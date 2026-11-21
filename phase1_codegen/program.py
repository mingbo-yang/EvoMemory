"""Versioned generated event programs and isolated invocation."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time

from .policy import (
    CAPABILITY_JSON_BYTES, CAPABILITY_RECEIPT_BYTES, INPUT_BYTES, ITERATION_BUDGET,
    MAX_MODEL_CALLS, OUTPUT_BYTES, CapabilityContractError, CapabilityInfrastructureError,
    GeneratedBudgetExceeded, Rejected, capability_request, json_copy,
    validate_output, validate_request, validate_source,
)

WORKER = Path(__file__).with_name('worker.py')
TOTAL_GENERATED_CPU_SECONDS = 1.0
TOTAL_WORKER_WALL_SECONDS = 5.0
WORKER_BOOTSTRAP_WALL_SECONDS = 30.0


class WorkerStartupError(Rejected):
    """Trusted startup failed before generated code was allowed to execute."""
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}



def _invoke_worker(request, timeout):
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded) > INPUT_BYTES:
        raise Rejected('worker input byte budget exceeded')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='evomem-program-') as directory:
        ready_read, ready_write = os.pipe()
        process = None
        try:
            with tempfile.TemporaryFile(dir=directory) as inp, tempfile.TemporaryFile(dir=directory) as out, tempfile.TemporaryFile(dir=directory) as err:
                inp.write(encoded); inp.seek(0)
                process = subprocess.Popen(
                    [sys.executable, '-s', '-S', str(WORKER)], stdin=inp,
                    stdout=out, stderr=err, cwd=directory, close_fds=True,
                    pass_fds=(ready_write,), start_new_session=True,
                    env={'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
                         'EVOMEM_WORKER_READY_FD': str(ready_write)},
                )
                os.close(ready_write); ready_write = None
                remaining = max(0., WORKER_BOOTSTRAP_WALL_SECONDS - (time.monotonic() - started))
                available, _, _ = select.select([ready_read], [], [], remaining)
                if not available or os.read(ready_read, 1) != b'R':
                    readiness = 'timeout' if not available else 'eof_before_ready'
                    returncode_before_cleanup = process.poll()
                    if returncode_before_cleanup is None:
                        process.kill()
                    process.wait()
                    out.seek(0); err.seek(0)
                    details = dict(stage='trusted_worker_bootstrap', readiness=readiness,
                        generated_execution_started=False, returncode_before_cleanup=returncode_before_cleanup,
                        returncode_after_cleanup=process.returncode,
                        elapsed_seconds=time.monotonic() - started,
                        startup_wall_limit_seconds=WORKER_BOOTSTRAP_WALL_SECONDS,
                        stdout_excerpt=out.read(500).decode(errors='replace'),
                        stderr_excerpt=err.read(500).decode(errors='replace'))
                    raise WorkerStartupError('trusted worker bootstrap failed before generated execution: ' +
                        json.dumps(details, ensure_ascii=False, sort_keys=True), details=details)
                bootstrap_seconds = time.monotonic() - started
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    raise GeneratedBudgetExceeded('worker exceeded cumulative five-second wall budget after trusted startup') from exc
                out.seek(0)
                raw = out.read(OUTPUT_BYTES + 1)
                err.seek(0)
                error = err.read(500)
        finally:
            os.close(ready_read)
            if ready_write is not None:
                os.close(ready_write)
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
    total_seconds = time.monotonic() - started
    if process.returncode != 0:
        raise Rejected('isolated worker rejected (%s): %s %s' % (
            process.returncode, raw[:500].decode(errors='replace'), error.decode(errors='replace')))
    if len(raw) > OUTPUT_BYTES:
        raise Rejected('worker output byte budget exceeded')
    try:
        receipt = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise Rejected('worker returned invalid JSON') from exc
    if type(receipt) is not dict:
        raise Rejected('worker receipt must be an object')
    isolation = receipt.get('isolation', {})
    if not isolation.get('seccomp') or not isolation.get('no_new_privs'):
        raise Rejected('worker did not verify kernel isolation')
    execution_seconds = isolation.get('worker_execution_wall_seconds')
    if type(execution_seconds) not in (int, float) or not math.isfinite(execution_seconds) or execution_seconds < 0:
        raise Rejected('worker returned invalid execution timing')
    if execution_seconds > timeout:
        raise GeneratedBudgetExceeded('worker exceeded cumulative five-second wall budget after trusted startup')
    receipt['bootstrap_seconds'] = bootstrap_seconds
    receipt['total_process_seconds'] = total_seconds
    return receipt, execution_seconds

# This is editable starting source, never a menu of selectable implementations.
# All text is constructed from own task history.  The generic implementation is
# deliberately simple; source and its functions are supplied to the Evolver.
BASELINE_SOURCE = '''def store_experience(state, experience, context):
    rows = [row for row in state.get("episodes", []) if row["id"] != experience["task_id"]]
    lines = ["Goal: " + experience.get("goal", "")]
    for step in experience.get("history", []):
        lines.append("Observation: " + step.get("observation", ""))
        lines.append("Action: " + step.get("action", ""))
        lines.append("Result: " + step.get("next_observation", ""))
    lines.append("Native reward: " + str(experience.get("score", 0)))
    text = "\\n".join(lines)
    rows.append({"id": experience["task_id"], "experience": text[:6000]})
    return {"episodes": rows[-8:]}

def retrieve_entries(state, context):
    words = set((context.get("goal", "") + " " + context.get("observation", "")).lower().split())
    rows = state.get("episodes", [])
    ranked = []
    for index, row in enumerate(rows):
        overlap = len(words.intersection(set(row["experience"].lower().split())))
        ranked.append([-overlap, index])
    result = []
    for pair in sorted(ranked):
        result.append(rows[pair[1]])
    return result

def on_event(state, event, context):
    memory = None
    if event.get("type") == "task_end":
        state = store_experience(state, event["experience"], context)
    if event.get("type") == "task_start":
        visible = event.get("visible", {})
        query = {"goal": visible.get("goal", context.get("goal", "")),
                 "observation": visible.get("obs", "")}
        memory = retrieve_entries(state, query)
    return {"state": state, "session": event.get("session", {}), "memory": memory}
'''


@dataclass(frozen=True)
class Program:
    source: str

    def __post_init__(self):
        tree = validate_source(self.source)
        # Cache only immutable metadata, never a mutable AST that instrumentation
        # could modify. Every isolated worker still independently validates source.
        object.__setattr__(self, '_entrypoints', frozenset({'on_event'}))
        object.__setattr__(self, '_source_hash', hashlib.sha256(self.source.encode()).hexdigest())
        canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
        object.__setattr__(self, '_identity_hash', hashlib.sha256(canonical.encode()).hexdigest())

    @property
    def entrypoints(self):
        return self._entrypoints

    @property
    def hash(self):
        return self._source_hash

    @property
    def identity_hash(self):
        return self._identity_hash

    def run(self, operation, state, experience=None, context=None, *, capabilities=None, diagnostics=False):
        if operation != 'event':
            raise Rejected('Program does not implement the requested ' + str(operation) + ' operation')
        if type(diagnostics) is not bool:
            raise Rejected('diagnostics must be a boolean')
        if not diagnostics:
            return self._run(operation, state, experience, context, capabilities=capabilities)
        captured = {}
        # Only this ordinary invocation determines mechanism validity and may
        # contact a configured model. Instrumentation has a separate CPU budget.
        authoritative = self._run(operation, state, experience, context,
                                  capabilities=capabilities, _capture=captured)
        class RecordedResultsOnly:
            def call(self, model_id, payload, *, operation):
                key = capability_request(model_id, payload, operation)['request_hash']
                if key not in captured:
                    raise Rejected('diagnostic replay requested an unrecorded model result')
                value = captured[key]
                return dict(value=json_copy(value['value'], CAPABILITY_JSON_BYTES),
                            receipt=json_copy(value['receipt'], CAPABILITY_RECEIPT_BYTES))
        started = time.monotonic()
        try:
            replay = self._run(operation, state, experience, context,
                               capabilities=RecordedResultsOnly(), diagnostics=True)
            wire = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            if wire(replay['value']) != wire(authoritative['value']):
                raise Rejected('diagnostic output differs from authoritative plain output')
            if replay['isolation']['iterations_used'] != authoritative['isolation']['iterations_used']:
                raise Rejected('diagnostic iteration count differs from authoritative plain execution')
            authoritative['execution_trace'] = replay['execution_trace']
            authoritative['diagnostics'] = dict(status='available', authoritative_execution='plain',
                worker_invocations=replay['worker_invocations'],
                worker_cpu_seconds=replay['isolation']['worker_cpu_seconds'],
                replayed_model_calls=len(replay['capability_calls']), new_model_calls=0,
                wall_seconds=time.monotonic() - started)
        except Exception as exc:
            # Diagnostic unavailability is evidence about observation coverage,
            # not invalidity of the already successful plain mechanism.
            authoritative['execution_trace'] = None
            authoritative['diagnostics'] = dict(status='unavailable', authoritative_execution='plain',
                error_type=type(exc).__name__, error=str(exc)[:1000], new_model_calls=0,
                wall_seconds=time.monotonic() - started)
        return authoritative

    def _run(self, operation, state, experience=None, context=None, *, capabilities=None,
             diagnostics=False, _capture=None):
        request = validate_request(operation, state, experience, {} if context is None else context)
        request['source'] = self.source
        if diagnostics:
            request['diagnostics'] = True
        results, references = {}, {}
        elapsed = cpu = generated_cpu = model_seconds = bootstrap_seconds = total_process_seconds = 0.0
        iterations = invocations = 0
        while True:
            remaining_wall = TOTAL_WORKER_WALL_SECONDS - elapsed
            if remaining_wall <= 0:
                raise GeneratedBudgetExceeded('worker exceeded cumulative five-second wall budget')
            request['iteration_budget'] = ITERATION_BUDGET - iterations
            request['capability_results'] = results
            receipt, seconds = _invoke_worker(request, remaining_wall)
            elapsed += seconds
            bootstrap_seconds += receipt.get('bootstrap_seconds', 0.)
            total_process_seconds += receipt.get('total_process_seconds', seconds)
            invocations += 1
            isolation = receipt['isolation']
            used = isolation.get('iterations_used')
            used_cpu = isolation.get('worker_cpu_seconds')
            generated = isolation.get('generated_cpu_seconds')
            if (type(used) is not int or not 0 <= used <= request['iteration_budget']
                    or type(used_cpu) not in (int, float) or not math.isfinite(used_cpu) or used_cpu < 0
                    or type(generated) not in (int, float) or not math.isfinite(generated)
                    or not 0 <= generated <= used_cpu):
                raise Rejected('invalid worker resource accounting')
            iterations += used
            cpu += used_cpu
            generated_cpu += generated
            # As with execution wall time, trusted interpreter/bootstrap work
            # is cost telemetry rather than generated execution. The kernel
            # still caps every worker's total CPU at one second; suspension
            # replays share this separate generated-execution allowance.
            if generated_cpu > TOTAL_GENERATED_CPU_SECONDS:
                raise GeneratedBudgetExceeded('worker exceeded cumulative one-second CPU budget')
            if elapsed > TOTAL_WORKER_WALL_SECONDS:
                raise GeneratedBudgetExceeded('worker exceeded cumulative five-second wall budget')
            calls = receipt.get('capability_calls', [])
            if type(calls) is not list or len(calls) > MAX_MODEL_CALLS:
                raise Rejected('invalid model capability call trace')
            for item in calls:
                if (type(item) is not dict or set(item) != {'request_hash', 'model_id', 'operation'}
                        or item['operation'] != operation):
                    raise Rejected('invalid model capability call trace')
            if 'needs_model' in receipt:
                pending = receipt['needs_model']
                if type(pending) is not dict or set(pending) != {'request_hash', 'model_id', 'operation', 'payload'}:
                    raise Rejected('invalid suspended model request')
                expected = capability_request(pending['model_id'], pending['payload'], operation)
                if pending != expected or not calls or calls[-1]['request_hash'] != pending['request_hash']:
                    raise Rejected('model request identity does not match the suspended call')
                key = pending['request_hash']
                if key in results or len(results) >= MAX_MODEL_CALLS:
                    raise Rejected('model capability replay failed to progress within eight calls')
                if capabilities is None or not callable(getattr(capabilities, 'call', None)):
                    raise Rejected('model_call requires an explicitly configured model capability broker')
                started_model = time.monotonic()
                try:
                    response = capabilities.call(pending['model_id'], json_copy(pending['payload']), operation=operation)
                except (GeneratedBudgetExceeded, CapabilityInfrastructureError, CapabilityContractError):
                    raise
                except Rejected as exc:
                    raise CapabilityContractError(str(exc)) from exc
                except Exception as exc:
                    raise CapabilityInfrastructureError('model capability broker failed: ' + str(exc)) from exc
                model_seconds += time.monotonic() - started_model
                if type(response) is not dict or set(response) != {'value', 'receipt'} or type(response['receipt']) is not dict:
                    raise CapabilityInfrastructureError('model capability broker must return value and receipt')
                try:
                    results[key] = json_copy(response['value'], CAPABILITY_JSON_BYTES)
                    references[key] = json_copy(response['receipt'], CAPABILITY_RECEIPT_BYTES)
                except Rejected as exc:
                    raise CapabilityInfrastructureError('Invalid model capability response: ' + str(exc)) from exc
                if _capture is not None:
                    _capture[key] = dict(value=json_copy(results[key], CAPABILITY_JSON_BYTES),
                                         receipt=json_copy(references[key], CAPABILITY_RECEIPT_BYTES))
                # Replay starts from the original serialized state/experience.
                # No intermediate worker output is committed to Task Memory.
                continue
            if 'value' not in receipt or any(item['request_hash'] not in references for item in calls):
                raise Rejected('worker returned incomplete model capability evidence')
            receipt['value'] = validate_output(operation, receipt['value'])
            receipt['capability_calls'] = [dict(item, receipt=references[item['request_hash']]) for item in calls]
            receipt['worker_invocations'] = invocations
            receipt['wall_seconds'] = elapsed
            receipt['bootstrap_seconds'] = bootstrap_seconds
            receipt['total_process_seconds'] = total_process_seconds
            receipt['capability_seconds'] = model_seconds
            isolation.update(iteration_budget=ITERATION_BUDGET, iterations_used=iterations,
                             generated_cpu_seconds=generated_cpu, worker_cpu_seconds=cpu)
            return receipt
