"""Audit within-environment shared-memory, cross-task-type experiments.

--write saves report.json and report.md only.

No model, environment, generated program, or mutable experiment module is imported.
An in-progress snapshot never treats missing results as failures or zero rewards.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
from pathlib import Path
import zipfile


if __package__:
    from phase1_codegen.outcomes import comparison_for_feedback
else:
    from outcomes import comparison_for_feedback


ENVIRONMENTS = ('alfworld', 'textworld')
CONDITIONS = ('no_external_em', 'cross_type_em')
INHERITED = 'inherited_no_target_evolution'
ROLES = ('source_adaptation', 'source_exploration', 'target_adaptation',
         'target_exploration', 'target_generalization')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def local_artifact(path, root):
    """Symlinked historical collections never become new physical execution."""
    return Path(path).resolve().is_relative_to(Path(root).resolve())


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.warnings = []
        self.memory_model_rows = {}

    def label(self, path):
        try:
            return str(Path(path).relative_to(self.root))
        except ValueError:
            return str(path)

    def json(self, path, default=None):
        path = Path(path)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            self.warnings.append(dict(file=self.label(path), error=str(exc)))
            return default

    def lines(self, path):
        path = Path(path)
        if not path.exists():
            return []
        rows = []
        try:
            with path.open() as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError('Expected a JSON object')
                        rows.append(row)
                    except ValueError as exc:
                        self.warnings.append(dict(file=self.label(path), line=number,
                                                  error=str(exc), note='Possibly an unfinished live ledger line'))
        except OSError as exc:
            self.warnings.append(dict(file=self.label(path), error=str(exc)))
        return rows

    def text(self, path):
        try:
            return Path(path).read_text() if Path(path).exists() else None
        except OSError as exc:
            self.warnings.append(dict(file=self.label(path), error=str(exc)))
            return None


def episode_summary(episodes, expected_ids=None):
    rows = episodes if isinstance(episodes, list) else []
    real = [e for e in rows if e.get('synthetic') is False]
    verified = [e for e in real if e.get('native_reward_verified') is True
                and not e.get('error') and finite(e.get('score'))]
    scores = [e['score'] for e in verified]
    return dict(recorded_rows=len(rows), real_rows=len(real), verified_rows=len(verified),
                synthetic_rows=sum(e.get('synthetic') is True for e in rows),
                unknown_provenance_rows=len(rows) - len(real) - sum(e.get('synthetic') is True for e in rows),
                unique_task_ids=len({e.get('task_id') for e in real if e.get('task_id')}),
                errors=sum(bool(e.get('error')) for e in rows),
                error_counts=dict(Counter(str(e['error']) for e in rows if e.get('error'))),
                native_reward_ge_1=sum(x >= 1 for x in scores),
                mean_verified_native_reward=sum(scores) / len(scores) if scores else None,
                native_steps=sum(len(e.get('history', [])) for e in real),
                recorded_episode_seconds=sum(e.get('elapsed', 0) for e in real if finite(e.get('elapsed'))),
                complete=(bool(expected_ids) and len(verified) == len(expected_ids)
                          and [e.get('task_id') for e in rows] == expected_ids)
                         if expected_ids is not None else None)


def model_summary(rows):
    calls = [r for r in rows if r.get('kind') == 'model' or r.get('api_attempted') is True]
    result = {}
    for role in sorted({r.get('role', 'unknown') for r in calls}):
        selected = [r for r in calls if r.get('role', 'unknown') == role]
        prompts, completions, missing = [], [], 0
        prices = []
        for row in selected:
            usage = row.get('usage') or {}
            a = row.get('prompt_tokens', usage.get('prompt_tokens', usage.get('input_tokens')))
            b = row.get('completion_tokens', usage.get('completion_tokens', usage.get('output_tokens')))
            if finite(a):
                prompts.append(a)
            if finite(b):
                completions.append(b)
            missing += not (finite(a) and finite(b))
            if finite(row.get('cost_usd')):
                prices.append(row['cost_usd'])
        result[role] = dict(model_attempts=len(selected), errors=sum(bool(r.get('error')) or r.get('ok') is False for r in selected),
                            known_prompt_tokens=sum(prompts), known_completion_tokens=sum(completions),
                            calls_missing_usage=missing,
                            recorded_call_seconds=sum(r.get('seconds', r.get('elapsed', 0)) for r in selected
                                                      if finite(r.get('seconds', r.get('elapsed', 0)))),
                            recorded_usd=sum(prices) if prices else None, calls_with_explicit_usd=len(prices),
                            pricing_note='Only explicit cost_usd is summed; local GPU dollar prices are not inferred')
    return result


def memory_model_counts(rows):
    """Count explicit broker facts; cache replay never adds physical token usage."""
    selected = [r for r in rows if r.get('kind') == 'memory_model_call']
    stats = dict(logical_requests=len(selected), cache_hits=0, backend_attempts=0,
                 inference_attempts=0, completed_physical=0, failed=0,
                 failed_backend_attempts=0, failed_inference_attempts=0, failed_cache_hits=0,
                 known_input_tokens=0, known_output_tokens=0,
                 inference_rows_missing_usage=0, deduplicated_usage_rows=0,
                 cache_usage_rows_ignored=0, logical_request_seconds=0.0, backend_attempt_seconds=0.0,
                 rows_missing_physical_attempt=0, rows_missing_inference_attempted=0)
    seen_usage = set()
    for index, row in enumerate(selected):
        hit = row.get('cache_hit') is True
        backend = row.get('physical_attempt') is True
        inference = row.get('inference_attempted') is True
        failed = row.get('ok') is False or bool(row.get('error'))
        stats['cache_hits'] += hit
        stats['backend_attempts'] += backend
        stats['inference_attempts'] += inference
        stats['completed_physical'] += row.get('physical') is True
        stats['failed'] += failed
        stats['failed_backend_attempts'] += backend and failed
        stats['failed_inference_attempts'] += inference and failed
        stats['failed_cache_hits'] += hit and failed
        stats['rows_missing_physical_attempt'] += 'physical_attempt' not in row
        stats['rows_missing_inference_attempted'] += 'inference_attempted' not in row
        seconds = row.get('seconds')
        if finite(seconds) and seconds >= 0:
            stats['logical_request_seconds'] += seconds
            stats['backend_attempt_seconds'] += seconds if backend else 0
        usage = row.get('usage') if isinstance(row.get('usage'), dict) else {}
        if hit:
            stats['cache_usage_rows_ignored'] += bool(usage)
            continue
        if not inference:
            continue
        # A repeated physical receipt can appear in copied ledgers. Count its
        # measured usage once; retain every logical ledger row in request counts.
        identity = tuple(row.get(k) for k in ('environment', 'model_id', 'capability_fingerprint',
                                              'cache_key', 'cache_record_hash'))
        if not all(isinstance(value, str) and value for value in identity):
            identity = ('unidentified', row.get('_audit_location', index))
        if identity in seen_usage:
            stats['deduplicated_usage_rows'] += 1
            continue
        seen_usage.add(identity)
        a = usage.get('prompt_tokens', usage.get('input_tokens'))
        b = usage.get('completion_tokens', usage.get('output_tokens'))
        valid_a, valid_b = type(a) is int and a >= 0, type(b) is int and b >= 0
        stats['known_input_tokens'] += a if valid_a else 0
        stats['known_output_tokens'] += b if valid_b else 0
        stats['inference_rows_missing_usage'] += not (valid_a and valid_b)
    return stats


def memory_model_groups(rows):
    groups = {}
    for row in rows:
        if row.get('kind') != 'memory_model_call':
            continue
        key = tuple(str(row.get(k, 'unknown')) for k in ('environment', 'model_id', 'phase'))
        groups.setdefault(key, []).append(row)
    return [dict(environment=key[0], model_id=key[1], phase=key[2], **memory_model_counts(value))
            for key, value in sorted(groups.items())]


def memory_model_report(reader, folder, environment, binding=None):
    """Audit formal-stage broker ledgers and actual read/write trace references.

    This never evaluates source text, imports a model, or executes a mechanism.
    Diagnostics and integration probes are outside the caller's stage whitelist.
    """
    folder = Path(folder)
    ledger = folder / 'capability_costs.jsonl'
    binding = binding or {}
    local = local_artifact(ledger, reader.root)
    rows = reader.lines(ledger) if local else []
    model_rows = [dict(row, _audit_location=reader.label(ledger) + ':' + str(i + 1))
                  for i, row in enumerate(rows) if row.get('kind') == 'memory_model_call']
    synthetic = [r for r in model_rows if r.get('synthetic') is True or binding.get('synthetic') is True]
    formal = [r for r in model_rows if r.get('synthetic') is not True and binding.get('synthetic') is not True]
    reader.memory_model_rows[str(ledger.resolve())] = formal
    # Runtime receipts are exact copies of broker rows; their outer request hash
    # includes operation, whereas the broker hash has a different documented scope.
    ledger_hashes = {digest({k: v for k, v in row.items() if k != '_audit_location'}) for row in formal}
    pool = reader.root / 'capability_pools' / str(environment)
    manifest_path = pool / 'manifest.json'
    manifest = reader.json(manifest_path, {}) or {}
    fingerprint = digest(manifest) if manifest else None
    expected_fingerprint = binding.get('model_capabilities')
    caches, violations, trace_refs = {}, [], []
    for row in formal:
        location = row['_audit_location']
        checks = dict(environment_matches=row.get('environment') == environment,
                      manifest_fingerprint_matches=(row.get('capability_fingerprint') == fingerprint)
                      if row.get('capability_fingerprint') is not None and fingerprint else None,
                      cache_flags_consistent=not (row.get('cache_hit') is True and any(
                          row.get(k) is True for k in ('physical_attempt', 'inference_attempted', 'physical'))),
                      completed_requires_inference=(row.get('inference_attempted') is True and row.get('ok') is True)
                      if row.get('physical') is True else True,
                      inference_requires_backend=row.get('physical_attempt') is True
                      if row.get('inference_attempted') is True else True)
        for name, value in checks.items():
            if value is False:
                violations.append(dict(path=location, check=name))
        if not row.get('cache_path'):
            continue  # Rejected payloads legitimately have no cache record.
        cache_path = Path(row['cache_path'])
        if not cache_path.is_absolute():
            cache_path = reader.root / cache_path
        identity = (str(cache_path), row.get('cache_record_hash'), row.get('capability_fingerprint'))
        if identity in caches:
            continue
        in_pool = cache_path.resolve().is_relative_to(pool.resolve())
        cache = reader.json(cache_path, {}) if in_pool else {}
        cache = cache or {}
        cache_checks = dict(path_in_environment_pool=in_pool, cache_exists=bool(cache),
            record_hash_matches=(cache.get('record_hash') == row.get('cache_record_hash') ==
                                 digest({k: v for k, v in cache.items() if k != 'record_hash'})) if cache else None,
            identity_matches=(cache.get('key') == row.get('cache_key') and cache.get('model_id') == row.get('model_id')
                              and cache.get('environment') == environment) if cache else None,
            capability_fingerprint_matches=(cache.get('capability_fingerprint') == row.get('capability_fingerprint') == fingerprint)
                                           if cache and fingerprint else None)
        caches[identity] = dict(cache_path=reader.label(cache_path), cache_key=row.get('cache_key'),
                               cache_record_hash=row.get('cache_record_hash'),
                               capability_fingerprint=row.get('capability_fingerprint'), checks=cache_checks)
        for name, value in cache_checks.items():
            if value is False:
                violations.append(dict(path=reader.label(cache_path), check=name))
    def record_calls(trace, artifact, position, operation, provenance, program_hash=None):
        from phase1_codegen.behavior_diagnostics import iter_execution_receipts
        if not isinstance(trace, dict):
            return
        try:
            receipts = list(iter_execution_receipts(trace, position))
        except (TypeError, AttributeError, ValueError):
            violations.append(dict(path=artifact + '#' + position, check='invalid_event_receipt_structure'))
            return
        for receipt, pointer, kind in receipts:
            # Historical report data may still carry read/write billing labels;
            # every nested current Hook receipt has the event operation.
            current_operation = 'event' if pointer != position else operation
            record_receipt(dict(receipt, task_id=trace.get('task_id', (trace.get('context') or {}).get('task_id'))),
                           artifact, pointer, current_operation, provenance, program_hash)

    def record_receipt(trace, artifact, position, operation, provenance, program_hash=None):
        calls = trace.get('capability_calls', [])
        if not isinstance(calls, list):
            violations.append(dict(path=artifact + '#' + position, check='capability_calls_is_list'))
            return
        for i, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            receipt = call.get('receipt') if isinstance(call.get('receipt'), dict) else {}
            verified = bool(receipt) and digest(receipt) in ledger_hashes
            fields_match = (call.get('operation') == receipt.get('operation') == operation
                            and call.get('model_id') == receipt.get('model_id')
                            and receipt.get('environment') == environment)
            trace_refs.append(dict(artifact=artifact, position=position + '/capability_calls/' + str(i),
                provenance=provenance, task_id=trace.get('task_id', (trace.get('context') or {}).get('task_id')),
                program_hash=program_hash or trace.get('program_hash'), operation=operation, model_id=call.get('model_id'),
                runtime_request_hash=call.get('request_hash'), broker_request_hash=receipt.get('request_hash'),
                broker_receipt_hash=digest(receipt) if receipt else None, receipt_found_in_stage_ledger=verified,
                receipt_identity_matches=fields_match, cache_key=receipt.get('cache_key'),
                cache_record_hash=receipt.get('cache_record_hash'), capability_fingerprint=receipt.get('capability_fingerprint')))
            if not verified or not fields_match:
                violations.append(dict(path=artifact + '#' + position + '/' + str(i), check='trace_capability_receipt_matches_ledger'))
    if binding.get('synthetic') is not True:
        for path in sorted((folder / 'trials').glob('*/behavior.json')):
            if not local_artifact(path, reader.root):
                continue
            behavior = reader.json(path, {}) or {}
            trial = reader.json(path.with_name('trial.json'), {}) or {}
            for side in ('parent', 'candidate'):
                value = behavior.get(side) or {}
                for field, operation in (('write_trace', 'write'), ('read_trace', 'read')):
                    for i, trace in enumerate(value.get(field, [])):
                        record_calls(trace, reader.label(path), side + '/' + field + '/' + str(i), operation,
                                     'mechanism_replay', trial.get(side + '_hash'))
        native_path = folder / 'runner/native/code_memory_traces.jsonl'
        if local_artifact(native_path, reader.root):
            for n, row in enumerate(reader.lines(native_path)):
                if row.get('kind') != 'generated_memory_native_episode' or row.get('synthetic') is not False:
                    continue
                if row.get('environment') != environment:
                    violations.append(dict(path=reader.label(native_path) + ':' + str(n + 1), check='native_trace_environment_matches'))
                    continue
                hook = row.get('memory_hook')
                if isinstance(hook, dict):
                    record_calls(dict(event_trace=hook.get('events'), task_id=row.get('task_id')),
                                 reader.label(native_path), str(n + 1) + '/memory_hook',
                                 'event', 'native_episode_memory_execution',
                                 (row.get('mechanism') or {}).get('program_hash'))
                    continue  # write_calls may contain a copy of these same events.
                for field, operation in (('write_calls', 'write'), ('read_calls', 'read')):
                    for i, trace in enumerate(row.get(field, [])):
                        record_calls(trace, reader.label(native_path), str(n + 1) + '/' + field + '/' + str(i),
                                     operation, 'native_episode_memory_execution')
    verified_refs = [r for r in trace_refs if r['receipt_found_in_stage_ledger'] and r['receipt_identity_matches']]
    binding_matches = expected_fingerprint == fingerprint if expected_fingerprint is not None and fingerprint else None
    if (expected_fingerprint is not None or any(r.get('capability_fingerprint') for r in formal)) and not manifest:
        violations.append(dict(path=reader.label(manifest_path), check='capability_manifest_available'))
    if manifest and manifest.get('environment') != environment:
        violations.append(dict(path=reader.label(manifest_path), check='manifest_environment_matches'))
    if binding_matches is False:
        violations.append(dict(path=reader.label(folder), check='binding_capability_manifest_matches'))
    return dict(available=ledger.exists() and local, ledger_path=reader.label(ledger),
        totals=memory_model_counts(formal), by_environment_model_phase=memory_model_groups(formal),
        excluded_synthetic=memory_model_counts(synthetic), excluded_external_ledger=ledger.exists() and not local,
        manifest=dict(path=reader.label(manifest_path), available=bool(manifest), fingerprint=fingerprint,
                      bound_fingerprint=expected_fingerprint, binding_matches=binding_matches,
                      environment_matches=manifest.get('environment') == environment if manifest else None),
        cache_receipts=list(caches.values()), integrity_findings=violations,
        mechanism_execution=dict(recorded_trace_call_references=len(trace_refs),
            verified_trace_call_references=len(verified_refs),
            unique_verified_broker_receipts=len({r['broker_receipt_hash'] for r in verified_refs}),
            model_ids=sorted({r['model_id'] for r in verified_refs if isinstance(r['model_id'], str)}),
            evidence=trace_refs,
            interpretation='Event or legacy read/write execution receipts, not source keywords. Replayed and native trace copies are witnesses, not additional logical or physical calls; calls do not establish memory improvement or transfer.'),
        counting_rule='Logical requests are broker ledger rows; physical flags are counted explicitly. Usage is measured non-cache inference usage, deduplicated by cache record identity; local token estimates and cached usage are excluded. Diagnostics are outside the formal-stage paths.')


def messages(row):
    request = row.get('request') or {}
    return request.get('messages', row.get('messages', []))


def user_payload(row):
    users = [m.get('content') for m in messages(row) if m.get('role') == 'user']
    if not users or not isinstance(users[-1], str):
        return {}
    try:
        value = json.loads(users[-1])
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


def displayed_records(payload):
    value = payload.get('evolution_memory', [])
    if isinstance(value, list):
        return [r for r in value if isinstance(r, dict) and isinstance(r.get('id'), str)]
    return []  # A clipped string mentioning an ID is not a displayed full record.


def program_evidence(source):
    if source is None:
        return dict(available=False)
    try:
        tree = ast.parse(source)
        functions = {node.name: digest(ast.dump(node, include_attributes=False))
                     for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        return dict(available=True, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    ast_sha256=digest(ast.dump(tree, include_attributes=False)), functions=functions,
                    lines=len(source.splitlines()), syntax_valid=True)
    except SyntaxError as exc:
        return dict(available=True, syntax_valid=False, syntax_error=str(exc))


def review_evidence(reader, folder, receipt):
    if not receipt:
        return dict(available=False)
    displayed = reader.json(folder / receipt.get('displayed_path', '__missing__'), {}) or {}
    report = receipt.get('report') or {}
    registry = {r.get('id') for r in displayed.get('evidence', [])}
    assessment = report.get('assessment') or {}
    cited = assessment.get('evidence_ids', []) + [eid for f in report.get('findings', []) for eid in f.get('evidence_ids', [])]
    return dict(available=True, ok=receipt.get('ok'), stage=receipt.get('stage'), error=receipt.get('error'),
                report_hash=receipt.get('report_hash'), report_hash_matches=digest(receipt.get('report')) == receipt.get('report_hash'),
                displayed_hash_matches=digest(displayed) == receipt.get('displayed_hash'),
                assessment=assessment, finding_count=len(report.get('findings', [])),
                cited_ids=sorted(set(cited)), all_citations_displayed=bool(report) and set(cited) <= registry,
                assessment_has_required_citation=bool(assessment.get('evidence_ids')) or assessment.get('verdict') == 'insufficient_evidence',
                positive_candidate_has_candidate_citation=(any(str(eid).startswith('candidate:') for eid in assessment.get('evidence_ids', []))
                    if assessment.get('subject') == 'candidate' and assessment.get('verdict') == 'positive' else None),
                displayed_design_record_ids=[r['id'] for r in displayed_records(displayed)])


def search_report(reader, folder, source_records):
    state = reader.json(folder / 'state.json', {}) or {}
    binding = state.get('binding') or {}
    freeze = reader.json(folder / 'freeze.json', {}) or {}
    status = reader.json(folder / 'status.json', {}) or {}
    evolver_rows = reader.lines(folder / 'evolver_costs.jsonl')
    reviewer_rows = reader.lines(folder / 'reviewer_costs.jsonl')
    native_rows = reader.lines(folder / 'runner/native/canonical_episodes.jsonl')
    native_costs = reader.lines(folder / 'runner/native/costs.jsonl')
    sources = {r['id']: r for r in source_records if isinstance(r, dict) and 'id' in r}
    committed = {r['path']: r['hash'] for r in state.get('history', [])}
    trial_reports = []
    previous_post = None
    parent_in_chain = state.get('initial_program')
    ancestry_ok = bool(state)
    for path in sorted((folder / 'trials').glob('*/trial.json')):
        trial = reader.json(path, {}) or {}
        directory = path.parent
        index = trial.get('round')
        parent_source = reader.text(folder / 'programs' / (str(trial.get('parent_hash')) + '.py'))
        candidate_source = reader.text(folder / 'programs' / (str(trial.get('candidate_hash')) + '.py'))
        proposal = reader.json(directory / 'proposal.json', {}) or {}
        if candidate_source is None:
            candidate_source = (proposal.get('value') or {}).get('source')
        parent, candidate = program_evidence(parent_source), program_evidence(candidate_source)
        behavior = reader.json(directory / 'behavior.json', {}) or {}
        before, after = behavior.get('parent') or {}, behavior.get('candidate') or {}
        pre = review_evidence(reader, folder, trial.get('pre_review'))
        post = review_evidence(reader, folder, trial.get('post_review'))
        retrieval = trial.get('evolution_memory_retrieval') or reader.json(directory / 'retrieval.json', {}) or {}
        expected_seed = binding.get('seed', 11) + index if type(index) is int else None
        calls = [row for row in evolver_rows if row.get('seed') == expected_seed and row.get('role') == 'code_evolver']
        actual = user_payload(calls[-1]) if calls else {}
        actual_records = displayed_records(actual)
        actual_ids = [r['id'] for r in actual_records]
        source_ids = retrieval.get('selected_source_ids', [])
        own_input = reader.json(directory / 'proposer_input.json', {}) or {}
        selected_by_id = {r['id']: r for r in displayed_records(own_input)}
        model_record_matches = {r['id']: digest(r) == digest(selected_by_id.get(r['id'])) for r in actual_records}
        review_calls = [row for row in reviewer_rows if type(index) is int and row.get('seed') in
                        (binding.get('seed', 11) + 20000 + 2 * index, binding.get('seed', 11) + 20001 + 2 * index)]
        reviewer_shown = {r['id'] for row in review_calls for r in displayed_records(user_payload(row))}
        own_review = own_input.get('mechanism_review') or {}
        pre_hash = pre.get('report_hash')
        review_field = actual.get('mechanism_review')
        review_sent = isinstance(review_field, dict) and review_field.get('original_hash') == digest(own_review)
        relative = str(path.relative_to(folder))
        is_committed = relative in committed
        receipt_matches = digest(trial) == committed.get(relative) if is_committed else None
        if is_committed:
            ancestry_ok &= receipt_matches and trial.get('parent_hash') == parent_in_chain
            if trial.get('accepted'):
                parent_in_chain = trial.get('candidate_hash')
        changed_functions = sorted(name for name in set(parent.get('functions', {})) | set(candidate.get('functions', {}))
                                   if parent.get('functions', {}).get(name) != candidate.get('functions', {}).get(name))
        diff = ''.join(difflib.unified_diff((parent_source or '').splitlines(True), (candidate_source or '').splitlines(True),
                                          fromfile='parent.py', tofile='candidate.py'))
        trial_reports.append(dict(
            round=index, committed=is_committed, committed_trial_hash_matches=receipt_matches,
            status=trial.get('public_status'), accepted=trial.get('accepted'), acceptance_basis=trial.get('acceptance_basis'),
            error=trial.get('error'), review_error=trial.get('review_error'),
            parent_hash=trial.get('parent_hash'), candidate_hash=trial.get('candidate_hash'),
            parent_program=parent, candidate_program=candidate,
            parent_source_hash_matches=parent.get('source_sha256') == trial.get('parent_hash') if parent.get('available') else None,
            candidate_source_hash_matches=candidate.get('source_sha256') == trial.get('candidate_hash') if trial.get('candidate_hash') else None,
            source_changed=parent_source != candidate_source if parent_source is not None and candidate_source is not None else None,
            ast_changed=parent.get('ast_sha256') != candidate.get('ast_sha256') if parent.get('syntax_valid') and candidate.get('syntax_valid') else None,
            changed_functions=changed_functions, code_diff=diff,
            mechanism_validated_receipt=trial.get('mechanism_validated'),
            behavior_receipt_matches=(digest(dict(parent=before, candidate=after)) == trial.get('behavior_evidence')) if behavior else None,
            changed_readouts=sum(a != b for a, b in zip(before.get('input_hashes', []), after.get('input_hashes', []))) if behavior else None,
            readout_comparison_scope=('task_start_only' if any(
                'on_event' in p.get('functions', {}) for p in (parent, candidate)) else 'task_level_readout'),
            compared_readouts=len(before.get('input_hashes', [])) if behavior else None,
            readout_counts_match=len(before.get('input_hashes', [])) == len(after.get('input_hashes', [])) if behavior else None,
            stored_state_changed=digest(before.get('stored_state')) != digest(after.get('stored_state')) if behavior else None,
            stored_tokens=dict(parent=before.get('stored_tokens'), candidate=after.get('stored_tokens')),
            physical_candidate_episodes=sum(e.get('program_hash') == trial.get('candidate_hash') for e in native_rows),
            exploration=comparison_for_feedback(trial.get('exploration')), pre_review=pre, post_review=post,
            role_dependencies=dict(pre_report_supplied_to_evolver=bool(pre_hash) and own_review.get('report_hash') == pre_hash,
                                   actual_request_review_origin_hash_matches=review_sent if calls else None,
                                   review_report_hash_present_in_displayed_text=(pre_hash in str(review_field)) if pre_hash and calls else None,
                                   previous_post_reused=pre_hash == previous_post if previous_post else None,
                                   proposal_three_field_schema=set((proposal.get('value') or {})) == {'source', 'hypothesis', 'risk'},
                                   note='Hash/data dependencies audit logical order; separate ledgers do not prove cross-role wall-clock order'),
            retrieval=retrieval,
            actual_model_exposure=dict(evolver_attempts=len(calls), evolver_record_ids=actual_ids,
                                       evolver_last_call_ok=calls[-1].get('ok') if calls else None,
                                       evolver_records_match_selected=model_record_matches,
                                       source_ids_shown_to_evolver=[rid for rid in source_ids if rid in actual_ids and model_record_matches.get(rid)],
                                       source_record_hash_matches_transfer={r['id']: digest(r) == digest(sources.get(r['id']))
                                                                            for r in actual_records if r['id'] in source_ids},
                                       source_ids_shown_to_current_round_reviewer=[rid for rid in source_ids if rid in reviewer_shown]),
            hypothesis=trial.get('hypothesis'),
            self_reported_source_usage_ids=[rid for rid in source_ids if rid in str(trial.get('hypothesis', ''))],
            source_record_environments={rid: sources[rid].get('observed_context', {}).get('environment') for rid in source_ids if rid in sources},
            source_record_families={rid: sources[rid].get('observed_context', {}).get('family') for rid in source_ids if rid in sources},
            target_family=(binding.get('context') or {}).get('family'),
            evidence_path=reader.label(path), interpretation='Retrieval/display/self-attribution are separate evidence levels; none alone establishes causal transfer',
        ))
        previous_post = post.get('report_hash') if post.get('ok') else None
    if state:
        ancestry_ok &= parent_in_chain == state.get('current_program') and state.get('next_round') == len(committed)
    prompt_sets = {}
    for role, rows in [('code_evolver', evolver_rows), ('mechanism_reviewer', reviewer_rows)]:
        prompt_sets[role] = sorted({digest(m.get('content')) for row in rows for m in messages(row) if m.get('role') == 'system'})
    completed = status.get('status') == 'complete' and state.get('next_round') == binding.get('max_rounds') and bool(state)
    return dict(path=reader.label(folder), status='complete' if completed else status.get('status', 'not_started'),
                complete=completed, frozen_plan=freeze.get('plan'), completed_rounds=state.get('next_round', 0),
                planned_rounds=binding.get('max_rounds', (freeze.get('plan') or {}).get('rounds')),
                accepted_revisions=state.get('revision', 0), current_program=state.get('current_program'),
                initial_program=state.get('initial_program'), binding=binding,
                state_binding_hash_matches=digest(binding) == state.get('binding_hash') if state else None,
                committed_ancestry_matches=ancestry_ok if state else None,
                prompt_hashes=prompt_sets,
                separate_role_prompts=(not set(prompt_sets['code_evolver']) & set(prompt_sets['mechanism_reviewer']))
                                     if all(prompt_sets.values()) else None,
                role_labels_match_ledgers=(all(r.get('role') == 'code_evolver' for r in evolver_rows)
                                          and all(r.get('role') == 'mechanism_reviewer' for r in reviewer_rows))
                                         if evolver_rows or reviewer_rows else None,
                native=episode_summary(native_rows), costs=model_summary(native_costs + evolver_rows + reviewer_rows),
                memory_model_calls=memory_model_report(reader, folder,
                    (binding.get('context') or {}).get('environment',
                        folder.name if folder.name in ENVIRONMENTS else folder.parent.name), binding),
                trials=trial_reports,
                unfinished_trial_directories=[reader.label(p) for p in sorted((folder / 'trials').glob('*'))
                                              if p.is_dir() and not (p / 'trial.json').exists()])


def heldout_report(reader, environment):
    folder = reader.root / 'generalization' / environment
    results = reader.json(folder / 'results.json', {}) or {}
    manifest = reader.json(reader.root / 'datasets' / environment / 'target_generalization.json', []) or []
    expected = [t['id'] for t in manifest]
    adaptation = reader.json(reader.root / 'datasets' / environment / 'target_adaptation.json', []) or []
    protocol = reader.json(reader.root / 'protocol.json', {}) or {}
    summaries = {}
    for name in (INHERITED,) + CONDITIONS:
        result = results.get(name) or {}
        summaries[name] = dict(program_hash=result.get('program_hash'),
                               **episode_summary(result.get('episodes', []), expected))
    pairs = {}
    for a, b in [(INHERITED, 'no_external_em'), (INHERITED, 'cross_type_em'), ('no_external_em', 'cross_type_em')]:
        complete = bool(expected) and summaries[a]['complete'] and summaries[b]['complete']
        if not complete:
            pairs[a + ' -> ' + b] = dict(status='incomplete', mean_delta=None)
            continue
        left, right = results[a]['episodes'], results[b]['episodes']
        delta = [y['score'] - x['score'] for x, y in zip(left, right)]
        pairs[a + ' -> ' + b] = dict(status='complete', n=len(delta), mean_delta=sum(delta) / len(delta),
                                     wins=sum(x > 0 for x in delta), ties=sum(x == 0 for x in delta), losses=sum(x < 0 for x in delta),
                                     same_program=summaries[a]['program_hash'] == summaries[b]['program_hash'])
    binding = reader.json(folder / 'binding.json', {}) or {}
    return dict(status='complete' if expected and all(s['complete'] for s in summaries.values()) else 'in_progress',
                expected_tasks=len(expected), sealed_declaration=binding.get('sealed_from_review_and_evolver'),
                versions=summaries, paired_native_results=pairs, binding=binding,
                memory_model_calls=memory_model_report(reader, folder, environment, binding),
                scope=dict(target_type_adaptation_tasks=len(adaptation), heldout_target_instances=len(expected),
                           zero_shot_unseen_task_type=False,
                           declared_scope=protocol.get('generalization_scope'),
                           declared_data_origin=(protocol.get('heldout_data_origin') or {}).get(environment),
                           manifest_split_counts=dict(Counter(t.get('official_split', t.get('split', 'unknown')) for t in manifest)),
                           official_train_archive_tasks=sum(t.get('archive_member', '').startswith('json_2.1.1/train/') for t in manifest)),
                no_heldout_writes=(all(e.get('memory_write_calls') == 0 and e.get('generated_memory_write_calls') == 0
                                      and e.get('read_only_memory') is True and bool(e.get('memory_hash_before'))
                                      and e.get('memory_hash_before') == e.get('memory_hash_after')
                                      for value in results.values() for e in value.get('episodes', [])) if results else None),
                interpretation='Paired observed native rewards on these held-out tasks; positive review/adoption is not task gain. Identical programs may share physical evaluation.')


def imported_adaptation(reader, environment, role, record, spec, tasks):
    """Attribute just the imported role's historical evidence/cost, never its siblings."""
    if not spec or not spec.get('collection'):
        return None
    origin = Path(spec['collection'])
    artifact_path = origin / (role + '.json')
    original = reader.json(artifact_path, {}) or {}
    original_binding = reader.json(origin / 'binding.json', {}) or {}
    ids = [t['id'] for t in tasks]
    selected_ids = set(ids)
    applied = record.get('newly_executed') is False and record.get('reuse_provenance') == spec
    historical_calls = [row for row in reader.lines(origin / 'runner/native/costs.jsonl')
                        if row.get('task_id') in selected_ids and row.get('phase') == role]
    cache_matches = []
    for task, episode in zip(tasks, original.get('episodes', [])):
        cache = reader.json(origin / 'episodes' / (digest(task) + '.json'), {}) or {}
        cache_matches.append(cache.get('task_hash') == digest(task)
                             and cache.get('binding_hash') == digest(original_binding)
                             and cache.get('episode') == episode)
    artifact_text = reader.text(artifact_path)
    summary = episode_summary(original.get('episodes', []), ids)
    return dict(environment=environment, role=role, status='imported' if applied else 'planned_not_yet_imported',
                source_collection=str(origin), source_artifact=str(artifact_path), provenance=spec,
                available_original_evidence=summary,
                imported_real_rows=summary['real_rows'] if applied else 0,
                newly_executed_here=False,
                artifact_hash_matches=hashlib.sha256(artifact_text.encode()).hexdigest() == spec.get('artifact_hash') if artifact_text is not None else None,
                binding_hash_matches=digest(original_binding) == spec.get('binding_hash') if original_binding else None,
                original_task_manifest_matches=(original_binding.get('tasks') or {}).get(role) == tasks if original_binding else None,
                original_episode_caches_match=len(cache_matches) == len(tasks) and all(cache_matches),
                imported_episodes_equal_original=record.get('episodes') == original.get('episodes') if applied else None,
                historical_costs_for_this_role_only=model_summary(historical_calls),
                cost_scope='Historical source artifact cost, not new model calls or new physical episodes in this root. Other old collection roles are excluded.')


def source_baseline_reuse(reader, environment, spec, tasks):
    """Independently audit a baseline import, without calling any runtime/helper.

    Missing imports are normal in old runs and in not-yet-started new sources.
    Only a fully verified import contributes historical episode counts.
    """
    folder = reader.root / 'source' / environment
    audit_path = folder / 'baseline_reuse.json'
    audit = reader.json(audit_path, {}) or {}
    result = dict(environment=environment, phase='round-000/exploration/parent',
                  status='planned_not_yet_imported' if spec else 'not_configured',
                  available=bool(audit), imported_real_rows=0, verified=False,
                  newly_executed_here=False, receipt_path=reader.label(audit_path),
                  checks={}, integrity_findings=[], historical_costs_for_this_phase_only={},
                  cost_scope='Historical initial source-parent exploration only; excluded from new execution and model_costs.')
    if not audit:
        if audit_path.exists():
            result['integrity_findings'].append(dict(path=reader.label(audit_path), check='readable_import_receipt'))
        return result
    checks = result['checks']
    fields = {'schema', 'source_run', 'state_sha256', 'measurement_key', 'measurement_sha256', 'native_costs_sha256'}
    sha = lambda value: type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    checks['protocol_origin_schema'] = (type(spec) is dict and set(spec) == fields
        and type(spec.get('schema')) is int and spec['schema'] == 1
        and type(spec.get('source_run')) is str and Path(spec['source_run']).is_absolute()
        and all(sha(spec.get(k)) for k in fields - {'schema', 'source_run'}))
    if not checks['protocol_origin_schema']:
        result.update(status='invalid_import', integrity_findings=[dict(path=reader.label(audit_path), check='protocol_origin_schema')])
        return result
    old = Path(spec['source_run']).resolve()
    old_state_path = old / 'state.json'
    old_measurement_path = old / 'measurements' / (spec['measurement_key'] + '.json')
    old_costs_path = old / 'runner/native/costs.jsonl'
    def pinned(path, value):
        try:
            with path.open('rb') as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            return h.hexdigest() == value
        except OSError:
            return False
    checks.update(source_directory=(old.name == environment and old.parent.name == 'source' and old != folder.resolve()),
                  local_import_receipt=local_artifact(audit_path, reader.root),
                  source_state_pin=pinned(old_state_path, spec['state_sha256']),
                  source_measurement_pin=pinned(old_measurement_path, spec['measurement_sha256']),
                  source_costs_pin=pinned(old_costs_path, spec['native_costs_sha256']))
    original_state = reader.json(old_state_path, {}) or {}
    state = reader.json(folder / 'state.json', {}) or {}
    old_binding, binding = original_state.get('binding') or {}, state.get('binding') or {}
    original = reader.json(old_measurement_path, {}) or {}
    program = original_state.get('initial_program')
    source = reader.text(old / 'programs' / (str(program) + '.py'))
    current_source = reader.text(folder / 'programs' / (str(state.get('initial_program')) + '.py'))
    from phase1_codegen.baseline_reuse import NON_EXECUTION_BINDINGS, RUNTIME_FILES, RUNTIME_PREFIXES
    ignored = NON_EXECUTION_BINDINGS
    runtime = set(RUNTIME_FILES)
    def execution_code(value):
        return {k: v for k, v in value.items() if k in runtime or k.startswith(RUNTIME_PREFIXES)}
    old_code, code = old_binding.get('code') or {}, binding.get('code') or {}
    checks.update(source_binding_hash=bool(old_binding) and original_state.get('binding_hash') == digest(old_binding),
                  destination_binding_hash=bool(binding) and state.get('binding_hash') == digest(binding) == audit.get('destination_binding_hash'),
                  initial_program_source=(sha(program) and program == state.get('initial_program')
                      and source is not None and hashlib.sha256(source.encode()).hexdigest() == program
                      and current_source == source),
                  native_execution_bindings=(bool(binding) and {k: v for k, v in old_binding.items() if k not in ignored}
                      == {k: v for k, v in binding.items() if k not in ignored}),
                  native_execution_code=(all(sha(old_code.get(k)) and sha(code.get(k)) for k in runtime)
                      and execution_code(old_code) == execution_code(code)),
                  source_scope=(binding.get('synthetic') is False and old_binding.get('synthetic') is False
                      and (binding.get('context') or {}).get('environment') == environment
                      and binding.get('source_memory_hash') == digest([])),
                  exploration_manifest=bool(tasks) and old_binding.get('exploration_tasks') == binding.get('exploration_tasks') == tasks)
    old_key = digest(dict(binding=old_binding, program=program, tasks=tasks))
    key = digest(dict(binding=binding, program=program, tasks=tasks))
    cache_path = folder / 'measurements' / (key + '.json')
    cache = reader.json(cache_path, {}) or {}
    phase = result['phase']
    episodes = original.get('episodes', [])
    episodes = episodes if isinstance(episodes, list) else []
    checks.update(source_measurement_identity=(original.get('key') == spec['measurement_key'] == old_key
                      and original.get('phase') == phase and original.get('synthetic') is False
                      and original.get('episodes_hash') == digest(episodes)
                      and 'reuse_provenance' not in original and original.get('physical', True) is True
                      and original.get('newly_executed', True) is True),
                  cache_identity=(cache.get('key') == audit.get('measurement_key') == key
                      and digest(cache) == audit.get('measurement_hash') and local_artifact(cache_path, reader.root)
                      and cache.get('phase') == phase and cache.get('synthetic') is False
                      and cache.get('physical') is False and cache.get('newly_executed') is False
                      and cache.get('seconds') == 0 and cache.get('episodes') == episodes
                      and cache.get('episodes_hash') == digest(episodes)),
                  import_receipt_identity=(audit.get('schema') == 1 and audit.get('kind') == 'source_baseline_import'
                      and audit.get('physical') is False and audit.get('newly_executed') is False
                      and audit.get('phase') == phase and audit.get('episodes') == len(episodes)))
    limit = binding.get('environment_step_limit')
    valid = bool(tasks) and type(limit) is int and limit > 0 and len(episodes) == len(tasks)
    for episode, task in zip(episodes, tasks):
        score = episode.get('score')
        valid &= (episode.get('task_id') == task['id'] and episode.get('group_id') == task.get('group_id')
                  and episode.get('family') == task.get('family') and episode.get('environment') == environment
                  and episode.get('phase') == phase and episode.get('synthetic') is False
                  and episode.get('native_reward_verified') is True and not episode.get('error') and finite(score)
                  and (episode.get('native_recorder') or {}).get('reward') == score
                  and episode.get('executor') == 'ALMA.Agent_Workflow.run_single_task'
                  and episode.get('environment_seed') == task.get('seed')
                  and episode.get('seed') == 42000 + int(digest(task['id'])[:6], 16) % 10000000
                  and len(episode.get('history', [])) <= (limit if type(limit) is int else 0)
                  and episode.get('program_hash') == program and sha(episode.get('program_identity_hash'))
                  and episode.get('read_only_memory') is True and episode.get('memory_read_calls') == 1
                  and episode.get('memory_write_calls') == 0 and episode.get('generated_memory_write_calls') == 0
                  and sha(episode.get('memory_hash_before'))
                  and episode.get('memory_hash_before') == episode.get('memory_hash_after'))
        for item in (episode, task):
            for field in ('phase', 'split', 'official_split', 'data_role'):
                value = str(item.get(field, '')).lower()
                valid &= value != 'test' and not any(s in value for s in ('heldout', 'held_out', 'final', 'confirm',
                                  'generalization', 'candidate', 'eval_in_distribution', 'eval_out_of_distribution'))
    checks['complete_real_parent_episodes'] = valid
    provenance = audit.get('reuse_provenance') or {}
    checks['reuse_provenance'] = (provenance == cache.get('reuse_provenance') and provenance.get('origin') == spec
        and provenance.get('source_run') == str(old) and provenance.get('source_binding_hash') == original_state.get('binding_hash')
        and provenance.get('source_measurement_path') == str(old_measurement_path)
        and provenance.get('source_measurement_key') == old_key and provenance.get('source_episodes_hash') == digest(episodes)
        and provenance.get('source_program_hash') == program)
    selected = [row for row in reader.lines(old_costs_path) if row.get('phase') == phase]
    models = [row for row in selected if row.get('kind') == 'model']
    steps = [row for row in selected if row.get('kind') == 'env_step']
    historical = provenance.get('historical_costs') or {}
    numeric_usage = all(type(row.get(k)) is int and row[k] >= 0 for row in models for k in ('prompt_tokens', 'completion_tokens'))
    checks['native_cost_rows'] = (bool(models) and numeric_usage and {row.get('task_id') for row in models} == {t['id'] for t in tasks}
        and all(row.get('role') == 'actor' and row.get('ok') is True and not row.get('error')
                and row.get('synthetic') is not True and finite(row.get('elapsed')) and row['elapsed'] >= 0 for row in models)
        and all(row.get('task_id') in {t['id'] for t in tasks} for row in steps)
        and all([row.get('step') for row in steps if row.get('task_id') == ep['task_id']]
                == list(range(len(ep.get('history', [])))) for ep in episodes))
    expected_costs = dict(ledger_path=str(old_costs_path), ledger_sha256=spec['native_costs_sha256'], phase=phase,
        selected_rows_hash=digest(selected), selected_rows=len(selected), actor_model_requests=len(models), env_steps=len(steps),
        prompt_tokens=sum(row.get('prompt_tokens', 0) for row in models) if numeric_usage else None,
        completion_tokens=sum(row.get('completion_tokens', 0) for row in models) if numeric_usage else None,
        actor_seconds=sum(row['elapsed'] for row in models if finite(row.get('elapsed'))), measurement_seconds=original.get('seconds'))
    checks['historical_cost_summary'] = (finite(original.get('seconds')) and original['seconds'] >= 0
                                       and all(historical.get(k) == v for k, v in expected_costs.items()))
    accesses = reader.lines(folder / 'measurement_access.jsonl')
    imported = [row for row in accesses if row.get('operation') == 'import']
    expected_access = dict(key=key, phase=phase, physical=False, operation='import', episodes=len(episodes), seconds=0.0,
                           reuse_receipt='baseline_reuse.json', reuse_receipt_hash=digest(audit))
    checks['import_access_receipt'] = (imported == [expected_access]
        and all(row.get('physical') is False for row in accesses if row.get('key') == key))
    verified = all(value is True for value in checks.values())
    result.update(status='imported' if verified else 'invalid_import', verified=verified,
        imported_real_rows=len(episodes) if verified else 0, recorded_import_rows=len(episodes),
        source_artifact=str(old_measurement_path), source_binding_hash=original_state.get('binding_hash'),
        destination_measurement=reader.label(cache_path), provenance=provenance,
        original_native=episode_summary(episodes, [t['id'] for t in tasks]),
        historical_costs_for_this_phase_only=model_summary(selected), historical_native_costs=expected_costs,
        integrity_findings=[dict(path=reader.label(audit_path), check=key) for key, value in checks.items() if value is False])
    return result


def linked_collection_origin(reader, environment, spec):
    """Verify the original collection snapshot, without comparing to changed code."""
    folder = reader.root / 'collection' / environment
    if not spec and not folder.is_symlink():
        return None
    spec = spec or {}
    origin = Path(spec.get('source_collection', folder.resolve()))
    binding = reader.json(origin / 'binding.json', {}) or {}
    manifest = reader.json(spec['snapshot_manifest'], {}) if spec.get('snapshot_manifest') else {}
    manifest = manifest or {}
    hashes = manifest.get('source_hashes', {})
    archive_hash = None
    archive_matches = None
    if spec.get('original_code_snapshot'):
        path = Path(spec['original_code_snapshot'])
        try:
            archive_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with zipfile.ZipFile(path) as archive:
                archive_matches = bool(hashes) and all(
                    hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in hashes.items())
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            reader.warnings.append(dict(file=str(path), error=str(exc)))
            archive_matches = False
    required = {'phase1_codegen/' + name: value for name, value in binding.get('own_code', {}).items()}
    if binding.get('executor_bridge'):
        required['evomem/alma_backend.py'] = binding['executor_bridge']
    return dict(source_collection=str(origin), specification=spec,
                resolved_collection_matches=folder.resolve() == origin.resolve(),
                same_environment=binding.get('environment') == environment if binding else None,
                new_model_execution_is_false=spec.get('new_model_execution') is False,
                snapshot_hash_matches=(archive_hash == spec.get('code_snapshot_sha256') == manifest.get('zip_hash'))
                                      if archive_hash else None,
                snapshot_members_match_manifest=archive_matches,
                executed_code_matches_snapshot=bool(required) and all(hashes.get(k) == v for k, v in required.items())
                                              if binding and manifest else None,
                note='Original native collection and original code snapshot. Reuse does not rerun these episodes in this root.')


def linked_adaptation(reader, environment, role, record, tasks, origin_audit):
    origin = Path(origin_audit['source_collection'])
    binding = reader.json(origin / 'binding.json', {}) or {}
    selected = record.get('episodes', [])
    ids = [t['id'] for t in tasks]
    selected_ids = set(ids)
    nested_spec = record.get('reuse_provenance')
    nested = imported_adaptation(reader, environment, role, record, nested_spec, tasks) if nested_spec else None
    # An already imported role has its cache and model calls in an earlier root.
    caches = []
    if not nested:
        for task, episode in zip(tasks, selected):
            cache = reader.json(origin / 'episodes' / (digest(task) + '.json'), {}) or {}
            caches.append(cache.get('task_hash') == digest(task)
                          and cache.get('binding_hash') == digest(binding) and cache.get('episode') == episode)
    costs = [r for r in reader.lines(origin / 'runner/native/costs.jsonl')
             if r.get('task_id') in selected_ids and r.get('phase') == role]
    summary = episode_summary(selected, ids)
    valid = bool(tasks) and all(e.get('task_id') in selected_ids for e in selected)
    return dict(environment=environment, role=role, status='imported' if selected else 'planned_not_yet_imported',
                source_collection=str(origin), source_artifact=str(origin / (role + '.json')),
                provenance=origin_audit['specification'], available_original_evidence=summary,
                imported_real_rows=summary['real_rows'], newly_executed_here=False,
                artifact_hash_matches=None, binding_hash_matches=None,
                original_task_manifest_matches=(binding.get('tasks') or {}).get(role) == tasks if binding else None,
                original_episode_caches_match=(nested['original_episode_caches_match'] if nested else
                                              valid and len(caches) == len(selected) and all(caches)) if selected else None,
                imported_episodes_equal_original=True if selected else None,
                nested_original_evidence=nested,
                historical_costs_for_this_role_only=(nested['historical_costs_for_this_role_only'] if nested else model_summary(costs)),
                complete=record.get('complete') is True and summary['complete'],
                cost_scope='Historical source role only; linked artifacts and earlier nested imports are counted once, never as new execution.')


def shared_memory_audit(reader, environment, manifests, collection_records, source, targets, generalization):
    """Bind mechanism inheritance and ordered shared evidence independently."""
    complete = all((collection_records.get(role) or {}).get('complete') is True
                   and episode_summary((collection_records.get(role) or {}).get('episodes', []),
                                       [t['id'] for t in manifests[role]])['complete']
                   for role in ('source_adaptation', 'target_adaptation'))
    logs = {role: [dict(e, environment=environment) for e in (collection_records.get(role) or {}).get('episodes', [])]
            for role in ('source_adaptation', 'target_adaptation')}
    shared = logs['source_adaptation'] + logs['target_adaptation']
    expected_families = sorted({t['family'] for role in ('source_adaptation', 'target_adaptation') for t in manifests[role]})
    provenance = []
    for role in ('source_adaptation', 'target_adaptation'):
        original = (collection_records.get(role) or {}).get('episodes', [])
        provenance += [e.get('environment', environment) == environment and e.get('family') == t.get('family')
                       and e.get('group_id') == t.get('group_id')
                       for e, t in zip(original, manifests[role])]
    source_state = reader.json(reader.root / 'source' / environment / 'state.json', {}) or {}
    source_hash = source.get('current_program')
    source_text = reader.text(reader.root / 'source' / environment / 'programs' / (str(source_hash) + '.py'))
    expected_inherited = dict(source_environment=environment, source_state_hash=digest(source_state),
                              inherited_program_hash=source_hash, source_adaptation_hash=digest(logs['source_adaptation']))
    source_em = reader.json(reader.root / 'transfer' / (environment + '.json'), []) or []
    checks = dict(adaptation_native_provenance_matches=all(provenance) if complete else None,
                  source_program_file_matches=hashlib.sha256(source_text.encode()).hexdigest() == source_hash
                                               if source_text is not None else None)
    source_plan, source_binding = source.get('frozen_plan'), source.get('binding') or {}
    checks['source_uses_source_adaptation_only'] = (source_plan.get('adaptation_hash') == digest(logs['source_adaptation'])
                                                   and source_binding.get('logs_hash') == digest(logs['source_adaptation'])) \
                                                  if source_plan and source_binding and complete else None
    target_checks = {}
    for condition in CONDITIONS:
        run = targets[condition]
        plan, binding = run.get('frozen_plan'), run.get('binding') or {}
        target_checks[condition] = dict(
            inherits_current_source_program=(run.get('initial_program') == plan.get('initial_program_hash') == source_hash)
                                             if plan and binding and source_hash else None,
            inherited_checkpoint_matches=(plan.get('inherited_checkpoint') == expected_inherited)
                                         if plan and source_state and complete else None,
            shared_ordered_logs_match=(plan.get('adaptation_hash') == binding.get('logs_hash') == digest(shared)
                                      and binding.get('ordered_logs_hash') == digest(json.dumps(shared, ensure_ascii=False, sort_keys=False)))
                                     if plan and binding and complete else None,
            environment_memory_scope=(plan.get('memory_scope') == binding.get('context', {}).get('memory_scope') == 'environment'
                                      and binding.get('context', {}).get('environment') == environment)
                                     if plan and binding else None,
            shared_family_declaration=(plan.get('task_memory_families') == expected_families
                                       and plan.get('task_memory_shared_across_types') is True) if plan else None,
            expected_source_em_matches=(binding.get('source_memory_hash') == digest(
                [] if condition == 'no_external_em' else source_em)) if binding else None)
    heldout = generalization.get('binding') or {}
    checks['generalization_shared_logs_match'] = heldout.get('logs_hash') == digest(shared) if heldout and complete else None
    checks['generalization_checkpoint_matches'] = heldout.get('inherited_checkpoint') == expected_inherited \
                                                 if heldout and source_state and complete else None
    checks['generalization_versions_match'] = heldout.get('versions') == {
        INHERITED: source_hash, **{c: targets[c].get('current_program') for c in CONDITIONS}} if heldout and source_hash else None
    checks['generalization_environment_scope'] = (heldout.get('memory_scope') == 'environment'
                                                  and heldout.get('task_memory_families') == expected_families) if heldout else None
    checks['generalization_sealed'] = heldout.get('sealed_from_review_and_evolver') is True if heldout else None
    checks['generalization_no_heldout_writes'] = generalization.get('no_heldout_writes')
    heldout_tasks = manifests['target_generalization']
    search_tasks = [t for role in ROLES if role != 'target_generalization' for t in manifests[role]]
    checks['heldout_id_and_group_disjoint'] = all(not {t.get(key) for t in heldout_tasks} & {t.get(key) for t in search_tasks}
                                                for key in ('id', 'group_id')) if heldout_tasks and search_tasks else None
    return dict(checks=checks, targets=target_checks, source_log_count=len(logs['source_adaptation']),
                target_log_count=len(logs['target_adaptation']), shared_log_count=len(shared),
                shared_log_hash=digest(shared) if complete else None, expected_families=expected_families,
                interpretation='Both target conditions inherit the source mechanism and replay identical source-then-target task evidence. Only access to source-type Evolution Memory differs; this is not total transfer versus no transfer. The bank is rebuilt by each mechanism, not asserted byte-identical to the earlier source-only bank.')


def build_report(root):
    reader = Reader(root)
    result = dict(schema=2, root=str(reader.root), generated_at=datetime.now(timezone.utc).isoformat(),
                  intended_protocol='within_environment_shared_task_and_evolution_memory; inherited_source_mechanism; both memory libraries independent across ALFWorld and TextWorld',
                  snapshot_note='Read-only live-file snapshot; stages may advance while files are read',
                  status='in_progress', datasets={}, collection={}, collection_origins={}, imported_evidence=[],
                  sources={}, transfer={}, targets={}, generalization={}, shared_memory_audit={}, source_baseline_reuse={})
    origins = reader.json(reader.root / 'collection_origins.json', {}) or {}
    baseline_origins = (reader.json(reader.root / 'protocol.json', {}) or {}).get('source_baseline_origins') or {}
    for environment in ENVIRONMENTS:
        manifests = {role: reader.json(reader.root / 'datasets' / environment / (role + '.json'), []) or [] for role in ROLES}
        result['datasets'][environment] = {role: len(tasks) for role, tasks in manifests.items()}
        result['collection'][environment] = {}
        collection_records = {}
        origin_audit = linked_collection_origin(reader, environment, origins.get(environment))
        if origin_audit:
            result['collection_origins'][environment] = origin_audit
        reuse_plan = reader.json(reader.root / 'collection' / environment / 'reuse.json', {}) or {}
        for role in ('source_adaptation', 'target_adaptation'):
            record = reader.json(reader.root / 'collection' / environment / (role + '.json'), {}) or {}
            collection_records[role] = record
            result['collection'][environment][role] = episode_summary(record.get('episodes', []), [t['id'] for t in manifests[role]])
            result['collection'][environment][role]['complete'] &= bool(manifests[role]) and record.get('complete') is True
            imported = (linked_adaptation(reader, environment, role, record, manifests[role], origin_audit)
                        if origin_audit else imported_adaptation(reader, environment, role, record,
                                           record.get('reuse_provenance') or reuse_plan.get(role), manifests[role]))
            result['collection'][environment][role]['origin'] = ('imported' if imported and imported['status'] == 'imported'
                                                                else 'this_root' if record else 'not_available')
            if imported:
                result['imported_evidence'].append(imported)
        records = reader.json(reader.root / 'transfer' / (environment + '.json'), []) or []
        provenance = reader.json(reader.root / 'transfer' / (environment + '.provenance.json'), {}) or {}
        source_families = sorted({t.get('family') for t in manifests['source_exploration'] if t.get('family')})
        target_families = sorted({t.get('family') for t in manifests['target_exploration'] if t.get('family')})
        source_environment = provenance.get('source_environment')
        source_state = (reader.json(reader.root / 'source' / source_environment / 'state.json', {}) or {}) if source_environment in ENVIRONMENTS else {}
        source_events = []
        if source_state:
            for receipt in source_state.get('history', []):
                original = reader.json(reader.root / 'source' / source_environment / receipt['path'], {}) or {}
                if original.get('evolution_event'):
                    source_events.append(original['evolution_event'])
        result['transfer'][environment] = dict(records=len(records), record_ids=[r.get('id') for r in records],
            provenance=provenance,
            expected_source_families=source_families, expected_target_families=target_families,
            all_new_code_trials=all(r.get('kind') == 'code_change_trial' for r in records) if records else None,
            source_environment_counts=dict(Counter(r.get('observed_context', {}).get('environment', 'unknown') for r in records)),
            source_family_counts=dict(Counter(r.get('observed_context', {}).get('family', 'unknown') for r in records)),
            same_environment=(source_environment == environment and provenance.get('target_environment') == environment
                              and all(r.get('observed_context', {}).get('environment') == environment for r in records)) if provenance and records else None,
            different_task_types=(bool(source_families) and bool(target_families) and not set(source_families) & set(target_families)
                                  and all(r.get('observed_context', {}).get('family') in source_families for r in records)) if records else None,
            observed_protocol=('cross_environment' if source_environment != environment else 'within_environment') if provenance else None,
            source_state_hash_matches=digest(source_state) == provenance.get('source_state_hash') if source_state and provenance else None,
            records_match_committed_source_trials=records == source_events if records and source_state else None,
            provenance_ids_match=provenance.get('record_ids') == [r.get('id') for r in records] if provenance else None)
        result['sources'][environment] = search_report(reader, reader.root / 'source' / environment, [])
        result['source_baseline_reuse'][environment] = source_baseline_reuse(
            reader, environment, baseline_origins.get(environment), manifests['source_exploration'])
        result['targets'][environment] = {condition: search_report(reader, reader.root / 'target' / environment / condition,
                                                                 records if condition == 'cross_type_em' else []) for condition in CONDITIONS}
        plans = [result['targets'][environment][c]['frozen_plan'] for c in CONDITIONS]
        fields = ('context', 'seed', 'rounds', 'acceptance_policy', 'exploration', 'adaptation_hash',
                  'initial_program_hash', 'native_step_limit', 'memory_scope', 'inherited_checkpoint',
                  'task_memory_shared_across_types', 'task_memory_families', 'task_memory_order')
        result['targets'][environment]['matching'] = {
            key: key in plans[0] and key in plans[1] and plans[0][key] == plans[1][key]
                 if all(plans) else None for key in fields}
        result['targets'][environment]['matching']['interpretation'] = 'Matched planned search inputs/budgets do not imply equal realized token costs'
        result['generalization'][environment] = heldout_report(reader, environment)
        result['shared_memory_audit'][environment] = shared_memory_audit(
            reader, environment, manifests, collection_records, result['sources'][environment],
            result['targets'][environment], result['generalization'][environment])
    # Count each canonical ledger exactly once; never add logical result/cached
    # episode arrays or the older raw episodes.jsonl ledger to physical totals.
    native_paths = sorted({p for p in reader.root.glob('**/native/canonical_episodes.jsonl')
                           if 'alma_audit' not in p.parts and local_artifact(p, reader.root)
                           and not p.is_relative_to(reader.root / 'diagnostics')
                           and not (p.is_relative_to(reader.root / 'collection') and origins)})
    all_native, all_costs, ledgers = [], [], {}
    cost_paths = set()
    for path in native_paths:
        rows = reader.lines(path)
        all_native.extend(rows)
        ledgers[reader.label(path)] = episode_summary(rows)
        cost_paths.add(path.with_name('costs.jsonl'))
    for pattern in ('source/*/*_costs.jsonl', 'target/*/*/*_costs.jsonl', '**/native/costs.jsonl'):
        cost_paths.update(p for p in reader.root.glob(pattern) if 'alma_audit' not in p.parts
                          and local_artifact(p, reader.root)
                          and not p.is_relative_to(reader.root / 'diagnostics')
                          and not (p.is_relative_to(reader.root / 'collection') and origins))
    for path in sorted(cost_paths):
        all_costs.extend(reader.lines(path))
    result['physical_native'] = dict(total=episode_summary(all_native), ledgers=ledgers,
                                    imported_existing_real_rows=sum(item['imported_real_rows'] for item in result['imported_evidence']),
                                    counting_rule='Only locally resolved canonical JSONL rows count as new recorded physical attempts. External symlink collections and their costs, imported evidence, and cached logical results are excluded; imported roles count once.')
    result['physical_native']['imported_adaptation_real_rows'] = result['physical_native']['imported_existing_real_rows']
    result['physical_native']['imported_source_baseline_real_rows'] = sum(
        item['imported_real_rows'] for item in result['source_baseline_reuse'].values())
    result['physical_native']['imported_existing_total_real_rows'] = (
        result['physical_native']['imported_adaptation_real_rows'] + result['physical_native']['imported_source_baseline_real_rows'])
    result['model_costs'] = dict(by_role=model_summary(all_costs), ledger_paths=[reader.label(p) for p in sorted(cost_paths) if p.exists()],
                                 sharing_note='Collection/source costs are separate from per-target search; generalization can reuse identical program/task measurements')
    capability_stages = [result['sources'][e]['memory_model_calls'] for e in ENVIRONMENTS]
    capability_stages += [result['targets'][e][c]['memory_model_calls'] for e in ENVIRONMENTS for c in CONDITIONS]
    capability_stages += [result['generalization'][e]['memory_model_calls'] for e in ENVIRONMENTS]
    capability_rows = [row for rows in reader.memory_model_rows.values() for row in rows]
    result['memory_model_calls'] = dict(totals=memory_model_counts(capability_rows),
        by_environment_model_phase=memory_model_groups(capability_rows),
        ledger_paths=[s['ledger_path'] for s in capability_stages if s['available']],
        excluded_synthetic_logical_requests=sum(s['excluded_synthetic']['logical_requests'] for s in capability_stages),
        excluded_roots=['diagnostics'],
        formal_stage_scope=['source/<environment>', 'target/<environment>/<condition>', 'generalization/<environment>'],
        verified_trace_call_references=sum(s['mechanism_execution']['verified_trace_call_references'] for s in capability_stages),
        interpretation='Memory model costs are additional to Actor/Evolver/Reviewer costs. Cache hits are logical uses, not new inference; repeated trace references are not added to broker counts. Real-model development probes and synthetic integration checks are not formal task results.')
    result['transfer_cases'] = []
    for environment in ENVIRONMENTS:
        for trial in result['targets'][environment]['cross_type_em']['trials']:
            shown = trial['actual_model_exposure']['source_ids_shown_to_evolver']
            same_environment_cross_type = bool(shown) and all(
                trial['source_record_environments'].get(rid) == environment
                and trial['source_record_families'].get(rid) is not None
                and trial['target_family'] is not None
                and trial['source_record_families'][rid] != trial['target_family'] for rid in shown)
            if shown and trial['ast_changed'] and same_environment_cross_type:
                result['transfer_cases'].append(dict(target_environment=environment, round=trial['round'],
                    source_ids_shown=shown, self_attributed_ids=trial['self_reported_source_usage_ids'],
                    source_environments=trial['source_record_environments'], changed_functions=trial['changed_functions'],
                    source_families=trial['source_record_families'], target_family=trial['target_family'],
                    same_environment_cross_type=True,
                    hypothesis=trial['hypothesis'], target_code_diff=trial['code_diff'],
                    source_evidence_file='transfer/' + environment + '.json',
                    assessment=trial['post_review'].get('assessment'), accepted=trial['accepted'],
                    exploration=trial['exploration'], evidence_path=trial['evidence_path'],
                    interpretation='Traceable exposure followed by target code change; self-attribution and temporal order are not causal proof'))
    complete = all(result['sources'][e]['complete'] and all(result['targets'][e][c]['complete'] for c in CONDITIONS)
                   and result['generalization'][e]['status'] == 'complete' for e in ENVIRONMENTS)
    integrity = [finding for stage in capability_stages for finding in stage['integrity_findings']]
    integrity.extend(finding for stage in result['source_baseline_reuse'].values() for finding in stage['integrity_findings'])
    for environment, origin in result['collection_origins'].items():
        for key in ('resolved_collection_matches', 'same_environment', 'new_model_execution_is_false',
                    'snapshot_hash_matches', 'snapshot_members_match_manifest', 'executed_code_matches_snapshot'):
            if origin[key] is False:
                integrity.append(dict(path='collection_origins.json/' + environment, check=key))
    for imported in result['imported_evidence']:
        for item in [imported] + ([imported['nested_original_evidence']] if imported.get('nested_original_evidence') else []):
            for key in ('artifact_hash_matches', 'binding_hash_matches', 'original_task_manifest_matches',
                        'original_episode_caches_match', 'imported_episodes_equal_original'):
                if item[key] is False:
                    integrity.append(dict(path=item['source_artifact'], check=key))
    for environment in ENVIRONMENTS:
        memory_audit = result['shared_memory_audit'][environment]
        for part, checks in [('shared', memory_audit['checks'])] + list(memory_audit['targets'].items()):
            for key, value in checks.items():
                if value is False:
                    integrity.append(dict(path=environment + '/' + part, check=key))
        for key, value in result['targets'][environment]['matching'].items():
            if value is False:
                integrity.append(dict(path='target/' + environment, check='matched_' + key))
        for key in ('same_environment', 'different_task_types', 'source_state_hash_matches',
                    'records_match_committed_source_trials', 'provenance_ids_match'):
            if result['transfer'][environment][key] is False:
                integrity.append(dict(path='transfer/' + environment + '.json', check=key))
        runs = [result['sources'][environment]] + [result['targets'][environment][c] for c in CONDITIONS]
        for run in runs:
            for key in ('state_binding_hash_matches', 'committed_ancestry_matches', 'separate_role_prompts', 'role_labels_match_ledgers'):
                if run[key] is False:
                    integrity.append(dict(path=run['path'], check=key))
            for trial in run['trials']:
                for key in ('committed_trial_hash_matches', 'parent_source_hash_matches', 'candidate_source_hash_matches',
                            'behavior_receipt_matches', 'readout_counts_match'):
                    if trial[key] is False:
                        integrity.append(dict(path=trial['evidence_path'], check=key))
    result['integrity_findings'] = integrity
    result['status'] = ('complete_with_audit_failures' if integrity else 'complete') if complete else 'in_progress'
    result['audit_warnings'] = reader.warnings
    result['limitations'] = [
        'Reviewer positive/adoption is an opinion-based search decision, not native task improvement.',
        'A source record selected, displayed, or named in a hypothesis does not prove causal use or transfer.',
        'ALFWorld and TextWorld have independent Task Memory and Evolution Memory. Task types inside each environment share both memories and inherit its source-evolved mechanism; cross-environment records are audit violations.',
        'NoExternalEM retains the same inherited source mechanism and shared task evidence as CrossTypeEM; it excludes source-type Evolution Memory only. The comparison does not isolate all benefits of transfer.',
        'InheritedNoTargetEvolution already includes source mechanism evolution and the same shared adaptation evidence. It is not a no-memory or unevolved-mechanism baseline.',
        'Generalization is evaluated after target-type adaptation, on unseen target instances; it is not zero-shot transfer to an unseen task type. ALFWorld uses group/room-disjoint tasks from its official training archive; TextWorld uses the official treasure_hunter evaluation split.',
        'Target banks are rebuilt by replaying source then target adaptation through each program. Shared evidence does not imply all resulting memory states are byte-identical or survive program-defined capacity limits.',
        'Source/target code and trace receipts are audited without executing generated programs again.',
        'Only the explicit held-out paired results compare task outcomes; no statistical significance claim is made.',
        'No dollar price is guessed for the local models; missing usage and unfinished/error receipts remain visible.',
        'Memory model calls are verified through runtime receipts and cache/manifest references, not source keywords. Model availability or invocation does not prove improved memory quality, task rewards, or Evolution Memory transfer.',
        'Capability costs include only formal source/target/generalization stage ledgers. Diagnostics and explicit synthetic records are excluded; historical cached inference is not charged again to a new phase.',
        'Separate role ledgers lack a shared event clock; hash dependencies establish logical flow, not independently verified wall-clock sequencing.',
    ]
    return result


def markdown(report):
    def cell(value):
        if value is None:
            return '未完成/未知'
        if isinstance(value, float):
            return '%.4f' % value
        return str(value).replace('|', '\\|').replace('\n', ' ')
    condition_names = dict(no_external_em='NoExternalEM', cross_type_em='CrossTypeEM',
                           inherited_no_target_evolution='InheritedNoTargetEvolution')
    lines = ['# 两个独立环境内的跨任务类型机制进化：证据报告', '', '状态：`' + report['status'] + '`。生成时间：' + report['generated_at'],
             '', 'ALFWorld 与 TextWorld 的 Task Memory 和 Evolution Memory 均独立。同一环境内，不同任务类型共享两类记忆并继承源任务进化后的机制；两个目标条件使用相同源机制、相同源后目标适配日志。NoExternalEM 只屏蔽源类型 Evolution Memory，仍保留机制继承与共享 Task Memory。积极评审与机制采用不等于任务收益。',
             '', '读取变化统计来自诊断输入；含 Hook 的候选仅覆盖 task_start，不能据此判断后续交互等价或有益。',
             '', '| 环境 | 阶段/条件 | 完成轮次/预算 | 采用数 | AST变化 | 诊断读取变化候选 | 实际展示源经验的轮次 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for env in ENVIRONMENTS:
        for label, run in [('source', report['sources'][env])] + [(c, report['targets'][env][c]) for c in CONDITIONS]:
            trials = run['trials']
            values = [env, condition_names.get(label, label), '%s/%s' % (run['completed_rounds'], run['planned_rounds']), run['accepted_revisions'],
                      sum(t['ast_changed'] is True for t in trials), sum((t['changed_readouts'] or 0) > 0 for t in trials),
                      sum(bool(t['actual_model_exposure']['source_ids_shown_to_evolver']) for t in trials)]
            lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    lines += ['', '## 独立泛化与实际成本', '', '| 环境 | 版本 | 完整任务数/计划 | 原生平均奖励 | 搜索Actor调用 | Evolver/Reviewer调用 | 已记录搜索输入/输出token |',
              '|---|---|---:|---:|---:|---:|---:|']
    for env in ENVIRONMENTS:
        for name, stats in report['generalization'][env]['versions'].items():
            run = report['targets'][env].get(name, {})
            costs = run.get('costs', {})
            values = [env, condition_names.get(name, name), '%s/%s' % (stats['verified_rows'], report['generalization'][env]['expected_tasks']),
                      stats['mean_verified_native_reward'], costs.get('actor', {}).get('model_attempts', 0),
                      '%s/%s' % (costs.get('code_evolver', {}).get('model_attempts', 0), costs.get('mechanism_reviewer', {}).get('model_attempts', 0)),
                      '%s/%s' % (sum(c['known_prompt_tokens'] for c in costs.values()), sum(c['known_completion_tokens'] for c in costs.values()))]
            lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    for env in ENVIRONMENTS:
        pair = report['generalization'][env]['paired_native_results']['no_external_em -> cross_type_em']
        lines += ['', '%s：CrossTypeEM 相对 NoExternalEM 的配对差值 %s，比较状态 `%s`。' %
                  (env, cell(pair.get('mean_delta')), pair['status']), '']
    lines += ['泛化范围：先使用目标类型适配经验（ALFWorld %s 题、TextWorld %s 题），再评测留出新实例，不是对完全未见任务类型的零样本迁移。ALFWorld 留出题来自官方 train archive 内按 group/room 隔离的切分；TextWorld 使用官方 treasure_hunter eval 的 %s 题。' %
              (report['generalization']['alfworld']['scope']['target_type_adaptation_tasks'],
               report['generalization']['textworld']['scope']['target_type_adaptation_tasks'],
               report['generalization']['textworld']['scope']['heldout_target_instances']), '']
    memory_calls = report.get('memory_model_calls')
    if memory_calls is not None:
        lines += ['## 记忆机制内部模型调用', '',
                  '| 环境 | 模型 | 阶段 | 逻辑请求 | 缓存命中 | 后端尝试 | 实际推理尝试 | 完成物理推理 | 失败 | 实测输入/输出token |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in memory_calls['by_environment_model_phase']:
            values = [row['environment'], row['model_id'], row['phase'], row['logical_requests'], row['cache_hits'],
                      row['backend_attempts'], row['inference_attempts'], row['completed_physical'], row['failed'],
                      '%s/%s' % (row['known_input_tokens'], row['known_output_tokens'])]
            lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
        if not memory_calls['ledger_paths']:
            lines.append('尚无正式阶段的记忆模型成本账本；接口探测与 synthetic 集成检查不计入实验结果。')
        lines += ['', '模型调用由 read/write 运行轨迹和缓存收据核对，不以源码关键词认定。缓存命中不重复计推理或 token；调用本身不证明机制改善或经验迁移。逐阶段审计见 JSON 的 memory_model_calls。', '']
    physical = report['physical_native']['total']
    lines += ['', '## 真实执行与未完成部分', '',
              '本目录新记录的物理原生 Episode：%s；唯一任务：%s；错误：%s。包括本目录先导、采集、源搜索、目标搜索和泛化；不重复累计逻辑缓存或导入旧证据。' %
              (physical['real_rows'], physical['unique_task_ids'], physical['errors']), '',
              '各角色 token、失败、缺失 usage、独立 prompt、前后评审引用和接受依据见 report.json。共享采集与源经验成本另列，不能把相同提案预算称为相同实际成本。',
              '已导入既有真实适配证据：%s 条，不计为本目录新增执行或新增成本。' % report['physical_native']['imported_existing_real_rows'], '']
    if report.get('source_baseline_reuse'):
        lines += ['另已核验并导入源机制 baseline 的原生探索结果：%s 条；与适配证据分别统计，历史 Actor token 不计入本版本新推理。' %
                  report['physical_native'].get('imported_source_baseline_real_rows', 0), '']
        for env, item in report['source_baseline_reuse'].items():
            if not item['available']:
                continue
            cost = item.get('historical_costs_for_this_phase_only', {}).get('actor', {})
            lines.append('- %s source baseline：%s，核验导入 %s 条；旧 phase `%s`，历史 Actor 调用 %s，输入/输出 token %s/%s。来源 `%s`；这些历史成本不新增计费。' %
                (env, item['status'], item['imported_real_rows'], item['phase'], cost.get('model_attempts', 0),
                 cost.get('known_prompt_tokens', 0), cost.get('known_completion_tokens', 0), item.get('source_artifact', '未知')))
        lines.append('')
    lines += ['InheritedNoTargetEvolution 表示“继承源机制后，不再做目标进化”，使用相同共享适配证据。泛化阶段冻结机制与记忆库，不将留出任务写回。共享日志顺序、继承 checkpoint、环境隔离、两条件匹配及冻结证据核验见 JSON 的 shared_memory_audit。', '']
    for imported in report['imported_evidence']:
        lines.append('- %s/%s：%s；已导入%s条；原证据 `%s`。原成本只统计该角色对应任务，单列于 JSON。' %
                     (imported['environment'], imported['role'], imported['status'], imported['imported_real_rows'], imported['source_artifact']))
    lines += ['', '## 可追踪的同环境跨任务类型修改', '']
    if not report['transfer_cases']:
        lines.append('尚无同时具备“真实请求展示源经验”和“目标代码 AST 变化”的完整记录。')
    for case in report['transfer_cases']:
        lines += ['- %s 第%s轮（%s → %s）：展示源记录 %s；修改函数 %s；自述使用 %s；采用=%s。证据：`%s`。' %
                  (case['target_environment'], case['round'], ', '.join(sorted(set(case['source_families'].values()))),
                   case['target_family'], ', '.join(case['source_ids_shown']),
                   ', '.join(case['changed_functions']), ', '.join(case['self_attributed_ids']) or '无明确ID',
                   case['accepted'], case['evidence_path'])]
    lines += ['', '这些是可审查的过程个案，不能单凭它们认定经验带来因果收益。', '', '## 证据边界', '']
    lines.extend('- ' + limitation for limitation in report['limitations'])
    if report['audit_warnings']:
        lines += ['', '读取警告：%s 项，详见 report.json；运行中的最后一行可能尚未写完。' % len(report['audit_warnings'])]
    if report['integrity_findings']:
        lines += ['', '协议或证据完整性核验未通过：%s 项，详见 report.json 的 integrity_findings。' % len(report['integrity_findings'])]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--write', action='store_true', help='Write root/report.json and root/report.md; otherwise print JSON only')
    args = parser.parse_args(argv)
    report = build_report(args.root)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if args.write:
        if not args.root.is_dir():
            raise ValueError('Campaign root must already exist')
        for name, content in [('report.json', encoded), ('report.md', markdown(report))]:
            temporary = args.root / (name + '.tmp')
            temporary.write_text(content)
            temporary.replace(args.root / name)
        print(json.dumps(dict(status=report['status'], physical_native=report['physical_native']['total'],
                              reports=[str(args.root / 'report.json'), str(args.root / 'report.md')]), ensure_ascii=False))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
