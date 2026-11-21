"""Trusted JSON-only bootstrap; generated code never executes before seccomp."""
import ast
import builtins
import json
import os
from pathlib import Path
import resource
import sys
import time

root = str(Path(__file__).resolve().parents[1])
# No cwd, generated files, user site, PYTHONPATH, or credentials are available.
# Keep interpreter stdlib paths and the trusted package root only.
sys.path[:] = [root] + [p for p in sys.path[1:] if p and 'site-packages' not in p]
from phase1_codegen.policy import (
    BudgetedLoops, CAPABILITY_CALLS, CAPABILITY_JSON_BYTES, INPUT_BYTES,
    ITERATION_BUDGET, MAX_MODEL_CALLS, OUTPUT_BYTES, SAFE_CALLS, EXCEPTION_NAMES,
    capability_request, json_copy, validate_output, validate_request, validate_source,
)
from phase1_codegen.isolation import install_filter
from phase1_codegen.behavior_diagnostics import ExecutionTrace, TraceTransformer


class NeedModelCall(BaseException):
    """Trusted suspension; generated programs cannot catch or construct this type."""
    def __init__(self, request):
        self.request = request


class TrustedTermination(BaseException):
    """A host/resource failure which generated data handlers cannot intercept."""


def main():
    raw = sys.stdin.buffer.read(INPUT_BYTES + 1)
    if len(raw) > INPUT_BYTES:
        raise ValueError('input byte budget')
    request = json.loads(raw)
    data = validate_request(request['operation'], request['state'], request.get('experience'), request['context'])
    tree = validate_source(request['source'])
    diagnostics = request.get('diagnostics', False)
    if type(diagnostics) is not bool:
        raise ValueError('diagnostics must be a boolean')
    trace = None
    if diagnostics:
        transformer = TraceTransformer(request['source'])
        tree = transformer.visit(tree)
        trace = ExecutionTrace(transformer.sites, data['operation'])
    tree = ast.fix_missing_locations(BudgetedLoops().visit(tree))
    code = compile(tree, '<generated-task-memory>', 'exec')
    safe = {name: getattr(builtins, name) for name in SAFE_CALLS - CAPABILITY_CALLS}
    safe.update({name: getattr(builtins, name) for name in EXCEPTION_NAMES})
    iteration_budget = request.get('iteration_budget', ITERATION_BUDGET)
    if type(iteration_budget) is not int or not 0 <= iteration_budget <= ITERATION_BUDGET:
        raise ValueError('invalid remaining iteration budget')
    cached_results = json_copy(request.get('capability_results', {}))
    if type(cached_results) is not dict or len(cached_results) > MAX_MODEL_CALLS:
        raise ValueError('invalid model capability result cache')
    for key, value in cached_results.items():
        if len(key) != 64:
            raise ValueError('invalid model capability request hash')
        json_copy(value, CAPABILITY_JSON_BYTES)
    remaining = [iteration_budget]
    calls = []
    def bounded_iter(iterable):
        for value in iterable:
            remaining[0] -= 1
            if remaining[0] < 0:
                raise TrustedTermination('shared iteration budget exceeded')
            yield value
    def model_call(model_id, payload):
        if len(calls) >= MAX_MODEL_CALLS:
            raise TrustedTermination('model_call budget exceeded: at most eight calls per event')
        try:
            item = capability_request(model_id, payload, data['operation'])
        except Exception as exc:
            raise TrustedTermination('model capability contract: ' + str(exc)) from None
        calls.append({key: item[key] for key in ('request_hash', 'model_id', 'operation')})
        if item['request_hash'] not in cached_results:
            raise NeedModelCall(item)
        # Each invocation gets its own value; local mutation cannot poison a
        # repeated call or any parent/broker cache.
        try:
            return json_copy(cached_results[item['request_hash']], CAPABILITY_JSON_BYTES)
        except Exception as exc:
            raise TrustedTermination('model result contract: ' + str(exc)) from None
    safe['model_call'] = model_call
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_BYTES, OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    install_filter()
    namespace = {'__builtins__': safe, '_bounded_iter': bounded_iter}
    if trace is not None:
        namespace.update(_diag_call=trace.call, _diag_value=trace.value,
                         _diag_condition=trace.condition, _diag_return=trace.returned,
                         _diag_locals=builtins.locals, _diag_iter=trace.iterate)
    ready_fd = os.environ.get('EVOMEM_WORKER_READY_FD')
    if ready_fd is not None:
        os.write(int(ready_fd), b'R')
        os.close(int(ready_fd))
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    try:
        exec(code, namespace, namespace)
        result = namespace['on_event'](data['state'], data['experience'], data['context'])
        output = dict(value=validate_output(data['operation'], result))
    except NeedModelCall as missing:
        output = dict(needs_model=missing.request)
    output.update(capability_calls=calls, isolation=dict(
        seccomp=True, no_new_privs=True, cpu_seconds=1,
        address_space_bytes=128 * 1024 * 1024,
        iteration_budget=iteration_budget,
        iterations_used=iteration_budget - remaining[0],
        generated_cpu_seconds=time.process_time() - started_cpu,
        worker_cpu_seconds=time.process_time(),
        worker_execution_wall_seconds=time.monotonic() - started_wall,
        imports_files_network_process_creation=False,
    ))
    if trace is not None and 'value' in output:
        # Diagnostic evidence must fit beside a valid unmodified result, even
        # when generated event output almost fills the existing JSON allowance.
        remaining_output = OUTPUT_BYTES - len(json.dumps(output, ensure_ascii=False,
                                         allow_nan=False, separators=(',', ':')).encode()) - 32
        output['execution_trace'] = trace.export(remaining_output)
    encoded = json.dumps(output, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':')).encode()
    if len(encoded) > OUTPUT_BYTES:
        raise ValueError('output byte budget')
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        sys.stdout.write(json.dumps({'error': type(exc).__name__, 'message': str(exc)[:300]}))
        sys.stdout.flush()
        os._exit(2)
