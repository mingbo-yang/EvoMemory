"""Persistent private-runtime CPU worker; one JSON request/result at a time."""
import argparse
from contextlib import redirect_stdout
import json
import sys

from phase1_codegen.policy import Rejected
from phase1_codegen.multimodal import _DirectMultimodalModel, REQUEST_BYTES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    output = sys.stdout

    def emit(value):
        output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        output.flush()

    model = _DirectMultimodalModel(args.model_path)
    original_loader = model._load_model
    hook = None
    visual_calls = []

    def load_with_visual_receipt():
        nonlocal hook
        original_loader()
        if hook is None:
            def visual_pre(module, positional, keywords):
                import torch
                pixels = positional[0]
                grid = keywords.get('grid_thw')
                visual_calls.append(dict(module_class=type(module).__name__,
                    pixels=dict(shape=list(pixels.shape), dtype=str(pixels.dtype), device=str(pixels.device),
                                all_finite=bool(torch.isfinite(pixels).all()), numel=pixels.numel()),
                    grid_thw=grid.tolist() if grid is not None else None))
            hook = model.model.model.visual.register_forward_pre_hook(visual_pre, with_kwargs=True)

    model._load_model = load_with_visual_receipt
    try:
        while True:
            raw = sys.stdin.buffer.readline(REQUEST_BYTES + 4097)
            if not raw:
                break
            if len(raw) > REQUEST_BYTES + 4096 or not raw.endswith(b'\n'):
                raise ValueError('Multimodal worker request is too large or incomplete')
            request = json.loads(raw)
            if request.get('command') == 'close':
                break
            request_id = request.get('request_id')
            if request.get('command') != 'generate' or type(request_id) is not int:
                raise ValueError('Invalid multimodal worker command')
            visual_calls.clear()
            model.on_inference_start = lambda: emit(dict(event='inference_start', request_id=request_id))
            response = dict(event='result', request_id=request_id, ok=False)
            try:
                with redirect_stdout(sys.stderr):
                    value, usage = model.generate(request['payload'])
                response.update(ok=True, value=value, usage=usage)
            except Exception as exc:
                response['failure_kind'] = 'request_contract' if isinstance(exc, Rejected) else 'infrastructure'
                response['error'] = type(exc).__name__ + ': ' + str(exc)[:2000]
            response.update(inference_attempted=model.last_inference_attempted,
                            input_metadata=model.last_input_metadata,
                            visual_forward_calls=list(visual_calls))
            if model.model is not None:
                import torch
                response['runtime_metadata'] = dict(
                    model_class=type(model.model).__name__, processor_class=type(model.processor).__name__,
                    parameter_devices=sorted({str(p.device) for p in model.model.parameters()}),
                    parameter_dtypes=sorted({str(p.dtype) for p in model.model.parameters()}),
                    torch_num_threads=torch.get_num_threads())
            emit(response)
    finally:
        if hook is not None:
            hook.remove()
        model.close()


if __name__ == '__main__':
    main()
