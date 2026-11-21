"""Synthetic multimodal contracts: no real weights, network, or task episodes."""
import base64
import hashlib
from contextlib import nullcontext
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from phase1_codegen import multimodal
from phase1_codegen.multimodal import _DirectMultimodalModel as LocalMultimodalModel, validate_payload
from phase1_codegen.policy import CapabilityInfrastructureError, Rejected


def embedded_image(size=(64, 64), fmt='PNG'):
    stream = BytesIO()
    Image.new('RGB', size, (32, 96, 160)).save(stream, format=fmt)
    return {'mime_type': 'image/png' if fmt == 'PNG' else 'image/jpeg',
            'data_base64': base64.b64encode(stream.getvalue()).decode('ascii')}


class FakeTensor:
    def __init__(self, rows, dtype='torch.int64'):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.dtype, self.device = dtype, 'cpu'

    def to(self, device):
        self.device = device
        return self

    def tolist(self):
        return self.rows


class FakeProcessor:
    def __init__(self, tokens=8, visual=True):
        self.tokens, self.visual = tokens, visual
        self.templates, self.calls, self.decoded = [], [], []
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return 'Synthetic rendered template'

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        value = {'input_ids': FakeTensor([list(range(self.tokens))]),
                 'attention_mask': FakeTensor([[1] * self.tokens]),
                 'mm_token_type_ids': FakeTensor([[0] * self.tokens])}
        if kwargs.get('images') and self.visual:
            value.update(pixel_values=FakeTensor([[0.25] * 4], dtype='torch.float32'),
                         image_grid_thw=FakeTensor([[1, 2, 2]]))
        return value

    def decode(self, ids, **kwargs):
        self.decoded.append((ids, kwargs))
        return 'Synthetic decoded output.'


class FakeModel:
    def __init__(self, suffix=None):
        self.suffix = [19, 99] if suffix is None else suffix
        self.calls = []
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.device = None
        self.evaluating = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return FakeTensor([kwargs['input_ids'].tolist()[0] + self.suffix])


class MultimodalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-multimodal-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.torch = SimpleNamespace(inference_mode=nullcontext, float32='float32',
                                     set_num_threads=Mock())
        self.torch_patch = patch.dict('sys.modules', {'torch': self.torch})
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def model(self, *, tokens=8, suffix=None, visual=True):
        value = LocalMultimodalModel(self.root)
        value.processor = FakeProcessor(tokens=tokens, visual=visual)
        value.model = FakeModel(suffix=suffix)
        self.addCleanup(value.close)
        return value

    def test_normalizes_optional_fields_and_accepts_embedded_png_or_jpeg(self):
        original = {'prompt': 'Neutral text.'}
        self.assertEqual(validate_payload(original), {'prompt': 'Neutral text.', 'images': [], 'max_tokens': 256})
        self.assertEqual(original, {'prompt': 'Neutral text.'})
        for fmt in ('PNG', 'JPEG'):
            payload = {'prompt': 'Describe visible colors.', 'images': [embedded_image(fmt=fmt)], 'max_tokens': 8}
            self.assertEqual(validate_payload(payload), payload)

    def test_rejects_paths_urls_extra_fields_and_invalid_image_data(self):
        bad_images = [{'url': 'http://example.org/image.png'}, {'path': '/tmp/image.png'},
                      {'mime_type': 'image/svg+xml', 'data_base64': 'AAAA'},
                      {'mime_type': 'image/png', 'data_base64': 'file:///tmp/image.png'},
                      {'mime_type': 'image/png', 'data_base64': 'data:image/png;base64,AAAA'},
                      {'mime_type': 'image/png', 'data_base64': 'not valid base64!!'},
                      {'mime_type': 'image/png', 'data_base64': base64.b64encode(b'not an image').decode()},
                      {'mime_type': 'image/png', 'data_base64': ''},
                      dict(embedded_image(), mime_type='image/jpeg')]
        for record in bad_images:
            with self.subTest(record=repr(record)[:100]), self.assertRaises(Rejected):
                validate_payload({'prompt': 'Neutral text.', 'images': [record]})
        with self.assertRaises(Rejected):
            validate_payload({'prompt': 'Neutral text.', 'path': '/tmp/model'})

    def test_image_count_edges_pixels_and_total_json_budget_are_enforced(self):
        for images in ([embedded_image(), embedded_image()], [embedded_image(size=(513, 1))],
                       [embedded_image(size=(1024, 1024))]):
            with self.subTest(images=len(images)), self.assertRaises(Rejected):
                validate_payload({'prompt': 'Neutral text.', 'images': images})
        with self.assertRaises(Rejected):
            validate_payload({'prompt': 'x' * (128 * 1024)})
        with self.assertRaises(Rejected):
            validate_payload({'prompt': 'x', 'images': [{'mime_type': 'image/png', 'data_base64': 'A' * (128 * 1024)}]})

    def test_payload_requires_finite_json_exact_types_and_bounded_output_tokens(self):
        for payload in ([], {'prompt': ''}, {'prompt': ' '}, {'prompt': 4},
                        {'prompt': 'x', 'images': None}, {'prompt': 'x', 'images': 'image.png'},
                        {'prompt': 'x', 'max_tokens': True}, {'prompt': 'x', 'max_tokens': 0},
                        {'prompt': 'x', 'max_tokens': 257}, {'prompt': 'x', 'max_tokens': float('nan')},
                        {'prompt': 'x', 'max_tokens': float('inf')}, {'prompt': 'x' * 8193}):
            with self.subTest(payload=payload), self.assertRaises(Rejected):
                validate_payload(payload)

    def test_image_and_all_processor_fields_reach_generation_without_sampling_or_thinking(self):
        model = self.model()
        result, usage = model.generate({'prompt': 'Neutral visual question.', 'images': [embedded_image()]})
        self.assertEqual(result, {'text': 'Synthetic decoded output.', 'truncated': False})
        self.assertEqual(usage, {'input_tokens': 8, 'prompt_tokens': 8, 'completion_tokens': 2,
                                 'total_tokens': 10, 'images': 1, 'device': 'cpu'})
        self.assertTrue(model.last_inference_attempted)
        call = model.model.calls[0]
        self.assertTrue({'pixel_values', 'image_grid_thw', 'mm_token_type_ids'} <= set(call))
        self.assertFalse(call['do_sample'])
        self.assertEqual(call['num_beams'], 1)
        self.assertEqual(call['max_new_tokens'], 256)
        self.assertFalse(model.processor.templates[0][1]['enable_thinking'])
        self.assertFalse(model.processor.calls[0]['truncation'])
        self.assertEqual(model.processor.calls[0]['images_kwargs']['size'],
                         {'shortest_edge': 1024, 'longest_edge': 262144})
        self.assertEqual(model.processor.decoded[0][0], [19, 99])

    def test_text_only_uses_same_model_without_visual_inputs(self):
        model = self.model()
        _, usage = model.generate({'prompt': 'Neutral text.'})
        self.assertEqual(usage['images'], 0)
        self.assertNotIn('images', model.processor.calls[0])
        self.assertNotIn('pixel_values', model.model.calls[0])

    def test_actual_token_limit_rejects_before_weights_or_generate_without_truncation(self):
        model = self.model(tokens=2049)
        with patch.object(model, '_load_model') as load, self.assertRaisesRegex(Rejected, 'no truncation'):
            model.generate({'prompt': 'Short text can have a long prepared input.'})
        load.assert_not_called()
        self.assertEqual(model.model.calls, [])
        self.assertFalse(model.last_inference_attempted)
        self.assertFalse(model.processor.calls[0]['truncation'])
        boundary = self.model(tokens=2048)
        _, usage = boundary.generate({'prompt': 'Boundary test.'})
        self.assertEqual(usage['input_tokens'], 2048)

    def test_missing_visual_tensors_and_bad_payload_never_infer(self):
        model = self.model(visual=False)
        with self.assertRaisesRegex(CapabilityInfrastructureError, 'visual tensors'):
            model.generate({'prompt': 'Neutral question.', 'images': [embedded_image()]})
        self.assertFalse(model.last_inference_attempted)
        self.assertEqual(model.model.calls, [])
        with self.assertRaises(Rejected):
            model.generate({'prompt': 'Neutral question.', 'images': [{'path': '/tmp/a.png'}]})
        self.assertFalse(model.last_inference_attempted)

    def test_output_budget_truncation_and_eos_at_boundary_are_distinguished(self):
        model = self.model(suffix=[19, 20])
        value, usage = model.generate({'prompt': 'Neutral text.', 'max_tokens': 2})
        self.assertTrue(value['truncated'])
        self.assertEqual(usage['completion_tokens'], 2)
        ended = self.model(suffix=[19, 99])
        value, _ = ended.generate({'prompt': 'Neutral text.', 'max_tokens': 2})
        self.assertFalse(value['truncated'])
        with self.assertRaisesRegex(CapabilityInfrastructureError, 'output token'):
            model.generate({'prompt': 'Neutral text.', 'max_tokens': 1})

    def test_generate_failure_marks_attempt_and_close_prevents_reuse(self):
        model = self.model()
        with patch.object(model.model, 'generate', side_effect=RuntimeError('synthetic inference failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic inference failure'):
                model.generate({'prompt': 'Neutral text.'})
        self.assertTrue(model.last_inference_attempted)
        model.close()
        self.assertIsNone(model.model)
        self.assertIsNone(model.processor)
        with self.assertRaisesRegex(CapabilityInfrastructureError, 'closed'):
            model.generate({'prompt': 'Neutral text.'})
        self.assertFalse(model.last_inference_attempted)

    def test_loader_uses_local_only_safe_code_cpu_float32_and_four_threads(self):
        processor, loaded = FakeProcessor(), FakeModel()
        auto_processor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        auto_model = SimpleNamespace(from_pretrained=Mock(return_value=loaded))
        transformers = SimpleNamespace(AutoProcessor=auto_processor, AutoModelForImageTextToText=auto_model)
        model = LocalMultimodalModel(self.root)
        self.addCleanup(model.close)
        with patch.dict('sys.modules', {'transformers': transformers}):
            model.generate({'prompt': 'Neutral text.'})
        auto_processor.from_pretrained.assert_called_once_with(str(self.root), local_files_only=True,
                                                               trust_remote_code=False)
        auto_model.from_pretrained.assert_called_once_with(str(self.root), local_files_only=True,
                                                           trust_remote_code=False, dtype='float32',
                                                           attn_implementation='eager')
        self.torch.set_num_threads.assert_called_once_with(4)
        self.assertEqual(loaded.device, 'cpu')
        self.assertTrue(loaded.evaluating)
        with self.assertRaises(ValueError):
            LocalMultimodalModel('http://example.org/model')

    def host(self, response_source):
        folder = self.root / 'helper-source'
        folder.mkdir(exist_ok=True)
        worker = folder / 'worker.py'
        worker.write_text(response_source)
        runtime = {'schema': 1, 'status': 'ready', 'helper_python': sys.executable,
                   'overlay_path': str(self.root)}
        runtime_patch = patch.object(multimodal, 'runtime_manifest', return_value=runtime)
        worker_patch = patch.object(multimodal, 'WORKER', worker)
        runtime_patch.start(); self.addCleanup(runtime_patch.stop)
        worker_patch.start(); self.addCleanup(worker_patch.stop)
        host = multimodal.LocalMultimodalModel(self.root)
        self.addCleanup(host.close)
        return host

    def test_host_reuses_one_lazy_local_helper_and_closes_only_that_process(self):
        host = self.host('''import json,os,sys
for line in sys.stdin:
    request=json.loads(line)
    if request['command']=='close': break
    ident=request['request_id']
    print(json.dumps(dict(event='inference_start',request_id=ident)),flush=True)
    print(json.dumps(dict(event='result',request_id=ident,ok=True,inference_attempted=True,
      value=dict(text=str(os.getpid()),truncated=False),
      usage=dict(input_tokens=8,prompt_tokens=8,completion_tokens=1,total_tokens=9,images=0,device='cpu'),
      runtime_metadata=dict(cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'],threads=os.environ['OMP_NUM_THREADS']))),flush=True)
''')
        self.assertIsNone(host._process)
        a, usage = host.generate({'prompt': 'Neutral text.'})
        first_pid = host._process.pid
        b, _ = host.generate({'prompt': 'Different neutral text.'})
        self.assertEqual(a['text'], b['text'])
        self.assertEqual(first_pid, host._process.pid)
        self.assertTrue(host.last_inference_attempted)
        self.assertEqual(host.last_runtime_metadata, {'cuda_visible': '', 'threads': '4'})
        host.close()
        self.assertIsNotNone(host._process.poll())
        with self.assertRaisesRegex(CapabilityInfrastructureError, 'closed'):
            host.generate({'prompt': 'Neutral text.'})

    def test_host_preserves_no_inference_for_child_token_rejection(self):
        host = self.host('''import json,sys
for line in sys.stdin:
    request=json.loads(line)
    if request['command']=='close': break
    print(json.dumps(dict(event='result',request_id=request['request_id'],ok=False,
      inference_attempted=False,failure_kind='request_contract',error='Rejected: token budget exceeded')),flush=True)
''')
        with self.assertRaisesRegex(Rejected, 'token budget'):
            host.generate({'prompt': 'Neutral text.'})
        self.assertFalse(host.last_inference_attempted)

    def test_host_timeout_records_started_inference_and_does_not_restart(self):
        host = self.host('''import json,sys,time
request=json.loads(sys.stdin.readline())
print(json.dumps(dict(event='inference_start',request_id=request['request_id'])),flush=True)
time.sleep(5)
''')
        # Observe the actual child notification before simulating deadline
        # expiry. A 0.25-second startup deadline races interpreter scheduling
        # and correctly times out *before* inference on a loaded host.
        original_line = host._line
        responses = []
        def notification_then_expired(deadline):
            if responses:
                return original_line(deadline - 11)
            response = original_line(deadline)
            responses.append(response)
            return response
        with patch.object(multimodal, 'HELPER_TIMEOUT_SECONDS', 10), \
                patch.object(host, '_line', side_effect=notification_then_expired):
            with self.assertRaisesRegex(CapabilityInfrastructureError, 'timeout'):
                host.generate({'prompt': 'Neutral text.'})
        self.assertEqual([row['event'] for row in responses], ['inference_start'])
        self.assertTrue(host.last_inference_attempted)
        first_pid = host._process.pid
        with self.assertRaisesRegex(CapabilityInfrastructureError, 'no automatic restart'):
            host.generate({'prompt': 'Other neutral text.'})
        self.assertEqual(first_pid, host._process.pid)

    def test_host_rejects_wrong_response_identity(self):
        host = self.host('''import json,sys
for line in sys.stdin:
    request=json.loads(line)
    if request['command']=='close': break
    print(json.dumps(dict(event='result',request_id=request['request_id']+1,ok=True)),flush=True)
''')
        with self.assertRaisesRegex(CapabilityInfrastructureError, 'identity'):
            host.generate({'prompt': 'Neutral text.'})

    def test_runtime_manifest_rechecks_overlay_and_interpreter_bytes(self):
        runtime = self.root / 'runtime'
        overlay = runtime / 'overlay'
        overlay.mkdir(parents=True)
        interpreter = runtime / 'python'
        interpreter.write_bytes(b'Synthetic interpreter fixture, never executed')
        source = overlay / 'fixture.py'
        source.write_text('synthetic = True\n')
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = dict(schema=1, status='ready', helper_python=str(interpreter),
                        interpreter_sha256=sha(interpreter), overlay_path=str(overlay),
                        overlay_files={'fixture.py': sha(source)}, reused_components={})
        (runtime / 'manifest.json').write_text(json.dumps(manifest))
        with patch.object(multimodal, 'RUNTIME_ROOT', runtime):
            actual = multimodal.runtime_manifest()
            self.assertEqual(actual['manifest_file_sha256'], sha(runtime / 'manifest.json'))
            source.write_text('synthetic = False\n')
            with self.assertRaisesRegex(RuntimeError, 'overlay files changed'):
                multimodal.runtime_manifest()
            source.write_text('synthetic = True\n')
            interpreter.write_bytes(b'Changed')
            with self.assertRaisesRegex(RuntimeError, 'interpreter changed'):
                multimodal.runtime_manifest()


if __name__ == '__main__':
    unittest.main()
