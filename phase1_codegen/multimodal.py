"""Bounded local CPU image/text inference, independent of memory algorithms.

Only embedded PNG/JPEG bytes are accepted.  The model and processor load from an
existing local directory without remote code, network images, or input truncation.
"""
from __future__ import annotations

import base64
import binascii
import gc
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
import warnings

from .policy import CapabilityInfrastructureError, Rejected, json_copy


REQUEST_BYTES = 128 * 1024
MAX_INPUT_TOKENS = 2048
MAX_OUTPUT_TOKENS = 256
MAX_IMAGES = 1
MAX_IMAGE_EDGE = 512
MAX_IMAGE_PIXELS = 512 * 512
CPU_THREADS = 4
MAX_PROMPT_CHARACTERS = 8192
HELPER_TIMEOUT_SECONDS = 300
RUNTIME_ROOT = Path(__file__).resolve().parents[1] / '.local' / 'multimodal-runtime'
WORKER = Path(__file__).with_name('multimodal_worker.py')
MIME_FORMATS = {'image/png': 'PNG', 'image/jpeg': 'JPEG'}


def _decode_image(record):
    from PIL import Image, ImageOps

    if type(record) is not dict or set(record) != {'mime_type', 'data_base64'}:
        raise Rejected('Each image is exactly {mime_type, data_base64}; paths and URLs are not accepted')
    mime, encoded = record['mime_type'], record['data_base64']
    if type(mime) is not str or mime not in MIME_FORMATS:
        raise Rejected('Only image/png and image/jpeg are supported')
    if type(encoded) is not str or not encoded or len(encoded) > REQUEST_BYTES:
        raise Rejected('Image data must be a nonempty bounded base64 string')
    try:
        raw = base64.b64decode(encoded.encode('ascii'), validate=True)
    except (UnicodeError, ValueError, binascii.Error) as exc:
        raise Rejected('Image data is not valid embedded base64') from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if image.format != MIME_FORMATS[mime]:
                    raise Rejected('Image MIME type does not match decoded image format')
                width, height = image.size
                if (min(width, height) < 1 or max(width, height) > MAX_IMAGE_EDGE
                        or width * height > MAX_IMAGE_PIXELS):
                    raise Rejected('Image dimensions exceed 512 per edge or 262144 pixels')
                if getattr(image, 'n_frames', 1) != 1:
                    raise Rejected('Only a single-frame image is supported')
                image.load()
                return ImageOps.exif_transpose(image).convert('RGB')
    except Rejected:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise Rejected('Image bytes cannot be decoded safely: ' + type(exc).__name__) from exc


def validate_payload(payload):
    """Return normalized finite JSON; validate image bytes without model loading."""
    if type(payload) is not dict:
        raise Rejected('Multimodal input must be a JSON object')
    value = json_copy(payload, REQUEST_BYTES)
    if set(value) - {'prompt', 'images', 'max_tokens'}:
        raise Rejected('Multimodal input permits prompt, images, and max_tokens only')
    if type(value.get('prompt')) is not str or not value['prompt'].strip():
        raise Rejected('Multimodal prompt must be a nonempty string')
    if len(value['prompt']) > MAX_PROMPT_CHARACTERS:
        raise Rejected('Multimodal prompt exceeds 8192 characters')
    images = value.get('images', [])
    if type(images) is not list or len(images) > MAX_IMAGES:
        raise Rejected('Multimodal input allows zero or one embedded image')
    maximum = value.get('max_tokens', MAX_OUTPUT_TOKENS)
    if type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT_TOKENS:
        raise Rejected('Multimodal output budget must be an integer from 1 to 256')
    for record in images:
        image = _decode_image(record)
        image.close()
    value['images'], value['max_tokens'] = images, maximum
    return json_copy(value, REQUEST_BYTES)


class _DirectMultimodalModel:
    """Lazy local processor/model; token rejection precedes weight loading."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise ValueError('Multimodal checkpoint must be an existing local directory')
        self.processor = self.model = None
        self.last_inference_attempted = False
        self.last_input_metadata = None
        self.closed = False
        self.on_inference_start = None

    def _load_processor(self):
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                str(self.path), local_files_only=True, trust_remote_code=False)

    def _load_model(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForImageTextToText

            torch.set_num_threads(CPU_THREADS)
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(self.path), local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, attn_implementation='eager')
            self.model.to('cpu')
            self.model.eval()

    def _prepare(self, payload):
        self._load_processor()
        images = [_decode_image(record) for record in payload['images']]
        try:
            content = [{'type': 'image'} for _ in images]
            content.append({'type': 'text', 'text': payload['prompt']})
            messages = [{'role': 'user', 'content': content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            kwargs = dict(text=[text], return_tensors='pt', truncation=False)
            if images:
                kwargs['images'] = images
                # These Qwen processor size fields are pixel areas, not edges.
                # Patch alignment may round dimensions to multiples of 32.
                kwargs['images_kwargs'] = {'size': {'shortest_edge': 32 * 32,
                                                    'longest_edge': MAX_IMAGE_PIXELS}}
            inputs = self.processor(**kwargs)
            # Qwen3.5 also needs mm_token_type_ids. Retain every processor tensor.
            inputs = {key: tensor.to('cpu') for key, tensor in inputs.items()}
        finally:
            for image in images:
                image.close()
        shape = tuple(inputs['input_ids'].shape)
        if len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
            raise CapabilityInfrastructureError('Multimodal processor returned an invalid input shape')
        if shape[1] > MAX_INPUT_TOKENS:
            raise Rejected('Multimodal actual input exceeds 2048 tokens; no truncation performed')
        if images and ('pixel_values' not in inputs or 'image_grid_thw' not in inputs):
            raise CapabilityInfrastructureError('Image processor omitted required visual tensors')
        self.last_input_metadata = dict(
            input_tokens=int(shape[1]), images=len(images),
            tensors={key: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                           'device': str(tensor.device)} for key, tensor in inputs.items()})
        return inputs

    def generate(self, payload):
        self.last_inference_attempted = False
        self.last_input_metadata = None
        if self.closed:
            raise CapabilityInfrastructureError('Multimodal model is closed')
        payload = validate_payload(payload)
        inputs = self._prepare(payload)
        self._load_model()
        import torch

        self.last_inference_attempted = True
        if self.on_inference_start is not None:
            self.on_inference_start()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=payload['max_tokens'], do_sample=False,
                num_beams=1, num_return_sequences=1, return_dict_in_generate=False,
                use_cache=True)
        rows = outputs.tolist()
        prompt_ids = inputs['input_ids'].tolist()[0]
        if (type(rows) is not list or len(rows) != 1 or type(rows[0]) is not list
                or rows[0][:len(prompt_ids)] != prompt_ids):
            raise CapabilityInfrastructureError('Multimodal model returned an unexpected generated sequence')
        generated = rows[0][len(prompt_ids):]
        if (len(generated) > payload['max_tokens']
                or any(type(token) is not int or token < 0 for token in generated)):
            raise CapabilityInfrastructureError('Multimodal model exceeded its output token contract')
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        if type(text) is not str:
            raise CapabilityInfrastructureError('Multimodal model did not decode to text')
        eos = getattr(self.model.generation_config, 'eos_token_id', None)
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        ended = bool(generated) and generated[-1] in eos_ids
        try:
            value = json_copy(dict(text=text, truncated=len(generated) >= payload['max_tokens'] and not ended),
                              REQUEST_BYTES)
        except Rejected as exc:
            raise CapabilityInfrastructureError('Invalid multimodal generation result') from exc
        count = len(prompt_ids)
        usage = dict(input_tokens=count, prompt_tokens=count, completion_tokens=len(generated),
                     total_tokens=count + len(generated), images=len(payload['images']), device='cpu')
        return value, usage

    def close(self):
        self.processor = self.model = None
        self.last_input_metadata = None
        self.closed = True
        gc.collect()


def runtime_manifest():
    """Frozen private overlay receipt; never install or alter dependencies here."""
    path = RUNTIME_ROOT / 'manifest.json'
    raw = path.read_bytes()
    value = json.loads(raw)
    if (type(value) is not dict or value.get('schema') != 1 or value.get('status') != 'ready'
            or not Path(value.get('helper_python', '')).is_file()
            or not Path(value.get('overlay_path', '')).is_dir()):
        raise RuntimeError('Private CPU multimodal runtime is not ready')
    def sha256(path):
        result = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                result.update(chunk)
        return result.hexdigest()
    if sha256(value['helper_python']) != value.get('interpreter_sha256'):
        raise RuntimeError('Private multimodal interpreter changed')
    overlay = Path(value['overlay_path'])
    files = value.get('overlay_files', {})
    if not files or any(sha256(overlay / name) != expected for name, expected in files.items()):
        raise RuntimeError('Private multimodal overlay files changed')
    for component in value.get('reused_components', {}).values():
        record = component['metadata_record']
        if sha256(record['path']) != record['sha256']:
            raise RuntimeError('Reused multimodal dependency metadata changed')
        for item in component['actual_rehashed_files']:
            if sha256(item['absolute_path']) != item['sha256']:
                raise RuntimeError('Reused multimodal dependency file changed')
    return dict(value, manifest_path=str(path), manifest_file_sha256=hashlib.sha256(raw).hexdigest())


class LocalMultimodalModel:
    """One lazy CPU helper using the pinned, torchvision-compatible environment."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise ValueError('Multimodal checkpoint must be an existing local directory')
        self.runtime = runtime_manifest()
        self.last_inference_attempted = False
        self.last_input_metadata = None
        self.last_visual_forward_calls = []
        self.last_runtime_metadata = None
        self.closed = False
        self._process = self._stderr = None
        self._buffer = b''
        self._request_id = 0

    def _start(self):
        if self._process is not None:
            if self._process.poll() is not None:
                raise CapabilityInfrastructureError('Multimodal helper exited; no automatic restart or retry')
            return
        environment = dict(os.environ)
        environment.pop('PYTHONHOME', None)
        environment.update(
            PYTHONPATH=os.pathsep.join([self.runtime['overlay_path'], str(WORKER.parent.parent)]),
            PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            [self.runtime['helper_python'], '-B', '-s', '-u', str(WORKER), '--model-path', str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            env=environment, cwd=str(WORKER.parent.parent), start_new_session=True)

    def _line(self, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while b'\n' not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    self._stop()
                    raise CapabilityInfrastructureError('Multimodal CPU helper exceeded its 300-second request timeout')
                chunk = os.read(self._process.stdout.fileno(), 65536)
                if not chunk:
                    self._stderr.seek(0)
                    detail = self._stderr.read()[-1000:].decode(errors='replace')
                    raise CapabilityInfrastructureError('Multimodal helper exited without a result; no retry: ' + detail)
                self._buffer += chunk
                if len(self._buffer) > REQUEST_BYTES + 16384:
                    self._stop()
                    raise CapabilityInfrastructureError('Multimodal helper response exceeded bounded JSON size')
        line, self._buffer = self._buffer.split(b'\n', 1)
        try:
            return json_copy(json.loads(line), REQUEST_BYTES + 16384)
        except (ValueError, TypeError) as exc:
            raise CapabilityInfrastructureError('Invalid JSON from multimodal helper') from exc

    def generate(self, payload):
        self.last_inference_attempted = False
        self.last_input_metadata = None
        self.last_visual_forward_calls = []
        if self.closed:
            raise CapabilityInfrastructureError('Multimodal model is closed')
        value = validate_payload(payload)
        self._start()
        self._request_id += 1
        request = dict(command='generate', request_id=self._request_id, payload=value)
        try:
            self._process.stdin.write((json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n').encode())
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CapabilityInfrastructureError('Multimodal helper communication failed; no retry') from exc
        deadline = time.monotonic() + HELPER_TIMEOUT_SECONDS
        while True:
            response = self._line(deadline)
            if type(response) is not dict or response.get('request_id') != self._request_id:
                raise CapabilityInfrastructureError('Multimodal helper response identity mismatch')
            if response.get('event') == 'inference_start':
                self.last_inference_attempted = True
                continue
            if response.get('event') != 'result':
                raise CapabilityInfrastructureError('Unknown multimodal helper response')
            self.last_inference_attempted = bool(response.get('inference_attempted'))
            self.last_input_metadata = response.get('input_metadata')
            self.last_visual_forward_calls = response.get('visual_forward_calls', [])
            self.last_runtime_metadata = response.get('runtime_metadata')
            if response.get('ok') is not True:
                error_type = Rejected if response.get('failure_kind') == 'request_contract' else CapabilityInfrastructureError
                raise error_type('Local CPU multimodal inference failed: ' + str(response.get('error')))
            output, usage = response.get('value'), response.get('usage')
            if (type(output) is not dict or set(output) != {'text', 'truncated'}
                    or type(output['text']) is not str or type(output['truncated']) is not bool
                    or type(usage) is not dict or usage.get('device') != 'cpu'):
                raise CapabilityInfrastructureError('Multimodal helper returned an invalid value or usage')
            for field in ('input_tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'images'):
                if type(usage.get(field)) is not int or usage[field] < 0:
                    raise CapabilityInfrastructureError('Multimodal helper usage must contain finite nonnegative integer counts')
            if (usage['input_tokens'] != usage['prompt_tokens'] or usage['input_tokens'] > MAX_INPUT_TOKENS
                    or usage['completion_tokens'] > value['max_tokens']
                    or usage['total_tokens'] != usage['input_tokens'] + usage['completion_tokens']
                    or usage['images'] != len(value['images'])):
                raise CapabilityInfrastructureError('Multimodal helper returned inconsistent token or image counts')
            try:
                return json_copy(output, REQUEST_BYTES), json_copy(usage, 4096)
            except Rejected as exc:
                raise CapabilityInfrastructureError('Invalid multimodal helper result or usage') from exc

    def _stop(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._process is not None:
            if self._process.poll() is None:
                try:
                    self._process.stdin.write(b'{"command":"close"}\n')
                    self._process.stdin.flush()
                    self._process.wait(timeout=10)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self._stop()
            self._process.stdin.close()
            self._process.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
