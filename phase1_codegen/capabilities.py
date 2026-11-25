"""Fixed local model API available to generated memory code, with auditable costs.

The catalog describes callable models and JSON contracts, never memory algorithms.
Model inference executes in the trusted host; generated code remains isolated.
Each environment owns its exact-request cache. Cache access cannot enumerate or
retrieve another request, and does not transfer task experience across datasets.
"""
from __future__ import annotations

import copy
import fcntl
import json
import math
from pathlib import Path
import os
import time

from evomem.common import MODEL_NAME, append, digest, file_hash, read, write
from .local_service import validate_base_url
from .multimodal import runtime_manifest as multimodal_runtime_manifest
from .policy import CapabilityContractError, CapabilityInfrastructureError, Rejected, json_copy


MODEL_ROOT = Path(os.environ.get('EVOMEMORY_CAPABILITY_MODEL_ROOT', str(Path(__file__).resolve().parents[1] / 'models'))).expanduser()
EMBEDDING_MODELS = {
    'multilingual-e5-small': (MODEL_ROOT / 'multilingual-e5-small', 384, 512),
    'all-MiniLM-L6-v2': (MODEL_ROOT / 'models--sentence-transformers--all-MiniLM-L6-v2'
                       / 'snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf', 384, 256),
    'sentence-t5-base': (MODEL_ROOT / 'sentence-t5-base', 768, 256),
}
GENERATION_MODEL = MODEL_NAME
MULTIMODAL_MODEL = 'qwen3.5-0.8b-local'
MULTIMODAL_PATH = MODEL_ROOT / 'qwen3.5-0.8b/model'
API_VERSION = 'local-model-json-v1'
LIMITS = dict(max_calls_per_invocation=8, max_embedding_texts_per_call=8,
              max_text_characters=8192, max_generation_input_tokens=2048,
              max_generation_output_tokens=256, request_result_bytes=128 * 1024,
              cpu_threads=4)
GENERATION_SYSTEM = 'You are a text model called by a memory program. Return the requested text. Do not claim that unobserved events occurred.'


def _manifest(path):
    files = [p for p in sorted(path.rglob('*')) if p.is_file()
             and p.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt'}]
    if not files or not any(p.suffix in {'.safetensors', '.bin'} for p in files):
        raise ValueError('Local model files are incomplete: ' + str(path))
    return {str(p.relative_to(path)): file_hash(p) for p in files}


def public_usage(usage):
    allowed = {'input_tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens',
               'local_input_tokens', 'batch_size', 'image_count', 'images'}
    value = {}
    for key in allowed:
        if key in usage:
            if type(usage[key]) is not int or usage[key] < 0:
                raise ValueError('Model usage counts must be nonnegative integers: ' + key)
            value[key] = usage[key]
    if 'device' in usage:
        if usage['device'] != 'cpu':
            raise ValueError('Unexpected local model device in usage receipt')
        value['device'] = 'cpu'
    return value


class LocalModelCapabilities:
    def __init__(self, root, environment, ledger, tokenizer, model_fingerprint, base_url,
                 *, phase='unspecified'):
        if environment not in {'alfworld', 'textworld', 'math', 'apps'}:
            raise ValueError('A capability pool requires one declared dataset')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.environment, self.ledger = environment, Path(ledger)
        self.tokenizer, self.base_url = tokenizer, validate_base_url(base_url)
        self.phase = phase
        self.models, self.client, self.multimodal = {}, None, None
        self._inference_attempted = False
        model_rows = []
        optional = os.environ.get("EVOMEMORY_ENABLE_OPTIONAL_MODELS", "1")
        if optional not in {"0", "1"}:
            raise ValueError("EVOMEMORY_ENABLE_OPTIONAL_MODELS must be 0 or 1")
        enabled = optional == "1"
        for identifier, (path, dimensions, max_tokens) in (EMBEDDING_MODELS.items() if enabled else []):
            manifest = _manifest(path)
            model_rows.append(dict(id=identifier, kind='embedding', path=str(path),
                                   dimensions=dimensions, max_input_tokens_per_text=max_tokens,
                                   model_files=manifest, model_files_hash=digest(manifest),
                                   input=dict(texts='list of 1..8 nonempty strings'),
                                   output=dict(embeddings='one normalized float vector per input string',
                                               input_tokens='one token count per input string'),
                                   normalization=True, decimal_places=6,
                                   input_prefix='caller supplies the complete text; no automatic prefix',
                                   truncation='reject overlength input; never silently truncate', device='cpu'))
        model_rows.append(dict(id=GENERATION_MODEL, kind='text_generation',
                               model_fingerprint=model_fingerprint, endpoint=self.base_url,
                               input=dict(prompt='nonempty string', max_tokens='optional integer 1..256'),
                               output=dict(text='generated string', truncated='boolean'),
                               parameters=dict(temperature=0, top_p=1, seed=42, enable_thinking=False),
                               system=GENERATION_SYSTEM))
        if enabled:
            vision_manifest = _manifest(MULTIMODAL_PATH)
            model_rows.append(dict(id=MULTIMODAL_MODEL, kind='multimodal_generation',
                                   path=str(MULTIMODAL_PATH), model_files=vision_manifest,
                                   model_files_hash=digest(vision_manifest),
                                   input=dict(prompt='nonempty string', images='optional list of at most one {mime_type,data_base64} PNG/JPEG image',
                                              max_tokens='optional integer 1..256'),
                                   output=dict(text='generated string', truncated='boolean'),
                                   image_limits=dict(max_width=512,max_height=512,max_pixels=262144),
                                   max_input_tokens=2048, device='cpu', dtype='float32',
                                   parameters=dict(do_sample=False, enable_thinking=False)))
        self.manifest = dict(api_version=API_VERSION, environment=environment, models=model_rows,
                             limits=LIMITS, no_remote_access=True, cache_scope='exact request within one dataset',
                             multimodal_runtime=multimodal_runtime_manifest() if enabled else None)
        self.fingerprint = digest(self.manifest)
        manifest_path = self.root / 'manifest.json'
        if manifest_path.exists() and read(manifest_path) != self.manifest:
            raise ValueError('Frozen local model capability manifest changed')
        if not manifest_path.exists():
            write(manifest_path, self.manifest)

    def _record_cost(self, row):
        # Different exact requests have different cache locks but share this ledger.
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            append(self.ledger, row)

    def describe(self):
        """Compact operational catalog; do not include mechanism suggestions."""
        rows = [{k: v for k, v in row.items() if k not in {'model_files', 'system'}}
                for row in self.manifest['models']]
        return dict(api='model_call(model_id, payload) -> JSON object', version=API_VERSION,
                    fingerprint=self.fingerprint, models=copy.deepcopy(rows), limits=dict(LIMITS))

    def _validated_payload(self, model_id, payload):
        if type(model_id) is not str or type(payload) is not dict:
            raise Rejected('model_call requires a model identifier and JSON object')
        if model_id not in {row['id'] for row in self.manifest['models']}:
            raise Rejected('model is not enabled in the configured capability catalog')
        payload = json_copy(payload, LIMITS['request_result_bytes'])
        if model_id in EMBEDDING_MODELS:
            texts = payload.get('texts')
            if (set(payload) != {'texts'} or type(texts) is not list or not 1 <= len(texts) <= 8
                    or any(type(s) is not str or not s.strip() or len(s) > LIMITS['max_text_characters'] for s in texts)):
                raise Rejected('embedding input is exactly {texts: list of 1..8 nonempty bounded strings}')
        elif model_id == MULTIMODAL_MODEL:
            from .multimodal import validate_payload
            payload = validate_payload(payload)
        elif model_id == GENERATION_MODEL:
            if set(payload) - {'prompt', 'max_tokens'} or type(payload.get('prompt')) is not str or not payload['prompt'].strip():
                raise Rejected('text model input requires prompt and optional max_tokens only')
            if len(payload['prompt']) > LIMITS['max_text_characters']:
                raise Rejected('text model character budget exceeded')
            maximum = payload.get('max_tokens', LIMITS['max_generation_output_tokens'])
            if type(maximum) is not int or not 1 <= maximum <= LIMITS['max_generation_output_tokens']:
                raise Rejected('text model output budget must be 1..256 tokens')
            payload['max_tokens'] = maximum
        else:
            raise Rejected('model is not in this frozen callable catalog')
        return payload

    def _embedding(self, model_id, payload):
        if model_id not in self.models:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(LIMITS['cpu_threads'])
            path, dimensions, maximum = EMBEDDING_MODELS[model_id]
            expected = next(row['model_files'] for row in self.manifest['models'] if row['id'] == model_id)
            if _manifest(path) != expected:
                raise ValueError('Model files changed after capability freeze')
            model = SentenceTransformer(str(path), device='cpu', local_files_only=True, trust_remote_code=False,
                                        model_kwargs={'dtype': torch.float32})
            model.eval()
            if model.get_sentence_embedding_dimension() != dimensions or model.max_seq_length != maximum:
                raise ValueError('Loaded model dimensions or maximum sequence length differ from frozen catalog')
            self.models[model_id] = model
        model = self.models[model_id]
        lengths = [len(ids) for ids in model.tokenizer(payload['texts'], truncation=False,
                                                     add_special_tokens=True)['input_ids']]
        if any(n > model.max_seq_length for n in lengths):
            raise Rejected('embedding input exceeds the model token limit; no truncation performed')
        self._inference_attempted = True
        vectors = model.encode(payload['texts'], batch_size=8, show_progress_bar=False,
                               normalize_embeddings=True, convert_to_numpy=True)
        value = dict(embeddings=[[round(float(x), 6) for x in vector] for vector in vectors], input_tokens=lengths)
        if len(value['embeddings']) != len(lengths) or any(len(v) != EMBEDDING_MODELS[model_id][1]
                                                        or not all(math.isfinite(x) for x in v) for v in value['embeddings']):
            raise ValueError('Embedding model returned malformed vectors')
        return value, dict(input_tokens=sum(lengths), completion_tokens=0, batch_size=len(lengths), device='cpu')

    def _generation(self, payload):
        import httpx
        if self.client is None:
            self.client = httpx.Client(timeout=180, trust_env=False, follow_redirects=False)
        messages = [dict(role='system', content=GENERATION_SYSTEM), dict(role='user', content=payload['prompt'])]
        count = self.tokenizer.chat_count(messages)
        if count > LIMITS['max_generation_input_tokens']:
            raise Rejected('text model input exceeds fixed 2048-token budget')
        request = dict(model=MODEL_NAME, messages=messages, temperature=0, top_p=1, seed=42,
                       max_tokens=payload['max_tokens'], chat_template_kwargs={'enable_thinking': False})
        self._inference_attempted = True
        response = self.client.post(self.base_url + '/chat/completions', json=request)
        response.raise_for_status()
        response = response.json()
        choice = response['choices'][0]
        if choice.get('finish_reason') not in {'stop', 'length'} or type(choice['message'].get('content')) is not str:
            raise ValueError('Text model did not return a bounded normal completion')
        return dict(text=choice['message']['content'], truncated=choice['finish_reason'] == 'length'), dict(
            request=request, response=response, local_input_tokens=count, **response.get('usage', {}))

    def _multimodal(self, payload):
        from .multimodal import LocalMultimodalModel
        if self.multimodal is None:
            if multimodal_runtime_manifest() != self.manifest['multimodal_runtime']:
                raise ValueError('Multimodal runtime changed after capability freeze')
            expected = next(row['model_files'] for row in self.manifest['models'] if row['id'] == MULTIMODAL_MODEL)
            if _manifest(MULTIMODAL_PATH) != expected:
                raise ValueError('Multimodal model files changed after capability freeze')
            self.multimodal = LocalMultimodalModel(MULTIMODAL_PATH)
        try:
            return self.multimodal.generate(payload)
        finally:
            self._inference_attempted = self.multimodal.last_inference_attempted

    def call(self, model_id, payload, *, operation):
        if operation != 'event':
            raise ValueError('Capabilities may only be called from memory on_event')
        started = time.monotonic()
        # Record even rejected requests, with finite bounded data only.
        original = json_copy(dict(model_id=model_id, payload=payload), LIMITS['request_result_bytes'])
        raw_hash = digest(original)
        try:
            normalized = self._validated_payload(model_id, payload)
        except Exception as exc:
            self._record_cost(dict(kind='memory_model_call', environment=self.environment, phase=self.phase,
                operation=operation, model_id=model_id, request_hash=raw_hash, ok=False,
                cache_hit=False, physical=False, error=type(exc).__name__ + ': ' + str(exc), seconds=time.monotonic()-started))
            raise
        key = digest(dict(fingerprint=self.fingerprint, model_id=model_id, payload=normalized))
        path = self.root / 'cache' / (key + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with (path.with_suffix('.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            hit = path.exists()
            if hit:
                cached = read(path)
                if (cached.get('key') != key or cached.get('model_id') != model_id
                        or cached.get('capability_fingerprint') != self.fingerprint
                        or cached.get('environment') != self.environment or cached.get('request') != normalized
                        or cached.get('record_hash') != digest({k:v for k,v in cached.items() if k!='record_hash'})):
                    raise ValueError('Model capability cache receipt changed')
            else:
                value, usage, error, failure_kind = None, {}, None, None
                self._inference_attempted = False
                try:
                    if model_id in EMBEDDING_MODELS:
                        value, usage = self._embedding(model_id, normalized)
                    elif model_id == MULTIMODAL_MODEL:
                        value, usage = self._multimodal(normalized)
                    else:
                        value, usage = self._generation(normalized)
                    try:
                        value = json_copy(value, LIMITS['request_result_bytes'])
                        usage = json_copy(usage, LIMITS['request_result_bytes'])
                        public_usage(usage)
                    except (ValueError, TypeError) as exc:
                        raise CapabilityInfrastructureError('Malformed model result or usage: ' + str(exc)) from exc
                except Exception as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
                    failure_kind = 'request_contract' if isinstance(exc, Rejected) else 'infrastructure'
                    value, usage = None, {}
                cached = dict(key=key, model_id=model_id, capability_fingerprint=self.fingerprint,
                              environment=self.environment, request=normalized, value=value, usage=usage,
                              error=error, failure_kind=failure_kind, ok=error is None, inference_attempted=self._inference_attempted,
                              seconds=time.monotonic()-started)
                cached['record_hash'] = digest(cached)
                if failure_kind == 'infrastructure':
                    # Preserve this physical attempt. Future authorized runs may make a new
                    # request; a temporary outage is not a deterministic response to cache.
                    path = self.root / 'failures' / (key + '-' + str(time.time_ns()) + '.json')
                write(path, cached)
            receipt = dict(kind='memory_model_call', environment=self.environment, phase=self.phase,
                           operation=operation, model_id=model_id, request_hash=raw_hash, cache_key=key,
                           cache_hit=hit, physical_attempt=not hit,
                           inference_attempted=not hit and cached.get('inference_attempted', False),
                           physical=not hit and cached['ok'] and cached.get('inference_attempted', False),
                           ok=cached['ok'], error=cached['error'], failure_kind=cached.get('failure_kind'),
                           cache_path=str(path), cache_record_hash=cached['record_hash'],
                           capability_fingerprint=self.fingerprint, seconds=time.monotonic()-started,
                           usage=public_usage(cached['usage']) if not hit else {},
                           measurement_kind='local_model_inference')
            self._record_cost(receipt)
            if not cached['ok']:
                error_type = (CapabilityContractError if cached.get('failure_kind') == 'request_contract'
                              else CapabilityInfrastructureError)
                raise error_type('Recorded local model call failed: ' + str(cached['error']))
            return dict(value=json_copy(cached['value'], LIMITS['request_result_bytes']), receipt=receipt)

    def close(self):
        if self.client is not None:
            self.client.close()
        if self.multimodal is not None:
            self.multimodal.close()
        self.models.clear()
