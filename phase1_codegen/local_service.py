"""Explicit routing to the experiment-owned loopback model services.

This module never starts, restarts, or stops a model server. Physical GPU values
are experiment assignments, not hardware measurements from the HTTP endpoint.
The shared model transport retains its existing implementation.
"""
from pathlib import Path
import os
import ipaddress
from urllib.parse import urlsplit

from evomem.common import BASE_URL, MODEL, MODEL_NAME, ROOT, digest, file_hash, read
from evomem.services import LocalLLM as HistoricalLocalLLM


ALLOWED_ENDPOINTS = {
    'http://127.0.0.1:8140/v1': 0,
    'http://127.0.0.1:8141/v1': 1,
    'http://127.0.0.1:8142/v1': 2,
}


if BASE_URL not in ALLOWED_ENDPOINTS:
    _url = urlsplit(BASE_URL)
    if _url.scheme != 'http' or _url.path != '/v1' or not ipaddress.ip_address(_url.hostname).is_loopback:
        raise ValueError('EVOMEMORY_BASE_URL must be an explicit loopback HTTP /v1 URL')
    ALLOWED_ENDPOINTS[BASE_URL] = None

def verify_model():
    """Verify the local checkpoint against the original frozen model manifest.

    This is a filesystem-only identity check. Small files are hashed and shard
    sizes are compared; it does not claim a fresh full-weight content hash.
    """
    manifest_path = Path(os.environ.get('EVOMEMORY_MODEL_MANIFEST', str(ROOT / 'runs/feasibility_v1/freeze.json')))
    if not manifest_path.is_file():
        raise FileNotFoundError('Register your local checkpoint with python -m scripts.register_model; set EVOMEMORY_MODEL_MANIFEST to its output.')
    model = read(manifest_path)['model']
    if Path(model['model_path']).resolve() != MODEL.resolve():
        raise ValueError('Frozen model is not the requested local Qwen3.8-27B checkpoint')
    for filename, expected in model['small_files'].items():
        if file_hash(MODEL / filename) != expected:
            raise ValueError('Model/tokenizer metadata changed: ' + filename)
    for filename, size in model['weight_sizes'].items():
        if (MODEL / filename).stat().st_size != size:
            raise ValueError('Model weight size changed: ' + filename)
    return model


def validate_base_url(base_url=BASE_URL):
    """Require an explicit allowlisted URL; no aliases or redirected endpoints."""
    if type(base_url) is not str or base_url not in ALLOWED_ENDPOINTS:
        raise ValueError('Allowed experiment loopback endpoints: ' + ', '.join(ALLOWED_ENDPOINTS))
    return base_url


def make_local_llm(tokenizer, ledger, model_fingerprint, base_url=BASE_URL, *, model_name=MODEL_NAME):
    """Construct the unchanged historical client at the selected allowed port.

    Call verify_local_service once before a formal run. Construction itself has
    no network side effects, which also keeps CPU contract tests independent of
    model availability.
    """
    endpoint = validate_base_url(base_url)
    llm = HistoricalLocalLLM(tokenizer, ledger, model_fingerprint, base_url=endpoint, model_name=model_name)
    # Measured generation can take >300s at the unchanged 6144-token limit.
    # This changes transport waiting only, without retries or generation changes.
    import httpx
    llm.client.timeout = httpx.Timeout(600)
    return llm


def verify_local_service(base_url=BASE_URL, client=None, *, model_name=MODEL_NAME, model_path=None):
    """Read /models and validate the expected served model and reported checkpoint.

    This verifies service-advertised identity, not the server process's physical
    GPU or model-weight bytes. The run freeze and owner process receipt provide
    those separate checks. A caller-provided client remains owned by the caller.
    """
    endpoint = validate_base_url(base_url)
    if type(model_name) is not str or not model_name.strip():
        raise ValueError('explicit nonempty model name required')
    if model_name != MODEL_NAME and model_path is None:
        raise ValueError('A nondefault model requires its explicit checkpoint path')
    expected_path = MODEL if model_path is None else Path(model_path)
    owned = client is None
    if owned:
        import httpx
        client = httpx.Client(timeout=10, trust_env=False, follow_redirects=False)
    try:
        response = client.get(endpoint + '/models')
        response.raise_for_status()
        payload = response.json()
        if type(payload) is not dict or type(payload.get('data')) is not list:
            raise ValueError('Model service returned an invalid models response')
        matches = [item for item in payload['data'] if type(item) is dict and item.get('id') == model_name]
        if len(matches) != 1:
            raise ValueError('Model service must advertise exactly one expected model ID: ' + model_name)
        record = matches[0]
        checkpoint = record.get('root')
        if checkpoint is not None and (type(checkpoint) is not str
                                      or Path(checkpoint).resolve() != expected_path.resolve()):
            raise ValueError('Model service reported an unexpected checkpoint root')
        return dict(base_url=endpoint, assigned_physical_gpu=ALLOWED_ENDPOINTS[endpoint],
                    model_name=model_name, reported_checkpoint=checkpoint,
                    checkpoint_root_verified=checkpoint is not None,
                    models_response_hash=digest(payload), identity_source='service-advertised /models')
    finally:
        if owned:
            client.close()
