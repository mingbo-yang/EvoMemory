from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LEGACY = Path(os.environ.get('EVOMEMORY_ASSET_ROOT', str(ROOT / 'data'))).expanduser()
MODEL = Path(os.environ.get('EVOMEMORY_MODEL_PATH', str(ROOT / 'models/Qwen3.8-27B'))).expanduser()
DATA = Path(os.environ.get('EVOMEMORY_DATA_ROOT', str(ROOT / 'data'))).expanduser()
RUN = ROOT / 'runs/feasibility_v1'
BASE_URL = os.environ.get('EVOMEMORY_BASE_URL', 'http://127.0.0.1:8140/v1')
MODEL_NAME = os.environ.get('EVOMEMORY_MODEL_NAME', 'qwen3.8-27b-local')

def dumps(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(x):
    return hashlib.sha256(dumps(x).encode()).hexdigest()

def file_hash(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(tmp, p)

def read(p):
    return json.loads(Path(p).read_text())

def append(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        f.write(dumps(value) + '\n'); f.flush()

def events(p):
    if not Path(p).exists(): return []
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def status(stage, **kw):
    row = dict(stage=stage, timestamp=time.time(), **kw)
    write(RUN / 'status.json', row); append(RUN / 'status_history.jsonl', row)
    print(dumps(row), flush=True)
