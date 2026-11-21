"""Record checkpoint metadata and shard sizes without loading or changing weights."""
import argparse
from pathlib import Path
from evomem.common import file_hash, write


def register(path, output):
    path = Path(path).expanduser().resolve()
    if not path.is_dir(): raise FileNotFoundError(path)
    small = {p.name: file_hash(p) for p in sorted(path.iterdir())
             if p.is_file() and (p.suffix == '.json' or 'template' in p.name) and p.stat().st_size < 20_000_000}
    weights = {p.name: p.stat().st_size for p in sorted(path.glob('*.safetensors'))}
    if 'config.json' not in small or not weights:
        raise ValueError('Expected a local Hugging Face checkpoint with config.json and safetensors shards')
    value = dict(model=dict(model_path=str(path), small_files=small, weight_sizes=weights),
                 identity_scope='metadata SHA256 and shard sizes, not a full weight-content hash')
    if Path(output).exists(): raise FileExistsError('Use a new manifest path; existing registrations are immutable')
    write(output, value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(); register(a.model, a.output)
    print('Registered checkpoint metadata:', a.output)

if __name__ == '__main__': main()
