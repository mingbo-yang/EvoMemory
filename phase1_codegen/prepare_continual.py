"""Prepare a fresh continual-memory development plan from task metadata only.

No episode outcomes, historical banks, models, native environments, or GPU state
are opened.  Reusing a previously used split is explicitly development reuse,
not a fresh independent effectiveness experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .protocol import WORKFLOW_VERSION, HOOK_VERSION

ENVIRONMENTS = ('alfworld', 'textworld')
ROLES = ('source_adaptation', 'source_exploration', 'target_adaptation',
         'target_exploration', 'target_generalization')
ROUNDS = 3


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def _tasks(value):
    rows = value.get('tasks') if type(value) is dict else value
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ValueError('Task manifest must be a list or an object containing a tasks list')
    return rows


def split_adaptation(rows, bootstrap):
    if type(bootstrap) is not int or bootstrap not in (1, 2):
        raise ValueError('bootstrap must explicitly be 1 or 2 tasks per type')
    if len(rows) < bootstrap + ROUNDS:
        raise ValueError('Insufficient adaptation tasks for bootstrap plus three nonempty learning blocks')
    remaining = rows[bootstrap:]
    quotient, remainder = divmod(len(remaining), ROUNDS)
    blocks, offset = [], 0
    for index in range(ROUNDS):
        count = quotient + int(index < remainder)
        blocks.append(remaining[offset:offset + count])
        offset += count
    return rows[:bootstrap], blocks


def prepare(from_root, out, *, bootstrap=2):
    """Publish only new metadata, refusing existing destinations or old-run edits."""
    source, destination = Path(from_root).resolve(), Path(out).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Destination must be a new directory outside the source run')
    protocol_path = source / 'protocol.json'
    protocol_bytes = protocol_path.read_bytes()
    original_protocol = json.loads(protocol_bytes)
    capabilities = original_protocol.get('local_model_capabilities', False)
    if type(capabilities) is not bool:
        raise ValueError('local_model_capabilities metadata must be a boolean')
    files, original_hashes, blocks, counts = {}, {'protocol.json': _hash(protocol_bytes)}, {}, {}
    for environment in ENVIRONMENTS:
        blocks[environment], counts[environment] = {}, {}
        for role in ROLES:
            relative = 'datasets/' + environment + '/' + role + '.json'
            raw = (source / relative).read_bytes()
            original_hashes[relative] = _hash(raw)
            rows = _tasks(json.loads(raw))
            if role.endswith('_adaptation'):
                prefix, pieces = split_adaptation(rows, bootstrap)
                files[relative] = prefix
                stage = role.removesuffix('_adaptation')
                blocks[environment][stage] = []
                for index, piece in enumerate(pieces):
                    path = 'datasets/' + environment + '/' + stage + '_learning_%03d.json' % index
                    files[path] = piece
                    blocks[environment][stage].append(path)
                counts[environment][stage] = dict(original_adaptation=len(rows), bootstrap=len(prefix),
                                                   learning_blocks=[len(piece) for piece in pieces])
            else:
                # Preserve exact task metadata and wrapper/asset references.
                files[relative] = raw
                counts[environment][role] = len(rows)
    protocol = dict(version='continual-task-memory-development-v3', task_memory_mode='continual',
                    workflow_version=WORKFLOW_VERSION, memory_hook_version=HOOK_VERSION,
                    mechanism_entrypoint='on_event(state, event, context)',
                    memory_scope='environment', rounds_per_search=ROUNDS, bootstrap_per_type=bootstrap,
                    learning_blocks=blocks, local_model_capabilities=capabilities,
                    conditions=['no_external_em', 'cross_type_em'], acceptance_policy='reviewer_and_reward',
                    seed=11, environment_seed=42, imported_results=False,
                    task_split_status='development_reuse_of_previously_used_task_split',
                    effectiveness_scope='Development validation only; this is not a fresh independent effectiveness experiment.',
                    learning_policy='Only the currently deployed mechanism writes completed own-environment learning episodes. '
                                    'Parent and candidate assessment uses one frozen learning frontier; candidate and heldout trajectories are not accumulated.',
                    transfer_scope='Within each dataset only; source and target task types share accumulated histories and inherit the mechanism.',
                    target_control='Both arms clone the same source-final bank plus target bootstrap, then learn in isolated branches.',
                    heldout_policy='Copied metadata only. Final assessment remains read-only and sealed from evolution and review.',
                    model_capability_policy='Optional existing interfaces only; no mechanism design suggestions.',
                    source_metadata=dict(root=str(source), hashes=original_hashes))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + destination.name + '.metadata-', dir=destination.parent) as temporary:
        staging = Path(temporary)
        for relative, value in files.items():
            path = staging / relative
            if isinstance(value, bytes):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            else:
                _write(path, value)
        _write(staging / 'protocol.json', protocol)
        # This validator examines IDs, groups, families and split metadata only.
        from .campaign import validate_dataset
        validation = {environment: validate_dataset(staging, environment) for environment in ENVIRONMENTS}
        preparation = dict(status='metadata_ready_no_execution', synthetic_results_created=False,
                           workflow_version=WORKFLOW_VERSION, memory_hook_version=HOOK_VERSION,
                           mechanism_entrypoint='on_event(state, event, context)',
                           read_scope=['protocol.json', 'datasets/{environment}/{role}.json'],
                           source=str(source), source_hashes=original_hashes, counts=counts,
                           dataset_validation=validation, development_split_reuse=True,
                           outcomes_read=False, models_called=False, native_episodes_executed=0)
        _write(staging / 'metadata_preparation.json', preparation)
        # Exclusive creation prevents any overwrite of an existing run. Publish
        # protocol last so campaign consumers fail closed during preparation.
        destination.mkdir(exist_ok=False)
        for child in sorted(staging.iterdir(), key=lambda item: item.name == 'protocol.json'):
            os.replace(child, destination / child.name)
    return preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    result = prepare(args.from_root, args.out, bootstrap=args.bootstrap)
    print(json.dumps(dict(out=str(args.out.resolve()), status=result['status'], counts=result['counts']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
