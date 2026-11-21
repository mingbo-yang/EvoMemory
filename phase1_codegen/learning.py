"""Immutable continual-learning snapshots and exactly-once episode receipts.

These helpers never execute a model/environment or authorize a mechanism.  A
caller validates the complete native episode before publishing its staged bank.
Unknown/pending attempts fail closed instead of silently repeating real work.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

from evomem.common import digest, read
from .protocol import native_executor


def ordered_wire(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def ordered_digest(value):
    return hashlib.sha256(ordered_wire(value).encode()).hexdigest()


def _copy(value):
    return json.loads(ordered_wire(value))


def _check_logs(logs, environment):
    if type(environment) is not str or not environment or type(logs) is not list:
        raise ValueError('Task Memory requires an environment and ordered experience list')
    seen = set()
    for ep in logs:
        if (type(ep) is not dict or ep.get('environment') != environment
                or type(ep.get('task_id')) is not str or not ep['task_id']
                or type(ep.get('family')) is not str or not ep['family'] or ep.get('error')):
            raise ValueError('Only valid own-environment learning experiences may enter Task Memory')
        if ep['task_id'] in seen:
            raise ValueError('Task Memory learning frontier cannot repeat a task ID')
        seen.add(ep['task_id'])
        sealed = ('heldout', 'held_out', 'generalization', 'confirmation', 'final',
                  'eval_in_distribution', 'eval_out_of_distribution')
        if any(any(word in str(ep.get(field, '')).lower() for word in sealed)
               or (field in ('split', 'official_split') and str(ep.get(field, '')).lower() == 'test')
               for field in ('phase', 'data_role', 'split', 'official_split')):
            raise ValueError('Sealed evaluation outcomes cannot enter the learning frontier')


def _check_memory(value, environment):
    if type(value) is not dict or value.get('version') != 'code-memory-state-v1':
        raise ValueError('An exported CodeMemory state snapshot is required')
    payload = {key: item for key, item in value.items() if key != 'snapshot_hash'}
    if value.get('snapshot_hash') != digest(payload):
        raise ValueError('Exported CodeMemory snapshot hash changed')
    if (value.get('environment') != environment or value.get('memory_scope') != 'environment'
            or not value.get('program_hash') or not value.get('memory_hash')):
        raise ValueError('Exported CodeMemory provenance differs from the learning environment')
    for name in ('state_wire', 'context_wire'):
        if type(value.get(name)) is not str or type(json.loads(value[name])) is not dict:
            raise ValueError('Exported CodeMemory must preserve JSON state/context objects')
    if json.loads(value['context_wire']).get('environment') != environment:
        raise ValueError('Exported memory context belongs to another environment')


def make_snapshot(logs, environment, *, memory_snapshot, parent_hash=None):
    """Bind raw cumulative experience to an actual exported deployed bank."""
    logs, memory_snapshot = _copy(logs), _copy(memory_snapshot)
    _check_logs(logs, environment)
    _check_memory(memory_snapshot, environment)
    return dict(version='continual-task-memory-v1', environment=environment,
                logs_wire=ordered_wire(logs), logs_hash=digest(logs),
                ordered_logs_hash=ordered_digest(logs),
                memory_snapshot_wire=ordered_wire(memory_snapshot),
                memory_snapshot_hash=digest(memory_snapshot),
                program_hash=memory_snapshot['program_hash'],
                memory_hash=memory_snapshot['memory_hash'], parent_hash=parent_hash)


def _validate_snapshot(snapshot):
    if type(snapshot) is not dict or snapshot.get('version') != 'continual-task-memory-v1':
        raise ValueError('Unknown continual Task Memory snapshot format')
    logs = json.loads(snapshot['logs_wire'])
    memory = json.loads(snapshot['memory_snapshot_wire'])
    expected = make_snapshot(logs, snapshot['environment'], memory_snapshot=memory,
                             parent_hash=snapshot.get('parent_hash'))
    if snapshot != expected:
        raise ValueError('Continual Task Memory snapshot contents or ordered hashes changed')
    return snapshot


def snapshot_logs(snapshot):
    _validate_snapshot(snapshot)
    return json.loads(snapshot['logs_wire'])


def snapshot_memory(snapshot):
    _validate_snapshot(snapshot)
    return json.loads(snapshot['memory_snapshot_wire'])


def _within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Receipt path escapes its own run')
    return path


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Immutable learning evidence changed: ' + str(path))
        return
    # An interrupted write is deliberately detectable; no silent native retry.
    try:
        with path.open('x') as stream:
            stream.write(ordered_wire(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if read(path) != value:
            raise RuntimeError('Concurrent immutable learning evidence mismatch')


def save_snapshot(root, snapshot):
    _validate_snapshot(snapshot)
    key = digest(snapshot)
    relative = 'task_memory/snapshots/' + key + '.json'
    _write_once(_within(root, relative), snapshot)
    return dict(path=relative, hash=key, logs_hash=snapshot['logs_hash'],
                ordered_logs_hash=snapshot['ordered_logs_hash'])


def load_snapshot(root, reference):
    snapshot = read(_within(root, reference['path']))
    _validate_snapshot(snapshot)
    expected = dict(path='task_memory/snapshots/' + digest(snapshot) + '.json',
                    hash=digest(snapshot), logs_hash=snapshot['logs_hash'],
                    ordered_logs_hash=snapshot['ordered_logs_hash'])
    if reference != expected:
        raise ValueError('Task Memory snapshot reference or artifact hash changed')
    return snapshot


def learning_key(binding, program_hash, snapshot_hash, task, phase):
    return digest(dict(binding_hash=digest(binding), program_hash=program_hash,
                       snapshot_hash=snapshot_hash, task_wire=ordered_wire(task), phase=phase,
                       mode='deployed-online-learning-v1'))


def _request(request):
    if (type(request) is not dict or not request.get('program_hash')
            or not request.get('snapshot_hash') or type(request.get('task')) is not dict
            or type(request.get('phase')) is not str):
        raise ValueError('Learning request must bind program, snapshot, task and phase')
    return dict(request_wire=ordered_wire(request), request_hash=ordered_digest(request))


def _pending_path(root, key):
    if type(key) is not str or len(key) != 64 or any(ch not in '0123456789abcdef' for ch in key):
        raise ValueError('Learning cache key must be a SHA256 digest')
    return _within(root, 'learning/pending/' + key + '.json')


def _receipt_path(root, key):
    _pending_path(root, key)
    return _within(root, 'learning/receipts/' + key + '.json')


def _validate_pending(pending, key, request):
    if (pending.get('version') != 'continual-learning-start-v1' or pending.get('key') != key
            or type(pending.get('timestamp')) not in (int, float)
            or not math.isfinite(pending['timestamp']) or pending['timestamp'] <= 0
            or any(pending.get(k) != v for k, v in _request(request).items())):
        raise ValueError('Learning episode start receipt changed')


def learning_receipt_ref(receipt):
    return dict(path='learning/receipts/' + receipt['key'] + '.json',
                hash=digest(receipt), key=receipt['key'])


def _validate_receipt(receipt, key, request):
    encoded = _request(request)
    if (receipt.get('version') != 'continual-learning-episode-v1' or receipt.get('key') != key
            or any(receipt.get(k) != v for k, v in encoded.items())
            or receipt.get('status') not in ('complete', 'failed')
            or digest(receipt.get('episode')) != receipt.get('episode_hash')
            or ordered_digest(receipt.get('episode')) != receipt.get('ordered_episode_hash')
            or (receipt['status'] == 'complete') != (receipt.get('output_snapshot_ref') is not None)):
        raise ValueError('Learning episode receipt or request binding changed')


def read_learning(root, key, request):
    """Return a recorded success/failure; an unresolved prior attempt is an error."""
    pending = _pending_path(root, key)
    path = _receipt_path(root, key)
    if path.exists():
        receipt = read(path)
        _validate_receipt(receipt, key, request)
        if not pending.exists():
            raise ValueError('Learning episode start receipt changed or is missing')
        start = read(pending)
        _validate_pending(start, key, request)
        if receipt.get('start_hash') != digest(start):
            raise ValueError('Learning episode start hash changed')
        return receipt
    if pending.exists():
        _validate_pending(read(pending), key, request)
        raise RuntimeError('Unresolved learning episode: inspect its native receipts; do not silently retry')
    return None


def begin_learning(root, key, request):
    if read_learning(root, key, request) is not None:
        raise RuntimeError('Learning episode already has a receipt; reuse it without execution')
    _write_once(_pending_path(root, key), dict(version='continual-learning-start-v1', key=key,
                                             timestamp=time.time(), **_request(request)))


def finish_learning(root, key, request, episode, output_snapshot_ref):
    """Persist success or failure and return its immutable {path,hash,key} ref."""
    pending = read(_pending_path(root, key))
    _validate_pending(pending, key, request)
    if output_snapshot_ref is not None:
        snapshot = load_snapshot(root, output_snapshot_ref)
        if (snapshot['parent_hash'] != request['snapshot_hash']
                or snapshot['program_hash'] != request['program_hash'] or episode.get('error')):
            raise ValueError('Successful learning bank does not continue the deployed mechanism')
    if output_snapshot_ref is None:
        episode = dict(episode, score=None, native_reward_verified=False)
    receipt = dict(version='continual-learning-episode-v1', key=key, **_request(request),
                   start_hash=digest(pending),
                   status='complete' if output_snapshot_ref is not None else 'failed',
                   episode=_copy(episode), episode_hash=digest(episode),
                   ordered_episode_hash=ordered_digest(episode), output_snapshot_ref=output_snapshot_ref)
    _validate_receipt(receipt, key, request)
    _write_once(_receipt_path(root, key), receipt)
    return learning_receipt_ref(receipt)


def validate_transition(trial, binding, prior_snapshot, *, root):
    """Validate a complete round's deployed-learning chain, then its code handoff.

    Native correctness and adoption are validated by the engine/RunStore.  This
    independently enforces exact task allocation and that candidate/heldout
    trajectories never enter the persistent learning history.
    """
    if binding.get('task_memory_mode') != 'continual':
        return prior_snapshot
    before = trial.get('task_memory_before')
    if before != prior_snapshot:
        raise ValueError('Trial learning does not start at the authoritative Task Memory pointer')
    current_ref, current = before, load_snapshot(root, before)
    environment = binding['context']['environment']
    parent = trial['parent_hash']
    if current['environment'] != environment or current['program_hash'] != parent:
        raise ValueError('Trial starts from a foreign environment or undeployed program')
    index = trial['round']
    blocks = binding['learning_blocks']
    if type(index) is not int or not 0 <= index < len(blocks):
        raise ValueError('Learning round has no frozen task block')
    tasks, refs = blocks[index], trial.get('learning_receipts', [])
    if len(tasks) != len(refs):
        raise ValueError('Learning receipts must exactly cover this round task block')
    for task, ref in zip(tasks, refs):
        receipt = read(_within(root, ref['path']))
        if ref != learning_receipt_ref(receipt):
            raise ValueError('Learning receipt reference or content hash changed')
        request = json.loads(receipt['request_wire'])
        if read_learning(root, ref['key'], request) != receipt:
            raise ValueError('Learning receipt cannot be verified')
        if (receipt['status'] != 'complete' or request['program_hash'] != parent
                or request.get('binding_hash') != digest(binding)
                or request['snapshot_hash'] != current_ref['hash']
                or ordered_digest(request['task']) != ordered_digest(task)
                or ref['key'] != learning_key(binding, parent, current_ref['hash'], task, request['phase'])):
            raise ValueError('Learning receipt does not follow the frozen deployed task sequence')
        episode = receipt['episode']
        if (episode.get('task_id') != task['id'] or episode.get('group_id') != task['group_id']
                or episode.get('family') != task.get('family') or episode.get('environment') != environment
                or episode.get('error') or episode.get('phase') != request['phase']):
            raise ValueError('Learning episode provenance disagrees with its task/request')
        next_ref = receipt['output_snapshot_ref']
        following = load_snapshot(root, next_ref)
        if (following['parent_hash'] != current_ref['hash'] or following['program_hash'] != parent
                or following['environment'] != environment
                or ordered_digest(snapshot_logs(following)) != ordered_digest(snapshot_logs(current) + [episode])):
            raise ValueError('Learning snapshot may append only its deployed episode once')
        if (episode.get('program_hash') != parent
                or episode.get('memory_hash_before') != current['memory_hash']
                or episode.get('memory_hash_after') != following['memory_hash']):
            raise ValueError('Learning episode does not bind the actual before/after deployed bank')
        if binding.get('synthetic') is not True:
            score = episode.get('score')
            if (episode.get('synthetic') is not False or episode.get('native_reward_verified') is not True
                    or episode.get('executor') != native_executor(environment)
                    or type(score) not in (int, float) or not math.isfinite(score)
                    or episode.get('native_recorder', {}).get('reward') != score
                    or episode.get('environment_seed') != task.get('seed')
                    or episode.get('seed') != 42000 + int(digest(task['id'])[:6], 16) % 10000000
                    or len(episode.get('history', [])) > binding['environment_step_limit']
                    or episode.get('memory_write_calls') != 1
                    or episode.get('generated_memory_write_calls') != 1
                    or episode.get('read_only_memory') is not False):
                raise ValueError('Formal learning transition lacks valid native episode evidence')
        current_ref, current = next_ref, following
    if trial.get('task_memory_evaluation') != current_ref:
        raise ValueError('Parent/candidate evaluation must use the completed frozen learning frontier')
    after_ref = trial.get('task_memory_after')
    after = load_snapshot(root, after_ref)
    deployed = trial['candidate_hash'] if trial['accepted'] else parent
    if (after['environment'] != environment or after['program_hash'] != deployed
            or after['ordered_logs_hash'] != current['ordered_logs_hash']
            or (after_ref != current_ref and after['parent_hash'] != current_ref['hash'])
            or (not trial['accepted'] and after_ref != current_ref)):
        raise ValueError('Mechanism handoff changed history or did not install the selected bank')
    if trial['accepted']:
        behavior = read(_within(root, trial['behavior_path']))
        if (behavior['parent'].get('runtime_snapshot') != snapshot_memory(current)
                or behavior['candidate'].get('runtime_snapshot') != snapshot_memory(after)):
            raise ValueError('Deployment must preserve the exact parent/candidate banks that were assessed')
    return after_ref
