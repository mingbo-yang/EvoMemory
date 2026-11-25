"""Private, sequential ALFWorld panels using the original ALMA actor and native reward."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path

from evomem.common import append, digest, file_hash, read, write
from phase1_codegen.hooks import replay_episode
from phase1_codegen.memory import CodeMemory
from phase1_codegen.native import NativeEvaluator
from phase1_codegen.outcomes import candidate_hook_failure
from phase1_codegen.policy import Rejected
from phase1_codegen.program import BASELINE_SOURCE, Program, WorkerStartupError
from . import witness
from .panel import COEVOLUTION_MIXED_10X60_PROTOCOL, NATIVE_PAIRED_PROTOCOLS
from .workflow import LearningMechanismError, PROTOCOL, _sync_directory, _write


def _wire(episode):
    # Replay preserves the order of visible mappings: candidates can iterate over them.
    return json.dumps(episode, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def _archive_episode(attempt, episode, learning_dir=None):
    """Publish the exact bytes seen by replay before a completion can reference them."""
    raw = _wire(episode)
    episode_hash = hashlib.sha256(raw).hexdigest()
    path = attempt / 'episode.json'
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # The canonical JSONL is key-sorted. Keep both the original ordering and
    # the independent runner/Actor receipts durable before publishing complete.json.
    for name in ('canonical_episodes.jsonl', 'actor.jsonl'):
        with (attempt / name).open('rb') as stream:
            os.fsync(stream.fileno())
    _sync_directory(attempt)
    if learning_dir is not None:
        learning_dir.mkdir(exist_ok=True)
        archive = learning_dir / (episode_hash + '.json')
        try:
            os.link(path, archive)
        except FileExistsError:
            if file_hash(archive) != episode_hash:
                raise ValueError('Conflicting learning episode archive')
        _sync_directory(learning_dir)
        _sync_directory(learning_dir.parent)
    return episode_hash


def _prefix(previous, episode_hash):
    return digest([previous, episode_hash])


def learning_failure_evidence(directory, result):
    row = result['row']
    return dict(task_id=row['task_id'], native_score=None, episode_hash=row['episode_hash'],
                candidate_error=row['candidate_error'], result_hash=digest(result),
                receipt=f'native/episodes/{directory.name}/learning_failure.json')


def _candidate_error(ep):
    rejected, = (e for e in ep['memory_hook']['events'] if e.get('status') == 'rejected')
    message = rejected['error']
    return dict(event=rejected['event']['type'], step=rejected['event']['step'], error_type=rejected['error_type'],
                message=message[:1024], omitted_chars=max(0, len(message)-1024), error_hash=digest(message))


def _evidence(ep, before, after, actors, *, episode_hash=None):
    hook = ep.get('memory_hook') or {}
    previous, changes, samples = {}, 0, []
    for item in hook.get('injections', []):
        if not item.get('actor_delivery_verified'):
            continue
        memory = item['memory']
        if memory != previous:
            changes += 1
            if len(samples) < 2:
                rows = memory.get('entries', [])
                samples.append(dict(step=item['step'], ids=[r['id'] for r in rows],
                                    sample=rows[0]['experience'][:240] if rows else ''))
        previous = memory
    history = ep.get('history') or []
    cost = dict(actor_requests=len(actors),
                actor_calls=sum(r.get('kind') == 'model' for r in actors),
                prompt_tokens=sum(r.get('prompt_tokens') or 0 for r in actors),
                completion_tokens=sum(r.get('completion_tokens') or 0 for r in actors),
                missing_usage=sum(r.get('prompt_tokens') is None or r.get('completion_tokens') is None
                                  for r in actors if r.get('kind') == 'model'),
                hook_model_calls=hook.get('usage', {}).get('model_calls', 0))
    evidence = dict(goal=ep.get('goal', '')[:300], steps=len(history),
                    first_action=history[0]['action'] if history else None,
                    last_action=history[-1]['action'] if history else None,
                    last_observation=history[-1].get('next_observation', '')[:300] if history else None,
                    stored_tokens_before=before['stored_tokens'], stored_tokens_after=after['stored_tokens'],
                    storage_sample_before=before['state_wire'][:240], storage_sample_after=after['state_wire'][:240],
                    exposed_memory_changes=changes, injected_readouts=samples,
                    events=len(hook.get('events', [])), model_calls=cost['hook_model_calls'],
                    review_witness=witness.review_witness(ep, actors, episode_hash=episode_hash, bank=before))
    return evidence, cost


def _verify_completed(ep, task, label, seed, actors, before, bank):
    steps = ep['native_recorder']['steps']
    reward = ep['native_recorder']['reward']
    history = ep['history']
    injections = ep['memory_hook']['injections']
    if not (ep['task_id'] == task['id'] and ep['family'] == task['family']
            and ep['environment'] == 'alfworld' and ep['phase'] == label
            and ep['executor'] == 'ALMA.Agent_Workflow.run_single_task' and ep['synthetic'] is False
            and ep['native_reward_verified'] is True and ep['seed'] == seed
            and ep['environment_seed'] == task['seed'] and ep['score'] == reward and reward in (0, 1)
            and reward == (steps[-1]['scores'] if steps else 0)
            and [r['action'] for r in history] == [s['action_took'] for s in steps]
            and 0 < len(actors) <= NativeEvaluator.step_limit and len(steps) <= len(actors)
            and len(actors) == len(injections)
            and all(r.get('ok') and i['actor_delivery_verified'] and r['request_hash'] == i['actor_request_hash']
                    for r, i in zip(actors, injections))
            and ep['memory_hash_before'] == before['memory_hash'] and ep['memory_hash_after'] == bank.fingerprint
            and ep['memory_hook']['complete'] and ep['memory_hook']['committed']
            and ep['generated_memory_write_calls'] == 1 and ep['program_hash'] == bank.program.hash):
        raise RuntimeError('ALFWorld actor, native reward or terminal memory receipt failed validation')


class Native:
    """Each public bank is a mapping from family to a validated, exact-program snapshot."""

    def __init__(self, root, tasks, tokenizer, model_fingerprint, *, base_url, model_name, capabilities=None,
                 learning_archive=None, protocol=PROTOCOL, runtime_root=None):
        if protocol not in (PROTOCOL, *NATIVE_PAIRED_PROTOCOLS):
            raise ValueError("Unknown ALFWorld evaluation protocol")
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.capabilities = capabilities
        self.protocol = protocol
        self.learning_archive = Path(learning_archive) if learning_archive is not None else self.root / "learning"
        self.engine = NativeEvaluator(self.root if runtime_root is None else runtime_root, tasks, tokenizer, model_fingerprint,
                                      {'environment': 'alfworld', 'memory_scope': 'family'},
                                      capabilities=capabilities, base_url=base_url, model_name=model_name)
        self.model_name = model_name
        self.model_fingerprint = model_fingerprint
        identity = dict(adapter=file_hash(__file__), witness=file_hash(witness.__file__), engine=self.engine.fingerprint)
        if protocol != PROTOCOL:
            identity['protocol'] = protocol
        self.fingerprint = digest(identity)
        self.job = None

    def close(self):
        self.engine.close()

    def set_job(self, path):
        self.job = str(Path(path).resolve())

    def _bank(self, program, memories, family):
        context = dict(environment='alfworld', memory_scope='family', family=family)
        if family not in memories:
            return CodeMemory(program, [], self.tokenizer, context, capabilities=self.capabilities)
        bank = CodeMemory.from_snapshot(program, memories[family], self.tokenizer, context,
                                        capabilities=self.capabilities)
        if bank.family != family:
            raise ValueError('Task Memory snapshot belongs to another family')
        return bank

    def _learning_episode(self, row):
        episode_hash = row['episode_hash']
        if not isinstance(episode_hash, str) or not re.fullmatch('[0-9a-f]{64}', episode_hash):
            raise ValueError('Invalid learning episode reference')
        path = self.root / 'learning' / (episode_hash + '.json')
        if not path.is_file():
            path = self.learning_archive / (episode_hash + '.json')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != episode_hash:
            raise ValueError('Learning episode digest changed')
        ep = json.loads(raw)
        if (ep['phase'] != 'learning' or row['phase'] != 'learning' or ep['error']
                or ep['task_id'] != row['task_id'] or ep['family'] != row['family']
                or ep['native_reward_verified'] is not True or ep['score'] != row['score']
                or ep['environment'] != 'alfworld' or ep['memory_hook']['committed'] is not True):
            raise ValueError('Ledger must reference verified learning episodes of this dataset')
        return ep

    def verified_learning_evidence(self, ledger):
        """Bind each learning row to the code that actually executed it."""
        return [dict(row, program_hash=self._learning_episode(row)['program_hash']) for row in ledger]

    def prepare(self, mechanism, starting_mechanism, initial_memory, ledger):
        """Clone exact source or replay the same family-specific formal learning prefix."""
        old_program, program = Program(starting_mechanism), Program(mechanism)
        chains, seen = {}, set()
        for row in ledger:
            if row['task_id'] in seen or row['phase'] != 'learning':
                raise ValueError('Repeated or non-learning task in the formal prefix')
            seen.add(row['task_id'])
            family = row['family']
            chains[family] = _prefix(chains.get(family, digest([])), row['episode_hash'])
        if set(chains) != set(initial_memory):
            raise ValueError('Formal learning prefix has different task families than the mother')
        for family, chain in chains.items():
            if self._bank(old_program, initial_memory, family).source_log_hash != chain:
                raise ValueError('Mother memory does not bind the registered learning prefix')
        if program.hash == old_program.hash:
            return dict(initial_memory)
        banks = {}
        started = time.monotonic()
        for row in ledger:
            ep = self._learning_episode(row)
            family = ep['family']
            if family not in banks:
                banks[family] = self._bank(program, {}, family)
            bank = banks[family]
            try:
                replay_episode(bank, ep)
            except WorkerStartupError:
                raise
            except Rejected as exc:
                from .workflow import ContractError
                raise ContractError(str(exc)) from exc
            bank.source_log_hash = _prefix(bank.source_log_hash, row['episode_hash'])
        prepared = {k: b.export_snapshot() for k, b in banks.items()}
        append(self.root/'preparations.jsonl', dict(source=program.hash, records=len(ledger),
               parents={f: s['snapshot_hash'] for f, s in initial_memory.items()},
               prepared={f: s['snapshot_hash'] for f, s in prepared.items()},
               elapsed=time.monotonic()-started, costs={f: b.stats for f, b in banks.items()}))
        return prepared

    def learn(self, mechanism, memory, task, seed):
        if Path(task['archive_member']).parts[1] != 'train':
            raise ValueError('Formal ALFWorld learning can only consume the training split')
        updated, row = self._episode(Program(mechanism), memory, task, seed, 'learning', None, 0)
        return updated, row

    def evaluate(self, mechanism, pre_memory, tasks, seeds):
        if len(tasks) != len(seeds) or not tasks:
            raise ValueError('Panel requires one seed per task')
        self.engine.check_tasks(tasks)
        program = Program(mechanism)
        panel = digest(dict(tasks=tasks, seeds=seeds, pre=pre_memory, program=program.hash,
                            evaluator=self.fingerprint))
        private = dict(pre_memory)
        rows = []
        for index, (task, seed) in enumerate(zip(tasks, seeds)):
            private, row = self._episode(program, private, task, seed, 'panel', panel, index)
            rows.append(row)
        return rows

    def _episode(self, program, memory, task, seed, label, panel, index):
        self.engine.check_tasks([task])
        family = task['family']
        bank = self._bank(program, memory, family)
        before = bank.export_snapshot()
        binding = dict(version=2, evaluator=self.fingerprint, source=program.hash,
                       initial=before['snapshot_hash'], task=digest(task), seed=seed,
                       scope=label, panel=panel, index=index, job=self.job)
        directory = self.root / 'episodes' / digest(binding)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory/'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            blocked = directory/'learning_failure.json'
            complete = directory/'complete.json'
            if blocked.is_file():
                if complete.exists():
                    raise ValueError('Learning task has both a failure and a completion')
                complete = blocked
            if complete.is_file():
                result = read(complete)
                if result['binding'] != binding or result['result_hash'] != digest(result['result']):
                    raise ValueError('Native completion binding or result changed')
                result = result['result']
                raw = directory / result['attempt']
                if (result['episode_wire'] != 'episode.json'
                        or file_hash(raw/'episode.json') != result['row']['episode_hash']
                        or file_hash(raw/'canonical_episodes.jsonl') != result['canonical_hash']
                        or file_hash(raw/'actor.jsonl') != result['actor_hash']):
                    raise ValueError('Native receipt contents changed')
                if label == 'learning' and complete != blocked:
                    self._learning_episode(result['row'])
            else:
                result = self._execute(directory, program, bank, before, task, seed, label)
                if result['row']['status'] == 'deployed_learning_hook_failure':
                    complete = blocked
                _write(complete, dict(binding=binding, result=result, result_hash=digest(result)))
            if complete == blocked:
                if result['row']['status'] != 'deployed_learning_hook_failure' or label != 'learning':
                    raise ValueError('Invalid deployed learning failure receipt')
                raise LearningMechanismError(learning_failure_evidence(directory, result))
            if result['row']['status'] == 'method_eval_failure':
                return dict(memory), result['row']
            snapshot = result['snapshot']
            if getattr(self, "protocol", PROTOCOL) == COEVOLUTION_MIXED_10X60_PROTOCOL and (
                    result['row']['family'] != family or snapshot['family'] != family
                    or snapshot['families'] != [family] or snapshot['memory_scope'] != 'family'
                    or json.loads(snapshot['context_wire']).get('family') != family):
                raise ValueError('Native episode returned a different task-family bank')
            updated = dict(memory)
            updated[family] = snapshot
            return updated, result['row']

    def _execute(self, directory, program, bank, before, task, seed, label):
        # A model response can be unknown after a crash. Retain the attempt and
        # never silently resample it. Explicitly classified infrastructure
        # failures allow one retry on a later invocation under the same binding.
        attempts = sorted(directory.glob('attempt-*'))
        if attempts and not (attempts[-1]/'infrastructure.json').is_file():
            raise RuntimeError(f'Ambiguous native attempt requires reconciliation: {attempts[-1]}')
        if len(attempts) >= 2:
            raise RuntimeError(f'Native infrastructure retry exhausted: {directory}')
        dest = directory / f'attempt-{len(attempts)}'
        dest.mkdir()
        ledger = dest/'actor.jsonl'
        ledger.touch()
        self.engine.runner.ledger = ledger
        try:
            ep = self.engine.runner.episode(task, bank, label, seed, allow_write=True)
            with (dest/'canonical_episodes.jsonl').open() as source:
                if json.loads(source.readline()) != ep or source.readline():
                    raise RuntimeError('Unexpected canonical ALFWorld episode')
            with ledger.open() as source:
                actors = [row for line in source if (row := json.loads(line)).get('role') == 'actor']
            if (ep['actor_ledger_start'] != 0 or ep['actor_ledger_end'] != ledger.stat().st_size
                    or any(r.get('task_id') != task['id'] or r.get('phase') != label
                           or r.get('model_name') != self.model_name
                           or r.get('model_fingerprint') != self.model_fingerprint
                           or r.get('original_message_hash') != digest(r['messages']) for r in actors)):
                raise RuntimeError('Native Actor receipts differ from registered task, request or model')
            failure = bool(ep.get('error'))
            learning_failure = failure and label == 'learning' and self.protocol in NATIVE_PAIRED_PROTOCOLS
            if failure:
                if (label != 'panel' and not learning_failure) or not candidate_hook_failure(ep, program.hash):
                    raise RuntimeError('Native task incomplete: ' + str(ep['error']))
                if bank.fingerprint != before['memory_hash'] or ep['generated_memory_write_calls']:
                    raise RuntimeError('Failed candidate changed its private bank')
                if (learning_failure or self.protocol == COEVOLUTION_MIXED_10X60_PROTOCOL) and not (
                        ep['task_id'] == task['id'] and ep['phase'] == label and ep['family'] == task['family']
                        and ep['environment'] == 'alfworld' and ep['seed'] == seed and ep['environment_seed'] == task['seed']
                        and ep['executor'] == 'ALMA.Agent_Workflow.run_single_task' and ep['synthetic'] is False
                        and ep['native_reward_verified'] is False and ep['score'] is None and not ep['memory_hook']['committed']
                        and ep['memory_hash_before'] == ep['memory_hash_after'] == before['memory_hash']):
                    raise RuntimeError('Incomplete binding for a deployed learning Hook failure')
            else:
                _verify_completed(ep, task, label, seed, actors, before, bank)
        except Exception as exc:
            write(dest/'infrastructure.json', dict(kind=type(exc).__name__, detail=str(exc)))
            raise
        ep_hash = _archive_episode(dest, ep, self.root/'learning' if label == 'learning' and not failure else None)
        if not failure:
            bank.source_log_hash = _prefix(bank.source_log_hash, ep_hash)
        snapshot = bank.export_snapshot()
        evidence, cost = _evidence(ep, before, snapshot, actors, episode_hash=ep_hash)
        row = dict(task_id=task['id'], family=task['family'], phase=label, environment='alfworld',
                   episode_hash=ep_hash, native_score=ep['score'], score=0 if failure else int(ep['score']),
                   status='method_eval_failure' if failure else 'completed', error=ep.get('error'),
                   evidence=evidence, cost=cost)
        if learning_failure:
            row.update(status='deployed_learning_hook_failure', score=None)
        if failure:
            row['candidate_error'] = _candidate_error(ep)
        return dict(row=row, snapshot=snapshot, attempt=dest.name, episode_wire='episode.json',
                    canonical_hash=file_hash(dest/'canonical_episodes.jsonl'), actor_hash=file_hash(ledger))
