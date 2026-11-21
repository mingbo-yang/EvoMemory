"""Independent environments with shared memory across their task types.

Collection, source search, target search and sealed generalization are explicit
phases. Each environment retains its source mechanism and cumulative experiences.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

from evomem.common import ROOT, digest, file_hash, read, write
from evomem.alma_backend import ACTOR_PARAMETERS
from .tokenization import Tokenizer
from .engine import EvolutionEngine, code_binding
from .experience import EvolutionMemory
from .local_service import verify_local_service, verify_model
from .protocol import WORKFLOW_VERSION, HOOK_VERSION
from .program import BASELINE_SOURCE, Program
from .proposer import CodeEvolver, EVOLVER_PROMPT, PARAMETERS as EVOLVER_PARAMETERS
from .review import MechanismReviewer, REVIEW_PROMPT, PARAMETERS as REVIEW_PARAMETERS
from .store import RunStore
from .capabilities import LocalModelCapabilities

ENDPOINTS = {'alfworld': 'http://127.0.0.1:8140/v1', 'textworld': 'http://127.0.0.1:8142/v1'}
ROUNDS = 3


def rounds_per_search(root):
    """Read the frozen proposal budget; do not infer it from candidate outcomes."""
    value = read(root / 'protocol.json').get('rounds_per_search', ROUNDS)
    if type(value) is not int or value <= 0:
        raise ValueError('rounds_per_search must be a positive frozen integer')
    return value


def workflow_context(environment, selected, protocol=None):
    """One fixed new workflow; values in receipts are not independent toggles."""
    protocol = protocol or {}
    if protocol.get('workflow_version', WORKFLOW_VERSION) != WORKFLOW_VERSION:
        raise ValueError('Campaign workflow version differs; prepare a new plan for ' + WORKFLOW_VERSION)
    if protocol.get('memory_hook_version', HOOK_VERSION) != HOOK_VERSION:
        raise ValueError('This workflow always uses memory-events-v1')
    return dict(context(environment, selected), memory_hook_version=HOOK_VERSION)


def task_file(root, environment, role):
    return root / 'datasets' / environment / (role + '.json')


def tasks(root, environment, role):
    value = read(task_file(root, environment, role))
    return value['tasks'] if isinstance(value, dict) else value


def context(environment, selected):
    families = {task['family'] for task in selected}
    if len(families) != 1:
        raise ValueError('A search bank must contain one target task family')
    return dict(environment=environment, family=next(iter(families)), memory_scope='environment')


def learning_blocks(root, environment, role):
    """Explicit learning allocation; never infer it from opened task outcomes."""
    if not (root / 'protocol.json').exists():
        raise ValueError('Current campaign requires an explicit continual protocol.json')
    protocol = read(root / 'protocol.json')
    if protocol.get('task_memory_mode') != 'continual':
        raise ValueError('Current campaign requires task_memory_mode=continual')
    if protocol.get('workflow_version') != WORKFLOW_VERSION or protocol.get('memory_hook_version') != HOOK_VERSION:
        raise ValueError('Current campaign requires explicit current workflow and event protocol versions')
    paths = protocol.get('learning_blocks', {}).get(environment, {}).get(role)
    if type(paths) is not list or len(paths) != rounds_per_search(root):
        raise ValueError('Continual campaign requires predeclared learning blocks for every round')
    blocks = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Learning task manifest must belong to the campaign')
        value = read(path)
        blocks.append(value['tasks'] if isinstance(value, dict) else value)
    if any(type(block) is not list or not block for block in blocks):
        raise ValueError('Continual campaign requires a nonempty learning block for every round')
    return blocks


def validate_dataset(root, environment):
    """Inspect split metadata only; held-out outcomes are never opened here."""
    seen_ids, seen_groups, seen_games = set(), set(), set()
    roles = ('source_adaptation', 'source_exploration', 'target_adaptation',
             'target_exploration', 'target_generalization')
    batches = {role: tasks(root, environment, role) for role in roles}
    for role in ('source', 'target'):
        for index, block in enumerate(learning_blocks(root, environment, role)):
            batches[role + '_learning_' + str(index)] = block
    families = {}
    for role, batch in batches.items():
        if not batch:
            raise ValueError('Every declared dataset role must be nonempty')
        family = {t['family'] for t in batch}
        if len(family) != 1:
            raise ValueError('Each dataset role must contain one task family')
        families[role] = next(iter(family))
        ids = [t['id'] for t in batch]
        groups = {t['group_id'] for t in batch}
        games = {t['game_sha256'] for t in batch if t.get('game_sha256')}
        if (len(set(ids)) != len(ids) or seen_ids.intersection(ids)
                or seen_groups.intersection(groups) or seen_games.intersection(games)):
            raise ValueError('Source, target and generalization tasks must be disjoint')
        if role != 'target_generalization':
            for t in batch:
                if environment == 'alfworld' and not t['archive_member'].startswith('json_2.1.1/train/'):
                    raise ValueError('Search may only use official training tasks')
                if environment == 'textworld' and t.get('official_split') != 'train':
                    raise ValueError('TextWorld search may only use its official training split')
                if t.get('split') in {'final', 'test', 'eval_in_distribution', 'eval_out_of_distribution'}:
                    raise ValueError('Held-out tasks cannot enter collection or search')
        seen_ids.update(ids); seen_groups.update(groups); seen_games.update(games)
    if (families['source_adaptation'] != families['source_exploration']
            or len({families[r] for r in roles if r.startswith('target_')}) != 1
            or families['source_exploration'] == families['target_exploration']):
        raise ValueError('Source and target must be distinct task families within one environment')
    if any(family != families['source_exploration'] for name, family in families.items() if name.startswith('source_')):
        raise ValueError('Source learning tasks must retain the declared source task family')
    if any(family != families['target_exploration'] for name, family in families.items() if name.startswith('target_')):
        raise ValueError('Target learning tasks must retain the declared target task family')
    return dict(unique_tasks=len(seen_ids), role_hashes={role: digest(batch) for role, batch in batches.items()})


def validate_transfer_records(root, environment, records):
    source_family = context(environment, tasks(root, environment, 'source_exploration'))['family']
    for record in records:
        observed = record.get('observed_context', {})
        if observed.get('environment') != environment or observed.get('family') != source_family:
            raise ValueError('Evolution Memory must come from the same environment and its declared source family')


def evaluator(environment, out, selected, tokenizer, model_hash, ctx=None, *, capabilities=None):
    options = {'capabilities': capabilities} if capabilities is not None else {}
    if environment == 'alfworld':
        from .native import NativeEvaluator
        return NativeEvaluator(out, selected, tokenizer, model_hash, ctx, **options)
    from .textworld_native import TextWorldNativeEvaluator
    return TextWorldNativeEvaluator(out, selected, tokenizer, model_hash, ctx,
                                   base_url=ENDPOINTS[environment], **options)


def capability_pool(root, environment, out, tokenizer, model_hash, phase):
    protocol = root / 'protocol.json'
    if not protocol.exists() or read(protocol).get('local_model_capabilities') is not True:
        return None
    return LocalModelCapabilities(root / 'capability_pools' / environment, environment,
        out / 'capability_costs.jsonl', tokenizer, model_hash, ENDPOINTS[environment], phase=phase)


def validate_episode(episode, task, limit):
    score = episode.get('score')
    if (episode.get('task_id') != task['id'] or episode.get('synthetic') is not False
            or not episode.get('native_reward_verified') or episode.get('error')
            or type(score) not in (int, float) or not math.isfinite(score)
            or episode.get('environment_seed') != task['seed']
            or len(episode.get('history', [])) > limit):
        raise RuntimeError('Incomplete real episode; preserve and inspect it: ' + task['id'])
    if episode.get('memory_hook') and (not episode['memory_hook'].get('complete')
                                       or not episode['memory_hook'].get('actor_delivery_verified')):
        raise RuntimeError('Incomplete Hook event or actual Actor delivery evidence')


def bind(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Frozen phase inputs changed: ' + str(path))
    else:
        write(path, value)


def reused_adaptation(spec, role, expected, binding, step_limit):
    """Reuse identical own-family collection evidence with its original receipt."""
    origin = Path(spec['collection'])
    original_binding = read(origin / 'binding.json')
    artifact = origin / (role + '.json')
    if digest(original_binding) != spec['binding_hash'] or file_hash(artifact) != spec['artifact_hash']:
        raise RuntimeError('Imported collection evidence changed')
    for key in ('version', 'environment', 'model_hash', 'endpoint', 'actor_parameters', 'own_code', 'executor_bridge'):
        if original_binding.get(key) != binding.get(key):
            raise RuntimeError('Imported collection execution binding differs: ' + key)
    if original_binding['tasks'][role] != expected:
        raise RuntimeError('Imported collection task split differs')
    record = read(artifact)
    if record.get('complete') is not True or [e['task_id'] for e in record['episodes']] != [t['id'] for t in expected]:
        raise RuntimeError('Imported adaptation collection incomplete')
    for episode, task in zip(record['episodes'], expected):
        cache = read(origin / 'episodes' / (digest(task) + '.json'))
        if (cache['task_hash'] != digest(task) or cache['binding_hash'] != digest(original_binding)
                or cache['episode'] != episode):
            raise RuntimeError('Imported episode disagrees with original cache receipt')
        validate_episode(episode, task, step_limit)
    return dict(record, reuse_provenance=spec, newly_executed=False)


def collection(root, environment, tokenizer, model_hash):
    validate_dataset(root, environment)
    out = root / 'collection' / environment
    selected = {role: tasks(root, environment, role) for role in ('source_adaptation', 'target_adaptation')}
    all_tasks = [t for batch in selected.values() for t in batch]
    out.mkdir(parents=True, exist_ok=True)
    # Collection does not run either meta role. Bind only code it executes, so
    # later development of the campaign reporting code cannot change its evidence.
    own_names = ['native.py', 'program.py', 'policy.py', 'worker.py', 'memory.py', 'protocol.py', 'isolation.py']
    if environment == 'textworld':
        own_names += ['textworld_native.py', 'textworld_tasks.py', 'local_service.py']
    binding = dict(version='real-no-memory-collection-v1', environment=environment,
                   tasks=selected, model_hash=model_hash, endpoint=ENDPOINTS[environment],
                   actor_parameters=ACTOR_PARAMETERS,
                   own_code={name: file_hash(Path(__file__).with_name(name)) for name in own_names},
                   executor_bridge=file_hash(ROOT / 'evomem/alma_backend.py'))
    reuse_path = out / 'reuse.json'
    reuse = read(reuse_path) if reuse_path.exists() else {}
    if set(reuse) - set(selected):
        raise ValueError('Unknown imported collection role')
    binding['reuse'] = reuse
    bind(out / 'binding.json', binding)
    native = evaluator(environment, out / 'runner', all_tasks, tokenizer, model_hash)
    try:
        for role, batch in selected.items():
            if role in reuse:
                record = reused_adaptation(reuse[role], role, batch, binding, native.step_limit)
                write(out / (role + '.json'), record)
                continue
            episodes = []
            for task in batch:
                cache = out / 'episodes' / (digest(task) + '.json')
                if cache.exists():
                    item = read(cache)
                    if item['task_hash'] != digest(task) or item['binding_hash'] != digest(binding):
                        raise RuntimeError('Collection cache binding changed')
                    episode = item['episode']
                else:
                    episode = native.no_memory([task], phase=role)[0]
                    write(cache, dict(task_hash=digest(task), binding_hash=digest(binding), episode=episode))
                episodes.append(episode)
                write(out / (role + '.json'), dict(episodes=episodes, synthetic=False,
                                                   collection='official no-memory Executor',
                                                   complete=len(episodes) == len(batch)))
                validate_episode(episode, task, native.step_limit)
                write(out / 'status.json', dict(status='collecting', role=role,
                                                completed=len(episodes), total=len(batch), timestamp=time.time()))
        write(out / 'status.json', dict(status='complete', episodes=len(all_tasks),
             reused_episodes=sum(len(selected[role]) for role in reuse), timestamp=time.time()))
    finally:
        native.close()


def adaptation(root, environment, role):
    path = root / 'collection' / environment / (role + '_adaptation.json')
    record = read(path)
    expected = tasks(root, environment, role + '_adaptation')
    if record.get('complete') is not True or [e['task_id'] for e in record['episodes']] != [t['id'] for t in expected]:
        raise RuntimeError('Adaptation collection is incomplete')
    origins_path = root / 'collection_origins.json'
    if origins_path.exists():
        origin = read(origins_path)[environment]
        folder = path.parent.resolve()
        if folder != Path(origin['source_collection']).resolve():
            raise RuntimeError('Imported collection points to a different environment source')
        archive = Path(origin['original_code_snapshot'])
        if file_hash(archive) != origin['code_snapshot_sha256']:
            raise RuntimeError('Original collection code snapshot changed')
        original_binding = read(folder / 'binding.json')
        if (original_binding['environment'] != environment
                or original_binding['endpoint'] != ENDPOINTS[environment]
                or original_binding['actor_parameters'] != ACTOR_PARAMETERS
                or original_binding['tasks'][role + '_adaptation'] != expected):
            raise RuntimeError('Original collection protocol does not match this environment and split')
        # Verify preserved code bytes instead of pretending the old recordings
        # executed the newer shared-memory runtime.
        import hashlib
        with zipfile.ZipFile(archive) as z:
            hashes = dict(original_binding['own_code'])
            for name, sha in hashes.items():
                if hashlib.sha256(z.read('phase1_codegen/' + name)).hexdigest() != sha:
                    raise RuntimeError('Original collection runtime snapshot differs from its binding')
            if hashlib.sha256(z.read('evomem/alma_backend.py')).hexdigest() != original_binding['executor_bridge']:
                raise RuntimeError('Original Executor bridge snapshot differs from its binding')
        if record.get('reuse_provenance'):
            verified = reused_adaptation(record['reuse_provenance'], role + '_adaptation', expected,
                                         original_binding, 30 if environment == 'alfworld' else 80)
            if verified['episodes'] != record['episodes']:
                raise RuntimeError('Recursively reused adaptation evidence differs')
        else:
            for episode, task in zip(record['episodes'], expected):
                cache = read(folder / 'episodes' / (digest(task) + '.json'))
                if (cache['task_hash'] != digest(task) or cache['binding_hash'] != digest(original_binding)
                        or cache['episode'] != episode):
                    raise RuntimeError('Original collection cache evidence changed')
    # Annotate an in-memory copy from the validated environment manifest. Preserve
    # the original no-memory episode artifacts and their original code receipts.
    episodes = []
    for episode, task in zip(record['episodes'], expected):
        validate_episode(episode, task, 30 if environment == 'alfworld' else 80)
        if (episode.get('environment', environment) != environment
                or episode.get('family') != task['family']
                or episode.get('group_id') != task['group_id']):
            raise RuntimeError('Adaptation provenance differs from its declared environment/task type')
        episodes.append(dict(episode, environment=environment))
    return episodes


def shared_logs(root, environment, role):
    if role == 'source':
        return adaptation(root, environment, 'source')
    if role != 'target':
        raise ValueError('Unknown campaign memory role')
    from .learning import load_snapshot, snapshot_logs
    folder = root / 'source' / environment
    state = read(folder / 'state.json')
    RunStore(folder).validate_state(state)
    return snapshot_logs(load_snapshot(folder, state['task_memory'])) + adaptation(root, environment, 'target')


def source_checkpoint(root, environment):
    folder = root / 'source' / environment
    if read(folder / 'status.json')['status'] != 'complete':
        raise RuntimeError('Source mechanism must be fixed before the target checkpoint is inherited')
    state = read(folder / 'state.json')
    if state['next_round'] != rounds_per_search(root) or state['binding']['context'].get('environment') != environment:
        raise RuntimeError('Incomplete or foreign source mechanism checkpoint')
    if state['binding'].get('workflow') != WORKFLOW_VERSION:
        raise RuntimeError('Source mechanism belongs to a different workflow; require ' + WORKFLOW_VERSION)
    program = RunStore(folder).current()
    return program, dict(source_environment=environment, source_state_hash=digest(state),
                         inherited_program_hash=program.hash,
                         task_memory_snapshot=state.get('task_memory'),
                         source_adaptation_hash=digest(adaptation(root, environment, 'source')))


def target_memory_checkpoint(root, environment, tokenizer, capabilities):
    """Create once, then clone the same actual inherited bank into both arms."""
    from .learning import load_snapshot, snapshot_memory, snapshot_logs, make_snapshot, save_snapshot
    from .memory import CodeMemory
    program, inherited = source_checkpoint(root, environment)
    folder = root / 'source' / environment
    state = read(folder / 'state.json')
    source = load_snapshot(folder, state['task_memory'])
    target = adaptation(root, environment, 'target')
    out = root / 'shared_task_memory' / environment
    binding = dict(source_snapshot=state['task_memory'], source_state_hash=digest(state),
                   target_adaptation_hash=digest(target), program_hash=program.hash)
    with RunStore(out).lock():
        bind(out / 'binding.json', binding)
        pointer = out / 'checkpoint.json'
        if pointer.exists():
            snapshot = load_snapshot(out, read(pointer))
        else:
            memory = CodeMemory.from_snapshot(program, snapshot_memory(source), tokenizer, capabilities=capabilities)
            for episode in target:
                memory.write(episode)
            logs = snapshot_logs(source) + target
            memory.source_log_hash = digest(logs)
            snapshot = make_snapshot(logs, environment, memory_snapshot=memory.export_snapshot(), parent_hash=digest(source))
            bind(pointer, save_snapshot(out, snapshot))
    return snapshot


def freeze_search(out, plan, model_hash):
    hashes = code_binding()
    freeze = dict(version=WORKFLOW_VERSION, plan=plan, model_hash=model_hash,
                  source_hashes=hashes, python=sys.version, actor_parameters=ACTOR_PARAMETERS,
                  evolver_prompt=EVOLVER_PROMPT, evolver_parameters=EVOLVER_PARAMETERS,
                  evolver_abi=HOOK_VERSION,
                  reviewer_prompt=REVIEW_PROMPT, reviewer_parameters=REVIEW_PARAMETERS,
                  dependencies=dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                                           if d.metadata['Name'])))
    out.mkdir(parents=True, exist_ok=True)
    bind(out / 'freeze.json', freeze)
    archive = out / 'code_snapshot.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for name in hashes:
                z.write(ROOT / name, name)
    return freeze


def search(root, environment, role, condition, tokenizer, model_hash):
    if (role, condition) not in {('source', 'local_only'), ('target', 'no_external_em'), ('target', 'cross_type_em')}:
        raise ValueError('Unknown search role or condition for within-environment transfer')
    validate_dataset(root, environment)
    rounds = rounds_per_search(root)
    if role == 'source':
        out = root / 'source' / environment
        records = []
        initial, inherited = Program(BASELINE_SOURCE), None
    else:
        out = root / 'target' / environment / condition
        records = [] if condition == 'no_external_em' else read(root / 'transfer' / (environment + '.json'))
        validate_transfer_records(root, environment, records)
        initial, inherited = source_checkpoint(root, environment)
    exploration = tasks(root, environment, role + '_exploration')
    logs = shared_logs(root, environment, role)
    learning = learning_blocks(root, environment, role)
    protocol = read(root / 'protocol.json')
    ctx = workflow_context(environment, exploration, protocol)
    plan = dict(role=role, condition=condition, context=ctx, seed=11, rounds=rounds,
                acceptance_policy='reviewer_and_reward', exploration=exploration,
                adaptation_hash=digest(logs), evolution_memory_hash=digest(records),
                endpoint=ENDPOINTS[environment], initial_program_hash=initial.hash,
                memory_scope='environment', inherited_checkpoint=inherited,
                task_memory_mode='continual',
                learning_blocks=learning,
                task_memory_shared_across_types=True,
                task_memory_families=sorted({e['family'] for e in logs}),
                task_memory_order='source adaptation then target adaptation; own environment only',
                collection_provenance=(read(root / 'collection_origins.json').get(environment)
                                       if (root / 'collection_origins.json').exists() else None),
                native_step_limit=30 if environment == 'alfworld' else 80)
    if role == 'source' and protocol.get('source_baseline_origins', {}).get(environment) is not None:
        raise ValueError('Continual learning cannot reuse a frozen-replay source baseline measurement')
    capabilities = capability_pool(root, environment, out, tokenizer, model_hash, role + '/' + condition)
    if capabilities is not None:
        plan['model_capabilities'] = capabilities.manifest
    initial_memory = None
    if role == 'target':
        from .learning import snapshot_memory, snapshot_logs
        snapshot = target_memory_checkpoint(root, environment, tokenizer, capabilities)
        logs = snapshot_logs(snapshot)
        initial_memory = snapshot_memory(snapshot)
        plan['shared_target_checkpoint_hash'] = digest(snapshot)
        plan['initial_runtime_memory_hash'] = snapshot['memory_hash']
        plan['initial_ordered_logs_hash'] = snapshot['ordered_logs_hash']
        plan['adaptation_hash'] = digest(logs)
        plan['task_memory_families'] = sorted({e['family'] for e in logs})
        plan['task_memory_order'] = 'source-final cumulative learning frontier then common target bootstrap; branch learning blocks remain separate'
    native = evaluator(environment, out / 'runner', exploration + [t for block in learning for t in block], tokenizer, model_hash, ctx,
                       **({'capabilities': capabilities} if capabilities is not None else {}))
    proposer = reviewer = None
    try:
        # Freeze after native construction so TextWorld's isolated dependency
        # versions and actual adapter fingerprint are recorded, before inference.
        plan['native_evaluator_fingerprint'] = native.fingerprint
        freeze_search(out, plan, model_hash)
        proposer = CodeEvolver(tokenizer, out / 'evolver_costs.jsonl', model_hash, base_url=ENDPOINTS[environment])
        reviewer = MechanismReviewer(tokenizer, out / 'reviewer_costs.jsonl', model_hash, base_url=ENDPOINTS[environment])
        engine = EvolutionEngine(out, tokenizer, proposer, native, EvolutionMemory(records), ctx, logs,
                                 exploration, [], seed=11, reviewer=reviewer,
                                 initial_source=initial.source,
                                 acceptance_policy='reviewer_and_reward', max_rounds=rounds, capabilities=capabilities,
                                 learning_blocks=learning, initial_memory_snapshot=initial_memory)
        write(out / 'status.json', dict(status='searching', timestamp=time.time()))
        state = engine.run(rounds)
        done = state['next_round'] == rounds and not any(t.get('review_error') for t in engine.store.history(state))
        write(out / 'status.json', dict(status='complete' if done else 'stopped_on_review_error',
                                        rounds=state['next_round'], revisions=state['revision'], timestamp=time.time()))
        if not done:
            raise RuntimeError('Search stopped on a real role failure; preserve reports before any correction')
        return state
    finally:
        if proposer:
            proposer.close()
        if reviewer:
            reviewer.close()
        native.close()
        if capabilities is not None:
            capabilities.close()


def transfer(root, environment=None):
    exports = []
    if environment is not None and environment not in ENDPOINTS:
        raise ValueError('Unknown environment')
    for destination in ([environment] if environment else ENDPOINTS):
        validate_dataset(root, destination)
        source_root = root / 'source' / destination
        state = read(source_root / 'state.json')
        if read(source_root / 'status.json')['status'] != 'complete' or state['next_round'] != rounds_per_search(root):
            raise RuntimeError('This environment source search must finish before freezing cross-type experiences')
        store = RunStore(source_root)
        store.validate_state(state)
        if state['binding'].get('synthetic') is not False:
            raise RuntimeError('Synthetic source runs cannot supply a formal transfer library')
        records = EvolutionMemory([trial['evolution_event'] for trial in store.history(state)]).records
        if not records or any(record.get('kind') != 'code_change_trial' or record.get('synthetic') is not False for record in records):
            raise RuntimeError('Cross-type transfer requires actual new code-change experiences')
        validate_transfer_records(root, destination, records)
        path = root / 'transfer' / (destination + '.json')
        exports.append((path, records, dict(source_environment=destination, target_environment=destination,
             source_family=context(destination, tasks(root, destination, 'source_exploration'))['family'],
             target_family=context(destination, tasks(root, destination, 'target_exploration'))['family'],
             source_state_hash=digest(state), record_ids=[r['id'] for r in records],
             includes_task_memory=False, includes_raw_trajectories=False, includes_final_answers=False,
             note='This export is Evolution Memory only. Shared Task Memory stays in the separate environment pool.',
             all_outcome_signs_retained=True)))
    # Each environment may publish independently when its own producer finishes.
    for path, records, provenance in exports:
        bind(path, records)
        bind(path.with_suffix('.provenance.json'), provenance)


def generalization(root, environment, tokenizer, model_hash):
    from .learning import load_snapshot, snapshot_logs, snapshot_memory
    validate_dataset(root, environment)
    selected = tasks(root, environment, 'target_generalization')
    logs = shared_logs(root, environment, 'target')
    initial, inherited = source_checkpoint(root, environment)
    variants = {'inherited_no_target_evolution': initial}
    variant_snapshots = {}
    for name in ('no_external_em', 'cross_type_em'):
        folder = root / 'target' / environment / name
        if read(folder / 'status.json')['status'] != 'complete':
            raise RuntimeError('All target search versions must be fixed before opening independent generalization')
        variants[name] = RunStore(folder).current()
        variant_snapshots[name] = load_snapshot(folder, read(folder / 'state.json')['task_memory'])
    out = root / 'generalization' / environment
    capabilities = capability_pool(root, environment, out, tokenizer, model_hash, 'sealed_generalization')
    variant_snapshots['inherited_no_target_evolution'] = target_memory_checkpoint(root, environment, tokenizer, capabilities)
    protocol = read(root / 'protocol.json')
    ctx = workflow_context(environment, selected, protocol)
    binding = dict(versions={key: p.hash for key, p in variants.items()}, tasks=selected, context=ctx,
                                  logs_hash=digest(logs), model_hash=model_hash, endpoint=ENDPOINTS[environment],
                                  memory_scope='environment', inherited_checkpoint=inherited,
                                  task_memory_families=sorted({e['family'] for e in logs}),
                                  sealed_from_review_and_evolver=True,
                                  model_capabilities=capabilities.fingerprint if capabilities is not None else None)
    binding.update(task_memory_mode='continual',
                   variant_snapshots={name: digest(snapshot) for name, snapshot in variant_snapshots.items()},
                   history_policy='Each learned arm uses its committed bank; inherited reference uses shared target-start bank')
    bind(out / 'binding.json', binding)
    native = evaluator(environment, out / 'runner', selected, tokenizer, model_hash, ctx,
                       **({'capabilities': capabilities} if capabilities is not None else {}))
    results = {}
    try:
        for name, program in variants.items():
            episodes = []
            current_logs = snapshot_logs(variant_snapshots[name])
            runtime_snapshot = snapshot_memory(variant_snapshots[name])
            for task in selected:
                key_data = dict(program=program.hash, task=task, logs=digest(current_logs))
                key_data['actual_memory_snapshot'] = digest(variant_snapshots[name])
                key = digest(key_data)
                cache = out / 'episodes' / (key + '.json')
                if cache.exists():
                    receipt = read(cache)
                    if (receipt.get('key') != key or receipt.get('binding_hash') != digest(binding)
                            or receipt.get('episode_hash') != digest(receipt['episode'])):
                        raise RuntimeError('Generalization cache evidence or binding changed')
                    episode = receipt['episode']
                else:
                    episode = native(program, current_logs, [task], phase='sealed_generalization/' + name,
                                     memory_snapshot=runtime_snapshot)[0]
                    write(cache, dict(key=key, episode=episode, episode_hash=digest(episode),
                                      binding_hash=digest(binding), physical_variant=name))
                episodes.append(episode)
                validate_episode(episode, task, native.step_limit)
                if (episode.get('read_only_memory') is not True
                        or episode.get('memory_write_calls') != 0 or episode.get('generated_memory_write_calls') != 0
                        or episode.get('program_hash') != program.hash
                        or episode.get('memory_hash_before') != runtime_snapshot['memory_hash']
                        or episode.get('memory_hash_after') != runtime_snapshot['memory_hash']):
                    raise RuntimeError('Held-out execution changed or did not use the committed read-only bank')
            results[name] = dict(program_hash=program.hash, episodes=episodes,
                                 mean_native_reward=sum(e['score'] for e in episodes) / len(episodes))
            results[name].update(task_memory_snapshot_hash=digest(variant_snapshots[name]),
                                 cumulative_learning_episodes=len(current_logs))
            write(out / 'results.json', results)
        write(out / 'status.json', dict(status='complete', variants=len(results), tasks=len(selected), timestamp=time.time()))
    finally:
        native.close()
        if capabilities is not None:
            capabilities.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m phase1_codegen', description=__doc__,
        epilog='This is the current ' + WORKFLOW_VERSION + ' workflow. '
               'All generated mechanisms implement on_event(state, event, context). '
               'python -m phase1_codegen and python -m phase1_codegen.campaign use this same entrypoint.')
    parser.add_argument('phase', choices=('collect', 'source', 'transfer', 'target', 'generalization'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--environment', choices=tuple(ENDPOINTS))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.phase == 'transfer':
        transfer(root, args.environment)
        return
    if args.environment is None:
        parser.error('--environment is required outside transfer')
    # Reject malformed metadata and workflow drift before touching a service or
    # loading a tokenizer. This opens task manifests, never held-out outcomes.
    validate_dataset(root, args.environment)
    protocol = read(root / 'protocol.json')
    selected_role = 'source_exploration' if args.phase in {'collect', 'source'} else 'target_exploration'
    workflow_context(args.environment, tasks(root, args.environment, selected_role), protocol)
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('Set PYTHONHASHSEED=0 before native execution')
    os.environ['TMPDIR'] = '/tmp'
    import tempfile
    tempfile.tempdir = '/tmp'
    identity = verify_local_service(ENDPOINTS[args.environment])
    write(root / 'services' / ('verified_' + args.environment + '.json'), identity)
    tokenizer = Tokenizer()
    model_hash = digest(verify_model())
    if args.phase == 'collect':
        collection(root, args.environment, tokenizer, model_hash)
    elif args.phase == 'source':
        search(root, args.environment, 'source', 'local_only', tokenizer, model_hash)
    elif args.phase == 'target':
        for condition in ('no_external_em', 'cross_type_em'):
            search(root, args.environment, 'target', condition, tokenizer, model_hash)
    else:
        generalization(root, args.environment, tokenizer, model_hash)


if __name__ == '__main__':
    main()
