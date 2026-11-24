"""Run two ordered ALFWorld families with separate task banks and a shared evolution archive."""
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

from evomem.common import file_hash
from .campaign import _checked_state, _link_learning, _native_panel
from .comparison import candidate_differences
from .panel import FAMILY_ROUNDS, save_registration, validate_sequential_manifest
from .runner import initial_state, open_stage
from .workflow import _lock, _read, _seed, digest, run_batch


ARMS = ('initial', 'final')


def _configuration(cfg):
    package = Path(__file__).resolve().parent
    revision = None
    if (package.parent / '.git').exists():
        revision = subprocess.check_output(['git', '-C', str(package.parent), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(model_config_hash=digest(cfg), code_commit=revision, sampling_scipy_version=version('scipy'),
                source_hashes={path.name: file_hash(path) for path in sorted(package.glob('*.py'))})


def _checkpoint(state):
    return {key: state[key] for key in ('mechanism', 'evolver')}


def _round_name(index):
    return f'evolution-{index + 1:02d}'


def _committed_round(root, index):
    stage = root / _round_name(index)
    state = _checked_state(stage)
    receipt = _read(stage / 'complete.json')
    if receipt != _round_receipt(stage, state):
        raise ValueError(f'Incomplete or changed family round: {stage}')
    return state


def _round_receipt(stage, state):
    return dict(state_hash=digest(state), resolved_panels_hash=file_hash(stage / 'batch/resolved_panels.json'),
                comparison_hash=file_hash(stage / 'candidate_differences.json'))


def _final_comparison(rows, tasks):
    if len(rows) != 2 or tuple(r.get('arm') for r in rows) != ARMS or rows[0]['evaluation_key'] == rows[1]['evaluation_key']:
        raise ValueError('Frozen arms must have two independent native evaluations')
    differences = Counter()
    by_arm = {}
    for result in rows:
        episodes = result['episodes']
        if ([e['task_id'] for e in episodes] != [t['id'] for t in tasks] or
                any(e['status'] not in ('completed', 'method_eval_failure') for e in episodes)):
            raise ValueError('Frozen arm lacks its ordered registered episodes')
        by_arm[result['arm']] = dict(tasks=len(episodes), scored_successes=sum(e['score'] for e in episodes),
                                      native_completed=sum(e['status'] == 'completed' for e in episodes),
                                      mechanism_failures=sum(e['status'] == 'method_eval_failure' for e in episodes))
    groups, cumulative = {}, {arm: [] for arm in ARMS}
    for task, a, b in zip(tasks, rows[0]['episodes'], rows[1]['episodes'], strict=True):
        group = groups.setdefault(task['group_id'], dict(tasks=0, initial_successes=0, final_successes=0))
        group['tasks'] += 1
        for arm, row in zip(ARMS, (a, b)):
            previous = cumulative[arm][-1] if cumulative[arm] else 0
            cumulative[arm].append(previous + row['score'])
            group[arm + '_successes'] += row['score']
        if a['status'] != 'completed' or b['status'] != 'completed':
            differences['without_native_pair'] += 1
        else:
            diff = b['native_score'] - a['native_score']
            differences['wins' if diff > 0 else 'losses' if diff < 0 else 'ties'] += 1
    for result in rows:
        arm = result['arm']
        summary = by_arm[arm]
        summary['success_rate'] = summary['scored_successes'] / summary['tasks']
        summary['recorded_environment_steps'] = sum(e['recorded_steps'] for e in result['episodes'])
    return dict(arms=by_arm, paired_native={k: differences[k] for k in ('wins', 'ties', 'losses', 'without_native_pair')},
                success_count_difference=by_arm['final']['scored_successes'] - by_arm['initial']['scored_successes'],
                cumulative_successes=cumulative, game_groups=len(groups), results_by_game_group=groups)


def _end_receipt(root, manifest, family):
    plan = manifest['families'][family]
    frozen, final = _read(root / 'frozen.json'), _read(root / 'final/summary.json')
    state = _read(root / _round_name(FAMILY_ROUNDS - 1) / 'batch/commit.json')
    records = _read(root / 'evolution_memory.json')
    ledger = state['state']['ledger']
    learning_ids = [t['id'] for batch in plan['batches'] for t in batch['learning']]
    if ([row['task_id'] for row in ledger] != learning_ids
            or any(row['family'] != family or row['phase'] != 'learning' for row in ledger)):
        raise ValueError('Final learning ledger differs from the registered family stream')
    if not (state['state_hash'] == digest(state['state'])
            and frozen['round_6_state_hash'] == state['state_hash']
            and frozen['frameworks'] == {'initial': _checkpoint(initial_state()), 'final': _checkpoint(state['state'])}
            and frozen['family'] == records['family'] == family
            and frozen['test_panel_hash'] == digest(plan['final'])
            and frozen['shared_archive_hash'] == digest(records)
            and records['round_6_state_hash'] == state['state_hash']
            and final['frozen_hash'] == digest(frozen)
            and final['comparison'] == _final_comparison(final['results'], plan['final'])):
        raise ValueError(f'Frozen final or evolution archive is not bound to {family} round six')
    for result in final['results']:
        arm = frozen['frameworks'][result['arm']]
        if result['mechanism_hash'] != digest(arm['mechanism']) or result['policy_hash'] != digest(arm['evolver']):
            raise ValueError('Stored final result refers to a different framework')
    return dict(family=family, registration_hash=digest(manifest), final_hash=digest(final),
                frozen_hash=digest(frozen), evolution_memory_hash=digest(records),
                learning_tasks=len(ledger), final_tasks_per_arm=len(plan['final']))


def _completed_family(root, manifest, family):
    if not (root / 'complete.json').is_file():
        raise ValueError(f'{family} must complete all rounds and both frozen tests before transfer')
    receipt = _read(root / 'complete.json')
    if receipt != _end_receipt(root, manifest, family):
        raise ValueError(f'{family} has no verified final completion')
    return receipt


def _run_round(root, family, index, cfg, manifest, shared):
    plan = manifest['families'][family]['batches']
    stage = root / _round_name(index)
    start = initial_state() if index == 0 else _committed_round(root, index - 1)
    previous = [task for batch in plan[:index] for task in batch['learning']]
    if [row['task_id'] for row in start['ledger']] != [task['id'] for task in previous]:
        raise ValueError('Committed learning stream differs from this family registration')
    current = plan[index]
    save_registration(stage / 'batch/starting_state.json', start)
    if (stage / 'complete.json').is_file():
        _committed_round(root, index)
        return
    tasks = previous + [t for phase in ('learning', 'primary', 'meta') for t in current[phase]]
    archive = root / 'learning'
    archive.mkdir(exist_ok=True)
    with open_stage(stage, cfg, manifest, tasks, learning_archive=archive,
                    shared_evolution_memory=shared) as adapter:
        state = run_batch(stage / 'batch', state=start, panels=current, adapter=adapter,
                          budget=manifest['budget']['per_round'],
                          learning_history_tasks=previous, seed=_seed(manifest['seed'], family, 'batch', index))
    _checked_state(stage)
    expected_ids = [t['id'] for t in previous + current['learning']]
    if [row['task_id'] for row in state['ledger']] != expected_ids:
        raise ValueError('Committed bank does not cover the exact learning stream')
    _link_learning(root, stage, state, len(previous))
    save_registration(stage / 'candidate_differences.json', candidate_differences(stage))
    save_registration(stage / 'complete.json', _round_receipt(stage, state))


def _run_final(root, family, cfg, manifest):
    state = _committed_round(root, FAMILY_ROUNDS - 1)
    plan = manifest['families'][family]
    if (root / 'complete.json').is_file():
        _completed_family(root, manifest, family)
        return
    tasks = plan['final']
    frameworks = {'initial': _checkpoint(initial_state()), 'final': _checkpoint(state)}
    # The official initializer needs a train asset even when every evaluation reset is held out.
    staged = [plan['batches'][0]['learning'][0], *tasks]
    with open_stage(root / 'final', cfg, manifest, staged) as adapter:
        records = dict(family=family, round_6_state_hash=digest(state), records=adapter.evolution_memory(state['history']))
        save_registration(root / 'evolution_memory.json', records)
        frozen = dict(family=family, round_6_state_hash=digest(state), frameworks=frameworks,
                      test_panel_hash=digest(tasks), shared_archive_hash=digest(records),
                      initial_task_memory='separate_empty_banks', task_memory_updates='online_same_family',
                      evolve_during_test=False)
        save_registration(root / 'frozen.json', frozen)
        results, costs = _native_panel(root / 'final', adapter, [frameworks[a] for a in ARMS],
                                       tasks, manifest['seed'], arms=ARMS)
    save_registration(root / 'final/summary.json',
                      dict(frozen_hash=digest(frozen), results=results, costs=costs,
                           comparison=_final_comparison(results, tasks)))
    save_registration(root / 'complete.json', _end_receipt(root, manifest, family))


def run_stage(root, cfg, manifest, name):
    validate_sequential_manifest(manifest)
    order = manifest['family_order']
    stages = {_round_name(i): i for i in range(FAMILY_ROUNDS)}
    if name not in {f'{family}/{stage}' for family in order for stage in (*stages, 'final')}:
        raise ValueError('Unregistered family or stage')
    family, stage_name = name.split('/')
    root = Path(root).resolve()
    if _read(root / 'registration.json') != manifest or _read(root / 'configuration.json') != _configuration(cfg):
        raise ValueError('Campaign source, model configuration or registration changed')
    shared = []
    if family == order[1]:
        first = root / order[0]
        _completed_family(first, manifest, order[0])
        shared = _read(first / 'evolution_memory.json')['records']
    family_root = root / family
    with _lock(family_root):
        if stage_name == 'final':
            _run_final(family_root, family, cfg, manifest)
        else:
            _run_round(family_root, family, stages[stage_name], cfg, manifest, shared)


def run_campaign(root, manifest_path, cfg_path):
    manifest_path, cfg_path = Path(manifest_path).resolve(), Path(cfg_path).resolve()
    root = Path(root).resolve()
    manifest = _read(manifest_path)
    validate_sequential_manifest(manifest)
    with _lock(root):
        save_registration(root / 'configuration.json', _configuration(_read(cfg_path)))
        save_registration(root / 'registration.json', manifest)
        completed = []
        for family in manifest['family_order']:
            for stage in [*(_round_name(i) for i in range(FAMILY_ROUNDS)), 'final']:
                name = f'{family}/{stage}'
                print(f'Campaign stage {name}', flush=True)
                subprocess.run([sys.executable, '-m', 'rsi_paired', 'campaign-stage', '--manifest', str(manifest_path),
                                '--model-config', str(cfg_path), '--out', str(root), '--stage', name], check=True)
            completed.append(_completed_family(root / family, manifest, family))
        save_registration(root / 'complete.json', dict(registration_hash=digest(manifest), families=completed))
