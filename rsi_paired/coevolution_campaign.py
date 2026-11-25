"""Isolated optimizer arms on paired data; a separately selected champion reaches the test."""
from pathlib import Path
import subprocess
import sys

from .campaign import _checked_state, _link_learning, _native_evaluation_key, _native_panel
from .coevolution_panel import validate_coevolution_manifest
from .comparison import candidate_differences
from .family_campaign import _configuration, _final_comparison, _round_name, _round_receipt
from .panel import save_registration
from .runner import initial_state, open_stage
from .workflow import _lock, _read, _seed, digest, run_batch

ARMS = ('A', 'B', 'C', 'D')


def _framework(state):
    return {key: state[key] for key in ('mechanism', 'reviewer', 'evolver')}


def _bound_round(arm_root, manifest, arm, index):
    """Check the audited completion seal and the checkpoint for this arm and registration."""
    stage = arm_root / _round_name(index)
    commit = _read(stage / 'batch/commit.json')
    state = commit['state']
    registration = _read(stage / 'batch/registration.json')
    learning_tasks = [t for batch in manifest['batches'][:index + 1] for t in batch['learning']]
    if manifest.get('family') == 'mixed' and (set(state['memory']) != set(manifest['families']) or
            [row['family'] for row in state['ledger']] != [t['family'] for t in learning_tasks]):
        raise ValueError('Mixed checkpoint has a missing family bank or a changed learning stream')
    learning = [t['id'] for t in learning_tasks]
    if (state.get('arm') != arm or commit['state_hash'] != digest(state)
            or [row['task_id'] for row in state['ledger']] != learning
            or len(state['history']) != index + 1
            or _read(stage / 'complete.json') != _round_receipt(stage, state)
            or registration['protocol'] != manifest['protocol']
            or registration['budget'] != manifest['budget']['per_round']
            or registration['panels'] != manifest['batches'][index]
            or registration['panel_hash'] != digest(manifest['batches'][index])):
        raise ValueError('Round checkpoint differs from its completed arm, games or registration')
    return state


def model_configuration_path(root, manifest, cfg):
    base = root / 'configuration.json'
    revised = root / 'configuration-evidence-recovery.json'
    if not revised.exists():
        return base
    old, new = _read(base), _read(revised)
    allowed = {'roles.py', 'runner.py', 'coevolution_campaign.py'}
    if (manifest.get('family') != 'mixed' or new != _configuration(cfg)
            or old != dict(new, code_commit=old['code_commit'], source_hashes={
                k: old['source_hashes'][k] if k in allowed else v for k, v in new['source_hashes'].items()
            })):
        raise ValueError('Unapproved campaign implementation change during evidence recovery')
    return revised


def _check_model_root(root, manifest, cfg):
    if (_read(root / 'registration.json') != manifest or
            _read(model_configuration_path(root, manifest, cfg)) != _configuration(cfg)):
        raise ValueError('Model source, registration or configuration changed')


def _run_round(model_root, cfg, manifest, arm, index):
    arm_root = model_root / arm
    stage = arm_root / _round_name(index)
    start = initial_state(arm=arm) if index == 0 else _bound_round(arm_root, manifest, arm, index - 1)
    save_registration(stage / 'batch/starting_state.json', start)
    if (stage / 'complete.json').is_file():
        _bound_round(arm_root, manifest, arm, index)
        return
    previous = [t for batch in manifest['batches'][:index] for t in batch['learning']]
    current = manifest['batches'][index]
    tasks = previous + [t for phase in ('learning','primary','meta') for t in current[phase]]
    archive = arm_root / 'learning'
    archive.mkdir(exist_ok=True)
    with open_stage(stage, cfg, manifest, tasks, learning_archive=archive, arm=arm) as adapter:
        state = run_batch(stage / 'batch', state=start, panels=current, adapter=adapter,
                          protocol=manifest['protocol'], budget=manifest['budget']['per_round'],
                          learning_history_tasks=previous, seed=_seed(manifest['seed'], manifest['family'], 'batch', index))
    _checked_state(stage)
    if [row['task_id'] for row in state['ledger']] != [t['id'] for t in previous + current['learning']]:
        raise ValueError('Committed learning prefix differs from the registered stream')
    _link_learning(arm_root, stage, state, len(previous))
    save_registration(stage / 'candidate_differences.json', candidate_differences(stage))
    save_registration(stage / 'complete.json', _round_receipt(stage, state))


def _selection_frozen(manifest, state, arm):
    return dict(arm=arm, registration_hash=digest(manifest), state_hash=digest(state), framework=_framework(state),
                panel_hash=digest(manifest['selection']), initial_task_memory='empty', evolve_during_test=False)


def _run_selection(root, cfg, manifest, arm):
    arm_root = root / arm
    state = _bound_round(arm_root, manifest, arm, manifest['rounds'] - 1)
    stage = arm_root / 'selection'
    frozen = _selection_frozen(manifest, state, arm)
    save_registration(stage / 'frozen.json', frozen)
    if (stage / 'complete.json').exists():
        validate_arm_selection(root, manifest, cfg, arm)
        return
    tasks = manifest['selection']
    staged = [manifest['batches'][0]['learning'][0], *tasks]
    with open_stage(stage, cfg, manifest, staged, arm=arm) as adapter:
        results, costs = _native_panel(stage, adapter, [frozen['framework']], tasks, manifest['seed'], arms=[arm])
    result = results[0]
    summary = dict(arm=arm, successes=result['counts']['all']['succeeded'], tasks=len(tasks),
                   panel_hash=digest(tasks), state_hash=frozen['state_hash'], registration_hash=digest(manifest),
                   model_config_hash=digest(cfg), frozen_hash=digest(frozen), result=result, costs=costs)
    save_registration(stage / 'summary.json', summary)
    save_registration(stage / 'complete.json', dict(summary_hash=digest(summary)))
    validate_arm_selection(root, manifest, cfg, arm)


def validate_arm_selection(model_root, manifest, cfg, arm):
    """Rebind the selection result to the sealed scores, rollout job and audited last checkpoint."""
    root = Path(model_root)
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    _check_model_root(root, manifest, cfg)
    state = _bound_round(root / arm, manifest, arm, manifest['rounds'] - 1)
    stage = root / arm / 'selection'
    summary = _read(stage / 'summary.json')
    frozen = _selection_frozen(manifest, state, arm)
    tasks = manifest['selection']
    result = summary['result']
    episodes = result['episodes']
    if (summary['arm'] != arm or result['arm'] != arm or summary['model_config_hash'] != digest(cfg)
            or summary['panel_hash'] != digest(tasks) or summary['state_hash'] != digest(state)
            or summary['registration_hash'] != digest(manifest) or summary['frozen_hash'] != digest(frozen)
            or _read(stage / 'frozen.json') != frozen or _read(stage / 'complete.json') != {'summary_hash': digest(summary)}
            or summary['tasks'] != len(tasks) or [r['task_id'] for r in episodes] != [t['id'] for t in tasks]
            or result['mechanism_hash'] != digest(state['mechanism']) or result['policy_hash'] != digest(state['evolver'])
            or result['counts']['all']['attempted'] != len(tasks) or summary['successes'] != sum(r['score'] for r in episodes)
            or summary['successes'] != result['counts']['all']['succeeded']):
        raise ValueError('Selection score, registered order, or frozen arm identity changed')
    key = result['evaluation_key']
    runtime = _read(stage / 'runtime.json')
    seeds = [_seed(manifest['seed'], 'native', t['id']) for t in tasks]
    if (runtime['arm'] != arm or runtime['panel_hash'] != digest(manifest)
            or key != _native_evaluation_key(digest(runtime), state['mechanism'], tasks, seeds, arm)):
        raise ValueError('Selection evaluation key differs from the frozen mechanism, runtime, seeds or arm')
    job = _read(stage / 'batch/jobs/evaluation' / key / 'result.json')
    native = job['outcome']
    if (not native['ok'] or digest(native) != job['outcome_hash'] or job['request_hash'] != digest(key) or
            [[r[k] for k in ('task_id','episode_hash','status','score','native_score')] for r in native['value']] !=
            [[r[k] for k in ('task_id','episode_hash','status','score','native_score')] for r in episodes]):
        raise ValueError('Selection summary differs from the audited native evaluation')
    return summary


def run_arm_stage(root, cfg, manifest, arm, stage):
    validate_coevolution_manifest(manifest)
    _check_model_root(root, manifest, cfg)
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    rounds = {_round_name(i): i for i in range(manifest['rounds'])}
    with _lock(root / arm):
        if stage == 'selection':
            _run_selection(root, cfg, manifest, arm)
        elif stage in rounds:
            _run_round(root, cfg, manifest, arm, rounds[stage])
        else:
            raise ValueError('Unregistered campaign stage')


def run_arm(root, manifest_path, cfg_path, arm, *, first_round_only=False):
    manifest_path, cfg_path = Path(manifest_path).resolve(), Path(cfg_path).resolve()
    root = Path(root).resolve()
    manifest, cfg = _read(manifest_path), _read(cfg_path)
    validate_coevolution_manifest(manifest)
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    with _lock(root):
        save_registration(root / 'registration.json', manifest)
        save_registration(model_configuration_path(root, manifest, cfg), _configuration(cfg))
        stages = [_round_name(0)] if first_round_only else [*(_round_name(i) for i in range(manifest['rounds'])), 'selection']
        for stage in stages:
            print(f'{arm} {stage}', flush=True)
            subprocess.run([sys.executable, '-m', 'rsi_paired', 'coevolve-stage', '--manifest', str(manifest_path),
                            '--model-config', str(cfg_path), '--out', str(root), '--arm', arm, '--stage', stage], check=True)


def run_champion(root, cfg, manifest, winner_path):
    from .coevolution_selection import verify_winner
    root = Path(root).resolve()
    winner_path = Path(winner_path).resolve()
    if winner_path != root.parent / 'winner.json':
        raise ValueError('The champion must be registered by the pooled two-model comparison')
    validate_coevolution_manifest(manifest)
    with _lock(root):
        _check_model_root(root, manifest, cfg)
        arm = verify_winner(winner_path, manifest, root, cfg)
        state = _bound_round(root / arm, manifest, arm, manifest['rounds'] - 1)
        tasks = manifest['final']
        frameworks = {'initial': _framework(initial_state(arm=arm)), 'final': _framework(state)}
        frozen = dict(arm=arm, state_hash=digest(state), winner_hash=digest(_read(winner_path)),
                      registration_hash=digest(manifest), frameworks=frameworks, panel_hash=digest(tasks),
                      initial_task_memory='separate_empty_banks', evolve_during_test=False)
        save_registration(root / 'final/frozen.json', frozen)
        with open_stage(root / 'final', cfg, manifest, [manifest['batches'][0]['learning'][0], *tasks], arm=arm) as adapter:
            results, costs = _native_panel(root / 'final', adapter, list(frameworks.values()), tasks, manifest['seed'],
                                           arms=('initial', 'final'))
        summary = dict(arm=arm, winner_hash=frozen['winner_hash'], frozen_hash=digest(frozen),
                       registration_hash=digest(manifest), model_config_hash=digest(cfg),
                       comparison=_final_comparison(results, tasks), results=results, costs=costs)
        save_registration(root / 'final/summary.json', summary)
        save_registration(root / 'complete.json', dict(final_hash=digest(summary), winner_hash=frozen['winner_hash'],
                                                       registration_hash=digest(manifest), arm=arm))
