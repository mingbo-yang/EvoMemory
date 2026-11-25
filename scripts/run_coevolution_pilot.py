"""Wait for both v3 clients, reuse their resident services, and run separate v4 arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from rsi_paired.coevolution_selection import (
    ARMS, FAMILIES, MODELS, register_suite, select_suite_winner, select_winner, suite_runs,
)
from rsi_paired.panel import COEVOLUTION_MIXED_10X60_PROTOCOL, save_registration
from rsi_paired.workflow import _lock, _read, _write, digest


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def prior_completed(prior, unit_state):
    for model in MODELS:
        if unit_state(f'evomemory-parallel4-{model}.service') != 'inactive':
            return False
        if not all((prior / model / receipt).is_file()
                   for receipt in ('complete.json', 'resident-after.json')):
            return False
    return True


def unit_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value'],
                            text=True, capture_output=True, timeout=15)
    return result.stdout.strip()


def legacy_env(source):
    return dict(os.environ, PYTHONPATH=str(source))


def check_prior(prior, source):
    # Import and verify with the original frozen checkout and native wire receipts.
    program = """
import runpy, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
verify = runpy.run_path(str(root / 'release-after-formal.py'))
read = verify['read']
manifest = read(root / 'panel.json')
verify['validate_sequential_manifest'](manifest)
for name in verify['MODELS']:
    verify['verify_campaign'](root / name, read(root / 'configs' / (name + '.json')), manifest)
"""
    subprocess.run([sys.executable, '-B', '-c', program, str(prior)], cwd=source,
                   env=legacy_env(source), check=True)


def await_idle(prior, source):
    program = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from monitor import MODELS, model_health, read, units
for name in MODELS:
    cfg = read(root / 'configs' / (name + '.json'))
    unit = cfg['resident_service']['unit']
    health = model_health(cfg, units([unit]).get(unit, {}))
    if not health['identity_verified']:
        raise RuntimeError('Registered model changed: ' + name)
    if health.get('running') != 0 or health.get('waiting') != 0:
        sys.exit(75)
"""
    while True:
        result = subprocess.run([sys.executable, '-B', '-c', program, str(prior)],
                                cwd=source, env=legacy_env(source))
        if result.returncode == 0:
            return
        if result.returncode != 75:
            result.check_returncode()
        time.sleep(20)


def guard(root, model, path):
    subprocess.run([sys.executable, '-B', str(path), str(root / 'configs' / f'{model}.json')],
                   check=True, timeout=60, stdout=subprocess.DEVNULL)


def run_model(root, model, task, guard_path):
    log_path = root / (model + '.log')
    phases = {'live_smoke': ('D',), 'arms': ARMS, 'final': ('final',)}[task]
    with log_path.open('a', buffering=1) as log:
        for phase in phases:
            guard(root, model, guard_path)
            _write(root / (model + '-status.json'), dict(phase=phase, status='running', at=now()))
            opts = (['coevolve-final', '--winner', str(root / 'winner.json')] if phase == 'final'
                    else ['coevolve-arm', '--arm', phase])
            if task == 'live_smoke':
                opts.append('--first-round-only')
            print(now(), model, *opts, file=log, flush=True)
            subprocess.run([sys.executable, '-Bu', '-m', 'rsi_paired', *opts,
                            '--manifest', str(root / 'panel.json'), '--out', str(root / model),
                            '--model-config', str(root / 'configs' / f'{model}.json')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            guard(root, model, guard_path)
            _write(root / (model + '-status.json'), dict(phase=phase, status='complete', at=now()))


def completed_pair(root, manifest):
    from rsi_paired.family_campaign import _final_comparison

    winner = _read(select_winner(root, manifest))
    hashes = {}
    for model in MODELS:
        directory = root / model
        summary = _read(directory / 'final/summary.json')
        frozen = _read(directory / 'final/frozen.json')
        expected = dict(final_hash=digest(summary), winner_hash=digest(winner),
                        registration_hash=digest(manifest), arm=winner['arm'])
        if (_read(directory / 'complete.json') != expected or summary['arm'] != winner['arm']
                or summary['registration_hash'] != digest(manifest)
                or summary['model_config_hash'] != digest(_read(root / 'configs' / (model + '.json')))
                or summary['winner_hash'] != digest(winner) or summary['frozen_hash'] != digest(frozen)
                or frozen['arm'] != winner['arm'] or frozen['panel_hash'] != digest(manifest['final'])
                or summary['comparison'] != _final_comparison(summary['results'], manifest['final'])):
            raise ValueError(f'Final comparison did not match the registered champion on {model}')
        hashes[model] = digest(summary)
    return dict(winner_hash=digest(winner), registration_hash=digest(manifest), models=hashes)


def pinned_checkout(checkout):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('Commit the tested source before registering the pilot')
    return dict(commit=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))


def study_inputs(root):
    if (root / 'study.json').is_file():
        return [(path, _read(path / 'panel.json')) for path in suite_runs(root).values()]
    return [(root, _read(root / 'panel.json'))]


def run_families(inputs, model, phase, guard_path):
    for study, _ in inputs:
        run_model(study, model, phase, guard_path)


def study_completion(root, inputs):
    if not (root / 'study.json').is_file():
        return completed_pair(*inputs[0])
    summaries = {plan['family']: completed_pair(study, plan) for study, plan in inputs}
    return dict(study_hash=digest(_read(root / 'study.json')), winner_hash=digest(_read(root / 'winner.json')),
                families=summaries)


def native_smoke_receipt(root, manifest):
    from rsi_paired.audit import audit_run
    from rsi_paired.coevolution_campaign import _bound_round, _check_model_root

    if manifest['protocol'] != COEVOLUTION_MIXED_10X60_PROTOCOL:
        raise ValueError('Expected the six-family mixed registration')
    models = {}
    for model in MODELS:
        directory = root / model
        cfg = _read(root / 'configs' / f'{model}.json')
        _check_model_root(directory, manifest, cfg)
        state = _bound_round(directory / 'D', manifest, 'D', 0)
        stage = directory / 'D/evolution-01'
        audit = audit_run(stage)
        event = state['history'][0]
        decisions = (event.get('decision') or {}).get('component_decisions') or {}
        if (audit['status'] not in ('verified', 'verified_fallback') or audit['findings']
                or not audit['coverage'].get('raw_model_and_native')
                or not audit['costs'].get('completed_native_episodes')
                or set(decisions) != {'reviewer', 'evolver'}
                or event['status'] in ('decision_error', 'budget_blocked')):
            raise ValueError(f'{model} mixed native smoke did not verify: {audit["status"]}, {event["status"]}')
        models[model] = dict(round_hash=digest(_read(stage / 'complete.json')), audit_status=audit['status'],
                             outcome=event['status'], paired_scored=(stage / 'batch/meta_results.json').exists(),
                             role_updates_requested={role: row['should_update'] for role, row in decisions.items()})
        if (stage / 'evidence-recovery.json').exists():
            binding = _read(stage / 'evidence-recovery.json')
            runtime = _read(stage / 'runtime-evidence-recovery.json')
            if (binding['previous_runtime_hash'] != digest(_read(stage / 'runtime.json'))
                    or any(runtime['software'][k] != v for k, v in binding['software'].items())):
                raise ValueError(f'{model} evidence recovery provenance differs from the runtime')
            models[model]['evidence_recovery_runtime_hash'] = digest(runtime)
    return dict(arm='D', round=1, registration_hash=digest(manifest), models=models)


def source_registration_path(root):
    revised = root / 'source-evidence-recovery.json'
    return revised if revised.exists() else root / 'source.json'


def smoke_publication_gate(root):
    marker = root / 'smoke-publication-ready.json'
    if not marker.exists():
        status = root / 'status.json'
        return dict(ready=False, status=_read(status)['phase'] if status.exists() else 'preparing')
    code = pinned_checkout(Path(__file__).resolve().parents[1])
    smoke = native_smoke_receipt(root, _read(root / 'panel.json'))
    expected = dict(**code, completion_hash=digest(smoke), target_branch='v4')
    if (_read(source_registration_path(root)) != code or _read(root / 'native-smoke.json') != smoke
            or _read(root / 'handoff.json')['registration_hash'] != smoke['registration_hash']
            or _read(marker) != expected):
        raise ValueError('Source or native smoke evidence changed before publication')
    return dict(ready=True, **expected)


def run(args):
    root, prior, source, guard_path = (p.resolve() for p in
                                       (args.root, args.prior, args.prior_source, args.guard))
    checkout = Path(__file__).resolve().parents[1]
    os.chdir(checkout)
    os.environ.update(PYTHONPATH=str(checkout), PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', EVOMEMORY_ENABLE_OPTIONAL_MODELS='0',
                      MODEL_CONTEXT_TOKENS='32768', TMPDIR='/tmp', CUDA_VISIBLE_DEVICES='')
    if args.suite:
        register_suite(root, {f: root / f for f in FAMILIES})
    inputs = study_inputs(root)
    suite = (root / 'study.json').is_file()
    mixed = not suite and inputs[0][1].get('protocol') == COEVOLUTION_MIXED_10X60_PROTOCOL
    registration_hash = digest(_read(root / 'study.json') if suite else inputs[0][1])
    with _lock(root / 'coordinator'):
        code = pinned_checkout(checkout)
        save_registration(source_registration_path(root), code)
        if (root / 'complete.json').exists() and (prior / 'release-result.json').exists():
            return
        for model in MODELS:
            cfg = _read(prior / 'configs' / f'{model}.json')
            for study, _ in inputs:
                save_registration(study / 'configs' / f'{model}.json', cfg)
        if not (root / 'handoff.json').exists():
            _write(root / 'status.json', dict(phase='waiting_for_both_v3_campaigns', since=now()))
            while not prior_completed(prior, unit_state):
                time.sleep(30)
            check_prior(prior, source)
            for model in MODELS:
                guard(inputs[0][0], model, guard_path)
            await_idle(prior, source)
            save_registration(root / 'handoff.json', dict(verified_at=now(), registration_hash=registration_hash,
                              prior_receipts={m: digest(_read(prior / m / 'complete.json')) for m in MODELS}))
        handoff = _read(root / 'handoff.json')
        if (not prior_completed(prior, unit_state) or handoff['registration_hash'] != registration_hash
                or handoff['prior_receipts'] != {m: digest(_read(prior / m / 'complete.json')) for m in MODELS}):
            raise ValueError('Earlier campaign or registration changed after handoff')
        if not (root / 'complete.json').is_file():
            phases = ('live_smoke', 'arms', 'final') if mixed else ('arms', 'final')
            for task in phases:
                if task == 'final':
                    if suite:
                        select_suite_winner(root)
                    else:
                        select_winner(*inputs[0])
                _write(root / 'status.json', dict(phase=task, since=now()))
                with ThreadPoolExecutor(max_workers=len(MODELS)) as workers:
                    jobs = [workers.submit(run_families, inputs, model, task, guard_path) for model in MODELS]
                    for job in jobs:
                        job.result()
                if task == 'live_smoke':
                    smoke = native_smoke_receipt(*inputs[0])
                    if pinned_checkout(checkout) != code:
                        raise ValueError('Source changed during the native smoke')
                    save_registration(root / 'native-smoke.json', smoke)
                    save_registration(root / 'smoke-publication-ready.json',
                                      dict(**code, completion_hash=digest(smoke), target_branch='v4'))
        completed = study_completion(root, inputs)
        if pinned_checkout(checkout) != code:
            raise ValueError('Source changed during the paired pilot')
        for study, _ in inputs:
            for model in MODELS:
                from rsi_paired.coevolution_campaign import model_configuration_path
                cfg = _read(study / 'configs' / f'{model}.json')
                configuration = model_configuration_path(study / model, _read(study / 'panel.json'), cfg)
                if _read(configuration)['code_commit'] != code['commit']:
                    raise ValueError(f'Different source was tested on {study.name}/{model}')
        save_registration(root / 'complete.json', completed)
        save_registration(root / 'publication-ready.json',
                          dict(**code, completion_hash=digest(completed), target_branch='v4'))
        _write(root / 'status.json', dict(phase='completed', at=now()))
        if args.release_after and not (prior / 'release-result.json').exists():
            # Honor the deferred v3 cleanup only after both new final jobs exit.
            subprocess.run([sys.executable, '-Bu', str(prior / 'release-after-formal.py'), str(prior),
                            str(guard_path)], cwd=source, env=legacy_env(source), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'prior', 'prior-source', 'guard'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--suite', action='store_true',
                        help='Register heat/panel.json and two/panel.json below the root for joint arm selection')
    parser.add_argument('--release-after', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        _write(args.root / 'status.json', dict(phase='failed', error=str(exc), at=now()))
        raise


if __name__ == '__main__':
    main()
