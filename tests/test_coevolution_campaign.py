"""Run the two-model, four-arm campaign contract with synthetic environment and role responses."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest

from rsi_paired import campaign, coevolution_campaign as c, coevolution_selection as selection
from rsi_paired.audit import audit_run
from rsi_paired.panel import COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL, COEVOLUTION_MIXED_10X60_PROTOCOL, COEVOLUTION_PROTOCOL, save_registration
from rsi_paired.workflow import InfrastructurePending, _lock, _read, _write, digest
from tests.test_coevolution_panel import panel
from tests.test_coevolution_workflow import ArmFixture
from tests.test_rsi_family_campaign import FixtureWireAudit


@pytest.mark.parametrize("protocol,train,holdout,final", [
    (COEVOLUTION_PROTOCOL, 150, 50, 50),
    (COEVOLUTION_6X30_PROTOCOL, 90, 30, 30),
    (COEVOLUTION_HEAT_5X23_PROTOCOL, 69, 30, 50),
    (COEVOLUTION_MIXED_10X60_PROTOCOL, None, 30, 100),
])
def test_two_models_four_isolated_arms_resume_and_select_one_champion(monkeypatch, tmp_path, protocol, train, holdout, final):
    mixed = protocol == COEVOLUTION_MIXED_10X60_PROTOCOL
    if mixed:
        from tests.test_mixed_panel import synthetic_mixed_manifest
        manifest = synthetic_mixed_manifest(tmp_path, monkeypatch)
    else:
        manifest = panel(monkeypatch, tmp_path, train, protocol=protocol, holdout_size=holdout, final_size=final)
    root = tmp_path / 'campaign'
    mp = root / 'panel.json'
    save_registration(mp, manifest)
    configs = {m: {'model': m} for m in selection.MODELS}
    # Represent two immutable checkout registrations even while developers edit the live tree.
    pinned = {digest(cfg): c._configuration(cfg) for cfg in configs.values()}
    monkeypatch.setattr(c, '_configuration', lambda cfg: pinned[digest(cfg)])
    for m, cfg in configs.items():
        save_registration(root / 'configs' / (m + '.json'), cfg)
    opened = Counter()
    seen_stages = []
    interrupt = [True]

    class Adapter(ArmFixture):
        def __init__(self, model, stage, arm):
            super().__init__(arm, ('reviewer', 'evolver'))
            self.model, self.stage = model, stage

        def learn(self, source, memory, task, seed):
            memory.setdefault(task['family'] if mixed else 'learning', []).append(task['id'])
            row = dict(task_id=task['id'], family=task['family'], phase='learning',
                       episode_hash=digest([self.arm, self.model, task['id']]), status='completed', native_score=0, score=0)
            save_registration(self.stage / 'native/learning' / (row['episode_hash'] + '.json'), row)
            return memory, row

        def prepare(self, source, parent, memory, ledger):
            if mixed:
                return {family: [row['task_id'] for row in ledger if row['family'] == family] for family in memory}
            return super().prepare(source, parent, memory, ledger)

        def decide_component(self, role, policy, packet):
            # Only the first round is made to self-revise in this fixture so
            # repeated real-width groups cover both paired and fallback paths.
            if mixed and self.stage.name != 'evolution-01':
                self.update.clear()
            return super().decide_component(role, policy, packet)

        def propose(self, policy, source, packet, seed, previous, error):
            focus = packet.get('pre', packet)['diagnosis']['synthetic_diagnostic_focus']
            body = f'synthetic-{1 + bool(focus) + 2 * (policy["propose"] == "new")}'
            return dict(source=body, loaded_evolver_hash=digest(policy), public_hash=digest(packet))

        def evaluate(self, source, memory, tasks, seeds):
            if self.stage.name in ('selection', 'final'):
                assert memory == {}
                opened[self.model, self.arm, self.stage.name] += 1
                if self.stage.name == 'selection' and self.model == selection.MODELS[0] and self.arm == 'A' and interrupt[0]:
                    interrupt[0] = False
                    raise OSError('synthetic interruption between training and validation')
            count = int(source.split('-')[-1]) if source.startswith('synthetic-') else 0
            return [dict(task_id=t['id'], family=t['family'], episode_hash=digest([str(self.job), t['id']]), evidence={'steps': 1},
                         status='completed', native_score=int(i < count), score=int(i < count)) for i, t in enumerate(tasks)]

    @contextmanager
    def stage(path, cfg, registered, tasks, *, arm, **kwargs):
        assert registered == manifest
        seen_stages.append((cfg['model'], arm, path.name))
        adapter = Adapter(cfg['model'], path, arm)
        adapter.protocol = registered['protocol']
        adapter.registered_batches = tuple(digest(b) for b in registered['batches'])
        if path.name in ('selection','final'):
            runtime = dict(arm=arm, panel_hash=digest(registered), synthetic=True)
            save_registration(path / 'runtime.json', runtime)
            adapter.fingerprint = digest(runtime)
        yield adapter

    def synthetic_commit(stage):
        result = audit_run(stage)
        assert result['status'] in ('verified_workflow_only', 'verified_fallback'), result
        assert not result['findings']
        return _read(stage / 'batch/commit.json')['state']

    # Only native provenance and service work are synthetic; the real workflow, journals,
    # locks, registration checks, scorer, reports and winner selection all run.
    monkeypatch.setattr(c, 'open_stage', stage)
    monkeypatch.setattr(c, '_checked_state', synthetic_commit)
    monkeypatch.setattr(c, 'candidate_differences', lambda stage: {'synthetic_wire': True})
    monkeypatch.setattr(campaign, 'Audit', FixtureWireAudit)
    original_subprocess = c.subprocess.run

    def inline(args, **kwargs):
        if 'coevolve-stage' not in args:
            return original_subprocess(args, **kwargs)
        values = dict(zip(args[4::2], args[5::2]))
        c.run_arm_stage(Path(values['--out']), _read(Path(values['--model-config'])),
                        _read(Path(values['--manifest'])), values['--arm'], values['--stage'])

    monkeypatch.setattr(c.subprocess, 'run', inline)
    winner_path = root / 'winner.json'
    if mixed:
        for name in selection.MODELS:
            cfg_path = root / 'configs' / f'{name}.json'
            c.run_arm(root / name, mp, cfg_path, 'D', first_round_only=True)
            first = root / name / 'D/evolution-01'
            assert (first / 'complete.json').is_file()
            assert not (first.parent / 'evolution-02').exists() and not (first.parent / 'selection').exists()
    for name in selection.MODELS:
        cfg_path = root / 'configs' / f'{name}.json'
        for arm in c.ARMS:
            if name == selection.MODELS[0] and arm == 'A':
                with _lock(root / name), pytest.raises(BlockingIOError):
                    c.run_arm(root / name, mp, cfg_path, arm)
                with pytest.raises(InfrastructurePending, match='interruption'):
                    c.run_arm(root / name, mp, cfg_path, arm)
                assert (root / name / arm / 'evolution-01/complete.json').exists()
                assert not (root / name / arm / 'selection/complete.json').exists()
                assert not winner_path.exists()
                with pytest.raises(FileNotFoundError):
                    c.run_champion(root / name, configs[name], manifest, winner_path)
                assert not (root / name / 'final').exists()
            c.run_arm(root / name, mp, cfg_path, arm)
            valid = c.validate_arm_selection(root / name, manifest, configs[name], arm)
            assert valid['successes'] == ('ABCD'.index(arm) + 1) and valid['tasks'] == holdout
            assert set(p.name for p in (root / name / arm / 'learning').iterdir()) == {
                digest([arm, name, t['id']]) + '.json' for batch in manifest['batches'] for t in batch['learning']}
    assert seen_stages.count((selection.MODELS[0], 'A', 'evolution-01')) == 1
    assert selection.select_winner(root, manifest) == winner_path
    assert _read(winner_path)['arm'] == 'D'
    # Even a correctly named winner file cannot substitute a different arm.
    winner = _read(winner_path)
    _write(winner_path, dict(winner, arm='C'))
    with pytest.raises(ValueError, match='winner differs'):
        c.run_champion(root / selection.MODELS[0], configs[selection.MODELS[0]], manifest, winner_path)
    _write(winner_path, winner)
    for name, cfg in configs.items():
        c.run_champion(root / name, cfg, manifest, winner_path)
        assert _read(root / name / 'complete.json')['arm'] == 'D'
        results = _read(root / name / 'final/summary.json')['results']
        assert [r['arm'] for r in results] == ['initial','final']
        assert [len(r['episodes']) for r in results] == [final, final]
        if mixed:
            assert seen_stages.count((name, 'D', 'evolution-01')) == 1
            stage = root / name / 'D/evolution-10'
            final_state = _read(stage / 'batch/commit.json')['state']
            assert len(final_state['history']) == 10 and len(final_state['ledger']) == 600
            assert all(len(final_state['memory'][f]) == 100 for f in manifest['families'])
            first = final_state['history'][0]
            assert set(first['paired_optimizer_family_outcomes']) == set(manifest['families'])
            assert all(sum(s['tasks'] for s in m['family_outcomes'].values()) == 60 for m in first['modification_records'])
            by_family = _read(root / name / 'final/summary.json')['comparison']['by_family']
            assert sorted(v['tasks'] for v in by_family.values()) == [16,16,17,17,17,17]
            assert sum(sum(v['paired_native'].values()) for v in by_family.values()) == final
        c.run_champion(root / name, cfg, manifest, winner_path)
        assert opened[name, 'D', 'final'] == 2   # one independent baseline and one winner, even after resume
        assert not any((root / name / arm / 'final').exists() for arm in c.ARMS)
