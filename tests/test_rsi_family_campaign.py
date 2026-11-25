"""Sequential orchestration and evidence boundaries using synthetic adapters, without a model."""
from contextlib import contextmanager
from copy import deepcopy
from threading import Barrier

import pytest

from rsi_paired import campaign, family_campaign as families
from rsi_paired.panel import HISTORICAL_SEQUENTIAL_PROTOCOL, SEQUENTIAL_PROTOCOL, save_registration
from rsi_paired.workflow import _read, _write, digest
from tests.test_rsi_family_panel import archives, manifest  # shared synthetic official-layout fixtures
from tests.test_rsi_roles import roles


class PanelAdapter:
    fingerprint = 'synthetic'

    def __init__(self):
        self.calls = []

    def set_job(self, path):
        self.job = path

    def evaluate(self, source, bank, tasks, seeds):
        assert bank == {}
        self.calls.append((self.job, source, tasks, seeds))
        bank['local'] = source
        return [dict(task_id=t['id'], native_score=1, score=1, status='completed', evidence={'steps': 3},
                     episode_hash=digest([str(self.job), t['id']])) for t in tasks]


class FixtureWireAudit:
    """Native wire validation is exercised by the native and audit suites."""
    def __init__(self, root):
        self.errors, self.native_missing_wire = [], []
        self.native_rows, self.native_verified = {}, set()
        self.report = {'costs': {}}

    def costs_and_wire(self):
        pass

    def bind_panel(self, key, tasks, seeds, source, bank, outcomes):
        assert bank == {} and [row['task_id'] for row in outcomes] == [t['id'] for t in tasks]


class ParallelPanelAdapter(PanelAdapter):
    candidate_workers = 2

    def __init__(self):
        super().__init__()
        self.both_arms_started = Barrier(2)

    def evaluate_isolated(self, job, source, bank, tasks, seeds):
        self.both_arms_started.wait(timeout=10)
        worker = PanelAdapter()
        worker.set_job(job)
        result = worker.evaluate(source, bank, tasks, seeds)
        self.calls.extend(worker.calls)
        return result


@pytest.mark.parametrize("adapter_class", [PanelAdapter, ParallelPanelAdapter])
def test_identical_final_frameworks_really_execute_twice_and_resume(tmp_path, monkeypatch, adapter_class):
    monkeypatch.setattr(campaign, 'Audit', FixtureWireAudit)
    states = [{'mechanism': 'same', 'evolver': 'same'}] * 2
    tasks = [dict(id=str(i), family='simple', split='valid_seen', group_id=f'room-{i % 5}') for i in range(100)]
    adapter = adapter_class()
    results, _ = campaign._native_panel(tmp_path, adapter, states, tasks, 33, arms=families.ARMS)
    assert len(adapter.calls) == 2 and adapter.calls[0][0] != adapter.calls[1][0]
    assert adapter.calls[0][1:] == adapter.calls[1][1:]
    comparison = families._final_comparison(results, tasks)
    assert comparison['paired_native']['ties'] == 100
    assert comparison['success_count_difference'] == 0 and comparison['game_groups'] == 5
    assert comparison['cumulative_successes']['final'] == list(range(1, 101))
    assert comparison['results_by_game_group']['room-4']['initial_successes'] == 20
    assert comparison['arms']['initial']['recorded_environment_steps'] == 300
    assert campaign._native_panel(tmp_path, adapter, states, tasks, 33, arms=families.ARMS)[0] == results
    assert len(adapter.calls) == 2


def test_only_compact_evolution_history_can_be_retrieved_in_second_family(tmp_path):
    r, _ = roles(tmp_path, {})
    event = {'task_memory': 'private bank contents', 'learning': 'raw native trajectory',
             'primary_review': {'findings': [{'judgment': 'insufficient_evidence',
                'evidence_ids': ['primary.P1'], 'observation': 'private task detail'}]},
             'modification_records': [{'slot': 'P1', 'success_count': 0,
                'diff_excerpt': ['+    value = "private answer"']}]}
    exported = r.evolution_memory([event])
    r.shared_evolution_memory = tuple(exported)
    public = r.compress({'phase': 'primary', 'history': [], 'diagnosis': {}})
    assert len(public['history']) == 1
    assert public['history'] == exported and 'private' not in str(exported)
    assert exported[0]['reviewer_judgments'] == ['insufficient_evidence'] and '[literal]' in str(exported)
    assert len(list((r.root / 'retrieval').glob('*.json'))) == 1


@pytest.mark.parametrize("protocol", [SEQUENTIAL_PROTOCOL, HISTORICAL_SEQUENTIAL_PROTOCOL])
def test_two_complete_families_and_only_development_experiences_cross(tmp_path, manifest, monkeypatch, protocol):
    manifest = dict(manifest, protocol=protocol)
    cfg = {'model': 'synthetic'}
    root = tmp_path / 'output'
    mp, cp = tmp_path / 'plan.json', tmp_path / 'model.json'
    save_registration(mp, manifest)
    save_registration(cp, cfg)
    save_registration(root / 'registration.json', manifest)
    save_registration(root / 'configuration.json', families._configuration(cfg))
    with pytest.raises(ValueError, match='both frozen tests before transfer'):
        families.run_stage(root, cfg, manifest, 'two/evolution-01')
    assert not (root / 'two').exists()

    stages, starts, seen_shared, final_calls = [], [], [], []
    archive = [dict(slot='P1', batch=i) for i in range(6)]

    class Adapter(PanelAdapter):
        def evolution_memory(self, history):
            assert len(history) == 6
            return archive

    @contextmanager
    def open_fake(path, config, plan, tasks, **kwargs):
        assert len({task['family'] for task in tasks}) == 1
        adapter = Adapter()
        seen_shared.append((path.parent.name, path.name, list(kwargs.get('shared_evolution_memory', ()))))
        yield adapter
        if path.name == 'final':
            final_calls.append(adapter.calls)

    def batch_fake(path, *, state, panels, adapter, learning_history_tasks, seed, budget, protocol):
        family = path.parent.parent.name
        assert protocol == manifest["protocol"]
        assert budget == {'episodes': 390, 'infrastructure_attempts': 2}
        assert len(panels['learning']) == 30
        assert [t['id'] for t in learning_history_tasks] == [r['task_id'] for r in state['ledger']]
        starts.append((family, deepcopy(state)))
        ledger = state['ledger'] + [dict(task_id=t['id'], family=family, phase='learning')
                                    for t in panels['learning']]
        new = dict(ledger=ledger, history=state['history'] + [len(ledger)], memory={family: len(ledger)},
                   mechanism=f'{family}/{len(ledger)}', evolver={'propose': f'{family} policy'})
        save_registration(path / 'commit.json', dict(state_hash=digest(new), state=new))
        save_registration(path / 'resolved_panels.json', {'seed': seed})
        return new

    monkeypatch.setattr(families, 'open_stage', open_fake)
    monkeypatch.setattr(families, 'run_batch', batch_fake)
    monkeypatch.setattr(families, '_link_learning', lambda *args: None)
    monkeypatch.setattr(families, 'candidate_differences', lambda stage: {'stage': stage.name})
    monkeypatch.setattr(families, '_checked_state', lambda stage: _read(stage / 'batch/commit.json')['state'])
    monkeypatch.setattr(campaign, 'Audit', FixtureWireAudit)

    real_run = families.subprocess.run
    def run_inline(command, **kwargs):
        if '--stage' not in command:
            return real_run(command, **kwargs)
        assert kwargs['check']
        stage = command[command.index('--stage') + 1]
        stages.append(stage)
        families.run_stage(root, cfg, manifest, stage)

    monkeypatch.setattr(families.subprocess, 'run', run_inline)
    families.run_campaign(root, mp, cp)
    expected = [f'{f}/{s}' for f in ('simple', 'two')
                for s in [*(f'evolution-{i:02d}' for i in range(1, 7)), 'final']]
    assert stages == expected
    assert [s for f, s in starts if not s['ledger']] == [families.initial_state()] * 2
    assert [len(s['ledger']) for f, s in starts] == list(range(0, 180, 30)) * 2
    assert all(entries == (archive if f == 'two' and stage != 'final' else [])
               for f, stage, entries in seen_shared)
    assert [len(calls) for calls in final_calls] == [2, 2]
    for family, calls in zip(manifest['family_order'], final_calls):
        plan = manifest['families'][family]
        assert [c[1] for c in calls] == [families.initial_state()['mechanism'], family + '/180']
        assert calls[0][2] == calls[1][2] == plan['final'] and calls[0][3] == calls[1][3]
        assert not (root / family / 'confirm').exists()
    assert [f['learning_tasks'] for f in _read(root / 'complete.json')['families']] == [180, 180]

    families.run_campaign(root, mp, cp)
    assert len(starts) == 12 and len(final_calls) == 2
    path = root / 'simple/evolution_memory.json'
    _write(path, dict(_read(path), records=['tampered']))
    with pytest.raises(ValueError, match='evolution archive'):
        families.run_stage(root, cfg, manifest, 'two/evolution-01')
