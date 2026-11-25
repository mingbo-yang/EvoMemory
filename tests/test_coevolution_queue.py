"""Exercise the actual suite coordinator without a model service or GPU."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_paired.panel import save_registration
from rsi_paired.workflow import _read
from scripts import run_coevolution_pilot as queue


def test_both_families_finish_selection_before_any_final(tmp_path, monkeypatch):
    root, prior = tmp_path / 'suite', tmp_path / 'prior'
    paths = {family: root / family for family in queue.FAMILIES}
    for family, p in paths.items():
        save_registration(p / 'panel.json', {'family': family})
    for m in queue.MODELS:
        save_registration(prior / 'configs' / (m + '.json'), {'model': m})
        save_registration(prior / m / 'complete.json', {'complete': True})
    events = []
    commit = 'synthetic-checkout'

    def register(path, roots):
        assert roots == paths
        save_registration(path / 'study.json', {'families': sorted(roots)})

    def model_phase(study, model, phase, guard_path):
        assert (root / 'handoff.json').is_file()
        assert _read(study / 'configs' / (model + '.json')) == {'model': model}
        if phase == 'final':
            assert (root / 'winner.json').is_file()
            save_registration(study / model / 'configuration.json', {'code_commit': commit})
        events.append((study.name, model, phase))

    def select(path):
        assert set(events) == {(f, m, 'arms') for f in paths for m in queue.MODELS}
        save_registration(path / 'winner.json', {'arm': 'D'})

    monkeypatch.setattr(queue, 'register_suite', register)
    monkeypatch.setattr(queue, 'suite_runs', lambda _: paths)
    monkeypatch.setattr(queue, 'pinned_checkout', lambda _: {'commit': commit, 'tree': 'synthetic-tree'})
    monkeypatch.setattr(queue, 'prior_completed', lambda *_: True)
    monkeypatch.setattr(queue, 'check_prior', lambda *_: None)
    monkeypatch.setattr(queue, 'await_idle', lambda *_: None)
    monkeypatch.setattr(queue, 'guard', lambda *_: None)
    monkeypatch.setattr(queue, 'run_model', model_phase)
    monkeypatch.setattr(queue, 'select_suite_winner', select)
    monkeypatch.setattr(queue, 'completed_pair', lambda p, manifest: {'family': p.name})
    monkeypatch.setattr(queue.os, 'environ', dict(queue.os.environ))
    monkeypatch.chdir(Path.cwd())
    args = SimpleNamespace(root=root, prior=prior, prior_source=tmp_path, guard=tmp_path,
                           release_after=False, suite=True)
    queue.run(args)
    assert len(events) == 8 and all((f, m, 'final') in events for f in paths for m in queue.MODELS)
    assert set(_read(root / 'complete.json')['families']) == set(paths)
    assert _read(root / 'publication-ready.json')['commit'] == commit
    queue.run(args)
    assert len(events) == 8


def test_mixed_two_model_smoke_then_four_arms_then_one_final(tmp_path, monkeypatch):
    root, prior = tmp_path / 'mixed', tmp_path / 'prior'
    plan = {'protocol': queue.COEVOLUTION_MIXED_10X60_PROTOCOL}
    save_registration(root / 'panel.json', plan)
    source = {'commit': 'same-commit', 'tree': 'same-tree'}
    for model in queue.MODELS:
        save_registration(prior / 'configs' / f'{model}.json', {'model': model})
        save_registration(prior / model / 'complete.json', {'complete': True})
    events = []

    def model_phase(path, model, phase, guard_path):
        assert path == root and (root / 'handoff.json').exists()
        if phase == 'arms':
            assert queue.smoke_publication_gate(root)['ready']
        if phase == 'final':
            assert _read(root / 'winner.json')['arm'] == 'C'
        save_registration(root / model / 'configuration.json', {'code_commit': source['commit']})
        events.append((phase, model))

    def select(path, manifest):
        assert set(events) == {(p, m) for p in ('live_smoke', 'arms') for m in queue.MODELS}
        save_registration(path / 'winner.json', {'arm': 'C'})

    def smoke(path, manifest):
        assert all(('live_smoke', m) in events for m in queue.MODELS)
        return {'registration_hash': queue.digest(manifest), 'models': list(queue.MODELS)}

    monkeypatch.setattr(queue, 'pinned_checkout', lambda _: source)
    monkeypatch.setattr(queue, 'prior_completed', lambda *_: True)
    monkeypatch.setattr(queue, 'check_prior', lambda *_: None)
    monkeypatch.setattr(queue, 'await_idle', lambda *_: None)
    monkeypatch.setattr(queue, 'guard', lambda *_: None)
    monkeypatch.setattr(queue, 'run_model', model_phase)
    monkeypatch.setattr(queue, 'native_smoke_receipt', smoke)
    monkeypatch.setattr(queue, 'select_winner', select)
    monkeypatch.setattr(queue, 'completed_pair', lambda *_: {'arm': 'C'})
    monkeypatch.setattr(queue.os, 'environ', dict(queue.os.environ))
    monkeypatch.chdir(Path.cwd())
    args = SimpleNamespace(root=root, prior=prior, prior_source=tmp_path, guard=tmp_path,
                           release_after=False, suite=False)
    assert not queue.smoke_publication_gate(root)['ready']
    queue.run(args)
    assert [p for p, _ in events] == ['live_smoke'] * 2 + ['arms'] * 2 + ['final'] * 2
    assert queue.smoke_publication_gate(root)['tree'] == source['tree']
    queue.run(args)
    assert len(events) == 6
    queue._write(root / 'native-smoke.json', {'changed': True})
    with pytest.raises(ValueError, match='changed'):
        queue.smoke_publication_gate(root)
