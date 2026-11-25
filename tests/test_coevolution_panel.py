"""Synthetic family inventory, including an exact capacity cutoff. No environment or model."""
from copy import deepcopy
from zipfile import ZipFile

import pytest

from evomem.common import digest
from rsi_paired import coevolution_panel as cp
from rsi_paired.panel import COEVOLUTION_PROTOCOL, COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL, resolve_family_panels


def panel(monkeypatch, tmp_path, train, *, protocol=COEVOLUTION_PROTOCOL, holdout_size=None, final_size=None):
    pools = {}
    for split, size in [('train', train), ('valid_seen', 51), ('valid_unseen', 49)]:
        rows = []
        for i in range(size):
            group = f'pick_heat_then_place_in_recep-cpu-{split}-{i}'
            member = f'json_2.1.1/{split}/{group}/synthetic/game.tw-pddl'
            rows.append((group, member, digest(member)))
        pools[split] = rows
    def task(member, sha, family, metadata, seed):
        _, split, group, _, _ = member.split('/')
        return dict(archive_member=member, game_sha256=sha, family=family, id='rsi-alfworld/' + digest(member)[:18],
                    group_id=group, seed=int(digest([seed, 'environment', member])[:8], 16) % 2147483647,
                    environment='alfworld', room=int(group.rsplit('-',1)[1]), split=split)
    monkeypatch.setattr(cp, '_task', task)
    monkeypatch.setattr(cp, '_catalogue', lambda *args: {'heat': pools})
    monkeypatch.setattr(cp, 'file_hash', lambda path: cp.GAME_SHA)
    archive = tmp_path / 'synthetic.zip'
    with ZipFile(archive, 'w'):
        pass
    return cp.coevolution_manifest(archive, archive, 42, protocol=protocol, holdout_size=holdout_size, final_size=final_size)


@pytest.mark.parametrize('train, expected_rounds', [(299,1),(300,2),(1500,10),(1800,10)])
def test_reduces_rounds_before_batch_size(monkeypatch, tmp_path, train, expected_rounds):
    manifest = panel(monkeypatch, tmp_path, train)
    assert manifest['rounds'] == expected_rounds
    assert len(manifest['selection']) == len(manifest['final']) == 50
    assert all(len(tasks) == 50 for batch in manifest['batches'] for tasks in batch.values())
    cp.validate_coevolution_manifest(manifest)
    bad = deepcopy(manifest)
    bad['selection'][0] = bad['final'][0]
    with pytest.raises(ValueError, match='disjoint game groups'):
        cp.validate_coevolution_manifest(bad)


def test_joint_gates_preserve_complete_learning_prefix(monkeypatch, tmp_path):
    manifest = panel(monkeypatch, tmp_path, 300)
    for i, batch in enumerate(manifest['batches']):
        previous = [t for b in manifest['batches'][:i+1] for t in b['learning']]
        ledger = [dict(task_id=t['id'], family='heat', phase='learning', status='completed',
                       score=0, native_score=0, episode_hash=digest(t)) for t in previous]
        result = resolve_family_panels(batch, previous, ledger, manifest['seed'], protocol=manifest['protocol'])
        assert result['panels'] == batch
        assert all(result['sampling'][gate]['counts']['fresh'] == 50 for gate in ('primary','meta'))


@pytest.mark.parametrize("holdout_size", [None, 30, 50])
def test_six_round_registration_and_independent_holdouts(monkeypatch, tmp_path, holdout_size):
    plan = panel(monkeypatch, tmp_path, 610, protocol=COEVOLUTION_6X30_PROTOCOL, holdout_size=holdout_size)
    assert plan['holdout_size'] == (holdout_size or 30)
    assert len(plan['final']) == len(plan['selection']) == plan['holdout_size']
    assert len(plan['batches']) == plan['requested_rounds'] == plan['rounds'] == 6
    assert all(len(tasks) == 30 for batch in plan['batches'] for tasks in batch.values())
    assert plan['budget']['per_round']['episodes'] == 390
    last = plan['batches'][-1]
    learning = [t for batch in plan['batches'] for t in batch['learning']]
    ledger = [dict(task_id=t['id'], family=t['family'], phase='learning', status='completed',
                   score=1, native_score=1, episode_hash=digest(t)) for t in learning]
    assert resolve_family_panels(last, learning, ledger, plan['seed'], protocol=plan['protocol'])['panels'] == last
    bad = deepcopy(plan)
    bad['holdout_size'] = 50 if plan['holdout_size'] == 30 else 30
    with pytest.raises(ValueError, match='holdout panel'):
        cp.validate_coevolution_manifest(bad)


def test_legacy_holdout_rule_and_insufficient_new_capacity(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Holdout size"):
        panel(monkeypatch, tmp_path, 610, protocol=COEVOLUTION_PROTOCOL, holdout_size=30)
    plan = panel(monkeypatch, tmp_path, 359, protocol=COEVOLUTION_6X30_PROTOCOL)
    assert plan['rounds'] == 3


def test_independent_final_size_preserves_arm_selection(monkeypatch, tmp_path):
    plan = panel(monkeypatch, tmp_path, 395, protocol=COEVOLUTION_6X30_PROTOCOL, final_size=50)
    assert len(plan['selection']) == plan['holdout_size'] == 30
    assert len(plan['final']) == plan['final_size'] == 50
    assert all(len(v) == 30 for b in plan['batches'] for v in b.values())


def test_heat_uses_five_23_game_rounds_and_separate_30_50_holdouts(monkeypatch, tmp_path):
    plan = panel(monkeypatch, tmp_path, 354, protocol=COEVOLUTION_HEAT_5X23_PROTOCOL)
    assert plan['rounds'] == plan['requested_rounds'] == len(plan['batches']) == 5
    assert plan['holdout_size'] == len(plan['selection']) == 30
    assert plan['final_size'] == len(plan['final']) == 50
    assert plan['budget']['per_round']['episodes'] == 299
    assert all(len(tasks) == 23 for b in plan['batches'] for tasks in b.values())
    previous = [t for b in plan['batches'] for t in b['learning']]
    ledger = [dict(task_id=t['id'], family='heat', phase='learning', status='completed',
                   score=1, native_score=1, episode_hash=digest(t)) for t in previous]
    batch = plan['batches'][-1]
    assert resolve_family_panels(batch, previous, ledger, plan['seed'], protocol=plan['protocol'])['panels'] == batch
    with pytest.raises(ValueError, match='specific to heat'):
        cp.validate_coevolution_manifest(dict(plan, family='two'))
    with pytest.raises(ValueError, match='30 selection games'):
        panel(monkeypatch, tmp_path, 354, protocol=COEVOLUTION_HEAT_5X23_PROTOCOL, final_size=30)
