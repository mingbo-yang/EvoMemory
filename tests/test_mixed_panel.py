"""Synthetic mixed registration and whole-game/gate isolation without native inference."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest

from evomem.common import digest
from rsi_paired import mixed_panel
from rsi_paired.coevolution_panel import validate_coevolution_manifest
from rsi_paired.panel import COEVOLUTION_MIXED_10X60_PROTOCOL, FAMILIES, MIXED_FAMILIES, resolve_family_panels


def synthetic_mixed_manifest(tmp_path, monkeypatch, seed=42):
    archive = tmp_path / 'mixed-synthetic.zip'
    with ZipFile(archive,'w'):
        pass
    labels = {alias: official for official, alias in FAMILIES.items()}
    pools = {}
    for family in MIXED_FAMILIES:
        pools[family] = {}
        for split, size in [('train',310),('valid_unseen',18),('valid_seen',10)]:
            rows = []
            for i in range(size):
                layout = 'shared' if split in ('train','valid_seen') else split
                group = f'{labels[family]}-synthetic-{layout}-{i}'
                member = f'json_2.1.1/{split}/{group}/synthetic/game.tw-pddl'
                rows.append((group, member, digest(member)))
            pools[family][split] = rows
    def task(member, sha, family, metadata, seed):
        _, split, group, _, _ = member.split('/')
        return dict(id='rsi-alfworld/' + digest(member)[:18], family=family, split=split, group_id=group,
                    game_sha256=sha, room=int(group.rsplit('-',1)[-1]), archive_member=member, environment='alfworld',
                    seed=int(digest([seed,'environment',member])[:8],16) % 2147483647)
    monkeypatch.setattr(mixed_panel, '_catalogue', lambda *args: pools)
    monkeypatch.setattr(mixed_panel, '_task', task)
    monkeypatch.setattr(mixed_panel, 'file_hash', lambda path: mixed_panel.GAME_SHA)
    return mixed_panel.mixed_manifest(archive,archive,seed)


def test_native_splits_balanced_families_and_resolved_history(tmp_path, monkeypatch):
    plan = synthetic_mixed_manifest(tmp_path,monkeypatch)
    assert plan == synthetic_mixed_manifest(tmp_path,monkeypatch)
    assert len(plan['batches']) == 10
    assert len(plan['selection']) == 30 and len(plan['final']) == 100
    counter = Counter(r['family'] for r in plan['final'])
    assert sorted(counter.values()) == [16,16,17,17,17,17]
    previous = []
    for batch in plan['batches']:
        for tasks in batch.values():
            assert len(tasks) == 60 and Counter(t['family'] for t in tasks) == dict.fromkeys(MIXED_FAMILIES,10)
            assert all({t['family'] for t in tasks[i:i+6]} == set(MIXED_FAMILIES) for i in range(0,60,6))
        previous.extend(batch['learning'])
    ledger = [dict(task_id=t['id'], family=t['family'], phase='learning', status='completed', score=0,
                   native_score=0,episode_hash=digest(t)) for t in previous]
    resolved = resolve_family_panels(batch, previous, ledger, 42, protocol=COEVOLUTION_MIXED_10X60_PROTOCOL)
    assert resolved['panels'] == batch and resolved['sampling']['meta']['counts']['fresh'] == 60
    assert plan['scope'] == 'alfworld_coevolution_mixed_families'
    validate_coevolution_manifest(plan)


@pytest.mark.parametrize('defect', ['quota', 'duplicate', 'test_exposure', 'order'])
def test_mixed_rejects_corrupted_manifest(tmp_path, monkeypatch, defect):
    plan = synthetic_mixed_manifest(tmp_path,monkeypatch)
    if defect == 'quota':
        plan['batches'][0]['learning'][0]['family'] = 'invalid'
    elif defect == 'duplicate':
        first = plan['batches'][0]['learning'][0]
        other = plan['batches'][1]['learning']
        other[next(i for i,t in enumerate(other) if t['family']==first['family'])] = first
    elif defect == 'test_exposure':
        first = plan['final'][0]
        other = plan['selection']
        other[next(i for i,t in enumerate(other) if t['family']==first['family'])] = first
    else:
        rows = plan['batches'][0]['learning']
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError):
        validate_coevolution_manifest(plan)
