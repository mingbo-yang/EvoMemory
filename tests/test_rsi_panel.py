import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from evomem.common import file_hash
from rsi_paired import panel


def test_smoke_uses_distinct_official_training_groups_and_games(tmp_path, monkeypatch):
    games, meta = tmp_path / 'games.zip', tmp_path / 'meta.zip'
    with ZipFile(games, 'w') as archive, ZipFile(meta, 'w') as metadata:
        for kind in panel.FAMILIES:
            for room in range(6):
                group = f'{kind}-fixture-{room}'
                entry = Path('json_2.1.1/train') / group / 'trial' / 'game.tw-pddl'
                archive.writestr(str(entry), group)
                metadata.writestr(str(entry.with_name('traj_data.json')), json.dumps({'task_type': kind}))
                if room == 0:
                    archive.writestr(str(entry).replace('/train/', '/valid_seen/'), group)
    monkeypatch.setattr(panel, 'GAME_SHA', file_hash(games))
    manifest = panel.smoke_manifest(games, meta, 42)
    tasks = [task for group in manifest['panels'].values() for task in group]
    assert len({task['group_id'] for task in tasks}) == len({task['game_sha256'] for task in tasks}) == 6
    assert all(task['room'] != 0 for task in tasks)
    assert manifest == panel.smoke_manifest(games, meta, 42)
    omitted = panel.smoke_manifest(games, meta, 42, {tasks[0]['group_id']})
    assert all(t['group_id'] != tasks[0]['group_id'] for rows in omitted['panels'].values() for t in rows)


def test_manifest_is_written_once(tmp_path):
    path = tmp_path / 'panel.json'
    panel.save_registration(path, {'seed': 1})
    panel.save_registration(path, {'seed': 1})
    with pytest.raises(ValueError, match='Frozen registration'):
        panel.save_registration(path, {'seed': 2})
    assert json.loads(path.read_text()) == {'seed': 1}
