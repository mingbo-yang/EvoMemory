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


def test_formal_panel_balances_roles_and_excludes_pilot(tmp_path, monkeypatch):
    from collections import Counter
    from rsi_paired.campaign import _campaign_tasks

    games, meta = tmp_path / 'formal-games.zip', tmp_path / 'formal-meta.zip'
    with ZipFile(games, 'w') as archive, ZipFile(meta, 'w') as metadata:
        for kind in panel.FAMILIES:
            for split, count in [('train', 24), ('valid_unseen', 19), ('valid_seen', 4)]:
                for n in range(count):
                    room = n if split == 'train' else 100 + n
                    group = f'{kind}-fixture-{room}'
                    entry = Path('json_2.1.1') / split / group / 'trial' / 'game.tw-pddl'
                    archive.writestr(str(entry), str(entry))
                    metadata.writestr(str(entry.with_name('traj_data.json')), json.dumps({'task_type': kind}))
    monkeypatch.setattr(panel, 'GAME_SHA', file_hash(games))
    excluded = {'pick_heat_then_place_in_recep-fixture-0'}
    manifest = panel.formal_manifest(games, meta, 42, excluded)
    development = _campaign_tasks(manifest)
    assert len(development) == len(manifest['final']) == 120
    assert Counter(x['split'] for x in development) == {'train': 120}
    assert manifest['final_by_official_split'] == {'valid_unseen': 114, 'valid_seen': 6}
    assert excluded.isdisjoint(t['group_id'] for t in development)
    assert len({t['game_sha256'] for t in development + manifest['final']}) == 240
    assert manifest == panel.formal_manifest(games, meta, 42, excluded)
    manifest['final'][0] = development[0]
    with pytest.raises(ValueError):
        _campaign_tasks(manifest)


def test_formal_confirmation_consumes_every_reserved_game(tmp_path, monkeypatch):
    from rsi_paired import campaign

    def games(prefix):
        return [{"id": f"{prefix}-{i}", "family": "heat"} for i in range(2)]

    batches = [{phase: games(f"{phase}-{i}") for phase in ("learning", "primary", "meta")}
               for i in range(4)]
    core = games("confirm")
    manifest = {"batches": batches, "confirmation": core}
    all_dev = [*core, *(t for b in batches for panel in b.values() for t in panel)]
    monkeypatch.setattr(campaign, "_campaign_tasks", lambda _: all_dev)
    monkeypatch.setattr(campaign, "_checked_state", lambda stage: dict(source=stage.name))

    def write(relative, value):
        panel.save_registration(tmp_path / relative, value)

    def scores(tasks):
        return [{"task_id": task["id"], "score": 0} for task in tasks]

    # The second and fourth batches scored their paired panels. Success values
    # play no role in whether the other reserved instances join confirmation.
    for i, batch in enumerate(batches):
        stage = f"evolution-{i+1:02d}/batch"
        for j, row in enumerate(scores(batch["learning"])):
            write(f"{stage}/jobs/learning/{j:03d}/result.json", {"outcome": {"value": [None, row]}})
        write(f"{stage}/primary_results.json", {"parent": {"evaluation": scores(batch["primary"])}})
        if i in (1, 3):
            write(f"{stage}/meta_results.json", {"parent": {"evaluation": scores(batch["meta"])}})

    tasks, states = campaign._confirmation_input(tmp_path, manifest)
    assert tasks == core + batches[0]["meta"] + batches[2]["meta"]
    assert len(states) == 5
    with pytest.raises(ValueError, match="do not cover"):
        campaign._covered_development(tmp_path, manifest, {"results": [{"episodes": scores(tasks[:-1])}]})
    summary = {"results": [{"episodes": scores(tasks)}]}
    receipt = campaign._covered_development(tmp_path, manifest, summary)
    assert receipt["distinct_tasks"] == len(all_dev)
    assert receipt["confirmation_hash"] == campaign.digest(summary)
