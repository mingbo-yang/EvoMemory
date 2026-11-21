"""Frozen ALMA/BALROG TextWorld assets and the exact upstream 50/50 split.

No game generation, oracle walkthrough loading, or outcome-based selection. The
JSON and Inform source sidecars are hashed for provenance, never given to agents.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import zipfile

from evomem.common import ROOT, digest, file_hash
from .policy import Rejected


GAMES = ROOT / 'data/textworld_balrog/tw_games'
ARCHIVE = ROOT / 'data/textworld_balrog/downloads/tw_games.zip'
ARCHIVE_SHA256 = 'c0e8841649b8dde9607bc5326e89a265a975e7859ddb432d53c1c17ddafdcbb9'
CATEGORIES = ('treasure_hunter', 'the_cooking_game')
OFFICIAL_COMMIT = '0a6ecb1f8ffbc704e15ac2c98a6973ea60caf70b'
ASSET_SEED_POLICY = ('Task manifest compiled_game_seed is the number in the asset filename. '
                     'Episode asset_seeds.settings_seed is metadata.settings.seed (null if absent); '
                     'asset_seeds.components contains the four recorded metadata.seeds values. '
                     'The compiled game is loaded by hash and is never regenerated.')


def build_manifest():
    if not ARCHIVE.is_file() or file_hash(ARCHIVE) != ARCHIVE_SHA256:
        raise Rejected('official BALROG TextWorld archive is missing or changed')
    categories = defaultdict(list)
    # Match Textworld_Env.__init__ enumeration and shared random.Random(42).
    for pattern in ('*.ulx', '*.z8'):
        for path in sorted(GAMES.rglob(pattern)):
            if path.parent.name in CATEGORIES:
                categories[path.parent.name].append(path)
    if {key: len(value) for key, value in categories.items()} != {
            'treasure_hunter': 25, 'the_cooking_game': 25}:
        raise Rejected('expected the full official 25 + 25 TextWorld game inventory')
    rng = random.Random(42)
    # The unchanged archive is the authority for the extracted assets too.
    # Merely hashing extracted files would silently bless pre-freeze edits.
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest()
                          for name in archive.namelist()
                          if not name.endswith('/') and name.split('/')[1] in CATEGORIES}
    tasks = []
    for category, paths in categories.items():
        rng.shuffle(paths)
        for index, path in enumerate(paths):
            sidecars = {suffix: file_hash(path.with_suffix(suffix)) for suffix in ('.json', '.ni')}
            game_hash = file_hash(path)
            for member, checksum in [(path, game_hash)] + [
                    (path.with_suffix(suffix), value) for suffix, value in sidecars.items()]:
                if archive_hashes.get('tw_games/' + str(member.relative_to(GAMES))) != checksum:
                    raise Rejected('extracted TextWorld asset differs from the frozen official archive')
            tasks.append(dict(
                id=f'textworld/{category}/{path.stem}',
                group_id=f'textworld/{category}/{path.stem}',
                room=path.stem, family=category, dataset='textworld',
                official_split='train' if index < len(paths) // 2 else 'eval_in_distribution',
                official_split_index=index, seed=42,
                # Historical task-schema key: an asset filename label, not a generator input.
                compiled_game_seed=int(re.search(r'seed_(\d+)', path.stem).group(1)),
                game_path=str(path.resolve()), game_sha256=game_hash,
                sidecar_sha256=sidecars, archive_member=str(path.relative_to(GAMES)),
                goal='',
            ))
    counts = {category: {split: sum(task['family'] == category and task['official_split'] == split
                                  for task in tasks)
                         for split in ('train', 'eval_in_distribution')}
              for category in CATEGORIES}
    return dict(
        dataset='ALMA official BALROG TextWorld compiled games',
        official_alma_commit=OFFICIAL_COMMIT, archive_sha256=ARCHIVE_SHA256,
        archive_bytes=ARCHIVE.stat().st_size, total_games=50,
        excluded_category='coin_collector (excluded by official ALMA adapter)',
        source='https://github.com/zksha/alma/blob/' + OFFICIAL_COMMIT
               + '/envs_docker/BALROG/balrog/environments/textworld/README.md',
        split_policy='official sorted *.ulx then *.z8; shared Random(42); first floor(n/2) train',
        asset_seed_policy=ASSET_SEED_POLICY,
        counts=counts, tasks=tasks, task_manifest_hash=digest(tasks),
    )


def load_tasks(category, split='train'):
    if category not in CATEGORIES or split not in ('train', 'eval_in_distribution'):
        raise Rejected('unknown official TextWorld category or split')
    return [task for task in build_manifest()['tasks']
            if task['family'] == category and task['official_split'] == split]


def verify_task_assets(task):
    """Check exact files before every episode; never inspect solution contents."""
    path = Path(task['game_path']).resolve()
    expected = (GAMES / task['archive_member']).resolve()
    if path != expected or not path.is_relative_to(GAMES.resolve()):
        raise Rejected('TextWorld game path escaped the official inventory')
    if file_hash(path) != task['game_sha256']:
        raise Rejected('TextWorld compiled game changed after task freeze')
    if set(task['sidecar_sha256']) != {'.json', '.ni'}:
        raise Rejected('TextWorld sidecar provenance is incomplete')
    for suffix, checksum in task['sidecar_sha256'].items():
        if file_hash(path.with_suffix(suffix)) != checksum:
            raise Rejected('TextWorld game sidecar changed after task freeze')
    return path


def asset_seed_receipt(task, verified_game):
    """Project only seed metadata from the authenticated sidecar of the selected game."""
    payload = verified_game.with_suffix('.json').read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != task['sidecar_sha256']['.json']:
        raise Rejected('TextWorld JSON sidecar changed before seed provenance was read')
    # Project in SQLite so walkthroughs, recipes and world state are never
    # materialized as Python objects or included in provenance.
    with closing(sqlite3.connect(':memory:')) as db:
        component_json, seed_type, settings_seed = db.execute(
            """SELECT json_extract(?1, '$.metadata.seeds'),
                      json_type(?1, '$.metadata.settings.seed'),
                      json_extract(?1, '$.metadata.settings.seed')""",
            (payload.decode('utf-8'),),
        ).fetchone()
    components = json.loads(component_json) if isinstance(component_json, str) else None
    if (not isinstance(components, dict) or set(components) != {'map', 'objects', 'quest', 'grammar'}
            or any(type(value) is not int for value in components.values())
            or seed_type not in (None, 'null', 'integer')):
        raise Rejected('invalid TextWorld generation seed metadata')
    return dict(filename_seed=task['compiled_game_seed'], settings_seed=settings_seed,
                components=components, sidecar_sha256=checksum)
