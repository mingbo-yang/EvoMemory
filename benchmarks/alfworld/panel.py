"""Load the public CrossEp-Emb test panel, preserving the original ALFWorld asset split."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from evomem.common import ROOT, DATA, LEGACY, file_hash, read
import os

AUDIT = DATA / 'alfworld_panel'
METADATA = Path(os.environ.get('EVOMEMORY_ALFWORLD_METADATA', str(LEGACY / 'downloads/json_2.1.1_json.zip')))
FAMILIES = {'pick_cool_then_place_in_recep': 'cool', 'pick_two_obj_and_place': 'two',
            'pick_clean_then_place_in_recep': 'clean', 'pick_and_place_simple': 'simple',
            'pick_heat_then_place_in_recep': 'heat', 'look_at_obj_in_light': 'light'}


def load_panel():
    manifest = read(AUDIT / 'crossep_emb_manifest_audit.json')
    if (manifest['source_sha256'] != 'fd24a8525a4c0464fd1fae130763cd736d326d6a687dcb7d48bda40d93e83c00'
            or file_hash(manifest['archive']) != manifest['archive_sha256']):
        raise ValueError('Official games or public mapping identity mismatch')
    tasks = []
    with ZipFile(manifest['archive']) as games, ZipFile(METADATA) as metadata:
        for row in manifest['items']:
            member = Path(row['member'])
            if (len(member.parts) != 5 or member.parts[:2] != ('json_2.1.1', 'valid_train')
                    or member.parts[2] != row['group'] or member.name != 'game.tw-pddl'
                    or row['family'] != row['group'].split('-', 1)[0]):
                raise ValueError('Bad asset provenance for item ' + str(row['item_id']))
            if hashlib.sha256(games.read(str(member))).hexdigest() != row['game_sha256']:
                raise ValueError('Game content differs from manifest')
            trajectory = json.loads(metadata.read(str(member.with_name('traj_data.json'))))
            if trajectory['task_type'] != row['family']:
                raise ValueError('Task type differs from matched official trajectory')
            tasks.append(dict(id=f"crossep-emb/{row['item_id']}", environment='alfworld',
                              family=FAMILIES[row['family']], group_id=row['group'],
                              room=int(row['group'].rsplit('-', 1)[-1]), seed=42,
                              goal='Complete the household task stated in the initial observation.',
                              archive_member=str(member), game_sha256=row['game_sha256'],
                              split='valid_train', benchmark_split='test', benchmark_item_id=row['item_id']))
    if (len(tasks) != 200 or len({t['id'] for t in tasks}) != 200
            or len({t['archive_member'] for t in tasks}) != 200):
        raise ValueError('Expected 200 distinct official panel instances')
    return tasks
