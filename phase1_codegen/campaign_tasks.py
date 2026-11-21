"""Freeze metadata-selected ALFWorld tasks; no model calls or episode-result reads.

This is the approved two-dataset campaign's ALFWorld task constructor.  It reads
only official archives and the existing task manifests/room split, then writes
five new manifests and their audit.  It does not stage or execute an environment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / 'evomemory/downloads/alfworld-tw-0.4.2.zip'
METADATA = ROOT / 'runs/feasibility_v1/alma_audit/downloads/json_2.1.1_json.zip'
OLD_MANIFESTS = ROOT / 'data/manifests'
ROOM_AUDIT = ROOT / 'data/audit.json'
SELECTION_SEED = 20261106
COUNTS = dict(source_adaptation=16, source_exploration=8,
              target_adaptation=16, target_exploration=12,
              target_generalization=16)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def round_robin(rows, rooms, count, *, hash_key, distinct_rooms=False):
    """Balance rooms before ranking groups by fixed structural metadata."""
    buckets = defaultdict(list)
    for row in rows:
        if row['room'] in rooms:
            buckets[row['room']].append(row)
    order = sorted(buckets, key=hash_key)
    for room in order:
        buckets[room].sort(key=lambda row: (-row['high_plan_length'],
                                           -row['scene_object_count'],
                                           hash_key(row['group_id'])))
    selected = []
    depth = 0
    while len(selected) < count:
        available = [buckets[room][depth] for room in order if len(buckets[room]) > depth]
        if not available or (distinct_rooms and depth):
            raise ValueError('Insufficient metadata-selected groups/rooms for the frozen role')
        selected.extend(available[:count - len(selected)])
        depth += 1
    return selected




def read_metadata(archive, game_member):
    return json.loads(archive.read(str(Path(game_member).with_name('traj_data.json'))))


def build_alfworld_cross_type(out, target_reference=None):
    """New heat -> two protocol; the prior campaign and target split stay intact."""
    out = Path(out).resolve()
    reference = Path(target_reference or ROOT / 'runs/two_dataset_mechanism_20261106/datasets/alfworld').resolve()
    if out == reference:
        raise ValueError('The cross-type campaign requires a new output directory')
    previous_audit = read_json(reference / 'audit.json')
    old_audit = read_json(ROOM_AUDIT)
    archive_hash = sha256(ARCHIVE)
    if archive_hash != old_audit['archive_sha256']:
        raise ValueError('Official game archive changed')
    excluded, exclusion_hashes, reference_hashes = set(), {}, {}
    for path in sorted(OLD_MANIFESTS.glob('*.json')):
        name = str(path.relative_to(ROOT / 'data'))
        actual = sha256(path)
        if old_audit['manifest_hashes'].get(name) != actual:
            raise ValueError('Historical manifest changed: ' + name)
        exclusion_hashes[str(path)] = actual
        excluded.update(t['group_id'] for t in read_json(path)
                        if t['archive_member'].startswith('json_2.1.1/train/'))
    reference_tasks = {}
    for role in COUNTS:
        path = reference / (role + '.json')
        actual = sha256(path)
        if previous_audit['manifest_sha256'].get(path.name) != actual:
            raise ValueError('Prior campaign task manifest changed: ' + path.name)
        reference_hashes[path.name] = actual
        reference_tasks[role] = read_json(path)
        excluded.update(t['group_id'] for t in reference_tasks[role]
                        if t['archive_member'].startswith('json_2.1.1/train/'))
    target_roles = ('target_adaptation', 'target_exploration', 'target_generalization')
    targets = [t for role in target_roles for t in reference_tasks[role]]
    if any(len(reference_tasks[role]) != COUNTS[role] for role in target_roles) or any(t['family'] != 'two' for t in targets):
        raise ValueError('The approved target split is exactly two: 16 adaptation, 12 exploration, 16 generalization')
    target_rooms = {t['room'] for t in targets}
    family = 'pick_heat_then_place_in_recep'
    prefix = 'codegen_crosstype_v1|20261106|'
    def order_key(value):
        return hashlib.sha256((prefix + str(value)).encode()).hexdigest()
    rows = []
    with zipfile.ZipFile(ARCHIVE) as games, zipfile.ZipFile(METADATA) as metadata:
        canonical = {}
        for name in sorted(games.namelist()):
            if name.startswith('json_2.1.1/train/') and name.endswith('/game.tw-pddl'):
                canonical.setdefault(name.split('/')[-3], name)
        for group, name in canonical.items():
            room = int(group.rsplit('-', 1)[1])
            if not group.startswith(family + '-') or group in excluded or room in target_rooms:
                continue
            record = read_metadata(metadata, name)
            if record['task_type'] != family:
                raise ValueError('Official heat game and metadata disagree')
            rows.append(dict(group_id=group, room=room, archive_member=name,
                             game_sha256=hashlib.sha256(games.read(name)).hexdigest(),
                             high_plan_length=len(record['plan']['high_pddl']),
                             scene_object_count=len(record['scene']['object_poses'])))
        for task in targets:
            if hashlib.sha256(games.read(task['archive_member'])).hexdigest() != task['game_sha256']:
                raise ValueError('Preserved target game archive hash changed')
    rooms = sorted({row['room'] for row in rows}, key=order_key)
    midpoint = len(rooms) // 2
    source_rooms = dict(source_adaptation=rooms[:midpoint], source_exploration=rooms[midpoint:])
    selected = {role: round_robin(rows, set(role_rooms), COUNTS[role], hash_key=order_key)
                for role, role_rooms in source_rooms.items()}
    manifests = {role: reference_tasks[role] for role in target_roles}
    for role, picked in selected.items():
        manifests[role] = [dict(id='alfworld:heat:' + role + ':' + order_key(row['group_id'])[:16],
            group_id=row['group_id'], room=row['room'], family='heat',
            split='adaptation' if role == 'source_adaptation' else 'exploration',
            goal='Complete the household task stated in the initial observation.', seed=42,
            game_sha256=row['game_sha256'], archive_member=row['archive_member']) for row in picked]
    seen = {field: set() for field in ('id', 'group_id', 'room', 'game_sha256')}
    for role, tasks in manifests.items():
        if len(tasks) != COUNTS[role]:
            raise ValueError('Frozen cross-type role size changed')
        for field in seen:
            values = {t[field] for t in tasks}
            if values & seen[field] or (field != 'room' and len(values) != len(tasks)):
                raise ValueError('Cross-type roles overlap in ' + field)
            seen[field].update(values)
    for role in source_rooms:
        if any(t['group_id'] in excluded for t in manifests[role]):
            raise ValueError('A previously selected task entered the new source')
    # Preserve all three target files byte-for-byte, including their IDs/seeds.
    files = {role + '.json': (reference / (role + '.json')).read_text() for role in target_roles}
    files.update({role + '.json': encode(manifests[role]) for role in source_rooms})
    historical_reserved = set(old_audit['room_assignment']['final'] + old_audit['room_assignment']['fresh'])
    used_source_rooms = {t['room'] for role in source_rooms for t in manifests[role]}
    audit = dict(schema=2, protocol='within_environment_cross_type', environment='alfworld',
        source_environment='alfworld', target_environment='alfworld', source_family='heat', target_family='two',
        native_execution=False, model_calls=0, selection_seed=SELECTION_SEED, environment_seed=42, proposed_evolver_seed=11,
        sources=dict(game_archive=dict(path=str(ARCHIVE), sha256=archive_hash),
                     metadata_archive=dict(path=str(METADATA), sha256=sha256(METADATA)),
                     generator=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
                     old_manifest_hashes=exclusion_hashes,
                     target_reference=dict(path=str(reference), manifest_sha256=reference_hashes)),
        exclusion=dict(rule='Source excludes all historical data/manifests train groups, all prior campaign selected groups, and every current target room; no outcome-based selection',
                       excluded_groups=len(excluded), source_available_groups=len(rows), source_available_rooms=len(rooms),
                       target_rooms_excluded=sorted(target_rooms)),
        historical_room_boundary=dict(
            prior_final_fresh_rooms_used_by_source=sorted(used_source_rooms & historical_reserved),
            note='Some new heat tasks use rooms reserved as final/fresh in an older experiment. Their task groups were never in its manifests. All current source/target rooms remain disjoint; historical room novelty is not claimed.'),
        ranking=dict(canonical_trial='Lexicographically first official train game per group',
                     hash="sha256(('codegen_crosstype_v1|20261106|' + str(value)).encode()).hexdigest()",
                     room_partition='Sort available heat rooms by the frozen hash; first floor(n/2) adaptation, remainder exploration',
                     group_order='Round-robin rooms; within a room descending high_pddl length, descending scene object_poses count, then ascending group hash',
                     difficulty='Structural metadata only; never model success/failure. Reference actions are not exported.'),
        role_capacity={role: dict(groups=sum(row['room'] in set(partition) for row in rows), rooms=len(partition),
                                  room_ids=sorted(partition)) for role, partition in source_rooms.items()},
        selected_counts={role: len(batch) for role, batch in manifests.items()},
        selected_room_ids={role: sorted({t['room'] for t in batch}) for role, batch in manifests.items()},
        selected_source_metadata={role: [dict(task_id=task['id'], group_id=row['group_id'], room=row['room'],
                                            high_plan_length=row['high_plan_length'], scene_object_count=row['scene_object_count'])
                                         for task, row in zip(manifests[role], picked)] for role, picked in selected.items()},
        manifest_sha256={name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()},
        search_protocol=dict(source_rounds=3, target_rounds_per_condition=3,
                             target_conditions=['NoExternalEM', 'CrossTypeEM'],
                             acceptance_policy='reviewer_and_reward',
                             experience_transfer='ALFWorld heat code-change trials -> ALFWorld two only. TextWorld has its own separate source, target and Evolution Memory.',
                             target_matching='Identical preserved target tasks, initial source, own adaptation logs, seed and proposal budget',
                             final_access='Target generalization remains sealed from Reviewer/Evolver and is evaluated only after target versions are fixed'),
        checks=dict(group_disjoint=True, room_disjoint=True, canonical_byte_disjoint=True,
                    source_old_task_groups_excluded=True, target_files_preserved_byte_for_byte=True,
                    same_environment=True, different_task_types=True, no_cross_environment_memory=True,
                    reference_actions_exported=False, model_outcomes_read=False))
    files['audit.json'] = encode(audit)
    for name, content in files.items():
        if (out / name).exists() and (out / name).read_text() != content:
            raise ValueError('Existing cross-type split differs: ' + name)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if not (out / name).exists():
            with (out / name).open('x') as stream:
                stream.write(content)
    return audit


def build_textworld(out):
    from .textworld_tasks import build_manifest
    from evomem.common import digest
    inventory = build_manifest()
    def ordered(category, split):
        values = [t for t in inventory['tasks'] if t['family'] == category and t['official_split'] == split]
        return sorted(values, key=lambda t: digest(['codegen_crossenv_v1', 20261106, t['id']]))
    source = ordered('the_cooking_game', 'train')
    target = ordered('treasure_hunter', 'train')
    heldout = ordered('treasure_hunter', 'eval_in_distribution')
    if (len(source), len(target), len(heldout)) != (12, 12, 13):
        raise ValueError('Official TextWorld inventory size changed')
    selected = dict(source_adaptation=source[:6], source_exploration=source[6:],
                    target_adaptation=target[:6], target_exploration=target[6:],
                    target_generalization=heldout)
    ids = [t['id'] for group in selected.values() for t in group]
    games = [t['game_sha256'] for group in selected.values() for t in group]
    if len(set(ids)) != len(ids) or len(set(games)) != len(games):
        raise ValueError('TextWorld roles reuse a compiled game')
    files = {role + '.json': encode(batch) for role, batch in selected.items()}
    audit = dict(dataset=inventory['dataset'], official_alma_commit=inventory['official_alma_commit'],
                 archive_sha256=inventory['archive_sha256'], official_inventory_counts=inventory['counts'],
                 official_split_policy=inventory['split_policy'],
                 role_selection='sha256(canonical JSON [codegen_crossenv_v1,20261106,task_id]); first 6 adaptation, remaining 6 exploration within each official training category',
                 source_family='the_cooking_game', target_family='treasure_hunter',
                 selected_counts={key: len(value) for key, value in selected.items()},
                 manifest_sha256={key: hashlib.sha256(value.encode()).hexdigest() for key, value in files.items()},
                 protocol=dict(kind='within_environment_cross_type', source_rounds=3, target_rounds_per_condition=3,
                               source_environment='textworld', target_environment='textworld',
                               target_conditions=['NoExternalEM', 'CrossTypeEM'],
                               final_versions=['initial', 'NoExternalEM', 'CrossTypeEM'],
                               acceptance_policy='reviewer_and_reward',
                               experience_transfer='TextWorld the_cooking_game -> TextWorld treasure_hunter only; ALFWorld Evolution Memory is never imported'),
                 limitation='Official dataset has only 24 training games across the two ALMA-supported categories; use all without moving evaluation games into search',
                 checks=dict(group_disjoint=True, canonical_byte_disjoint=True, official_train_only_search=True,
                             reference_actions_exported=False, model_outcomes_read=False))
    files['audit.json'] = encode(audit)
    out = Path(out)
    for name, content in files.items():
        if (out / name).exists() and (out / name).read_text() != content:
            raise ValueError('Existing frozen TextWorld split differs')
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if not (out / name).exists():
            (out / name).write_text(content)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--environment', choices=('alfworld', 'textworld'), default='alfworld')
    parser.add_argument('--target-reference', type=Path, help='Existing ALFWorld target split to preserve byte-for-byte')
    args = parser.parse_args(argv)
    if args.environment == 'textworld':
        audit = build_textworld(args.out)
    else:
        audit = build_alfworld_cross_type(args.out, args.target_reference)
    print(encode(dict(out=str(args.out.resolve()), counts=audit['selected_counts'],
                      capacity=audit.get('role_capacity'), checks=audit['checks'])))


if __name__ == '__main__':
    main()
