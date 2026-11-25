"""Preregister a single multi-step ALFWorld family at its registered batch width."""
from pathlib import Path
import re
from zipfile import ZipFile

from benchmarks.alfworld.panel import FAMILIES
from evomem.common import digest, file_hash
from scripts.prepare_assets import GAME_SHA
from .panel import (COEVOLUTION_HEAT_5X23_PROTOCOL, COEVOLUTION_MIXED_10X60_PROTOCOL, COEVOLUTION_PROTOCOL,
                    coevolution_shape, GATES, _catalogue, _check_fresh_groups,
                    _holdout_groups, _partition_fresh, _spread_groups, _task)

HARD_FAMILIES = ('heat', 'cool', 'clean', 'two')
SCOPE = 'alfworld_coevolution_hard_family'


def exclusion_rows(document, final_only):
    """Accept v2/v3 formal plans; read task records only, never experimental outcomes."""
    if 'family_order' in document:
        plans = document['families'].values()
    else:
        plans = [document]
    rows = []
    for plan in plans:
        final = plan.get('final', plan.get('test'))
        if not isinstance(final, list):
            raise ValueError('Exposure exclusions require a registered final/test panel')
        rows.extend(final)
        if not final_only:
            rows.extend(plan.get('confirmation', []))
            for batch in plan['batches']:
                for tasks in batch.values():
                    if isinstance(tasks, list):
                        rows.extend(tasks)
    if any(not isinstance(row, dict) or 'group_id' not in row or 'game_sha256' not in row for row in rows):
        raise ValueError('Malformed exposure-exclusion task')
    return rows


def _holdout(train, official, blocked_groups, seed, tag, size):
    """Reserve whole groups. Prefer official validation games, then isolated train groups."""
    eligible = [row for row in official if row[0] not in blocked_groups]
    chosen = eligible[:size]
    training = [row for row in train if row[0] not in blocked_groups]
    missing = size - len(chosen)
    groups = _holdout_groups(training, missing, 0, seed, tag)
    chosen.extend(_spread_groups([row for row in training if row[0] in groups])[:missing])
    if len(chosen) != size:
        raise ValueError(f'{tag} needs a complete {size}-game holdout')
    return chosen


def _holdout_sizes(protocol, selection, final):
    default = 50 if protocol == COEVOLUTION_PROTOCOL else 30
    selection = default if selection is None else selection
    if final is None:
        final = 50 if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL else selection
    if type(selection) is not int or type(final) is not int:
        raise ValueError('Holdout sizes must be integers')
    if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL and (selection, final) != (30, 50):
        raise ValueError('The heat 5x23 registration requires 30 selection games and 50 final games')
    allowed = (50,) if protocol == COEVOLUTION_PROTOCOL else (30, 50)
    if selection not in allowed or final not in allowed:
        raise ValueError(f'Holdout sizes for {protocol} must be in {allowed}')
    return selection, final


def coevolution_manifest(games_path, metadata_path, seed, family='heat', excluded=(), holdout_excluded=(),
                         *, protocol=COEVOLUTION_PROTOCOL, holdout_size=None, final_size=None):
    if protocol == COEVOLUTION_MIXED_10X60_PROTOCOL:
        raise ValueError("Use mixed_manifest to register the six-family treatment")
    max_rounds, size = coevolution_shape(protocol)
    holdout_size, final_size = _holdout_sizes(protocol, holdout_size, final_size)
    if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL and family != 'heat':
        raise ValueError('The 5x23 registration is specific to heat')
    if family not in HARD_FAMILIES:
        raise ValueError('Choose a multi-step ALFWorld family')
    if file_hash(games_path) != GAME_SHA:
        raise ValueError('Game archive differs from the pinned ALFWorld release')
    groups = {task['group_id'] for task in excluded}
    hashes = {task['game_sha256'] for task in excluded}
    historical = {task['group_id'] for task in holdout_excluded}
    historical_hashes = {task['game_sha256'] for task in holdout_excluded}
    with ZipFile(games_path) as games, ZipFile(metadata_path) as metadata:
        pools = _catalogue(games, metadata, (family,), groups, hashes | historical_hashes, seed)[family]
        official = _spread_groups(pools['valid_unseen']) + _spread_groups(pools['valid_seen'])
        # Exclude content aliases even when an older release placed them in a different game group.
        historical.update(row[0] for row in pools['train'] + official if row[2] in historical_hashes)
        final = _holdout(pools['train'], official, historical, seed, family + '/final', final_size)
        selection = _holdout(pools['train'], official, historical | {r[0] for r in final}, seed, family + '/selection', holdout_size)
        reserved = {r[0] for r in final + selection}
        dev = _spread_groups([r for r in pools['train'] if r[0] not in reserved])
        # All previous exact instances are absent. Prefer new group identities at the gates.
        dev.sort(key=lambda row: row[0] in historical)
        rounds = min(max_rounds, len(dev) // (3 * size))
        if rounds < 1:
            raise ValueError(f'{family} cannot supply a single full round and two independent holdouts')

        def convert(rows):
            return [_task(path, sha, family, metadata, seed) for _, path, sha in rows]

        def order(rows, label):
            return sorted(convert(rows), key=lambda row: digest([seed, label, row['id']]))

        batches = []
        for i in range(rounds):
            gate_start = 2 * i * size
            learn_start = (2 * rounds + i) * size
            learning = dev[learn_start:learn_start + size]
            primary, meta = _partition_fresh(dev[gate_start:gate_start + 2 * size], size)
            batches.append(dict(learning=convert(learning), primary=convert(primary), meta=convert(meta)))
        exposure = {phase: dict(
            content=sum(t['game_sha256'] in historical_hashes for batch in batches for t in batch[phase]),
            group=sum(t['group_id'] in historical for batch in batches for t in batch[phase]))
            for phase in ('learning', *GATES)}
        plan = dict(protocol=protocol, scope=SCOPE, seed=seed, family=family,
                    requested_rounds=max_rounds, rounds=rounds, batch_size=size, batches=batches,
                    games_sha256=GAME_SHA, metadata_sha256=file_hash(metadata_path),
                    selection=order(selection, 'selection'), final=order(final, 'final'),
                    eligible_by_split={split: len(rows) for split, rows in pools.items()},
                    development_capacity=len(dev), previous_registration_overlap=exposure, budget={'per_round': {'episodes': 13 * size, 'infrastructure_attempts': 2}},
                    exclusions={'all_group_hash': digest(sorted(groups)), 'all_content_hash': digest(sorted(hashes)),
                                'holdout_group_hash': digest(sorted(historical)), 'prior_content_excluded_everywhere_hash': digest(sorted(historical_hashes))})
    if protocol != COEVOLUTION_PROTOCOL:
        plan['holdout_size'] = holdout_size
        if final_size != holdout_size:
            plan['final_size'] = final_size
    validate_coevolution_manifest(plan)
    return plan


def validate_coevolution_manifest(plan):
    if plan['protocol'] == COEVOLUTION_MIXED_10X60_PROTOCOL:
        from .mixed_panel import validate_mixed_manifest
        return validate_mixed_manifest(plan)
    family, protocol = plan['family'], plan['protocol']
    max_rounds, size = coevolution_shape(protocol)
    holdout_size, final_size = _holdout_sizes(protocol, plan.get('holdout_size'), plan.get('final_size'))
    if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL and family != 'heat':
        raise ValueError('The 5x23 registration is specific to heat')
    if protocol != COEVOLUTION_PROTOCOL and 'holdout_size' not in plan:
        raise ValueError('Missing selection panel size')
    if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL and 'final_size' not in plan:
        raise ValueError('Missing final panel size')
    if (plan['scope'] != SCOPE or plan['games_sha256'] != GAME_SHA
            or family not in HARD_FAMILIES or type(plan['seed']) is not int
            or type(plan['batch_size']) is not int or plan['batch_size'] != size
            or type(plan['requested_rounds']) is not int or type(plan['rounds']) is not int
            or plan['requested_rounds'] != max_rounds or plan['rounds'] != len(plan['batches'])
            or type(plan['development_capacity']) is not int
            or not 1 <= plan['rounds'] == min(max_rounds, plan['development_capacity'] // (3 * size))
            or plan['budget'] != {'per_round': {'episodes': 13 * size, 'infrastructure_attempts': 2}}):
        raise ValueError('Invalid coevolution registration or capacity')
    dev = []
    for batch in plan['batches']:
        if set(batch) != {'learning', *GATES} or any(len(pool) != size for pool in batch.values()):
            raise ValueError(f'Each registered phase needs exactly {size} fresh games')
        _check_fresh_groups(batch)
        dev.extend(task for pool in batch.values() for task in pool)
    selection, final = plan['selection'], plan['final']
    if len(selection) != holdout_size or len(final) != final_size or any(row['split'] != 'train' for row in dev):
        raise ValueError('Invalid development or holdout panel')
    splits = plan['eligible_by_split']
    if (set(splits) != {'train','valid_seen','valid_unseen'} or
            any(type(v) is not int or v < 0 for v in splits.values()) or plan['development_capacity'] > splits['train']):
        raise ValueError('Invalid catalogue counts')
    seen_groups = set()
    for tasks in (dev, selection, final):
        groups = {t['group_id'] for t in tasks}
        if seen_groups & groups:
            raise ValueError('Development, arm selection and final test must use disjoint game groups')
        seen_groups.update(groups)
    rows = dev + selection + final
    for row in rows:
        validate_registered_task(row, plan['seed'], family, splits)
    if any(len({row[field] for row in rows}) != len(rows) for field in ('game_sha256','archive_member','id')):
        raise ValueError('A registered task occurs twice')


def validate_registered_task(row, seed, family, splits):
    member = row['archive_member']
    parts = Path(member).parts
    if (len(parts) != 5 or parts[0] != 'json_2.1.1' or parts[-1] != 'game.tw-pddl' or '..' in parts
            or row['split'] not in splits or row['split'] != parts[1] or row['group_id'] != parts[2]
            or FAMILIES.get(parts[2].split('-', 1)[0]) != family or row['family'] != family
            or row['room'] != int(parts[2].rsplit('-', 1)[-1]) or row['environment'] != 'alfworld'
            or not re.fullmatch(r'[0-9a-f]{64}', row['game_sha256']) or row['id'] != 'rsi-alfworld/' + digest(member)[:18]
            or row['seed'] != int(digest([seed, 'environment', member])[:8], 16) % 2147483647):
        raise ValueError('Game identity differs from the family registration')
