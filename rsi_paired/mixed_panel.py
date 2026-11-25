"""Preregister one interleaved search over the six official ALFWorld families."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from zipfile import ZipFile

from benchmarks.alfworld.panel import FAMILIES
from evomem.common import digest, file_hash
from scripts.prepare_assets import GAME_SHA
from .coevolution_panel import validate_registered_task
from .panel import (COEVOLUTION_MIXED_10X60_PROTOCOL, GATES, MIXED_FAMILIES, _partition_fresh,
                    _spread_groups, _task, validate_mixed_round)

SCOPE = 'alfworld_coevolution_mixed_families'
VAL_SPLITS = ('valid_unseen', 'valid_seen')
POLICY = 'official_train_development_exclude_all_official_validation_content_and_selected_holdout_groups_v1'


def _catalogue(games, metadata, seed):
    """Never use a validation file/content for development; unused validation group names are not embargoed."""
    names = set(metadata.namelist())
    splits = ('train', *VAL_SPLITS)
    catalog = {f: {split: [] for split in splits} for f in MIXED_FAMILIES}
    validation_content = set()
    for name in sorted(games.namelist()):
        path = Path(name)
        parts = path.parts
        if len(parts) != 5 or parts[0] != 'json_2.1.1' or parts[-1] != 'game.tw-pddl':
            continue
        split, group = parts[1:3]
        sha = hashlib.sha256(games.read(name)).hexdigest()
        if split != 'train':
            validation_content.add(sha)
        family = FAMILIES.get(group.split('-', 1)[0])
        if family and split in splits and str(path.with_name('traj_data.json')) in names:
            catalog[family][split].append((group, name, sha))
    seen = set()
    for split in (*VAL_SPLITS, 'train'):
        for family in MIXED_FAMILIES:
            unique = []
            for row in sorted(catalog[family][split], key=lambda r: digest([seed, 'trial', r[1]])):
                sha = row[2]
                if sha in seen or (split == 'train' and sha in validation_content):
                    continue
                unique.append(row)
                seen.add(sha)
            catalog[family][split] = unique
    return catalog


def final_quotas(seed):
    order = sorted(MIXED_FAMILIES, key=lambda f: digest([seed, 'final-extra', f]))
    return {family: 16 + (index < 4) for index, family in enumerate(order)}


def _reserve(pools, size, unavailable, train_counts, seed, label):
    """Prefer unseen validation; among available groups consume the least training capacity first."""
    result = []
    for split in VAL_SPLITS:
        grouped = defaultdict(list)
        for row in pools[split]:
            if row[0] not in unavailable:
                grouped[row[0]].append(row)
        groups = sorted(grouped, key=lambda g: (train_counts[g], digest([seed, label, g])))
        for group in groups:
            if len(result) == size:
                return result
            result.extend(grouped[group][:size - len(result)])
            unavailable.add(group)
    if len(result) != size:
        raise ValueError(f'Insufficient separate official validation groups for {label}')
    return result


def _interleave(by_family, seed, label):
    """Round-robin across a new deterministic permutation for each block of six."""
    rows = []
    for index in range(max(map(len, by_family.values()))):
        order = sorted(by_family, key=lambda f: digest([seed, label, index, f]))
        rows.extend(by_family[f][index] for f in order if index < len(by_family[f]))
    return rows


def mixed_manifest(games_path, metadata_path, seed):
    if type(seed) is not int or file_hash(games_path) != GAME_SHA:
        raise ValueError('Mixed registration needs an integer seed and the pinned ALFWorld archive')
    quotas = final_quotas(seed)
    plans = {}
    capacities = {}
    with ZipFile(games_path) as games, ZipFile(metadata_path) as metadata:
        catalog = _catalogue(games, metadata, seed)
        for family in MIXED_FAMILIES:
            pools = catalog[family]
            reserved = set()
            train_counts = Counter(row[0] for row in pools['train'])
            final = _reserve(pools, quotas[family], reserved, train_counts, seed, 'final')
            selection = _reserve(pools, 5, reserved, train_counts, seed, 'selection')
            dev = _spread_groups([row for row in pools['train'] if row[0] not in reserved])
            if len(dev) < 300:
                raise ValueError(f'{family}: ten balanced rounds require 300 distinct eligible train games; found {len(dev)}')
            capacities[family] = dict(eligible_by_official_split={s: len(v) for s,v in pools.items()},
                                      development_capacity=len(dev), reserved_group_count=len(reserved))
            def convert(rows):
                return [_task(name, sha, family, metadata, seed) for _, name, sha in rows]
            batches = []
            for index in range(10):
                primary, meta = _partition_fresh(dev[index * 20:(index + 1) * 20], 10)
                learning = dev[200 + index * 10:210 + index * 10]
                batches.append(dict(learning=convert(learning), primary=convert(primary), meta=convert(meta)))
            plans[family] = dict(batches=batches, selection=convert(selection), final=convert(final))
    batches = [{phase: _interleave({f: plans[f]['batches'][i][phase] for f in MIXED_FAMILIES}, seed, [i, phase])
                for phase in ('learning', *GATES)} for i in range(10)]
    plan = dict(protocol=COEVOLUTION_MIXED_10X60_PROTOCOL, scope=SCOPE, family='mixed', families=list(MIXED_FAMILIES),
                seed=seed, games_sha256=GAME_SHA, metadata_sha256=file_hash(metadata_path), data_policy=POLICY,
                rounds=10, requested_rounds=10, batch_size=60, holdout_size=30, final_size=100,
                budget={'per_round': {'episodes': 780, 'infrastructure_attempts': 2}},
                batches=batches, capacities=capacities, final_family_quotas=quotas,
                selection=_interleave({f: plans[f]['selection'] for f in MIXED_FAMILIES}, seed, 'selection'),
                final=_interleave({f: plans[f]['final'] for f in MIXED_FAMILIES}, seed, 'final'))
    validate_mixed_manifest(plan)
    return plan


def _check_order(rows, seed, label):
    by_family = {f: [t for t in rows if t['family'] == f] for f in MIXED_FAMILIES}
    if rows != _interleave(by_family, seed, label):
        raise ValueError('Mixed panel order differs from its registered family interleaving')


def validate_mixed_manifest(plan):
    seed = plan['seed']
    if (plan['protocol'] != COEVOLUTION_MIXED_10X60_PROTOCOL or plan['scope'] != SCOPE
            or plan['family'] != 'mixed' or plan['families'] != list(MIXED_FAMILIES)
            or plan['data_policy'] != POLICY or type(seed) is not int or plan['games_sha256'] != GAME_SHA
            or (plan['requested_rounds'], plan['rounds'], plan['batch_size'], plan['holdout_size'], plan['final_size']) != (10, 10, 60, 30, 100)
            or plan['budget'] != {'per_round': {'episodes': 780, 'infrastructure_attempts': 2}}
            or len(plan['batches']) != 10 or plan['final_family_quotas'] != final_quotas(seed)
            or set(plan['capacities']) != set(MIXED_FAMILIES)):
        raise ValueError('Invalid mixed six-family protocol, quotas or budget')
    dev = []
    for i,batch in enumerate(plan['batches']):
        validate_mixed_round(batch)
        for phase, rows in batch.items():
            _check_order(rows, seed, [i,phase])
            dev.extend(rows)
    selection, final = plan['selection'], plan['final']
    if (Counter(t['family'] for t in selection) != dict.fromkeys(MIXED_FAMILIES, 5) or
            Counter(t['family'] for t in final) != final_quotas(seed)):
        raise ValueError('Invalid per-family mixed selection or final quotas')
    _check_order(selection, seed, 'selection')
    _check_order(final, seed, 'final')
    groups_used = set()
    for tasks, splits in ((dev, {'train'}), (selection, VAL_SPLITS), (final, VAL_SPLITS)):
        groups = {t['group_id'] for t in tasks}
        if groups & groups_used:
            raise ValueError('Development, selection and final must contain disjoint game groups')
        groups_used.update(groups)
        for t in tasks:
            validate_registered_task(t, seed, t['family'], splits)
    rows = dev + selection + final
    for field in ('game_sha256', 'id', 'archive_member'):
        if len({t[field] for t in rows}) != len(rows):
            raise ValueError(f'Duplicate mixed registration {field}')
    for family, counts in plan['capacities'].items():
        eligible = counts['eligible_by_official_split']
        reserved = {t['group_id'] for t in selection + final if t['family'] == family}
        if (set(eligible) != {'train', *VAL_SPLITS} or any(type(n) is not int or n < 0 for n in eligible.values())
                or type(counts['development_capacity']) is not int
                or not 300 <= counts['development_capacity'] <= eligible['train'] or counts['reserved_group_count'] != len(reserved)):
            raise ValueError('Invalid mixed registration capacity or holdout group count')
