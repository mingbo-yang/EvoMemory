"""Register official ALFWorld games and resolve each protocol's development panels."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from benchmarks.alfworld.panel import FAMILIES
from evomem.common import digest, file_hash
from scripts.prepare_assets import GAME_SHA


def _task(member, game_hash, family, metadata, seed):
    path = Path(member)
    trajectory = json.loads(metadata.read(str(path.with_name("traj_data.json"))))
    if FAMILIES.get(trajectory["task_type"]) != family:
        raise ValueError("Game family does not match official metadata")
    group = path.parts[2]
    return dict(id="rsi-alfworld/" + digest(member)[:18], environment="alfworld", family=family,
                group_id=group, seed=int(digest([seed, "environment", member])[:8], 16) % 2147483647,
                room=int(group.rsplit("-", 1)[-1]), archive_member=member, game_sha256=game_hash,
                split=path.parts[1], goal="Complete the task stated in the initial observation.")


def smoke_manifest(games_path, metadata_path, seed, excluded_groups=(), excluded_games=()):
    if file_hash(games_path) != GAME_SHA:
        raise ValueError("ALFWorld game archive differs from the pinned release")
    blocked_groups = set(excluded_groups)
    blocked_games = set(excluded_games)
    by_group = defaultdict(list)
    with ZipFile(games_path) as games, ZipFile(metadata_path) as metadata:
        for member in sorted(games.namelist()):
            parts = Path(member).parts
            if len(parts) != 5 or parts[0] != "json_2.1.1" or parts[-1] != "game.tw-pddl":
                continue
            if parts[1] == "train":
                by_group[parts[2]].append(member)
            else:
                blocked_groups.add(parts[2])
                blocked_games.add(hashlib.sha256(games.read(member)).hexdigest())
        ordered = sorted(by_group, key=lambda group: digest([seed, group]))
        # Use the same two families in each phase so an empty bank is not forced
        # by choosing a different family at every evaluation boundary.
        paired_families = sorted(set(FAMILIES.values()), key=lambda family: digest([seed, "family", family]))[:2]
        panels = {}
        for phase in ("learning", "primary", "meta"):
            tasks = []
            for family in paired_families:
                for group in ordered:
                    if group in blocked_groups or FAMILIES.get(group.split("-", 1)[0]) != family:
                        continue
                    member = by_group[group][0]
                    game_hash = hashlib.sha256(games.read(member)).hexdigest()
                    if game_hash in blocked_games:
                        continue
                    tasks.append(_task(member, game_hash, family, metadata, seed))
                    blocked_groups.add(group)
                    blocked_games.add(game_hash)
                    break
                else:
                    raise ValueError("Too few distinct groups for the selected panel")
            panels[phase] = tasks
    return dict(protocol="rsi_original_parent_paired_v2", scope="engineering", seed=seed,
                games_sha256=GAME_SHA, metadata_sha256=file_hash(metadata_path),
                excluded_groups_hash=digest(sorted(excluded_groups)), excluded_games_hash=digest(sorted(excluded_games)),
                panels=panels)


def save_registration(path, value):
    """Write once and compare on resume, without overwriting a previous registration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        name = tmp.name
        try:
            json.dump(value, tmp, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
            try:
                os.link(name, path)
            except FileExistsError:
                if json.loads(path.read_text()) != value:
                    raise ValueError("Frozen registration differs: " + str(path))
        finally:
            Path(name).unlink(missing_ok=True)


def formal_manifest(games_path, metadata_path, seed, excluded_groups=(), excluded_games=()):
    """Stratify by native family and register all 240 distinct instances without scoring."""
    if file_hash(games_path) != GAME_SHA:
        raise ValueError("ALFWorld game archive differs from the pinned release")
    blocked_groups, used_hashes = set(excluded_groups), set(excluded_games)
    train, validation = defaultdict(list), defaultdict(list)
    with ZipFile(games_path) as games, ZipFile(metadata_path) as metadata:
        metadata_names = set(metadata.namelist())
        groups_outside_train = set()
        for name in sorted(games.namelist()):
            member = Path(name)
            if len(member.parts) != 5 or member.parts[0] != "json_2.1.1" or member.name != "game.tw-pddl":
                continue
            _, split, group, _, _ = member.parts
            family = FAMILIES.get(group.split("-", 1)[0])
            if split != "train":
                groups_outside_train.add(group)
            if not family or group in blocked_groups or str(member.with_name("traj_data.json")) not in metadata_names:
                continue
            if split == "train":
                train[family].append((group, name))
            elif split in ("valid_unseen", "valid_seen"):
                validation[family].append((group, name))
        used_groups = set()

        def take(items, n, family, subset, one_per_group=False):
            order = sorted(items, key=lambda p: (
                # Prefer unseen instances for the final panel and spread trials across groups.
                "/valid_seen/" in p[1], digest([seed, subset, p[0]]), digest([seed, subset, p[1]])))
            if not one_per_group:
                grouped = defaultdict(list)
                for group, name in order:
                    grouped[group].append(name)
                # Cycle groups within each official split before considering the next split.
                order = sorted(
                    ((group, name, i) for group, names in grouped.items() for i, name in enumerate(names)),
                    key=lambda p: ("/valid_seen/" in p[1], p[2], digest([seed, subset, p[0]])))
                order = [(group, name) for group, name, _ in order]
            tasks = []
            for group, name in order:
                if one_per_group and (group in used_groups or group in groups_outside_train):
                    continue
                raw = hashlib.sha256(games.read(name)).hexdigest()
                if raw in used_hashes:
                    continue
                tasks.append(_task(name, raw, family, metadata, seed))
                used_hashes.add(raw)
                used_groups.add(group)
                if len(tasks) == n:
                    return tasks
            raise ValueError(f"Not enough disjoint official games for {subset}/{family}: got {len(tasks)}, need {n}")

        families = sorted(FAMILIES.values())
        final = [task for f in families for task in take(validation[f], 20, f, "final")]
        development = {f: take(train[f], 20, f, "development", one_per_group=True) for f in families}

    def ordered(rows, tag):
        return sorted(rows, key=lambda task: digest([seed, tag, task["id"]]))

    specs = (("learning", 0, 2), ("primary", 2, 3), ("meta", 3, 4))
    batches = [{name: ordered([t for f in families for t in development[f][i*4+start:i*4+end]], [i, name])
                for name, start, end in specs} for i in range(4)]
    confirmation = ordered([t for f in families for t in development[f][16:20]], "confirmation")
    return dict(protocol="rsi_original_parent_paired_v2", scope="formal_alfworld_240", seed=seed,
                games_sha256=GAME_SHA, metadata_sha256=file_hash(metadata_path), batches=batches,
                confirmation=confirmation, final=ordered(final, "final-order"),
                final_by_official_split=dict(Counter(t["split"] for t in final)),
                excluded_groups_hash=digest(sorted(excluded_groups)),
                excluded_games_hash=digest(sorted(excluded_games)))


HISTORICAL_SEQUENTIAL_PROTOCOL = "rsi_original_parent_paired_v3_6x30"
SEQUENTIAL_PROTOCOL = "rsi_original_parent_paired_v3_6x30_fresh"
SEQUENTIAL_PROTOCOLS = (HISTORICAL_SEQUENTIAL_PROTOCOL, SEQUENTIAL_PROTOCOL)
COEVOLUTION_PROTOCOL = "rsi_reviewer_evolver_v4_hard_10x50"
COEVOLUTION_BATCH_SIZE = 50
COEVOLUTION_MAX_ROUNDS = 10
COEVOLUTION_6X30_PROTOCOL = "rsi_reviewer_evolver_v4_hard_6x30"
COEVOLUTION_HEAT_5X23_PROTOCOL = "rsi_reviewer_evolver_v4_heat_5x23"
COEVOLUTION_MIXED_10X60_PROTOCOL = "rsi_reviewer_evolver_v4_mixed_10x60"
MIXED_FAMILIES = tuple(sorted(FAMILIES.values()))
COEVOLUTION_PROTOCOLS = (COEVOLUTION_PROTOCOL, COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL,
                         COEVOLUTION_MIXED_10X60_PROTOCOL)
NATIVE_PAIRED_PROTOCOLS = (*SEQUENTIAL_PROTOCOLS, *COEVOLUTION_PROTOCOLS)


def coevolution_shape(protocol):
    if protocol == COEVOLUTION_PROTOCOL:
        return COEVOLUTION_MAX_ROUNDS, COEVOLUTION_BATCH_SIZE
    if protocol == COEVOLUTION_6X30_PROTOCOL:
        return 6, 30
    if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL:
        return 5, 23
    if protocol == COEVOLUTION_MIXED_10X60_PROTOCOL:
        return 10, 60
    raise ValueError(f"Unknown coevolution protocol: {protocol}")



SEQUENTIAL_SCOPE = "formal_alfworld_sequential_families"
DEFAULT_FAMILIES = ("simple", "two")
FAMILY_ROUNDS = 6
FAMILY_BATCH_SIZE = 30
FAMILY_TEST_SIZE = 100
HISTORICAL_QUOTA = 6
GATES = ("primary", "meta")



def _sequential_budget():
    final = 2 * FAMILY_TEST_SIZE
    maximum = 390 * FAMILY_ROUNDS + final
    return dict(per_round=dict(episodes=390, infrastructure_attempts=2), final_per_family=final,
                max_native_episodes_per_family=maximum, max_native_episodes_total=2 * maximum)


def _partition_fresh(rows, size=FAMILY_BATCH_SIZE):
    """Give both fresh reserves the registered width while keeping sibling trials on one side."""
    first, second = rows[:size], rows[size:]
    if not ({row[0] for row in first} & {row[0] for row in second}):
        return first, second
    # Assign sibling trials to one side. Prefer groups in their seeded input order.
    sizes = Counter(row[0] for row in rows)
    picked = {0: ()}
    for group, group_size in sizes.items():
        for total, previous in tuple(picked.items()):
            if total + group_size <= size:
                picked.setdefault(total + group_size, previous + (group,))
        if size in picked:
            break
    if size not in picked:
        raise ValueError("Cannot register two complete group-disjoint fresh reserves")
    primary = set(picked[size])
    return [r for r in rows if r[0] in primary], [r for r in rows if r[0] not in primary]


def _check_fresh_groups(panels):
    if {r["group_id"] for r in panels["primary"]} & {r["group_id"] for r in panels["meta"]}:
        raise ValueError("Fresh reserve game groups overlap between primary and meta")

def _catalogue(games, metadata, families, excluded_groups, excluded_games, seed):
    """Eligible trial paths and content hashes. Reserve all non-training groups and content."""
    names = set(metadata.namelist())
    available = {family: defaultdict(list) for family in families}
    nontrain_groups, nontrain_content = set(), set()
    for name in sorted(games.namelist()):
        path = Path(name)
        if len(path.parts) != 5 or path.parts[0] != "json_2.1.1" or path.name != "game.tw-pddl":
            continue
        _, split, group, _, _ = path.parts
        family = FAMILIES.get(group.split("-", 1)[0])
        if split == "train" and family not in available:
            continue
        sha = hashlib.sha256(games.read(name)).hexdigest()
        if split != "train":
            nontrain_groups.add(group)
            nontrain_content.add(sha)
        if (family in available and split in ("train", "valid_unseen", "valid_seen")
                and group not in excluded_groups and sha not in excluded_games
                and str(path.with_name("traj_data.json")) in names):
            available[family][split].append((group, name, sha))
    # A duplicated game must not enter another split or another family as a distinct task.
    seen = set()
    for split in ("valid_unseen", "valid_seen", "train"):
        for pools in available.values():
            unique = []
            for group, name, sha in sorted(pools[split], key=lambda row: digest([seed, "trial", row[1]])):
                if sha in seen or (split == "train" and (group in nontrain_groups or sha in nontrain_content)):
                    continue
                unique.append((group, name, sha))
                seen.add(sha)
            pools[split] = unique
    return available


def _holdout_groups(train, required, reserve, seed, family):
    """Choose whole groups using seed and size, minimizing unused training games."""
    grouped = defaultdict(list)
    for row in train:
        grouped[row[0]].append(row)
    surplus = len(train) - reserve
    if surplus < required:
        raise ValueError(f"{family}: need {reserve} development games and {required} train holdouts, have {len(train)}")
    # Subset sums avoid a greedy last group consuming games reserved for development.
    reachable = {0: None}
    for group in sorted(grouped, key=lambda g: digest([seed, family, "holdout-group", g])):
        if required in reachable:
            break
        size = len(grouped[group])
        for previous in tuple(reachable):
            count = previous + size
            if count <= surplus and count not in reachable:
                reachable[count] = previous, group
    count = min((n for n in reachable if n >= required), default=None)
    if count is None:
        raise ValueError(f"{family}: the training groups cannot supply a disjoint {required}-game holdout")
    groups = set()
    while count:
        count, group = reachable[count]
        groups.add(group)
    return groups


def _spread_groups(rows):
    """Preserve seed order within and between groups, using one trial per group before its siblings."""
    counts = Counter()
    ranked = []
    for row in rows:
        group = row[0]
        ranked.append((counts[group], row))
        counts[group] += 1
    return [row for _, row in sorted(ranked, key=lambda item: item[0])]


def sequential_manifest(games_path, metadata_path, seed, families=DEFAULT_FAMILIES,
                        excluded_groups=(), excluded_games=()):
    """Freeze all learning, fresh, and final trial identities before any outcomes are known."""
    if len(families) != 2 or len(set(families)) != 2 or not set(families) <= set(FAMILIES.values()):
        raise ValueError("Select two different native ALFWorld families")
    if file_hash(games_path) != GAME_SHA:
        raise ValueError("ALFWorld game archive differs from the pinned release")
    excluded_groups, excluded_games = set(excluded_groups), set(excluded_games)
    outputs = {}
    reserve = FAMILY_ROUNDS * (1 + len(GATES)) * FAMILY_BATCH_SIZE
    with ZipFile(games_path) as games, ZipFile(metadata_path) as metadata:
        catalogue = _catalogue(games, metadata, families, excluded_groups, excluded_games, seed)
        for family, pools in catalogue.items():
            official = _spread_groups(pools["valid_unseen"]) + _spread_groups(pools["valid_seen"])
            final = official[:FAMILY_TEST_SIZE]
            missing = FAMILY_TEST_SIZE - len(final)
            train_groups = _holdout_groups(pools["train"], missing, reserve, seed, family)
            final += _spread_groups([r for r in pools["train"] if r[0] in train_groups])[:missing]
            dev = _spread_groups([r for r in pools["train"] if r[0] not in train_groups])[:reserve]

            def tasks(rows):
                return [_task(name, sha, family, metadata, seed) for _, name, sha in rows]

            batches = []
            cursor = 0
            for _ in range(FAMILY_ROUNDS):
                learning = dev[cursor:cursor + FAMILY_BATCH_SIZE]
                cursor += FAMILY_BATCH_SIZE
                end = cursor + len(GATES) * FAMILY_BATCH_SIZE
                primary, meta = _partition_fresh(dev[cursor:end])
                batches.append(dict(learning=tasks(learning), primary=tasks(primary), meta=tasks(meta)))
                cursor = end
            tests = sorted(tasks(final), key=lambda t: digest([seed, "final-order", t["id"]]))
            outputs[family] = dict(batches=batches, final=tests,
                                   final_by_official_split=dict(Counter(t["split"] for t in tests)))
    manifest = dict(protocol=SEQUENTIAL_PROTOCOL, scope=SEQUENTIAL_SCOPE, seed=seed, budget=_sequential_budget(),
                    family_order=list(families), families=outputs, games_sha256=GAME_SHA,
                    metadata_sha256=file_hash(metadata_path), excluded_groups_hash=digest(sorted(excluded_groups)),
                    excluded_games_hash=digest(sorted(excluded_games)))
    validate_sequential_manifest(manifest)
    return manifest


def validate_sequential_manifest(manifest):
    order, families = manifest["family_order"], manifest["families"]
    if (manifest.get("protocol") not in SEQUENTIAL_PROTOCOLS or manifest.get("scope") != SEQUENTIAL_SCOPE
            or type(manifest["seed"]) is not int or manifest["games_sha256"] != GAME_SHA
            or len(order) != 2 or len(set(order)) != 2 or set(order) != set(families)
            or not set(order) <= set(FAMILIES.values())):
        raise ValueError("Invalid two-family registration")
    if digest(manifest.get("budget")) != digest(_sequential_budget()):
        raise ValueError("Missing or mismatched registered sequential runtime budget")
    all_tasks, dev_groups, final_groups = [], set(), set()
    for family in order:
        plan = families[family]
        batches, final = plan["batches"], plan["final"]
        if len(batches) != FAMILY_ROUNDS or len(final) != FAMILY_TEST_SIZE:
            raise ValueError("Expected six rounds and 100 final games per family")
        training = []
        for batch in batches:
            if set(batch) != {"learning", *GATES} or any(len(v) != FAMILY_BATCH_SIZE for v in batch.values()):
                raise ValueError("Expected 30 learning games and two disjoint fresh reserves of 30 games")
            _check_fresh_groups(batch)
            training.extend(t for pool in batch.values() for t in pool)
        if (any(t["split"] != "train" for t in training)
                or any(t["split"] not in ("train", "valid_seen", "valid_unseen") for t in final)
                or plan["final_by_official_split"] != dict(Counter(t["split"] for t in final))):
            raise ValueError("Invalid development or final split")
        for task in training + final:
            name = task["archive_member"]
            parts = Path(name).parts
            if (len(parts) != 5 or parts[0] != "json_2.1.1" or parts[-1] != "game.tw-pddl"
                    or parts[2] != task["group_id"] or parts[1] != task["split"] or ".." in parts
                    or FAMILIES.get(parts[2].split("-", 1)[0]) != family or task["family"] != family
                    or task["environment"] != "alfworld" or not re.fullmatch(r"[0-9a-f]{64}", task["game_sha256"])
                    or task["id"] != "rsi-alfworld/" + digest(name)[:18]
                    or task["room"] != int(parts[2].rsplit("-", 1)[-1])
                    or task["seed"] != int(digest([manifest["seed"], "environment", name])[:8], 16) % 2147483647):
                raise ValueError("Task does not match its registered family, archive path or identity")
        all_tasks.extend(training + final)
        dev_groups.update(t["group_id"] for t in training)
        final_groups.update(t["group_id"] for t in final)
    if dev_groups & final_groups:
        raise ValueError("Final game group occurs in development")
    for key in ("id", "game_sha256", "archive_member"):
        if len({t[key] for t in all_tasks}) != len(all_tasks):
            raise ValueError(f"Repeated official game {key} in the static registration")



def _allocate_gates(panels, history, seed):
    """Jointly fill both gates while allowing siblings only on the same side.

    Maximize historical coverage first. At equal coverage, successive seats in
    each gate/outcome stratum have diminishing priority, so scarcity is shared
    fairly. Frozen reserve order and seeded historical ranks break ties.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    costs, constraints, lower, upper = [], [], [], []
    options = {gate: [] for gate in GATES}
    group_columns = defaultdict(lambda: defaultdict(list))

    def variable(cost):
        costs.append(cost)
        return len(costs) - 1

    def constrain(coefficients, lo, hi):
        constraints.append(coefficients)
        lower.append(lo)
        upper.append(hi)

    def option(gate, task, evidence, rank):
        column = variable(rank)
        options[gate].append((column, task, evidence))
        group_columns[task["group_id"]][gate].append(column)
        return column

    seats = len(GATES) * FAMILY_BATCH_SIZE
    rank_bound = max(FAMILY_BATCH_SIZE, *(len(pool) for pool in history.values()))
    fairness_weight = seats * rank_bound + 1
    # One extra historical seat dominates all possible fairness and rank differences.
    history_weight = fairness_weight * (len(GATES) * len(history) * HISTORICAL_QUOTA ** 2 + 1)
    for gate in GATES:
        for rank, task in enumerate(panels[gate]):
            option(gate, task, dict(task_id=task["id"], source="fresh"), rank)
        for score, pool in history.items():
            tag = "historical_success" if score else "historical_failure"
            ordered = sorted(pool, key=lambda item: digest([seed, "history", score, gate, item[1]["id"]]))
            count = {}
            for rank, (index, task, row) in enumerate(ordered):
                evidence = dict(task_id=task["id"], source=tag, learning_index=index,
                                learning_round=index // FAMILY_BATCH_SIZE + 1, episode_hash=row["episode_hash"])
                count[option(gate, task, evidence, rank)] = 1
            # Each selected historical task fills one of the six historical seats.
            for priority in range(HISTORICAL_QUOTA, 0, -1):
                count[variable(-history_weight - priority * fairness_weight)] = -1
            constrain(count, 0, 0)
        constrain({column: 1 for column, _, _ in options[gate]}, FAMILY_BATCH_SIZE, FAMILY_BATCH_SIZE)

    for sides in group_columns.values():
        if len(sides) == 2:
            owner = variable(0)  # 1 assigns the group to primary, 0 to meta
            for gate in GATES:
                rows = sides[gate]
                coefficients = dict.fromkeys(rows, 1)
                if gate == "primary":
                    coefficients[owner] = -len(rows)
                    constrain(coefficients, -np.inf, 0)
                else:
                    coefficients[owner] = len(rows)
                    constrain(coefficients, -np.inf, len(rows))

    values, rows, cols = [], [], []
    for index, coefficients in enumerate(constraints):
        for column, coefficient in coefficients.items():
            rows.append(index)
            cols.append(column)
            values.append(coefficient)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(constraints), len(costs))).tocsc()
    solution = milp(costs, integrality=np.ones(len(costs)), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix, lower, upper), options={"mip_rel_gap": 0})
    if not solution.success:
        raise ValueError("Cannot resolve two complete group-disjoint validation panels: " + solution.message)
    chosen = {gate: [(task, entry) for col, task, entry in options[gate] if solution.x[col] > 0.5]
              for gate in GATES}
    if (any(len(rows) != FAMILY_BATCH_SIZE for rows in chosen.values())
            or {t["group_id"] for t, _ in chosen["primary"]} & {t["group_id"] for t, _ in chosen["meta"]}):
        raise ValueError("Incomplete or overlapping group assignment")
    return chosen

def validate_mixed_round(panels):
    phases = ("learning", *GATES)
    if set(panels) != set(phases) or any(Counter(t["family"] for t in panels[p]) != dict.fromkeys(MIXED_FAMILIES, 10)
                                         for p in phases):
        raise ValueError("Each mixed phase requires exactly ten tasks from each of six families")
    _check_fresh_groups(panels)


def resolve_family_panels(panels, learning_tasks, ledger, seed, *, protocol=SEQUENTIAL_PROTOCOL):
    """Use registered games in order. Earlier registrations retain outcome-stratified sampling."""
    if protocol not in NATIVE_PAIRED_PROTOCOLS:
        raise ValueError("Unknown sequential panel protocol")
    size, rounds = FAMILY_BATCH_SIZE, FAMILY_ROUNDS
    if protocol in COEVOLUTION_PROTOCOLS:
        rounds, size = coevolution_shape(protocol)
    if set(panels) != {"learning", *GATES} or any(len(v) != size for v in panels.values()):
        raise ValueError("Invalid fresh reserve sizes")
    if (not size <= len(learning_tasks) <= rounds * size
            or len(learning_tasks) % size or len(learning_tasks) != len(ledger)
            or learning_tasks[-size:] != panels["learning"]):
        raise ValueError("Learning prefix must end with this complete learning batch")
    cutoff_tasks = learning_tasks + panels["primary"] + panels["meta"]
    families = {t["family"] for t in cutoff_tasks}
    if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL and families != {"heat"}:
        raise ValueError("The 5x23 registration is specific to heat")
    if protocol == COEVOLUTION_MIXED_10X60_PROTOCOL:
        validate_mixed_round(panels)
        expected = dict.fromkeys(MIXED_FAMILIES, len(learning_tasks) // len(MIXED_FAMILIES))
        if Counter(t["family"] for t in learning_tasks) != expected:
            raise ValueError("Mixed learning history differs from the six registered family quotas")
    elif len(families) != 1:
        raise ValueError("Historical and fresh tasks must belong to the same family")
    for key in ("id", "archive_member", "game_sha256"):
        if len({t[key] for t in cutoff_tasks}) != len(cutoff_tasks):
            raise ValueError("Duplicate game between learning history and fresh reserves")
    _check_fresh_groups(panels)
    history = {0: [], 1: []}
    for index, (task, row) in enumerate(zip(learning_tasks, ledger)):
        score = row.get("native_score")
        if (task["id"] != row.get("task_id") or task["family"] != row.get("family")
                or row.get("phase") != "learning" or row.get("status") != "completed"
                or type(score) not in (int, float) or score not in (0, 1)
                or type(row.get("score")) not in (int, float) or row["score"] != score
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("episode_hash") or "")):
            raise ValueError("Validation gates require ordered completed learning episodes with native binary scores")
        history[int(score)].append((index, task, row))
    if protocol == HISTORICAL_SEQUENTIAL_PROTOCOL:
        chosen = _allocate_gates(panels, history, seed)
        for gate in GATES:
            chosen[gate].sort(key=lambda p: digest([seed, "gate-order", gate, p[0]["id"]]))
    else:
        chosen = {gate: [(task, {"task_id": task["id"], "source": "fresh"}) for task in panels[gate]]
                  for gate in GATES}
    resolved, sampling = {"learning": list(panels["learning"])}, {}
    for gate in GATES:
        selected = chosen[gate]
        resolved[gate] = [task for task, _ in selected]
        entries = [entry for _, entry in selected]
        counts = Counter(entry["source"] for entry in entries)
        sampling[gate] = dict(entries=entries, counts={
            label: counts[label] for label in ("historical_failure", "historical_success", "fresh")})
    return {"panels": resolved, "sampling": sampling}
