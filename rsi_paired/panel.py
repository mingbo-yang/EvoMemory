"""Prepare the six ALFWorld instances used for engineering validation."""
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from benchmarks.alfworld.panel import FAMILIES
from evomem.common import digest, file_hash
from scripts.prepare_assets import GAME_SHA


def smoke_manifest(games_path, metadata_path, seed, excluded_groups=(), excluded_games=()):
    if file_hash(games_path) != GAME_SHA:
        raise ValueError("ALFWorld game archive differs from the pinned release")
    blocked_groups = set(excluded_groups)
    blocked_games = set(excluded_games)
    by_group = defaultdict(list)
    with ZipFile(games_path) as games, ZipFile(metadata_path) as metadata:
        import hashlib
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
                    trajectory = json.loads(metadata.read(str(Path(member).with_name("traj_data.json"))))
                    if FAMILIES.get(trajectory["task_type"]) != family:
                        raise ValueError("Game family does not match official metadata")
                    tasks.append(dict(id="rsi-alfworld/" + digest(member)[:18], environment="alfworld",
                                      family=family, group_id=group,
                                      seed=int(digest([seed, "environment", member])[:8], 16) % 2147483647,
                                      room=int(group.rsplit("-", 1)[-1]),
                                      archive_member=member, game_sha256=game_hash, split="train",
                                      goal="Complete the task stated in the initial observation."))
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
    import hashlib
    from collections import Counter

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
                member = Path(name)
                trajectory = json.loads(metadata.read(str(member.with_name("traj_data.json"))))
                if FAMILIES.get(trajectory["task_type"]) != family:
                    raise ValueError("Game family does not match official metadata")
                tasks.append(dict(id="rsi-alfworld/" + digest(name)[:18], environment="alfworld",
                                  family=family, group_id=group,
                                  seed=int(digest([seed, "environment", name])[:8], 16) % 2147483647,
                                  room=int(group.rsplit("-", 1)[-1]), archive_member=name,
                                  game_sha256=raw, split=member.parts[1],
                                  goal="Complete the task stated in the initial observation."))
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
