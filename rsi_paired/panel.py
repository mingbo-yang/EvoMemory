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
