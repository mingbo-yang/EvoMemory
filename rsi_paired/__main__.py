"""Register and run paired memory evolution on ALFWorld."""
import argparse
import json
import os
from pathlib import Path

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command, help_text in (("prepare-smoke", "Freeze six official games"),
                               ("prepare-formal", "Register two sequential task families and two independent 100-game tests")):
        register = sub.add_parser(command, help=help_text)
        register.add_argument("--games", required=True, type=Path)
        register.add_argument("--metadata", required=True, type=Path)
        register.add_argument("--seed", default=42, type=int)
        register.add_argument("--exclude", action="append", default=[], type=Path,
                              help="Exclude group IDs and game hashes from a task list or a v3 formal registration.")
        register.add_argument("--out", required=True, type=Path)
        if command == "prepare-formal":
            register.add_argument("--families", nargs=2, default=("simple", "two"),
                                  metavar=("FIRST", "SECOND"), help="Execution order; both must have sufficient games.")
        else:
            register.add_argument("--v3", action="store_true", help="Exercise the v3 decision and scoring rules.")
    for command in ("smoke", "campaign", "campaign-stage"):
        run = sub.add_parser(command)
        run.add_argument("--manifest", required=True, type=Path)
        run.add_argument("--model-config", required=True, type=Path)
        run.add_argument("--out", required=True, type=Path)
        if command == "smoke":
            run.add_argument("--exercise-pair", action="store_true")
        if command == "campaign-stage":
            run.add_argument("--stage", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ("prepare-smoke", "prepare-formal"):
        from .panel import (SEQUENTIAL_PROTOCOL, sequential_manifest, save_registration,
                            smoke_manifest, validate_sequential_manifest)
        groups, games = set(), set()
        for path in args.exclude:
            rows = json.loads(path.read_text())
            if isinstance(rows, dict):
                validate_sequential_manifest(rows)
                families = rows["families"]
                rows = []
                for plan in families.values():
                    rows.extend(plan["final"])
                    for batch in plan["batches"]:
                        for pool in batch.values():
                            rows.extend(pool)
            for row in rows:
                groups.add(row["group_id"])
                games.add(row["game_sha256"])
        if args.command == "prepare-smoke":
            manifest = smoke_manifest(args.games, args.metadata, args.seed, groups, games)
            if args.v3:
                manifest["protocol"] = SEQUENTIAL_PROTOCOL
        else:
            manifest = sequential_manifest(args.games, args.metadata, args.seed, families=args.families,
                                           excluded_groups=groups, excluded_games=games)
        save_registration(args.out, manifest)
        print("Registered", manifest["scope"], "tasks in", args.out)
        return
    args.out = args.out.resolve()
    cfg = json.loads(args.model_config.read_text())
    manifest = json.loads(args.manifest.read_text())
    # Bind the environment before importing any native runners with module-level defaults.
    os.environ.update(EVOMEMORY_BASE_URL=cfg["base_url"], EVOMEMORY_MODEL_NAME=cfg["model_name"],
                      EVOMEMORY_MODEL_PATH=cfg["tokenizer_path"],
                      EVOMEMORY_MODEL_MANIFEST=str(Path(cfg["checkpoint_manifest"]).resolve()))
    if args.command == "smoke":
        from .runner import run_smoke
        run_smoke(args.out, cfg, manifest, args.exercise_pair)
    else:
        if manifest.get("scope") == "formal_alfworld_sequential_families":
            from .family_campaign import run_campaign, run_stage
        else:
            from .campaign import run_campaign, run_stage
        if args.command == "campaign":
            run_campaign(args.out, args.manifest, args.model_config)
        else:
            run_stage(args.out, cfg, manifest, args.stage)


if __name__ == "__main__":
    main()
