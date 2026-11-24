"""Register a small ALFWorld engineering panel, or run its bounded paired search."""
import argparse
import json
import os
from pathlib import Path



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command, help_text in (("prepare-smoke", "Freeze six official games"),
                               ("prepare-formal", "Freeze 120 development and 120 final games")):
        register = sub.add_parser(command, help=help_text)
        register.add_argument("--games", required=True, type=Path)
        register.add_argument("--metadata", required=True, type=Path)
        register.add_argument("--seed", default=42, type=int)
        register.add_argument("--exclude", action="append", default=[], type=Path,
                              help="Exclude game hashes and group IDs listed in task-list JSON.")
        register.add_argument("--out", required=True, type=Path)
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
        from .panel import formal_manifest, save_registration, smoke_manifest
        groups, games = set(), set()
        for path in args.exclude:
            for row in json.loads(path.read_text()):
                groups.add(row["group_id"])
                games.add(row["game_sha256"])
        prepare = smoke_manifest if args.command == "prepare-smoke" else formal_manifest
        manifest = prepare(args.games, args.metadata, args.seed, groups, games)
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
        from .campaign import run_campaign, run_stage
        if args.command == "campaign":
            run_campaign(args.out, args.manifest, args.model_config)
        else:
            run_stage(args.out, cfg, manifest, args.stage)


if __name__ == "__main__":
    main()
