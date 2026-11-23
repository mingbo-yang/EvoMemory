"""Register a small ALFWorld engineering panel, or run its bounded paired search."""
import argparse
import json
import os
from pathlib import Path



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    register = sub.add_parser("prepare-smoke", help="Freeze six disjoint official games; no model calls.")
    register.add_argument("--games", required=True, type=Path)
    register.add_argument("--metadata", required=True, type=Path)
    register.add_argument("--seed", default=42, type=int)
    register.add_argument("--exclude", action="append", default=[], type=Path,
                          help="Additional task-list JSON to exclude by game hash and group ID.")
    register.add_argument("--out", required=True, type=Path)
    run = sub.add_parser("smoke", help="Run or resume one local engineering batch, up to 26 native episodes.")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--model-config", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--exercise-pair", action="store_true", help="Exercise the paired path as an engineering fixture.")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare-smoke":
        from .panel import save_registration, smoke_manifest
        groups, games = set(), set()
        for path in args.exclude:
            for row in json.loads(path.read_text()):
                groups.add(row["group_id"])
                games.add(row["game_sha256"])
        manifest = smoke_manifest(args.games, args.metadata, args.seed, groups, games)
        save_registration(args.out, manifest)
        print("Registered 2 learning, 2 primary and 2 meta tasks in", args.out)
        return
    args.out = args.out.resolve()
    cfg = json.loads(args.model_config.read_text())
    manifest = json.loads(args.manifest.read_text())
    # Bind the environment before importing any native runners with module-level defaults.
    os.environ.update(EVOMEMORY_BASE_URL=cfg["base_url"], EVOMEMORY_MODEL_NAME=cfg["model_name"],
                      EVOMEMORY_MODEL_PATH=cfg["tokenizer_path"],
                      EVOMEMORY_MODEL_MANIFEST=str(Path(cfg["checkpoint_manifest"]).resolve()))
    from .runner import run_smoke
    run_smoke(args.out, cfg, manifest, args.exercise_pair)


if __name__ == "__main__":
    main()
