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
    prepare = sub.add_parser("prepare-coevolution", help="Register fixed-width batches for mixed or single-family ALFWorld evolution")
    prepare.add_argument("--protocol", choices=("10x50", "6x30", "5x23", "mixed-10x60"), default="10x50")
    prepare.add_argument("--holdout-size", type=int, choices=(30, 50), help="Default size of arm selection and final test for this protocol")
    prepare.add_argument("--final-size", type=int, choices=(30, 50), help="Override only the independent final test size")
    prepare.add_argument("--games", type=Path, required=True)
    prepare.add_argument("--metadata", type=Path, required=True)
    prepare.add_argument("--family", choices=("heat", "clean", "cool", "two"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--exclude-final", type=Path, action="append", default=[],
                         help="Exclude all groups/content of this prior final panel across the new registration")
    prepare.add_argument("--exclude-holdout", type=Path, action="append", default=[],
                         help="Exclude previous instances throughout, and their whole groups from selection and final")
    prepare.add_argument("--out", type=Path, required=True)
    for command in ("smoke", "campaign", "campaign-stage", "coevolve-arm", "coevolve-stage", "coevolve-final"):
        run = sub.add_parser(command)
        run.add_argument("--manifest", required=True, type=Path)
        run.add_argument("--model-config", required=True, type=Path)
        run.add_argument("--out", required=True, type=Path)
        if command == "smoke":
            run.add_argument("--exercise-pair", action="store_true")
        if command in ("coevolve-arm", "coevolve-stage"):
            run.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
        if command == "coevolve-arm":
            run.add_argument("--first-round-only", action="store_true", help="Run only this arm's first registered evolution round")
        if command == "coevolve-final":
            run.add_argument("--winner", required=True, type=Path)
        if command in ("campaign-stage", "coevolve-stage"):
            run.add_argument("--stage", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare-coevolution":
        from .coevolution_panel import coevolution_manifest, exclusion_rows
        from .panel import COEVOLUTION_PROTOCOL, COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL, save_registration
        if args.protocol == "mixed-10x60":
            if args.family or args.exclude_final or args.exclude_holdout or args.holdout_size or args.final_size:
                raise ValueError("The mixed protocol fixes all six families, official split policy, and 30/100 holdouts")
            from .mixed_panel import mixed_manifest
            manifest = mixed_manifest(args.games, args.metadata, args.seed)
            save_registration(args.out, manifest)
            print(f"Registered {manifest['rounds']} balanced six-family rounds in {args.out}")
            return
        exclude = [t for p in args.exclude_final for t in exclusion_rows(json.loads(p.read_bytes()), True)]
        holdout = [t for p in args.exclude_holdout for t in exclusion_rows(json.loads(p.read_bytes()), False)]
        protocol = {"10x50": COEVOLUTION_PROTOCOL, "6x30": COEVOLUTION_6X30_PROTOCOL,
                    "5x23": COEVOLUTION_HEAT_5X23_PROTOCOL}[args.protocol]
        manifest = coevolution_manifest(args.games, args.metadata, args.seed, args.family or "heat", exclude, holdout,
                                        protocol=protocol, holdout_size=args.holdout_size, final_size=args.final_size)
        save_registration(args.out, manifest)
        print(f"Registered {manifest['rounds']} complete rounds of {manifest['batch_size']} games per phase in {args.out}")
        return
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
    if args.command.startswith("coevolve-"):
        from .coevolution_campaign import run_arm, run_arm_stage, run_champion
        if args.command == "coevolve-arm":
            if args.first_round_only:
                run_arm(args.out, args.manifest, args.model_config, args.arm, first_round_only=True)
            else:
                run_arm(args.out, args.manifest, args.model_config, args.arm)
        elif args.command == "coevolve-stage":
            run_arm_stage(args.out, cfg, manifest, args.arm, args.stage)
        else:
            run_champion(args.out, cfg, manifest, args.winner)
        return
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
