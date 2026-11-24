"""Run balanced ALFWorld evolution, confirmation and frozen evaluation."""
from collections import Counter, defaultdict
import os
from pathlib import Path
import subprocess
import sys

from evomem.common import file_hash

from .audit import Audit, audit_run
from .panel import FAMILIES, save_registration
from .runner import initial_state, open_stage
from .workflow import _Journal, _lock, _read, _seed, _validate_score, digest


def _campaign_tasks(manifest):
    batches = manifest["batches"]
    confirmation, final = manifest["confirmation"], manifest["final"]
    kinds = set(FAMILIES.values())

    def balanced(rows, count):
        if Counter(t["family"] for t in rows) != dict.fromkeys(kinds, count):
            raise ValueError("Registered tasks are not balanced across the six ALFWorld families")

    if (manifest["protocol"] != "rsi_original_parent_paired_v2"
            or manifest["scope"] != "formal_alfworld_240" or len(batches) != 4):
        raise ValueError("Expected the registered four-batch ALFWorld protocol")
    evolution = list(confirmation)
    balanced(confirmation, 4)
    balanced(final, 20)
    for batch in batches:
        if set(batch) != {"learning", "primary", "meta"}:
            raise ValueError("Invalid panel roles")
        for phase, count in (("learning", 2), ("primary", 1), ("meta", 1)):
            balanced(batch[phase], count)
            evolution.extend(batch[phase])
    if (len({r["group_id"] for r in evolution}) != len(evolution)
            or {r["group_id"] for r in evolution} & {r["group_id"] for r in final}):
        raise ValueError("Development and final game groups overlap or development repeats a group")
    for field in ("id", "game_sha256", "archive_member"):
        if len({t[field] for t in evolution + final}) != 240:
            raise ValueError(f"Repeated registered game {field}")
    for rows, allowed in ((evolution, {"train"}), (final, {"valid_seen", "valid_unseen"})):
        for t in rows:
            if t["split"] not in allowed or Path(t["archive_member"]).parts[1] != t["split"]:
                raise ValueError("Wrong official asset split")
    return evolution


def _checked_state(batch_root):
    receipt = _read(batch_root / "batch/commit.json")
    if digest(receipt["state"]) != receipt["state_hash"]:
        raise ValueError("Previous stage state digest changed")
    result = audit_run(batch_root)
    if result["status"] not in {"verified", "verified_fallback"} or result["findings"]:
        raise ValueError(f"Evolution batch evidence did not verify: {batch_root}")
    return receipt["state"]


def _confirmation_input(root, manifest):
    """Use the core panel plus registered meta games that never reached comparison."""
    stages = [root / f"evolution-{i+1:02d}" for i in range(len(manifest["batches"]))]
    states = [initial_state()] + [_checked_state(stage) for stage in stages]
    supplemental = [i for i, stage in enumerate(stages)
                    if not (stage / "batch/meta_results.json").is_file()]
    tasks = [*manifest["confirmation"]]
    for i in supplemental:
        tasks.extend(manifest["batches"][i]["meta"])
    save_registration(root / "confirm/panel.json", dict(
        tasks=tasks, supplemental_batch_indices=supplemental,
        checkpoint_hashes=[digest(state) for state in states]))
    return tasks, states


def _stage_spec(root, manifest, name):
    if name == "confirm":
        return _confirmation_input(root, manifest)
    if name == "final":
        frozen = _read(root / "frozen.json")
        confirmed = _read(root / "confirm/summary.json")
        if frozen["confirmation_hash"] != digest(confirmed) or frozen["checkpoint"] != confirmed["selected"]:
            raise ValueError("Frozen deployment no longer matches confirmation")
        if frozen["test_panel_hash"] != digest(manifest["final"]):
            raise ValueError("Frozen deployment no longer matches final panel")
        coverage = _read(root / "development_coverage.json")
        if frozen["development_coverage_hash"] != digest(coverage):
            raise ValueError("Frozen deployment no longer matches the complete development evidence")
        return manifest["final"], [frozen["checkpoint"]]
    raise ValueError("Unknown panel stage")


def _native_panel(root, adapter, states, tasks, seed):
    journal = _Journal(root / "batch", adapter, 2)
    seeds = [_seed(seed, "native", t["id"]) for t in tasks]
    programs = {}
    results = []
    for index, state in enumerate(states):
        source, policy = state["mechanism"], state["evolver"]
        key = digest([adapter.fingerprint, source, tasks, seeds, "empty-family-memory"])
        if key not in programs:
            def evaluate():
                rows = adapter.evaluate(source, {}, tasks, seeds)
                if len(rows) != len(tasks):
                    raise ValueError("Incomplete frozen panel")
                for row, task in zip(rows, tasks):
                    _validate_score(row, task)
                return rows

            programs[key] = journal.job("evaluation/" + key, key, evaluate)
        rows = programs[key]
        categories = defaultdict(Counter)
        for task, row in zip(tasks, rows):
            for k in (task["family"], "split/" + task["split"], "all"):
                c = categories[k]
                c["attempted"] += 1
                c["succeeded"] += row["score"]
                c["verified_native"] += row["status"] == "completed"
                c["mechanism_failures"] += row["status"] == "method_eval_failure"
        results.append(dict(index=index, mechanism_hash=digest(source), policy_hash=digest(policy),
                            evaluation_key=key, counts={k: dict(v) for k, v in categories.items()},
                            episodes=[dict(task_id=r["task_id"], score=r["score"], native_score=r["native_score"],
                                           status=r["status"], episode_hash=r["episode_hash"]) for r in rows]))

    audit = Audit(root)
    audit.costs_and_wire()
    for state, item in zip(states, results):
        key = item["evaluation_key"]
        audit.bind_panel(key, tasks, seeds, state["mechanism"], {}, programs[key])
    if audit.errors or audit.native_missing_wire or set(audit.native_rows) != audit.native_verified:
        raise ValueError(f"Frozen panel does not have complete bound evidence: {audit.errors}")
    return results, audit.report["costs"]


def _link_learning(root, source, state, prior_count):
    for row in state["ledger"][prior_count:]:
        filename = row["episode_hash"] + ".json"
        path, destination = source / "native/learning" / filename, root / "learning" / filename
        try:
            os.link(path, destination)
        except FileExistsError:
            if path.read_bytes() != destination.read_bytes():
                raise ValueError("Shared learning archive has conflicting contents")


def run_stage(root, cfg, manifest, name):
    root, stage = Path(root), Path(root) / name
    _campaign_tasks(manifest)
    if _read(root / "registration.json") != manifest:
        raise ValueError("Campaign manifest differs from the frozen registration")
    if name.startswith("evolution-"):
        index = int(name.rsplit("-", 1)[1]) - 1
        if index not in range(4) or name != f"evolution-{index+1:02d}":
            raise ValueError("Unknown evolution stage")
        start = initial_state() if index == 0 else _checked_state(root / f"evolution-{index:02d}")
        panels = manifest["batches"][index]
        tasks = [t for rows in panels.values() for t in rows]
        save_registration(stage / "batch/starting_state.json", start)
        from .workflow import run_batch
        with open_stage(stage, cfg, manifest, tasks, learning_archive=root / "learning") as adapter:
            state = run_batch(stage / "batch", state=start, panels=panels, adapter=adapter,
                              seed=_seed(manifest["seed"], "batch", index))
        _checked_state(stage)
        _link_learning(root, stage, state, len(start["ledger"]))
    else:
        tasks, states = _stage_spec(root, manifest, name)
        # ALMA's initializer expects a training asset. The registered held-out
        # task list still supplies every evaluated environment reset explicitly.
        staged = tasks if name == "confirm" else [manifest["batches"][0]["learning"][0], *tasks]
        with open_stage(stage, cfg, manifest, staged) as adapter:
            results, costs = _native_panel(stage, adapter, states, tasks, manifest["seed"])
        winner = max(results, key=lambda result: result["counts"]["all"]["succeeded"])
        selection = dict(checkpoint_index=winner["index"], mechanism=states[winner["index"]]["mechanism"],
                         evolver=states[winner["index"]]["evolver"])
        if name == "final":
            selection = states[0]
        save_registration(stage / "summary.json", dict(selected=selection, results=results, costs=costs,
                                                        tasks_hash=digest(tasks)))


def _covered_development(root, manifest, confirmation):
    """Verify instance use before releasing the independent final panel."""
    covered = set()

    def include(rows):
        covered.update(row["task_id"] for row in rows)

    for i, batch in enumerate(manifest["batches"]):
        stage = root / f"evolution-{i+1:02d}" / "batch"
        for j, _ in enumerate(batch["learning"]):
            receipt = _read(stage / f"jobs/learning/{j:03d}/result.json")
            include([receipt["outcome"]["value"][1]])
        include(_read(stage / "primary_results.json")["parent"]["evaluation"])
        meta = stage / "meta_results.json"
        if meta.is_file():
            include(_read(meta)["parent"]["evaluation"])
    for result in confirmation["results"]:
        include(result["episodes"])

    registered = _campaign_tasks(manifest)
    if covered != {task["id"] for task in registered}:
        raise ValueError("Native receipts do not cover the registered 120 development instances")
    coverage = dict(task_ids=sorted(covered), distinct_tasks=len(covered),
                    families=dict(Counter(task["family"] for task in registered)),
                    confirmation_hash=digest(confirmation))
    save_registration(root / "development_coverage.json", coverage)
    return coverage


def run_campaign(root, manifest_path, cfg_path):
    root = Path(root).resolve()
    manifest = _read(Path(manifest_path))
    _campaign_tasks(manifest)
    with _lock(root):
        save_registration(root / "configuration.json", dict(
            model_config_hash=digest(_read(Path(cfg_path))),
            source_hashes={p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}))
        save_registration(root / "registration.json", manifest)
        (root / "learning").mkdir(exist_ok=True)
        for stage in [f"evolution-{i+1:02d}" for i in range(4)] + ["confirm", "final"]:
            if stage == "final":
                confirmed = _read(root / "confirm/summary.json")
                coverage = _covered_development(root, manifest, confirmed)
                save_registration(root / "frozen.json",
                                  dict(checkpoint=confirmed["selected"], confirmation_hash=digest(confirmed),
                                       development_coverage_hash=digest(coverage),
                                       test_panel_hash=digest(manifest["final"]), initial_task_memory="empty",
                                       task_memory_updates="online_same_family", evolve_during_test=False))
            print(f"Campaign stage {stage}", flush=True)
            subprocess.run([sys.executable, "-m", "rsi_paired", "campaign-stage",
                            "--manifest", str(Path(manifest_path).resolve()), "--model-config",
                            str(Path(cfg_path).resolve()), "--out", str(root), "--stage", stage], check=True)
        final = _read(root / "final/summary.json")
        frozen = _read(root / "frozen.json")
        if final["selected"] != frozen["checkpoint"] or len(final["results"][0]["episodes"]) != 120:
            raise ValueError("Final summary does not cover the frozen 120 tasks")
        save_registration(root / "complete.json", dict(registration_hash=digest(manifest),
                          frozen_hash=digest(frozen), final_hash=digest(final), tasks=240))
