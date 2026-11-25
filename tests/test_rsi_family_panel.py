"""Synthetic official-layout archives and learning receipts for the sequential protocol."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from evomem.common import digest, file_hash
from rsi_paired import panel


@pytest.fixture(scope="module")
def archives(tmp_path_factory):
    root = tmp_path_factory.mktemp("family-panels")
    games_path, metadata_path = root / "games.zip", root / "meta.zip"
    with ZipFile(games_path, "w") as games, ZipFile(metadata_path, "w") as metadata:
        for kind in ("pick_and_place_simple", "pick_two_obj_and_place"):
            def write(split, room, trial, contents=None):
                group = f"{kind}-synthetic-{room}"
                member = Path("json_2.1.1") / split / group / str(trial) / "game.tw-pddl"
                games.writestr(str(member), contents or str(member))
                metadata.writestr(str(member.with_name("traj_data.json")), json.dumps({"task_type": kind}))
            for i in range(610):
                write("train", i // 3, i)
            for i in range(22):
                write("valid_seen", 1000 + i, i)
            for i in range(18):
                write("valid_unseen", 2000 + i, i)
            write("valid_train", 0, "official-pilot")
            write("train", 1000, "overlap-group")
            write("train", 2100, "content-match",
                  str(Path("json_2.1.1/valid_seen") / f"{kind}-synthetic-1000" / "0/game.tw-pddl"))
    return games_path, metadata_path


@pytest.fixture
def manifest(archives, monkeypatch):
    games, metadata = archives
    monkeypatch.setattr(panel, "GAME_SHA", file_hash(games))
    return panel.sequential_manifest(games, metadata, 41)


def test_static_panel_complete_groups_and_content_isolation(manifest, archives):
    assert manifest["family_order"] == ["simple", "two"]
    assert manifest == panel.sequential_manifest(*archives, 41)
    for family, plan in manifest["families"].items():
        learning = [t for b in plan["batches"] for t in b["learning"]]
        reserves = [t for b in plan["batches"] for role in panel.GATES for t in b[role]]
        assert (len(learning), len(reserves), len(plan["final"])) == (180, 360, 100)
        assert plan["final_by_official_split"] == {"valid_unseen": 18, "valid_seen": 22, "train": 60}
        dev = learning + reserves
        assert {t["group_id"] for t in dev}.isdisjoint(t["group_id"] for t in plan["final"])
        assert len({t["id"] for t in dev + plan["final"]}) == 640
        assert len({t["game_sha256"] for t in dev + plan["final"]}) == 640
        assert all(t["family"] == family and t["room"] not in (0, 1000, 2100) for t in dev)
    first = manifest["families"]["simple"]["final"][0]
    blocked = panel.sequential_manifest(*archives, 41, excluded_games=[first["game_sha256"]])
    for family, plan in blocked["families"].items():
        assert all(t["game_sha256"] != first["game_sha256"] for t in plan["final"])


def test_registration_rejects_changed_ids_and_cross_split_groups(manifest):
    m = deepcopy(manifest)
    b = m["families"]["two"]["batches"][0]
    b["primary"][0] = b["learning"][0]
    with pytest.raises(ValueError, match="Repeated"):
        panel.validate_sequential_manifest(m)
    m = deepcopy(manifest)
    plan = m["families"]["simple"]
    dev = plan["batches"][0]["primary"][0]
    index = next(i for i, t in enumerate(plan["final"]) if t["split"] == "train")
    plan["final"][index] = dev
    with pytest.raises(ValueError, match="Final game group"):
        panel.validate_sequential_manifest(m)
    m = deepcopy(manifest)
    m["families"]["two"]["final"][0]["family"] = "simple"
    with pytest.raises(ValueError, match="registered family"):
        panel.validate_sequential_manifest(m)


def _learning_rows(tasks, labels):
    return [dict(task_id=task["id"], family=task["family"], phase="learning", status="completed",
                 native_score=score, score=score, episode_hash=digest([task["id"], score]))
            for task, score in zip(tasks, labels, strict=True)]


def test_fresh_gates_use_only_their_registered_order_independent_of_learning(manifest):
    first, current = manifest["families"]["two"]["batches"][:2]
    learned = first["learning"] + current["learning"]
    for seed, scores in [(19, [0] * 60), (63, [1] * 60), (19, [0, 1] * 30)]:
        resolved = panel.resolve_family_panels(current, learned, _learning_rows(learned, scores), seed)
        assert resolved["panels"] == current
        for gate in panel.GATES:
            sampling = resolved["sampling"][gate]
            assert sampling["entries"] == [{"task_id": t["id"], "source": "fresh"} for t in current[gate]]
            assert sampling["counts"] == dict(fresh=30, historical_success=0, historical_failure=0)


@pytest.mark.parametrize("labels,successes,failures", [
    ([0, 1] * 30, [6, 6], [6, 6]),
    ([0] * 60, [0, 0], [6, 6]),
    ([1] * 59 + [0], [6, 6], [0, 1]),
])
def test_resolve_fair_disjoint_gates_and_fill_shortages(manifest, labels, successes, failures):
    first, current = manifest["families"]["simple"]["batches"][:2]
    learned = first["learning"] + current["learning"]
    ledger = _learning_rows(learned, labels)
    result = panel.resolve_family_panels(current, learned, ledger, 63, protocol=panel.HISTORICAL_SEQUENTIAL_PROTOCOL)
    assert result == panel.resolve_family_panels(current, learned, ledger, 63,
                                                 protocol=panel.HISTORICAL_SEQUENTIAL_PROTOCOL)
    assert sorted(v["counts"]["historical_success"] for v in result["sampling"].values()) == successes
    assert sorted(v["counts"]["historical_failure"] for v in result["sampling"].values()) == failures
    p, m = ([t["id"] for t in result["panels"][g]] for g in panel.GATES)
    assert len(set(p + m)) == 60 and result["panels"]["learning"] == current["learning"]
    prior_ids = {r["id"] for r in learned}
    rounds = set()
    for gate in panel.GATES:
        entries = result["sampling"][gate]["entries"]
        assert len(entries) == 30 and [t["id"] for t in result["panels"][gate]] == [e["task_id"] for e in entries]
        assert Counter(e["source"] for e in entries) == Counter(result["sampling"][gate]["counts"])
        fresh = {t["id"] for t in current[gate]}
        for e in entries:
            if e["source"] == "fresh":
                assert e["task_id"] in fresh - prior_ids
            else:
                row = ledger[e["learning_index"]]
                assert e["task_id"] == row["task_id"] and e["episode_hash"] == row["episode_hash"]
                assert e["source"] == ("historical_success" if row["native_score"] else "historical_failure")
                rounds.add(e["learning_round"])
    assert rounds == {1, 2}


@pytest.mark.parametrize("field,value", [("phase", "panel"), ("family", "two"), ("native_score", None),
                                       ("score", False), ("status", "method_eval_failure"), ("episode_hash", "broken")])
def test_unverified_or_non_learning_outcomes_cannot_seed_history(manifest, field, value):
    batch = manifest["families"]["simple"]["batches"][0]
    rows = _learning_rows(batch["learning"], [0] * 30)
    rows[-1][field] = value
    with pytest.raises(ValueError, match="completed learning"):
        panel.resolve_family_panels(batch, batch["learning"], rows, 9)


def test_exhausted_or_inseparable_training_pools_fail_before_execution():
    with pytest.raises(ValueError, match="development games"):
        panel._holdout_groups([("one", "a", "h")], 1, 1, 42, "simple")
    with pytest.raises(ValueError, match="disjoint"):
        panel._holdout_groups([("one", "a", "h"), ("one", "b", "j")], 1, 1, 42, "simple")


def test_v3_budget_and_two_fresh_reserves_are_preregistered(manifest):
    assert manifest["protocol"] == panel.SEQUENTIAL_PROTOCOL == "rsi_original_parent_paired_v3_6x30_fresh"
    historical = dict(manifest, protocol="rsi_original_parent_paired_v3_6x30")
    assert historical["protocol"] == panel.HISTORICAL_SEQUENTIAL_PROTOCOL
    panel.validate_sequential_manifest(historical)
    assert manifest["budget"] == {
        "per_round": {"episodes": 390, "infrastructure_attempts": 2},
        "final_per_family": 200, "max_native_episodes_per_family": 2540, "max_native_episodes_total": 5080}
    for family in manifest["family_order"]:
        for batch in manifest["families"][family]["batches"]:
            assert {t["group_id"] for t in batch["primary"]}.isdisjoint(t["group_id"] for t in batch["meta"])
    for remove in (True, False):
        altered = deepcopy(manifest)
        if remove:
            del altered["budget"]
        else:
            altered["budget"]["per_round"]["episodes"] = 391
        with pytest.raises(ValueError, match="registered sequential runtime budget"):
            panel.validate_sequential_manifest(altered)


def test_crossing_fresh_siblings_are_repartitioned_or_rejected():
    rows = [("group-" + str(i % 30), i, str(i)) for i in range(60)]
    first, second = panel._partition_fresh(rows)
    assert len(first) == len(second) == 30
    assert {r[0] for r in first}.isdisjoint(r[0] for r in second)
    assert sorted(first + second) == sorted(rows)
    with pytest.raises(ValueError, match="disjoint fresh"):
        panel._partition_fresh([("inseparable", i, str(i)) for i in range(60)])


def test_siblings_may_fill_history_within_a_gate_but_never_across(manifest):
    batch = deepcopy(manifest["families"]["simple"]["batches"][0])
    history = batch["learning"]
    for i, task in enumerate(history):
        # Two groups each supply enough trials of both labels to fill an entire gate.
        # Their sibling fresh reserve is also legal only if assigned to the same side.
        group = batch[("primary", "meta")[i % 2]][0]["group_id"]
        task["group_id"] = group
    ledger = _learning_rows(history, [(i // 2) % 2 for i in range(len(history))])
    result = panel.resolve_family_panels(batch, history, ledger, 91, protocol=panel.HISTORICAL_SEQUENTIAL_PROTOCOL)
    primary, meta = (result["panels"][name] for name in panel.GATES)
    assert {t["group_id"] for t in primary}.isdisjoint(t["group_id"] for t in meta)
    for gate in panel.GATES:
        assert result["sampling"][gate]["counts"] == dict(historical_failure=6, historical_success=6, fresh=18)
        assert len({t["group_id"] for t in result["panels"][gate]}) < 30


def test_group_constrained_historical_shortfall_is_filled_without_duplicate_trials(manifest):
    batch = deepcopy(manifest["families"]["simple"]["batches"][0])
    for task in batch["learning"]:
        task["group_id"] = "one-historical-group"
    ledger = _learning_rows(batch["learning"], [0, 1] * 15)
    result = panel.resolve_family_panels(batch, batch["learning"], ledger, 91,
                                         protocol=panel.HISTORICAL_SEQUENTIAL_PROTOCOL)
    assert sorted(s["counts"]["fresh"] for s in result["sampling"].values()) == [18, 30]
    all_rows = result["panels"]["primary"] + result["panels"]["meta"]
    assert len({t["id"] for t in all_rows}) == 60
    assert {r["group_id"] for r in all_rows[:30]}.isdisjoint(r["group_id"] for r in all_rows[30:])
    # Detect invalid reserves before offering them to an Evolver, even if outcomes later permit a workaround.
    batch["meta"][0]["group_id"] = batch["primary"][0]["group_id"]
    with pytest.raises(ValueError, match="overlap between primary and meta"):
        panel.resolve_family_panels(batch, batch["learning"], ledger, 91, protocol=panel.HISTORICAL_SEQUENTIAL_PROTOCOL)
