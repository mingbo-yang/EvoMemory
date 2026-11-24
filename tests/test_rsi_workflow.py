"""Synthetic protocol checks. Scores here are fixtures, not experimental results."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsi_paired.workflow import BudgetExhausted, ContractError, InfrastructurePending, digest, run_batch


def state():
    return {"mechanism": "original", "evolver": {"schema": 1, "propose": "old"},
            "memory": {"learning": []}, "ledger": [], "history": []}


def panels():
    return {name: [{"id": f"{name}-{i}"} for i in range(2)] for name in ("learning", "primary", "meta")}


class Synthetic:
    fingerprint = "synthetic-v1"

    def __init__(self):
        self.calls = []
        self.evals = []
        self.scores = {"P2": 2, "N3": 2, "O1": 1}
        self.first_passes = {}
        self.triggers = True
        self.corrupt = False
        self.break_meta = False
        self.malform_decision = False

    def set_job(self, path):
        self.job = path

    def learn(self, mechanism, memory, task, seed):
        memory["learning"].append(task["id"])
        return memory, {"task_id": task["id"], "score": 0, "family": "cool"}

    def diagnose(self, mechanism, memory, ledger, history):
        return {"scope": "public"}

    def propose(self, policy, mechanism, packet, seed, previous, error):
        name = self.job.parent.parent.name
        index = int(self.job.parent.name)
        self.calls.append((name, index, mechanism, policy["propose"], digest(packet), seed, previous, error))
        source = name if index >= self.first_passes.get(name, 0) else name+"-bad"
        return {"source": source, "loaded_evolver_hash": digest(policy),
                "public_hash": "wrong" if self.corrupt else digest(packet)}

    def check(self, source):
        if source.endswith("-bad"):
            raise ContractError("Wrong Hook signature")

    def decide(self, policy, packet):
        if self.malform_decision:
            return {"should_update": False}
        return {"should_update": self.triggers, "reason": "bound evidence",
                "evidence_ids": ["post.review"], "target_components": ["propose"],
                "expected_effect": "more valid drafts"}

    def revise(self, policy, packet):
        return {**policy, "propose": "new"}

    def prepare(self, source, parent, memory, ledger):
        assert [s["task_id"] for s in ledger] == memory["learning"]
        return {"learning": memory["learning"], "prepared":source}

    def evaluate(self, source, memory, tasks, seeds):
        phase = tasks[0]["id"].split("-")[0]
        if phase == "meta" and self.break_meta:
            raise OSError("transient")
        assert "panel" not in memory
        memory["panel"] = phase
        self.evals.append((phase, source))
        count = self.scores.get(source, 0)
        if phase == "meta" and source == "P2":
            count = 1
        return [{"task_id": t["id"], "score": int(i < count)} for i,t in enumerate(tasks)]

    def review(self, source, records):
        return {"scope": "native"}


class PairedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = Synthetic()

    def execute(self, **kwargs):
        return run_batch(self.root, state=state(), panels=panels(), adapter=self.a, **kwargs)

    def test_same_mother_matched_public_evidence_and_real_slots(self):
        result = self.execute()
        event = result["history"][-1]
        self.assertEqual((event["primary_best"],event["old_best"],event["new_best"]),("P2","retained","N3"))
        self.assertEqual(result["mechanism"],"N3")
        self.assertEqual(result["evolver"]["propose"],"new")
        self.assertEqual(result["memory"],{"learning":["learning-0","learning-1"],"prepared":"N3"})
        self.assertEqual(len(self.a.calls),9)
        arms = self.a.calls[3:]
        self.assertEqual({x[2] for x in arms},{"original"})
        self.assertEqual({x[3] for x in arms[:3]},{"old"})
        self.assertEqual({x[3] for x in arms[3:]},{"new"})
        self.assertEqual(len({x[4] for x in arms}),1)
        self.assertNotEqual(arms[0][4],self.a.calls[0][4])
        self.assertEqual([x[5] for x in arms[:3]],[x[5] for x in arms[3:]])
        self.assertEqual(len({x[5] for x in arms[:3]}),3)
        self.assertEqual(len(self.a.evals),12)
        self.assertEqual(result,self.execute())
        self.assertEqual(len(self.a.calls),9)
        with self.assertRaisesRegex(ValueError,"Frozen"):
            self.execute(seed=1)

    def test_third_repair_passes_or_exhausts_without_replacement(self):
        self.a.first_passes = {"P1":3,"O2":4}
        self.execute()
        p = json.loads((self.root/"slots/P1.json").read_text())
        o = json.loads((self.root/"slots/O2.json").read_text())
        self.assertTrue(p["valid"])
        self.assertFalse(o["valid"])
        self.assertEqual(len(p["drafts"]),4)
        self.assertEqual(len(o["drafts"]),4)
        self.assertIsNone(json.loads((self.root/"meta_results.json").read_text())["O2"]["success_count"])
        self.assertEqual([x[1] for x in self.a.calls if x[0]=="N1"],[0])

    def test_tie_parent_fallback_and_deduplicated_score(self):
        self.a.scores.clear()
        value = self.execute()
        self.assertEqual(value["history"][0]["primary_best"],"parent")
        self.assertEqual(value["history"][0]["old_best"],"retained")
        self.assertEqual(value["evolver"],state()["evolver"])
        self.assertEqual(self.a.evals.count(("meta","original")),1)

    def test_old_search_can_deploy_and_original_state_is_immutable(self):
        self.a.scores.update(N3=0,O1=2)
        original = state()
        value = run_batch(self.root,state=original,panels=panels(),adapter=self.a)
        self.assertEqual(value["mechanism"],"O1")
        self.assertEqual(value["evolver"],original["evolver"])
        self.assertEqual(original,state())

    def test_no_trigger_and_budget_reservation(self):
        self.a.triggers = False
        a = self.execute()
        self.assertEqual(a["history"][-1]["status"],"not_triggered")
        self.assertEqual(len(self.a.calls),3)
        self.a.triggers = True
        b = run_batch(self.root/"budget",state=state(),panels=panels(),adapter=self.a,budget={"episodes":10})
        self.assertEqual(b["history"][-1]["status"],"budget_blocked")
        with self.assertRaises(BudgetExhausted):
            run_batch(self.root/"small",state=state(),panels=panels(),adapter=self.a,budget={"episodes":9})

    def test_no_tampered_packet_or_partial_native_measurements(self):
        self.a.corrupt = True
        with self.assertRaisesRegex(InfrastructurePending,"different Evolver"):
            self.execute()
        self.assertFalse((self.root/"commit.json").exists())

    def test_retry_preserves_six_slots_and_seals(self):
        self.a.break_meta = True
        with self.assertRaises(InfrastructurePending):
            self.execute()
        self.assertEqual(len(self.a.calls),9)
        self.assertTrue((self.root/"paired_seal.json").is_file())
        self.a.break_meta = False
        self.assertEqual(self.execute()["mechanism"],"N3")
        self.assertEqual(len(self.a.calls),9)

    def test_meta_exhaustion_never_accepts_partial_arm(self):
        self.a.break_meta = True
        with self.assertRaises(InfrastructurePending):
            self.execute(budget={"infrastructure_attempts":1})
        result = self.execute(budget={"infrastructure_attempts":1})
        self.assertEqual(result["mechanism"],"P2")
        self.assertEqual(result["history"][-1]["status"],"meta_incomplete")

    def test_bad_trigger_is_recorded_separately(self):
        self.a.malform_decision = True
        result = self.execute()
        self.assertEqual(result["history"][-1]["status"],"decision_error")
        self.assertEqual(len(self.a.calls),3)

    def test_forced_engineering_record_declares_its_origin(self):
        self.a.triggers = False
        event = self.execute(force_pair=True)["history"][-1]
        self.assertTrue(event["forced_pair"])
        self.assertFalse(event["decision"]["should_update"])
        self.assertEqual(event["status"],"paired_complete")

    def test_malformed_self_revision_cannot_change_guard(self):
        self.a.revise = lambda policy,packet: {"propose":"new","schema":2}
        self.assertEqual(self.execute()["history"][0]["status"],"invalid_self_revision")

    def test_subsequent_batch_loads_actual_winner_and_policy(self):
        committed = self.execute()
        with self.assertRaisesRegex(ValueError, "learning ledger"):
            run_batch(self.root/"overlap", state=committed, panels=panels(), adapter=self.a)
        future = {name: [{**row, "id": row["id"] + "-next"} for row in group]
                  for name, group in panels().items()}
        self.a.triggers = False
        self.a.calls.clear()
        result = run_batch(self.root/"next", state=committed, panels=future, adapter=self.a)
        self.assertTrue(all(call[2:4] == ("N3", "new") for call in self.a.calls))
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(len(result["ledger"]), 4)

    def test_cross_split_game_duplicate_rejected(self):
        tasks = panels()
        for bucket in ("primary","meta"):
            tasks[bucket][0]["game_sha256"] = "shared"
        with self.assertRaisesRegex(ValueError,"Repeated task"):
            run_batch(self.root,state=state(),panels=tasks,adapter=self.a)
