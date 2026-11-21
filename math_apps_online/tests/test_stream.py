"""Online stream integration tests with controlled, non-research scores and decisions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from evomem.common import digest, read, write
from math_apps_online.online import run_stream
from phase1_codegen.program import BASELINE_SOURCE, Program


def task(item, family="algebra", domain="math"):
    ident = f"{domain}/{family}/test/{item}"
    return dict(id=ident, group_id=ident, seed=item, split="test", family=family,
                environment=domain, room=family, prompt_hash=digest(ident))


def make_snapshot(source, logs, item):
    return dict(program_hash=Program(source).hash, source_log_hash=digest(logs),
                environment=item["environment"], memory_scope="family", family=item["family"],
                memory_hash=digest([source, [ep["task_id"] for ep in logs]]))


def episode(source, item, score, label):
    return dict(task_id=item["id"], family=item["family"], environment=item["environment"],
                program_hash=Program(source).hash, native_reward_verified=True,
                error=None, score=score, synthetic=True, fixture_marker=label)


class Backend:
    """Native and paired-review doubles; only the stream controller is under test."""
    def __init__(self, first_scores, judgments=None):
        self.first_scores = first_scores
        self.judgments = judgments or {}
        self.evaluations = []
        self.repairs = []
        self.commits = []

    def evaluate(self, task, source, logs, snapshot, directory):
        self.evaluations.append((task["id"], source, deepcopy(logs), deepcopy(snapshot)))
        origin = make_snapshot(source, logs, task)
        if snapshot is not None:
            assert origin == snapshot
        return episode(source, task, self.first_scores[task["id"]], "first"), origin

    def repair(self, task, source, logs, snapshot, first_episode, records, directory):
        self.repairs.append((task["id"], list(records), list(logs), deepcopy(snapshot)))
        assert first_episode["score"] == 0
        assert snapshot == make_snapshot(source, logs, task)
        assert all(ep["task_id"] != task["id"] for ep in logs)
        selected, deployed, origin = first_episode, source, snapshot
        attempts, events = [], []
        accepted = solved = False
        for index, (score, decision) in enumerate(self.judgments[task["id"]], 1):
            marker = repr(f"{task['id']}-revision-{index}")
            candidate = ('def on_event(state, event, context):\n'
                         f'    return {{"state": state, "session": {{"marker": {marker}}}, "memory": None}}\n')
            accepted = decision == "adopt"
            solved |= score == 1
            attempts.append(dict(attempt=index, accepted=accepted, candidate_score=score,
                                 decision=decision, reason=f"fixture decision {index}", candidate_hash=Program(candidate).hash))
            events.append(dict(id=digest([task["id"], index])[:24], accepted=accepted,
                               observed_context=dict(environment=task["environment"], family=task["family"])))
            if accepted:
                deployed = candidate
                selected = episode(candidate, task, score, f"attempt-{index}")
                origin = make_snapshot(candidate, logs, task)
            if decision != "continue":
                break
        return dict(source=deployed, episode=selected, snapshot=origin, events=events,
                    attempts=attempts, accepted=accepted, solved=solved)

    def commit(self, task, source, logs, snapshot, episode, directory):
        assert snapshot == make_snapshot(source, logs, task)
        self.commits.append((task["id"], source, episode["fixture_marker"]))
        return make_snapshot(source, [*logs, episode], task)


class TestOnlineStream(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="evomemory-online-stream-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def run_items(self, items, backend, *, condition="cross_type_em", root=None):
        return run_stream(root=self.root if root is None else root, domain="math", tasks=items,
                          binding={"synthetic_test": True}, evaluate=backend.evaluate, repair=backend.repair,
                          commit=backend.commit, condition=condition)

    def test_first_success_and_restart_do_not_propose_or_regrade(self):
        t = task(1)
        b = Backend({t["id"]: 1})
        first = self.run_items([t], b)
        self.assertEqual((first["completed"], first["first_correct"], first["solved"], first["adopted"],
                          first["proposals"]), (1, 1, 1, 0, 0))
        self.assertEqual(first["correct_by_attempt"], [1])
        self.assertEqual((len(b.evaluations), len(b.repairs), len(b.commits)), (1, 0, 1))
        self.assertEqual(self.run_items([t], b), first)
        self.assertEqual((len(b.evaluations), len(b.repairs), len(b.commits)), (1, 0, 1))

    def test_fourth_attempt_adoption_and_solved_without_adoption(self):
        a, b = task(1), task(2)
        backend = Backend({a["id"]: 0, b["id"]: 0}, {
            a["id"]: [(None, "continue"), (0, "continue"), (0, "continue"), (1, "adopt")],
            b["id"]: [(1, "continue"), (0, "stop")],
        })
        summary = self.run_items([a, b], backend)
        self.assertEqual((summary["first_correct"], summary["solved"], summary["adopted"],
                          summary["proposals"]), (0, 2, 1, 6))
        self.assertEqual(summary["adoptions_by_attempt"], [0, 0, 0, 1])
        self.assertEqual(summary["stops_by_attempt"], [0, 1, 0, 0])
        self.assertEqual(summary["native_comparisons_by_attempt"], [1, 2, 1, 1])
        self.assertEqual(summary["proposals_by_attempt"], [2, 2, 1, 1])
        self.assertEqual(summary["correct_by_attempt"], [0, 1, 1, 1, 2])
        self.assertEqual(summary["solved_on_retry_without_adoption"], 1)
        self.assertEqual([x[2] for x in backend.commits], ["attempt-4", "first"])
        self.assertEqual(backend.commits[0][1], backend.commits[1][1])
        self.assertEqual([ep["fixture_marker"] for ep in backend.repairs[1][2]], ["attempt-4"])
        self.assertEqual(self.run_items([a, b], backend), summary)
        self.assertEqual(len(backend.repairs), 2)

    def test_each_family_owns_its_task_history_both_arms_inherit_global_code(self):
        a, b, c = task(1), task(2, "number_theory"), task(3)
        for arm in ("cross_type_em", "no_external_em"):
            with self.subTest(arm=arm):
                engine = Backend({a["id"]: 0, b["id"]: 0, c["id"]: 0}, {
                    a["id"]: [(1, "adopt")], b["id"]: [(0, "stop")], c["id"]: [(0, "stop")]})
                self.run_items([a, b, c], engine, condition=arm, root=self.root/arm)
                _, code_b, logs_b, snap_b = engine.evaluations[1]
                self.assertEqual(code_b, engine.commits[0][1])
                self.assertEqual(logs_b, [])
                self.assertIsNone(snap_b)
                self.assertEqual([e["task_id"] for e in engine.evaluations[2][2]], [a["id"]])
                shown = [[r["observed_context"]["family"] for r in call[1]] for call in engine.repairs]
                expected = [[], ["algebra"], ["algebra", "number_theory"]]
                self.assertEqual(shown, expected if arm == "cross_type_em" else [[], [], ["algebra"]])

    def test_reviewer_can_stop_after_first_proposal_and_preserve_deployment(self):
        a, b = task(1), task(2)
        backend = Backend({a["id"]: 0, b["id"]: 1}, {a["id"]: [(0, "stop"), (1, "adopt")]})
        out = self.run_items([a, b], backend)
        self.assertEqual((out["first_correct"], out["solved"], out["proposals"], out["adopted"]), (1, 1, 1, 0))
        self.assertEqual([v[1] for v in backend.commits], [BASELINE_SOURCE, BASELINE_SOURCE])
        self.assertEqual(backend.repairs[0][2], [])
        self.assertEqual(backend.evaluations[1][2][0]["score"], 0)

    def test_invalid_stream_and_changed_binding_rejected_before_new_observations(self):
        a = task(1)
        backend = Backend({a["id"]: 1})
        self.run_items([a], backend)
        with self.assertRaisesRegex(ValueError, "Changed registered"):
            self.run_items([a], backend, condition="no_external_em")
        for items in ([a, a], [a | {"split": "train"}], [task(1, domain="apps")]):
            with self.subTest(items=items), self.assertRaises((ValueError, RuntimeError)):
                self.run_items(items, backend)
        self.assertEqual(len(backend.evaluations), 1)

    def test_v1_plan_rejected_before_first_answer(self):
        a = task(1)
        backend = Backend({a["id"]: 1})
        write(self.root/"online_plan.json", dict(protocol="online-same-question-v1", budget=3))
        with self.assertRaisesRegex(ValueError, "Changed registered"):
            self.run_items([a], backend)
        self.assertFalse(backend.evaluations)

    def test_unfinished_or_impossible_decision_never_commits(self):
        a = task(1)
        for name, decisions in (("no-proposal", []), ("unfinished", [(0, "continue")]),
                                ("unsolved-adopt", [(0, "adopt")]), ("unmeasured-adopt", [(None, "adopt")])):
            backend = Backend({a["id"]: 0}, {a["id"]: decisions})
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_items([a], backend, root=self.root/name)
            self.assertFalse(backend.commits)

    def test_missing_earlier_item_does_not_repeat_work(self):
        a, b = task(1), task(2)
        backend = Backend({a["id"]: 1, b["id"]: 1})
        self.run_items([a, b], backend)
        plan = read(self.root/"online_plan.json")
        self.assertEqual(plan["protocol"], "online-same-question-v2")
        self.assertNotIn("budget", plan)
        (self.root/"items"/"000000"/"item.json").unlink()
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.run_items([a, b], backend)
        self.assertEqual(len(backend.evaluations), 2)

    def test_final_summary_binds_the_last_item_on_resume(self):
        a, b = task(1), task(2)
        backend = Backend({a["id"]: 1, b["id"]: 1})
        self.run_items([a, b], backend)
        summary_path = self.root/"summary.json"
        summary = read(summary_path)
        write(summary_path, dict(summary, final_receipt="changed"))
        with self.assertRaisesRegex(ValueError, "receipt chain"):
            self.run_items([a, b], backend)
        write(summary_path, summary)
        (self.root/"items"/"000001"/"item.json").unlink()
        with self.assertRaisesRegex(ValueError, "complete set"):
            self.run_items([a, b], backend)
        self.assertEqual(len(backend.evaluations), 2)

    def test_paid_later_item_prevents_replaying_a_missing_frontier(self):
        a, b = task(1), task(2)
        backend = Backend({a["id"]: 1, b["id"]: 1})
        def interrupt_before_first(**kwargs):
            raise OSError("engineering interruption")
        evaluate = backend.evaluate
        backend.evaluate = interrupt_before_first
        with self.assertRaises(OSError):
            self.run_items([a, b], backend)
        later = self.root/"items"/"000001"/"repair"/"trials"/"000"
        later.mkdir(parents=True)
        write(later/"proposal_intent.json", {"engineering": True})
        backend.evaluate = evaluate
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.run_items([a, b], backend)
        self.assertFalse(backend.evaluations)

    def test_infrastructure_failure_after_committed_item_and_resume(self):
        a, b = task(1), task(2)
        backend = Backend({a["id"]: 1, b["id"]: 1})
        first_pass = backend.evaluate
        def interrupted(**kwargs):
            if kwargs["task"]["id"] == b["id"]:
                raise OSError("engineering injected process interruption")
            return first_pass(**kwargs)
        backend.evaluate = interrupted
        with self.assertRaises((OSError, RuntimeError)):
            self.run_items([a, b], backend)
        self.assertEqual([row[0] for row in backend.commits], [a["id"]])
        directory = self.root/"items"/"000001"
        pause, = (directory/"interruptions").glob("*.json")
        self.assertEqual(read(pause)["task_id"], b["id"])
        self.assertFalse((directory/"item.json").exists())
        backend.evaluate = first_pass
        summary = self.run_items([a, b], backend)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual([row[0] for row in backend.commits], [a["id"], b["id"]])


if __name__ == "__main__":
    unittest.main()
