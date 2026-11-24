"""Adversarial changes to synthetic receipts. No service or model requests."""
import hashlib
import json
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
import unittest

from phase1_codegen.program import BASELINE_SOURCE, Program
from rsi_paired.audit import Audit, audit_run, file_hash
from rsi_paired.workflow import digest, run_batch
from tests.test_rsi_workflow import Synthetic, panels, state


class AuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        run_batch(Path(cls.fixture.name)/"batch", state=state(), panels=panels(), adapter=Synthetic())

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        copytree(Path(self.fixture.name)/"batch", self.root/"batch")

    def edit(self, filename, rewrite):
        path = self.root/"batch"/filename
        data = json.loads(path.read_text())
        rewrite(data)
        path.write_text(json.dumps(data))

    def test_complete_paired_selection_and_no_claim_of_native_run(self):
        report = audit_run(self.root)
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["status"], "verified_workflow_only")
        self.assertEqual(report["selection"]["selected"], "N3")
        self.assertEqual(len(report["slots"]), 9)
        self.assertEqual(report["scores"]["meta"]["distinct_panel_executions"], 8)

    def test_score_and_commit_tampering(self):
        self.edit("meta_results.json", lambda d: d["N3"].update(success_count=0))
        self.assertEqual(audit_run(self.root)["status"], "audit_failed")
        self.edit("commit.json", lambda d: d["state"].update(mechanism="foreign"))
        self.assertIn("Commit identity/learning cutoff mismatch", audit_run(self.root)["findings"])

    def test_parent_and_eval_memory_and_barrier(self):
        self.edit("slots/N1.json", lambda d: d.update(parent_framework_hash="wrong"))
        self.edit("meta_results.json", lambda d: d["retained"].update(pre_memory_hash="private"))
        (self.root/"batch/paired_seal.json").unlink()
        issues = audit_run(self.root)["findings"]
        self.assertTrue(any("Wrong mother" in msg for msg in issues))
        self.assertTrue(any("without all proposals sealed" in msg for msg in issues))
        self.assertTrue(any("pre-evaluation memory" in msg for msg in issues))

    def test_partial_receipts(self):
        (self.root/"batch/commit.json").unlink()
        report = audit_run(self.root)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["stage"], "awaiting_final_commit")
        self.assertEqual(report["scores"]["primary"]["success_counts"]["P2"], 2)

    def test_per_job_wire_binding_verifies_actual_body(self):
        a = Audit(self.root)
        frozen = {"evidence_ids": ["pre.diagnosis"]}
        body = dict(current_source="wrong", public_evidence=frozen)
        req = {"messages": [{"role":"user", "content":json.dumps(body)}]}
        key = digest(dict(role="propose", job="batch/jobs/proposals/O1/0/private",
                          model_fingerprint="model", policy_hash="policy", request=req))
        a.intents = {key: [{"role":"propose", "model_fingerprint":"model", "policy_hash":"policy",
                           "request":req, "public_hash":digest(frozen)}]}
        self.assertEqual(len(a._slot_wire("O1",0,"policy",frozen,"original",None,None)), 1)
        self.assertTrue(a.errors)
        self.assertEqual(a._slot_wire("N1",0,"policy",frozen,"original",None,None), [])

    def test_real_evidence_cannot_be_claimed_from_synthetic_workflow(self):
        (self.root/"runtime.json").write_text("{}")
        report = audit_run(self.root)
        self.assertEqual(report["status"],"audit_failed")
        self.assertIn("Missing actual proposal request P1/0",report["findings"])


class NativeBindingChecks(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "runtime.json").write_text(json.dumps({"native_fingerprint":"engine"}))
        self.task = {"id":"kitchen", "family":"heat", "seed":8}
        self.program = Program(BASELINE_SOURCE)
        self.seed = 13
        self.panel = digest(dict(tasks=[self.task], seeds=[self.seed], pre={},
                                 program=self.program.hash, evaluator="engine"))

    def receipt(self, *, job="evaluation/first", learning=False, version=2):
        source = self.program.hash
        scope = "learning" if learning else "panel"
        before = dict(version="code-memory-state-v1", program_hash=source,
                      program_identity_hash=self.program.identity_hash, state_wire="{}",
                      context_wire='{"environment":"alfworld","memory_scope":"family","family":"heat"}',
                      environment="alfworld", memory_scope="family", family=None, families=[],
                      source_log_hash=digest([]), capacity=16384, read_budget=2048, top_k=4, stored_tokens=1)
        before["memory_hash"] = Audit._memory_hash(before)
        before["snapshot_hash"] = digest(before)
        after = dict(before, family="heat", families=["heat"], state_wire='{"a":1}')
        after["memory_hash"] = Audit._memory_hash(after)
        ep = dict(seed=self.seed, environment_seed=self.task["seed"], task_id=self.task["id"], family="heat",
                  environment="alfworld", phase=scope, program_hash=source, memory_hash_before=before["memory_hash"],
                  memory_hash_after=after["memory_hash"], synthetic=False, score=1, native_reward_verified=True,
                  native_recorder={"reward":1, "steps":[{"scores":1}], "observed":{"second":"b", "first":"a"}}, error=None)
        raw = json.dumps(ep, ensure_ascii=False, separators=(",", ":")).encode()
        ep_hash = hashlib.sha256(raw).hexdigest()
        after["source_log_hash"] = digest([before["source_log_hash"], ep_hash])
        after["snapshot_hash"] = digest({k:v for k,v in after.items() if k != "snapshot_hash"})
        row = dict(task_id=ep["task_id"], phase=scope, family="heat", environment="alfworld", score=1,
                   native_score=1, status="completed", error=None, episode_hash=ep_hash,
                   evidence={"stored_tokens_before":1})
        binding = dict(version=version, source=source, evaluator="engine", seed=self.seed, task=digest(self.task),
                       scope=scope, index=0, panel=None if learning else self.panel, initial=before["snapshot_hash"],
                       job="/original/location/batch/jobs/"+job+"/private")
        dest = self.root / "native/episodes" / digest(binding) / "attempt-0"
        dest.mkdir(parents=True)
        canonical = dest / "canonical_episodes.jsonl"
        canonical.write_text(json.dumps(ep, sort_keys=True)+"\n")
        actor = dest / "actor.jsonl"
        actor.touch()
        result = dict(attempt="attempt-0", row=row, snapshot=after, canonical_hash=file_hash(canonical),
                      actor_hash=file_hash(actor))
        if version == 2:
            result["episode_wire"] = "episode.json"
            (dest / "episode.json").write_bytes(raw)
        if learning:
            archive = self.root / "native/learning" / (ep_hash+".json")
            archive.parent.mkdir(exist_ok=True)
            archive.write_bytes(raw)
        complete = dest.parent / "complete.json"
        complete.write_text(json.dumps(dict(binding=binding, result=result, result_hash=digest(result))))
        return dest, row, before

    def indexed(self):
        audit = Audit(self.root)
        audit.costs_and_wire()
        return audit

    def test_exact_wire_and_relocated_registered_panel(self):
        dest, row, _ = self.receipt()
        audit = self.indexed()
        audit.bind_panel("first", [self.task], [self.seed], BASELINE_SOURCE, {}, [row])
        self.assertEqual(audit.errors, [])
        self.assertEqual(audit.report["native_wire"], {"verified":1, "unavailable_legacy_panels":[]})
        ordered = dest / "episode.json"
        obj = json.loads(ordered.read_bytes())
        # Only change the insertion order of a nested mapping. Canonical JSON still compares equal.
        obj["native_recorder"]["observed"] = {"first":"a", "second":"b"}
        ordered.write_bytes(json.dumps(obj, separators=(",", ":")).encode())
        self.assertIn("Native ordered episode digest/content changed", self.indexed().errors)

    def test_rehashed_foreign_score_or_sorted_canonical_data_fails(self):
        dest, _, _ = self.receipt()
        complete = dest.parent / "complete.json"
        data = json.loads(complete.read_bytes())
        data["result"]["row"]["episode_hash"] = "0"*64
        data["result_hash"] = digest(data["result"])
        complete.write_text(json.dumps(data))
        self.assertIn("Native ordered episode digest/content changed", self.indexed().errors)
        # Restore the original row and independently alter canonical content and its outer checksums.
        data["result"]["row"]["episode_hash"] = file_hash(dest/"episode.json")
        canonical = dest/"canonical_episodes.jsonl"
        obj = json.loads(canonical.read_bytes())
        obj["native_recorder"]["observed"]["first"] = "different"
        canonical.write_text(json.dumps(obj, sort_keys=True)+"\n")
        data["result"]["canonical_hash"] = file_hash(canonical)
        data["result_hash"] = digest(data["result"])
        complete.write_text(json.dumps(data))
        self.assertIn("Native ordered episode digest/content changed", self.indexed().errors)

    def test_identical_episode_row_does_not_authorize_another_job_or_memory(self):
        _, row, before = self.receipt()
        source = self.program.hash
        cases = [("evaluation/sibling",0,self.task,self.seed,source,self.panel,{}),
                 ("evaluation/first",1,self.task,self.seed,source,self.panel,{}),
                 ("evaluation/first",0,dict(self.task, variant=1),self.seed,source,self.panel,{}),
                 ("evaluation/first",0,self.task,self.seed+1,source,self.panel,{}),
                 ("evaluation/first",0,self.task,self.seed,digest(BASELINE_SOURCE),self.panel,{}),
                 ("evaluation/first",0,self.task,self.seed,source,None,{}),
                 ("evaluation/first",0,self.task,self.seed,source,"wrong-panel",{}),
                 ("evaluation/first",0,self.task,self.seed,source,self.panel,{"heat":dict(before,snapshot_hash="wrong")})]
        for args in cases:
            with self.subTest(job=args[0], index=args[1], panel=args[5]):
                audit = self.indexed()
                audit.bind_native(*args, row)
                self.assertTrue(audit.errors)

    def test_legacy_learning_archive_and_panel_wire_coverage(self):
        _, row, _ = self.receipt(job="learning/000", learning=True, version=1)
        dest, _, _ = self.receipt(version=1)
        audit = self.indexed()
        audit.bind_native("learning/000",0,self.task,self.seed,self.program.hash,None,{},row)
        self.assertEqual(audit.errors, [])
        self.assertEqual(audit.report["native_wire"], {"verified":1, "unavailable_legacy_panels":[dest.parent.name]})
        learning = self.root / "native/learning" / (row["episode_hash"]+".json")
        learning.unlink()
        self.assertIn("Native learning archive changed", self.indexed().errors)

    def test_current_native_receipt_requires_the_ordered_witness(self):
        dest, _, _ = self.receipt()
        (dest/"episode.json").unlink()
        audit = self.indexed()
        self.assertIn("Missing or invalid native ordered episode witness", audit.errors)
        self.assertEqual(audit.report["native_wire"]["unavailable_legacy_panels"], [])
