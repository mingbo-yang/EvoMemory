"""Adversarial changes to synthetic receipts. No service or model requests."""
import json
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
import unittest

from rsi_paired.audit import Audit, audit_run
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
