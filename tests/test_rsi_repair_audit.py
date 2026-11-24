"""Synthetic malformed role responses with the same file boundaries as a real run."""
import json
import tempfile
import unittest
from pathlib import Path

from rsi_paired.audit import Audit
from rsi_paired.roles import decode_role_response, proposal_source
from rsi_paired.workflow import ContractError, digest, run_batch
from tests.test_rsi_workflow import Synthetic, panels, state


def response(text, finish="stop"):
    return dict(model="test", choices=[dict(message=dict(content=text), finish_reason=finish)])


class Malformed(Synthetic):
    def propose(self, policy, mechanism, packet, seed, previous, error):
        result = super().propose(policy, mechanism, packet, seed, previous, error)
        name, index = self.job.parent.parent.name, int(self.job.parent.name)
        if name == "P1" and index in (1, 2):
            reply = response(None, "length") if index == 1 else response("  {")
            proposal_source(decode_role_response(reply, "test"))
        return result


class RepairChainAudit(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        engine = Malformed()
        engine.first_passes = {"P1": 1}
        run_batch(self.root/"batch", state=state(), panels=panels(), adapter=engine)
        self.audit = Audit(self.root)
        self.audit.real = True
        self.packet = self.audit.load("H_pre.json")
        self.mother = self.audit.load("parent_snapshot.json")
        self.row = self.audit.load("slots/P1.json")
        self.policy = digest(state()["evolver"])
        self.requests = []
        responses = [response(json.dumps({"source":"P1-bad"})), response(None, "length"),
                     response("  {"), response(json.dumps({"source":"P1"}))]
        for index, reply in enumerate(responses):
            body = dict(public_evidence=self.packet, current_source=self.mother["framework"])
            if index:
                prior = self.row["drafts"][index-1]
                body.update(own_draft=prior["source"], own_contract_error=prior["error"])
            request = dict(messages=[{"role":"user","content":json.dumps(body)}],
                           seed=self.row["drafts"][index]["seed"], model="test")
            key = digest(dict(role="propose",job=f"batch/jobs/proposals/P1/{index}/private",
                              model_fingerprint="test",policy_hash=self.policy,request=request))
            intent = dict(request=request,policy_hash=self.policy,role="propose",
                          model_fingerprint="test",public_hash=digest(self.packet))
            self.audit.intents[key] = [intent]
            out = self.root/"roles/requests"/key
            out.mkdir(parents=True)
            (out/"response.json").write_text(json.dumps(reply))
            self.requests.append((key,out,body))

    def inspect(self):
        self.audit.slot("P1",self.packet,self.mother,self.policy,None)
        return self.audit.errors

    def change(self, relative, modify):
        path = self.root/"batch"/relative
        data = json.loads(path.read_text())
        modify(data)
        path.write_text(json.dumps(data))

    def test_absent_and_malformed_generation_are_separately_bound(self):
        key = self.requests[1][0]
        self.audit.intents[key] *= 2
        self.assertEqual(self.inspect(), [])
        attempts = self.audit.slots["P1"]["attempts"]
        self.assertEqual([a["provenance"] for a in attempts],
                         ["generated_source", "generation_error_without_draft",
                          "generation_error_with_draft", "generated_source"])
        self.assertIsNone(attempts[1]["source_hash"])
        self.assertEqual(attempts[2]["source_hash"], digest("  {"))
        send, = attempts[1]["actual_requests"]
        self.assertEqual((send["sends"], send["response_attribution"]), (2, "request_key"))

    def test_historical_source_cannot_be_attributed_to_missing_generation(self):
        self.change("slots/P1.json",lambda row: row["drafts"][1].update(source="P1-bad"))
        self.assertIn("Wrong recorded draft source P1/1",self.inspect())

    def test_exact_check_error_and_next_request_inputs_are_bound(self):
        def rewrite(data):
            data["outcome"]["error"] = "Other check error"
            data["outcome_hash"] = digest(data["outcome"])
        self.change("jobs/checks/P1/0/result.json",rewrite)
        errors = self.inspect()
        self.assertIn("Wrong recorded draft error P1/0", errors)
        self.assertIn("Wrong job inputs for proposals/P1/1", errors)
        self.assertIn("Incorrect actual proposal input P1/1", errors)

    def test_malformed_response_must_explain_journal_and_draft(self):
        _,folder,_ = self.requests[2]
        (folder/"response.json").write_text(json.dumps(response(" {")))
        self.assertIn("Generation failure differs from actual response P1/2", self.inspect())

    def test_registration_downgrade_and_modified_seed_rejected(self):
        self.change("registration.json", lambda r:r.pop("repair_version"))
        errors = self.inspect()
        self.assertIn("Versioned draft in legacy registration P1/0",errors)
        self.assertIn("Wrong job inputs for proposals/P1/0",errors)
        self.assertIn("Wrong model request seed P1/1",errors)

    def test_legacy_carried_source_is_reported_but_never_certified(self):
        # Rebuild the batch-scoped V1 input bindings used by the historic journal.
        # V1 kept the previous source after attempt 1 received no draft.
        self.change("registration.json", lambda r: r.pop("repair_version"))
        row = self.row
        binding = {k:v for k,v in row.items() if k not in
                   ("drafts", "source", "source_hash", "error", "valid", "loaded_evolver_hash")}
        previous = error = None
        for index,draft in enumerate(row["drafts"]):
            draft.pop("seed")
            if index == 1:
                draft["source"] = previous
            request_hash = digest(dict(**binding, draft=index, previous_hash=digest(previous), error=error))
            folder = self.root/"batch"/"jobs"/"proposals"/"P1"/str(index)
            for path in folder.glob("*.json"):
                record = json.loads(path.read_text())
                path.write_text(json.dumps(dict(record, request_hash=request_hash)))
            previous, error = draft["source"], draft["error"]
        (self.root/"batch"/"slots"/"P1.json").write_text(json.dumps(row))
        self.audit.real = False
        self.audit.intents.clear()
        self.assertEqual(self.inspect(), [])
        self.assertEqual(self.audit.report["legacy_unbound_repairs"], ["P1/1"])
        attempt = self.audit.slots["P1"]["attempts"][1]
        self.assertEqual(attempt["provenance"], "legacy_carried_previous_draft")
        self.assertIsNone(attempt["source_hash"])

    def test_absent_body_is_distinct_from_absent_response(self):
        _,folder,_ = self.requests[1]
        (folder/"response.json").unlink()
        self.assertIn("Generation failure differs from actual response P1/1",self.inspect())
