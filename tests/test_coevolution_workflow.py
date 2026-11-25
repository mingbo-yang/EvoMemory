"""Synthetic arm isolation, paired attribution, natural decisions and recovery."""
from copy import deepcopy
import json

import pytest

from rsi_paired.audit import audit_run
from rsi_paired.panel import COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL, COEVOLUTION_PROTOCOL, coevolution_shape
from rsi_paired.runner import initial_state
from rsi_paired.workflow import ContractError, InfrastructurePending, ReviewOutputError, digest, run_batch
from tests.test_rsi_workflow import Synthetic, state


class ArmFixture(Synthetic):
    def __init__(self, arm, update):
        super().__init__()
        self.arm, self.update = arm, set(update)
        self.packets, self.diagnoses, self.decisions, self.revisions = {}, [], [], []
        self.decision_evidence = {}
        self.reject = self.interrupt = self.bad_diagnosis = False

    def diagnose_with_policy(self, policy, mechanism, memory, ledger, history):
        return {"synthetic_diagnostic_focus": policy["prompt"], "phase": "pre_primary"}

    def diagnose_post_primary(self, policy, mechanism, memory, ledger, history, primary_records, *, primary_review):
        raw_hash = digest([mechanism, memory, ledger, history, primary_records, primary_review])
        self.diagnoses.append((policy["prompt"], raw_hash))
        if policy["prompt"] and self.interrupt:
            self.interrupt = False
            raise OSError("synthetic diagnosis interruption")
        if policy["prompt"] and self.bad_diagnosis:
            raise ReviewOutputError("Unavailable cited evidence")
        return {"synthetic_diagnostic_focus": policy["prompt"], "phase": "post_primary", "raw_evidence_hash": raw_hash}

    def decide_component(self, role, policy, packet):
        self.decisions.append(role)
        self.decision_evidence[role] = deepcopy(packet)
        should = role in self.update
        return dict(should_update=should, evidence_ids=["primary.P1"], reason="Observed test evidence",
                    expected_effect="Better tests", target_components=[{"evolver": "propose", "reviewer": "diagnose"}[role]] if should else [])

    def revise_component(self, role, policy, packet):
        self.revisions.append(role)
        if role == "reviewer":
            if self.reject:
                raise ContractError("Invalid format")
            return dict(policy, prompt="Check relevant delivered details.")
        return dict(policy, propose="new")

    def propose_slot(self, policy, parent, packet, seed, previous, error, slot_index):
        name = self.job.parent.parent.name
        assert slot_index + 1 == int(name[1])
        self.packets[name] = deepcopy(packet)
        return self.propose(policy, parent, packet, seed, previous, error)

    def evaluate(self, source, memory, tasks, seeds):
        return [dict(row, status="completed", native_score=row["score"]) for row in super().evaluate(source, memory, tasks, seeds)]


def panels(protocol=COEVOLUTION_PROTOCOL):
    size = coevolution_shape(protocol)[1]
    family = "heat" if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL else "two"
    return {phase: [{"id": f"{phase}-{i}", "family": family} for i in range(size)] for phase in ("learning", "primary", "meta")}


def execute(root, adapter, protocol=COEVOLUTION_PROTOCOL):
    start = dict(state(), arm=adapter.arm, reviewer=initial_state(adapter.arm)["reviewer"])
    return run_batch(root / "batch", state=start, panels=panels(protocol), adapter=adapter, protocol=protocol)


@pytest.mark.parametrize("arm,eligible", [("A", ()), ("B", ("reviewer",)), ("C", ("evolver",)),
                                            ("D", ("reviewer", "evolver"))])
def test_arm_eligibility_evidence_routing_and_restart(tmp_path, arm, eligible):
    adapter = ArmFixture(arm, ["reviewer", "evolver"])
    result = execute(tmp_path, adapter)
    assert adapter.decisions == list(eligible) == adapter.revisions
    primary_context = adapter.packets["P1"]
    assert primary_context["diagnosis"] == {"synthetic_diagnostic_focus": "", "phase": "pre_primary"}
    if eligible:
        assert result["history"][-1]["adopted"]
        before = adapter.packets["O1"]["pre"]["diagnosis"]
        after = adapter.packets["N1"]["pre"]["diagnosis"]
        assert before["phase"] == after["phase"] == "post_primary"
        assert before["raw_evidence_hash"] == after["raw_evidence_hash"]
        assert (before != after) == ("reviewer" in eligible)
        if "reviewer" in eligible:
            assert len(adapter.diagnoses) == 2 and adapter.diagnoses[0][1] == adapter.diagnoses[1][1]
            assert bool(result["reviewer"]["prompt"])
        assert (result["evolver"]["propose"] == "new") == ("evolver" in eligible)
    else:
        assert result["history"][-1]["status"] == "not_triggered" and len(adapter.packets) == 3
    report = audit_run(tmp_path)
    assert report["status"] == "verified_workflow_only", report["findings"]
    assert execute(tmp_path, adapter) == result and adapter.decisions == list(eligible)


def test_independent_decisions_and_tie(tmp_path):
    adapter = ArmFixture("D", ["evolver"])
    adapter.scores.update(N3=0, O1=2)
    result = execute(tmp_path, adapter)
    assert adapter.revisions == ["evolver"] and len(adapter.diagnoses) == 1
    assert result["evolver"]["propose"] == "old" and result["reviewer"]["prompt"] == ""
    assert not result["history"][-1]["adopted"]
    assert audit_run(tmp_path)["status"] == "verified_workflow_only"


def test_interrupted_revised_diagnosis_is_resumed_without_revisiting_decisions(tmp_path):
    adapter = ArmFixture("B", ["reviewer"])
    adapter.interrupt = True
    with pytest.raises(InfrastructurePending, match="paired_diagnoses/N"):
        execute(tmp_path, adapter)
    assert not (tmp_path / "batch/commit.json").exists()
    result = execute(tmp_path, adapter)
    assert adapter.decisions == adapter.revisions == ["reviewer"] and result["history"][-1]["adopted"]
    path = tmp_path / "batch/H_meta_new.json"
    value = json.loads(path.read_text())
    value["pre"]["diagnosis"] = {}
    path.write_text(json.dumps(value))
    assert audit_run(tmp_path)["status"] == "audit_failed"


def test_invalid_review_does_not_revise_independent_evolver_or_publish_a_diagnosis(tmp_path):
    adapter = ArmFixture("B", ["reviewer"])
    adapter.bad_diagnosis = True
    result = execute(tmp_path, adapter)
    assert result["history"][-1]["status"] == "invalid_paired_diagnosis"
    assert result["reviewer"]["prompt"] == "" and result["evolver"]["propose"] == "old"
    assert audit_run(tmp_path)["status"] == "verified_fallback"


def test_rejected_component_rolls_back_the_whole_requested_revision(tmp_path):
    adapter = ArmFixture("D", ["reviewer", "evolver"])
    adapter.reject = True
    result = execute(tmp_path, adapter)
    assert not result["reviewer"]["prompt"] and result["evolver"]["propose"] == "old"
    assert result["history"][-1]["status"] == "invalid_self_revision"
    assert audit_run(tmp_path)["status"] == "verified_fallback"


def test_policy_compares_offspring_while_stronger_control_can_be_deployed(tmp_path):
    adapter = ArmFixture("B", ["reviewer"])
    adapter.scores.update(original=2, O1=0, N3=1)
    result = execute(tmp_path, adapter)
    event = result["history"][-1]
    assert event["adopted"] and event["selected"] == "retained" and result["mechanism"] == "original"
    assert (event["old_offspring_count"], event["new_offspring_count"]) == (0, 1)
    assert result["reviewer"]["prompt"] and audit_run(tmp_path)["status"] == "verified_workflow_only"


def test_missing_all_valid_old_offspring_cannot_adopt_new_optimizer(tmp_path):
    adapter = ArmFixture("D", ["reviewer", "evolver"])
    adapter.first_passes = {"O1": 4, "O2": 4, "O3": 4}
    result = execute(tmp_path, adapter)
    event = result["history"][-1]
    assert event["old_offspring_count"] is None and event["new_offspring_count"] == 2
    assert not event["adopted"] and not result["reviewer"]["prompt"]
    assert audit_run(tmp_path)["status"] == "verified_workflow_only"


@pytest.mark.parametrize("protocol", [COEVOLUTION_HEAT_5X23_PROTOCOL, COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_PROTOCOL])
@pytest.mark.parametrize("adopt", [True, False])
def test_following_round_exposes_only_own_sealed_result_and_uses_deployed_pair(tmp_path, adopt, protocol):
    first = ArmFixture("D", ["evolver", "reviewer"])
    if not adopt:
        first.scores["O1"] = first.scores["N3"]
    prior = execute(tmp_path / "first", first, protocol)
    event = prior["history"][-1]
    assert event["adopted"] == adopt
    second = ArmFixture("D", [])
    from tests.test_rsi_roles import roles
    packing, calls = roles(tmp_path / "packing", None)
    second.compress = packing.compress
    next_panels = {p: [dict(row, id=row["id"] + "-next") for row in rows] for p, rows in panels(protocol).items()}
    # Verified ledger/score ordering is covered by the native family tests.
    root = tmp_path / "second"
    state2 = run_batch(root / "batch", state=prior, panels=next_panels, adapter=second, protocol=protocol)
    pre = second.packets["P1"]
    assert pre["diagnosis"]["synthetic_diagnostic_focus"] == prior["reviewer"]["prompt"]
    assert second.calls[0][3] == prior["evolver"]["propose"]
    ref = pre["prior_optimizer_result"]
    assert set(ref) == {"round", "arm", "event_hash", "status", "policies", "adopted", "old_offspring_count", "new_offspring_count"}
    assert ref["arm"] == "D" and ref["event_hash"] == digest(event) and ref["round"] == 1
    assert ref["policies"] == event["optimizer_versions"]
    assert ref["old_offspring_count"] == event["old_offspring_count"]
    assert ref["adopted"] == adopt and ref["new_offspring_count"] == event["new_offspring_count"]
    shared = json.loads((root / "batch/H_shared.json").read_text())
    assert "pre.prior_optimizer_result" in shared["evidence_ids"] and shared["pre"]["prior_optimizer_result"] == ref
    assert second.decision_evidence == {"reviewer": shared, "evolver": shared} and calls == []
    assert state2["history"][:-1] == prior["history"]
    assert audit_run(root)["status"] == "verified_workflow_only"

@pytest.mark.parametrize("protocol,wrong", [(COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_PROTOCOL),
                                           (COEVOLUTION_PROTOCOL, COEVOLUTION_6X30_PROTOCOL),
                                           (COEVOLUTION_HEAT_5X23_PROTOCOL, COEVOLUTION_6X30_PROTOCOL)])
def test_shapes_and_runtime_bindings_fail_before_learning(tmp_path, protocol, wrong):
    adapter = ArmFixture("A", [])
    adapter.protocol = protocol
    start = dict(state(), arm="A", reviewer=initial_state("A")["reviewer"])
    args = dict(state=start, adapter=adapter)
    with pytest.raises(ValueError, match="protocol width"):
        run_batch(tmp_path / "batch", panels=panels(wrong), protocol=protocol, **args)
    with pytest.raises(ValueError, match="Runtime protocol"):
        run_batch(tmp_path / "batch", panels=panels(wrong), protocol=wrong, **args)
    assert not (tmp_path / "batch").exists()
    registered = panels(protocol)
    adapter.registered_batches = (digest(registered),)
    swapped = dict(registered, primary=registered["meta"], meta=registered["primary"])
    with pytest.raises(ValueError, match="registered campaign panel"):
        run_batch(tmp_path / "batch", panels=swapped, **args)
    if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL:
        wrong_family = deepcopy(registered)
        wrong_family["learning"][0]["family"] = "two"
        with pytest.raises(ValueError, match="heat tasks"):
            run_batch(tmp_path / "batch", panels=wrong_family, **args)
    result = run_batch(tmp_path / "batch", panels=registered, **args)
    assert len(result["ledger"]) == coevolution_shape(protocol)[1]
    assert audit_run(tmp_path)["status"] == "verified_workflow_only"
    record = tmp_path / "batch/registration.json"
    registered = json.loads(record.read_text())
    registered["protocol"] = wrong
    record.write_text(json.dumps(registered))
    assert "Co-evolution panel width differs from registered protocol" in audit_run(tmp_path)["findings"]
