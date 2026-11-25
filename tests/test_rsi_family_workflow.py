"""Family-round lifecycle and journal recovery on synthetic native-result fixtures."""
import json

import pytest

from rsi_paired.audit import Audit, audit_run
from rsi_paired.panel import HISTORICAL_SEQUENTIAL_PROTOCOL, SEQUENTIAL_PROTOCOL
from rsi_paired.workflow import ContractError, InfrastructurePending, ReviewOutputError, digest, run_batch
from tests.test_rsi_roles import roles
from tests.test_rsi_workflow import Synthetic, panels, state


def reserves(round_index):
    def task(phase, i):
        key = f"{phase}-{round_index}-{i}"
        return dict(id=key, family="two", group_id=key, environment="alfworld",
                    archive_member=f"json_2.1.1/train/{key}/trial/game.tw-pddl",
                    game_sha256=digest(key), split="train", seed=i, room=1, goal="Synthetic task")
    return {phase: [task(phase, i) for i in range(30)] for phase in ("learning", "primary", "meta")}


class FamilyFixture(Synthetic):
    def __init__(self):
        super().__init__()
        self.learned = []
        self.interrupt = False

    def learn(self, source, memory, task, seed):
        memory, row = super().learn(source, memory, task, seed)
        self.learned.append(task["id"])
        score = int(task["id"].split("-")[-1]) % 2
        return memory, dict(row, family="two", phase="learning", status="completed", native_score=score,
                            score=score, episode_hash=digest(task["id"]), environment="alfworld", error=None)

    def diagnose(self, mechanism, memory, ledger, history):
        if self.interrupt:
            self.interrupt = False
            raise OSError("synthetic interruption after learning")
        return super().diagnose(mechanism, memory, ledger, history)

    def evaluate(self, source, memory, tasks, seeds):
        return [dict(row, native_score=row["score"], status="completed")
                for row in super().evaluate(source, memory, tasks, seeds)]


@pytest.mark.parametrize("protocol", [None, HISTORICAL_SEQUENTIAL_PROTOCOL], ids=["fresh-default", "historical"])
def test_sealed_gates_use_registered_policy_and_survive_restart(tmp_path, protocol):
    first = reserves(1)
    adapter = FamilyFixture()
    root = tmp_path / "round" / "batch"
    args = dict(state=state(), panels=first, adapter=adapter, learning_history_tasks=[], seed=52)
    if protocol is not None:
        args["protocol"] = protocol
    adapter.interrupt = True
    with pytest.raises(InfrastructurePending, match="pre_review"):
        run_batch(root, **args)
    selected = json.loads((root / "resolved_panels.json").read_text())
    selected_hash = digest(selected)
    assert len(adapter.learned) == 30 and not adapter.calls
    result = run_batch(root, **args)
    resume_args = {key: value for key, value in args.items() if key != "protocol"}
    assert result == run_batch(root, **resume_args)
    if protocol == HISTORICAL_SEQUENTIAL_PROTOCOL:
        with pytest.raises(ValueError, match="Frozen"):
            run_batch(root, **resume_args, protocol=SEQUENTIAL_PROTOCOL)
    assert len(adapter.learned) == len(set(adapter.learned)) == 30
    assert [row["task_id"] for row in result["ledger"]] == adapter.learned
    assert result["memory"]["learning"] == adapter.learned
    assert "panel" not in result["memory"]
    assert len(adapter.calls) == 9
    assert json.loads((root / "parent_snapshot.json").read_text())["resolved_panels_hash"] == selected_hash
    assert json.loads((root / "resolved_panels.json").read_text()) == selected
    ids = {}
    learning = {row["task_id"]: row["native_score"] for row in result["ledger"]}
    for phase in ("primary", "meta"):
        rows = selected["panels"][phase]
        ids[phase] = [row["id"] for row in rows]
        historical = [learning[i] for i in ids[phase] if i in learning]
        assert len(ids[phase]) == 30
        if protocol == HISTORICAL_SEQUENTIAL_PROTOCOL:
            assert historical.count(1) == historical.count(0) == 6
        else:
            assert not historical
            assert rows == first[phase]
        evaluated = json.loads((root / f"{phase}_results.json").read_text())
        assert all([r["task_id"] for r in c["evaluation"]] == ids[phase] for c in evaluated.values())
    assert not set(ids["primary"]).intersection(ids["meta"])
    pre = json.loads((root / "H_pre.json").read_text())
    assert all(t["id"] not in json.dumps(pre) for t in first["primary"] + first["meta"])
    shared = json.loads((root / "H_shared.json").read_text())
    strata = {entry["task_id"]: entry["source"] for entry in selected["sampling"]["primary"]["entries"]}
    assert all(row["gate_source"] == strata[row["task_id"]] for slot in shared["primary"].values()
               for row in slot["evaluation"])
    report = audit_run(root.parent)
    assert report["status"] == "verified_workflow_only"
    if protocol != HISTORICAL_SEQUENTIAL_PROTOCOL:
        for phase in ("primary", "meta"):
            assert report["panel_sampling"][phase]["entries"] == [
                {"task_id": task_id, "source": "fresh"} for task_id in ids[phase]]
    shared_path = root / "H_shared.json"
    original = shared_path.read_bytes()
    shared["primary"]["parent"]["evaluation"][0]["gate_source"] = "incorrect"
    shared_path.write_text(json.dumps(shared))
    assert "Shared feedback mislabels primary evaluation rows: parent" in audit_run(root.parent)["findings"]
    shared_path.write_bytes(original)
    selected["panels"]["primary"].reverse()
    (root / "resolved_panels.json").write_text(json.dumps(selected))
    if protocol != HISTORICAL_SEQUENTIAL_PROTOCOL:
        # Even a matching rewritten parent checksum cannot legitimize a panel that
        # differs from the registered order.
        parent_path = root / "parent_snapshot.json"
        parent = json.loads(parent_path.read_bytes())
        parent["resolved_panels_hash"] = digest(selected)
        parent_path.write_text(json.dumps(parent))
    assert audit_run(root.parent)["status"] == "audit_failed"
    with pytest.raises(ValueError, match="Frozen validation gates"):
        run_batch(root, **args)


def test_later_round_can_draw_prior_experience_but_never_relearn_a_game(tmp_path):
    adapter = FamilyFixture()
    adapter.triggers = False
    first, second = reserves(1), reserves(2)
    start = run_batch(tmp_path / "r1" / "batch", state=state(), panels=first, adapter=adapter,
                      learning_history_tasks=[], protocol=HISTORICAL_SEQUENTIAL_PROTOCOL)
    root = tmp_path / "r2" / "batch"
    args = dict(state=start, adapter=adapter, learning_history_tasks=first["learning"],
                protocol=HISTORICAL_SEQUENTIAL_PROTOCOL)
    with pytest.raises(ValueError, match="Repeated task"):
        run_batch(root, panels=first, **args)
    with pytest.raises(ValueError, match="historical tasks differ"):
        run_batch(root, state=start, adapter=adapter, panels=second, learning_history_tasks=[])
    assert len(adapter.learned) == 30
    finish = run_batch(root, panels=second, **args)
    assert len(finish["ledger"]) == len(set(adapter.learned)) == 60
    selected = json.loads((root / "resolved_panels.json").read_text())["panels"]
    earlier = {t["id"] for t in first["learning"]}
    assert any(t["id"] in earlier for phase in ("primary", "meta") for t in selected[phase])
    assert not (root / "meta_results.json").exists()
    assert audit_run(root.parent)["status"] == "verified_workflow_only"


def test_audit_resolves_public_code_and_episode_references(tmp_path):
    class Repeated(FamilyFixture):
        def propose(self, *args):
            return dict(super().propose(*args), source="original")

        def evaluate(self, source, memory, tasks, seeds):
            return [dict(row, episode_hash=digest(row), evidence={"steps": i, "storage_sample_after": "readout " * 180})
                    for i, row in
                    enumerate(super().evaluate(source, memory, tasks, seeds))]

    adapter = Repeated()
    adapter.triggers = False
    root = tmp_path / "round"
    # Exercise the actual role packet compressor without contacting any model.
    packing, model_calls = roles(root, None)
    adapter.compress = packing.compress
    run_batch(root / "batch", state=state(), panels=reserves(1), adapter=adapter, learning_history_tasks=[])
    assert not model_calls
    assert audit_run(root)["status"] == "verified_workflow_only"
    path = root / "batch/H_shared.json"
    shared = json.loads(path.read_text())
    p = shared["primary"]
    assert p["P2"]["source_ref"] == "/primary/P1/source"
    assert p["P2"]["evaluation"][2]["evidence_ref"] == "/primary/P1/evaluation/2/evidence"
    p["P1"]["evaluation"][0]["evidence"]["steps"] = 77
    p["P2"]["source_ref"] = "/primary/P3/source"
    p["P2"]["evaluation"][2]["evidence_ref"] = "/primary/P1/evaluation/3/evidence"
    path.write_text(json.dumps(shared))
    findings = audit_run(root)["findings"]
    assert "Shared source reference differs from the actual candidate: P2" in findings
    assert "Shared evidence reference differs from its episode: /primary/P2/evaluation/2/evidence" in findings
    assert "Directly displayed evidence differs from the archived facts: /primary/P1/evaluation/0" in findings


def test_single_decision_and_v3_evaluation_namespace(tmp_path):
    adapter = FamilyFixture()
    adapter.malform_decision = True
    root = tmp_path / "batch"
    args = dict(state=state(), panels=reserves(1), adapter=adapter, learning_history_tasks=[])
    done = run_batch(root, **args)
    assert done["history"][-1]["status"] == "decision_error"
    assert len(adapter.calls) == 3
    assert [p.name for p in (root / "jobs/decision").iterdir()] == ["0"]
    assert json.loads((root / "registration.json").read_text())["protocol"] == SEQUENTIAL_PROTOCOL
    assert audit_run(tmp_path)["status"] == "verified_fallback"
    assert run_batch(root, **args) == done
    (root / "jobs/decision/1").mkdir()
    assert "v3 allocated a second self-decision opportunity" in audit_run(tmp_path)["findings"]


class ReviewerFixture(FamilyFixture):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.reviews = []

    def evaluate(self, source, memory, tasks, seeds):
        rows = super().evaluate(source, memory, tasks, seeds)
        if source == "N3":
            rows[0].update(status="method_eval_failure", score=0, native_score=None,
                           candidate_error={"step": 0, "error_type": "Rejected"})
            for row in rows[1:]:
                row.update(score=1, native_score=1)
        return rows

    def review(self, source, records):
        if "retained" in records:
            self.reviews.append(records)
            if self.mode == "infrastructure":
                raise OSError("Synthetic transport failure")
            if self.mode == "unrelated_contract":
                raise ContractError("Synthetic unrelated contract failure")
            if self.mode != "valid":
                raise ReviewOutputError("Malformed Reviewer outcome", draft={"receipt": "first"})
        return {"findings": []}

    def review_retry(self, source, records, prior):
        assert prior == {"error": "Malformed Reviewer outcome", "previous_response": {"receipt": "first"}}
        assert self.reviews[-1] == records
        self.reviews.append(records)
        if self.mode == "interrupted":
            self.mode = "corrected"
            raise OSError("Synthetic interrupted correction")
        if self.mode == "missing":
            raise ReviewOutputError("Reviewer cited a missing ID", draft={"receipt": "second"})
        return {"findings": ["synthetic opinion"]}


@pytest.mark.parametrize("mode,status", [
    ("valid", "available"), ("corrected", "corrected"), ("missing", "missing"), ("interrupted", "corrected")])
def test_completed_meta_scores_follow_one_format_only_correction(tmp_path, mode, status):
    adapter = ReviewerFixture(mode)
    root = tmp_path / "batch"
    args = dict(state=state(), panels=reserves(1), adapter=adapter, learning_history_tasks=[],
                protocol=HISTORICAL_SEQUENTIAL_PROTOCOL)
    if mode == "interrupted":
        with pytest.raises(InfrastructurePending, match="reviews/meta_correction"):
            run_batch(root, **args)
        assert (root / "meta_results.json").exists() and not (root / "commit.json").exists()
    done = run_batch(root, **args)
    event = done["history"][-1]
    assert event["adopted"] and event["selected"] == "N3" and event["status"] == "paired_complete"
    assert event["meta_results"]["N3"]["success_count"] == 29
    assert event["meta_review"]["status"] == status
    assert len(event["meta_review"]["attempt_errors"]) == {"available": 0, "corrected": 1, "missing": 2}[status]
    assert len(adapter.reviews) == {"valid": 1, "corrected": 2, "missing": 2, "interrupted": 3}[mode]
    displayed = adapter.reviews[0]
    original = displayed["parent"]["provenance"]
    assert original["origin"] == "deployment" and original["origin_slot"] is None
    assert displayed["retained"]["provenance"]["origin_slot"] == event["primary_best"]
    assert displayed["N3"]["provenance"]["loaded_evolver_hash"] == digest(done["evolver"])
    assert any("historical_learning" in r for r in displayed["N3"]["evaluation"])
    assert "provenance" not in json.loads((root / "H_shared.json").read_text())["primary"]["P1"]
    assert len(adapter.calls) == 9 and len(adapter.learned) == 30
    assert run_batch(root, **args) == done
    assert audit_run(tmp_path)["status"] == "verified_workflow_only"
    commit = json.loads((root / "commit.json").read_text())
    commit["state"]["history"][-1]["meta_review"]["status"] = "invented"
    commit["state_hash"] = digest(commit["state"])
    (root / "commit.json").write_text(json.dumps(commit))
    assert "Published meta review differs from counted attempts" in audit_run(tmp_path)["findings"]


@pytest.mark.parametrize("mode,exception", [("infrastructure", InfrastructurePending), ("unrelated_contract", ContractError)])
def test_uncertain_review_is_never_treated_as_a_format_failure(tmp_path, mode, exception):
    root = tmp_path / "batch"
    with pytest.raises(exception):
        run_batch(root, state=state(), panels=reserves(1), adapter=ReviewerFixture(mode), learning_history_tasks=[])
    assert (root / "meta_results.json").is_file()
    assert not (root / "jobs/reviews/meta_correction").exists() and not (root / "commit.json").exists()


def test_exhausted_reviewer_transport_keeps_complete_meta_scores_pending(tmp_path):
    adapter = ReviewerFixture("infrastructure")
    adapter.scores = {}
    root = tmp_path / "batch"
    args = dict(state=state(), panels=panels(), adapter=adapter, protocol=SEQUENTIAL_PROTOCOL)
    for _ in range(2):
        with pytest.raises(InfrastructurePending, match="reviews/meta"):
            run_batch(root, **args)
    scores = (root / "meta_results.json").read_bytes()
    sealed = json.loads(scores)
    assert sealed["N3"]["success_count"] == 1 and all(sealed[k]["success_count"] == 0
                                                      for k in ("retained", "parent", "O1", "O2", "O3"))
    with pytest.raises(InfrastructurePending, match="Meta scores sealed"):
        run_batch(root, **args)
    assert scores == (root / "meta_results.json").read_bytes() and len(adapter.reviews) == 2
    assert not (root / "jobs/reviews/meta_correction").exists() and not (root / "commit.json").exists()
    audit = audit_run(tmp_path)
    assert audit["stage"] == "awaiting_meta_reviewer_recovery" and not audit["findings"]


def test_exhausted_incomplete_meta_panel_falls_back_to_primary(tmp_path):
    class InterruptedPanel(ReviewerFixture):
        def evaluate(self, source, memory, tasks, seeds):
            if source == "O1":
                raise OSError("Synthetic interrupted native panel")
            return super().evaluate(source, memory, tasks, seeds)
    adapter = InterruptedPanel("valid")
    root = tmp_path / "batch"
    args = dict(state=state(), panels=panels(), adapter=adapter, protocol=SEQUENTIAL_PROTOCOL)
    for _ in range(2):
        with pytest.raises(InfrastructurePending, match="evaluation/"):
            run_batch(root, **args)
    result = run_batch(root, **args)
    event = result["history"][-1]
    assert event["status"] == "meta_incomplete" and event["selected"] == event["primary_best"] == "P2"
    assert not (root / "meta_results.json").exists() and not adapter.reviews
    assert run_batch(root, **args) == result and audit_run(tmp_path)["status"] == "verified_fallback"


def test_missing_meta_opinion_gives_no_free_adoption_on_a_tie(tmp_path):
    class Tied(ReviewerFixture):
        evaluate = FamilyFixture.evaluate
    adapter = Tied("missing")
    adapter.scores = {}
    result = run_batch(tmp_path / "batch", state=state(), panels=panels(), adapter=adapter, protocol=SEQUENTIAL_PROTOCOL)
    event = result["history"][-1]
    assert event["meta_review"]["status"] == "missing"
    assert event["old_best"] == event["new_best"] == "retained" and not event["adopted"]
    assert result["evolver"] == state()["evolver"] and audit_run(tmp_path)["status"] == "verified_workflow_only"
    path = tmp_path / "batch/commit.json"
    result["history"][-1]["status"] = "meta_incomplete"
    path.write_text(json.dumps({"state": result, "state_hash": digest(result)}))
    assert "Sealed meta scores cannot be discarded by fallback" in audit_run(tmp_path)["findings"]


def test_six_task_v3_smoke_can_audit_actual_reviewer_wires_with_stubbed_transport(tmp_path):
    reviewer, _ = roles(tmp_path, None)
    def respond(request):
        body = json.loads(request["messages"][-1]["content"])
        group = "meta" if "meta.retained" in body["evidence"] else "primary"
        refs = [f"{group}.parent"] if group == "primary" or "format_correction" in body else ["unknown"]
        result = {"findings": [dict(evidence_ids=refs, judgment="insufficient_evidence", observation="Fixture")]}
        return dict(model=request["model"], choices=[dict(finish_reason="stop", message=dict(content=json.dumps(result)))],
                    usage=dict(prompt_tokens=120, completion_tokens=24))
    reviewer.transport = respond

    class WithRole(FamilyFixture):
        def set_job(self, path):
            super().set_job(path)
            reviewer.set_job(path)

        def evaluate(self, source, memory, tasks, seeds):
            rows = super().evaluate(source, memory, tasks, seeds)
            for r in rows:
                r["evidence"] = dict(steps=1, review_witness={"episode": digest([r["task_id"], source])})
            return rows

        review = staticmethod(reviewer.review)
        review_retry = staticmethod(reviewer.review_retry)

    root = tmp_path / "batch"
    run_batch(root, state=state(), panels=panels(), adapter=WithRole(), protocol=SEQUENTIAL_PROTOCOL)
    read = lambda name: json.loads((root / name).read_bytes())
    event = read("commit.json")["state"]["history"][-1]
    assert event["meta_review"]["status"] == "corrected"
    audit = Audit(tmp_path)
    audit.costs_and_wire()
    audit.real = True  # Check the recorded model wire; no native environment is part of this CPU fixture.
    m, p = read("parent_snapshot.json"), read("primary_results.json")
    audit.review_receipts("primary", p, m, event["primary_review"])
    keep = event["primary_best"]
    audit.review_receipts("meta", read("meta_results.json"), m, event["meta_review"], (keep, p[keep]))
    assert not audit.errors
    job = root / "jobs/reviews/meta_correction/result.json"
    receipt = json.loads(job.read_bytes())
    receipt["request_hash"] = digest("a different evidence packet")
    job.write_text(json.dumps(receipt))
    audit.review_receipts("meta", read("meta_results.json"), m, event["meta_review"], (keep, p[keep]))
    assert "Wrong job inputs for reviews/meta_correction" in audit.errors
