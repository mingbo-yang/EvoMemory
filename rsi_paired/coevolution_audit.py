"""Reconstruct the independent natural decisions and paired policy commitments."""
import json

from .coevolution_workflow import ARM_COMPONENTS, TARGETS, policy_versions, previous_result
from .roles import _displayed_hash, _review_findings, decode_role_response
from .workflow import ContractError, _review_records, digest


def _wire(audit, job, role, receipt):
    if not audit.real or receipt is None:
        return None
    matched = []
    for key, requests in audit.intents.items():
        first = requests[0]
        if first["role"] != role:
            continue
        binding = {k: first[k] for k in ("role", "model_fingerprint", "policy_hash", "request")}
        if digest(dict(binding, job=f"batch/jobs/{job}/private")) == key:
            matched.append((key, first))
    if not audit.check(len(matched) == 1, f"Unbound component model request: {job}"):
        return None
    key, item = matched[0]
    raw = json.loads((audit.root / "roles/requests" / key / "response.json").read_text())
    try:
        answer = decode_role_response(raw, item["request"]["model"])
    except ContractError:
        answer = None
    body = json.loads(item["request"]["messages"][-1]["content"])
    return body, answer, item["policy_hash"]


def _diagnosis(audit, name, observation, policy, outcome):
    audit._job_binding(name, [observation, digest(policy)])
    wire = _wire(audit, name, "diagnose", outcome)
    if wire:
        body, decoded, h = wire
        audit.check(h == digest(policy), f"Wrong Reviewer policy on wire: {name}")
        if outcome["ok"]:
            findings = _review_findings(decoded, body["evidence"])
            result = outcome["value"]
            audit.check(result["findings"] == findings["findings"] and result["evidence_hash"] == _displayed_hash(body),
                        f"Diagnosis differs from cited model output: {name}")
        return body.get("evidence_source_hash", digest(body["evidence"]))


def verify_components(audit, start, mother, state, aggregate, revision, shared):
    arm = start["arm"]
    components = ARM_COMPONENTS[arm]
    event = state["history"][-1]
    audit.check(state["arm"] == event["arm"] == arm and not event["forced_pair"], "Changed or forced experimental arm")
    if audit.real:
        runtime = json.loads((audit.root / "runtime.json").read_text())
        audit.check(runtime["arm"] == arm, "Runtime differs from registered arm")
    prior = previous_result(start["history"]) if start["history"] else None
    for packet in (audit.load("H_pre.json"), shared):
        displayed = packet.get("pre", packet)
        audit.check(displayed.get("prior_optimizer_result") == prior and
                    (("pre.prior_optimizer_result" in packet["evidence_ids"]) == (prior is not None)),
                    "Previous optimizer evidence differs from the preceding committed event")
    pre = audit.outcome("pre_review")
    _diagnosis(audit, "pre_review", digest(mother), start["reviewer"], pre)
    audit.check(pre and pre["ok"] and audit.load("H_pre.json")["diagnosis"] == pre["value"]
                and shared["pre"]["diagnosis"] == pre["value"], "Primary and old branch must use the deployed diagnosis")
    jobroot = audit.batch / "jobs"
    decision_dirs = {p.name for p in (jobroot/"self_decisions").glob("*")}
    audit.check(decision_dirs == set(components), "Arm requested the wrong component decisions")
    decisions = {}
    for role in components:
        job = "self_decisions/" + role
        rec = audit.outcome(job)
        audit._job_binding(job, [digest(start[role]), digest(shared)])
        decisions[role] = rec["value"] if rec and rec["ok"] else None
        wire = _wire(audit, job, "reviewer_decide" if role == "reviewer" else "decide", rec)
        if wire:
            body, decoded, policy_hash = wire
            audit.check(policy_hash == digest(start[role]) and body == {"current_strategy": start[role], "public_evidence": shared},
                        f"Wrong self-decision inputs: {role}")
            if rec["ok"]:
                if isinstance(decoded, dict) and isinstance(decoded.get("evidence_ids"), list):
                    decoded = dict(decoded, evidence_ids=list(dict.fromkeys(decoded["evidence_ids"])))
                audit.check(decoded == rec["value"], f"Model and self-decision differ: {role}")
    triggers = [r for r, value in decisions.items() if value and value["should_update"]]
    if aggregate:
        audit.check(aggregate["component_decisions"] == decisions and aggregate["arm"] == arm
                    and aggregate["target_components"] == [TARGETS[r] for r in triggers]
                    and aggregate["should_update"] == bool(triggers), "Combined trigger differs from role decisions")
    else:
        audit.check(any(value is None for value in decisions.values()), "Valid role decision discarded")
    changed = {r: start[r] for r in TARGETS}
    revised_dirs = {p.name for p in (jobroot/"self_revisions").glob("*")}
    if revision:
        audit.check(revised_dirs == set(triggers), "Unrequested or missing component revision")
    else:
        audit.check(not revised_dirs or event["status"] == "meta_incomplete", "Untracked component revisions")
    for role in revised_dirs & set(TARGETS):
        job = "self_revisions/" + role
        rec = audit.outcome(job)
        audit._job_binding(job, [digest(start[role]), digest(shared)])
        wire = _wire(audit, job, "reviewer_revise" if role == "reviewer" else "revise", rec)
        if wire:
            body, decoded, policy_hash = wire
            audit.check(body == {"current_strategy": start[role], "public_evidence": shared}
                        and policy_hash == digest(start[role]), f"Wrong revision inputs: {role}")
            if rec["ok"]:
                value = rec["value"]
                actual = {"prompt": value["prompt"]} if role == "reviewer" else value["propose"]
                audit.check(decoded == actual, f"Model and revised policy differ: {role}")
        if rec and rec["ok"]:
            changed[role] = rec["value"]
            audit.check(changed[role] != start[role], f"Unchanged revision: {role}")
    if revision and revision["ok"]:
        audit.check(revision["value"] == changed, "Paired revision differs from its component receipts")
    bindings = audit.load("paired_diagnoses.json")
    if event["status"] == "paired_complete":
        audit.check(bindings is not None, "Missing paired diagnostic bindings")
    if bindings or event["status"] == "invalid_paired_diagnosis":
        primary = audit.load("primary_results.json")
        entries = audit.report.get("panel_sampling", {}).get("primary", {}).get("entries", [])
        records = _review_records(primary, entries, "primary", digest(mother), mother["learning_cutoff"])
        obs = [digest(mother), digest(records), digest(shared["review"])]
        same = start["reviewer"] == changed["reviewer"]
        seen, rejected = [], False
        for label, policies, filename in (("O", start, "H_meta_old.json"), ("N", changed, "H_meta_new.json")):
            job = "paired_diagnoses/O" if label == "O" or same else "paired_diagnoses/N"
            outcome = audit.outcome(job)
            if not outcome:
                break
            raw_hash = _diagnosis(audit, job, obs, policies["reviewer"], outcome)
            if not outcome["ok"]:
                rejected = outcome.get("error_kind") == "review_output"
                break
            diagnosis = outcome["value"]
            # Witness indexing on the wire references full raw receipts by hash.
            # Compare both representations across arms in their own namespaces.
            seen.append((raw_hash, diagnosis["raw_evidence_hash"], diagnosis.get("evidence_hash")))
            packet = audit.load(filename)
            audit.check(packet and packet["pre"]["diagnosis"] == diagnosis and packet["diagnosis_phase"] == "post_primary"
                        and all(packet[k] == shared[k] for k in ("retained", "review", "evidence_ids"))
                        and packet["pre"]["history"] == shared["pre"]["history"], "Paired branch lost its own diagnostic findings")
            if bindings:
                audit.check(bindings["observation"] == obs and bindings["branches"][label] == dict(
                    reviewer_hash=digest(policies["reviewer"]), evolver_hash=digest(policies["evolver"]),
                    diagnosis_job=job, diagnosis_hash=digest(diagnosis), packet_hash=digest(packet)),
                    "Wrong paired optimizer receipt: " + label)
        if event["status"] == "paired_complete":
            audit.check(len(seen) == 2 and seen[0] == seen[1], "Paired Reviewers did not receive the same raw observations")
        if event["status"] == "invalid_paired_diagnosis":
            audit.check(rejected, "Rejected paired diagnosis lacks its format rejection")
    expected = changed if event.get("adopted") else start
    audit.check(all(state[r] == expected[r] for r in TARGETS)
                and event["selected_reviewer_hash"] == digest(state["reviewer"])
                and event["selected_optimizer_hash"] == digest({r: state[r] for r in TARGETS}),
                "Deployed policies differ from independent revisions and paired scores")
    proposed = policy_versions(changed) if revision and revision["ok"] else None
    audit.check(event["optimizer_versions"] == dict(starting=policy_versions(start), proposed=proposed,
                                                    deployed=policy_versions(state)),
                "Incorrect policy version history for the next round")
    audit.report["self_decision"] = dict(arm=arm, eligible_roles=list(components), triggered_roles=triggers)
