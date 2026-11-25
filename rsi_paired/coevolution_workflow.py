"""Independent Reviewer/Evolver self-revision layered on the native paired tournament."""
from copy import deepcopy

from .workflow import ContractError, _Batch, _check_policy, _choose, _read, _review_records, digest


ARM_COMPONENTS = {"A": (), "B": ("reviewer",), "C": ("evolver",), "D": ("reviewer", "evolver")}
TARGETS = {"reviewer": "diagnose", "evolver": "propose"}


def check_state(state, adapter, force_pair):
    arm = state.get("arm")
    if arm not in ARM_COMPONENTS or getattr(adapter, "arm", arm) != arm:
        raise ValueError("The registered arm must match the deployed arm")
    if force_pair:
        raise ValueError("Co-evolution uses independent natural triggers only")
    if not isinstance(state.get("reviewer"), dict):
        raise ValueError("Missing deployed Reviewer policy")


def policy_versions(policies):
    return {role: digest(policies[role]) for role in TARGETS}


def previous_result(history):
    """The only previous optimizer receipt exposed alongside mechanism experiences."""
    event = history[-1]
    return dict(round=len(history), arm=event["arm"], event_hash=digest(event), status=event["status"],
                policies=event["optimizer_versions"], adopted=event.get("adopted", False),
                old_offspring_count=event.get("old_offspring_count"), new_offspring_count=event.get("new_offspring_count"))


class CoevolutionBatch(_Batch):
    def __init__(self, *args):
        super().__init__(*args)
        self.j.seal("starting_state", self.state)
        self.reviewer = self.state["reviewer"]
        self.arm = self.state["arm"]

    def _packet(self, name, packet):
        history = self.state["history"]
        if name == "H_pre" and history:
            packet = dict(packet, prior_optimizer_result=previous_result(history),
                          evidence_ids=[*packet["evidence_ids"], "pre.prior_optimizer_result"])
        elif packet.get("pre", {}).get("prior_optimizer_result") is not None:
            packet = dict(packet, evidence_ids=[*packet["evidence_ids"], "pre.prior_optimizer_result"])
        return super()._packet(name, packet)

    def _run_diagnosis(self, name, policy):
        return self.j.job(name, [self.mother_hash, digest(policy)], lambda: self.api.diagnose_with_policy(
            deepcopy(policy), self.original, deepcopy(self.memory), deepcopy(self.ledger), deepcopy(self.state["history"])))

    def _diagnosis(self):
        return self._run_diagnosis("pre_review", self.reviewer)

    def _propose(self, index, policy, packet, seed, previous, error):
        return self.api.propose_slot(deepcopy(policy), self.original, deepcopy(packet), seed, previous, error, index)

    def _decision(self, shared, attempt, prior_error):
        decisions = {}
        for role in ARM_COMPONENTS[self.arm]:
            policy = self.state[role]
            def decide():
                value = self.api.decide_component(role, deepcopy(policy), deepcopy(shared))
                if not isinstance(value, dict) or type(value.get("should_update")) is not bool:
                    raise ContractError("Invalid component self-decision")
                targets = [TARGETS[role]] if value["should_update"] else []
                ids = value.get("evidence_ids")
                if (value.get("target_components") != targets or not isinstance(ids, list) or not ids
                        or not all(isinstance(i, str) and i in shared["evidence_ids"] for i in ids)
                        or not all(isinstance(value.get(k), str) and value[k].strip() for k in ("reason", "expected_effect"))):
                    raise ContractError("Unbound component self-decision")
                return value
            try:
                decisions[role] = self.j.job(f"self_decisions/{role}", [digest(policy), digest(shared)], decide)
            except ContractError:
                decisions[role] = None
        if any(value is None for value in decisions.values()):
            raise ContractError("An eligible role did not return a valid self-decision")
        targets = [role for role, value in decisions.items() if value and value["should_update"]]
        evidence_ids = sorted({i for value in decisions.values() if value for i in value["evidence_ids"]})
        return dict(should_update=bool(targets), reason="Independent registered role decisions",
                    expected_effect="Compare requested revisions under the paired gate",
                    target_components=[TARGETS[role] for role in targets], evidence_ids=evidence_ids or ["primary.selection"],
                    component_decisions=decisions, arm=self.arm)

    def _revision(self, shared, decision):
        updated = {role: self.state[role] for role in TARGETS}
        rejected = []
        for role in ARM_COMPONENTS[self.arm]:
            choice = decision["component_decisions"][role]
            if not choice or not choice["should_update"]:
                continue
            current = self.state[role]
            def revise():
                result = self.api.revise_component(role, deepcopy(current), deepcopy(shared))
                if role == "evolver":
                    _check_policy(current, result)
                else:
                    from .coevolution_roles import validate_reviewer
                    validate_reviewer(result)
                    if result == current:
                        raise ContractError("Reviewer revision returned the same strategy")
                return result
            try:
                updated[role] = self.j.job(f"self_revisions/{role}", [digest(current), digest(shared)], revise)
            except ContractError:
                rejected.append(role)
        if rejected:
            raise ContractError("Rejected joint policy revision: " + ", ".join(rejected))
        return updated

    def _paired_proposals(self, new, shared, raw_shared, primary):
        # Each optimizer starts with the identical frozen parent, learning ledger
        # and scored primary records. No old Reviewer output substitutes for a
        # newly proposed diagnostic policy. The fixed post-review is public to both.
        entries = self.sampling.get("primary", {}).get("entries", [])
        records = _review_records(primary, entries, "primary", self.mother_hash, len(self.ledger))
        observation = [self.mother_hash, digest(records), digest(raw_shared["review"])]
        plans = (("O", self.reviewer, self.old), ("N", new["reviewer"], new["evolver"]))
        diagnoses, packets, bindings = {}, [], {}
        common_evidence = None
        for label, reviewer, evolver in plans:
            key = digest(reviewer)
            if key not in diagnoses:
                job = "paired_diagnoses/" + label
                diagnosis = self.j.job(job, [observation, key], lambda: self.api.diagnose_post_primary(
                    deepcopy(reviewer), self.original, deepcopy(self.memory), deepcopy(self.ledger),
                    deepcopy(self.state["history"]), deepcopy(records), primary_review=deepcopy(raw_shared["review"])))
                diagnoses[key] = (job, diagnosis)
            job, diagnosis = diagnoses[key]
            evidence = (diagnosis.get("raw_evidence_hash"), diagnosis.get("evidence_hash"))
            if not isinstance(evidence[0], str) or (common_evidence is not None and evidence != common_evidence):
                raise ValueError("Paired Reviewer diagnoses must use identical recorded evidence")
            common_evidence = evidence
            raw = deepcopy(raw_shared)
            raw["pre"]["diagnosis"] = diagnosis
            raw["diagnosis_phase"] = "post_primary"
            packet = self._packet("H_meta_old" if label == "O" else "H_meta_new", raw)
            bindings[label] = dict(reviewer_hash=key, evolver_hash=digest(evolver), diagnosis_job=job,
                                   diagnosis_hash=digest(diagnosis), packet_hash=digest(packet))
            packets.append((label, evolver, packet))
        self.j.seal("paired_diagnoses", dict(observation=observation, branches=bindings))
        return packets

    def _select_pair(self, meta):
        old, new, _, _ = super()._select_pair(meta)
        scores = {}
        for label, prefix in (("old", "O"), ("new", "N")):
            names = [f"{prefix}{i}" for i in (1, 2, 3) if meta[f"{prefix}{i}"].get("success_count") is not None]
            best = _choose(names, meta) if names else None
            scores[label + "_offspring_best"] = best
            scores[label + "_offspring_count"] = meta[best]["success_count"] if best else None
        left, right = scores["old_offspring_count"], scores["new_offspring_count"]
        changed = left is not None and right is not None and right > left
        from .panel import COEVOLUTION_MIXED_10X60_PROTOCOL
        if self.protocol == COEVOLUTION_MIXED_10X60_PROTOCOL:
            from .coevolution_roles import paired_family_outcomes
            scores["paired_optimizer_family_outcomes"] = None
            if left is not None and right is not None:
                scores["paired_optimizer_family_outcomes"] = paired_family_outcomes(
                    meta[scores["old_offspring_best"]]["evaluation"], meta[scores["new_offspring_best"]]["evaluation"])
        return old, new, changed, scores

    def _commit_new(self, new, changed):
        if changed:
            self.reviewer = new["reviewer"]
        return new["evolver"] if changed else self.old

    def _commit(self, name, winner, evolver, status, event):
        # Commit both policy versions in the same sealed receipt as the mechanism.
        deployed = {"reviewer": self.reviewer, "evolver": evolver}
        revision_path = self.j.root / "jobs/revision/result.json"
        revision = _read(revision_path)["outcome"] if revision_path.is_file() else None
        proposed = policy_versions(revision["value"]) if revision and revision["ok"] else None
        versions = dict(starting=policy_versions(self.state), proposed=proposed, deployed=policy_versions(deployed))
        self.state["reviewer"] = self.reviewer
        event = dict(event, selected_reviewer_hash=digest(self.reviewer), arm=self.arm,
                     selected_optimizer_hash=digest(deployed), optimizer_versions=versions)
        return super()._commit(name, winner, evolver, status, event)
