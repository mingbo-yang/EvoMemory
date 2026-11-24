"""Crash-resumable, original-parent paired search. Adapters only receive public JSON or their own work."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from difflib import unified_diff
import fcntl
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


PROTOCOL = "rsi_original_parent_paired_v2"
SLOTS = 3
REPAIRS = 3
REPAIR_VERSION = 2


class ContractError(ValueError):
    """Candidate-attributable pre-score failure; preserve a malformed draft if available."""

    def __init__(self, message, draft=None):
        super().__init__(message)
        self.draft = draft


class ReviewOutputError(ContractError):
    """A returned Reviewer response failed its format or evidence-citation contract."""


class LearningMechanismError(RuntimeError):
    """The deployed mechanism failed during formal learning and its task remains unscored."""

    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(f"Deployed learning Hook failed on {evidence['task_id']}: {evidence['candidate_error']['message']}")


class InfrastructurePending(RuntimeError):
    """A job did not return a usable receipt. Partial results never acquire a score."""


class BudgetExhausted(RuntimeError):
    """A registered resource ceiling has terminally prevented completion."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)
    _sync_directory(path.parent)


@contextmanager
def _lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".batch.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class _Journal:
    def __init__(self, root, adapter, attempts):
        self.root = root
        self.adapter = adapter
        self.attempts = attempts

    def seal(self, name, value):
        path = self.root / (name + ".json")
        if path.exists():
            if canonical(_read(path)) != canonical(value):
                raise ValueError(f"Frozen input or receipt changed: {name}")
        else:
            _write(path, value)

    def job(self, name, request, callback):
        """Use the durable original slot. A retry after an uncertain call is billed separately."""
        directory = self.root / "jobs" / name
        key = digest(request)
        receipt = directory / "result.json"
        if receipt.exists():
            result = _read(receipt)
            if (result["request_hash"] != key
                    or digest(result["outcome"]) != result["outcome_hash"]):
                raise ValueError(f"Invalid job binding: {name}")
            return self._unwrap(result["outcome"])
        directory.mkdir(parents=True, exist_ok=True)
        previous = sorted(directory.glob("attempt-*.json"))
        for prior in previous:
            event = _read(prior)
            if event["request_hash"] != key:
                raise ValueError(f"Pending job input changed: {name}")
            if event["status"] == "started":
                event["status"] = "ambiguous_attempt"
                _write(prior, event)
        if len(previous) >= self.attempts:
            raise BudgetExhausted(f"Registered infrastructure attempts exhausted: {name}")
        attempt = directory / f"attempt-{len(previous) + 1:03d}.json"
        intent = {"status": "started", "request_hash": key}
        _write(attempt, intent)
        try:
            if hasattr(self.adapter, "set_job"):
                self.adapter.set_job(directory / "private")
            value = callback()
            # Normalize tuples etc. to ensure replay sees exactly what live execution saw.
            outcome = {"ok": True, "value": json.loads(canonical(value))}
        except LearningMechanismError as error:
            outcome = {"ok": False, "error_kind": "deployed_learning_hook", "error": str(error), "evidence": error.evidence}
        except ContractError as error:
            outcome = {"ok": False, "error": str(error), "draft": error.draft}
            if isinstance(error, ReviewOutputError):
                outcome["error_kind"] = "review_output"
        except BudgetExhausted:
            _write(attempt, {**intent, "status": "budget_exhausted"})
            raise
        except Exception as error:
            _write(attempt, {**intent, "status": "infrastructure_pending",
                             "error_type": type(error).__name__, "error": str(error)})
            raise InfrastructurePending(f"{name}: {error}") from error
        _write(receipt, {"request_hash": key, "outcome": outcome, "outcome_hash": digest(outcome)})
        status = "learning_hook_failure" if outcome.get("error_kind") == "deployed_learning_hook" else "finished"
        _write(attempt, {**intent, "status": status})
        return self._unwrap(outcome)

    @staticmethod
    def _unwrap(outcome):
        if not outcome["ok"]:
            if outcome.get("error_kind") == "deployed_learning_hook":
                raise LearningMechanismError(outcome["evidence"])
            cls = ReviewOutputError if outcome.get("error_kind") == "review_output" else ContractError
            raise cls(outcome["error"], draft=outcome.get("draft"))
        return outcome["value"]


def _seed(seed, *parts):
    return int(digest([seed, *parts])[:8], 16) % 2147483647


def _draft_seed(slot_seed, index):
    """Keep first drafts unchanged and match repair draws across paired slots."""
    return slot_seed if index == 0 else _seed(slot_seed, "repair", index)


def _validate_panels(panels, *, distinct_groups=True):
    if set(panels) != {"learning", "primary", "meta"}:
        raise ValueError("Expected exactly learning, primary and meta panels")
    fields = ("id", "group_id", "game_sha256") if distinct_groups else ("id", "archive_member", "game_sha256")
    for field in fields:
        seen = set()
        for panel in panels.values():
            if not isinstance(panel, list) or not panel:
                raise ValueError("All panels require a non-empty ordered task list")
            for task in panel:
                value = task.get(field)
                if field == "id" and (not isinstance(value, str) or not value):
                    raise ValueError("Missing task id")
                if value is not None:
                    if value in seen:
                        raise ValueError(f"Repeated task {field} across registered panels: {value}")
                    seen.add(value)


def _validate_score(row, task, *, classified=False):
    if (not isinstance(row, dict) or row.get("task_id") != task["id"]
            or type(row.get("score")) not in (float, int) or row["score"] not in (0, 1)):
        raise ValueError("A complete native evaluation must bind the registered task and a binary score")
    if classified:
        outcome = row.get("status")
        if outcome == "completed" and type(row.get("native_score")) in (int, float) and row["native_score"] == row["score"]:
            return
        if (outcome == "method_eval_failure" and "native_score" in row and row["native_score"] is None
                and row["score"] == 0 and isinstance(row.get("candidate_error"), dict) and row["candidate_error"]):
            return
        raise ValueError("Panel row lacks a verified native outcome or a documented candidate Hook penalty")


def _public_episode(row, *, for_review=False):
    # Never pass a private archive path, benchmark solution, engine metadata or hidden state.
    allowed = ("task_id", "family", "score", "done", "error", "steps", "visible_summary",
               "summary", "diagnosis", "candidate_error", "native_score", "status")
    result = {k: row[k] for k in (*allowed, "phase", "episode_hash") if k in row}
    fields = ("goal", "steps", "first_action", "last_action", "last_observation",
              "stored_tokens_before", "stored_tokens_after", "storage_sample_before",
              "storage_sample_after", "exposed_memory_changes", "injected_readouts", "events",
              "model_calls")
    if isinstance(row.get("evidence"), dict):
        result["evidence"] = {k: row["evidence"][k] for k in fields if k in row["evidence"]}
        if for_review and "review_witness" in row["evidence"]:
            result["evidence"]["review_witness"] = row["evidence"]["review_witness"]
    return result


def _experience(parent, label, row):
    """Persistent modification evidence. Full programs and trial receipts live at their immutable hashes."""
    draft = row.get("source")
    if draft is None and row.get("drafts"):
        draft = row["drafts"][-1]["source"]
    diff = list(unified_diff(parent.splitlines(), draft.splitlines(), lineterm="", n=1)) if isinstance(draft, str) else []
    # Store a bounded excerpt independently from the full diff identity. Explicitly mark both clipping modes.
    excerpt = [line[:200] for line in diff[:16]]
    return {"slot": label, "parent_hash": digest(parent), "draft_hash": digest(draft),
            "diff_hash": digest(diff), "diff_excerpt": excerpt, "omitted_lines": max(0, len(diff) - len(excerpt)),
            "truncated_lines": sum(len(line) > 200 for line in diff[:16]),
            "success_count": row.get("success_count"), "task_count": len(row.get("evaluation", [])),
            "error": row.get("error"), "slot_ref": "slots/" + label + ".json",
            "evaluation_key": row.get("eval_key")}


def _public_record(record, *, for_review=False):
    allowed = ("source", "source_hash", "valid", "drafts", "error", "success_count", "evaluation")
    result = {k: record[k] for k in allowed if k in record}
    if "evaluation" in result:
        result["evaluation"] = [_public_episode(row, for_review=for_review) for row in result["evaluation"]]
    return result


def _published_records(records, gate_sources, *, for_review=False):
    published = {key: _public_record(row, for_review=for_review) for key, row in records.items()}
    if gate_sources:
        for candidate in published.values():
            for row in candidate.get("evaluation", []):
                row["gate_source"] = gate_sources[row["task_id"]]
    return published


def _review_records(records, entries, phase, mother_hash, cutoff, retained=None):
    by_id = {entry["task_id"]: entry for entry in entries}
    published = _published_records(records, {i: e["source"] for i, e in by_id.items()}, for_review=True)
    for name, view in published.items():
        row = records[name]
        origin_name, origin = retained if name == "retained" else (name, row)
        source = row.get("source")
        view["provenance"] = dict(
            phase=phase, arm={"O": "old", "N": "new", "P": "primary"}.get(name[:1], "common"),
            origin="primary_winner" if name == "retained" else "deployment" if name == "parent" else "proposal",
            origin_slot=origin_name if origin_name != "parent" else None,
            requested_evolver_hash=origin.get("requested_evolver_hash"), loaded_evolver_hash=origin.get("loaded_evolver_hash"),
            program_hash=hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
            parent_snapshot_hash=mother_hash, learning_cutoff=cutoff, pre_eval_memory_hash=row.get("pre_memory_hash"),
            evaluation_key=row.get("eval_key"))
        for item in view.get("evaluation", []):
            entry = by_id.get(item["task_id"], {})
            if "episode_hash" in entry:
                item["historical_learning"] = dict(episode_hash=entry["episode_hash"], round=entry["learning_round"])
    return published


def _choose(names, records):
    valid = [name for name in names if records[name].get("success_count") is not None]
    if not valid:
        raise ValueError("No deployable common parent")
    return max(valid, key=lambda name: records[name]["success_count"])


def _check_policy(old, new):
    if not isinstance(new, dict) or "propose" not in new:
        raise ContractError("New Evolver does not carry a proposal strategy")
    original_fixed = {k: v for k, v in old.items() if k != "propose"}
    revised_fixed = {k: v for k, v in new.items() if k != "propose"}
    if canonical(original_fixed) != canonical(revised_fixed):
        raise ContractError("Only the proposal strategy may change")
    if digest(old) == digest(new):
        raise ContractError("Self-revision returned the original strategy unchanged")


class _Batch:
    def __init__(self, root, state, panels, adapter, seed, force_pair, budget, learning_history_tasks=None,
                 protocol=PROTOCOL):
        self.state, self.panels, self.api = state, panels, adapter
        self.seed, self.force_pair = seed, force_pair
        self.budget = budget
        self.j = _Journal(root, adapter, budget.get("infrastructure_attempts", 2))
        self.original = state["mechanism"]
        self.old = state["evolver"]
        self.memory = deepcopy(state["memory"])
        self.ledger = deepcopy(state["ledger"])
        self.prepared = {}
        self.gate_sources = {}
        self.sampling = {}
        self.fingerprint = str(getattr(adapter, "fingerprint", type(adapter).__qualname__))
        self.protocol = protocol
        registration = dict(protocol=protocol, repair_version=REPAIR_VERSION, state_hash=digest(state),
                            panel_hash=digest(panels), panels=panels, seed=seed, force_pair=force_pair,
                            budget=budget, adapter_fingerprint=self.fingerprint)
        self.learning_history_tasks = learning_history_tasks
        if learning_history_tasks is not None:
            registration["learning_history_tasks"] = learning_history_tasks
            self.j.seal("starting_state", state)
        self.j.seal("registration", registration)

    def resolve_panels(self, ledger):
        from .panel import resolve_family_panels
        history = self.learning_history_tasks + self.panels["learning"]
        return resolve_family_panels(self.panels, history, ledger, self.seed)

    def _packet(self, name, packet):
        value = (self.api.compress(deepcopy(packet)) if hasattr(self.api, "compress") else packet)
        self.j.seal(name, value)
        return value

    def _collect(self):
        recent = []
        for i, task in enumerate(self.panels["learning"]):
            seed = _seed(self.seed, "native", task["id"])
            request = [digest(self.original), digest(self.memory), task, seed]
            def learn():
                memory, row = self.api.learn(self.original, deepcopy(self.memory), deepcopy(task), seed)
                _validate_score(row, task)
                return memory, row
            self.memory, row = self.j.job(f"learning/{i:03d}", request, learn)
            recent.append(row)
            self.ledger.append(row)
        mother = dict(framework=self.original, memory=self.memory, learning_cutoff=len(self.ledger),
                      learning_hash=digest(self.ledger), batch_trajectory_hash=digest(recent),
                      configuration_hash=digest(_read(self.j.root / "registration.json")))
        if self.learning_history_tasks is not None:
            # Select from completed learning rows only. Proposal inputs include
            # the snapshot identity, without revealing the prospective gates.
            resolved = self.resolve_panels(self.ledger)
            self.j.seal("resolved_panels", resolved)
            mother["resolved_panels_hash"] = digest(resolved)
            self.panels = resolved["panels"]
            self.sampling = resolved["sampling"]
            self.gate_sources = {entry["task_id"]: entry["source"] for gate in resolved["sampling"].values()
                                 for entry in gate["entries"]}
        self.j.seal("parent_snapshot", mother)
        self.mother_hash = digest(mother)
        diagnosis = self.j.job("pre_review", self.mother_hash, lambda: self.api.diagnose(
            self.original, deepcopy(self.memory), deepcopy(self.ledger), deepcopy(self.state["history"])))
        packet = dict(phase="primary", parent_snapshot_hash=self.mother_hash, diagnosis=diagnosis,
                      history=deepcopy(self.state["history"]), recent_learning=[_public_episode(r) for r in recent],
                      evidence_ids=["pre.diagnosis", "pre.recent_learning", "pre.history"])
        return self._packet("H_pre", packet)

    def _slot(self, name, i, evolver, packet, phase):
        slot_seed = _seed(self.seed, "primary" if phase == "primary" else "paired", i)
        binding = dict(slot=name, phase=phase, slot_index=i+1, seed=slot_seed,
                       parent_snapshot_hash=self.mother_hash, parent_framework_hash=digest(self.original),
                       learning_cutoff=len(self.ledger), source_memory_hash=digest(self.memory),
                       requested_evolver_hash=digest(evolver), proposal_context_hash=digest(packet))
        drafts, previous, error = [], None, None
        source = None
        loaded = None
        for n in range(REPAIRS + 1):
            seed = _draft_seed(slot_seed, n)
            req = dict(**binding, draft=n, draft_seed=seed, previous_hash=digest(previous), error=error)
            def generate():
                response = self.api.propose(deepcopy(evolver), self.original, deepcopy(packet),
                                            seed, previous, error)
                if (response.get("loaded_evolver_hash") != binding["requested_evolver_hash"]
                        or response.get("public_hash") != binding["proposal_context_hash"]):
                    raise ValueError("Model request used a different Evolver or public evidence")
                if not isinstance(response.get("source"), str) or not response["source"].strip():
                    raise ContractError("Candidate response did not contain source text")
                return response
            try:
                proposal = self.j.job(f"proposals/{name}/{n}", req, generate)
            except ContractError as exc:
                previous = exc.draft if isinstance(exc.draft, str) and exc.draft.strip() else None
                error = str(exc)
            else:
                previous = proposal["source"]
                loaded = proposal["loaded_evolver_hash"]
                try:
                    self.j.job(f"checks/{name}/{n}", [digest(previous), self.mother_hash],
                               lambda: self.api.check(previous))
                    error = None
                except ContractError as exc:
                    error = str(exc)
            drafts.append(dict(index=n, seed=seed, source=previous, error=error))
            if error is None:
                source = previous
                break
        row = {**binding, "loaded_evolver_hash": loaded, "valid": source is not None, "source": source,
               "source_hash": digest(source) if source is not None else None, "drafts": drafts,
               "error": error}
        self.j.seal("slots/" + name, row)
        return row

    def _evaluate(self, records, panel_name):
        tasks = self.panels[panel_name]
        seeds = [_seed(self.seed, "native", t["id"]) for t in tasks]
        for name, row in records.items():
            if not row["valid"]:
                row["success_count"] = None
                continue
            source = row["source"]
            key = digest(source)
            try:
                if key not in self.prepared:
                    if source == self.original:
                        self.prepared[key] = self.memory
                    else:
                        self.prepared[key] = self.j.job(
                            f"preparation/{key}", [self.mother_hash, key],
                            lambda: self.api.prepare(source, self.original, deepcopy(self.memory),
                                                     deepcopy(self.ledger)))
            except ContractError as error:
                row.update(valid=False, error=str(error), success_count=None)
                continue
            memory = self.prepared[key]
            eval_key = digest([self.protocol, self.fingerprint, key, digest(memory),
                               panel_name, tasks, seeds, "private-online-memory"])
            def evaluate():
                results = self.api.evaluate(source, deepcopy(memory), deepcopy(tasks), list(seeds))
                if not isinstance(results, list) or len(results) != len(tasks):
                    raise ValueError("Incomplete native panel")
                for result, task in zip(results, tasks):
                    _validate_score(result, task, classified=self.protocol != PROTOCOL)
                return results
            rows = self.j.job(f"evaluation/{eval_key}", eval_key, evaluate)
            row.update(evaluation=rows, eval_key=eval_key, pre_memory_hash=digest(memory),
                       success_count=sum(int(item["score"]) for item in rows))
        self.j.seal(f"{panel_name}_results", records)

    def _review(self, phase, records, *, retained=None):
        if self.protocol == PROTOCOL:
            published = _published_records(records, self.gate_sources, for_review=True)
        else:
            entries = self.sampling.get(phase, {}).get("entries", [])
            published = _review_records(records, entries, phase, self.mother_hash, len(self.ledger), retained)
        records_hash = digest(published)
        errors = []
        recoverable = phase == "meta" and self.protocol != PROTOCOL
        try:
            review = self.j.job(f"reviews/{phase}", records_hash, lambda:
                                self.api.review(self.original, deepcopy(published)))
        except ReviewOutputError as first:
            if not recoverable:
                raise
            prior = {"error": str(first), "previous_response": first.draft}
            errors.append(prior)
            try:
                review = self.j.job("reviews/meta_correction", [records_hash, prior], lambda:
                                    self.api.review_retry(self.original, deepcopy(published), deepcopy(prior)))
            except ReviewOutputError as second:
                errors.append({"error": str(second), "previous_response": second.draft})
                review = {"findings": []}
        if recoverable:
            status = ("available", "corrected", "missing")[len(errors)]
            return dict(review, status=status, records_hash=records_hash, attempt_errors=errors)
        return review

    def _commit(self, name, winner, evolver, status, event):
        bank = self.prepared[digest(winner["source"])]
        event = {**event, "status": status, "selected": name, "selected_framework_hash": digest(winner["source"]),
                 "selected_evolver_hash": digest(evolver), "committed_memory_hash": digest(bank),
                 "learning_cutoff": len(self.ledger)}
        history = self.state["history"] + [event]
        state = {**self.state, "mechanism": winner["source"], "evolver": evolver,
                 "memory": bank, "ledger": self.ledger, "history": history}
        self.j.seal("commit", {"state_hash": digest(state), "state": state})
        return deepcopy(state)

    def run(self):
        parent = {"valid": True, "source": self.original, "source_hash": digest(self.original)}
        learning, primary_count, meta_count = (len(self.panels[n]) for n in ("learning", "primary", "meta"))
        if learning + 4 * primary_count > self.budget["episodes"]:
            raise BudgetExhausted("Insufficient complete primary-phase reservation")
        h_pre = self._collect()
        primary = {"parent": parent}
        for i in range(SLOTS):
            name = f"P{i+1}"
            primary[name] = self._slot(name, i, self.old, h_pre, "primary")
        self.j.seal("primary_seal", {k:digest(v) for k,v in primary.items()})
        self._evaluate(primary, "primary")
        keep = _choose(primary, primary)
        review = self._review("primary", primary)
        public = _published_records(primary, self.gate_sources)
        ids = ["pre.diagnosis", "post.review", "primary.selection", *["primary." + k for k in primary]]
        shared = self._packet("H_shared", dict(phase="post_primary", pre=h_pre, primary=public,
                              retained=keep, review=review, evidence_ids=ids))
        decision, decision_error = None, None
        for attempt in range(2 if self.protocol == PROTOCOL else 1):
            try:
                def decide():
                    if attempt and hasattr(self.api, "decide_retry"):
                        value = self.api.decide_retry(deepcopy(self.old), deepcopy(shared), decision_error)
                    else:
                        value = self.api.decide(deepcopy(self.old), deepcopy(shared))
                    if (not isinstance(value, dict) or type(value.get("should_update")) is not bool
                            or not isinstance(value.get("reason"), str) or not value["reason"].strip()
                            or not isinstance(value.get("evidence_ids"), list) or not value["evidence_ids"]
                            or not all(e in shared["evidence_ids"] for e in value["evidence_ids"])
                            or not isinstance(value.get("target_components"), list)
                            or not isinstance(value.get("expected_effect"), str)):
                        raise ContractError("Decision must cite registered public evidence and the required fields")
                    return value
                decision = self.j.job(f"decision/{attempt}", [digest(self.old), digest(shared), decision_error], decide)
                break
            except ContractError as exc:
                decision_error = str(exc)
        event = dict(parent_snapshot_hash=self.mother_hash, public_context_hash=digest(shared),
                     primary_best=keep, decision=decision, forced_pair=self.force_pair,
                     modification_records=[_experience(self.original,k,v) for k,v in primary.items() if k != "parent"],
                     primary_results={k: {s: v.get(s) for s in ("source_hash","success_count","error")}
                                      for k, v in primary.items()}, primary_review=review)
        def fallback(reason):
            return self._commit(keep, primary[keep], self.old, reason, event)
        if decision is None:
            return fallback("decision_error")
        if not self.force_pair and not decision["should_update"]:
            return fallback("not_triggered")
        used = learning + primary_count * len({v["eval_key"] for v in primary.values() if "eval_key" in v})
        if used + 8 * meta_count > self.budget["episodes"]:
            return fallback("budget_blocked")
        if hasattr(self.api, "reserve_pair") and not self.api.reserve_pair(deepcopy(self.budget)):
            return fallback("budget_blocked")
        try:
            def revision():
                value = self.api.revise(deepcopy(self.old), deepcopy(shared))
                _check_policy(self.old, value)
                return value
            try:
                new = self.j.job("revision", [digest(self.old), digest(shared)], revision)
            except ContractError:
                return fallback("invalid_self_revision")
            arms = {}
            for label, policy in (("O", self.old), ("N", new)):
                for i in range(SLOTS):
                    name = f"{label}{i+1}"
                    arms[name] = self._slot(name, i, policy, shared, "meta_old" if label=="O" else "meta_new")
            self.j.seal("paired_seal", {k:digest(v) for k,v in arms.items()})
            meta = {"retained": dict(valid=True, source=primary[keep]["source"],
                                     source_hash=primary[keep]["source_hash"]),
                    "parent": dict(valid=True, source=self.original, source_hash=digest(self.original)), **arms}
            self._evaluate(meta, "meta")
        except BudgetExhausted:
            return fallback("meta_incomplete")
        try:
            event["meta_review"] = self._review("meta", meta, retained=(keep, primary[keep]))
        except BudgetExhausted as exc:
            if self.protocol == PROTOCOL:
                return fallback("meta_incomplete")
            # All scores already exist. A transport failure gives no Reviewer
            # output to correct and cannot change which framework wins.
            raise InfrastructurePending(f"Meta scores sealed; Reviewer infrastructure requires recovery: {exc}") from exc
        old_name = _choose(["retained","parent","O1","O2","O3"], meta)
        new_name = _choose(["retained","parent","N1","N2","N3"], meta)
        changed = meta[new_name]["success_count"] >= meta[old_name]["success_count"] + 1
        chosen = new_name if changed else old_name
        event["modification_records"].extend(_experience(self.original,k,meta[k]) for k in arms)
        event.update(old_best=old_name, new_best=new_name, adopted=changed,
                     meta_results={k: {s:v.get(s) for s in ("source_hash","success_count","error")}
                                   for k,v in meta.items()})
        return self._commit(chosen, meta[chosen], new if changed else self.old, "paired_complete", event)


def run_batch(root, *, state, panels, adapter, seed=42, force_pair=False, budget=None,
              learning_history_tasks=None, protocol=None):
    """Run or resume one frozen batch. Its only published state is commit.json.

    Adapter callbacks: learn(source, banks, task, seed) -> (banks, episode);
    diagnose(source, banks, ledger, history); propose(policy, source, packet, seed,
    previous_source, contract_error) -> {source, loaded_evolver_hash, public_hash};
    check(source); decide(policy, packet); revise(policy, packet); prepare(source,
    original_source, original_banks, learning_ledger); evaluate(source, banks, tasks,
    seeds) -> ordered binary-scored episode summaries; review(source, public_records).
    The optional set_job(path) supplies private durable callback storage; compress
    seals one common bounded public packet. The optional reserve_pair(budget) checks
    non-episode resource reservations. A live adapter should bind its configuration,
    model, tokenizer, runtime and scorer in its fingerprint.

    For a sequential family campaign, panels contains fresh learning tasks and
    both gates' fresh reserves. Pass the previous learning tasks in ledger order
    (an empty list for round one). The observed learning outcomes determine
    the two sealed evaluation gates before any proposal is generated.
    """
    from .panel import SEQUENTIAL_PROTOCOL
    if protocol is None:
        protocol = SEQUENTIAL_PROTOCOL if learning_history_tasks is not None else PROTOCOL
    if protocol not in (PROTOCOL, SEQUENTIAL_PROTOCOL) or (learning_history_tasks is not None and protocol == PROTOCOL):
        raise ValueError("Unknown or incompatible batch protocol")
    if not {"mechanism","evolver","memory","ledger","history"} <= set(state):
        raise ValueError("Incomplete deployment state")
    if not isinstance(state["evolver"], dict) or "propose" not in state["evolver"]:
        raise ValueError("Deployment requires an explicit proposal strategy")
    if learning_history_tasks is None:
        _validate_panels(panels)
    else:
        from .panel import FAMILY_BATCH_SIZE
        if (not isinstance(learning_history_tasks, list)
                or [t["id"] for t in learning_history_tasks] != [row["task_id"] for row in state["ledger"]]):
            raise ValueError("Registered historical tasks differ from the starting learning ledger")
        if any(len(panels.get(phase, [])) != FAMILY_BATCH_SIZE for phase in ("learning", "primary", "meta")):
            raise ValueError("Unexpected family round learning or fresh gate reserve size")
        catalog = {**panels, "learning": learning_history_tasks + panels["learning"]}
        _validate_panels(catalog, distinct_groups=False)
        if len({t["family"] for tasks in catalog.values() for t in tasks}) != 1:
            raise ValueError("A family round cannot mix task families")
    prior_ids = {row["task_id"] for row in state["ledger"]}
    if any(task["id"] in prior_ids for panel in panels.values() for task in panel):
        raise ValueError("A registered task already appears in the deployed learning ledger")
    budget = dict(budget or {})
    budget.setdefault("episodes", len(panels["learning"])+4*len(panels["primary"])+8*len(panels["meta"]))
    for name, value in (("episodes",budget["episodes"]),("infrastructure_attempts", budget.get("infrastructure_attempts",2))):
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid positive budget: {name}")
    root = Path(root)
    with _lock(root):
        batch = _Batch(root, deepcopy(state), deepcopy(panels), adapter, seed, force_pair, budget,
                       deepcopy(learning_history_tasks), protocol)
        path = root/"commit.json"
        if path.exists():
            committed = _read(path)
            if digest(committed["state"]) != committed["state_hash"]:
                raise ValueError("Damaged deployment commit")
            if learning_history_tasks is not None:
                resolved = batch.resolve_panels(committed["state"]["ledger"])
                if (resolved != _read(root / "resolved_panels.json")
                        or digest(resolved) != _read(root / "parent_snapshot.json")["resolved_panels_hash"]):
                    raise ValueError("Frozen validation gates differ from the registered learning outcomes")
            return committed["state"]
        return batch.run()
