"""Read-only RSI engineering-run audit. Usage: python -m rsi_paired.audit RUN [--require-complete]."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

from phase1_codegen.outcomes import candidate_hook_failure

from .workflow import (ContractError, LearningMechanismError, PROTOCOL, REPAIR_VERSION, _draft_seed, _public_episode,
                       _review_records, _seed, _validate_score, digest)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    with path.open("rb") as stream:
        for line in stream:
            if line.endswith(b"\n"):
                yield json.loads(line)


def file_hash(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def totals(rows):
    rows = list(rows)
    return dict(requests=len(rows), prompt_tokens=sum((r.get("prompt_tokens") or 0) for r in rows),
                completion_tokens=sum((r.get("completion_tokens") or 0) for r in rows),
                missing_usage=sum(r.get("prompt_tokens") is None or r.get("completion_tokens") is None for r in rows))


class Audit:
    def __init__(self, root):
        self.root, self.batch = root, root / "batch"
        self.errors = []
        self.slots = {}
        self.intents = {}
        self.native_rows = {}
        self.learning_failures = {}
        self.native_origins = set()
        self.native_verified = set()
        self.native_missing_wire = []
        self.ordered_wire_verified = 0
        self.native_fingerprint = None
        self.report = dict(status="partial", findings=self.errors, slots=self.slots, scores={}, costs={},
                           coverage={}, unresolved_jobs=[], barrier={})
        self.real = (root / "runtime.json").exists()

    def check(self, ok, message):
        if not ok and message not in self.errors:
            self.errors.append(message)
        return ok

    def load(self, path, default=None):
        full = self.batch / path
        return read(full) if full.exists() else default

    def outcome(self, job):
        rec = self.load("jobs/" + job + "/result.json")
        if rec is not None:
            self.check(digest(rec["outcome"]) == rec["outcome_hash"], f"Damaged job {job}")
            return rec["outcome"]

    def costs_and_wire(self):
        root = self.root
        role_costs = root / "roles/role_costs.jsonl"
        roles = list(lines(role_costs)) if role_costs.exists() else []
        by_role = {role: totals(r for r in roles if r["role"] == role) for role in sorted({r["role"] for r in roles})}
        for path in sorted((root / "roles/requests").glob("*/intent*.json")):
            item = read(path)
            self.intents.setdefault(path.parent.name, []).append(item)
        costs = self.report["costs"]
        costs.update(roles=by_role, role_intents=sum(map(len,self.intents.values())),
                     role_receipts_by_status=dict(Counter(r["status"] for r in roles)))
        actors = (r for path in (root/"native/episodes").glob("*/attempt-*/actor.jsonl")
                  for r in lines(path) if r.get("role") == "actor" and r.get("kind") == "model")
        costs["actor"] = totals(actors)
        caps = root/"capability_costs.jsonl"
        caps_rows = list(lines(caps)) if caps.exists() else []
        usage = Counter()
        for row in caps_rows:
            if row.get("physical"):
                usage.update({k:v for k,v in (row.get("usage") or {}).items()
                              if k in ("prompt_tokens","input_tokens","completion_tokens","total_tokens") and type(v) is int})
        costs["memory_model"] = dict(receipts=len(caps_rows), physical=sum(bool(r.get("physical")) for r in caps_rows),
                                   cache_hits=sum(bool(r.get("cache_hit")) for r in caps_rows), observed_usage=dict(usage))
        if self.real:
            self.native_fingerprint = read(root / "runtime.json").get("native_fingerprint")
        for path in (root/"native/episodes").glob("*/complete.json"):
            self._index_native(path)
        for path in (root/"native/episodes").glob("*/learning_failure.json"):
            self._index_native(path)
        self.check(len(self.native_origins) <= 1, "Native jobs come from different original run directories")
        costs["completed_native_episodes"] = len(self.native_rows) - len(self.learning_failures)
        costs["unscored_learning_failures"] = len(self.learning_failures)
        costs["native_attempt_directories"] = sum(1 for _ in (root/"native/episodes").glob("*/attempt-*"))
        self.report["native_wire"] = dict(verified=self.ordered_wire_verified,
                                          unavailable_legacy_panels=sorted(self.native_missing_wire))

    def _index_native(self, path):
        data = read(path)
        binding, result = data["binding"], data["result"]
        self.check(digest(result) == data["result_hash"] and digest(binding) == path.parent.name,
                   "Native episode completion identity changed")
        if not re.fullmatch(r"attempt-[0-9]+", result["attempt"]):
            raise ValueError("Invalid native attempt reference")
        attempt = path.parent / result["attempt"]
        canonical = attempt / "canonical_episodes.jsonl"
        self.check(file_hash(canonical) == result["canonical_hash"]
                   and file_hash(attempt / "actor.jsonl") == result["actor_hash"], "Native evidence files changed")
        # A partial second line is also inconsistent with a sealed, single-episode file.
        ep = json.loads(canonical.read_bytes())
        row, snapshot = result["row"], result["snapshot"]
        episode_hash = row["episode_hash"]
        if not isinstance(episode_hash, str) or not re.fullmatch("[a-f0-9]{64}", episode_hash):
            raise ValueError("Invalid native episode hash")
        version = binding["version"]
        self.check(type(version) is int and version in (1, 2), "Unknown native binding version")
        exact = attempt / "episode.json"
        if version == 2 or "episode_wire" in result:
            self.check(result.get("episode_wire") == exact.name and exact.is_file(),
                       "Missing or invalid native ordered episode witness")
        learning = binding["scope"] == "learning"
        archive = self.root / "native/learning" / (episode_hash + ".json")
        blocked = path.name == "learning_failure.json"
        if blocked:
            self.check(learning and not archive.exists() and not (path.parent / "complete.json").exists(),
                       "Blocked learning was also published as a completion or experience")
        elif learning:
            self.check(archive.is_file() and file_hash(archive) == episode_hash, "Native learning archive changed")
        witness = exact if exact.is_file() else archive if learning and archive.is_file() else None
        exact_bytes = witness.read_bytes() if witness else None
        ordered = json.loads(exact_bytes) if witness else ep
        # Program execution observes mapping insertion order. Historical canonical JSONL sorted its keys.
        wire = json.dumps(ordered, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        valid_wire = hashlib.sha256(wire).hexdigest() == episode_hash
        if version == 1 and not witness and not learning and not valid_wire:
            self.native_missing_wire.append(path.parent.name)
        else:
            if self.check(valid_wire and digest(ordered) == digest(ep) and
                          (witness is None or wire == exact_bytes), "Native ordered episode digest/content changed"):
                self.ordered_wire_verified += 1
        valid = (ep["task_id"] == row["task_id"] and ep["phase"] == row["phase"] == binding["scope"]
                 and ep["environment"] == row["environment"] == "alfworld" and ep["family"] == row["family"]
                 and ep["seed"] == binding["seed"] and ep["program_hash"] == binding["source"]
                 and ep["synthetic"] is False and row["native_score"] == ep["score"]
                 and ep.get("error") == row.get("error"))
        if blocked:
            from .native import _candidate_error
            valid &= (row["status"] == "deployed_learning_hook_failure" and row["score"] is None
                      and ep["executor"] == 'ALMA.Agent_Workflow.run_single_task' and ep['native_reward_verified'] is False
                      and ep["score"] is None and candidate_hook_failure(ep, binding["source"])
                      and row["candidate_error"] == _candidate_error(ep) and not ep["memory_hook"]["committed"]
                      and ep["generated_memory_write_calls"] == 0 and ep["memory_hash_before"] == ep["memory_hash_after"])
        elif row["status"] == "method_eval_failure":
            valid &= (not learning and row["score"] == 0 and ep["score"] is None
                      and candidate_hook_failure(ep, binding["source"]))
        else:
            recorder = ep["native_recorder"]
            reward = recorder["steps"][-1]["scores"] if recorder["steps"] else 0
            valid &= (type(row["score"]) in (int, float) and row["score"] in (0, 1)
                      and row["status"] == "completed" and not ep["error"] and ep["native_reward_verified"] is True
                      and row["score"] == reward == recorder["reward"] == ep["score"])
        self.check(valid, "A completed native task does not reproduce its reported outcome")
        self.check(snapshot["snapshot_hash"] == digest({k:v for k,v in snapshot.items() if k != "snapshot_hash"})
                   and snapshot["program_hash"] == binding["source"]
                   and snapshot["memory_hash"] == ep["memory_hash_after"]
                   and snapshot["memory_hash"] == self._memory_hash(snapshot), "Native result snapshot changed")
        # Native stores an absolute job path. Preserve identity after copying a complete run elsewhere.
        job_path = Path(binding["job"])
        parts = job_path.parts
        if not (job_path.is_absolute() and len(parts) >= 6 and parts[-5:-3] == ("batch", "jobs")
                and parts[-1] == "private" and parts[-3] in ("learning", "evaluation")):
            raise ValueError("Unexpected native job path")
        self.native_origins.add(job_path.parents[4])
        key = ("/".join(parts[-3:-1]), binding["index"])
        if blocked:
            from .native import learning_failure_evidence
            self.learning_failures[key] = learning_failure_evidence(path.parent, result)
        self.check(key not in self.native_rows, f"Duplicate native receipt for {key}")
        self.native_rows[key] = (binding, row, snapshot, ep["environment_seed"], ep["memory_hash_before"])

    @staticmethod
    def _memory_hash(snapshot):
        return digest({k:snapshot[k] for k in ("program_hash", "state_wire", "context_wire",
                                                "capacity", "read_budget", "top_k")})

    def bind_native(self, job, index, task, seed, source_hash, panel, bank, row=None):
        key = (job, index)
        if key not in self.native_rows:
            self.check(row is None, f"Missing native receipt for {job}/{index}")
            return bank
        self.native_verified.add(key)
        binding, actual, after, env_seed, before_hash = self.native_rows[key]
        family = task["family"]
        before = bank.get(family)
        if before is None:
            # For the family's first task, reconstruct CodeMemory's empty snapshot from its fixed context.
            before = dict(after, state_wire="{}", family=None, families=[], source_log_hash=digest([]),
                          stored_tokens=actual["evidence"]["stored_tokens_before"])
            before["memory_hash"] = self._memory_hash(before)
            before["snapshot_hash"] = digest({k:v for k,v in before.items() if k != "snapshot_hash"})
        expected = dict(evaluator=self.native_fingerprint, source=source_hash, initial=before["snapshot_hash"],
                        task=digest(task), seed=seed, scope="learning" if panel is None else "panel",
                        panel=panel, index=index)
        self.check(all(binding.get(k) == v for k,v in expected.items())
                   and actual["task_id"] == task["id"] and actual["family"] == family and env_seed == task["seed"]
                   and before_hash == before["memory_hash"] and json.loads(after["context_wire"])["family"] == family
                   and (row is None or digest(actual) == digest(row)), f"Native task binding mismatch for {job}/{index}")
        if actual["status"] in ("method_eval_failure", "deployed_learning_hook_failure"):
            self.check(after == before, f"Failed native candidate changed memory for {job}/{index}")
            return bank
        self.check(after["family"] == family and after["source_log_hash"] ==
                   digest([before["source_log_hash"], actual["episode_hash"]]), f"Native memory chain mismatch for {job}/{index}")
        return {**bank, family:after}

    def bind_panel(self, key, tasks, seeds, source, bank, outcomes):
        source_hash = hashlib.sha256(source.encode()).hexdigest()  # Program.hash hashes the original UTF-8 source.
        panel = digest(dict(tasks=tasks, seeds=seeds, pre=bank, program=source_hash, evaluator=self.native_fingerprint))
        for index, (task, seed) in enumerate(zip(tasks, seeds)):
            row = outcomes[index] if outcomes is not None else None
            bank = self.bind_native("evaluation/"+key, index, task, seed, source_hash, panel, bank, row)

    def _slot_wire(self, label, index, expected, packet, original, previous, error):
        from .roles import decode_role_response, proposal_source

        scope = f"batch/jobs/proposals/{label}/{index}/private"
        answer = []
        for key, intents in self.intents.items():
            item = intents[0]
            if item["role"] != "propose":
                continue
            binding = dict(role="propose", job=scope, model_fingerprint=item["model_fingerprint"],
                           policy_hash=item["policy_hash"], request=item["request"])
            if digest(binding) != key:
                continue
            for send in intents:
                self.check(all(send[k] == binding[k] for k in ("role", "model_fingerprint", "policy_hash", "request"))
                           and send["public_hash"] == digest(packet),
                           f"Retry changed actual proposal request {label}/{index}")
            request = binding["request"]
            body = json.loads(request["messages"][-1]["content"])
            feedback = {k: body[k] for k in ("own_draft", "own_contract_error") if k in body}
            self.check(digest(body["public_evidence"]) == digest(packet) and item["policy_hash"] == expected
                       and body["current_source"] == original
                       and feedback == ({} if index == 0 else dict(own_draft=previous, own_contract_error=error)),
                       f"Incorrect actual proposal input {label}/{index}")
            receipt = self.root / "roles/requests" / key / "response.json"
            evidence = dict(request_key=key, policy_hash=item["policy_hash"], public_hash=item["public_hash"],
                            seed=request.get("seed"), sends=len(intents), response_available=receipt.is_file())
            if receipt.is_file():
                # An identical retry shares a request key and one saved response. Never
                # attribute that response or its generated draft to each physical send.
                result = read(receipt)
                raw = result["choices"][0]["message"].get("content")
                evidence.update(response_raw_hash=digest(raw),
                                response_attribution="request_key" if len(intents) > 1 else "single_send")
                try:
                    source = proposal_source(decode_role_response(result, request["model"]))
                except ContractError as exc:
                    evidence.update(response_error=str(exc), response_draft_hash=digest(exc.draft))
                except RuntimeError as exc:
                    self.check(False, f"Invalid proposal response {label}/{index}: {exc}")
                else:
                    evidence["response_source_hash"] = digest(source)
            answer.append(evidence)
        self.check(len(answer) <= 1, f"Multiple proposal request keys for {label}/{index}")
        return answer

    def _job_binding(self, job, request):
        key = digest(request)
        rec = self.load(f"jobs/{job}/result.json")
        if rec:
            self.check(rec["request_hash"] == key, f"Wrong job inputs for {job}")
            for path in (self.batch/"jobs"/job).glob("attempt-*.json"):
                self.check(read(path)["request_hash"] == key, f"Wrong physical attempt inputs for {job}")

    def slot(self, name, packet, mother, old_hash, new_hash):
        row = self.load("slots/"+name+".json")
        if row is None:
            return
        prefix = name[0]
        policy = new_hash if prefix == "N" else old_hash
        expected = dict(parent_snapshot_hash=digest(mother), parent_framework_hash=digest(mother["framework"]),
                        learning_cutoff=mother["learning_cutoff"], source_memory_hash=digest(mother["memory"]),
                        requested_evolver_hash=policy, proposal_context_hash=digest(packet),
                        slot=name, phase={"P":"primary","O":"meta_old","N":"meta_new"}[prefix], slot_index=int(name[1:]))
        self.check(all(row.get(k) == v for k,v in expected.items()), f"Wrong mother, public packet or policy: {name}")
        drafts = row["drafts"]
        self.check(1 <= len(drafts) <= 4 and [d["index"] for d in drafts] == list(range(len(drafts)))
                   and all(d["error"] for d in drafts[:-1]), f"Repair allowance or ordering changed: {name}")
        self.check((row["valid"] and row["source_hash"] == digest(row["source"])
                    and row["source"] == drafts[-1]["source"] and drafts[-1]["error"] is None)
                   or (not row["valid"] and len(drafts) == 4 and bool(drafts[-1]["error"])
                       and row["source"] is None and row["source_hash"] is None), f"Invalid sealed slot: {name}")
        summary = {k: row[k] for k in (*expected,"seed","loaded_evolver_hash","valid","source_hash")}
        registration = self.load("registration.json")
        version = registration.get("repair_version", 1)
        slot_seed = _seed(registration["seed"], "primary" if prefix == "P" else "paired", int(name[1:])-1)
        self.check(type(row["seed"]) is int and row["seed"] == slot_seed, f"Wrong registered slot seed {name}")
        binding = dict(**expected, seed=slot_seed)
        attempts, previous, error = [], None, None
        for d in drafts:
            i = d["index"]
            label = f"{name}/{i}"
            generation_job, check_job = f"proposals/{label}", f"checks/{label}"
            generated, checked = self.outcome(generation_job), self.outcome(check_job)
            seed = _draft_seed(slot_seed, i) if version == REPAIR_VERSION else slot_seed
            req = dict(**binding, draft=i, previous_hash=digest(previous), error=error)
            if version == REPAIR_VERSION:
                req["draft_seed"] = seed
                self.check(type(d.get("seed")) is int and d["seed"] == seed, f"Wrong draft seed {label}")
            else:
                self.check("seed" not in d, f"Versioned draft in legacy registration {label}")
            self._job_binding(generation_job, req)
            wire = self._slot_wire(name, i, policy, packet, mother["framework"], previous, error)
            self.check(all(type(w["seed"]) is int and w["seed"] == seed for w in wire),
                       f"Wrong model request seed {label}")
            source = error = None
            provenance = "generation_receipt_missing"
            self.check(generated is not None, f"Missing proposal receipt {label}")
            if generated and generated["ok"]:
                value = generated["value"]
                source = value["source"]
                self._job_binding(check_job, [digest(source), digest(mother)])
                self.check(isinstance(source, str) and value["loaded_evolver_hash"] == policy
                           and value["public_hash"] == digest(packet) and checked is not None,
                           f"Unbound compiled draft {label}")
                if checked and not checked["ok"]:
                    error = checked["error"]
                    self.check(isinstance(error, str) and bool(error), f"Missing code check error {label}")
                if self.real:
                    self.check(bool(wire), f"Missing actual proposal request {label}")
                    self.check(any(w.get("response_source_hash") == digest(source) for w in wire),
                               f"Draft not found in actual model response {label}")
                provenance = "generated_source"
            elif generated is not None:
                raw, error = generated.get("draft"), generated["error"]
                self.check(isinstance(error, str) and bool(error), f"Missing generation error {label}")
                source = raw if isinstance(raw, str) and raw.strip() else None
                self.check(checked is None, f"Failed generation acquired a code check {label}")
                if wire:
                    self.check(any(w.get("response_error") == error and w.get("response_draft_hash") == digest(raw)
                                   for w in wire), f"Generation failure differs from actual response {label}")
                elif self.real and (raw is not None or str(error).startswith(("Role output incomplete", "Expected a JSON object",
                                                                         "Expected only a complete source string"))):
                    self.check(False, f"Missing actual proposal request {label}")
                provenance = "generation_error_with_draft" if source is not None else "generation_error_without_draft"
            self.check(error == d["error"], f"Wrong recorded draft error {label}")
            recorded = source
            if version == 1 and generated and not generated["ok"]:
                recorded = previous if raw is None else raw
                if raw is None and previous is not None:
                    # Legacy replay reports the old handoff separately from the
                    # actual source of this generation attempt (which is absent).
                    provenance = "legacy_carried_previous_draft"
                    self.report.setdefault("legacy_unbound_repairs", []).append(label)
            self.check(d["source"] == recorded, f"Wrong recorded draft source {label}")
            previous = recorded
            attempts.append(dict(index=i, seed=seed, passed=error is None,
                                 source_hash=None if source is None else digest(source),
                                 provenance=provenance, actual_requests=wire, error=error))
        self.check(row["error"] == error, f"Wrong terminal slot error {name}")
        if row["valid"]:
            self.check(row["loaded_evolver_hash"] == policy, f"Wrong loaded policy {name}")
        self.slots[name] = {**summary, "attempts": attempts}

    def panel(self, phase, registered, mother, primary):
        records = self.load(phase+"_results.json")
        seal = self.load("primary_seal.json" if phase == "primary" else "paired_seal.json")
        names = ["P1","P2","P3"] if phase == "primary" else ["O1","O2","O3","N1","N2","N3"]
        if seal is not None:
            expected = {n:digest(self.load("slots/"+n+".json")) for n in names}
            if phase == "primary":
                source = mother["framework"]
                expected["parent"] = digest(dict(valid=True, source=source, source_hash=digest(source)))
            self.check(seal == expected, f"{phase} seal differs from immutable slots")
        if records is None and (seal is None or not self.real):
            return None
        commons = ["parent"] if phase == "primary" else ["retained","parent"]
        if records is not None:
            self.check(seal is not None, f"{phase} scores without all proposals sealed")
            self.check(set(records) == set(commons+names), f"Unexpected {phase} membership")
        tasks = registered["panels"][phase]
        seeds = [int(digest([registered["seed"],"native",t["id"]])[:8], 16) % 2147483647 for t in tasks]
        scores, evals = {}, {}
        for n in commons+names:
            source = (mother["framework"] if n=="parent" else primary[0][primary[1]]["source"] if n=="retained"
                      else self.load("slots/"+n+".json")["source"])
            row = records[n] if records is not None else None
            if row is not None:
                self.check(row["source"] == source, f"Scored a different program: {phase}/{n}")
            if row is not None and not row["valid"]:
                self.check(row["success_count"] is None and "evaluation" not in row,
                           f"Invalid candidate has a native score: {phase}/{n}")
                scores[n] = None
                continue
            if source is None:
                continue
            if source == mother["framework"]:
                bank = mother["memory"]
            else:
                prepared = self.outcome("preparation/"+digest(source))
                if not prepared or not prepared["ok"]:
                    self.check(row is None, f"Missing prepared memory: {phase}/{n}")
                    continue
                bank = prepared["value"]
            key = digest([registered["protocol"], registered["adapter_fingerprint"], digest(source),
                          digest(bank), phase, tasks, seeds, "private-online-memory"])
            job_name = "evaluation/"+key
            job = self.outcome(job_name)
            if job and self.real:
                self.check(self.load("jobs/"+job_name+"/result.json")["request_hash"] == digest(key),
                           f"Evaluation job request mismatch: {phase}/{n}")
            if row is None:
                # Evaluation jobs are durable before the aggregate results file is sealed.
                outcomes = job["value"] if job and job["ok"] else None
                if outcomes is not None:
                    self.check(len(outcomes) == len(tasks), f"Incomplete native job: {phase}/{n}")
                self.bind_panel(key, tasks, seeds, source, bank, outcomes)
                continue
            self.check(key == row["eval_key"] and digest(bank) == row["pre_memory_hash"],
                       f"Wrong private pre-evaluation memory or gate: {phase}/{n}")
            self.check(job and job["ok"] and row["evaluation"] == job["value"], f"Unbound score receipt {phase}/{n}")
            outcomes = row["evaluation"]
            self.check([r["task_id"] for r in outcomes] == [t["id"] for t in tasks]
                       and all(type(r["score"]) in (int,float) and r["score"] in (0,1) for r in outcomes)
                       and sum(r["score"] for r in outcomes) == row["success_count"], f"Incomplete scores {phase}/{n}")
            if registered["protocol"] != PROTOCOL:
                for episode, task in zip(outcomes, tasks):
                    _validate_score(episode, task, classified=True)
            if self.real:
                self.bind_panel(key, tasks, seeds, source, bank, outcomes)
            scores[n] = row["success_count"]
            evals[n] = key
        if records is None:
            return None
        # The signed content relationship is verifiable after copying. The order check
        # below additionally uses original filesystem mtimes, which some copies discard.
        path = self.batch / ("primary_seal.json" if phase == "primary" else "paired_seal.json")
        times = [(self.batch/"jobs/evaluation"/k/"result.json").stat().st_mtime_ns for k in set(evals.values())]
        self.report["barrier"][phase] = dict(seal_matches_slots=(seal == expected if seal is not None else False),
            seal_precedes_first_panel_receipt_by_mtime=path.stat().st_mtime_ns <= min(times) if path.exists() and times else None)
        self.report["scores"][phase] = dict(success_counts=scores, tasks=len(tasks),
                                              valid_versions=len(evals), distinct_panel_executions=len(set(evals.values())))
        return records

    def _displayed_evidence(self, rows, omission):
        from .roles import _evidence_counts
        originals = {path + "/evidence": row["evidence"] for path, row, shown in rows
                     if "evidence" in row and isinstance(shown.get("evidence"), dict)}
        reduced = []
        for path, row, shown in rows:
            facts, visible = row.get("evidence"), shown.get("evidence")
            self.check(all(row.get(k) == shown.get(k) for k in
                           ("task_id", "score", "native_score", "episode_hash", "status")),
                       f"Shared evidence differs from its episode identity or native result: {path}")
            if "evidence_ref" in shown:
                ref = shown["evidence_ref"]
                self.check("evidence" not in shown and ref in originals and originals[ref] == facts,
                           f"Shared evidence reference differs from its episode: {path}/evidence")
            elif facts != visible:
                fingerprint = digest(facts)
                self.check(isinstance(facts, dict) and visible == _evidence_counts(facts)
                           and (shown.get("episode_hash") or shown.get("full_witness_hash") == fingerprint),
                           f"Directly displayed evidence differs from the archived facts: {path}")
                reduced.append([path, fingerprint])
        expected = ({"episode_witnesses_reduced": len(reduced), "witnesses_hash": digest(reduced)}
                    if reduced else None)
        self.check(omission == expected, "Displayed evidence omission receipt differs from the archived facts")

    def family_feedback(self, shared, primary, recent_rows):
        """Resolve any deduplicated public values against the original episode and source."""
        slots = shared["primary"]
        sources = {f"/primary/{name}/source": val["source"] for name, val in slots.items()
                   if isinstance(val.get("source"), str)}
        # An evidence reference must end at a directly displayed row. Equality
        # is checked on the inputs to compaction, including any prior H_pre omissions.
        pre = self.load("H_pre.json")
        recent = pre.get("recent_learning", [])
        visible = shared["pre"].get("recent_learning", [])
        self.check(len(recent_rows) == len(recent) == len(visible), "Shared feedback omits registered recent learning rows")
        self._displayed_evidence([(f"/recent_learning/{i}", _public_episode(row), shown)
                                  for i, (row, shown) in enumerate(zip(recent_rows, recent))], pre.get("omission"))
        rows = [(f"/pre/recent_learning/{i}", original, shown)
                for i, (original, shown) in enumerate(zip(recent, visible))]
        sampling = self.report.get("panel_sampling")
        for name, record in primary.items():
            view = slots[name]
            if "source_ref" in view:
                self.check("source" not in view and view["source_ref"] in sources
                           and sources[view["source_ref"]] == record["source"],
                           f"Shared source reference differs from the actual candidate: {name}")
            else:
                self.check(view.get("source") == record["source"],
                           f"Shared source differs from the actual candidate: {name}")
            if "evaluation" not in record:
                continue
            evaluated = record["evaluation"]
            shown = view.get("evaluation", [])
            expected = [(r["task_id"], sampling["primary"]["entries"][i]["source"] if sampling else None, r["score"])
                        for i, r in enumerate(evaluated)]
            labels = [(r.get("task_id"), r.get("gate_source"), r.get("score")) for r in shown]
            self.check(labels == expected, f"Shared feedback mislabels primary evaluation rows: {name}")
            rows.extend((f"/primary/{name}/evaluation/{i}", _public_episode(row), displayed)
                        for i, (row, displayed) in enumerate(zip(evaluated, shown)))
        self._displayed_evidence(rows, shared.get("omission"))

    def review_receipts(self, phase, records, mother, reported, retained=None):
        from .roles import _displayed_hash, _review_findings, _review_witness_index, _score_view, decode_role_response
        names = ["parent", "P1", "P2", "P3"] if phase == "primary" else [
            "retained", "parent", "O1", "O2", "O3", "N1", "N2", "N3"]
        entries = self.report.get("panel_sampling", {}).get(phase, {}).get("entries", [])
        published = _review_records({name: records[name] for name in names}, entries, phase, digest(mother),
                                    mother["learning_cutoff"], retained)
        records_hash = digest(published)
        evidence = {"parent.source": mother["framework"], **{f"{phase}.{k}": _score_view(r) for k, r in published.items()}}
        witnesses = _review_witness_index(evidence)
        errors = []
        value = None
        for job in ([f"reviews/{phase}"] if phase == "primary" else ["reviews/meta", "reviews/meta_correction"]):
            rec = self.outcome(job)
            if value is not None:
                self.check(not (self.batch / "jobs" / job).exists(), "A successfully returned meta review was corrected again")
                continue
            if not self.check(rec is not None, f"Missing scored-panel review receipt: {job}"):
                return
            prior = errors[-1] if errors else None
            self._job_binding(job, [records_hash, prior] if prior else records_hash)
            if self.real:
                # Bind the actual model request and response to this journal job.
                matched = []
                for key, sends in self.intents.items():
                    item = sends[0]
                    if item["role"] != "review":
                        continue
                    binding = {k: item[k] for k in ("role", "model_fingerprint", "policy_hash", "request")}
                    if digest(dict(binding, job=f"batch/jobs/{job}/private")) == key:
                        matched.append(key)
                if not self.check(len(matched) == 1, f"Unbound Reviewer model request: {job}"):
                    return
                key, = matched
                request = self.intents[key][0]["request"]
                body = json.loads(request["messages"][-1]["content"])
                kept = body.get("episode_witnesses", {})
                self.check(body["evidence"] == evidence and kept == dict(list(witnesses.items())[:len(kept)])
                           and body.get("omitted_witness_count", 0) == len(witnesses) - len(kept)
                           and body.get("format_correction") == prior,
                           f"Reviewer saw different evidence, attribution or correction: {job}")
                wire_hash = _displayed_hash(body)
                response = read(self.root / "roles/requests" / key / "response.json")
                try:
                    actual = _review_findings(decode_role_response(response, request["model"]), evidence)
                except ContractError as exc:
                    bad = dict(request_key=key, response_hash=digest(response),
                               output_hash=digest(response["choices"][0]["message"].get("content")),
                               displayed_evidence_hash=wire_hash)
                    self.check(not rec["ok"] and rec["error"] == str(exc) and rec["draft"] == bad,
                               f"Reviewer rejection differs from its model response: {job}")
                except RuntimeError as exc:
                    self.check(False, f"Invalid Reviewer response identity for {job}: {exc}")
                else:
                    self.check(rec["ok"] and rec["value"] == dict(actual, evidence_hash=wire_hash),
                               f"Reviewer opinion differs from its model response: {job}")
            if rec["ok"]:
                value = rec["value"]
            else:
                self.check(phase == "meta" and rec.get("error_kind") == "review_output",
                           f"A non-format failure was processed as a Reviewer correction: {job}")
                errors.append({"error": rec["error"], "previous_response": rec.get("draft")})
        if phase == "meta":
            value = dict(value or {"findings": []}, records_hash=records_hash, attempt_errors=errors,
                         status=("available", "corrected", "missing")[len(errors)])
        self.check(reported == value, f"Published {phase} review differs from counted attempts")

    def inspect(self):
        self.costs_and_wire()
        job_root = self.batch/"jobs"
        for folder in sorted({p.parent for p in job_root.rglob("attempt-*.json")}):
            if (folder/"result.json").exists():
                continue
            attempts = sorted(folder.glob("attempt-*.json"))
            last = read(attempts[-1])
            self.report["unresolved_jobs"].append(dict(job=folder.relative_to(job_root).as_posix(),
                    attempts=len(attempts), status=last["status"], error_type=last.get("error_type")))
        registration = self.load("registration.json")
        if not registration:
            self.report["stage"] = "awaiting_batch_registration"
            return
        from .panel import SEQUENTIAL_PROTOCOL
        v3 = registration["protocol"] == SEQUENTIAL_PROTOCOL
        self.check(v3 or registration["protocol"] == PROTOCOL, "Unknown registered batch protocol")
        if v3:
            decision_dirs = {p.name for p in (self.batch/"jobs/decision").glob("*") if p.is_dir()}
            self.check(decision_dirs <= {"0"}, "v3 allocated a second self-decision opportunity")
            reviews = {p.name for p in (self.batch/"jobs/reviews").glob("*") if p.is_dir()}
            self.check(reviews <= {"primary", "meta", "meta_correction"}, "v3 allocated an unregistered review")
            decisions = {k: sends for k, sends in self.intents.items() if sends[0]["role"] == "decide"}
            self.report["self_decision"] = dict(journal_slots=len(decision_dirs), request_keys=len(decisions),
                                                physical_sends=sum(map(len, decisions.values())))
            self.check(len(decisions) <= 1, "v3 sent multiple logical self-decision requests")
            for key, sends in decisions.items():
                for item in sends:
                    binding = {k: item[k] for k in ("role", "model_fingerprint", "policy_hash", "request")}
                    body = json.loads(item["request"]["messages"][-1]["content"])
                    self.check(digest(dict(binding, job="batch/jobs/decision/0/private")) == key
                               and body.get("public_evidence") == self.load("H_shared.json")
                               and set(body) == {"public_evidence", "current_strategy"},
                               "v3 decision request differs from its first sealed opportunity")
        if "repair_version" in registration:
            self.check(type(registration["repair_version"]) is int and registration["repair_version"] == REPAIR_VERSION,
                       "Unsupported repair receipt version")
        self.report["repair_version"] = registration.get("repair_version", 1)
        self.check(digest(registration["panels"]) == registration["panel_hash"], "Task panel hash mismatch")
        mother = self.load("parent_snapshot.json")
        starting = self.load("starting_state.json")
        if starting is not None:
            self.check(digest(starting) == registration["state_hash"], "Starting state differs from registration")
        learning_jobs = {p.parent.name: self.outcome(f"learning/{p.parent.name}") for p in
                         (self.batch/"jobs/learning").glob("*/result.json")}
        blocked = {n: out for n, out in learning_jobs.items() if out.get("error_kind") == "deployed_learning_hook"}
        self.report["learning_receipts"] = sum(out["ok"] for out in learning_jobs.values())
        if blocked:
            n, out = min(blocked.items())
            self.check(v3 and len(blocked) == 1 and n == max(learning_jobs) and not mother
                       and not (self.batch / "commit.json").exists(),
                       "A deployed learning failure advanced or committed the round")
            self.report["blocked_learning"] = out["evidence"]
        if self.real:
            from phase1_codegen.program import BASELINE_SOURCE
            source = starting["mechanism"] if starting is not None else BASELINE_SOURCE
            self.check(mother is None or mother["framework"] == source, "Mother uses the wrong starting mechanism")
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            bank = starting["memory"] if starting is not None else {}
            for i, task in enumerate(registration["panels"]["learning"]):
                job = f"learning/{i:03d}"
                outcome = self.outcome(job)
                seed = int(digest([registration["seed"], "native", task["id"]])[:8], 16) % 2147483647
                row = outcome["value"][1] if outcome and outcome["ok"] else None
                if outcome and outcome.get("error_kind") == "deployed_learning_hook":
                    self._job_binding(job, [digest(source), digest(bank), task, seed])
                    self.check(outcome["evidence"] == self.learning_failures.get((job, 0))
                               and outcome["error"] == str(LearningMechanismError(outcome["evidence"])),
                               f"Deployed learning failure differs from the native receipt: {job}")
                updated = self.bind_native(job, 0, task, seed, source_hash, None, bank, row)
                if row is not None:
                    self.check(outcome["value"][0] == updated, f"Learning job returned a different memory: {job}")
                    receipt = self.load("jobs/"+job+"/result.json")
                    self.check(receipt["request_hash"] == digest([digest(source), digest(bank), task, seed]),
                               f"Learning job request mismatch: {job}")
                bank = updated
        if not mother:
            self.report["stage"] = "blocked_deployed_learning_hook" if blocked else "collecting_learning_episodes"
            return
        self.check(mother["configuration_hash"] == digest(registration), "Mother differs from registration")
        recent = [self.outcome(f"learning/{i:03d}")["value"] for i in range(len(registration["panels"]["learning"]))]
        recent_rows = [val[1] for val in recent]
        earlier = starting["ledger"] if starting is not None else []
        self.check(digest(recent_rows) == mother["batch_trajectory_hash"]
                   and (starting is None or digest(earlier + recent_rows) == mother["learning_hash"])
                   and recent[-1][0] == mother["memory"], "Mother differs from the learning receipts")
        self.check(("resolved_panels_hash" in mother) == ("learning_history_tasks" in registration),
                   "Validation gate protocol differs from the registered selection mode")
        if "learning_history_tasks" in registration:
            from .panel import resolve_family_panels
            if not self.check(starting is not None, "Historical sampling lacks its starting learning ledger"):
                return
            plan = registration["panels"]
            resolved = resolve_family_panels(plan, registration["learning_history_tasks"] + plan["learning"],
                                             earlier + recent_rows, registration["seed"])
            if not self.check(self.load("resolved_panels.json") == resolved
                              and mother.get("resolved_panels_hash") == digest(resolved),
                              "Resolved validation gates differ from the registered learning outcomes"):
                return
            # The registration hash in the mother binds the original fresh
            # reserves. Native evaluation jobs bind the gates resolved above.
            registration = {**registration, "panels": resolved["panels"]}
            self.report["panel_sampling"] = resolved["sampling"]
        old_hash = None
        revision = self.outcome("revision")
        new_hash = digest(revision["value"]) if revision and revision["ok"] else None
        for intents in self.intents.values():
            item = intents[0]
            if item["role"] == "decide":
                old_policy = json.loads(item["request"]["messages"][-1]["content"])["current_strategy"]
                old_hash = digest(old_policy)
                if new_hash is not None:
                    new_policy = revision["value"]
                    self.check({k:v for k,v in old_policy.items() if k != "propose"} ==
                               {k:v for k,v in new_policy.items() if k != "propose"} and new_hash != old_hash,
                               "Revision changed fixed strategy fields or produced the old strategy")
                break
        first = self.load("slots/P1.json")
        if old_hash is None and first:
            old_hash = first["requested_evolver_hash"]
        if starting is not None and old_hash is not None:
            self.check(old_hash == digest(starting["evolver"]), "Proposal policy differs from the batch starting state")
        for prefix in "PON":
            packet = self.load("H_pre.json" if prefix == "P" else "H_shared.json")
            for i in (1,2,3):
                self.slot(f"{prefix}{i}",packet,mother,old_hash,new_hash)
        for i in (1,2,3):
            a,b = (self.slots.get(f"{p}{i}") for p in "ON")
            if a and b:
                self.check(a["seed"] == b["seed"], "Mismatch in paired slot seeds")
        for prefix in "PON":
            arm = [row["seed"] for label,row in self.slots.items() if label[0] == prefix]
            self.check(len(arm) == len(set(arm)), f"Repeated {prefix} slot seeds")
        primary = self.panel("primary",registration,mother,None)
        commit = self.load("commit.json")
        self.report["coverage"] = dict(raw_model_and_native=self.real, old_evolver_hash=old_hash,
                                      new_evolver_hash=new_hash, parent_snapshot_hash=digest(mother),
                                      primary_sealed=self.load("primary_seal.json") is not None,
                                      paired_sealed=self.load("paired_seal.json") is not None)
        if primary is None:
            self.report["stage"] = "primary_search_or_evaluation"
            return
        def winner(records, names):
            eligible = [n for n in names if records[n].get("success_count") is not None]
            return max(eligible, key=lambda n: records[n]["success_count"])
        keep = winner(primary,["parent","P1","P2","P3"])
        shared = self.load("H_shared.json")
        if shared is not None:
            self.check(shared.get("retained") == keep, "Shared feedback misstates the primary winner")
            for name, row in primary.items():
                displayed = shared["primary"].get(name, {})
                self.check(displayed.get("success_count") == row.get("success_count")
                           and displayed.get("source_hash") == row.get("source_hash"),
                           f"Shared feedback misstates primary candidate {name}")
            if v3:
                self.family_feedback(shared, primary, recent_rows)
                self.review_receipts("primary", primary, mother, shared["review"])
        meta = self.panel("meta",registration,mother,(primary,keep))
        if commit is None:
            self.report["stage"] = "decision_revision_or_paired_search" if meta is None else "awaiting_final_commit"
            if v3 and meta is not None and any(j["job"].startswith("reviews/meta") for j in self.report["unresolved_jobs"]):
                self.report["stage"] = "awaiting_meta_reviewer_recovery"
            return
        state = commit["state"]
        event = state["history"][-1]
        if v3 and meta is not None:
            self.check(event["status"] == "paired_complete", "Sealed meta scores cannot be discarded by fallback")
        self.check(digest(state) == commit["state_hash"] and event["primary_best"] == keep
                   and digest(state["ledger"]) == mother["learning_hash"]
                   and len(state["ledger"]) == event["learning_cutoff"] == mother["learning_cutoff"]
                   and event["forced_pair"] == registration["force_pair"], "Commit identity/learning cutoff mismatch")
        if starting is not None:
            self.check(state["history"][:-1] == starting["history"], "Commit changed prior batch history")
        self.check(event["parent_snapshot_hash"] == digest(mother) and
                   event["public_context_hash"] == digest(self.load("H_shared.json")), "Commit public packet mismatch")
        result = dict(status=event["status"], forced_engineering=event["forced_pair"], primary=keep)
        selected, records, policy_hash = keep, primary, old_hash
        if event["status"] == "paired_complete":
            if v3:
                self.review_receipts("meta", meta, mother, event["meta_review"], (keep, primary[keep]))
            old = winner(meta,["retained","parent","O1","O2","O3"])
            new = winner(meta,["retained","parent","N1","N2","N3"])
            adopted = meta[new]["success_count"] >= meta[old]["success_count"] + 1
            self.check((event["old_best"],event["new_best"],event["adopted"]) == (old,new,adopted),
                       "Decision does not reproduce on paired scores")
            selected, records = (new if adopted else old), meta
            policy_hash = new_hash if adopted else old_hash
            result.update(old_best=old,new_best=new,adopted=adopted)
        else:
            self.check(event["status"] in ("budget_blocked","not_triggered","decision_error",
                                          "invalid_self_revision","meta_incomplete"), "Unrecognized fallback")
        source = records[selected]["source"]
        self.check(event["selected"] == selected and state["mechanism"] == source
                   and event["selected_framework_hash"] == digest(source)
                   and digest(state["evolver"]) == event["selected_evolver_hash"] == policy_hash
                   and digest(state["memory"]) == records[selected]["pre_memory_hash"] == event["committed_memory_hash"],
                   "Deployed source, policy or memory differs from selected pre-evaluation state")
        decision = next((out["value"] for i in (0,1)
                         if (out := self.outcome(f"decision/{i}")) and out["ok"]), None)
        self.check(event["decision"] == decision, "Trigger decision disagrees with its durable receipt")
        if v3 and event["status"] == "decision_error":
            first = self.outcome("decision/0")
            self.check(first is not None and not first["ok"] and event["decision"] is None,
                       "Invalid v3 self-decision lacks its first and only rejection")
        if event["status"] == "not_triggered":
            self.check(decision is not None and not decision["should_update"] and not event["forced_pair"],
                       "Fallback does not follow original trigger decision")
        if event["status"] == "invalid_self_revision":
            self.check(revision is not None and not revision["ok"], "Invalid revision fallback lacks a rejection")
        completed = event["status"] in ("paired_complete", "not_triggered")
        status = ("verified" if self.real else "verified_workflow_only") if completed else "verified_fallback"
        self.report.update(status=status, stage="committed",
                           selection={**result,"selected":selected})


def audit_run(run_dir):
    audit = Audit(Path(run_dir))
    try:
        audit.inspect()
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        audit.errors.append(f"Unreadable or inconsistent receipt: {type(exc).__name__}: {exc}")
    if audit.real and audit.report["status"] in ("verified", "verified_fallback"):
        audit.check(set(audit.native_rows) == audit.native_verified, "Native completions outside registered jobs")
        if audit.native_missing_wire:
            audit.report["status"] = "unverified_native_wire"
    if audit.report.get("legacy_unbound_repairs") and audit.report["status"].startswith("verified"):
        audit.report["status"] = "unverified_legacy_repair_attribution"
    if audit.errors:
        audit.report["status"] = "audit_failed"
    return audit.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    report = audit_run(args.run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["status"] == "audit_failed" or args.require_complete and report["status"] != "verified")


if __name__ == "__main__":
    raise SystemExit(main())
