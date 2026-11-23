"""Read-only RSI engineering-run audit. Usage: python -m rsi_paired.audit RUN [--require-complete]."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .workflow import digest


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
        for path in (root/"native/episodes").glob("*/complete.json"):
            data = read(path)
            res = data["result"]
            self.check(digest(res) == data["result_hash"] and digest(data["binding"]) == path.parent.name,
                       "Native episode completion identity changed")
            attempt = path.parent / res["attempt"]
            if Path(res["attempt"]).name != res["attempt"] or not res["attempt"].startswith("attempt-"):
                raise ValueError("Invalid native attempt reference")
            row = res["row"]
            canonical = attempt / "canonical_episodes.jsonl"
            self.check(file_hash(canonical) == res["canonical_hash"]
                       and file_hash(attempt / "actor.jsonl") == res["actor_hash"], "Native evidence files changed")
            episodes = list(lines(canonical))
            self.check(len(episodes) == 1, "Native attempt must have exactly one episode")
            ep = episodes[0]
            valid = (ep["task_id"] == row["task_id"] and ep["phase"] == row["phase"]
                     and ep["synthetic"] is False and row["native_score"] == ep["score"])
            if row["status"] == "method_eval_failure":
                valid &= row["score"] == 0 and bool(ep["error"])
            else:
                valid &= (not ep["error"] and ep["native_reward_verified"] is True
                          and row["score"] == ep["native_recorder"]["reward"] == ep["score"])
            self.check(valid, "A completed native task does not reproduce its reported outcome")
            self.native_rows.setdefault(row["episode_hash"], set()).add(digest(row))
        costs["completed_native_episodes"] = sum(1 for _ in (root/"native/episodes").glob("*/complete.json"))
        costs["native_attempt_directories"] = sum(1 for _ in (root/"native/episodes").glob("*/attempt-*"))

    def _slot_wire(self, label, index, expected, packet, original, previous, error):
        scope = f"batch/jobs/proposals/{label}/{index}/private"
        answer = []
        # Scope is bound by the original request digest; no branch label is placed in model-visible input.
        for key, intents in self.intents.items():
            item = intents[0]
            if item["role"] != "propose":
                continue
            request = item["request"]
            if key != digest(dict(role="propose", job=scope, model_fingerprint=item["model_fingerprint"],
                                  policy_hash=item["policy_hash"], request=request)):
                continue
            for item in intents:
                actual = json.loads(item["request"]["messages"][-1]["content"])
                self.check(digest(actual["public_evidence"]) == digest(packet) == item["public_hash"]
                           and item["policy_hash"] == expected and actual["current_source"] == original
                           and actual.get("own_draft") == previous and actual.get("own_contract_error") == error,
                           f"Incorrect actual proposal input {label}/{index}")
                receipt = self.root/"roles/requests"/key/"response.json"
                source_hash = None
                if receipt.exists():
                    choice = read(receipt)["choices"][0]
                    try:
                        obj = json.loads(choice["message"]["content"])
                    except (TypeError, ValueError):
                        obj = None
                    if choice.get("finish_reason") == "stop" and isinstance(obj, dict) and isinstance(obj.get("source"), str):
                        source_hash = digest(obj["source"])
                answer.append(dict(request_key=key, policy_hash=item["policy_hash"], public_hash=item["public_hash"],
                                   response_source_hash=source_hash, seed=item["request"].get("seed")))
        return answer

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
        attempts, previous, error = [], None, None
        for d in drafts:
            i = d["index"]
            generated = self.outcome(f"proposals/{name}/{i}")
            checked = self.outcome(f"checks/{name}/{i}")
            wire = self._slot_wire(name, i, policy, packet, mother["framework"], previous, error)
            self.check(generated is not None, f"Missing proposal receipt {name}/{i}")
            self.check(all(r["seed"] == row["seed"] for r in wire), f"Wrong model request seed {name}/{i}")
            if generated and generated["ok"]:
                value = generated["value"]
                self.check(value["source"] == d["source"] and value["loaded_evolver_hash"] == policy
                           and value["public_hash"] == digest(packet) and checked is not None
                           and checked["ok"] == (d["error"] is None), f"Unbound compiled draft {name}/{i}")
                if self.real:
                    self.check(bool(wire), f"Missing actual proposal request {name}/{i}")
                    self.check(any(r["response_source_hash"] == digest(d["source"]) for r in wire),
                               f"Draft not found in actual model response {name}/{i}")
            elif generated is not None:
                self.check(d["error"] == generated["error"] and not checked, f"Unbound failed proposal {name}/{i}")
            attempts.append(dict(index=i, passed=d["error"] is None, source_hash=digest(d["source"]),
                                 actual_requests=wire, error=d["error"]))
            previous, error = d["source"], d["error"]
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
        if records is None:
            return None
        self.check(seal is not None, f"{phase} scores without all proposals sealed")
        commons = ["parent"] if phase == "primary" else ["retained","parent"]
        self.check(set(records) == set(commons+names), f"Unexpected {phase} membership")
        tasks = registered["panels"][phase]
        seeds = [int(digest([registered["seed"],"native",t["id"]])[:8], 16) % 2147483647 for t in tasks]
        scores, evals = {}, {}
        for n in commons+names:
            row = records[n]
            source = (mother["framework"] if n=="parent" else primary[0][primary[1]]["source"] if n=="retained"
                      else self.load("slots/"+n+".json")["source"])
            self.check(row["source"] == source, f"Scored a different program: {phase}/{n}")
            if not row["valid"]:
                self.check(row["success_count"] is None and "evaluation" not in row,
                           f"Invalid candidate has a native score: {phase}/{n}")
                scores[n] = None
                continue
            bank = mother["memory"] if source == mother["framework"] else self.outcome("preparation/"+digest(source))["value"]
            key = digest([registered["protocol"], registered["adapter_fingerprint"], digest(source),
                          digest(bank), phase, tasks, seeds, "private-online-memory"])
            self.check(key == row["eval_key"] and digest(bank) == row["pre_memory_hash"],
                       f"Wrong private pre-evaluation memory or gate: {phase}/{n}")
            job = self.outcome("evaluation/"+key)
            self.check(job and job["ok"] and row["evaluation"] == job["value"], f"Unbound score receipt {phase}/{n}")
            outcomes = row["evaluation"]
            self.check([r["task_id"] for r in outcomes] == [t["id"] for t in tasks]
                       and all(type(r["score"]) in (int,float) and r["score"] in (0,1) for r in outcomes)
                       and sum(r["score"] for r in outcomes) == row["success_count"], f"Incomplete scores {phase}/{n}")
            if self.real:
                self.check(all(digest(r) in self.native_rows.get(r.get("episode_hash"), set()) for r in outcomes),
                           f"Unbound native score {phase}/{n}")
            scores[n] = row["success_count"]
            evals[n] = key
        # The signed content relationship is verifiable after copying. The order check
        # below additionally uses original filesystem mtimes, which some copies discard.
        path = self.batch / ("primary_seal.json" if phase == "primary" else "paired_seal.json")
        times = [(self.batch/"jobs/evaluation"/k/"result.json").stat().st_mtime_ns for k in set(evals.values())]
        self.report["barrier"][phase] = dict(seal_matches_slots=(seal == expected if seal is not None else False),
            seal_precedes_first_panel_receipt_by_mtime=path.stat().st_mtime_ns <= min(times) if path.exists() and times else None)
        self.report["scores"][phase] = dict(success_counts=scores, tasks=len(tasks),
                                              valid_versions=len(evals), distinct_panel_executions=len(set(evals.values())))
        return records

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
        self.check(digest(registration["panels"]) == registration["panel_hash"], "Task panel hash mismatch")
        mother = self.load("parent_snapshot.json")
        self.report["learning_receipts"] = len(list((self.batch/"jobs/learning").glob("*/result.json")))
        if not mother:
            self.report["stage"] = "collecting_learning_episodes"
            return
        self.check(mother["configuration_hash"] == digest(registration), "Mother differs from registration")
        recent = [self.outcome(f"learning/{i:03d}")["value"] for i in range(len(registration["panels"]["learning"]))]
        self.check(digest([val[1] for val in recent]) == mother["batch_trajectory_hash"]
                   and recent[-1][0] == mother["memory"], "Mother differs from the learning receipts")
        if self.real:
            self.check(all(digest(val[1]) in self.native_rows.get(val[1].get("episode_hash"),set())
                           for val in recent), "Missing native learning receipt")
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
        meta = self.panel("meta",registration,mother,(primary,keep))
        if commit is None:
            self.report["stage"] = "decision_revision_or_paired_search" if meta is None else "awaiting_final_commit"
            return
        state = commit["state"]
        event = state["history"][-1]
        self.check(digest(state) == commit["state_hash"] and event["primary_best"] == keep
                   and digest(state["ledger"]) == mother["learning_hash"]
                   and len(state["ledger"]) == event["learning_cutoff"] == mother["learning_cutoff"]
                   and event["forced_pair"] == registration["force_pair"], "Commit identity/learning cutoff mismatch")
        self.check(event["parent_snapshot_hash"] == digest(mother) and
                   event["public_context_hash"] == digest(self.load("H_shared.json")), "Commit public packet mismatch")
        result = dict(status=event["status"], forced_engineering=event["forced_pair"], primary=keep)
        selected, records, policy_hash = keep, primary, old_hash
        if event["status"] == "paired_complete":
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
