"""Real isolated Hook and bank operations, with synthetic ALFWorld episodes and no model."""
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from phase1_codegen.hooks import HookSession
from phase1_codegen.policy import Rejected
from phase1_codegen.program import Program
from rsi_paired import campaign
from rsi_paired.audit import Audit
from rsi_paired.native import Native, _archive_episode, _prefix
from rsi_paired.panel import COEVOLUTION_MIXED_10X60_PROTOCOL, MIXED_FAMILIES
from rsi_paired.workflow import digest
from tests.test_rsi_native import Tokens
from tests.test_rsi_family_campaign import PanelAdapter


SOURCE = """def on_event(state, event, context):
    ids = state.get("ids", [])
    if event["type"] == "task_end":
        ids = ids + [event["experience"]["task_id"]]
    state = {"ids": ids}
    if event["type"] == "task_start":
        entries = [{"id": item, "experience": item} for item in ids]
        return {"state": state, "session": {}, "memory": entries}
    return {"state": state, "session": {}, "memory": None}
"""


def tasks(prefix, order=MIXED_FAMILIES):
    return [dict(id=f"{prefix}/{f}", family=f, environment="alfworld",
                 archive_member=f"json_2.1.1/train/{f}/synthetic/game.tw-pddl") for f in order]


@pytest.mark.parametrize("arms", [("D",), ("initial", "final")])
def test_frozen_panel_entry_rejects_a_cross_family_snapshot(tmp_path, monkeypatch, arms):
    # Model the native archive at the point after wire indexing, then exercise
    # the actual frozen-panel summary and the auditor's bind_panel/bind_native.
    class IndexedAudit(Audit):
        def bind_panel(self, key, tasks, seeds, source, bank, outcomes):
            game, = tasks
            outcome, = outcomes
            outcome.update(family=game["family"])
            outcome["evidence"]["stored_tokens_before"] = 0
            source_hash = sha256(source.encode()).hexdigest()
            before = dict(program_hash=source_hash, state_wire="{}", context_wire='{"family":"clean"}',
                          capacity=512, read_budget=64, top_k=5, memory_scope="family",
                          source_log_hash=digest([]), stored_tokens=0, family=None, families=[])
            before["memory_hash"] = self._memory_hash(before)
            before["snapshot_hash"] = digest(before)
            after = dict(before, family="clean", families=["clean", "cool"],
                         source_log_hash=digest([before["source_log_hash"], outcome["episode_hash"]]))
            panel = digest(dict(tasks=tasks, seeds=seeds, pre=bank, program=source_hash, evaluator=self.native_fingerprint))
            binding = dict(evaluator=self.native_fingerprint, source=source_hash, initial=before["snapshot_hash"],
                           task=digest(game), seed=seeds[0], scope="panel", panel=panel, index=0)
            self.native_rows[("evaluation/"+key, 0)] = (binding, outcome, after, game["seed"], before["memory_hash"])
            super().bind_panel(key, tasks, seeds, source, bank, outcomes)

    monkeypatch.setattr(campaign, "Audit", IndexedAudit)
    adapter = PanelAdapter()
    adapter.protocol = COEVOLUTION_MIXED_10X60_PROTOCOL
    states = [dict(mechanism=SOURCE, evolver={}) for _ in arms]
    panel = [dict(id="heldout", family="clean", seed=13, split="valid_seen")]
    with pytest.raises(ValueError, match="Mixed evaluation reused a different task-family bank"):
        campaign._native_panel(tmp_path, adapter, states, panel, 11, arms=arms)


def test_mixed_live_hook_replay_frozen_writes_and_recovery_remain_type_local(tmp_path):
    native = object.__new__(Native)
    native.root, native.learning_archive = tmp_path, tmp_path / "learning"
    native.protocol, native.fingerprint = COEVOLUTION_MIXED_10X60_PROTOCOL, "synthetic"
    native.tokenizer, native.capabilities = Tokens(), None
    native.engine = SimpleNamespace(check_tasks=lambda rows: None)
    shown = {}

    def synthetic_episode(directory, program, bank, before, task, seed, label):
        family, ident = task["family"], task["id"]
        session = HookSession(bank, task, allow_write=True)
        session.emit("task_start", {"step": 0, "visible": {"obs": "synthetic"}})
        shown[ident] = [item["experience"] for item in session.current_memory.get("entries", [])]
        assert session.context["family"] == family
        ep = dict(task_id=ident, family=family, environment="alfworld", synthetic=True, phase=label, seed=seed,
                  program_hash=program.hash, goal="synthetic", native_reward_verified=True, score=1, history=[], error=None)
        session.finish(ep)
        session.publish()
        ep["memory_hook"] = session.summary()
        attempt = directory / "attempt-0"
        attempt.mkdir()
        canonical, actor = attempt / "canonical_episodes.jsonl", attempt / "actor.jsonl"
        canonical.write_text(json.dumps(ep) + "\n")
        actor.touch()
        ep_hash = _archive_episode(attempt, ep, native.learning_archive if label == "learning" else None)
        bank.source_log_hash = _prefix(bank.source_log_hash, ep_hash)
        return dict(row=dict(task_id=ident, family=family, status="completed", phase=label, score=1, native_score=1,
                             episode_hash=ep_hash), snapshot=bank.export_snapshot(), episode_wire="episode.json",
                    canonical_hash=sha256(canonical.read_bytes()).hexdigest(), actor_hash=sha256(b"").hexdigest(), attempt=attempt.name)

    native._execute = synthetic_episode
    ledger, memory = [], {}
    for row in tasks("a") + tasks("b", reversed(MIXED_FAMILIES)):
        native.set_job(tmp_path / "round" / row["id"])
        memory, out = native.learn(SOURCE, memory, row, 42)
        ledger.append(out)
    assert set(memory) == set(MIXED_FAMILIES)
    for f in MIXED_FAMILIES:
        assert shown[f"a/{f}"] == [] and shown[f"b/{f}"] == [f"a/{f}"]
        assert memory[f]["family"] == f and memory[f]["program_hash"] == Program(SOURCE).hash
    wrong = dict(memory, clean=memory["cool"])
    with pytest.raises((Rejected, ValueError)):
        native._bank(Program(SOURCE), wrong, "clean")

    # The new program replays all twelve registered episodes. Each individual
    # Hook sees only its family's snapshot and its own native event.
    changed = SOURCE.replace('state = {"ids": ids}', 'state = {"ids": ids, "new": True}')
    prepared = native.prepare(changed, SOURCE, memory, ledger)
    for family in MIXED_FAMILIES:
        snap = prepared[family]
        assert json.loads(snap["state_wire"]) == {"ids": [f"a/{family}", f"b/{family}"], "new": True}
        assert snap["source_log_hash"] == memory[family]["source_log_hash"] and snap["program_hash"] == Program(changed).hash

    # Independent candidate/final arms never share online panel writes. Restore
    # the same receipt twice to verify the indexed update survives resumption.
    original = digest(prepared)
    for name, start in (("candidate", prepared), ("frozen", {}), ("other-frozen", {})):
        panel = tasks(name+"-1") + tasks(name+"-2")
        native.set_job(tmp_path / name)
        results = native.evaluate(changed, start, panel, list(range(12)))
        assert native.evaluate(changed, start, panel, list(range(12))) == results
        for f in MIXED_FAMILIES:
            prefix = [f"a/{f}", f"b/{f}"] if start else []
            assert shown[f"{name}-1/{f}"] == prefix
            assert shown[f"{name}-2/{f}"] == [*prefix, f"{name}-1/{f}"]
        assert digest(prepared) == original
