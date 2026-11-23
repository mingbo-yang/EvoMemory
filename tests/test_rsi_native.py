"""CPU contract checks. Constructed episodes below are fixtures, not research outcomes."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from phase1_codegen.hooks import HookSession
from phase1_codegen.program import Program
from rsi_paired.native import BASELINE_SOURCE, Native, _evidence, _prefix, _wire


class Tokens:
    def count(self, value):
        return len(value.split())

    def clip(self, value, limit):
        return ' '.join(value.split()[:limit])


class NativeMemoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.native = object.__new__(Native)
        self.native.root = Path(temp.name)
        self.native.tokenizer = Tokens()
        self.native.capabilities = None
        self.native.fingerprint = 'fixture'
        self.native.engine = SimpleNamespace(check_tasks=lambda tasks: None)

    def archive_learning(self, family, task_id):
        task = {'id': task_id, 'family': family, 'environment': 'alfworld'}
        bank = self.native._bank(Program(BASELINE_SOURCE), {}, family)
        session = HookSession(bank, task, allow_write=True)
        ep = dict(task_id=task_id, family=family, environment='alfworld', phase='learning',
                  native_reward_verified=True, score=0.0, error=None, goal='fixture observation',
                  seed=42, done=False, history=[])
        session.emit('task_start', dict(step=0, visible={'obs': 'visible fixture'}))
        session.finish(ep)
        session.publish()
        ep['memory_hook'] = session.summary()
        raw = _wire(ep)
        identity = hashlib.sha256(raw).hexdigest()
        path = self.native.root/'learning'/f'{identity}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        bank.source_log_hash = _prefix(bank.source_log_hash, identity)
        row = dict(task_id=task_id, family=family, score=0, phase='learning', episode_hash=identity)
        return bank.export_snapshot(), row

    def test_identical_source_reuses_and_changed_source_replays_same_family_prefix(self):
        ice, a = self.archive_learning('cool', 'cool/a')
        heat, b = self.archive_learning('heat', 'heat/b')
        memory = {'heat': heat, 'cool': ice}
        self.assertEqual(self.native.prepare(BASELINE_SOURCE, BASELINE_SOURCE, memory, [a, b]), memory)
        candidate = '''def on_event(state, event, context):
    if event["type"] == "task_end":
        state["ids"] = state.get("ids", []) + [event["experience"]["task_id"]]
    return {"state": state, "session": {}, "memory": None}
'''
        prepared = self.native.prepare(candidate, BASELINE_SOURCE, memory, [a, b])
        self.assertEqual(json.loads(prepared['cool']['state_wire']), {'ids': ['cool/a']})
        self.assertEqual(json.loads(prepared['heat']['state_wire']), {'ids': ['heat/b']})
        self.assertEqual(prepared['heat']['source_log_hash'], heat['source_log_hash'])
        self.assertNotEqual(prepared['heat']['program_hash'], heat['program_hash'])
        with self.assertRaisesRegex(ValueError, 'non-learning'):
            self.native.prepare(candidate, BASELINE_SOURCE, memory, [dict(a, phase='panel'), b])
        with self.assertRaisesRegex(ValueError, 'learning prefix'):
            self.native.prepare(candidate, BASELINE_SOURCE, memory, [dict(a, episode_hash='0'*64), b])

    def test_panel_updates_private_bank_in_order_and_does_not_modify_input(self):
        initial = {'cool': {'value': 8}}
        calls = []

        def episode(program, bank, task, seed, phase, panel, index):
            self.assertEqual(phase, 'panel')
            calls.append((bank['cool']['value'], panel, seed))
            return {'cool': {'value': bank['cool']['value']+1}}, {'task_id': task['id'], 'score': 0}

        self.native._episode = episode
        rows = self.native.evaluate(BASELINE_SOURCE, initial, [{'id': 'a'}, {'id': 'b'}], [40, 42])
        self.assertEqual([r['task_id'] for r in rows], ['a', 'b'])
        self.assertEqual([(x[0], x[2]) for x in calls], [(8, 40), (9, 42)])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(initial['cool']['value'], 8)

    def test_witness_requires_real_delivery_and_tracks_emptying_memory(self):
        bank = {'state_wire': '{}', 'stored_tokens': 1}
        delivered = {'entries': [{'id': 'h', 'experience': 'visible evidence'}]}
        ep = {'history': [], 'memory_hook': {'injections': [
            {'actor_delivery_verified': False, 'memory': delivered},
            {'actor_delivery_verified': True, 'memory': delivered, 'step': 1},
            {'actor_delivery_verified': True, 'memory': delivered, 'step': 2},
            {'actor_delivery_verified': True, 'memory': {}, 'step': 3}]}}
        evidence, cost = _evidence(ep, bank, bank, [])
        self.assertEqual(evidence['exposed_memory_changes'], 2)
        self.assertEqual([s['sample'] for s in evidence['injected_readouts']], ['visible evidence', ''])
        self.assertEqual(cost['actor_calls'], 0)
