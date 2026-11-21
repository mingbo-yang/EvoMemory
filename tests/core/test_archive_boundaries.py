"""Host archive growth must not relax or masquerade as worker input limits.

Synthetic CPU regression fixtures execute real isolated event programs. They
are boundary checks, not task-effect evidence or generated mechanism discoveries.
"""
import copy
import unittest

from phase1_codegen.hooks import HookSession
from phase1_codegen.memory import CodeMemory, digest
from phase1_codegen.policy import INPUT_BYTES, STATE_BYTES, Rejected, archive_copy, json_copy, validate_state
from phase1_codegen.program import Program
from tests.core.test_program_memory import CharacterTokenizer, fixture


SOURCE = '''def on_event(state, event, context):
    if event["type"] == "task_end":
        state["count"] = state.get("count", 0) + 1
    return {"state": state, "session": event.get("session", {}), "memory": None}
'''


class ArchiveBoundaryTests(unittest.TestCase):
    def bank(self, logs=()):
        return CodeMemory(Program(SOURCE), list(logs), CharacterTokenizer(),
                          dict(environment='alfworld', memory_scope='environment'))

    def test_cumulative_archive_larger_than_one_invocation_rebuilds(self):
        logs = [fixture('archive-one'), fixture('archive-two')]
        for row in logs:
            row['host_audit'] = 'x' * (INPUT_BYTES // 2 + 32)
        original = copy.deepcopy(logs)
        with self.assertRaisesRegex(Rejected, 'JSON byte budget'):
            json_copy(logs)
        bank = self.bank(logs)
        self.assertEqual(bank.state, {'count': 2})
        self.assertEqual(bank.source_log_hash, digest(logs))
        self.assertEqual(bank.stats['write_calls'], 2)
        self.assertEqual(logs, original)

    def test_one_large_historical_receipt_is_not_sent_to_the_program(self):
        bank = self.bank()
        episode = fixture('event-archive')
        task = dict(id=episode['task_id'], family=episode['family'], environment='alfworld')
        session = HookSession(bank, task, True)
        session.emit('task_start', dict(step=0, visible={'obs': 'synthetic visible data'}))
        session.finish(episode)
        session.publish()
        episode['memory_hook'] = session.summary()
        episode['memory_hook']['events'][0]['receipt']['host_audit'] = 'x' * INPUT_BYTES
        with self.assertRaisesRegex(Rejected, 'JSON byte budget'):
            json_copy(episode)
        rebuilt = self.bank([episode])
        self.assertEqual(rebuilt.state, bank.state)
        self.assertNotIn('host_audit', rebuilt.write_trace[0]['event_trace'][0]['event'])

    def test_whole_archive_node_count_is_not_a_worker_invocation(self):
        archive = {'audit': [0] * 100001}
        with self.assertRaisesRegex(Rejected, 'depth/node budget'):
            json_copy(archive)
        cloned = archive_copy(archive)
        self.assertEqual(cloned, archive)
        cloned['audit'][0] = 1
        self.assertEqual(archive['audit'][0], 0)

    def test_summary_and_commit_retain_large_host_receipts(self):
        bank = self.bank()
        episode = fixture('large-receipt')
        session = HookSession(bank, dict(id=episode['task_id'], family=episode['family'], environment='alfworld'), True)
        session.emit('task_start', dict(step=0, visible={'obs': 'small worker input'}))
        session.trace[0]['host_audit'] = 'x' * INPUT_BYTES
        session.finish(episode)
        session.publish()
        self.assertEqual(bank.state, {'count': 1})
        self.assertTrue(session.summary()['committed'])
        self.assertEqual(len(bank.write_trace[0]['event_trace'][0]['host_audit']), INPUT_BYTES)

    def test_live_event_keeps_two_mebibyte_input_limit(self):
        bank = self.bank()
        session = HookSession(bank, dict(id='oversize', family='synthetic-clean', environment='alfworld'))
        before = bank.export_snapshot()
        with self.assertRaisesRegex(Rejected, 'JSON byte budget'):
            session.emit('task_start', dict(visible={'obs': 'x' * INPUT_BYTES}))
        self.assertEqual(before, bank.export_snapshot())

    def test_persistent_state_keeps_its_own_limit(self):
        with self.assertRaisesRegex(Rejected, 'JSON byte budget'):
            validate_state({'text': 'x' * STATE_BYTES})

    def test_archives_remain_finite_exact_json(self):
        for value in ({'bad': float('nan')}, {'bad': float('inf')}, {1: 'wrong key'},
                      {'bad': {1, 2}}, {'bad': (1, 2)}):
            with self.subTest(value=type(value)), self.assertRaises(Rejected):
                archive_copy(value)
        cycle = []; cycle.append(cycle)
        with self.assertRaises(Rejected):
            archive_copy(cycle)


if __name__ == '__main__':
    unittest.main()
