"""Synthetic contract checks, never effectiveness evidence."""
import copy
import unittest

from phase1_codegen.hooks import HookLimits, HookSession, replay_episode
from phase1_codegen.memory import CodeMemory
from phase1_codegen.policy import Rejected
from phase1_codegen.program import BASELINE_SOURCE, Program
from tests.core.test_engine import TinyTokenizer

SOURCE = '''
def on_event(state, event, context):
    session = event["session"]
    session["seen"] = session.get("seen", []) + [event["type"]]
    memory = None
    if event["type"] == "before_action":
        memory = [{"id": "synthetic", "experience": "fixture memory"}]
    if event["type"] == "task_end":
        state["episodes"] = [{"id": event["experience"]["task_id"], "experience": "fixture"}]
    return {"state": state, "session": session, "memory": memory}
'''


class HookTests(unittest.TestCase):
    def test_default_episode_allowance_permits_multiple_full_slots_and_remains_bounded(self):
        from phase1_codegen.protocol import READ_TOKENS, EPISODE_INJECTED_TOKENS
        source = '''
def on_event(state, event, context):
    session = event["session"]
    session["n"] = session.get("n", 0) + 1
    memory = [{"id": str(session["n"]), "experience": "x" * 1950}]
    return {"state": state, "session": session, "memory": memory}
'''
        hook = HookSession(self.bank(source), self.task())
        hook.emit('task_start', {})
        first = hook.usage['injected_tokens']
        self.assertGreater(first, READ_TOKENS // 2)
        hook.emit('before_action', {})
        self.assertEqual(hook.usage['injected_tokens'], 2 * first)
        self.assertEqual(hook.limits.injected_tokens, EPISODE_INJECTED_TOKENS)
        hook.emit('after_observation', {})
        hook.emit('before_action', {})
        with self.assertRaisesRegex(Rejected, 'injection budget'):
            hook.emit('after_observation', {})
        self.assertLessEqual(hook.usage['injected_tokens'], EPISODE_INJECTED_TOKENS)
        self.assertFalse(hook.summary()['committed'])

    def test_event_contract_rejects_extra_memory_fields(self):
        from phase1_codegen.policy import validate_output
        with self.assertRaises(Rejected):
            validate_output('event', dict(state={}, session={}, memory=[dict(id='a', experience='b', family='extra')]))

    def bank(self, source=SOURCE):
        return CodeMemory(Program(source), [], TinyTokenizer(), dict(environment='alfworld',
            memory_scope='environment', memory_hook_version='memory-events-v1'))

    def task(self):
        return dict(id='synthetic-task', family='synthetic-type', environment='alfworld')

    def episode(self):
        return dict(task_id='synthetic-task', family='synthetic-type', environment='alfworld',
                    goal='synthetic goal', history=[], score=0., done=True, error=None)

    def test_rejects_unknown_version(self):
        with self.assertRaisesRegex(Rejected, 'explicitly frozen'):
            CodeMemory(Program(SOURCE), [], TinyTokenizer(), {'memory_hook_version': 'unknown'})

    def test_actual_isolated_event_and_readonly(self):
        bank = self.bank()
        before = bank.export_snapshot()
        hook = HookSession(bank, self.task())
        hook.emit('task_start', dict(visible={'obs': 'only current information'}, step=0))
        self.assertEqual(hook.current_memory, {})
        hook.emit('before_action', dict(step=0))
        self.assertEqual(hook.current_memory['entries'][0]['id'], 'synthetic')
        hook.finish(self.episode())
        self.assertTrue(hook.summary()['complete'])
        self.assertFalse(hook.summary()['committed'])
        self.assertEqual(before, bank.export_snapshot())
        self.assertTrue(hook.trace[0]['receipt']['isolation']['seccomp'])
        self.assertNotIn('experience', hook.trace[0]['event'])

    def test_single_commit_and_exact_visible_event_rebuild(self):
        bank = self.bank()
        hook = HookSession(bank, self.task(), True)
        hook.emit('task_start', dict(visible={'obs': 'visible'}, step=0))
        hook.emit('before_action', dict(step=0))
        hook.finish(self.episode())
        self.assertEqual(bank.state, {})
        hook.publish()
        self.assertEqual(len(bank.write_trace), 1)
        ep = dict(self.episode(), memory_hook=hook.summary())
        rebuilt = self.bank()
        replay_episode(rebuilt, ep)
        self.assertEqual(bank.state, rebuilt.state)
        self.assertEqual(bank.fingerprint, rebuilt.fingerprint)
        with self.assertRaisesRegex(Rejected, 'already'):
            hook.finish(self.episode())

    def test_replay_does_not_inherit_previous_program_scratch(self):
        bank = self.bank()
        hook = HookSession(bank, self.task(), True)
        hook.emit('task_start', dict(step=0))
        hook.finish(self.episode())
        hook.publish()
        recorded = hook.summary()
        recorded['events'][0]['event']['session'] = {'injected_foreign_value': True}
        rebuilt = self.bank()
        replay_episode(rebuilt, dict(self.episode(), memory_hook=recorded))
        self.assertEqual(bank.state, rebuilt.state)

    def test_repeated_memory_does_not_clear_or_charge_twice(self):
        hook = HookSession(self.bank(), self.task(), limits=HookLimits(injected_tokens=100))
        hook.emit('task_start', {})
        hook.emit('before_action', {})
        initial, cost = copy.deepcopy(hook.current_memory), hook.usage['injected_tokens']
        hook.emit('after_observation', {})
        hook.emit('before_action', {})
        self.assertEqual(initial, hook.current_memory)
        self.assertEqual(cost, hook.usage['injected_tokens'])

    def test_error_never_commits_partial_state(self):
        source = SOURCE.replace('memory = None', 'state["partial"] = True\n    memory = None')
        bank = self.bank(source)
        hook = HookSession(bank, self.task(), True)
        before = bank.export_snapshot()
        hook.emit('task_start', {})
        hook.abort('native failure')
        self.assertFalse(hook.summary()['committed'])
        self.assertEqual(before, bank.export_snapshot())

    def test_contract_error_cannot_publish(self):
        source = SOURCE.replace('"session": session, "memory": memory', '"session": session, "memory": memory, "action": "forbidden"')
        bank = self.bank(source)
        hook = HookSession(bank, self.task(), True)
        with self.assertRaisesRegex(Rejected, 'exactly state'):
            hook.emit('task_start', {})
        self.assertEqual(bank.state, {})
        self.assertTrue(hook.failed)

    def test_episode_model_budget_checked_before_dispatch(self):
        source = SOURCE.replace('memory = None', 'value = model_call("fixture", {})\n    memory = None')
        bank = self.bank(source)
        hook = HookSession(bank, self.task(), limits=HookLimits(model_calls=0))
        with self.assertRaisesRegex(Rejected, 'model-call budget'):
            hook.emit('task_start', {})
        from phase1_codegen.outcomes import candidate_hook_failure
        self.assertEqual(hook.trace[-1]['error_type'], 'GeneratedBudgetExceeded')
        self.assertTrue(candidate_hook_failure(dict(error='HookExecutionError', program_hash=bank.program.hash,
                        memory_hook=hook.summary()), bank.program.hash))
        self.assertEqual(bank.state, {})

    def test_environment_boundary(self):
        task = dict(self.task(), environment='textworld')
        with self.assertRaisesRegex(Rejected, 'different environment'):
            HookSession(self.bank(), task)

    def test_history_import_ingests_raw_terminal_experience(self):
        bank = self.bank()
        bank.write(self.episode())
        self.assertEqual(bank.state['episodes'][0]['id'], self.task()['id'])

    def replay_recording(self):
        hook = HookSession(self.bank(), self.task(), True)
        hook.emit('task_start', dict(step=0))
        hook.emit('before_action', dict(step=0))
        hook.finish(self.episode())
        return dict(self.episode(), memory_hook=hook.summary())

    def test_replay_preflights_complete_event_order_before_execution(self):
        from unittest.mock import patch
        original = self.replay_recording()
        variants = []
        duplicate_end = copy.deepcopy(original)
        duplicate_end['memory_hook']['events'].append(copy.deepcopy(duplicate_end['memory_hook']['events'][-1]))
        variants.append(duplicate_end)
        end_not_last = copy.deepcopy(original)
        end_not_last['memory_hook']['events'].append(copy.deepcopy(end_not_last['memory_hook']['events'][1]))
        variants.append(end_not_last)
        duplicate_start = copy.deepcopy(original)
        duplicate_start['memory_hook']['events'].insert(1, copy.deepcopy(duplicate_start['memory_hook']['events'][0]))
        variants.append(duplicate_start)
        invalid_row = copy.deepcopy(original)
        invalid_row['memory_hook']['events'][-1]['status'] = 'rejected'
        variants.append(invalid_row)
        for index, bad in enumerate(variants):
            with self.subTest(index=index):
                bank = self.bank()
                before = bank.export_snapshot()
                with patch.object(Program, 'run', side_effect=AssertionError('must preflight before execution')) as run:
                    with self.assertRaises(Rejected):
                        replay_episode(bank, bad)
                run.assert_not_called()
                self.assertEqual(bank.export_snapshot(), before)

    def test_replay_binds_terminal_native_values_and_environment(self):
        from unittest.mock import patch
        original = self.replay_recording()
        for key, changed in [('task_id', 'another-task'), ('family', 'another-type'),
                             ('environment', 'textworld'), ('score', 1.)]:
            with self.subTest(key=key):
                bad = copy.deepcopy(original)
                bad['memory_hook']['events'][-1]['event']['experience'][key] = changed
                bank = self.bank()
                before = bank.export_snapshot()
                with patch.object(Program, 'run', side_effect=AssertionError('must preflight before execution')) as run:
                    with self.assertRaisesRegex(Rejected, 'outer native episode'):
                        replay_episode(bank, bad)
                run.assert_not_called()
                self.assertEqual(bank.export_snapshot(), before)
        foreign = copy.deepcopy(original)
        foreign['environment'] = 'textworld'
        foreign['memory_hook']['events'][-1]['event']['experience']['environment'] = 'textworld'
        with self.assertRaisesRegex(Rejected, 'different environment'):
            replay_episode(self.bank(), foreign)

    def test_replay_finish_failure_cannot_publish(self):
        from unittest.mock import patch
        original = self.replay_recording()
        bank = self.bank()
        before = bank.export_snapshot()
        finish = HookSession.finish

        def fail_after_finish(session, experience):
            finish(session, experience)
            self.assertEqual(bank.export_snapshot(), before)
            raise RuntimeError('synthetic failure before publication')

        with patch.object(HookSession, 'finish', fail_after_finish):
            with self.assertRaisesRegex(RuntimeError, 'before publication'):
                replay_episode(bank, original)
        self.assertEqual(bank.export_snapshot(), before)
        self.assertEqual(bank.write_trace, [])
        self.assertEqual(bank.stats['write_calls'], 0)

    def test_replay_worker_failure_never_commits_partial_state(self):
        original = self.replay_recording()
        source = SOURCE.replace('memory = None',
            'state["partial"] = True\n    memory = None\n'
            '    if event["type"] == "before_action":\n        memory = state["missing"]')
        bank = self.bank(source)
        before = bank.export_snapshot()
        with self.assertRaisesRegex(Rejected, 'missing'):
            replay_episode(bank, original)
        self.assertEqual(bank.export_snapshot(), before)
        self.assertEqual(bank.write_trace, [])

    def test_abort_is_idempotent_and_staged_finish_never_needs_rollback(self):
        bank = self.bank()
        before = bank.export_snapshot()
        hook = HookSession(bank, self.task(), True)
        hook.emit('task_start', {})
        hook.finish(self.episode())
        self.assertFalse(hook.summary()['committed'])
        hook.abort('native close failed')
        hook.abort('same failure returned by outer runner')
        self.assertEqual(sum(row['status'] == 'aborted' for row in hook.trace), 1)
        self.assertEqual(bank.export_snapshot(), before)
        self.assertEqual(hook.state, {})
        with self.assertRaisesRegex(Rejected, 'not publishable'):
            hook.publish()

    def test_pure_hook_history_import_is_only_explicit_terminal_event(self):
        source = '''def on_event(state, event, context):
    if event["type"] == "task_end":
        state["seen"] = state.get("seen", []) + [{"type": event["type"],
            "origin": event.get("origin", "native"), "mode": event.get("mode", "native"),
            "id": event["experience"]["task_id"]}]
    return {"state": state, "session": event["session"], "memory": None}
'''
        bank = self.bank(source)
        bank.write(self.episode())
        self.assertEqual(bank.state['seen'], [{'type': 'task_end', 'origin': 'history_import',
            'mode': 'history_import', 'id': 'synthetic-task'}])
        trace = bank.write_trace[0]
        self.assertEqual(trace['origin'], 'history_import')
        self.assertEqual(len(trace['event_trace']), 1)
        self.assertEqual(trace['event_trace'][0]['event']['type'], 'task_end')

    def test_diagnostic_read_closes_temporary_session_without_fake_terminal(self):
        from unittest.mock import patch
        bank = self.bank()
        before = bank.export_snapshot()
        closed = []
        close = HookSession.close
        def record_close(session):
            close(session)
            closed.append(session)
        with patch.object(HookSession, 'close', record_close):
            bank.read({'task_id': 'synthetic-task', 'family': 'synthetic-type', 'goal': 'visible'})
        self.assertEqual(len(closed), 1)
        self.assertTrue(closed[0].closed)
        self.assertFalse(closed[0].finished)
        self.assertEqual([row['event']['type'] for row in closed[0].trace], ['task_start'])
        self.assertEqual(closed[0].state, {})
        self.assertEqual(bank.export_snapshot(), before)

    def test_hook_uses_shared_readout_clipping(self):
        from phase1_codegen.memory import bound_readout
        source = '''def on_event(state, event, context):
    return {"state": state, "session": {}, "memory": [
        {"id": str(i), "experience": "long " * 1000} for i in range(6)]}
'''
        bank = self.bank(source)
        hook = HookSession(bank, self.task())
        hook.emit('task_start', {})
        rows = [{'id': str(i), 'experience': 'long ' * 1000} for i in range(6)]
        expected, tokens = bound_readout(rows, bank.tokenizer)
        self.assertEqual(hook.current_memory, expected)
        self.assertEqual(hook.usage['injected_tokens'], tokens)
        self.assertLessEqual(tokens, bank.read_budget)
        hook.close()


if __name__ == '__main__':
    unittest.main()
