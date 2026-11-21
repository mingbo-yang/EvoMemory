"""Synthetic input-boundary checks, not native task effectiveness evidence."""
import copy
import unittest

from phase1_codegen.hooks import HookSession
from phase1_codegen.memory import CodeMemory
from phase1_codegen.policy import Rejected
from phase1_codegen.program import Program
from phase1_codegen.tests.test_engine import TinyTokenizer


SOURCE = '''def on_event(state, event, context):
    visible = event.get("visible", {})
    text = str(sorted(visible.keys())) + ":" + visible.get("obs", "")
    text = text + ":" + visible.get("goal", "missing")
    state["scratch"] = "not committed"
    return {"state": state, "session": event["session"],
            "memory": [{"id": "input-witness", "experience": text}]}
'''


class EventInputTests(unittest.TestCase):
    def setUp(self):
        self.bank = CodeMemory(Program(SOURCE), [], TinyTokenizer(),
                               dict(environment='alfworld', family='source',
                                    memory_scope='environment'))
        self.task = dict(id='synthetic-input-task', family='target', environment='alfworld')
        self.view = self.bank.for_task(self.task, 'alfworld')

    def test_diagnostic_matches_native_event_without_synthesized_goal(self):
        visible = {'obs': 'Your task is to: move a fixture.', 'actions_list': ['look']}
        original = copy.deepcopy(visible)
        before = self.bank.export_snapshot()
        hook = HookSession(self.view, self.task)
        hook.emit('task_start', dict(step=0, visible=visible))
        expected = copy.deepcopy(hook.current_memory['entries'])
        expected_event = copy.deepcopy(hook.trace[0]['event'])
        hook.close()

        actual = self.view.read({}, visible=visible)
        self.assertEqual(actual, expected)
        trace = self.bank.read_trace[-1]
        self.assertEqual(trace['hook_trace'][0]['event'], expected_event)
        self.assertEqual(trace['event_visible'], original)
        self.assertNotIn('goal', trace['event_visible'])
        self.assertEqual(trace['runtime_context']['family'], 'target')
        self.assertEqual(trace['runtime_context']['task_id'], self.task['id'])
        self.assertNotIn('visible', trace['runtime_context'])
        self.assertNotIn('observation', trace['runtime_context'])
        self.assertEqual(trace['input_origin'], 'external_visible')
        self.assertEqual(visible, original)
        self.assertEqual(self.bank.export_snapshot(), before)

    def test_explicit_empty_visible_does_not_fall_back_to_query(self):
        self.view.read({'goal': 'not an environment field', 'observation': 'not visible'}, visible={})
        trace = self.bank.read_trace[-1]
        self.assertEqual(trace['event_visible'], {})
        self.assertEqual(trace['hook_trace'][0]['event']['visible'], {})
        self.assertEqual(trace['raw_readout'][0]['experience'], '[]::missing')
        self.assertNotIn('goal', trace['runtime_context'])

    def test_supplied_goal_and_other_native_fields_survive_without_normalization(self):
        visible = dict(obs='current room', goal='explicit task', actions_list=['look'],
                       short_term_context='', action_dict={'a': 'left'})
        self.view.read({}, visible=visible)
        self.assertEqual(self.bank.read_trace[-1]['event_visible'], visible)
        self.assertEqual(self.bank.read_trace[-1]['hook_trace'][0]['event']['visible'], visible)

    def test_explicit_query_diagnostic_does_not_claim_native_event_origin(self):
        self.view.retrieve('explicit query', 'explicit observation')
        trace = self.bank.read_trace[-1]
        self.assertEqual(trace['event_visible'], {'goal': 'explicit query', 'obs': 'explicit observation'})
        self.assertEqual(trace['input_origin'], 'explicit_query_diagnostic')
        self.view.read({})
        self.assertEqual(self.bank.read_trace[-1]['event_visible'], {})

    def test_invalid_visible_input_rejected_before_worker_and_bank_mutation(self):
        before = self.bank.export_snapshot()
        for visible in ([], 'not an object', 0):
            with self.subTest(visible=visible), self.assertRaisesRegex(Rejected, 'visible input'):
                self.view.read({}, visible=visible)
        self.assertEqual(self.bank.read_trace, [])
        self.assertEqual(self.bank.export_snapshot(), before)


if __name__ == '__main__':
    unittest.main()
