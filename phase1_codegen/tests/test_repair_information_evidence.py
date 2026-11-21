"""Input fidelity and finite behavior evidence; no mechanism quality assumptions."""
import copy
import unittest

from phase1_codegen.behavior_diagnostics import diagnose
from phase1_codegen.engine import observed_behavior_comparison, own_trace
from phase1_codegen.program import Program
from phase1_codegen.review import (_bounded_memory_data, _build_diagnostic_groups,
                                   bounded_review_payload, native_trace_witness)
from phase1_codegen.tests.test_review import ApproximateTokenizer
from phase1_codegen.tests.test_review_contract import contract_payload


SOURCE = '''def on_event(state, event, context):
    goal = event["visible"].get("goal", context.get("goal", ""))
    return {"state": state, "session": event["session"], "memory": [{"id": "one", "experience": goal}]}
'''


class InformationEvidenceTests(unittest.TestCase):
    def test_current_goal_and_verified_actor_receipt_survive_trace_budget(self):
        goal = 'Recover the latchkey from the bowl.'
        receipt = dict(actor_requests=80, direct_goal_verified_requests=80,
                       all_actor_requests_verified=True, request_receipts_hash='bound-request-hash')
        ep = dict(task_id='current', goal=goal, score=0, done=False, task_input_receipt=receipt,
                  history=[dict(action='go north', observation='ROOM ' + 'x' * 9000 + ' banana is here',
                                next_observation='RESPONSE ' + 'y' * 9000 + ' final native feedback') for _ in range(80)])
        original = copy.deepcopy(ep)
        result = native_trace_witness(own_trace(ep), ApproximateTokenizer())
        self.assertEqual(result['goal'], goal)
        self.assertEqual(result['task_input_receipt'], receipt)
        self.assertEqual(result['steps'][-1]['fields']['action']['text'], 'go north')
        self.assertTrue(result['steps'][-1]['fields']['observation']['suffix'].endswith('banana is here'))
        self.assertTrue(result['steps'][-1]['fields']['next_observation']['suffix'].endswith('final native feedback'))
        self.assertEqual(ep, original)

    def test_read_query_does_not_fabricate_actor_goal_delivery(self):
        payload = contract_payload()
        for row in payload['evidence']:
            if row['kind'] == 'execution_trace':
                row['data']['goal'] = 'current task goal'
            if row['kind'] == 'memory_readout':
                row['data'].setdefault('query', {})['goal'] = 'current task goal'
        shown = bounded_review_payload(payload, ApproximateTokenizer())
        traces = [row['data'] for row in shown['evidence'] if row['kind'] == 'execution_trace']
        self.assertTrue(traces)
        self.assertTrue(all(row['goal'] == 'current task goal' for row in traces))
        self.assertTrue(all('task_input_receipt' not in row for row in traces))

    def test_memory_tail_result_and_first_last_units_survive_without_splicing(self):
        rows = [{'id': str(i), 'experience': 'Goal: native goal ' + 'x' * 5000 + ' Native reward: 0.0'} for i in range(12)]
        data = dict(state={'rows': rows}, stored_tokens=9000)
        result, clipped = _bounded_memory_data('memory_storage', data, ApproximateTokenizer(), 2048, {})
        self.assertTrue(clipped)
        self.assertEqual(result['fragments'][0]['path'], ['state', 'rows', 0])
        self.assertEqual(result['fragments'][-1]['path'], ['state', 'rows', 11])
        self.assertGreater(result['omitted_unit_count'], 0)
        for item in result['fragments']:
            fragment = item['fragment']
            self.assertIn('Native reward: 0.0', fragment['suffix'])
            self.assertNotIn('Native reward: 0.0', fragment['text'])
            self.assertGreater(fragment['omitted_character_range'][1], fragment['omitted_character_range'][0])

    def _nested_default(self, visible):
        program = Program(SOURCE)
        event = dict(type='task_start', visible=visible, session={})
        receipt = program.run('event', {}, experience=event, context={}, diagnostics=True)
        behavior = dict(stored_state={}, read_trace=[dict(context={'task_id': 'one'},
            hook_trace=[dict(event=event, receipt=receipt)])], write_trace=[])
        diagnostics = diagnose(program, behavior, [])
        return receipt, diagnostics

    def test_evaluated_fallback_is_not_used_when_outer_goal_exists(self):
        receipt, diagnostics = self._nested_default({'goal': 'actual query'})
        self.assertEqual(receipt['value']['memory'][0]['experience'], 'actual query')
        missing = next(row for row in diagnostics['facts']['missing_dict_keys']
                       if row['expression'] == 'context.get("goal", "")')
        outer = missing['enclosing_default_calls'][0]
        self.assertEqual(outer['present_count'], 1)
        self.assertEqual(outer['missing_count'], 0)
        self.assertFalse(outer['samples'][0]['default_used'])
        self.assertEqual(outer['samples'][0]['result'], 'actual query')
        payload = contract_payload()
        payload['evidence'].append(dict(id='parent:behavior', kind='behavior_diagnostics', data=diagnostics))
        groups, _ = _build_diagnostic_groups(payload)
        group = next(g for g in groups if g['expression'] == 'context.get("goal", "")')
        self.assertEqual(group['enclosing_default_calls'], missing['enclosing_default_calls'])
        self.assertEqual(group['missing_witness']['result'], '')

    def test_missing_outer_goal_really_uses_empty_default(self):
        receipt, diagnostics = self._nested_default({})
        self.assertEqual(receipt['value']['memory'][0]['experience'], '')
        missing = next(row for row in diagnostics['facts']['missing_dict_keys']
                       if row['expression'] == 'context.get("goal", "")')
        sample = missing['enclosing_default_calls'][0]['samples'][0]
        self.assertTrue(sample['default_used'])
        self.assertEqual(sample['result'], '')

    def test_state_comparison_preserves_snapshot_order_even_after_json_archive_sorting(self):
        before = dict(context_hashes=['q1'], stored_state={'a': 1, 'b': 2}, readouts=[{}],
                      input_hashes=['slot-wire-a'], runtime_snapshot={'state_wire': '{"a":1,"b":2}'})
        after = copy.deepcopy(before)
        after['runtime_snapshot']['state_wire'] = '{"b":2,"a":1}'
        after['input_hashes'] = ['slot-wire-b']
        comparison = observed_behavior_comparison(before, after)
        self.assertFalse(comparison['stored_state_equal'])
        self.assertFalse(comparison['task_start_readouts_equal'])
        self.assertEqual(comparison['stored_state_comparison_source'], 'exact_snapshot_state_wire')
        self.assertEqual(comparison['readout_comparison_source'], 'exact_slot_wire_hashes')

    def test_finite_comparison_does_not_infer_unmeasured_dynamic_equivalence(self):
        before = dict(context_hashes=['q1', 'q2'], stored_state={'rows': []}, readouts=[{}, {}])
        after = copy.deepcopy(before)
        comparison = observed_behavior_comparison(before, after)
        self.assertTrue(comparison['comparison_complete'])
        self.assertTrue(comparison['stored_state_equal'])
        self.assertTrue(comparison['task_start_readouts_equal'])
        self.assertIn('not a later-event', comparison['boundary'])
        payload = contract_payload()
        payload['behavior_comparison'] = comparison
        shown = bounded_review_payload(payload, ApproximateTokenizer())
        facts = next(row['data'] for row in shown['evidence'] if row['id'] == 'current:facts')
        self.assertEqual(facts['behavior_comparison'], comparison)
        after['context_hashes'] = ['other', 'q2']
        incomplete = observed_behavior_comparison(before, after)
        self.assertFalse(incomplete['comparison_complete'])
        self.assertIsNone(incomplete['task_start_readouts_equal'])


if __name__ == '__main__':
    unittest.main()
