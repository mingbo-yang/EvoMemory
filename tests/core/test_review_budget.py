"""Mandatory evidence is allocated as a set, preserving all task results."""
import copy
import unittest
from unittest.mock import patch
from evomem.common import digest
from phase1_codegen.roles import reviewer as review
from tests.core.test_review import ApproximateTokenizer
from tests.core.test_review_contract import contract_payload


class JointReviewBudgetTests(unittest.TestCase):
    def test_joint_packet_preserves_long_goal_trajectories_and_delivered_evidence(self):
        payload = contract_payload()
        tokenizer = ApproximateTokenizer()
        traces = [r for r in payload['evidence'] if r['kind'] == 'execution_trace']
        for row, step_count in zip(traces, (59, 80)):
            side = row['id'].split(':')[0]
            row['data'].update(
                goal='public goal component ' * 320,
                program_hash=side + '-hash',
                provenance=dict(program_hash=side + '-hash', memory_snapshot_hash='snapshot-2'),
                task_input_receipt=dict(version='textworld-goal-delivery-v2', delivery_verified=True),
                history=[dict(action=f'{side} action {step}', observation=f'room {step}',
                              next_observation=f'feedback {step}') for step in range(step_count)])
            with self.assertRaisesRegex(ValueError, 'witness budget'):
                review.native_trace_witness(row['data'], tokenizer, limit=1536)

        shown = review.bounded_review_payload(payload, tokenizer)
        self.assertLessEqual(tokenizer.chat_count(review._messages(shown)), review.MAX_INPUT_TOKENS)
        visible = {r['id']: r['data'] for r in shown['evidence']}
        registry = shown['citation_registry']
        for row in traces:
            raw = row['data']
            witness = visible[row['id']]
            for key in ('goal', 'score', 'done', 'error', 'task_id', 'program_hash',
                        'provenance', 'task_input_receipt'):
                self.assertEqual(witness[key], raw[key])
            self.assertEqual(witness['original_hash'], digest(raw))
            size = len(raw['history'])
            self.assertEqual([s['index'] for s in witness['steps']], [0, size - 2, size - 1])
            self.assertEqual(witness['omitted_step_count'], size - 3)
            pointers = {p for group in registry['evidence'][row['id']]
                        for p in review.citation_pointers(group)}
            self.assertIn('/task_input_receipt/delivery_verified', pointers)
            for i, step in enumerate(witness['steps']):
                for name, fragment in step['fields'].items():
                    self.assertEqual(fragment['text'], raw['history'][step['index']][name])
                    self.assertIn(f'/steps/{i}/fields/{name}/text', pointers)
            self.assertFalse(any(p.startswith('/steps/3/') for p in pointers))
            for group in registry['evidence'][row['id']]:
                scope = registry['scopes'][group['scope_id']]
                self.assertEqual(scope['program_hash'], raw['program_hash'])
                self.assertEqual(scope['task_id'], raw['task_id'])

    def test_eight_task_pair_does_not_displace_memory_or_native_outcomes(self):
        payload = contract_payload()
        traces = {side: next(r for r in payload['evidence'] if r['id'] == side + ':execution:0')
                  for side in ('parent', 'candidate')}
        payload['evidence'] = [r for r in payload['evidence'] if r['kind'] != 'execution_trace']
        expected = {}
        for side, prototype in traces.items():
            for index in range(8):
                row = copy.deepcopy(prototype)
                row['id'] = side + ':execution:' + str(index)
                row['data'].update(task_id='task-' + str(index), score=float(index % 2),
                    history=[dict(action='look', observation='recorded input ' * 500,
                                  next_observation='recorded feedback ' * 500) for _ in range(30)])
                payload['evidence'].append(row)
                expected[row['id']] = row['data']
        tokenizer = ApproximateTokenizer()
        # Isolate the mandatory packet. Optional expansions cannot make it fit.
        with patch.object(review, '_bounded_memory_data', return_value=(None, True)):
            full = review.bounded_review_payload(payload, tokenizer)
            budget = tokenizer.chat_count(review._messages(full)) - 1500
            with patch.object(review, 'MAX_INPUT_TOKENS', budget):
                shown = review.bounded_review_payload(payload, tokenizer)
        self.assertLess(shown['input_selection']['native_trace_witness_tokens'], 1536)
        self.assertLessEqual(tokenizer.chat_count(review._messages(shown)), budget)
        visible = {r['id']: r for r in shown['evidence']}
        for eid, data in expected.items():
            witness = visible[eid]['data']
            self.assertEqual(witness['original_hash'], digest(data))
            for key in ('task_id', 'score', 'done', 'error'):
                self.assertEqual(witness[key], data[key])
        self.assertEqual(len([r for r in shown['evidence'] if r['kind'] == 'execution_trace']), 16)
        for side in ('parent', 'candidate'):
            self.assertEqual(visible[side + ':program']['data'], next(
                r['data'] for r in payload['evidence'] if r['id'] == side + ':program'))
            self.assertTrue(visible[side + ':storage']['data']['fragments'])
            self.assertTrue(visible[side + ':read:0']['data']['fragments'])


if __name__ == '__main__':
    unittest.main()
