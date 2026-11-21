"""Contracts for incomplete evidence and host-vs-candidate failure attribution."""
import unittest
from unittest.mock import patch

from evomem.common import read
from phase1_codegen.engine import MeasurementFailure, own_trace
from phase1_codegen.outcomes import candidate_hook_failure, comparison_for_feedback, reported_score
from phase1_codegen.roles.reviewer import _computed_current_facts, native_trace_witness
from tests.core import test_acceptance_policy as fixture
from tests.core.test_engine import TinyTokenizer


class MeasurementOutcomeTests(unittest.TestCase):
    def test_failed_values_and_verified_zero(self):
        valid = dict(score=0., synthetic=False, native_reward_verified=True)
        self.assertEqual(reported_score(valid), 0.)
        for bad in (dict(valid, native_reward_verified=False), dict(valid, error='transport')):
            self.assertIsNone(reported_score(bad))
            self.assertIsNone(own_trace(bad)['score'])
            self.assertIsNone(native_trace_witness(bad, TinyTokenizer())['score'])
        comparison = dict(complete=False, parent=1., candidate=0., delta=0., task_ids=['t'])
        shown = comparison_for_feedback(comparison)
        self.assertEqual([shown[k] for k in ('parent', 'candidate', 'delta')], [None]*3)
        self.assertEqual(comparison['candidate'], 0.)

    def test_partial_pair_preserves_good_pairs(self):
        evidence = []
        for side in ('parent', 'candidate'):
            evidence.append(dict(id=side+':program', kind='program', data=dict(source='example')))
            for i in (0, 1):
                evidence.append(dict(id=f'{side}:execution:{i}', kind='execution_trace',
                    data=dict(task_id=str(i), score=i, done=True, synthetic=False,
                              native_reward_verified=(side, i) != ('candidate', 1))))
        facts = _computed_current_facts(dict(stage='post', context={}, evidence=evidence))
        self.assertEqual(facts['programs']['parent']['mean_native_score'], .5)
        self.assertIsNone(facts['programs']['candidate']['mean_native_score'])
        pair = facts['comparison']
        self.assertFalse(pair['complete'])
        self.assertTrue(all(pair[key] is None for key in ('parent_mean', 'candidate_mean', 'delta')))
        self.assertEqual(pair['tasks'][0]['delta'], 0.)
        self.assertIsNone(pair['tasks'][1]['candidate_score'])
        self.assertIsNone(pair['tasks'][1]['delta'])

    def test_generated_hook_failure_requires_bound_rejection(self):
        row = dict(status='rejected', error_type='Rejected', program_hash='draft',
                   error='isolated worker rejected (2): {"error":"TypeError","message":"wrong argument"}')
        episode = dict(error='generated hook failed', program_hash='draft', memory_hook=dict(events=[row]))
        self.assertTrue(candidate_hook_failure(episode, 'draft'))
        self.assertFalse(candidate_hook_failure(episode, 'other'))
        self.assertFalse(candidate_hook_failure(dict(episode, infrastructure_failure={'kind':'host'}), 'draft'))
        for old_message in ('worker exceeded cumulative one-second CPU budget',
                            'worker exceeded cumulative five-second wall budget after trusted startup'):
            row['error'] = old_message
            self.assertTrue(candidate_hook_failure(episode, 'draft'))
        for ambiguous in ('isolated worker rejected (-9): ', 'trusted worker bootstrap failed'):
            row['error'] = ambiguous
            self.assertFalse(candidate_hook_failure(episode, 'draft'))


class CandidateHostFailureTest(unittest.TestCase):
    setUp = fixture.AcceptancePolicyTests.setUp
    components = fixture.AcceptancePolicyTests.components
    engine = fixture.AcceptancePolicyTests.engine

    def test_host_exception_on_candidate_stops_before_post_review_or_commit(self):
        proposer, evaluator, reviewer = self.components()
        original_call = type(evaluator).__call__
        calls = []
        def fail_candidate(evaluator, program, logs, tasks, phase):
            calls.append(phase)
            if phase.endswith('/candidate'):
                raise OSError('simulated environment unavailable')
            return original_call(evaluator, program, logs, tasks, phase)
        with patch.object(type(evaluator), '__call__', new=fail_candidate):
            for _ in range(2):
                with self.assertRaisesRegex(MeasurementFailure, 'environment unavailable'):
                    self.engine(proposer, evaluator, reviewer).run(1)
        self.assertEqual(sum(phase.endswith('/candidate') for phase in calls), 1)
        self.assertEqual(len(proposer.calls), 1)
        self.assertEqual(len(reviewer.calls), 1)
        self.assertFalse((self.root / 'trials/000/trial.json').exists())
        self.assertEqual(read(self.root/'state.json')['next_round'], 0)
        failure, = (self.root/'measurement_failures').glob('*.json')
        self.assertEqual(read(failure)['error_type'], 'OSError')
        self.assertEqual(len(list((self.root/'measurements').glob('*.json'))), 1)
