"""Rule B rejects score swaps, unsupported novelty and known current defects."""
import copy
import importlib.util
from pathlib import Path
import unittest

from phase1_codegen import acceptance as module


class RuleAnalysisTests(unittest.TestCase):
    def setUp(self):
        observations = []
        for side, text in [('parent', 'previous information'), ('candidate', 'new useful information')]:
            observations.append(dict(id=side, evidence_id=side + ':storage', pointer='/fragments/0/fragment/text',
                scope=dict(program_hash=side, phase='current_exploration', task_id=None), value=text))
        self.contract = dict(observations=observations, evaluations=[dict(
            id='e1', dimension='Actionability', polarity='advantage', uncertainty=['none_observed'],
            observation_ids=['parent', 'candidate'])], diagnostic_assessments=[])
        self.trial = dict(mechanism_validated=True, parent_hash='parent', candidate_hash='candidate',
            exploration=dict(complete=True, parent=1., candidate=1., delta=0., task_ids=['a', 'b']),
            post_review=dict(ok=True, stage='post', report=dict(assessment=dict(
                subject='candidate', verdict='positive'), evidence_contract=self.contract)))
        self.pairs = [dict(task_id=key, parent_score=1., candidate_score=1.) for key in ['a', 'b']]

    def decision(self):
        return module.equal_utility_decision(self.trial, self.pairs)

    def test_grounded_utility_can_qualify(self):
        self.assertTrue(self.decision()['eligible'])

    def test_bound_suffix_supports_same_b_utility_but_cannot_relax_strict_a(self):
        for kind in ('storage', 'hook_read:0'):
            with self.subTest(kind=kind):
                for observation in self.contract['observations']:
                    observation['pointer'] = '/fragments/0/fragment/suffix'
                    observation['evidence_id'] = observation['id'] + ':' + kind
                result = self.decision()
                self.assertTrue(result['eligible'])
                self.assertEqual(result['advantages'][0]['basis'], 'bound_content_change_and_quality_advantage')
                binding = dict(acceptance_policy='reviewer_and_reward', review_contract=2,
                               exploration_tasks=[{'id': key} for key in ['a', 'b']])
                self.assertFalse(module.eligible_for_adoption(binding, self.trial))

    def test_suffix_still_requires_matching_current_paired_content_and_no_defect(self):
        for observation in self.contract['observations']:
            observation['pointer'] = '/fragments/0/fragment/suffix'
        self.assertTrue(self.decision()['eligible'])
        self.contract['observations'][0]['pointer'] = '/fragments/0/fragment/text'
        self.assertFalse(self.decision()['eligible'])
        self.contract['observations'][0]['pointer'] = '/fragments/0/fragment/suffix'
        self.contract['observations'][1]['scope']['phase'] = 'historical'
        self.assertFalse(self.decision()['eligible'])
        self.contract['observations'][1]['scope']['phase'] = 'current_exploration'
        self.contract['diagnostic_assessments'] = [dict(group_id='d1', subject='candidate', judgment='supported_defect')]
        self.assertFalse(self.decision()['eligible'])

    def test_excerpt_metadata_never_counts_as_content(self):
        for pointer in ('/fragments/0/fragment/original_hash', '/fragments/0/fragment/character_ranges',
                        '/fragments/0/fragment/suffix_length', '/fragments/0/fragment/text/extra'):
            with self.subTest(pointer=pointer):
                for observation in self.contract['observations']:
                    observation['pointer'] = pointer
                self.assertFalse(self.decision()['eligible'])

    def test_suffix_supports_efficiency_only_with_bound_saving_and_preserved_quality(self):
        for observation in self.contract['observations']:
            observation['pointer'] = '/fragments/0/fragment/suffix'
        self.contract['evaluations'][0]['dimension'] = 'Efficiency'
        for dimension in ['Faithfulness', 'Relevance', 'Actionability']:
            self.contract['evaluations'].append(dict(id=dimension, dimension=dimension, polarity='unchanged',
                uncertainty=['none_observed'], observation_ids=['parent', 'candidate']))
        for side, value in [('parent', 100), ('candidate', 50)]:
            self.contract['observations'].append(dict(id=side + '-tokens', evidence_id='current:facts',
                pointer='/programs/' + side + '/stored_tokens',
                scope=dict(program_hash=side, phase='current_exploration'), value=value))
        self.contract['evaluations'][0]['observation_ids'] = ['parent-tokens', 'candidate-tokens']
        self.assertTrue(self.decision()['eligible'])
        self.contract['evaluations'][1]['uncertainty'] = ['clipped_evidence']
        self.assertFalse(self.decision()['eligible'])

    def test_same_mean_with_per_task_tradeoff_does_not_qualify(self):
        self.pairs[0].update(parent_score=1., candidate_score=0.)
        self.pairs[1].update(parent_score=0., candidate_score=1.)
        self.trial['exploration'].update(parent=.5, candidate=.5)
        self.assertIn('per_task_scores_not_identical', self.decision()['reasons'])

    def test_identical_zero_and_unknown_scores_do_not_qualify(self):
        for row in self.pairs:
            row.update(parent_score=0., candidate_score=0.)
        self.trial['exploration'].update(parent=0., candidate=0.)
        self.assertIn('zero_or_negative_mean', self.decision()['reasons'])
        self.pairs[0]['candidate_score'] = None
        self.assertIn('missing_finite_per_task_scores', self.decision()['reasons'])

    def test_unexecuted_candidate_has_no_pair_and_is_not_assigned_zero(self):
        self.trial.update(mechanism_validated=False, candidate_hash=None, exploration=None)
        self.pairs = []
        result = self.decision()
        self.assertFalse(result['eligible'])
        self.assertIn('missing_complete_current_pairs', result['reasons'])

    def test_identical_content_or_source_novelty_is_not_utility(self):
        self.contract['observations'][1]['value'] = self.contract['observations'][0]['value']
        self.assertFalse(self.decision()['eligible'])
        self.contract['observations'][0]['evidence_id'] = 'parent:source'
        self.contract['observations'][1]['evidence_id'] = 'candidate:source'
        self.assertFalse(self.decision()['eligible'])

    def test_current_defect_blocks_extra_branch_parent_defect_does_not(self):
        self.contract['diagnostic_assessments'] = [dict(group_id='d1', subject='candidate', judgment='supported_defect')]
        self.assertIn('confirmed_current_candidate_functional_defect', self.decision()['reasons'])
        self.contract['diagnostic_assessments'][0]['subject'] = 'parent'
        self.assertTrue(self.decision()['eligible'])

    def test_efficiency_alone_cannot_make_up_missing_quality(self):
        self.contract['evaluations'][0]['dimension'] = 'Efficiency'
        self.assertFalse(self.decision()['eligible'])
        for dimension in ['Faithfulness', 'Relevance', 'Actionability']:
            self.contract['evaluations'].append(dict(id=dimension, dimension=dimension, polarity='unchanged',
                uncertainty=['none_observed'], observation_ids=['parent', 'candidate']))
        for side, value in [('parent', 100), ('candidate', 50)]:
            self.contract['observations'].append(dict(id=side + '-tokens', evidence_id='current:facts',
                pointer='/programs/' + side + '/stored_tokens',
                scope=dict(program_hash=side, phase='current_exploration'), value=value))
        self.contract['evaluations'][0]['observation_ids'] = ['parent-tokens', 'candidate-tokens']
        self.assertTrue(self.decision()['eligible'])
        self.contract['evaluations'][1]['uncertainty'] = ['clipped_evidence']
        self.assertFalse(self.decision()['eligible'])

    def test_historical_content_does_not_support_current_advantage(self):
        self.contract['observations'][1]['scope']['phase'] = 'historical'
        self.assertFalse(self.decision()['eligible'])

    def test_read_saving_requires_the_same_actual_task_and_unit(self):
        self.contract['evaluations'][0]['dimension'] = 'Efficiency'
        for dimension in ['Faithfulness', 'Relevance', 'Actionability']:
            self.contract['evaluations'].append(dict(id=dimension, dimension=dimension, polarity='unchanged',
                uncertainty=['none_observed'], observation_ids=['parent', 'candidate']))
        facts = {'programs': {}}
        for side, value in [('parent', 100), ('candidate', 50)]:
            facts['programs'][side] = dict(reads=[dict(task_id='a', unit='one_actor_memory_slot',
                read_tokens=value, actual_trace_available=True)])
            self.contract['observations'].append(dict(id=side + '-read', evidence_id='current:facts',
                pointer='/programs/' + side + '/reads/0/read_tokens',
                scope=dict(program_hash=side, phase='current_exploration', task_id='a'), value=value))
        self.contract['evaluations'][0]['observation_ids'] = ['parent-read', 'candidate-read']
        self.assertTrue(module.equal_utility_decision(self.trial, self.pairs, facts)['eligible'])
        facts['programs']['candidate']['reads'][0]['unit'] = 'different_measurement'
        self.assertFalse(module.equal_utility_decision(self.trial, self.pairs, facts)['eligible'])
        facts['programs']['candidate']['reads'][0]['unit'] = 'one_actor_memory_slot'
        facts['programs']['candidate']['reads'][0]['missing_token_measurements'] = 1
        self.assertFalse(module.equal_utility_decision(self.trial, self.pairs, facts)['eligible'])


if __name__ == '__main__':
    unittest.main()
