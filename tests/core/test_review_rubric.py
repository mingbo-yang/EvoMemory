"""CPU regression tests: fixed quality criteria and faithful bounded evidence.

Recorded assessment excerpts are regression fixtures, not new model results.
"""
import copy
import unittest

from evomem.common import digest, dumps
from phase1_codegen.roles.reviewer import MAX_INPUT_TOKENS, NONPROGRAM_TOKENS, REVIEW_PROMPT, REVIEW_RUBRIC, _bounded_memory_data, _messages, bounded_review_payload, validate_report
from tests.core.test_review import CharacterTokenizer, SOURCE, payload, report


class HeadTailTokenizer(CharacterTokenizer):
    def clip(self, text, count):
        if len(text) <= count:
            return text
        first = count // 2
        return text[:first] + text[-(count - first):]


def post_with_validation(valid=False):
    result = payload('post')
    result['evidence'].append(dict(id='candidate:validation', kind='validation', data=dict(
        status='readout_unchanged' if valid else 'invalid', mechanism_validated=valid,
        candidate_execution_performed=False,
        error=None if valid else 'Rejected: private names and builtin shadowing forbidden')))
    return result


def readout_data():
    entries = [dict(id='episode-a', experience='A_BEGIN ' + 'a' * 9000 + ' A_END'),
               dict(id='episode-b', experience='B_BEGIN ' + 'b' * 9000 + ' B_END')]
    return dict(task_id='current-target',
                query=dict(task_id='current-target', goal='find the target key',
                           environment='textworld', family='treasure_hunter'),
                native_score=0.0, read_tokens=2301, read_tokens_unit='whole_task_readout',
                actual_trace_available=True, replay_matches_actual=True,
                actual_injected_memory={'entries': entries}, replayed_readout={'entries': entries})


class ReviewRubricTests(unittest.TestCase):
    def test_recorded_invalid_post_subject_errors_are_explicitly_rejected(self):
        # Both real failed outputs switched to current because the candidate was
        # unexecuted/rejected. Those conditions never change the subject rule.
        observed = [
            ('alfworld', 'negative',
             'The candidate was rejected (candidate:validation), so the current mechanism remains the active implementation.'),
            ('textworld', 'neutral',
             'The candidate mechanism was rejected and not executed, so the assessment subject is the current (parent) implementation.'),
        ]
        for environment, verdict, rationale in observed:
            with self.subTest(environment=environment):
                source = post_with_validation()
                source['context']['environment'] = environment
                value = report('candidate:validation', subject='current', verdict=verdict)
                value['assessment']['rationale'] = rationale
                for evidence in (source, bounded_review_payload(source, CharacterTokenizer())):
                    with self.assertRaisesRegex(ValueError, 'subject must be candidate'):
                        validate_report(value, evidence)
                shown = bounded_review_payload(source, CharacterTokenizer())
                self.assertEqual(shown['expected_assessment_subject'], 'candidate')
                value['assessment']['subject'] = 'candidate'
                self.assertEqual(validate_report(value, shown), value)
        # The cases above verify subject selection directly; wording may change
        # when the wire format changes without altering this rule.

    def test_subject_is_derived_not_overridden_by_input_metadata(self):
        source = post_with_validation()
        source['expected_assessment_subject'] = 'current'
        with self.assertRaisesRegex(ValueError, 'fixed stage/program rule'):
            bounded_review_payload(source, CharacterTokenizer())
        source.pop('expected_assessment_subject')
        source['stage'] = 'pre'
        self.assertEqual(bounded_review_payload(source, CharacterTokenizer())['expected_assessment_subject'], 'current')

    def test_validation_error_is_required_and_invalid_positive_is_rejected(self):
        source = post_with_validation()
        shown = bounded_review_payload(source, CharacterTokenizer())
        self.assertEqual(shown['candidate_validation_summary']['error'], source['evidence'][-1]['data']['error'])
        self.assertFalse(shown['candidate_validation_summary']['mechanism_validated'])
        self.assertTrue(shown['candidate_validation_summary']['error_present'])
        positive = report('candidate:program', subject='candidate', verdict='positive')
        for evidence in (source, shown):
            with self.assertRaisesRegex(ValueError, 'cannot receive a positive'):
                validate_report(positive, evidence)
        source['evidence'][-1]['data']['error'] += ' very long error ' * 3000
        with self.assertRaisesRegex(ValueError, 'fixed input budget'):
            bounded_review_payload(source, CharacterTokenizer())

    def test_valid_unexecuted_candidate_may_have_grounded_non_reward_positive(self):
        source = post_with_validation(valid=True)
        source['evidence'].append(dict(id='candidate:storage', kind='memory_storage', data=dict(
            state={'records': []}, stored_tokens=2, stored_tokens_unit='whole_bank')))
        positive = report('candidate:storage', subject='candidate', verdict='positive')
        positive['assessment']['rationale'] = (
            'Valid replay verifies duplicate retention was removed while preserving the relevant facts; '
            'candidate Actor execution is absent, so this establishes no task-performance improvement.')
        self.assertEqual(validate_report(positive, bounded_review_payload(source, CharacterTokenizer())), positive)

    def test_readout_head_tail_fragments_keep_separate_ranges_and_entries(self):
        source = payload()
        source['context'].update(environment='textworld', memory_scope='environment')
        source['evidence'].append(dict(id='parent:read:0', kind='memory_readout', data=readout_data()))
        original = copy.deepcopy(source)
        shown = bounded_review_payload(source, HeadTailTokenizer())
        row = next(row for row in shown['evidence'] if row['id'] == 'parent:read:0')['data']
        self.assertEqual(row['query'], readout_data()['query'])
        self.assertEqual(row['read_tokens'], 2301)
        self.assertEqual(row['read_tokens_unit'], 'whole_task_readout')
        self.assertEqual(row['read_tokens_source'], 'replayed_readout')
        self.assertEqual(row['memory_scope'], 'environment')
        self.assertEqual(row['native_score'], 0.0)
        self.assertTrue(row['replay_matches_actual'])
        self.assertEqual([item['id'] for item in row['fragments']], ['episode-a', 'episode-b'])
        for item, marker, other in zip(row['fragments'], ('A', 'B'), ('B', 'A')):
            self.assertEqual(item['scopes'], ['actual_injected_memory', 'replayed_readout'])
            part = item['fragment']
            self.assertTrue(part['text'].startswith(marker + '_BEGIN'))
            self.assertNotIn(marker + '_END', part['text'])
            self.assertNotIn(other + '_BEGIN', part['text'])
            self.assertTrue(part['clipped'])
            self.assertEqual(part['fragment_policy'], 'separate_head_tail')
            self.assertTrue(part['suffix'].endswith(marker + '_END'))
            self.assertEqual(part['character_ranges'][0][0], 0)
            self.assertEqual(part['omitted_character_range'],
                             [part['character_ranges'][0][1], part['character_ranges'][1][0]])
        self.assertLessEqual(HeadTailTokenizer().count(dumps(row)), NONPROGRAM_TOKENS)
        self.assertLessEqual(HeadTailTokenizer().chat_count(_messages(shown)), MAX_INPUT_TOKENS)
        self.assertEqual(source, original)

    def test_different_actual_and_replay_are_labelled_independently(self):
        data = readout_data()
        data['actual_injected_memory']['entries'] = [dict(id='actual-only', experience='actual text')]
        data['replay_matches_actual'] = False
        shown, _ = _bounded_memory_data('memory_readout', data, CharacterTokenizer(), 2048, {})
        scopes = {tuple(row['scopes']) for row in shown['fragments']}
        self.assertEqual(scopes, {('actual_injected_memory',), ('replayed_readout',)})
        self.assertFalse(shown['replay_matches_actual'])
        self.assertEqual(shown['read_tokens_source'], 'replayed_readout')

    def test_storage_totals_keep_whole_bank_units_and_separate_state_paths(self):
        data = dict(state={'episodes': [dict(id='one', history='ONE ' + 'a' * 8000),
                                       dict(id='two', history='TWO ' + 'b' * 8000)]},
                    stored_tokens=10702, environment='alfworld', memory_scope='environment',
                    memory_hash='bank-hash', provenance='replayed_own_adaptation')
        shown, clipped = _bounded_memory_data('memory_storage', data, HeadTailTokenizer(), 2048, {})
        self.assertTrue(clipped)
        self.assertEqual(shown['stored_tokens'], 10702)
        self.assertEqual(shown['stored_tokens_unit'], 'whole_bank')
        self.assertEqual(shown['state_top_level_keys'], ['episodes'])
        self.assertEqual([row['path'] for row in shown['fragments']],
                         [['state', 'episodes', 0], ['state', 'episodes', 1]])
        for row, value in zip(shown['fragments'], data['state']['episodes']):
            self.assertEqual(row['fragment']['original_hash'], digest(value))
            self.assertTrue(dumps(value).startswith(row['fragment']['text']))
        self.assertLessEqual(HeadTailTokenizer().count(dumps(shown)), 2048)

    def test_legacy_memory_metadata_and_explicit_omission_remain_supported(self):
        data = dict(task_id='legacy-task', read_tokens=3, replayed_readout={}, actual_trace_available=False)
        shown, clipped = _bounded_memory_data('memory_readout', data, CharacterTokenizer(), 2048,
                                              {'environment': 'alfworld', 'family': 'heat'})
        self.assertFalse(clipped)
        self.assertEqual(shown['query']['environment'], 'alfworld')
        self.assertEqual(shown['query']['task_id'], 'legacy-task')
        self.assertEqual(shown['read_tokens_unit'], 'whole_task_readout')
        self.assertEqual(shown['fragments'], [])
        self.assertEqual(_bounded_memory_data('memory_readout', data, CharacterTokenizer(), 64, {}), (None, True))

    def test_quality_principles_do_not_force_architecture_positive_or_category_enum(self):
        for principle in ('Faithfulness', 'Relevance', 'Actionability', 'Applicability', 'Efficiency'):
            self.assertIn(principle, REVIEW_RUBRIC)
        self.assertIn('No dimension score or particular internal architecture is required', REVIEW_RUBRIC)
        self.assertIn('success-only storage is not a quality requirement', REVIEW_RUBRIC)
        self.assertIn('Task score need not improve', REVIEW_RUBRIC)
        self.assertIn('Novel code alone is not a verified gain', REVIEW_RUBRIC)
        self.assertIn('Only the separate Evolver chooses', REVIEW_PROMPT)
        for verdict in ('positive', 'neutral', 'negative', 'insufficient_evidence'):
            value = report(verdict=verdict)
            value['findings'][0]['category'] = 'legacy-freeform-storage-finding'
            self.assertEqual(validate_report(value, payload()), value)


if __name__ == '__main__':
    unittest.main()
