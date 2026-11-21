"""Historical native feedback must survive long trajectories for either decision."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evomem.common import digest, dumps, read
from phase1_codegen import proposer, review
from phase1_codegen.tests.test_review import ApproximateTokenizer, SOURCE


def episode_row(side, index, *, score, error=None, action='look'):
    program = digest(side)
    data = dict(
        task_id=f'textworld/fixture/seed_{index}', environment='textworld',
        family='fixture', phase=f'round-000/exploration/{side}', seed=42 + index,
        goal='Read the sign.', score=score, done=error is None, error=error,
        program_hash=program, native_reward_verified=error is None,
        provenance=dict(program_hash=program, memory_snapshot_hash='earlier-bank'),
        task_input_receipt=dict(direct_goal_verified=True),
        history=[dict(action=action, observation=f'room {i}', next_observation=f'feedback {i}')
                 for i in range(8)])
    return dict(id=f'{side}:execution:{index}', kind='execution_trace', data=data)


def displayed_row(raw, tokenizer):
    return dict(raw, data=review.native_trace_witness(raw['data'], tokenizer, limit=30000))


class HistoricalNativeFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ApproximateTokenizer()

    def test_verified_witnesses_follow_original_task_order_and_are_reused_intact(self):
        raw = [episode_row(side, 0, score=score) for side, score in [('parent', 0), ('candidate', 1)]]
        shown = [displayed_row(row, self.tokenizer) for row in raw]
        unrelated = dict(id='adaptation:0', kind='adaptation_trace', data={'score': 1})
        result = review.historical_native_execution([unrelated, *raw], shown[::-1], self.tokenizer)
        self.assertEqual(result, [dict(id=row['id'], data=row['data']) for row in shown])

    def test_wrong_hash_task_native_result_and_provenance_are_rejected(self):
        raw = episode_row('candidate', 0, score=0)
        shown = displayed_row(raw, self.tokenizer)
        for field, invalid in [('original_hash', 'other-digest'), ('task_id', 'other-task'),
                               ('score', 1), ('program_hash', 'another-program'),
                               ('provenance', {'memory_snapshot_hash': 'future-bank'})]:
            with self.subTest(field=field):
                changed = copy.deepcopy(shown)
                changed['data'][field] = invalid
                with self.assertRaises((ValueError, RuntimeError)):
                    review.historical_native_execution([raw], [changed], self.tokenizer)

    def test_short_legacy_trace_keeps_existing_renderer_and_long_fallback_keeps_null(self):
        short = episode_row('parent', 0, score=0)
        long = episode_row('candidate', 0, score=0, error='Worker interrupted', action='word ' * 2200)
        with self.assertRaisesRegex(ValueError, 'witness budget'):
            review.native_trace_witness(long['data'], self.tokenizer)
        expected_short = review.native_trace_witness(short['data'], self.tokenizer)
        for shown in ([], [short, long]):
            with self.subTest(legacy_raw_shown=bool(shown)):
                result = review.historical_native_execution([short, long], shown, self.tokenizer)
                self.assertEqual([r['id'] for r in result], [short['id'], long['id']])
                self.assertEqual(result[0]['data'], expected_short)
                self.assertEqual(result[0]['data']['score'], 0)
                reference = result[1]['data']
                self.assertIsNone(reference['score'])
                self.assertEqual(reference['error'], long['data']['error'])
                for key in ('task_id', 'phase', 'seed', 'program_hash'):
                    self.assertEqual(reference[key], long['data'][key])
                self.assertEqual(reference['original_hash'], digest(long['data']))
                self.assertEqual(reference['step_count'], len(long['data']['history']))
                self.assertEqual(reference['omitted_step_count'], reference['step_count'])

    def test_total_proposal_budget_preserves_sources_errors_and_every_paired_outcome(self):
        raw = [episode_row(side, i, score=i % 2, error='Unverified execution' if i == 2 else None,
                           action='recorded action ' * 450)
               for i in range(3) for side in ('parent', 'candidate')]
        shown = [displayed_row(r, self.tokenizer) for r in raw]
        traces = review.historical_native_execution(raw, shown, self.tokenizer)
        draft = SOURCE + '# previous complete proposal\n'
        current = SOURCE + '# current complete deployment\n'
        payload = dict(context={}, current_source=current, previous_attempt=dict(
            source=draft, error='exact admission error', native_execution=traces))
        payload_hash = digest(payload)
        author = proposer.CodeEvolver.__new__(proposer.CodeEvolver)
        author.tokenizer = self.tokenizer
        author.llm = Mock(model_name='synthetic-test-model')
        author._invoke = Mock(return_value={})
        messages = [dict(role='system', content=proposer.EVOLVER_PROMPT),
                    dict(role='user', content=dumps(payload))]
        unabridged = self.tokenizer.chat_count(messages)
        # All six complete witnesses overflow together. The bare task references fit.
        context_limit = unabridged // 2 + proposer.PARAMETERS['max_tokens'] + 512
        with patch.object(proposer, 'META_CONTEXT_TOKENS', context_limit):
            author(payload, 10)
        request, tokens, _ = author._invoke.call_args.args
        sent = json.loads(request['messages'][1]['content'])
        prior = sent['previous_attempt']
        self.assertEqual(author._invoke.call_count, 1)
        self.assertEqual(tokens, self.tokenizer.chat_count(request['messages']))
        self.assertLessEqual(tokens + proposer.PARAMETERS['max_tokens'] + 512, context_limit)
        self.assertEqual(digest(payload), payload_hash)
        self.assertEqual((sent['current_source'], prior['source'], prior['error']),
                         (current, draft, 'exact admission error'))
        self.assertEqual([row['id'] for row in prior['native_execution']], [r['id'] for r in raw])
        for out, original in zip(prior['native_execution'], raw):
            receipt, data = out['data'], original['data']
            self.assertEqual(receipt['original_hash'], digest(data))
            for key in ('task_id', 'program_hash', 'seed', 'phase', 'done', 'error'):
                self.assertEqual(receipt[key], data[key])
            self.assertEqual(receipt['score'], None if data['error'] else data['score'])
            self.assertEqual(receipt['step_count'], len(data['history']))
            self.assertGreater(receipt['omitted_step_count'], 0)

    def test_real_accepted_and_rejected_archives_preserve_post_review_witnesses(self):
        validation = Path(__file__).resolve().parents[2] / 'validation'
        cases = [
            ('reverse_cooking_three_seeds_20261118/study/source/treasure/trials/000', True),
            ('gpu1_27b_20261118/paired_recovery/study/target/textworld/cross_type_em/trials/000', False),
        ]
        for directory, accepted in cases:
            with self.subTest(archive=directory):
                root = validation / directory
                if not root.is_dir():
                    self.skipTest('Frozen development archive is unavailable in this checkout')
                trial = read(root / 'trial.json')
                raw = read(root / 'post_review_input.json')['evidence']
                displayed = read(root / 'post_review_displayed.json')['evidence']
                shown_by_id = {r['id']: r['data'] for r in displayed if r['kind'] == 'execution_trace'}
                expected = [r for r in raw if r['kind'] == 'execution_trace']
                self.assertEqual(trial['accepted'], accepted)
                self.assertEqual(len(expected), 4)
                self.assertEqual(digest(read(root / 'post_review_displayed.json')), trial['post_review']['displayed_hash'])
                result = review.historical_native_execution(raw, displayed, self.tokenizer)
                self.assertEqual([r['id'] for r in result], [r['id'] for r in expected])
                for received, original in zip(result, expected):
                    self.assertEqual(received['data'], shown_by_id[original['id']])
                    self.assertEqual(received['data']['original_hash'], digest(original['data']))
                    for key in ('task_id', 'score', 'program_hash', 'provenance'):
                        self.assertEqual(received['data'][key], original['data'][key])


if __name__ == '__main__':
    unittest.main()
