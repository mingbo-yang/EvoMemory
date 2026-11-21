"""Synthetic reviewer contracts with fake HTTP; no model/environment calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phase1_codegen.review import (
    MAX_INPUT_TOKENS, PARAMETERS, REVIEW_PROMPT, MechanismReviewer,
    bounded_review_payload, validate_report,
)


class CharacterTokenizer:
    def count(self, text):
        return len(text)

    def clip(self, text, count):
        return text[:max(0, count)]

    def chat_count(self, messages):
        return sum(len(message['content']) + 16 for message in messages) + 16


class ApproximateTokenizer:
    """CPU review fixtures: four UTF-8 bytes per token, not model tokenization."""
    def count(self, text):
        return (len(text.encode('utf-8')) + 3) // 4

    def clip(self, text, count):
        return text.encode('utf-8')[:max(0, count) * 4].decode('utf-8', errors='ignore')

    def chat_count(self, messages):
        return sum(self.count(message['content']) + 16 for message in messages) + 16


SOURCE = 'def on_event(state, event, context):\n    return {"state": state, "session": event.get("session", {}), "memory": []}\n'


def payload(stage='pre'):
    evidence = [dict(id='program-parent', kind='program', data={'source': SOURCE, 'role': 'parent'}),
                dict(id='storage-parent', kind='storage', data={'stored_tokens': 2}),
                dict(id='trajectory-parent', kind='exploration', data={'reward': 0, 'actions': ['look']})]
    if stage == 'post':
        evidence.extend([dict(id='candidate:program', kind='program', data=SOURCE + '# candidate\n'),
                         dict(id='storage-candidate', kind='storage', data={'stored_tokens': 4})])
    return dict(stage=stage, context={'environment': 'synthetic', 'family': 'synthetic-clean'},
                evidence=evidence, evolution_memory=[], previous_post_review=None, round=0)


def report(reference='storage-parent', subject='current', verdict='neutral'):
    return dict(summary='Synthetic diagnostic only.', findings=[dict(
        category='storage', claim='The displayed storage measurement is small.',
        evidence_ids=[reference], interpretation='observation')],
        limitations=['Synthetic fixture; no native task performance evidence.'],
        assessment=dict(subject=subject, verdict=verdict,
                        rationale='The displayed mechanism evidence does not establish a relative advantage.',
                        evidence_ids=[reference]))


class FakeResponse:
    def __init__(self, value=None, finish='stop', status_error=None, content=None):
        self.value = {'choices': [{'finish_reason': finish, 'message': {
            'content': json.dumps(report() if value is None else value) if content is None else content}}],
            'usage': {'prompt_tokens': 300, 'completion_tokens': 100}}
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return copy.deepcopy(self.value)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = False

    def post(self, url, json):
        self.requests.append((url, copy.deepcopy(json)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


class FakeLLM:
    def __init__(self, client, model_fingerprint, base_url='http://127.0.0.1:8140/v1', model_name='qwen3.8-27b-local'):
        self.client = client
        self.base = base_url
        self.fingerprint = model_fingerprint
        self.model_name = model_name


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ApproximateTokenizer()

    def make_reviewer(self, directory, responses):
        client = FakeClient(responses)
        factory = lambda tokenizer, ledger, model_fingerprint, base_url, model_name: FakeLLM(client, model_fingerprint, base_url, model_name)
        with patch('phase1_codegen.review.LocalLLM', factory):
            reviewer = MechanismReviewer(self.tokenizer, Path(directory) / 'reviewer.jsonl', 'synthetic-model')
        return reviewer, client

    def test_one_fixed_prompt_and_parameters_for_pre_and_post(self):
        with tempfile.TemporaryDirectory() as directory:
            from phase1_codegen.tests.test_review_contract import contract_payload, bundled_raw_report
            inputs = [contract_payload(stage) for stage in ('pre', 'post')]
            shown = [bounded_review_payload(value, self.tokenizer) for value in inputs]
            raw = [bundled_raw_report(value) for value in shown]
            reviewer, client = self.make_reviewer(directory, [FakeResponse(value) for value in raw])
            first = reviewer(inputs[0], 11)
            second = reviewer(inputs[1], 12)
            self.assertEqual(first, validate_report(raw[0], shown[0]))
            self.assertEqual(second, validate_report(raw[1], shown[1]))
            self.assertEqual(len(client.requests), 2)
            for stage, (url, request) in zip(('pre', 'post'), client.requests):
                self.assertEqual(url, 'http://127.0.0.1:8140/v1/chat/completions')
                self.assertEqual(request['messages'][0]['content'], REVIEW_PROMPT)
                self.assertEqual(json.loads(request['messages'][1]['content'])['stage'], stage)
                for name, value in PARAMETERS.items():
                    self.assertEqual(request[name], value)
            snapshot = reviewer.last_payload
            snapshot['evidence'].clear()
            self.assertTrue(reviewer.last_payload['evidence'])
            receipts = [json.loads(line) for line in reviewer.ledger.read_text().splitlines()]
            self.assertEqual([r['stage'] for r in receipts], ['pre', 'post'])
            for index, receipt in enumerate(receipts):
                self.assertEqual(receipt['role'], 'mechanism_reviewer')
                self.assertTrue(receipt['ok'])
                self.assertTrue(receipt['api_attempted'])
                self.assertEqual(receipt['usage'], {'prompt_tokens': 300, 'completion_tokens': 100})
                self.assertEqual(receipt['validated_report'], (first, second)[index])
            reviewer.close()
            self.assertTrue(client.closed)

    def test_report_schema_citations_and_uncertainty_are_enforced(self):
        source = payload()
        self.assertEqual(validate_report(report(), source), report())
        empty = dict(summary='No defensible finding.', findings=[], limitations=['Insufficient evidence.'],
                     assessment=dict(subject='current', verdict='insufficient_evidence',
                                     rationale='Evidence is insufficient.', evidence_ids=[]))
        self.assertEqual(validate_report(empty, source), empty)
        cases = []
        value = report(); value['approve'] = True; cases.append(value)
        value = report(); value['score'] = 1; cases.append(value)
        value = report(); value['findings'][0]['evidence_ids'] = ['not-shown']; cases.append(value)
        value = report(); value['findings'][0]['evidence_ids'] = []; cases.append(value)
        value = report(); value['findings'][0]['interpretation'] = 'proven_cause'; cases.append(value)
        value = report(); value['findings'] *= 9; cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_report(value, source)

    def test_positive_mechanism_assessment_does_not_require_reward_improvement(self):
        source = payload('post')
        next(row for row in source['evidence'] if row['id'] == 'candidate:program')['data'] = (
            'def on_event(state, event, context):\n'
            '    state = {"recent": event.get("experience", {}).get("history", [])[-4:]}\n'
            '    return {"state": state, "session": {}, "memory": []}\n')
        source['evidence'].append(dict(id='candidate:execution:0', kind='exploration',
                                       data={'reward': 0, 'actions': ['look']}))
        value = report(reference='candidate:program', subject='candidate', verdict='positive')
        value['assessment']['rationale'] = (
            'Relative to the current implementation, the candidate explicitly bounds retained history; '
            'this is a mechanism-level opinion. Both displayed task rewards remain zero.')
        value['assessment']['evidence_ids'] = ['program-parent', 'candidate:program', 'candidate:execution:0']
        value['findings'][0]['claim'] = 'The candidate event source limits stored history to four entries.'
        self.assertEqual(validate_report(value, source), value)
        with tempfile.TemporaryDirectory() as directory:
            from phase1_codegen.tests.test_review_contract import contract_payload, bundled_raw_report
            source = contract_payload()
            value = bundled_raw_report(bounded_review_payload(source, self.tokenizer), verdict='positive')
            reviewer, client = self.make_reviewer(directory, [FakeResponse(value)])
            self.assertEqual(reviewer(source, 22)['assessment']['verdict'], 'positive')
            receipt = json.loads(reviewer.ledger.read_text())
            self.assertTrue(receipt['ok'])
            self.assertNotIn('approve', receipt['validated_report'])
            self.assertEqual(len(client.requests), 1)

    def test_positive_candidate_requires_candidate_evidence_not_only_parent(self):
        source = payload('post')
        positive = report(reference='program-parent', subject='candidate', verdict='positive')
        with self.assertRaisesRegex(ValueError, 'must cite displayed candidate: evidence'):
            validate_report(positive, source)
        positive['assessment']['evidence_ids'].append('candidate:program')
        self.assertEqual(validate_report(positive, source), positive)
        positive['assessment']['evidence_ids'] = ['candidate:not-shown']
        with self.assertRaisesRegex(ValueError, 'absent from displayed evidence: candidate:not-shown'):
            validate_report(positive, source)

    def test_evaluation_only_findings_reject_the_removed_suggestion_field(self):
        value = report()
        self.assertEqual(set(value['findings'][0]),
                         {'category', 'claim', 'evidence_ids', 'interpretation'})
        value['findings'][0]['suggestion'] = 'A proposed change.'
        with self.assertRaisesRegex(ValueError, 'must contain exactly category, claim, evidence_ids and interpretation'):
            validate_report(value, payload())
        # Prompt contract only: these assertions do not establish model compliance.
        prompt = ' '.join(REVIEW_PROMPT.split())
        self.assertIn('observations, causal hypotheses and uncertainties', prompt)
        self.assertIn('Do not recommend mechanism', prompt)
        self.assertIn('Judge only implementations actually displayed.', prompt)
        self.assertIn('Do not use hypothetical or unimplemented alternatives as evaluation standards', prompt)
        self.assertIn('label the attribution as a hypothesis and state the missing comparison', prompt)
        self.assertIn('do not infer that unseen content or fields were preserved', prompt.lower())

    def test_all_unique_displayed_evidence_ids_can_be_cited_without_eight_id_cap(self):
        source = payload()
        source['evidence'].extend(dict(id='evidence-%d' % index, kind='exploration',
                                      data={'reward': 0}) for index in range(32))
        visible = [row['id'] for row in source['evidence']]
        self.assertEqual(len(visible), 35)
        value = report()
        value['assessment']['evidence_ids'] = visible[:26]
        value['findings'][0]['evidence_ids'] = visible[:18]
        self.assertEqual(validate_report(value, source), value)
        value['assessment']['evidence_ids'] = visible
        value['findings'][0]['evidence_ids'] = list(reversed(visible))
        self.assertEqual(validate_report(value, source), value)

    def test_invalid_citation_errors_distinguish_type_empty_duplicate_and_unknown(self):
        cases = [
            ('storage-parent', 'must be a list'),
            ([1], 'must contain strings only'),
            ([], 'must cite at least one displayed evidence ID'),
            (['storage-parent', 'storage-parent'], 'must be unique; duplicate IDs were provided'),
            (['unseen'], 'contain IDs absent from displayed evidence: unseen'),
        ]
        for role in ('assessment', 'finding'):
            for references, message in cases:
                with self.subTest(role=role, references=references):
                    value = report()
                    target = value['assessment'] if role == 'assessment' else value['findings'][0]
                    target['evidence_ids'] = references
                    label = 'Reviewer assessment' if role == 'assessment' else 'Reviewer finding 0'
                    with self.assertRaisesRegex(ValueError, label + ' evidence_ids ' + message):
                        validate_report(value, payload())

    def test_assessment_subject_uses_stage_and_actual_candidate_evidence_not_hash_metadata(self):
        post = payload('post')
        with self.assertRaisesRegex(ValueError, 'subject must be candidate'):
            validate_report(report(), post)
        pre_with_candidate = copy.deepcopy(post)
        pre_with_candidate['stage'] = 'pre'
        self.assertEqual(validate_report(report(), pre_with_candidate)['assessment']['subject'], 'current')
        post_without_candidate = payload('pre')
        post_without_candidate['stage'] = 'post'
        post_without_candidate['candidate_program_hash'] = 'metadata-is-not-program-evidence'
        self.assertEqual(validate_report(report(), post_without_candidate)['assessment']['subject'], 'current')
        renamed = copy.deepcopy(post)
        next(row for row in renamed['evidence'] if row['id'] == 'candidate:program')['id'] = 'program-candidate'
        self.assertEqual(validate_report(report(), renamed)['assessment']['subject'], 'current')

    def test_assessment_requires_grounded_verdict_and_bounded_rationale(self):
        source = payload()
        cases = []
        value = report(); value.pop('assessment'); cases.append(value)
        value = report(); value['assessment']['verdict'] = 'approved'; cases.append(value)
        value = report(); value['assessment']['verdict'] = ['positive']; cases.append(value)
        value = report(); value['assessment']['rationale'] = ' '; cases.append(value)
        value = report(); value['assessment']['rationale'] = 'x' * 2001; cases.append(value)
        value = report(); value['assessment']['evidence_ids'] = ['unseen']; cases.append(value)
        value = report(); value['assessment']['evidence_ids'] = ['storage-parent'] * 2; cases.append(value)
        for verdict in ('positive', 'neutral', 'negative'):
            value = report(verdict=verdict)
            value['assessment']['evidence_ids'] = []
            cases.append(value)
            self.assertEqual(validate_report(report(verdict=verdict), source)['assessment']['verdict'], verdict)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_report(value, source)
        insufficient = report(verdict='insufficient_evidence')
        insufficient['assessment']['evidence_ids'] = []
        self.assertEqual(validate_report(insufficient, source), insufficient)

    def test_budget_preserves_complete_programs_and_marks_clipped_evidence(self):
        source = payload('post')
        source['evidence'][1]['data'] = {'history': 'observation and action ' * 4000}
        source['evolution_memory'] = [{'id': 'design-old', 'hypothesis': 'A design hypothesis.'}]
        source['previous_post_review'] = report()
        original = copy.deepcopy(source)
        displayed = bounded_review_payload(source, self.tokenizer)
        actual_programs = [row for row in displayed['evidence'] if row['kind'] == 'program']
        expected_programs = [row for row in source['evidence'] if row['kind'] == 'program']
        self.assertEqual(actual_programs, expected_programs)
        clipped = next(row for row in displayed['evidence'] if row['id'] == 'storage-parent')
        self.assertTrue(clipped['data']['clipped'])
        self.assertIn('storage-parent', displayed['input_selection']['clipped_evidence_ids'])
        self.assertIn('original_hash', clipped['data'])
        messages = [{'role': 'system', 'content': REVIEW_PROMPT},
                    {'role': 'user', 'content': json.dumps(displayed, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}]
        self.assertLessEqual(self.tokenizer.chat_count(messages), MAX_INPUT_TOKENS)
        self.assertEqual(displayed['evolution_memory'], source['evolution_memory'])
        self.assertEqual(source, original)
        self.assertEqual(displayed, bounded_review_payload(source, self.tokenizer))

    def test_model_cannot_cite_omitted_evidence_from_original_payload(self):
        source = payload()
        # This case tests omitted references, not an oversized-source rejection.
        # The synthetic tokenizer approximates four UTF-8 bytes per token.
        # Force omissions at the configured capacity rather than hard-coding
        # the old small-run input limit.
        source['evidence'] = [dict(id='program-parent', kind='program', data='# ' + 'x' * 7000 + '\n' + SOURCE)]
        source['evidence'].extend(dict(id='hidden-%d' % i, kind='exploration', data={'history': 'long ' * 200})
                                  for i in range(MAX_INPUT_TOKENS // 200 + 100))
        displayed = bounded_review_payload(source, self.tokenizer)
        self.assertGreater(displayed['input_selection']['omitted_evidence_count'], 0)
        shown = {row['id'] for row in displayed['evidence']}
        absent = next(row['id'] for row in source['evidence'] if row['id'] not in shown)
        # This reference exists in the original input, but the model never saw it.
        self.assertEqual(validate_report(report(absent), source), report(absent))
        with tempfile.TemporaryDirectory() as directory:
            from phase1_codegen.tests.test_review_contract import bundled_raw_report
            source['review_contract'] = 2
            shown_v2 = bounded_review_payload(source, self.tokenizer)
            invalid = bundled_raw_report(shown_v2)
            invalid['focus_evidence_ids'] = ['not-a-displayed-citation']
            reviewer, client = self.make_reviewer(directory, [FakeResponse(invalid)])
            with self.assertRaisesRegex(ValueError, 'displayed citation'):
                reviewer(source, 13)
            receipt = json.loads(reviewer.ledger.read_text())
            self.assertFalse(receipt['ok'])
            self.assertIsNone(receipt['validated_report'])
            self.assertEqual(len(client.requests), 1)

    def test_design_memory_and_previous_review_are_reserved_before_long_traces(self):
        source = payload('post')
        source['evidence'][0]['data']['source'] = '# ' + 'x' * 10000 + '\n' + SOURCE
        source['evidence'].extend(dict(id='long-%d' % i, kind='exploration', data={'history': 'long trace ' * 2000})
                                  for i in range(16))
        source['evolution_memory'] = [dict(id='reserved-design', hypothesis='x' * 2400)]
        source['previous_post_review'] = report()
        source['previous_post_review']['summary'] = 'p' * 500
        displayed = bounded_review_payload(source, self.tokenizer)
        self.assertEqual(displayed['evolution_memory'], source['evolution_memory'])
        self.assertEqual(displayed['previous_post_review'], source['previous_post_review'])
        self.assertTrue(displayed['input_selection']['previous_post_review_shown'])
        self.assertTrue(displayed['input_selection']['clipped_evidence_ids'] or
                        displayed['input_selection']['omitted_evidence_count'])

    def test_design_condition_is_never_silently_dropped_to_fit_budget(self):
        source = payload()
        source['evolution_memory'] = [dict(id='too-large-design', hypothesis='x' * 14000)]
        with self.assertRaisesRegex(ValueError, 'silent no-EM'):
            bounded_review_payload(source, self.tokenizer)
        source['evolution_memory'] = [dict(id='large-design', hypothesis='x' * 10000)]
        baseline = bounded_review_payload(payload(), self.tokenizer)
        baseline_tokens = self.tokenizer.chat_count([
            {'role': 'system', 'content': REVIEW_PROMPT},
            {'role': 'user', 'content': json.dumps(baseline, ensure_ascii=False, sort_keys=True, separators=(',', ':'))},
        ])
        source_size = (MAX_INPUT_TOKENS - baseline_tokens - 1000) * 4
        source['evidence'][0]['data']['source'] = '# ' + 'x' * source_size + '\n' + SOURCE
        with self.assertRaisesRegex(ValueError, 'reserved Evolution Memory'):
            bounded_review_payload(source, self.tokenizer)

    def test_http_and_truncated_completion_failures_preserve_receipts(self):
        for response in (RuntimeError('synthetic HTTP transport failure'), FakeResponse(finish='length'),
                         FakeResponse(content='not JSON'), FakeResponse(dict(report(), approve=True))):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                reviewer, client = self.make_reviewer(directory, [response])
                with self.assertRaises(Exception):
                    reviewer(dict(payload(), review_contract=2), 14)
                receipt = json.loads(reviewer.ledger.read_text())
                self.assertFalse(receipt['ok'])
                self.assertTrue(receipt['api_attempted'])
                self.assertTrue(receipt['error'])
                self.assertIsNone(receipt['validated_report'])
                self.assertEqual(receipt['request'], client.requests[0][1])
                self.assertEqual(receipt['system_prompt'], REVIEW_PROMPT)
                self.assertIn('usage', receipt)

    def test_heldout_or_invalid_stage_rejected_without_http(self):
        variants = []
        value = payload(); value['stage'] = 'final'; variants.append(value)
        value = payload(); value['evidence'][2]['data']['phase'] = 'round-0/confirmation'; variants.append(value)
        value = payload(); value['evidence'][2]['data']['final_reward'] = 1; variants.append(value)
        for value in variants:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                value['review_contract'] = 2
                reviewer, client = self.make_reviewer(directory, [])
                with self.assertRaises(ValueError):
                    reviewer(value, 15)
                receipt = json.loads(reviewer.ledger.read_text())
                self.assertFalse(receipt['api_attempted'])
                self.assertEqual(receipt['kind'], 'review_input_rejected')
                self.assertEqual(client.requests, [])

    def test_new_http_calls_cannot_select_historical_unbound_report_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            reviewer, client = self.make_reviewer(directory, [])
            with self.assertRaisesRegex(ValueError, 'legacy reports are read-only'):
                reviewer(payload(), 15)
            self.assertFalse(client.requests)

    def test_oversized_full_source_is_rejected_instead_of_truncated(self):
        source = payload()
        source['evidence'][0]['data']['source'] = '# ' + 'x' * (MAX_INPUT_TOKENS * 4 + 1000) + '\n' + SOURCE
        with tempfile.TemporaryDirectory() as directory:
            reviewer, client = self.make_reviewer(directory, [])
            with self.assertRaisesRegex(ValueError, 'source was not truncated'):
                reviewer(dict(source, review_contract=2), 16)
            self.assertEqual(client.requests, [])
            receipt = json.loads(reviewer.ledger.read_text())
            self.assertFalse(receipt['api_attempted'])
            self.assertIsNone(receipt['displayed_payload'])


if __name__ == '__main__':
    unittest.main()
