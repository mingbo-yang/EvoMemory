"""Contract diagnostics are exact model input, never generated-code repair."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phase1_codegen.policy import Rejected, validate_output, validate_source
from phase1_codegen.protocol import EVENT_CONTRACT
from phase1_codegen.review import (
    MAX_INPUT_TOKENS, MechanismReviewer, bounded_review_payload, validate_report,
    expected_assessment_subject,
)
from phase1_codegen.tests.test_review import ApproximateTokenizer, FakeClient, FakeResponse
from phase1_codegen.tests.test_review_contract import contract_payload, bundled_raw_report


def rejected_payload(error):
    payload = contract_payload()
    validation = next(row['data'] for row in payload['evidence'] if row['id'] == 'candidate:validation')
    validation.update(status='invalid', error=error, mechanism_validated=False,
                      candidate_execution_performed=False)
    # A rejected implementation has no fabricated candidate execution evidence.
    payload['evidence'] = [row for row in payload['evidence'] if not (
        row['id'].startswith('candidate:') and row['kind'] not in {'program', 'validation'})]
    return payload


class ContractFeedbackReviewTests(unittest.TestCase):
    def test_source_reports_callbacks_lambda_and_unsafe_handlers_with_locations_together(self):
        source = '''def on_event(state, event, context):
    rows = sorted([], key=lambda x: x)
    try:
        value = 1
    except Exception:
        value = 0
    return {"state": state, "session": {}, "memory": []}
'''
        with self.assertRaises(Rejected) as caught:
            validate_source(source)
        error = str(caught.exception)
        for expected in ("actual keyword: 'key'", 'forbidden syntax: Lambda',
                         'except requires static permitted data exceptions',
                         'line 2', 'line 5'):
            self.assertIn(expected, error)

    def test_invalid_python_and_wrong_entrypoint_identify_location(self):
        for source, expected in [
            ('def on_event(state, event, context)\n    return {}', 'line 1'),
            ('def on_event(state, query):\n    return {}', "actual arguments: ['state', 'query']"),
        ]:
            with self.subTest(source=source), self.assertRaises(Rejected) as caught:
                validate_source(source)
            self.assertIn(expected, str(caught.exception))

    def test_many_source_violations_keep_bounded_explicit_diagnostics(self):
        source = 'def on_event(state, event, context):\n' + '    value = lambda x: x\n' * 100
        with self.assertRaises(Rejected) as caught:
            validate_source(source)
        error = str(caught.exception)
        self.assertEqual(error.count('forbidden syntax: Lambda'), 12)
        self.assertIn('88 additional source violations omitted', error)
        self.assertLess(len(error), 4000)

    def test_output_metadata_remains_legal_inside_state_but_not_in_actor_rows(self):
        row = dict(id='episode-1', experience='recorded fact', keywords=['fact'])
        value = dict(state={'episodes': [row]}, session={}, memory=[row])
        original = copy.deepcopy(value)
        with self.assertRaises(Rejected) as caught:
            validate_output('event', value)
        self.assertIn('path=/memory/0', str(caught.exception))
        self.assertIn("actual=['experience', 'id', 'keywords']", str(caught.exception))
        self.assertEqual(value, original)
        value['memory'] = [dict(id=row['id'], experience=row['experience'])]
        self.assertEqual(validate_output('event', value), value)

    def test_wrong_row_types_and_duplicates_have_precise_paths(self):
        rows = [dict(id='same', experience='a'), dict(id='same', experience='b')]
        with self.assertRaises(Rejected) as caught:
            validate_output('event', dict(state={}, session={}, memory=rows))
        self.assertIn('path=/memory/1', str(caught.exception))
        self.assertIn('duplicate id=True', str(caught.exception))
        with self.assertRaises(Rejected) as caught:
            validate_output('event', dict(state={}, session=[], memory=None))
        self.assertIn('path=/session; actual=list', str(caught.exception))
        with self.assertRaises(Rejected) as caught:
            validate_output('event', dict(state={'entries': [{'value': {1, 2}}]}, session={}, memory=None))
        self.assertIn('non-JSON value: set; path=/state/entries/0/value', str(caught.exception))

    def test_exact_error_survives_optional_trace_pressure_and_is_citable(self):
        error = "Rejected: forbidden syntax: Try (line 58, column 9)"
        payload = rejected_payload(error)
        payload['evidence'].extend(dict(id='extra-' + str(i), kind='execution_trace',
            data={'history': 'optional trace ' * 4000}) for i in range(5))
        shown = bounded_review_payload(payload, ApproximateTokenizer())
        facts = next(row['data'] for row in shown['evidence'] if row['id'] == 'current:facts')
        self.assertEqual(shown['candidate_validation_summary']['error'], error)
        self.assertEqual(facts['validation']['error'], error)
        pointers = [pointer for group in shown['citation_registry']['evidence']['current:facts']
                    for pointer in group['pointers'].values()]
        self.assertIn('/validation/error', pointers)
        self.assertEqual(shown['event_contract'], EVENT_CONTRACT)
        self.assertNotIn('event_contract', shown['citation_registry']['evidence'])

    def test_oversized_error_is_rejected_instead_of_truncated(self):
        payload = rejected_payload('x' * (MAX_INPUT_TOKENS * 4))
        with self.assertRaisesRegex(ValueError, 'fixed input budget'):
            bounded_review_payload(payload, ApproximateTokenizer())

    def test_actual_reviewer_request_keeps_error_and_compiler_checks_exact_citation(self):
        error = "Rejected: event memory row 0 requires exactly id and experience; path=/memory/0; actual=['experience', 'id', 'keywords']"
        payload = rejected_payload(error)
        shown = bounded_review_payload(payload, ApproximateTokenizer())
        cid = next(cid for group in shown['citation_registry']['evidence']['current:facts']
                   for cid, pointer in group['pointers'].items() if pointer == '/validation/error')
        raw = bundled_raw_report(shown, verdict='insufficient_evidence')
        raw['focus_evidence_ids'] = [cid]
        client = FakeClient([FakeResponse(raw)])
        with tempfile.TemporaryDirectory() as directory:
            with patch('httpx.Client', return_value=client):
                reviewer = MechanismReviewer(ApproximateTokenizer(), Path(directory) / 'review.jsonl', 'synthetic')
            self.addCleanup(reviewer.close)
            report = reviewer(payload, 11)
        actual = json.loads(client.requests[0][1]['messages'][1]['content'])
        self.assertEqual(actual['candidate_validation_summary']['error'], error)
        self.assertEqual(actual['event_contract'], EVENT_CONTRACT)
        observation = next(row for row in report['evidence_contract']['observations']
                           if row['pointer'] == '/validation/error')
        self.assertEqual(observation['value'], error)
        changed = copy.deepcopy(report)
        next(row for row in changed['evidence_contract']['observations']
             if row['pointer'] == '/validation/error')['value'] = 'invented error'
        with self.assertRaises(ValueError):
            validate_report(changed, shown)
        raw['assessment']['verdict'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'cannot receive a positive'):
            validate_report(raw, shown)

    def test_empty_source_candidate_completes_post_review_from_validation_evidence(self):
        error = 'Rejected: source must be nonempty Python within 32768 bytes'
        payload = rejected_payload(error)
        payload['candidate_program_hash'] = None
        payload['evidence'] = [row for row in payload['evidence'] if row['id'] != 'candidate:program']
        shown = bounded_review_payload(payload, ApproximateTokenizer())
        self.assertEqual(expected_assessment_subject(payload), 'candidate')
        self.assertEqual(expected_assessment_subject(shown), 'candidate')
        self.assertEqual(shown['expected_assessment_subject'], 'candidate')
        # The optional original validation row may be omitted under pressure.
        shown['evidence'] = [row for row in shown['evidence'] if row['id'] != 'candidate:validation']
        self.assertEqual(expected_assessment_subject(shown), 'candidate')
        cid = next(cid for group in shown['citation_registry']['evidence']['current:facts']
                   for cid, pointer in group['pointers'].items() if pointer == '/validation/error')
        raw = bundled_raw_report(shown, verdict='insufficient_evidence')
        raw['focus_evidence_ids'] = [cid]
        client = FakeClient([FakeResponse(raw)])
        with tempfile.TemporaryDirectory() as directory:
            with patch('httpx.Client', return_value=client):
                reviewer = MechanismReviewer(ApproximateTokenizer(), Path(directory) / 'review.jsonl', 'synthetic')
            self.addCleanup(reviewer.close)
            report = reviewer(payload, 12)
            receipt = json.loads((Path(directory) / 'review.jsonl').read_text())
        actual = json.loads(client.requests[0][1]['messages'][1]['content'])
        self.assertTrue(receipt['ok'])
        self.assertEqual(actual['expected_assessment_subject'], 'candidate')
        self.assertFalse(any(row['id'] == 'candidate:program' for row in actual['evidence']))
        self.assertEqual(report['assessment']['subject'], 'candidate')
        self.assertEqual(report['assessment']['verdict'], 'insufficient_evidence')
        self.assertEqual(next(row['value'] for row in report['evidence_contract']['observations']
                              if row['pointer'] == '/validation/error'), error)
        raw['assessment']['verdict'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'cannot receive a positive'):
            validate_report(raw, actual)
        self.assertEqual(expected_assessment_subject(dict(payload, stage='pre')), 'current')
