"""Synthetic transport checks: verified reports reach the designer intact."""
import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phase1_codegen.proposer import CodeEvolver
from phase1_codegen.protocol import REVIEW_FEEDBACK_TOKENS
from phase1_codegen.review import compact_review_feedback, expand_review_feedback
from phase1_codegen.engine import review_observation_context
from phase1_codegen.tests.test_review import CharacterTokenizer, FakeClient, FakeResponse


class ReviewDeliveryTests(unittest.TestCase):
    def test_numeric_diagnostics_retain_exact_code_location_without_unrelated_content(self):
        shown = dict(evidence=[dict(id='candidate:behavior', data=dict(facts=dict(branch_counts=[
            dict(function='on_event', line=42, expression='len(rows) > 4', operation='task_end',
                 true_count=28, false_count=4, unrelated_text='do not copy this')])))] )
        observations = [dict(id='o1', evidence_id='candidate:behavior', pointer='/facts/branch_counts/0/true_count'),
                        dict(id='o2', evidence_id='candidate:behavior', pointer='/facts/branch_counts/0/false_count')]
        result = review_observation_context(shown, dict(evidence_contract=dict(observations=observations)))
        self.assertEqual(result, [dict(evidence_id='candidate:behavior', pointer='/facts/branch_counts/0',
            observation_ids=['o1', 'o2'], metadata=dict(function='on_event', line=42,
                                                      expression='len(rows) > 4', operation='task_end'))])

    def test_large_duplicate_evidence_cannot_displace_complete_verified_report(self):
        contract = dict(schema=2, observations=[dict(id='o1', value='exact permitted fact')],
                        evaluations=[], assessment=dict(subject='current', verdict='neutral'),
                        displayed_payload_hash='synthetic-display')
        package = dict(current_program_hash='synthetic-program', report_hash='synthetic-report',
                       evidence_refs=[dict(data='duplicate display ' * 10000)],
                       report=dict(evidence_contract=contract, summary='duplicated rendering ' * 1000))
        response = dict(source='synthetic source', hypothesis='Synthetic fixture', risk='No effect claim')
        client = FakeClient([FakeResponse(response)])
        with tempfile.TemporaryDirectory() as directory:
            with patch('httpx.Client', return_value=client):
                evolver = CodeEvolver(CharacterTokenizer(), Path(directory) / 'costs.jsonl', 'synthetic-model')
            self.addCleanup(evolver.close)
            self.assertEqual(evolver(dict(mechanism_review=package), 1), response)
        transported = json.loads(client.requests[0][1]['messages'][1]['content'])['mechanism_review']
        self.assertEqual(expand_review_feedback(transported['report']), contract)
        self.assertEqual(transported['report_hash'], 'synthetic-report')
        self.assertNotIn('evidence_refs', transported)
        self.assertNotIn('serialized_evidence', transported)

    def test_over_budget_contract_fails_before_inference_instead_of_truncating_facts(self):
        client = FakeClient([])
        package = dict(report=dict(evidence_contract=dict(observations=['x' * (REVIEW_FEEDBACK_TOKENS + 1)])))
        with tempfile.TemporaryDirectory() as directory:
            with patch('httpx.Client', return_value=client):
                evolver = CodeEvolver(CharacterTokenizer(), Path(directory) / 'costs.jsonl', 'synthetic-model')
            self.addCleanup(evolver.close)
            with self.assertRaisesRegex(ValueError, 'Complete verified mechanism review'):
                evolver(dict(mechanism_review=package), 1)
        self.assertEqual(client.requests, [])

    def test_shared_scopes_and_witnesses_are_lossless_and_tamper_evident(self):
        scope = dict(program_hash='program', task_id=None, memory_snapshot_hash='snapshot',
                     environment='alfworld', round=0, phase='current')
        contract = dict(schema=2, observations=[dict(scope=scope, value=i) for i in range(12)],
                        diagnostic_assessments=[dict(scope=scope, original_hash='x' * 64,
                            assignment_witnesses=[{'sample_at_assignment': {'type': 'str', 'value': '1.0'}}])
                            for i in range(12)])
        packed = compact_review_feedback(contract)
        self.assertEqual(expand_review_feedback(packed), contract)
        self.assertLess(len(json.dumps(packed)), len(json.dumps(contract)))
        packed['shared_values'][next(iter(packed['shared_values']))] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            expand_review_feedback(packed)

    def test_previous_feedback_uses_same_lossless_transport_without_mutating_payload(self):
        contract = dict(schema=2, observations=[dict(scope={'program_hash': 'p'}, value=1)],
                        evaluations=[], assessment={'subject': 'candidate', 'verdict': 'neutral'})
        payload = dict(previous_attempt=dict(source='complete earlier source',
            post_review={'report': contract}), current_source='complete deployed source')
        original = copy.deepcopy(payload)
        response = dict(source='synthetic source', hypothesis='Synthetic fixture', risk='No effect claim')
        client = FakeClient([FakeResponse(response)])
        with tempfile.TemporaryDirectory() as directory:
            with patch('httpx.Client', return_value=client):
                evolver = CodeEvolver(CharacterTokenizer(), Path(directory) / 'costs.jsonl', 'synthetic-model')
            self.addCleanup(evolver.close)
            evolver(payload, 1)
        sent = json.loads(client.requests[0][1]['messages'][1]['content'])
        self.assertEqual(expand_review_feedback(sent['previous_attempt']['post_review']['report']), contract)
        self.assertEqual(payload, original)
