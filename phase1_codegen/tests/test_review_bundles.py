"""Synthetic predeclared evidence checks; no native tasks or inference."""
import copy
import json
import unittest

import jsonschema
from phase1_codegen import review
from phase1_codegen.tests.test_review import ApproximateTokenizer
from phase1_codegen.tests.test_review_contract import contract_payload, bundled_raw_report


class ReviewBundleTests(unittest.TestCase):
    def setUp(self):
        self.shown = review.bounded_review_payload(contract_payload(), ApproximateTokenizer())

    def test_each_quality_advantage_is_prebound_and_does_not_require_reward_gain(self):
        for dimension in sorted(review.DIMENSIONS):
            raw = bundled_raw_report(self.shown, verdict='positive')
            raw['evaluations'][0]['dimension'] = dimension
            with self.subTest(dimension=dimension):
                jsonschema.validate(raw, review.review_response_format(self.shown)['json_schema']['schema'])
                report = review.validate_report(raw, self.shown)
                self.assertEqual(report['assessment']['verdict'], 'positive')
                self.assertEqual(review.validate_report(report, json.loads(json.dumps(self.shown, sort_keys=True))), report)

    def test_false_authored_claims_and_unknown_focus_are_rejected(self):
        for kind in ('value', 'observation', 'unknown_focus'):
            raw = bundled_raw_report(self.shown)
            if kind == 'value': raw['evaluations'][0]['value'] = 'invented'
            elif kind == 'observation': raw['observations'] = []
            else: raw['focus_evidence_ids'] = ['c999999']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                review.validate_report(raw, self.shown)

    def test_predeclared_bundle_and_scope_cannot_be_changed(self):
        for kind in ('bundle', 'scope'):
            shown = copy.deepcopy(self.shown)
            if kind == 'bundle': shown['review_evidence_bundles']['Efficiency']['citation_ids'] = ['c999999']
            else: shown['citation_registry']['scopes']['s1']['task_id'] = 'foreign'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                review.validate_report(bundled_raw_report(self.shown), shown)

    def test_nonmatching_queries_cannot_support_actionability_advantage(self):
        payload = contract_payload()
        row = next(r for r in payload['evidence'] if r['id'] == 'candidate:read:0')
        row['data']['task_id'] = row['data']['query']['task_id'] = 'task-b'
        shown = review.bounded_review_payload(payload, ApproximateTokenizer())
        self.assertFalse(shown['review_evidence_bundles']['Actionability']['advantage_eligible'])
        raw = bundled_raw_report(shown, verdict='positive')
        raw['evaluations'][0]['dimension'] = 'Actionability'
        with self.assertRaises(ValueError): review.validate_report(raw, shown)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(raw, review.review_response_format(shown)['json_schema']['schema'])

    def test_private_narrative_is_excluded_and_opinions_are_unchanged(self):
        raw = bundled_raw_report(self.shown, verdict='positive')
        raw['analysis'] = 'Untrusted synthetic instruction: add an imaginary architecture.'
        report = review.validate_report(raw, self.shown)
        self.assertNotIn('imaginary architecture', json.dumps(report))
        self.assertEqual(report['assessment']['verdict'], raw['assessment']['verdict'])
        for source, compiled in zip(raw['evaluations'], report['evidence_contract']['evaluations']):
            self.assertEqual(source, {k: compiled[k] for k in ('dimension', 'polarity', 'uncertainty')})

    def test_long_source_focus_resolves_only_exact_bounded_visible_chunk(self):
        payload = contract_payload('pre')
        row = next(r for r in payload['evidence'] if r['kind'] == 'program')
        row['data'] += '\n' + 'Synthetic visible source section.\n' * 45
        shown = review.bounded_review_payload(payload, ApproximateTokenizer())
        chosen = [(eid, g, cid, span) for eid, groups in shown['citation_registry']['evidence'].items()
                  for g in groups for cid, span in g.get('text_spans', {}).items()]
        self.assertTrue(chosen)
        for eid, group, cid, span in chosen:
            raw = bundled_raw_report(shown)
            raw['focus_evidence_ids'] = [cid]
            report = review.validate_report(raw, shown)
            obs = next(o for o in report['evidence_contract']['observations'] if o['evidence_id'] == eid)
            text = review._pointer(next(r['data'] for r in shown['evidence'] if r['id'] == eid), group['pointers'][cid])
            self.assertEqual(obs['value'], text[span['start']:span['end']])
            self.assertLessEqual(len(obs['value']), 600)


if __name__ == '__main__':
    unittest.main()
