"""Explicit diagnostic judgments bind displayed facts, without judging by rule."""
import copy
import unittest
from evomem.common import digest
from phase1_codegen.review import bounded_review_payload, validate_report, review_response_format
from phase1_codegen.tests.test_review import ApproximateTokenizer
from phase1_codegen.tests.test_review_contract import contract_payload, bundled_raw_report


def payload():
    value = contract_payload()
    diagnostic = dict(facts=dict(missing_dict_keys=[dict(function='retrieve_entries', line=30,
        operation='task_start', expression='context.get("visible", {})', calls=14,
        first_missing=dict(key='visible', receiver_keys=['environment', 'task_id'], receiver_key_count=2,
                           result={'type': 'dict', 'length': 0}, default={'type': 'dict', 'length': 0}))],
        branch_counts=[dict(function='score_from_text', line=5, operation='task_end',
            expression='tail.isdigit()', calls=12, true_count=0, false_count=12)]))
    value['evidence'].append(dict(id='candidate:behavior', kind='behavior_diagnostics', data=diagnostic))
    return value


class FunctionalReviewTests(unittest.TestCase):
    def setUp(self):
        self.shown = bounded_review_payload(payload(), ApproximateTokenizer())

    def test_every_group_is_required_in_wire_and_bound_in_report(self):
        groups = self.shown['diagnostic_groups']
        self.assertEqual(len(groups), 2)
        schema = review_response_format(self.shown)['json_schema']['schema']
        self.assertEqual(set(schema['properties']['diagnostic_assessments']['required']),
                         {group['group_id'] for group in groups})
        raw = bundled_raw_report(self.shown)
        for item in raw['diagnostic_assessments'].values():
            item['judgment'] = 'supported_defect'
        report = validate_report(raw, self.shown)
        diagnoses = report['evidence_contract']['diagnostic_assessments']
        self.assertEqual(len(diagnoses), 2)
        self.assertTrue(all(item['judgment'] == 'supported_defect' for item in diagnoses))
        self.assertEqual(validate_report(report, self.shown), report)
        self.assertEqual(report['assessment']['verdict'], 'neutral')

    def test_missing_group_or_authored_facts_rejected(self):
        raw = bundled_raw_report(self.shown)
        raw['diagnostic_assessments'].pop(next(iter(raw['diagnostic_assessments'])))
        with self.assertRaisesRegex(ValueError, 'every displayed'):
            validate_report(raw, self.shown)
        raw = bundled_raw_report(self.shown)
        next(iter(raw['diagnostic_assessments'].values()))['code'] = 'a prescribed repair'
        with self.assertRaisesRegex(ValueError, 'judgment and uncertainty only'):
            validate_report(raw, self.shown)

    def test_repeated_wire_tags_preserve_first_order_and_all_opinions(self):
        raw = bundled_raw_report(self.shown)
        opinions = [*raw['evaluations'], *raw['diagnostic_assessments'].values()]
        for item in opinions:
            item['uncertainty'] = ['limited_tasks', 'clipped_evidence', 'limited_tasks']
        original = copy.deepcopy(raw)
        report = validate_report(raw, self.shown)
        self.assertEqual(original, raw)
        for item in opinions:
            item['uncertainty'] = ['limited_tasks', 'clipped_evidence']
        self.assertEqual(validate_report(raw, self.shown), report)
        self.assertEqual(validate_report(report, self.shown), report)
        for group in ('evaluations', 'diagnostic_assessments'):
            altered = copy.deepcopy(report)
            altered['evidence_contract'][group][0]['uncertainty'].append('limited_tasks')
            with self.subTest(group=group), self.assertRaises(ValueError):
                validate_report(altered, self.shown)

    def test_duplicate_tags_do_not_mask_invalid_wire_input(self):
        for tags in ([], 'limited_tasks', ['limited_tasks'] * 7,
                     ['limited_tasks', 'limited_tasks', 'Limited_tasks'], [['limited_tasks']]):
            for group in ('evaluations', 'diagnostic_assessments'):
                raw = bundled_raw_report(self.shown)
                opinions = raw[group]
                item = opinions[0] if group == 'evaluations' else next(iter(opinions.values()))
                item['uncertainty'] = tags
                with self.subTest(group=group, tags=tags), self.assertRaisesRegex(ValueError, 'fixed tags'):
                    validate_report(raw, self.shown)

    def test_no_automatic_missing_key_defect_or_positive_verdict(self):
        for judgment in ['no_defect_demonstrated', 'insufficient_evidence']:
            raw = bundled_raw_report(self.shown)
            for item in raw['diagnostic_assessments'].values():
                item['judgment'] = judgment
            report = validate_report(raw, self.shown)
            self.assertTrue(all(d['judgment'] == judgment for d in report['diagnostics']))
            self.assertEqual(report['assessment']['verdict'], 'neutral')

    def test_diagnostic_counts_location_and_snapshot_cannot_be_changed(self):
        report = validate_report(bundled_raw_report(self.shown), self.shown)
        for key, val in [('line', 999), ('original_hash', 'changed')]:
            changed = copy.deepcopy(report)
            changed['evidence_contract']['diagnostic_assessments'][0][key] = val
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'facts.*changed'):
                validate_report(changed, self.shown)
        changed = copy.deepcopy(report)
        changed['evidence_contract']['diagnostic_assessments'][0]['scope']['program_hash'] = 'old-snapshot'
        with self.assertRaisesRegex(ValueError, 'facts.*changed'):
            validate_report(changed, self.shown)

    def test_group_coverage_bounded_and_omissions_exact(self):
        raw = payload()
        row = raw['evidence'][-1]['data']
        row['facts']['branch_counts'] = [dict(function='site'+str(i), line=i+1,
            operation='task_start', expression='flag', calls=1, true_count=0, false_count=1) for i in range(40)]
        shown = bounded_review_payload(raw, ApproximateTokenizer())
        coverage = shown['input_selection']['diagnostic_group_coverage']
        self.assertEqual(coverage['total_groups'], 41)
        self.assertEqual(coverage['omitted_groups'], 29)
        self.assertEqual(len(shown['diagnostic_groups']), 12)
        for group in shown['diagnostic_groups']:
            self.assertEqual(group['group_id'], 'd'+digest({k:v for k,v in group.items() if k!='group_id'})[:12])


if __name__ == '__main__':
    unittest.main()
