"""Read-only report auditing of fabricated CPU import receipts, no inference."""
import copy
from unittest import TestCase

from evomem.common import append, file_hash, read, write
from phase1_codegen.baseline_reuse import seed_source_baseline
from phase1_codegen.reporting.campaign import Reader, build_report, markdown, source_baseline_reuse
from tests.core import test_baseline_reuse as fixtures


class BaselineReuseReportTests(TestCase):
    def setUp(self):
        self.fixture = fixtures.BaselineReuseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root / 'new'
        seed_source_baseline(self.fixture.engine, self.fixture.origin)
        self.fixture.engine.store.initialize(self.fixture.program, self.fixture.engine.binding)
        write(self.root / 'protocol.json', {'source_baseline_origins': {'textworld': self.fixture.origin}})
        write(self.root / 'datasets/textworld/source_exploration.json', self.fixture.tasks)

    def audit(self):
        return source_baseline_reuse(Reader(self.root), 'textworld', self.fixture.origin, self.fixture.tasks)

    def test_valid_import_remains_separate_from_new_native_and_inference_cost(self):
        before = {p: file_hash(p) for p in self.fixture.root.rglob('*') if p.is_file()}
        audit = self.audit()
        self.assertTrue(audit['verified'], audit['checks'])
        self.assertEqual(audit['imported_real_rows'], 2)
        self.assertEqual(audit['historical_costs_for_this_phase_only']['actor']['known_prompt_tokens'], 200)
        self.assertEqual(audit['historical_native_costs']['env_steps'], 2)
        self.assertEqual(audit['integrity_findings'], [])
        report = build_report(self.root)
        self.assertEqual(report['physical_native']['total']['real_rows'], 0)
        self.assertEqual(report['physical_native']['imported_adaptation_real_rows'], 0)
        self.assertEqual(report['physical_native']['imported_source_baseline_real_rows'], 2)
        self.assertEqual(report['physical_native']['imported_existing_total_real_rows'], 2)
        self.assertEqual(report['model_costs']['by_role'], {})
        self.assertIn('历史 Actor token 不计入本版本新推理', markdown(report))
        after = {p: file_hash(p) for p in self.fixture.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        # Later physical execution is separate; imported historical usage stays excluded.
        episode = dict(self.fixture.episodes[0], phase='round-000/exploration/candidate')
        append(self.fixture.new / 'runner/native/canonical_episodes.jsonl', episode)
        append(self.fixture.new / 'runner/native/costs.jsonl', dict(kind='model', role='actor',
            phase=episode['phase'], prompt_tokens=5, completion_tokens=2, ok=True))
        report = build_report(self.root)
        self.assertEqual(report['physical_native']['total']['real_rows'], 1)
        self.assertEqual(report['physical_native']['imported_source_baseline_real_rows'], 2)
        self.assertEqual(report['model_costs']['by_role']['actor']['known_prompt_tokens'], 5)

    def test_cache_receipt_protocol_pins_and_access_tampering_are_flagged(self):
        audit = read(self.fixture.new / 'baseline_reuse.json')
        cache_path = self.fixture.new / 'measurements' / (audit['measurement_key'] + '.json')
        changes = [
            (cache_path, lambda row: row['episodes'][0].update(score=0.3), 'cache_identity'),
            (self.fixture.new / 'baseline_reuse.json', lambda row: row.update(physical=True), 'import_receipt_identity'),
            (self.fixture.old / 'state.json', lambda row: row.update(binding_hash='0' * 64), 'source_state_pin'),
            (self.fixture.old / 'measurements' / (self.fixture.origin['measurement_key'] + '.json'),
             lambda row: row.update(phase='heldout'), 'source_measurement_pin'),
        ]
        for path, mutate, expected in changes:
            with self.subTest(check=expected):
                original_bytes = path.read_bytes()
                row = read(path); mutate(row); write(path, row)
                result = self.audit()
                self.assertFalse(result['verified'])
                self.assertEqual(result['imported_real_rows'], 0)
                self.assertIn(expected, {item['check'] for item in result['integrity_findings']})
                path.write_bytes(original_bytes)
        append(self.fixture.new / 'measurement_access.jsonl', dict(key=audit['measurement_key'], physical=True))
        self.assertFalse(self.audit()['checks']['import_access_receipt'])

    def test_historical_cost_drift_and_missing_protocol_do_not_verify(self):
        append(self.fixture.old / 'runner/native/costs.jsonl', dict(kind='model', phase='irrelevant'))
        self.assertFalse(self.audit()['checks']['source_costs_pin'])
        missing = source_baseline_reuse(Reader(self.root), 'textworld', None, self.fixture.tasks)
        self.assertFalse(missing['checks']['protocol_origin_schema'])
        self.assertEqual(missing['imported_real_rows'], 0)

    def test_old_runs_without_import_are_normal_and_legacy_json_still_renders(self):
        root = self.fixture.root / 'legacy'
        result = build_report(root)
        self.assertEqual(result['source_baseline_reuse']['textworld']['status'], 'not_configured')
        self.assertEqual(result['source_baseline_reuse']['textworld']['integrity_findings'], [])
        self.assertEqual(result['physical_native']['imported_source_baseline_real_rows'], 0)
        legacy = copy.deepcopy(result)
        legacy.pop('source_baseline_reuse')
        for field in ('imported_adaptation_real_rows', 'imported_source_baseline_real_rows', 'imported_existing_total_real_rows'):
            legacy['physical_native'].pop(field)
        self.assertNotIn('另已核验并导入源机制', markdown(legacy))
