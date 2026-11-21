"""Read-only reporting contracts with synthetic on-disk receipts, no model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from phase1_codegen.reporting.campaign import Reader, build_report, digest, markdown, memory_model_counts, memory_model_groups, memory_model_report


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class CapabilityReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-capability-report-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def example(self, env='alfworld', phase='source/no_external_em', model='test-model'):
        pool = self.root / 'capability_pools' / env
        manifest = dict(environment=env, models=[dict(id=model, kind='test-fake')])
        save(pool / 'manifest.json', manifest)
        fingerprint = digest(manifest)
        cache = dict(key='test-cache', model_id=model, capability_fingerprint=fingerprint,
                     environment=env, request={'text': 'synthetic fixture'}, value={'answer': 'fake'},
                     usage={'input_tokens': 11, 'completion_tokens': 0}, ok=True, error=None,
                     inference_attempted=True, seconds=.2)
        cache['record_hash'] = digest(cache)
        path = pool / 'cache/test-cache.json'
        save(path, cache)
        row = dict(kind='memory_model_call', environment=env, phase=phase, operation='write',
                   model_id=model, request_hash='broker-identity', cache_key='test-cache', cache_hit=False,
                   physical_attempt=True, inference_attempted=True, physical=True, ok=True, error=None,
                   cache_path=str(path), cache_record_hash=cache['record_hash'],
                   capability_fingerprint=fingerprint, usage=cache['usage'], seconds=.2,
                   measurement_kind='synthetic unit-test fixture')
        return row, dict(synthetic=False, model_capabilities=fingerprint)

    def test_requests_backend_inference_failures_and_tokens_have_separate_counts(self):
        physical, _ = self.example()
        hit = dict(physical, cache_hit=True, physical_attempt=False, inference_attempted=False,
                   physical=False, usage={'input_tokens': 9999, 'completion_tokens': 99})
        backend_failure = dict(physical, cache_key='load-failed', cache_record_hash='load-failed',
                               ok=False, error='load failed', inference_attempted=False, physical=False, usage={})
        inference_failure = dict(physical, cache_key='inference-failed', cache_record_hash='inference-failed',
                                 ok=False, error='inference failed', physical=False,
                                 usage={'prompt_tokens': 7, 'completion_tokens': 2})
        rejected = dict(kind='memory_model_call', environment='alfworld', phase='source/no_external_em',
                        model_id='test-model', ok=False, error='invalid payload', cache_hit=False, physical=False)
        rows = [physical, hit, backend_failure, inference_failure, rejected, copy.deepcopy(physical)]
        stats = memory_model_counts(rows)
        self.assertEqual(stats['logical_requests'], 6)
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['backend_attempts'], 4)
        self.assertEqual(stats['inference_attempts'], 3)
        self.assertEqual(stats['completed_physical'], 2)
        self.assertEqual(stats['failed'], 3)
        self.assertEqual(stats['failed_backend_attempts'], 2)
        self.assertEqual(stats['failed_inference_attempts'], 1)
        self.assertEqual(stats['known_input_tokens'], 18)
        self.assertEqual(stats['known_output_tokens'], 2)
        self.assertEqual(stats['deduplicated_usage_rows'], 1)
        self.assertEqual(stats['cache_usage_rows_ignored'], 1)
        self.assertEqual(stats['rows_missing_inference_attempted'], 1)

    def test_usage_never_substitutes_local_estimates_and_models_are_not_hardcoded(self):
        row, _ = self.example(model='new-multimodal-model')
        row['usage'] = {'local_input_tokens': 300, 'batch_size': 1}
        other = dict(row, environment='textworld', phase='sealed_generalization')
        groups = memory_model_groups([row, other])
        self.assertEqual(len(groups), 2)
        for value in groups:
            self.assertEqual(value['model_id'], 'new-multimodal-model')
            self.assertEqual(value['known_input_tokens'], 0)
            self.assertEqual(value['inference_rows_missing_usage'], 1)

    def test_actual_trace_receipts_are_verified_without_counting_copies_as_new_calls(self):
        row, binding = self.example()
        folder = self.root / 'source/alfworld'
        ledger(folder / 'capability_costs.jsonl', [row])
        call = dict(model_id=row['model_id'], operation='write', request_hash='runtime-identity', receipt=row)
        trace = dict(task_id='fixture-task', capability_calls=[call])
        save(folder / 'trials/000/behavior.json', {'parent': {'write_trace': [trace]},
                                                  'candidate': {'write_trace': [trace]}})
        ledger(folder / 'runner/native/code_memory_traces.jsonl', [dict(
            kind='generated_memory_native_episode', synthetic=False, environment='alfworld',
            read_calls=[], write_calls=[trace])])
        audit = memory_model_report(Reader(self.root), folder, 'alfworld', binding)
        self.assertEqual(audit['totals']['logical_requests'], 1)
        self.assertEqual(audit['mechanism_execution']['recorded_trace_call_references'], 3)
        self.assertEqual(audit['mechanism_execution']['verified_trace_call_references'], 3)
        self.assertEqual(audit['mechanism_execution']['unique_verified_broker_receipts'], 1)
        self.assertEqual(audit['integrity_findings'], [])
        self.assertTrue(audit['manifest']['binding_matches'])
        self.assertTrue(all(audit['cache_receipts'][0]['checks'].values()))
        provenance = {item['provenance'] for item in audit['mechanism_execution']['evidence']}
        self.assertEqual(provenance, {'mechanism_replay', 'native_episode_memory_execution'})
        # Different runtime/broker request hash scopes are expected, not corruption.
        self.assertNotEqual(call['request_hash'], row['request_hash'])

    def test_static_source_keyword_is_not_execution_evidence_and_tampering_is_detected(self):
        row, binding = self.example()
        folder = self.root / 'source/alfworld'
        ledger(folder / 'capability_costs.jsonl', [row])
        (folder / 'programs').mkdir(parents=True)
        (folder / 'programs/fixture.py').write_text('def read(state, context):\n    return model_call("fake", {})\n')
        first = memory_model_report(Reader(self.root), folder, 'alfworld', binding)
        self.assertEqual(first['mechanism_execution']['verified_trace_call_references'], 0)
        cache_path = Path(row['cache_path'])
        value = json.loads(cache_path.read_text()); value['value'] = 'tampered'
        save(cache_path, value)
        save(self.root / 'capability_pools/alfworld/manifest.json', {'environment': 'alfworld', 'models': []})
        bad_receipt = dict(row, request_hash='not-in-stage-ledger')
        save(folder / 'trials/000/behavior.json', {'candidate': {'read_trace': [dict(capability_calls=[dict(
            operation='read', model_id=row['model_id'], receipt=bad_receipt)])]}})
        audit = memory_model_report(Reader(self.root), folder, 'alfworld', binding)
        failed = {item['check'] for item in audit['integrity_findings']}
        self.assertIn('record_hash_matches', failed)
        self.assertIn('binding_capability_manifest_matches', failed)
        self.assertIn('trace_capability_receipt_matches_ledger', failed)
        self.assertFalse(audit['manifest']['binding_matches'])

    def test_hook_receipts_cover_readonly_native_and_initial_diagnostics_without_double_counting(self):
        row, binding = self.example()
        row['operation'] = 'event'
        folder = self.root / 'source/alfworld'
        ledger(folder / 'capability_costs.jsonl', [row])
        call = dict(operation='event', model_id=row['model_id'],
                    request_hash='worker-event-request', receipt=row)
        event = dict(status='valid', program_hash='hook-source',
                     receipt=dict(capability_calls=[call]))
        # Initial Hook diagnostics also carry a convenience copy of the calls;
        # only their event receipt should be traversed.
        save(folder / 'trials/000/behavior.json', dict(candidate=dict(
            read_trace=[dict(task_id='task', hook_trace=[event], capability_calls=[call])],
            write_trace=[])))
        ledger(folder / 'runner/native/code_memory_traces.jsonl', [dict(
            kind='generated_memory_native_episode', synthetic=False, environment='alfworld',
            task_id='task', read_calls=[], write_calls=[dict(event_trace=[event])],
            memory_hook=dict(events=[event], committed=False))])
        audit = memory_model_report(Reader(self.root), folder, 'alfworld', binding)
        self.assertEqual(audit['integrity_findings'], [])
        self.assertEqual(audit['mechanism_execution']['verified_trace_call_references'], 2)
        self.assertEqual(audit['mechanism_execution']['unique_verified_broker_receipts'], 1)
        self.assertEqual(audit['totals']['logical_requests'], 1)
        self.assertEqual({x['operation'] for x in audit['mechanism_execution']['evidence']}, {'event'})
        # The wrapper cannot hide an operation mismatch in a trusted receipt.
        event['receipt']['capability_calls'][0] = dict(call, operation='read')
        save(folder / 'trials/000/behavior.json', dict(candidate=dict(
            read_trace=[dict(hook_trace=[event])], write_trace=[])))
        broken = memory_model_report(Reader(self.root), folder, 'alfworld', binding)
        self.assertIn('trace_capability_receipt_matches_ledger',
                      {x['check'] for x in broken['integrity_findings']})

    def test_synthetic_stages_and_rows_are_excluded_but_visible(self):
        row, binding = self.example()
        folder = self.root / 'source/alfworld'
        ledger(folder / 'capability_costs.jsonl', [row, dict(row, synthetic=True)])
        mixed = memory_model_report(Reader(self.root), folder, 'alfworld', binding)
        self.assertEqual(mixed['totals']['logical_requests'], 1)
        self.assertEqual(mixed['excluded_synthetic']['logical_requests'], 1)
        integration = memory_model_report(Reader(self.root), folder, 'alfworld', dict(binding, synthetic=True))
        self.assertEqual(integration['totals']['logical_requests'], 0)
        self.assertEqual(integration['excluded_synthetic']['logical_requests'], 2)
        self.assertEqual(integration['mechanism_execution']['evidence'], [])

    def test_formal_summary_excludes_diagnostic_calls_and_native_smoke(self):
        source_row, binding = self.example()
        source = self.root / 'source/alfworld'
        ledger(source / 'capability_costs.jsonl', [source_row])
        # Generalization uses the same cache, so no new inference or token usage.
        cached = dict(source_row, phase='sealed_generalization', cache_hit=True,
                      physical_attempt=False, inference_attempted=False, physical=False, usage={})
        ledger(self.root / 'generalization/alfworld/capability_costs.jsonl', [cached])
        diagnostic = self.root / 'diagnostics/integration/source/alfworld'
        ledger(diagnostic / 'capability_costs.jsonl', [source_row] * 3)
        ledger(diagnostic / 'runner/native/costs.jsonl', [dict(kind='model', role='actor', usage={
            'prompt_tokens': 500, 'completion_tokens': 20})])
        ledger(diagnostic / 'runner/native/canonical_episodes.jsonl', [dict(synthetic=False,
            native_reward_verified=True, task_id='smoke', score=1, history=[])])
        result = build_report(self.root)
        self.assertEqual(result['memory_model_calls']['totals']['logical_requests'], 2)
        self.assertEqual(result['memory_model_calls']['totals']['completed_physical'], 1)
        self.assertEqual(result['memory_model_calls']['totals']['known_input_tokens'], 11)
        self.assertEqual(result['sources']['alfworld']['memory_model_calls']['totals']['logical_requests'], 1)
        self.assertEqual(result['generalization']['alfworld']['memory_model_calls']['totals']['cache_hits'], 1)
        self.assertEqual(result['physical_native']['total']['real_rows'], 0)
        self.assertEqual(result['model_costs']['by_role'], {})
        self.assertTrue(all(not path.startswith('diagnostics/') for path in result['memory_model_calls']['ledger_paths']))
        text = markdown(result)
        self.assertIn('记忆机制内部模型调用', text)
        self.assertIn('调用本身不证明', text)
        # Older pre-capability JSON reports still render.
        legacy = copy.deepcopy(result); legacy.pop('memory_model_calls')
        self.assertNotIn('记忆机制内部模型调用', markdown(legacy))

    def test_missing_legacy_ledger_is_unknown_evidence_not_execution(self):
        audit = memory_model_report(Reader(self.root), self.root / 'source/alfworld', 'alfworld')
        self.assertFalse(audit['available'])
        self.assertEqual(audit['totals']['logical_requests'], 0)
        self.assertEqual(audit['mechanism_execution']['verified_trace_call_references'], 0)
        self.assertIsNone(audit['manifest']['binding_matches'])
        self.assertEqual(audit['integrity_findings'], [])


if __name__ == '__main__':
    unittest.main()
