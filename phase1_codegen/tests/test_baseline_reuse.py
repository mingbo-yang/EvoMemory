"""CPU-only provenance fixtures, never real task results or effectiveness evidence."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evomem.common import append, digest, events, file_hash, read, write
from phase1_codegen.baseline_reuse import PHASE, RUNTIME_FILES, seed_source_baseline
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.program import BASELINE_SOURCE, Program


class NeverExecute:
    fingerprint = 'cpu-only-fabricated-native-provenance'
    step_limit = 80

    def __call__(self, *args):
        raise AssertionError('Reuse must not execute an environment or model')


class BaselineReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cpu-fixture-baseline-reuse-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.old = self.root / 'old/source/textworld'
        self.new = self.root / 'new/source/textworld'
        self.program = Program(BASELINE_SOURCE)
        self.context = dict(environment='textworld', family='cooking', memory_scope='environment')
        self.tasks = [dict(id='source-task-' + str(i), group_id='group-' + str(i),
                           family='cooking', seed=42, official_split='train') for i in range(2)]
        self.logs = [dict(task_id='adaptation', group_id='adaptation-group', environment='textworld',
                          family='cooking', history=[], goal='CPU fixture', score=1.0,
                          synthetic=False, native_reward_verified=True, error=None)]
        code = {name: 'a' * 64 for name in RUNTIME_FILES}
        code.update({'evomem/common.py': 'b' * 64, 'phase1_codegen/review.py': 'c' * 64})
        with patch('phase1_codegen.engine.code_binding', return_value=code):
            self.engine = EvolutionEngine(
                self.new, None, SimpleNamespace(fingerprint='new-proposer'), NeverExecute(),
                EvolutionMemory(), self.context, self.logs, self.tasks, [],
                reviewer=SimpleNamespace(fingerprint='new-reviewer'), max_rounds=3,
                capabilities=SimpleNamespace(environment='textworld', fingerprint='model-capability-fixture'))
        binding = copy.deepcopy(self.engine.binding)
        binding.update(proposer='old-proposer', reviewer='old-reviewer')
        binding['code']['phase1_codegen/review.py'] = 'd' * 64
        self.state = dict(binding=binding, binding_hash=digest(binding), initial_program=self.program.hash,
                          current_program=self.program.hash, next_round=0, revision=0, history=[])
        write(self.old / 'state.json', self.state)
        p = self.old / 'programs' / (self.program.hash + '.py')
        p.parent.mkdir(parents=True)
        p.write_text(BASELINE_SOURCE)
        self.episodes = []
        for index, task in enumerate(self.tasks):
            self.episodes.append(dict(task_id=task['id'], group_id=task['group_id'], family='cooking',
                environment='textworld', phase=PHASE, score=float(index), error=None, synthetic=False,
                native_reward_verified=True, executor='ALMA.Agent_Workflow.run_single_task',
                environment_seed=task['seed'], seed=42000 + int(digest(task['id'])[:6], 16) % 10000000,
                native_recorder=dict(reward=float(index)), history=[dict(action='look')],
                program_hash=self.program.hash, program_identity_hash=self.program.identity_hash,
                memory_hash_before='f' * 64, memory_hash_after='f' * 64,
                read_only_memory=True, memory_read_calls=1, memory_write_calls=0,
                generated_memory_write_calls=0))
        self.measurement = dict(key=digest(dict(binding=binding, program=self.program.hash, tasks=self.tasks)),
            episodes=self.episodes, episodes_hash=digest(self.episodes), phase=PHASE,
            synthetic=False, seconds=12.5)
        self.origin = dict(schema=1, source_run=str(self.old), measurement_key=self.measurement['key'])
        self._write_measurement()
        for task in self.tasks:
            append(self.old / 'runner/native/costs.jsonl', dict(kind='model', phase=PHASE,
                role='actor', task_id=task['id'], ok=True, prompt_tokens=100, completion_tokens=4, elapsed=1.2))
            append(self.old / 'runner/native/costs.jsonl', dict(kind='env_step', phase=PHASE,
                task_id=task['id'], step=0))
        # Same old ledger can contain later diagnostics: never charge them to the baseline.
        append(self.old / 'runner/native/costs.jsonl', dict(kind='model', phase='round-000/exploration/candidate',
            role='actor', task_id=self.tasks[0]['id'], ok=True, prompt_tokens=999, completion_tokens=99, elapsed=9))
        self._pins()

    def _write_measurement(self):
        self.measurement_path = self.old / 'measurements' / (self.origin['measurement_key'] + '.json')
        write(self.measurement_path, self.measurement)

    def _pins(self):
        self.origin.update(state_sha256=file_hash(self.old / 'state.json'),
                           measurement_sha256=file_hash(self.measurement_path),
                           native_costs_sha256=file_hash(self.old / 'runner/native/costs.jsonl'))

    def assert_refused(self):
        with self.assertRaises((RuntimeError, ValueError, KeyError)):
            seed_source_baseline(self.engine, self.origin)
        self.assertFalse((self.new / 'baseline_reuse.json').exists())
        self.assertFalse(list((self.new / 'measurements').glob('*.json')))

    def test_import_changes_cache_binding_preserves_raw_episodes_and_cost_provenance(self):
        before = {p.relative_to(self.old): file_hash(p) for p in self.old.rglob('*') if p.is_file()}
        receipt = seed_source_baseline(self.engine, self.origin)
        self.assertNotEqual(receipt['measurement_key'], self.origin['measurement_key'])
        cache = read(self.new / 'measurements' / (receipt['measurement_key'] + '.json'))
        self.assertEqual(cache['episodes'], self.episodes)
        self.assertEqual(cache['episodes_hash'], digest(self.episodes))
        self.assertEqual(cache['seconds'], 0)
        self.assertFalse(cache['newly_executed'])
        costs = receipt['reuse_provenance']['historical_costs']
        self.assertEqual((costs['actor_model_requests'], costs['env_steps']), (2, 2))
        self.assertEqual((costs['prompt_tokens'], costs['completion_tokens']), (200, 8))
        self.assertEqual(costs['measurement_seconds'], 12.5)
        self.assertEqual(seed_source_baseline(self.engine, self.origin), receipt)
        self.assertEqual(len(events(self.new / 'measurement_access.jsonl')), 1)
        self.assertEqual(self.engine._measure(self.program, self.tasks, PHASE), self.episodes)
        accesses = events(self.new / 'measurement_access.jsonl')
        self.assertEqual(accesses[0]['operation'], 'import')
        self.assertTrue(all(row['physical'] is False for row in accesses))
        self.engine.store.initialize(self.program, self.engine.binding)
        (self.new / 'trials').mkdir()
        append(self.new / 'runner/native/costs.jsonl', {'kind': 'model', 'phase': 'later-candidate'})
        self.assertEqual(seed_source_baseline(self.engine, self.origin), receipt)
        self.assertEqual(events(self.new / 'measurement_access.jsonl'), accesses)
        after = {p.relative_to(self.old): file_hash(p) for p in self.old.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_execution_binding_drift_is_rejected(self):
        fields = ('evaluator', 'logs_hash', 'ordered_logs_hash', 'ordered_context_hash',
                  'environment_step_limit', 'storage_tokens', 'read_tokens', 'top_k',
                  'model_capabilities', 'seed', 'exploration_tasks', 'context')
        for field in fields:
            with self.subTest(field=field):
                original = self.engine.binding[field]
                self.engine.binding[field] = 'changed'
                self.assert_refused()
                self.engine.binding[field] = original

    def test_runtime_code_drift_is_rejected_even_with_unchanged_evaluator_fingerprint(self):
        for name in ('phase1_codegen/memory.py', 'phase1_codegen/capabilities.py', 'evomem/common.py',
                     'phase1_codegen/protocol.py', 'phase1_codegen/hooks.py', 'phase1_codegen/hook_adapter.py',
                     'phase1_codegen/isolation.py'):
            with self.subTest(name=name):
                old = self.engine.binding['code'][name]
                self.engine.binding['code'][name] = 'e' * 64
                self.assert_refused()
                self.engine.binding['code'][name] = old

    def test_origin_artifact_hash_drift_is_rejected(self):
        for key in ('state_sha256', 'measurement_sha256', 'native_costs_sha256', 'measurement_key'):
            with self.subTest(key=key):
                original = self.origin[key]
                self.origin[key] = '0' * 64
                self.assert_refused()
                self.origin[key] = original

    def test_internal_state_or_episode_hash_drift_is_rejected_even_if_file_re_pinned(self):
        self.state['binding_hash'] = '0' * 64
        write(self.old / 'state.json', self.state)
        self._pins()
        self.assert_refused()
        self.state['binding_hash'] = digest(self.state['binding'])
        write(self.old / 'state.json', self.state)
        self.measurement['episodes_hash'] = '0' * 64
        self._write_measurement(); self._pins()
        self.assert_refused()

    def test_invalid_synthetic_candidate_heldout_or_writing_episodes_are_rejected(self):
        changes = ({'synthetic': True}, {'error': 'runtime failure'}, {'phase': 'round-000/exploration/candidate'},
                   {'data_role': 'generalization'}, {'program_hash': '0' * 64}, {'read_only_memory': False},
                   {'memory_hash_after': '0' * 64}, {'memory_write_calls': 1}, {'generated_memory_write_calls': 1},
                   {'environment_seed': 9}, {'score': float('nan')})
        pristine = copy.deepcopy(self.episodes[0])
        for change in changes:
            with self.subTest(change=change):
                self.episodes[0] = dict(pristine, **change)
                self.measurement['episodes'] = self.episodes
                self.measurement['episodes_hash'] = digest(self.episodes)
                self._write_measurement(); self._pins()
                self.assert_refused()
        self.episodes[0] = pristine

    def test_only_fresh_source_and_exact_baseline_can_import(self):
        for directory in (self.root / 'new/target/textworld/no_external_em', self.root / 'new/source/alfworld'):
            original = self.engine.store.root
            self.engine.store.root = directory
            self.assert_refused()
            self.engine.store.root = original
        self.engine.initial = Program(BASELINE_SOURCE + '\n')
        self.assert_refused()
        self.engine.initial = self.program
        self.engine.store.initialize(self.program, self.engine.binding)
        self.assert_refused()

    def test_old_source_changed_or_chained_import_is_rejected(self):
        source = self.old / 'programs' / (self.program.hash + '.py')
        source.write_text(BASELINE_SOURCE + '\n')
        self.assert_refused()
        source.write_text(BASELINE_SOURCE)
        self.measurement['reuse_provenance'] = {}
        self._write_measurement(); self._pins()
        self.assert_refused()

    def test_cost_ledger_cannot_lose_steps_or_include_failed_model_rows(self):
        path = self.old / 'runner/native/costs.jsonl'
        pristine = events(path)
        variants = ([row for row in pristine if row['kind'] != 'env_step'],
                    [dict(row, ok=False) if row['kind'] == 'model' else row for row in pristine])
        for rows in variants:
            path.unlink()
            for row in rows:
                append(path, row)
            self._pins()
            self.assert_refused()

    def test_existing_import_tampering_is_not_overwritten(self):
        receipt = seed_source_baseline(self.engine, self.origin)
        path = self.new / 'measurements' / (receipt['measurement_key'] + '.json')
        damaged = read(path)
        damaged['episodes'][0]['score'] = 0.5
        write(path, damaged)
        with self.assertRaises(RuntimeError):
            seed_source_baseline(self.engine, self.origin)
        self.assertEqual(read(path), damaged)


if __name__ == '__main__':
    unittest.main()
