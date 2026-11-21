"""Independent synthetic regression checks for continual-memory commit boundaries.

No native environment, LLM, or GPU is used; task rewards are test fixtures.
"""
import copy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from evomem.common import digest, read, write
from phase1_codegen import campaign
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.learning import load_snapshot, snapshot_logs, snapshot_memory, make_snapshot, save_snapshot
from phase1_codegen.memory import CodeMemory
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.tests.test_engine import FIRST_SOURCE, SECOND_SOURCE, SyntheticProposer, TinyTokenizer, fixture_log, task
from phase1_codegen.tests.test_continual_engine import LearningEvaluator


class ContinualIntegrationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-continual-integration-audit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        binding = patch('phase1_codegen.engine.code_binding', return_value={'synthetic_audit_fixture': 'stable'})
        binding.start()
        self.addCleanup(binding.stop)
        self.logs = [dict(fixture_log(), environment='alfworld')]

    @staticmethod
    def task(name, family='synthetic-clean'):
        return dict(task(name), family=family, environment='alfworld', split='train')

    def engine(self, **overrides):
        options = dict(out=self.root / 'run', tokenizer=TinyTokenizer(),
                       proposer=SyntheticProposer((FIRST_SOURCE, SECOND_SOURCE)), evaluator=LearningEvaluator(),
                       evolution_memory=EvolutionMemory(),
                       context=dict(environment='alfworld', family='synthetic-clean', memory_scope='environment'),
                       logs=self.logs, exploration_tasks=[self.task('probe')],
                       confirmation_blocks=[[self.task('confirm-0')], [self.task('confirm-1')]],
                       learning_blocks=[[self.task('learn-0')], [self.task('learn-1')]],
                       synthetic=True, acceptance_policy='native_reward', max_rounds=2)
        options.update(overrides)
        return EvolutionEngine(**options)

    def test_constructor_restart_restores_saved_initial_bank_before_state_initialization(self):
        initial = self.engine()
        reference = initial.binding['initial_task_memory_snapshot']
        self.assertTrue((initial.store.root / reference['path']).is_file())
        self.assertFalse((initial.store.root / 'state.json').exists())
        # An initial generated bank already exists. Reconstructing the engine
        # must not repeat its writes/model calls merely because run() was not entered.
        with patch.object(CodeMemory, 'write', side_effect=AssertionError('initial bank was replayed')):
            restored = self.engine()
        self.assertEqual(restored.binding['initial_task_memory_snapshot'], reference)

    def test_interrupted_initial_construction_leaves_pending_and_never_silently_replays(self):
        with patch.object(CodeMemory, 'write', side_effect=KeyboardInterrupt('SyntheticInitialInterruption')):
            with self.assertRaisesRegex(KeyboardInterrupt, 'SyntheticInitialInterruption'):
                self.engine()
        folder = self.root / 'run'
        self.assertTrue((folder / 'initial_task_memory_attempt.json').exists())
        self.assertFalse((folder / 'initial_task_memory.json').exists())
        self.assertFalse((folder / 'state.json').exists())
        with patch.object(CodeMemory, 'write', side_effect=AssertionError('pending attempt was replayed')) as generated:
            with self.assertRaisesRegex(RuntimeError, 'Unresolved initial Task Memory'):
                self.engine()
        generated.assert_not_called()

    def test_changed_bootstrap_inputs_are_rejected_before_reusing_or_rebuilding_initial_bank(self):
        first = self.engine()
        anchor = first.store.root / 'initial_task_memory.json'
        before = anchor.read_bytes()
        changed = copy.deepcopy(self.logs)
        changed[0]['goal'] = 'Changed synthetic bootstrap input'
        with patch.object(CodeMemory, 'write', side_effect=AssertionError('changed bootstrap was replayed')) as generated:
            with self.assertRaisesRegex(RuntimeError, 'inputs or runtime changed'):
                self.engine(logs=changed)
        generated.assert_not_called()
        self.assertEqual(anchor.read_bytes(), before)

    def test_failure_after_one_successful_learning_episode_keeps_atomic_pointer_and_no_reexecution(self):
        class FailSecond(LearningEvaluator):
            def learn(self, *args, **kwargs):
                result = super().learn(*args, **kwargs)
                if len(self.learning_calls) == 2:
                    result['episode']['error'] = 'SyntheticLateError'
                return result
        evaluator = FailSecond()
        blocks = [[self.task('first-learning'), self.task('second-learning')], [self.task('third-learning')]]
        engine = self.engine(evaluator=evaluator, learning_blocks=blocks)
        before = engine.run(0)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete native learning'):
            engine.run(1)
        self.assertEqual(read(engine.store.root / 'state.json'), before)
        current = load_snapshot(engine.store.root, before['task_memory'])
        self.assertEqual(snapshot_logs(current), self.logs)
        with self.assertRaisesRegex(RuntimeError, 'Previously failed'):
            self.engine(evaluator=evaluator, learning_blocks=blocks).run(1)
        self.assertEqual(len(evaluator.learning_calls), 2)
        receipts = [read(path) for path in (engine.store.root / 'learning/receipts').glob('*.json')]
        self.assertEqual(sorted(row['status'] for row in receipts), ['complete', 'failed'])
        self.assertEqual(evaluator.calls, [])

    def test_cross_type_arms_inherit_actual_bank_then_diverge_without_mutating_source(self):
        source_engine = self.engine(out=self.root / 'source')
        source_state = source_engine.run(1)
        source = load_snapshot(source_engine.store.root, source_state['task_memory'])
        source_program = source_engine.store.current()
        source_state_before = digest(read(source_engine.store.root / 'state.json'))
        initial_refs, final_states = [], []
        for branch, confirmation in [('accepting', 'better'), ('rejecting', 'tie')]:
            evaluator = LearningEvaluator(confirmation=confirmation)
            target = self.engine(out=self.root / branch, evaluator=evaluator,
                                 proposer=SyntheticProposer((SECOND_SOURCE,)),
                                 logs=snapshot_logs(source), initial_source=source_program.source,
                                 initial_memory_snapshot=snapshot_memory(source),
                                 context=dict(environment='alfworld', family='new-target-family', memory_scope='environment'),
                                 exploration_tasks=[self.task('target-probe-a', 'new-target-family'),
                                                    self.task('target-probe-b', 'new-target-family')],
                                 confirmation_blocks=[[self.task('target-confirm-a', 'new-target-family'),
                                                       self.task('target-confirm-b', 'new-target-family')]],
                                 learning_blocks=[[self.task('target-learn', 'new-target-family')]], max_rounds=1)
            ref = target.binding['initial_task_memory_snapshot']
            initial_refs.append(ref)
            self.assertEqual(snapshot_memory(load_snapshot(target.store.root, ref)), snapshot_memory(source))
            final_states.append(target.run())
            final = load_snapshot(target.store.root, final_states[-1]['task_memory'])
            self.assertEqual({ep['family'] for ep in snapshot_logs(final)}, {'synthetic-clean', 'new-target-family'})
            self.assertEqual(final['program_hash'], final_states[-1]['current_program'])
            self.assertTrue(all(not ep['task_id'].startswith('target-probe') for ep in snapshot_logs(final)))
        self.assertEqual(initial_refs[0], initial_refs[1])
        self.assertNotEqual(final_states[0]['current_program'], final_states[1]['current_program'])
        self.assertEqual(digest(read(source_engine.store.root / 'state.json')), source_state_before)
        self.assertEqual(snapshot_logs(load_snapshot(source_engine.store.root, source_state['task_memory'])), snapshot_logs(source))

    def test_final_assessment_requires_both_complete_uses_frozen_logs_and_rejects_write_receipts(self):
        program = Program(BASELINE_SOURCE)
        context = dict(environment='alfworld', family='synthetic-clean', memory_scope='environment')
        memory = CodeMemory(program, self.logs, TinyTokenizer(), context)
        snapshot = make_snapshot(self.logs, 'alfworld', memory_snapshot=memory.export_snapshot())
        selected = [self.task('heldout-never-learned')]
        root = self.root / 'campaign'
        write(root / 'protocol.json', dict(task_memory_mode='continual', workflow_version=campaign.WORKFLOW_VERSION,
                                          memory_hook_version=campaign.HOOK_VERSION))
        roots = [root / 'target/alfworld' / name for name in ('no_external_em', 'cross_type_em')]
        for folder in roots:
            write(folder / 'state.json', {'task_memory': save_snapshot(folder, snapshot)})
            write(folder / 'status.json', {'status': 'complete'})
        saved = {str(folder): (folder / 'state.json').read_bytes() for folder in roots}

        class ReadonlyFixture:
            step_limit = 30

            def __init__(self):
                self.calls = []
                self.corrupt = False

            def __call__(self, current, logs, tasks, phase, *, memory_snapshot):
                self.calls.append(dict(logs=copy.deepcopy(logs), snapshot=copy.deepcopy(memory_snapshot)))
                return [dict(task_id=item['id'], synthetic=False, fixture_origin='synthetic-native-shaped-test-only',
                             native_reward_verified=True, error=None, score=0., environment_seed=item['seed'],
                             history=[], read_only_memory=not self.corrupt,
                             memory_write_calls=int(self.corrupt), generated_memory_write_calls=int(self.corrupt),
                             program_hash=current.hash, memory_hash_before=memory_snapshot['memory_hash'],
                             memory_hash_after=memory_snapshot['memory_hash']) for item in tasks]

            def close(self):
                pass

        native = ReadonlyFixture()
        with ExitStack() as stack:
            for name, replacement in {
                'validate_dataset': lambda *args: None,
                'tasks': lambda *args: selected,
                'shared_logs': lambda *args: copy.deepcopy(self.logs),
                'source_checkpoint': lambda *args: (program, {'synthetic_fixture': True}),
                'learning_blocks': lambda *args: [[self.task('allocated-learning')]],
                'target_memory_checkpoint': lambda *args: snapshot,
                'capability_pool': lambda *args: None,
            }.items():
                stack.enter_context(patch.object(campaign, name, replacement))
            stack.enter_context(patch.object(campaign.RunStore, 'current', return_value=program))
            factory = stack.enter_context(patch.object(campaign, 'evaluator', return_value=native))
            write(roots[1] / 'status.json', {'status': 'searching'})
            with self.assertRaisesRegex(RuntimeError, 'All target search'):
                campaign.generalization(root, 'alfworld', TinyTokenizer(), 'synthetic-model')
            factory.assert_not_called()
            write(roots[1] / 'status.json', {'status': 'complete'})
            campaign.generalization(root, 'alfworld', TinyTokenizer(), 'synthetic-model')
            self.assertEqual(len(native.calls), 1)  # Three exact variants, one physical fixture.
            self.assertEqual(native.calls[0]['logs'], self.logs)
            self.assertEqual(native.calls[0]['snapshot'], snapshot_memory(snapshot))
            self.assertTrue(all('heldout-never-learned' not in str(row['logs']) for row in native.calls))
            for folder in roots:
                self.assertEqual((folder / 'state.json').read_bytes(), saved[str(folder)])
                self.assertEqual(snapshot_logs(load_snapshot(folder, read(folder / 'state.json')['task_memory'])), self.logs)
            # Corrupt a cached assertion without changing the frozen bank: even a
            # hash-consistent cached episode is rejected as a readonly violation.
            cache = next((root / 'generalization/alfworld/episodes').glob('*.json'))
            receipt = read(cache)
            receipt['episode']['read_only_memory'] = False
            receipt['episode_hash'] = digest(receipt['episode'])
            write(cache, receipt)
            with self.assertRaisesRegex(RuntimeError, 'read-only bank'):
                campaign.generalization(root, 'alfworld', TinyTokenizer(), 'synthetic-model')
            self.assertEqual(len(native.calls), 1)

    def test_continual_search_rejects_old_source_baseline_import_before_any_runtime(self):
        root = self.root / 'old-baseline-disallowed'
        write(root / 'protocol.json', dict(source_baseline_origins={'alfworld': {'old': 'measurement'}}))
        with patch.object(campaign, 'validate_dataset'), \
             patch.object(campaign, 'tasks', return_value=[self.task('explore')]), \
             patch.object(campaign, 'shared_logs', return_value=self.logs), \
             patch.object(campaign, 'learning_blocks', return_value=[[self.task('learn')]]), \
             patch.object(campaign, 'capability_pool') as capabilities, \
             patch.object(campaign, 'evaluator') as evaluator:
            with self.assertRaisesRegex(ValueError, 'frozen-replay source baseline'):
                campaign.search(root, 'alfworld', 'source', 'local_only', TinyTokenizer(), 'synthetic-model')
        capabilities.assert_not_called()
        evaluator.assert_not_called()

    def test_target_plan_distinguishes_shared_checkpoint_from_run_local_snapshot_identity(self):
        program = Program(BASELINE_SOURCE)
        target_log = dict(fixture_log(), environment='alfworld', family='new-target-family',
                          task_id='target-bootstrap', group_id='group-target-bootstrap')
        cumulative = self.logs + [target_log]
        context = dict(environment='alfworld', family='synthetic-clean', memory_scope='environment')
        memory = CodeMemory(program, cumulative, TinyTokenizer(), context)
        snapshot = make_snapshot(cumulative, 'alfworld', memory_snapshot=memory.export_snapshot(),
                                 parent_hash='synthetic-source-checkpoint')
        plans, engine_inputs = [], []
        write(self.root / 'target-plan/protocol.json', dict(task_memory_mode='continual',
              workflow_version=campaign.WORKFLOW_VERSION, memory_hook_version=campaign.HOOK_VERSION))

        class FakeEngine:
            def __init__(self, *args, **kwargs):
                engine_inputs.append(dict(logs=args[6], kwargs=kwargs))
                self.store = SimpleNamespace(history=lambda state: [])

            def run(self, rounds):
                return dict(next_round=rounds, revision=0)

        with ExitStack() as stack:
            replacements = dict(validate_dataset=lambda *args: None,
                                tasks=lambda *args: [self.task('target-probe', 'new-target-family')],
                                shared_logs=lambda *args: self.logs,
                                learning_blocks=lambda *args: [[self.task('target-learning', 'new-target-family')]],
                                source_checkpoint=lambda *args: (program, {}),
                                validate_transfer_records=lambda *args: None,
                                target_memory_checkpoint=lambda *args: snapshot,
                                capability_pool=lambda *args: None,
                                evaluator=lambda *args, **kwargs: SimpleNamespace(fingerprint='synthetic-native', close=lambda: None),
                                CodeEvolver=lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
                                MechanismReviewer=lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
                                freeze_search=lambda out, plan, model: plans.append(copy.deepcopy(plan)),
                                EvolutionEngine=FakeEngine)
            for name, value in replacements.items():
                stack.enter_context(patch.object(campaign, name, value))
            campaign.search(self.root / 'target-plan', 'alfworld', 'target', 'no_external_em', TinyTokenizer(), 'synthetic-model')
        self.assertEqual(plans[0]['shared_target_checkpoint_hash'], digest(snapshot))
        self.assertEqual(plans[0]['initial_runtime_memory_hash'], snapshot['memory_hash'])
        self.assertEqual(plans[0]['initial_ordered_logs_hash'], snapshot['ordered_logs_hash'])
        self.assertNotIn('initial_task_memory_snapshot_hash', plans[0])
        self.assertEqual(plans[0]['adaptation_hash'], digest(cumulative))
        self.assertEqual(plans[0]['task_memory_families'], ['new-target-family', 'synthetic-clean'])
        self.assertEqual(engine_inputs[0]['logs'], cumulative)
        self.assertEqual(engine_inputs[0]['kwargs']['initial_memory_snapshot'], snapshot_memory(snapshot))


if __name__ == '__main__':
    unittest.main()
