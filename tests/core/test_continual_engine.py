"""Synthetic learning/environment contracts with real isolated memory execution.

These tests establish persistence and control flow only, never task improvement.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evomem.common import digest, read, write
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.learning import load_snapshot, snapshot_logs, snapshot_memory
from phase1_codegen.memory import CodeMemory
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.store import RunStore
from tests.core.test_engine import FIRST_SOURCE, SECOND_SOURCE, SyntheticEvaluator, SyntheticProposer, TinyTokenizer, fixture_log, task


class LearningEvaluator(SyntheticEvaluator):
    fingerprint = 'synthetic-continuous-evaluator-no-environment'

    def __init__(self, *args, fail_learning=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_calls, self.memory_inputs = [], []
        self.fail_learning = fail_learning

    def __call__(self, program, logs, tasks, phase, *, memory_snapshot=None):
        self.memory_inputs.append(dict(program=program.hash, logs=copy.deepcopy(logs),
                                       snapshot=copy.deepcopy(memory_snapshot), phase=phase))
        return super().__call__(program, logs, tasks, phase)

    def learn(self, program, logs, item, phase, *, memory_snapshot):
        self.learning_calls.append((program.hash, item['id']))
        memory = CodeMemory.from_snapshot(program, memory_snapshot, TinyTokenizer())
        before = memory.fingerprint
        episode = dict(fixture_log(), task_id=item['id'], group_id=item['group_id'],
                       family=item['family'], environment=item['environment'], goal=item['goal'],
                       phase=phase, score=0., seed=42, program_hash=program.hash)
        memory.write(episode)
        episode.update(memory_hash_before=before, memory_hash_after=memory.fingerprint,
                       memory_write_calls=1, generated_memory_write_calls=1, read_only_memory=False)
        if self.fail_learning:
            episode['error'] = 'SyntheticErrorDetectedAfterGeneralUpdate'
        memory.source_log_hash = digest(logs + [episode])
        return dict(episode=episode, memory_snapshot=memory.export_snapshot())


class ContinualEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-continual-engine-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = [dict(fixture_log(), environment='alfworld')]
        self.explore = [self.task('probe-a'), self.task('probe-b')]
        self.blocks = [[self.task('confirmation-%d-a' % i), self.task('confirmation-%d-b' % i)] for i in range(2)]
        self.learning = [[self.task('learn-one')], [self.task('learn-two')]]
        # Parallel development edits are not part of this synthetic runtime.
        self.binding_patch = patch('phase1_codegen.engine.code_binding', return_value={'synthetic-runtime': 'test'})
        self.binding_patch.start(); self.addCleanup(self.binding_patch.stop)

    @staticmethod
    def task(name, family='synthetic-clean', environment='alfworld'):
        return dict(task(name), family=family, environment=environment, split='train')

    def engine(self, evaluator=None, sources=(FIRST_SOURCE, SECOND_SOURCE), root=None, **overrides):
        options = dict(out=root or self.root, tokenizer=TinyTokenizer(), proposer=SyntheticProposer(sources),
                       evaluator=evaluator or LearningEvaluator(), evolution_memory=EvolutionMemory(),
                       context=dict(environment='alfworld', family='synthetic-clean', memory_scope='environment'),
                       logs=self.logs, exploration_tasks=self.explore, confirmation_blocks=self.blocks,
                       seed=11, synthetic=True, acceptance_policy='native_reward', max_rounds=2,
                       learning_blocks=self.learning)
        options.update(overrides)
        return EvolutionEngine(**options)

    def test_learning_is_cumulative_and_only_deployed_policy_enters_memory(self):
        evaluator = LearningEvaluator()
        engine = self.engine(evaluator)
        state = engine.run()
        self.assertEqual(state['revision'], 2)
        self.assertEqual(evaluator.learning_calls,
                         [(Program(BASELINE_SOURCE).hash, 'learn-one'), (Program(FIRST_SOURCE).hash, 'learn-two')])
        snapshot = load_snapshot(self.root, state['task_memory'])
        self.assertEqual([ep['task_id'] for ep in snapshot_logs(snapshot)],
                         ['adaptation-only', 'learn-one', 'learn-two'])
        self.assertEqual(snapshot['program_hash'], Program(SECOND_SOURCE).hash)
        for index, trial in enumerate(engine.store.history(state)):
            frozen = load_snapshot(self.root, trial['task_memory_evaluation'])
            self.assertEqual(len(snapshot_logs(frozen)), 2 + index)
            self.assertNotIn('probe-a', frozen['logs_wire'])
            self.assertNotIn('confirmation-', frozen['logs_wire'])
        self.assertTrue(all(row['snapshot'] is not None for row in evaluator.memory_inputs))
        self.assertEqual(RunStore(self.root).current().hash, state['current_program'])

    def test_three_candidate_iterations_share_one_frontier_and_preserve_rejections(self):
        from tests.core.test_previous_attempt_feedback import BAD_SYNTAX
        from tests.core.test_review_loop import SyntheticReviewer
        proposer = SyntheticProposer([BAD_SYNTAX, FIRST_SOURCE, SECOND_SOURCE])
        evaluator = LearningEvaluator()
        reviewer = SyntheticReviewer([], verdict='negative')
        engine = self.engine(evaluator, proposer=proposer, reviewer=reviewer,
            max_rounds=3, learning_blocks=[self.learning[0], [], []],
            confirmation_blocks=[], acceptance_policy='reviewer_and_reward')
        state = engine.run(3)
        trials = engine.store.history(state)
        self.assertEqual(len(proposer.calls), 3)
        self.assertEqual(len(evaluator.learning_calls), 1)
        self.assertEqual(state['revision'], 0)
        self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
        self.assertEqual([t['mechanism_validated'] for t in trials], [False, True, True])
        self.assertEqual(len({t['task_memory_evaluation']['hash'] for t in trials}), 1)
        self.assertEqual([c['payload']['stage'] for c in reviewer.calls], ['pre', 'post'] * 3)
        for i in (1, 2):
            prior = proposer.calls[i]['payload']['previous_attempt']
            self.assertEqual(prior['source'], [BAD_SYNTAX, FIRST_SOURCE][i - 1])
            self.assertFalse(prior['accepted'])
            self.assertEqual(proposer.calls[i]['payload']['current_source'], BASELINE_SOURCE)
            self.assertEqual(proposer.calls[i]['payload']['mechanism_review']['report']['assessment']['subject'], 'current')
        for call in reviewer.calls:
            self.assertNotIn('evolution_memory', call['payload'])
            self.assertNotIn('previous_post_review', call['payload'])
        before = (len(proposer.calls), len(reviewer.calls), len(evaluator.calls))
        self.assertEqual(engine.run(99), state)
        self.assertEqual((len(proposer.calls), len(reviewer.calls), len(evaluator.calls)), before)

    def test_resume_does_not_repeat_learning_or_rebuild_committed_bank(self):
        first_evaluator = LearningEvaluator()
        first = self.engine(first_evaluator).run(1)
        second_evaluator = LearningEvaluator()
        resumed = self.engine(second_evaluator, sources=(SECOND_SOURCE,)).run(2)
        self.assertEqual(second_evaluator.learning_calls, [(first['current_program'], 'learn-two')])
        final = load_snapshot(self.root, resumed['task_memory'])
        self.assertEqual(len(snapshot_logs(final)), 3)
        no_calls = LearningEvaluator()
        self.assertEqual(self.engine(no_calls, sources=()).run(2), resumed)
        self.assertEqual(no_calls.learning_calls, [])
        self.assertEqual(no_calls.calls, [])

    def test_snapshot_change_invalidates_same_program_evaluation_cache(self):
        evaluator = LearningEvaluator(confirmation='tie')
        state = self.engine(evaluator).run()
        self.assertEqual(state['revision'], 0)
        baseline = Program(BASELINE_SOURCE).hash
        parent_calls = [r for r in evaluator.memory_inputs if r['program'] == baseline and 'exploration/parent' in r['phase']]
        self.assertEqual(len(parent_calls), 2)
        self.assertNotEqual(parent_calls[0]['snapshot']['memory_hash'], parent_calls[1]['snapshot']['memory_hash'])

    def test_post_write_native_error_does_not_commit_and_is_not_silently_retried(self):
        evaluator = LearningEvaluator(fail_learning=True)
        engine = self.engine(evaluator)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete native learning'):
            engine.run(1)
        state = read(self.root / 'state.json')
        self.assertEqual(state['next_round'], 0)
        self.assertEqual(len(snapshot_logs(load_snapshot(self.root, state['task_memory']))), 1)
        with self.assertRaisesRegex(RuntimeError, 'Previously failed'):
            self.engine(evaluator).run(1)
        self.assertEqual(len(evaluator.learning_calls), 1)
        receipts = list((self.root / 'learning/receipts').glob('*.json'))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(read(receipts[0])['status'], 'failed')

    def test_interrupted_trial_reuses_completed_learning_and_actual_candidate_bank(self):
        evaluator = LearningEvaluator()
        engine = self.engine(evaluator)
        original = engine.store.commit
        with patch.object(engine.store, 'commit', side_effect=RuntimeError('SyntheticCrashBeforeCommit')):
            with self.assertRaisesRegex(RuntimeError, 'SyntheticCrash'):
                engine.run(1)
        count = len(evaluator.calls)
        engine.store.commit = original
        state = engine.run(1)
        self.assertEqual(state['next_round'], 1)
        self.assertEqual(len(evaluator.learning_calls), 1)
        self.assertEqual(len(evaluator.calls), count)

    def test_learning_task_overlap_and_foreign_environment_are_rejected(self):
        for blocks in ([[self.explore[0]], self.learning[1]],
                       [[self.task('foreign', environment='textworld')], self.learning[1]],
                       [[dict(self.task('sealed'), split='test')], self.learning[1]]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                self.engine(learning_blocks=blocks)

    def test_inherited_actual_state_survives_cross_type_initialization(self):
        source = self.engine().run(1)
        snapshot = load_snapshot(self.root, source['task_memory'])
        program = RunStore(self.root).current()
        target = self.root / 'target'
        ctx = dict(environment='alfworld', family='other-family', memory_scope='environment')
        target_engine = self.engine(root=target, logs=snapshot_logs(snapshot), initial_source=program.source,
                                   initial_memory_snapshot=snapshot_memory(snapshot), context=ctx,
                                   exploration_tasks=[self.task('target-probe', 'other-family')],
                                   confirmation_blocks=[[self.task('target-confirm', 'other-family')]], acceptance_policy='native_reward',
                                   max_rounds=1, learning_blocks=[[self.task('target-learn', 'other-family')]])
        inherited = load_snapshot(target, target_engine.binding['initial_task_memory_snapshot'])
        self.assertEqual(snapshot_memory(inherited), snapshot_memory(snapshot))
        self.assertEqual(snapshot_logs(inherited), snapshot_logs(snapshot))
        self.assertEqual(target_engine.evaluator.calls, [])

    def test_committed_pointer_tampering_is_detected(self):
        state = self.engine().run(1)
        state['task_memory'] = state['binding']['initial_task_memory_snapshot']
        write(self.root / 'state.json', state)
        with self.assertRaisesRegex(RuntimeError, 'Task Memory pointer'):
            RunStore(self.root).current()

    def test_search_measurement_cache_cannot_substitute_another_program_or_bank(self):
        engine = self.engine()
        state = engine.run(0)
        engine._activate_snapshot(state['task_memory'])
        bank = snapshot_memory(engine.active_snapshot)
        program = Program(BASELINE_SOURCE)
        calls = []
        def native_shaped_fixture(program, logs, tasks, phase, **options):
            calls.append(options)
            return [dict(task_id=t['id'], program_hash=program.hash, error=None,
                         score=0., synthetic=False, native_reward_verified=True, native_recorder={'reward': 0.},
                         executor='ALMA.Agent_Workflow.run_single_task', environment_seed=t['seed'],
                         seed=42000 + int(digest(t['id'])[:6], 16) % 10000000,
                         memory_hash_before=bank['memory_hash'], memory_hash_after=bank['memory_hash'],
                         memory_write_calls=0, generated_memory_write_calls=0, read_only_memory=True,
                         fixture_origin='synthetic-contract-test-no-native-execution') for t in tasks]
        engine.evaluator = native_shaped_fixture
        engine.synthetic = False  # Exercise only the formal receipt checker with explicit mock data.
        expected = engine._measure(program, self.explore, 'synthetic-check')
        self.assertEqual(calls[0]['memory_snapshot'], bank)
        self.assertEqual(engine._measure(program, self.explore, 'cache-hit'), expected)
        self.assertEqual(len(calls), 1)
        path = next((self.root / 'measurements').glob('*.json'))
        original = read(path)
        for field, value in [('program_hash', 'another-program'),
                             ('memory_hash_before', 'another-bank'), ('read_only_memory', False)]:
            receipt = copy.deepcopy(original)
            receipt['episodes'][0][field] = value
            receipt['episodes_hash'] = digest(receipt['episodes'])
            write(path, receipt)
            with self.assertRaisesRegex(RuntimeError, 'fixed read-only program and bank'):
                engine._measure(program, self.explore, 'tampered-cache')
            # Start the next independent tamper case with no prior failure receipt.
            (self.root / 'measurement_failures' / path.name).unlink()
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
