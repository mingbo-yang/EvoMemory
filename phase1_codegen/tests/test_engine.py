"""Synthetic workflow contracts; fake rewards are never evidence of task gains.

Generated programs still execute in the real bounded subprocess runtime. The
evaluator below is deliberately synthetic so these tests need no LLM or GPU.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from evomem.common import read, write
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.store import RunStore


class TinyTokenizer:
    """Character counting is a deterministic synthetic test fixture only."""

    def count(self, text):
        return len(text)

    def clip(self, text, count):
        return text[:max(0, count)]


def task(name):
    return dict(id=name, group_id='group-' + name, room='room-' + name,
                family='synthetic-clean', seed=42, goal='clean apple ' + name)


def fixture_log():
    return dict(task_id='adaptation-only', group_id='group-adaptation-only',
                room='room-adaptation-only', family='synthetic-clean', environment='alfworld',
                goal='clean apple', score=1.0, error=None, done=True,
                history=[dict(observation='apple on table', action='take apple',
                              next_observation='apple carried')],
                phase='synthetic-adaptation', seed=42, synthetic=True)


def custom_source(prefix):
    """Change both schema and text construction, outside the old enum space."""
    return '''def on_event(state, event, context):
    memory = None
    if event.get("type") == "task_end":
        experience = event["experience"]
        events = state.get("events", [])
        events.append({"id": experience["task_id"], "experience": PREFIX + experience["goal"]})
        state = {"events": events[-5:]}
    if event.get("type") == "task_start":
        memory = state.get("events", [])
    return {"state": state, "session": event.get("session", {}), "memory": memory}
'''.replace('PREFIX', repr(prefix))


FIRST_SOURCE = custom_source('Recorded task goal: ')
SECOND_SOURCE = custom_source('Adaptation goal summary: ')


class SyntheticProposer:
    def __init__(self, sources):
        self.sources = list(sources)
        self.calls = []

    def __call__(self, payload, seed):
        index = len(self.calls)
        self.calls.append(dict(payload=copy.deepcopy(payload), seed=seed))
        return dict(source=self.sources[index],
                    hypothesis='Synthetic workflow fixture only.',
                    risk='Fake scores do not establish real task effectiveness.')


class SyntheticEvaluator:
    fingerprint = 'synthetic-evaluator-v1-no-model-or-environment'
    synthetic = True

    def __init__(self, sources=(FIRST_SOURCE, SECOND_SOURCE), confirmation='better'):
        self.ranks = {Program(BASELINE_SOURCE).hash: 0}
        self.ranks.update({Program(source).hash: i + 1 for i, source in enumerate(sources)})
        self.confirmation = confirmation
        self.calls = []

    def __call__(self, program, logs, tasks, phase):
        self.calls.append(dict(program_hash=program.hash, tasks=copy.deepcopy(tasks), phase=phase))
        confirm = 'confirm' in phase.lower()
        rank = self.ranks.get(program.hash, 1)
        rows = []
        for index, item in enumerate(tasks):
            score = float(index < rank)
            error = None
            if confirm and self.confirmation == 'tie':
                score = 0.0
            if confirm and self.confirmation == 'worse':
                score = float(rank == 0)
            if confirm and self.confirmation == 'error' and rank:
                error = 'SyntheticNativeFailure'
            rows.append(dict(task_id=item['id'], score=score, error=error,
                             synthetic=True, native_reward_verified=False,
                             native_recorder=dict(initial={
                                 'obs': 'Public observation for ' + item['id'],
                                 'goal': item['goal'],
                             })))
        return rows


class EvolutionEngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-code-evolution-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.logs = [fixture_log()]
        self.explore = [task('exploration-a'), task('exploration-b')]
        self.blocks = [[task('heldout-secret-%s-a' % i), task('heldout-secret-%s-b' % i)]
                       for i in range(2)]

    def engine(self, proposer, evaluator, *, out=None, blocks=None):
        return EvolutionEngine(
            out=self.root if out is None else out,
            tokenizer=TinyTokenizer(), proposer=proposer, evaluator=evaluator,
            evolution_memory=EvolutionMemory(), context={'family': 'synthetic-clean', 'environment': 'alfworld'},
            logs=self.logs, exploration_tasks=self.explore,
            confirmation_blocks=self.blocks if blocks is None else blocks,
            initial_source=BASELINE_SOURCE, seed=11, synthetic=True, acceptance_policy='native_reward',
        )

    def trials(self, state, out=None):
        return RunStore(self.root if out is None else out).history(state)

    def assert_no_heldout_payload(self, proposer):
        payloads = json.dumps([row['payload'] for row in proposer.calls], sort_keys=True)
        self.assertNotIn('heldout-secret-', payloads)
        self.assertNotIn('SyntheticNativeFailure', payloads)

    def test_positive_exploration_cannot_replace_without_strict_complete_confirmation(self):
        for condition in ('tie', 'worse', 'error'):
            with self.subTest(confirmation=condition):
                directory = self.root / condition
                proposer = SyntheticProposer([FIRST_SOURCE])
                evaluator = SyntheticEvaluator(confirmation=condition)
                state = self.engine(proposer, evaluator, out=directory).run(rounds=1)
                trial = self.trials(state, directory)[0]
                self.assertGreater(trial['exploration']['delta'], 0)
                self.assertFalse(trial['accepted'])
                self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
                self.assertEqual(state['revision'], 0)
                self.assertEqual(state['next_round'], 1)
                self.assertTrue(any('confirm' in call['phase'].lower() for call in evaluator.calls))
                self.assert_no_heldout_payload(proposer)

    def test_strict_confirmation_replaces_program_and_keeps_immutable_parent(self):
        proposer = SyntheticProposer([FIRST_SOURCE])
        evaluator = SyntheticEvaluator()
        original_logs = copy.deepcopy(self.logs)
        state = self.engine(proposer, evaluator).run(rounds=1)
        trial = self.trials(state)[0]
        self.assertTrue(trial['accepted'])
        self.assertTrue(trial['confirmation']['complete'])
        self.assertGreater(trial['confirmation']['delta'], 0)
        self.assertEqual(state['current_program'], Program(FIRST_SOURCE).hash)
        self.assertEqual(state['revision'], 1)
        self.assertEqual(self.logs, original_logs)
        self.assertEqual(RunStore(self.root).program(Program(BASELINE_SOURCE).hash).source,
                         BASELINE_SOURCE)
        self.assert_no_heldout_payload(proposer)

    def test_equal_initial_readout_still_runs_complete_event_candidate(self):
        source = BASELINE_SOURCE.replace(
            'return {"episodes": rows[-8:]}',
            'return {"episodes": rows[-8:], "unused_count": len(rows)}',
        )
        self.assertNotEqual(Program(source).identity_hash, Program(BASELINE_SOURCE).identity_hash)
        proposer = SyntheticProposer([source])
        evaluator = SyntheticEvaluator(sources=[source])
        state = self.engine(proposer, evaluator).run(rounds=1)
        trial = self.trials(state)[0]
        self.assertTrue(trial['accepted'])
        self.assertEqual(state['current_program'], Program(source).hash)
        self.assertTrue(any(call['program_hash'] == Program(source).hash for call in evaluator.calls))
        self.assertTrue(trial['exploration']['complete'])
        self.assertTrue(trial['confirmation']['complete'])

    def test_two_adaptive_rounds_consume_disjoint_confirmation_blocks(self):
        proposer = SyntheticProposer([FIRST_SOURCE, SECOND_SOURCE])
        evaluator = SyntheticEvaluator()
        state = self.engine(proposer, evaluator).run(rounds=2)
        trials = self.trials(state)
        self.assertEqual(len(trials), 2)
        self.assertTrue(all(trial['accepted'] for trial in trials))
        self.assertEqual(trials[1]['parent_hash'], Program(FIRST_SOURCE).hash)
        self.assertEqual(state['current_program'], Program(SECOND_SOURCE).hash)
        self.assertEqual(state['revision'], 2)
        confirm_calls = [call for call in evaluator.calls if 'confirm' in call['phase'].lower()]
        self.assertEqual(len(confirm_calls), 4)
        used = [{row['id'] for row in call['tasks']} for call in confirm_calls]
        self.assertEqual(used[0], used[1])
        self.assertEqual(used[2], used[3])
        self.assertFalse(used[0] & used[2])
        self.assert_no_heldout_payload(proposer)

    def test_resume_completed_round_does_not_repeat_proposals_or_evaluations(self):
        proposer = SyntheticProposer([FIRST_SOURCE])
        evaluator = SyntheticEvaluator()
        before = self.engine(proposer, evaluator).run(rounds=1)
        calls_before = copy.deepcopy(evaluator.calls)
        proposal_calls_before = copy.deepcopy(proposer.calls)
        after = self.engine(proposer, evaluator).run(rounds=1)
        self.assertEqual(after, before)
        self.assertEqual(evaluator.calls, calls_before)
        self.assertEqual(proposer.calls, proposal_calls_before)
        self.assertEqual(len(self.trials(after)), 1)

    def test_invalid_generated_source_preserves_current_and_records_failure(self):
        proposer = SyntheticProposer(['import os\n' + BASELINE_SOURCE])
        evaluator = SyntheticEvaluator()
        state = self.engine(proposer, evaluator).run(rounds=1)
        trial = self.trials(state)[0]
        self.assertFalse(trial['accepted'])
        self.assertTrue(trial.get('error'))
        self.assertEqual(state['revision'], 0)
        self.assertEqual(state['next_round'], 1)
        self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
        self.assertFalse(any('confirm' in call['phase'].lower() for call in evaluator.calls))
        self.assert_no_heldout_payload(proposer)

    def test_crash_after_evidence_before_commit_resumes_without_duplicate_calls(self):
        proposer = SyntheticProposer([FIRST_SOURCE])
        evaluator = SyntheticEvaluator()
        engine = self.engine(proposer, evaluator)
        with mock.patch.object(RunStore, 'commit', side_effect=InterruptedError('synthetic interruption')):
            with self.assertRaises(InterruptedError):
                engine.run(rounds=1)
        pending = read(self.root / 'state.json')
        self.assertEqual(pending['current_program'], Program(BASELINE_SOURCE).hash)
        self.assertEqual(pending['next_round'], 0)
        calls_before = copy.deepcopy(evaluator.calls)
        proposal_calls_before = copy.deepcopy(proposer.calls)
        state = self.engine(proposer, evaluator).run(rounds=1)
        self.assertEqual(state['current_program'], Program(FIRST_SOURCE).hash)
        self.assertEqual(state['next_round'], 1)
        self.assertEqual(evaluator.calls, calls_before)
        self.assertEqual(proposer.calls, proposal_calls_before)
        self.assertEqual(len(self.trials(state)), 1)
        exported = read(self.root / 'evolution_memory.json')
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]['after_hash'], Program(FIRST_SOURCE).hash)

    def test_resume_rejects_pointer_binding_or_history_drift(self):
        proposer = SyntheticProposer([FIRST_SOURCE])
        evaluator = SyntheticEvaluator()
        original = self.engine(proposer, evaluator).run(rounds=1)
        mutations = {
            'current_without_history': ('current_program', Program(BASELINE_SOURCE).hash),
            'wrong_revision': ('revision', 99),
            'wrong_round': ('next_round', 99),
            'missing_receipt': ('history', []),
        }
        for name, (field, value) in mutations.items():
            with self.subTest(mutation=name):
                damaged = copy.deepcopy(original)
                damaged[field] = value
                write(self.root / 'state.json', damaged)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.engine(proposer, evaluator).run(rounds=1)
                write(self.root / 'state.json', original)
        damaged = copy.deepcopy(original)
        damaged['binding']['synthetic'] = False
        write(self.root / 'state.json', damaged)
        with self.assertRaises((RuntimeError, ValueError)):
            self.engine(proposer, evaluator).run(rounds=1)
        write(self.root / 'state.json', original)

    def test_pending_trial_cannot_bypass_finite_positive_paired_acceptance(self):
        proposer = SyntheticProposer([FIRST_SOURCE])
        evaluator = SyntheticEvaluator()
        with mock.patch.object(RunStore, 'commit', side_effect=InterruptedError('synthetic interruption')):
            with self.assertRaises(InterruptedError):
                self.engine(proposer, evaluator).run(rounds=1)
        state_before = read(self.root / 'state.json')
        trial_path = self.root / 'trials' / '000' / 'trial.json'
        original = read(trial_path)
        cases = ('nan_confirmation', 'negative_exploration', 'wrong_confirmation_ids', 'wrong_synthetic')
        for name in cases:
            with self.subTest(mutation=name):
                trial = copy.deepcopy(original)
                if name == 'nan_confirmation':
                    trial['confirmation']['delta'] = float('nan')
                elif name == 'negative_exploration':
                    trial['exploration']['delta'] = -0.5
                elif name == 'wrong_confirmation_ids':
                    trial['confirmation']['task_ids'] = [row['id'] for row in self.blocks[1]]
                else:
                    trial['synthetic'] = False
                write(trial_path, trial)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.engine(proposer, evaluator).run(rounds=1)
                self.assertEqual(read(self.root / 'state.json'), state_before)
        write(trial_path, original)

    def test_transferable_memory_rejects_nested_task_or_heldout_data(self):
        for forbidden in ('history', 'native_recorder', 'raw_logs', 'confirmation', 'final'):
            with self.subTest(nested_key=forbidden):
                record = dict(id='synthetic-source-event', kind='code_change_trial',
                              observed_context={'family': 'source-family', 'nested': {
                                  forbidden: [{'private_source_task': 'source-answer'}]}},
                              synthetic=True)
                with self.assertRaises(ValueError):
                    EvolutionMemory([record])

    def test_failed_current_evaluation_is_preserved_and_not_silently_retried(self):
        class FailingSyntheticEvaluator(SyntheticEvaluator):
            def __call__(self, program, logs, tasks, phase):
                self.calls.append(dict(program_hash=program.hash, tasks=copy.deepcopy(tasks), phase=phase))
                raise RuntimeError('Deliberate synthetic environment failure')

        proposer = SyntheticProposer([FIRST_SOURCE])
        evaluator = FailingSyntheticEvaluator()
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'Deliberate synthetic environment failure'):
                self.engine(proposer, evaluator).run(rounds=1)
        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(proposer.calls, [])
        state = read(self.root / 'state.json')
        self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
        self.assertEqual(state['next_round'], 0)
        self.assertFalse(list((self.root / 'measurements').glob('*.json')))
        failures = list((self.root / 'measurement_failures').glob('*.json'))
        self.assertEqual(len(failures), 1)
        self.assertIs(read(failures[0])['candidate_measurement_committed'], False)


if __name__ == '__main__':
    unittest.main()
