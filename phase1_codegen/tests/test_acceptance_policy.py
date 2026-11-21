"""Synthetic acceptance-policy contracts, never evidence of mechanism effectiveness.

The reviewer is an independent test callable. Generated code is checked by the
real isolated worker; synthetic scores make policy boundaries deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from evomem.common import digest, read, write
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.store import RunStore
from phase1_codegen.tests.test_engine import FIRST_SOURCE, SECOND_SOURCE, TinyTokenizer, fixture_log, task
from phase1_codegen.tests.test_review_loop import EventEvaluator, EventProposer, SyntheticReviewer


# A task-ID index changes the storage representation and deduplication lookup.
# The readout intentionally remains identical on this one-episode corpus, so
# the deterministic synthetic score is independent of this representation test.
INDEXED_SOURCE = BASELINE_SOURCE.replace(
    'state.get("episodes", [])', 'list(state.get("by_id", {}).values())'
).replace(
    '    return {"episodes": rows[-8:]}',
    '    by_id = {}\n    for row in rows[-8:]:\n        by_id[row["id"]] = row\n    return {"by_id": by_id}'
)

RUNTIME_FAILURE_SOURCE = '''def on_event(state, event, context):
    return {"state": {"broken": 1 / 0}, "session": {}, "memory": None}
''' 


class PolicyEvaluator(EventEvaluator):
    fingerprint = 'synthetic-policy-evaluator-v1'

    def __init__(self, events, parent_score=0.5, candidate_score=1.0, candidate_error=False):
        super().__init__(events)
        self.parent_score = parent_score
        self.candidate_score = candidate_score
        self.candidate_error = candidate_error

    def __call__(self, program, logs, tasks, phase):
        rows = super().__call__(program, logs, tasks, phase)
        candidate = program.hash != Program(BASELINE_SOURCE).hash
        for row in rows:
            row['score'] = self.candidate_score if candidate else self.parent_score
            row['history'] = [dict(observation='synthetic visible state',
                                   action='synthetic action', next_observation='synthetic environment feedback')]
            row['done'] = True
            if candidate and self.candidate_error:
                row['error'] = 'SyntheticCandidateExecutionFailure'
        return rows


class AcceptancePolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-reviewer-policy-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.events = []
        self.logs = [fixture_log()]
        self.explore = [task('exploration-a'), task('exploration-b')]

    def components(self, source=FIRST_SOURCE, verdict='positive', **evaluator_options):
        return (EventProposer([source], self.events), PolicyEvaluator(self.events, **evaluator_options),
                SyntheticReviewer(self.events, verdict=verdict))

    def engine(self, proposer, evaluator, reviewer, out=None, policy=None, blocks=None, max_rounds=1):
        options = {} if policy is None else {'acceptance_policy': policy}
        return EvolutionEngine(
            out=self.root if out is None else out, tokenizer=TinyTokenizer(),
            proposer=proposer, evaluator=evaluator, reviewer=reviewer,
            evolution_memory=EvolutionMemory(), context={'family': 'synthetic-clean', 'environment': 'alfworld'},
            logs=self.logs, exploration_tasks=self.explore,
            confirmation_blocks=[] if blocks is None else blocks,
            initial_source=BASELINE_SOURCE, seed=11, synthetic=True,
            max_rounds=max_rounds, **options,
        )

    def trials(self, state, out=None):
        return RunStore(self.root if out is None else out).history(state)

    def test_positive_opinion_requires_positive_strictly_improved_native_score(self):
        for parent_score, candidate_score, accepted in ((0.5, 0.5, False), (0.5, 0., False),
                (0., 0., False), (-1., 0., False), (0., 1., True)):
            with self.subTest(parent=parent_score, candidate=candidate_score):
                out = self.root / ('%s-%s' % (parent_score, candidate_score))
                proposer, evaluator, reviewer = self.components(parent_score=parent_score, candidate_score=candidate_score)
                state = self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                trial = self.trials(state, out)[0]
                self.assertEqual(trial['acceptance_policy'], 'reviewer_and_reward')
                self.assertEqual(trial['accepted'], accepted)
                self.assertEqual(trial['acceptance_feedback']['eligible'], accepted)
                self.assertTrue(trial['mechanism_validated'])
                self.assertTrue(trial['exploration']['complete'])
                self.assertEqual(state['current_program'], Program(FIRST_SOURCE if accepted else BASELINE_SOURCE).hash)
                self.assertFalse(trial.get('confirmation'))
                self.assertEqual([call['payload']['stage'] for call in reviewer.calls], ['pre', 'post'])

    def test_all_disagreements_continue_with_native_trace_feedback_under_one_budget(self):
        for verdict, parent_score, candidate_score in (
                ('positive', 0.5, 0.), ('negative', 0., 1.), ('negative', 0.5, 0.)):
            with self.subTest(verdict=verdict, candidate=candidate_score):
                out = self.root / ('%s-%s' % (verdict, candidate_score))
                proposer = EventProposer([FIRST_SOURCE, SECOND_SOURCE], self.events)
                evaluator = PolicyEvaluator(self.events, parent_score=parent_score, candidate_score=candidate_score)
                reviewer = SyntheticReviewer(self.events, verdict=verdict)
                engine = self.engine(proposer, evaluator, reviewer, out=out, max_rounds=2)
                state = engine.run(99)
                trials = self.trials(state, out)
                self.assertEqual(len(trials), 2)
                self.assertEqual(len(proposer.calls), 2)
                self.assertEqual(state['revision'], 0)
                self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
                for i, trial in enumerate(trials):
                    self.assertFalse(trial['accepted'])
                    self.assertEqual(trial['exploration']['parent'], parent_score)
                    self.assertEqual(trial['exploration']['candidate'], candidate_score)
                    receipt = trial['post_review']
                    shown = read(out / receipt['displayed_path'])
                    traces = {r['id']: r['data'] for r in shown['evidence'] if r['kind'] == 'execution_trace'}
                    self.assertEqual(traces['parent:execution:0']['score'], parent_score)
                    self.assertEqual(traces['candidate:execution:0']['score'], candidate_score)
                    self.assertTrue(traces['candidate:execution:0']['steps'])
                    self.assertIn('action', traces['candidate:execution:0']['steps'][0]['fields'])
                    self.assertIn('candidate:execution:0', shown['citation_registry']['evidence'])
                feedback = proposer.calls[1]['payload']['previous_attempt']
                self.assertEqual(feedback['native_validation'], trials[0]['exploration'])
                self.assertEqual(feedback['acceptance_feedback'], trials[0]['acceptance_feedback'])
                candidate_trace = next(r['data'] for r in feedback['native_execution'] if r['id'] == 'candidate:execution:0')
                self.assertEqual(candidate_trace['score'], candidate_score)
                self.assertEqual(candidate_trace['provenance']['round'], 0)
                self.assertEqual(candidate_trace['provenance']['program_hash'], trials[0]['candidate_hash'])
                self.assertEqual(engine.run(99), state)
                self.assertEqual(len(proposer.calls), 2)

    def test_neutral_negative_and_insufficient_opinions_remain_unaccepted(self):
        for verdict in ('neutral', 'negative', 'insufficient_evidence'):
            with self.subTest(verdict=verdict):
                out = self.root / verdict
                proposer, evaluator, reviewer = self.components(verdict=verdict, candidate_score=1.0)
                state = self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                trial = self.trials(state, out)[0]
                self.assertGreater(trial['exploration']['delta'], 0)
                self.assertFalse(trial['accepted'])
                self.assertIsNone(trial['acceptance_basis'])
                self.assertEqual(trial['post_review']['report']['assessment']['verdict'], verdict)
                self.assertEqual(state['revision'], 0)
                self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
                self.assertFalse(any('confirm' in call['phase'] for call in evaluator.calls))

    def test_positive_opinion_can_accept_valid_restructured_bank_with_unchanged_readout(self):
        proposer, evaluator, reviewer = self.components(source=INDEXED_SOURCE)
        state = self.engine(proposer, evaluator, reviewer).run(rounds=1)
        trial = self.trials(state)[0]
        self.assertTrue(trial['mechanism_validated'])
        self.assertTrue(trial['accepted'])
        self.assertEqual(trial['acceptance_basis'], 'reviewer_and_reward')
        self.assertEqual(state['current_program'], Program(INDEXED_SOURCE).hash)
        self.assertTrue(trial['exploration']['complete'])
        self.assertGreater(trial['exploration']['delta'], 0)
        self.assertFalse(trial.get('confirmation'))
        self.assertTrue(any(call['program_hash'] == Program(INDEXED_SOURCE).hash for call in evaluator.calls))
        evidence = reviewer.calls[1]['payload']['evidence']
        storage = {e['id']: e['data']['state'] for e in evidence if e['kind'] == 'memory_storage'}
        self.assertIn('episodes', storage['parent:storage'])
        self.assertIn('by_id', storage['candidate:storage'])
        readouts = {e['id']: e['data'] for e in evidence if e['kind'] == 'memory_readout'}
        self.assertEqual(readouts['parent:read:0']['replayed_readout'], readouts['candidate:read:0']['replayed_readout'])
        self.assertFalse(readouts['candidate:read:0']['actual_trace_available'])

    def test_positive_opinion_cannot_override_invalid_runtime_failure_or_duplicate_source(self):
        for name, source in [('invalid', 'import os\n' + BASELINE_SOURCE),
                             ('runtime_failure', RUNTIME_FAILURE_SOURCE),
                             ('duplicate', BASELINE_SOURCE)]:
            with self.subTest(kind=name):
                out = self.root / name
                proposer, evaluator, reviewer = self.components(source=source)
                reviewer.force_invalid_positive = True  # Deliberately malformed synthetic judgment.
                state = self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                trial = self.trials(state, out)[0]
                self.assertFalse(trial['accepted'])
                self.assertFalse(trial['mechanism_validated'])
                self.assertIsNone(trial['acceptance_basis'])
                self.assertTrue(trial.get('error'))
                self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
                self.assertFalse(trial['post_review']['ok'])
                self.assertIsNone(trial['post_review']['report'])
                self.assertIn('positive', trial['post_review']['error'].lower())
                self.assertFalse(any('confirm' in call['phase'] for call in evaluator.calls))

    def test_incomplete_candidate_execution_cannot_be_accepted_by_positive_opinion(self):
        proposer, evaluator, reviewer = self.components(candidate_error=True)
        state = self.engine(proposer, evaluator, reviewer).run(rounds=1)
        trial = self.trials(state)[0]
        self.assertFalse(trial['exploration']['complete'])
        self.assertFalse(trial['accepted'])
        self.assertIsNone(trial['acceptance_basis'])
        self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
        self.assertFalse(any('confirm' in call['phase'] for call in evaluator.calls))

    def test_default_policy_requires_independent_reviewer_even_for_synthetic_workflows(self):
        proposer, evaluator, reviewer = self.components()
        with self.assertRaisesRegex(ValueError, '[Rr]eviewer'):
            self.engine(proposer, evaluator, None)
        # A fingerprint must not make one object eligible to both propose and
        # approve its own generated candidate.
        proposer.fingerprint = 'synthetic-proposer-with-valid-fingerprint'
        with self.assertRaisesRegex(ValueError, 'distinct|separate|role'):
            self.engine(proposer, evaluator, proposer)
        self.assertFalse(proposer.calls)
        self.assertFalse(evaluator.calls)
        self.assertFalse(reviewer.calls)

    def test_reviewer_policy_can_use_explicit_round_limit_without_confirmation_tasks(self):
        proposer, evaluator, reviewer = self.components()
        state = self.engine(proposer, evaluator, reviewer, max_rounds=1).run()
        self.assertEqual(state['next_round'], 1)
        self.assertEqual(len(proposer.calls), 1)
        self.assertEqual(state['binding']['max_rounds'], 1)
        self.assertEqual(state['binding']['confirmation_blocks'], [])
        self.assertFalse(any('confirm' in call['phase'] for call in evaluator.calls))

    def test_empty_confirmation_plan_requires_positive_explicit_round_budget(self):
        for max_rounds in (None, 0, -1, True):
            with self.subTest(max_rounds=max_rounds):
                proposer, evaluator, reviewer = self.components()
                with self.assertRaises((ValueError, TypeError)):
                    self.engine(proposer, evaluator, reviewer, max_rounds=max_rounds)

    def test_resuming_frozen_run_cannot_switch_acceptance_policy(self):
        proposer, evaluator, reviewer = self.components()
        blocks = [[task('heldout-secret-a'), task('heldout-secret-b')]]
        self.engine(proposer, evaluator, reviewer, blocks=blocks).run(rounds=1)
        with self.assertRaises((RuntimeError, ValueError)):
            self.engine(proposer, evaluator, reviewer, blocks=blocks, policy='native_reward').run(rounds=1)
        self.assertFalse(any('confirm' in call['phase'] for call in evaluator.calls))
        self.assertNotIn('heldout-secret', json.dumps([c['payload'] for c in reviewer.calls]))

    def test_pending_positive_trial_rejects_modified_review_receipt(self):
        proposer, evaluator, reviewer = self.components()
        with mock.patch.object(RunStore, 'commit', side_effect=InterruptedError('synthetic interruption')):
            with self.assertRaises(InterruptedError):
                self.engine(proposer, evaluator, reviewer).run(rounds=1)
        state_before = read(self.root / 'state.json')
        path = self.root / 'trials' / '000' / 'post_review.json'
        receipt = read(path)
        receipt['report']['assessment']['verdict'] = 'negative'
        write(path, receipt)
        with self.assertRaises((RuntimeError, ValueError)):
            self.engine(proposer, evaluator, reviewer).run(rounds=1)
        self.assertEqual(read(self.root / 'state.json'), state_before)

    def test_forged_positive_trial_does_not_replace_persisted_negative_review(self):
        proposer, evaluator, reviewer = self.components(verdict='negative')
        with mock.patch.object(RunStore, 'commit', side_effect=InterruptedError('synthetic interruption')):
            with self.assertRaises(InterruptedError):
                self.engine(proposer, evaluator, reviewer).run(rounds=1)
        state_before = read(self.root / 'state.json')
        trial_path = self.root / 'trials' / '000' / 'trial.json'
        trial = read(trial_path)
        trial['accepted'] = True
        trial['acceptance_basis'] = 'reviewer_and_reward'
        trial['post_review']['report']['assessment']['verdict'] = 'positive'
        trial['post_review']['report_hash'] = digest(trial['post_review']['report'])
        write(trial_path, trial)
        with self.assertRaises((RuntimeError, ValueError)):
            self.engine(proposer, evaluator, reviewer).run(rounds=1)
        self.assertEqual(read(self.root / 'state.json'), state_before)


if __name__ == '__main__':
    unittest.main()
