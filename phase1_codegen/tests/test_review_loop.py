"""Synthetic single-Reviewer contracts; scores here never establish task gains.

Generated memory programs still execute in the real restricted worker. No model,
GPU, network, or native ALFWorld episode is used by this test module.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from evomem.common import read
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.store import RunStore
from phase1_codegen.tests.test_engine import (
    FIRST_SOURCE, SECOND_SOURCE, SyntheticEvaluator, SyntheticProposer,
    TinyTokenizer, fixture_log, task,
)


class SyntheticReviewer:
    """One callable for both stages, with deterministic evidence-only reports."""
    fingerprint = 'synthetic-single-reviewer-v2-bound-evidence-no-model'

    def __init__(self, events, fail_stage=None, invalid=None, verdict='positive', force_invalid_positive=False):
        self.events = events
        self.calls = []
        self.fail_stage = fail_stage
        self.invalid = invalid
        self.verdict = verdict
        self.force_invalid_positive = force_invalid_positive

    def __call__(self, payload, seed):
        self.last_payload = None
        self.calls.append(dict(payload=copy.deepcopy(payload), seed=seed))
        stage = payload['stage']
        self.events.append('review:' + stage)
        if stage == self.fail_stage:
            raise RuntimeError('SyntheticReviewerFailure')
        from phase1_codegen.review import bounded_review_payload, validate_report, _candidate_invalid
        from phase1_codegen.tests.test_review_contract import synthetic_raw_report, observation
        class SyntheticReviewTokenizer:
            # Synthetic packing only. A character-per-token model exaggerates
            # JSON overhead; strict 22k boundary cases live in test_review.py.
            def count(self, text): return (len(text) + 3) // 4
            def clip(self, text, count): return text[:max(0, count) * 4]
            def chat_count(self, messages):
                return sum(self.count(message['content']) + 16 for message in messages) + 16
        self.last_payload = bounded_review_payload(dict(payload, review_contract=2), SyntheticReviewTokenizer())
        verdict = self.verdict
        if _candidate_invalid(self.last_payload) and not self.force_invalid_positive:
            verdict = 'negative'
        raw = synthetic_raw_report(self.last_payload, verdict)
        if _candidate_invalid(self.last_payload):
            raw['observations'] = [observation(self.last_payload, 'current:facts',
                                              '/validation/mechanism_validated', 'o1')]
            raw['evaluations'][0]['observation_ids'] = ['o1']
        if self.invalid == 'approval':
            raw['approve'] = True
        if self.invalid == 'unsupported_evidence':
            raw['observations'][0]['evidence_id'] = 'missing-evidence-id'
        if self.invalid == 'bad_interpretation':
            raw['evaluations'][0]['polarity'] = 'proven_cause'
        return validate_report(raw, self.last_payload)


class EventProposer(SyntheticProposer):
    def __init__(self, sources, events):
        super().__init__(sources)
        self.events = events

    def __call__(self, payload, seed):
        self.events.append('propose')
        return super().__call__(payload, seed)


class EventEvaluator(SyntheticEvaluator):
    def __init__(self, events, sources=(FIRST_SOURCE, SECOND_SOURCE), confirmation='better'):
        super().__init__(sources, confirmation)
        self.events = events

    def __call__(self, program, logs, tasks, phase):
        self.events.append('evaluate:' + phase)
        return super().__call__(program, logs, tasks, phase)


class ReviewLoopTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-code-review-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.events = []
        self.logs = [fixture_log()]
        self.explore = [task('exploration-a'), task('exploration-b')]
        self.blocks = [[task('heldout-secret-%s-a' % i), task('heldout-secret-%s-b' % i)]
                       for i in range(2)]

    def engine(self, proposer, evaluator, reviewer, out=None):
        return EvolutionEngine(
            out=self.root if out is None else out, tokenizer=TinyTokenizer(),
            proposer=proposer, evaluator=evaluator, reviewer=reviewer,
            evolution_memory=EvolutionMemory(), context={'family': 'synthetic-clean', 'environment': 'alfworld'},
            logs=self.logs, exploration_tasks=self.explore, confirmation_blocks=self.blocks,
            initial_source=BASELINE_SOURCE, seed=11, synthetic=True, acceptance_policy='native_reward',
        )

    def components(self, sources=(FIRST_SOURCE,), **reviewer_options):
        return (EventProposer(sources, self.events), EventEvaluator(self.events, sources),
                SyntheticReviewer(self.events, **reviewer_options))

    def trials(self, state, out=None):
        return RunStore(self.root if out is None else out).history(state)

    def assert_no_confirmation_leak(self, proposer, reviewer):
        text = json.dumps(dict(proposer=[c['payload'] for c in proposer.calls],
                               reviewer=[c['payload'] for c in reviewer.calls]), sort_keys=True)
        self.assertNotIn('heldout-secret-', text)
        self.assertNotIn('SyntheticNativeFailure', text)

    def test_single_reviewer_precedes_proposal_and_reviews_changes_before_confirmation(self):
        proposer, evaluator, reviewer = self.components()
        state = self.engine(proposer, evaluator, reviewer).run(rounds=1)
        trial = self.trials(state)[0]
        self.assertTrue(trial['accepted'])
        self.assertEqual([c['payload']['stage'] for c in reviewer.calls], ['pre', 'post'])
        self.assertLess(self.events.index('review:pre'), self.events.index('propose'))
        self.assertLess(self.events.index('propose'), self.events.index('review:post'))
        candidate_exploration = [i for i, e in enumerate(self.events)
                                 if '/exploration/candidate' in e]
        confirmation = [i for i, e in enumerate(self.events) if '/confirmation/' in e]
        self.assertTrue(candidate_exploration)
        self.assertTrue(confirmation)
        self.assertLess(candidate_exploration[0], self.events.index('review:post'))
        self.assertLess(self.events.index('review:post'), min(confirmation))
        self.assertIn('pre_review', trial)
        self.assertIn('post_review', trial)
        self.assertIn('evidence_contract', proposer.calls[0]['payload']['mechanism_review']['report'])
        post = json.dumps(reviewer.calls[1]['payload'], sort_keys=True)
        self.assertIn(Program(FIRST_SOURCE).hash, post)
        self.assertIn('Recorded task goal:', post)
        self.assert_no_confirmation_leak(proposer, reviewer)

    def test_diagnostics_include_stored_state_and_exact_actor_readout(self):
        proposer, evaluator, reviewer = self.components()
        self.engine(proposer, evaluator, reviewer).run(rounds=1)
        evidence = reviewer.calls[0]['payload']['evidence']
        self.assertIsInstance(evidence, list)
        self.assertTrue(evidence)
        ids = [row['id'] for row in evidence]
        self.assertEqual(len(ids), len(set(ids)))
        for row in evidence:
            self.assertIn('kind', row)
            self.assertIn('data', row)
        # Native experience text must appear in storage/readout evidence, not just
        # abstract success rates or the source code string.
        diagnostic = [row for row in evidence if any(word in row['kind'].lower()
                      for word in ('memory', 'storage', 'retrieval', 'readout'))]
        self.assertTrue(diagnostic)
        payload = json.dumps(diagnostic, ensure_ascii=False)
        self.assertIn('adaptation-only', payload)
        self.assertIn('take apple', payload)
        self.assertIn('apple carried', payload)
        self.assertIn('entries', payload)
        self.assert_no_confirmation_leak(proposer, reviewer)

    def test_two_rounds_reuse_previous_post_as_next_pre_with_three_total_calls(self):
        proposer, evaluator, reviewer = self.components((FIRST_SOURCE, SECOND_SOURCE))
        state = self.engine(proposer, evaluator, reviewer).run(rounds=2)
        trials = self.trials(state)
        self.assertEqual(len(trials), 2)
        self.assertEqual([c['payload']['stage'] for c in reviewer.calls], ['pre', 'post', 'post'])
        self.assertEqual(state['revision'], 2)
        second_input = proposer.calls[1]['payload']['mechanism_review']
        self.assertEqual(second_input['report'], trials[0]['post_review']['report'])
        self.assertIn(Program(FIRST_SOURCE).hash, json.dumps(second_input))
        self.assertEqual(trials[1]['pre_review']['report'], trials[0]['post_review']['report'])
        self.assertNotIn('previous_post_review', reviewer.calls[2]['payload'])
        self.assertNotIn('evolution_memory', reviewer.calls[2]['payload'])
        self.assertEqual(proposer.calls[1]['payload']['previous_attempt']['source'], FIRST_SOURCE)
        self.assert_no_confirmation_leak(proposer, reviewer)

    def test_complete_candidate_and_invalid_code_both_get_post_review(self):
        no_effect_source = BASELINE_SOURCE.replace(
            'return {"episodes": rows[-8:]}',
            'return {"episodes": rows[-8:], "unused_count": len(rows)}')
        for name, source in [('no_effect', no_effect_source), ('invalid', 'import os\n' + BASELINE_SOURCE)]:
            with self.subTest(kind=name):
                out = self.root / name
                proposer = EventProposer((source,), self.events)
                reviewer = SyntheticReviewer(self.events)
                # Invalid source cannot be compiled even by the synthetic scorer.
                evaluator = EventEvaluator(self.events, (source,)) if name != 'invalid' else EventEvaluator(self.events)
                state = self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                trial = self.trials(state, out)[0]
                self.assertEqual(trial['accepted'], name != 'invalid')
                self.assertEqual([c['payload']['stage'] for c in reviewer.calls], ['pre', 'post'])
                self.assertIn('post_review', trial)
                post_evidence = reviewer.calls[1]['payload']['evidence']
                if name == 'invalid':
                    self.assertIn('import os', json.dumps(post_evidence))
                    validation = [e for e in post_evidence if e['kind'] == 'validation']
                    self.assertTrue(validation)
                    self.assertTrue(validation[0]['data'].get('error'))
                else:
                    candidate_readouts = [e for e in post_evidence
                                          if e['kind'] == 'memory_readout' and e['id'].startswith('candidate:')]
                    self.assertTrue(candidate_readouts)
                    self.assertFalse(any(e['data'].get('actual_trace_available') for e in candidate_readouts))
                self.assertEqual(any('confirm' in c['phase'] for c in evaluator.calls), name != 'invalid')
                self.assert_no_confirmation_leak(proposer, reviewer)

    def test_review_failure_is_persisted_and_cannot_promote_or_retry_on_resume(self):
        for stage in ('pre', 'post'):
            with self.subTest(stage=stage):
                out = self.root / stage
                proposer, evaluator, reviewer = self.components(fail_stage=stage)
                state = self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                trial = self.trials(state, out)[0]
                self.assertFalse(trial['accepted'])
                self.assertTrue(trial.get('review_error'))
                self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
                self.assertFalse(any('confirm' in c['phase'] for c in evaluator.calls))
                if stage == 'pre':
                    self.assertFalse(proposer.calls)
                counts = (len(proposer.calls), len(evaluator.calls), len(reviewer.calls))
                resumed = self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                self.assertEqual(resumed, state)
                self.assertEqual((len(proposer.calls), len(evaluator.calls), len(reviewer.calls)), counts)

    def test_invalid_report_schema_or_unresolved_evidence_cannot_authorize_acceptance(self):
        for invalid in ('approval', 'unsupported_evidence', 'bad_interpretation'):
            with self.subTest(invalid=invalid):
                out = self.root / invalid
                proposer, evaluator, reviewer = self.components(invalid=invalid)
                state = self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                trial = self.trials(state, out)[0]
                self.assertFalse(trial['accepted'])
                self.assertTrue(trial.get('review_error'))
                self.assertFalse(proposer.calls)
                self.assertFalse(any('confirm' in c['phase'] for c in evaluator.calls))

    def test_completed_or_interrupted_commit_resume_does_not_repeat_reviews(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt):
                out = self.root / str(interrupt)
                proposer, evaluator, reviewer = self.components()
                if interrupt:
                    with mock.patch.object(RunStore, 'commit', side_effect=InterruptedError('synthetic interruption')):
                        with self.assertRaises(InterruptedError):
                            self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                    self.assertEqual(read(out / 'state.json')['next_round'], 0)
                else:
                    self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                counts = (len(proposer.calls), len(evaluator.calls), len(reviewer.calls))
                state = self.engine(proposer, evaluator, reviewer, out=out).run(rounds=1)
                self.assertEqual(state['revision'], 1)
                self.assertEqual((len(proposer.calls), len(evaluator.calls), len(reviewer.calls)), counts)

    def test_valid_review_does_not_override_tied_independent_confirmation(self):
        proposer, evaluator, reviewer = self.components()
        evaluator.confirmation = 'tie'
        state = self.engine(proposer, evaluator, reviewer).run(rounds=1)
        trial = self.trials(state)[0]
        self.assertEqual([c['payload']['stage'] for c in reviewer.calls], ['pre', 'post'])
        self.assertGreater(trial['exploration']['delta'], 0)
        self.assertEqual(trial['confirmation']['delta'], 0)
        self.assertFalse(trial['accepted'])
        self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)
        self.assert_no_confirmation_leak(proposer, reviewer)

    def test_formal_evolution_cannot_omit_the_fixed_reviewer(self):
        proposer, evaluator, _ = self.components()
        with self.assertRaisesRegex(ValueError, '[Rr]eview'):
            EvolutionEngine(
                out=self.root, tokenizer=TinyTokenizer(), proposer=proposer,
                evaluator=evaluator, reviewer=None, evolution_memory=EvolutionMemory(),
                context={'family': 'synthetic-clean', 'environment': 'alfworld'}, logs=self.logs,
                exploration_tasks=self.explore, confirmation_blocks=self.blocks,
                initial_source=BASELINE_SOURCE, seed=11, synthetic=False,
            )
        self.assertFalse(proposer.calls)
        self.assertFalse(evaluator.calls)

    def test_resuming_with_changed_reviewer_fingerprint_is_rejected(self):
        proposer, evaluator, reviewer = self.components()
        self.engine(proposer, evaluator, reviewer).run(rounds=1)
        reviewer.fingerprint = 'different-reviewer-prompt-or-parameters'
        with self.assertRaises((ValueError, RuntimeError)):
            self.engine(proposer, evaluator, reviewer).run(rounds=1)


if __name__ == '__main__':
    unittest.main()
