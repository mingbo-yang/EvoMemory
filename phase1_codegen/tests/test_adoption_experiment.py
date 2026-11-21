"""Experimental B contracts; all scores and reviews are synthetic."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from evomem.common import digest, read, write
from phase1_codegen.acceptance import POLICIES, adoption_feedback, eligible_for_adoption
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.proposer import CodeEvolver
from phase1_codegen.review import citation_pointers, validate_report
from phase1_codegen.store import RunStore
from phase1_codegen.tests import test_acceptance_policy as fixtures
from phase1_codegen.tests.test_engine import FIRST_SOURCE
from phase1_codegen.tests.test_review import CharacterTokenizer
from phase1_codegen.tests.test_review_contract import observation, synthetic_raw_report
from phase1_codegen.tests.test_review_loop import SyntheticReviewer

B = 'reviewer_and_reward_or_equal_utility'


class UtilityReviewer(SyntheticReviewer):
    def __call__(self, payload, seed):
        report = super().__call__(payload, seed)
        if payload['stage'] != 'post':
            return report
        shown = self.last_payload
        raw = synthetic_raw_report(shown, self.verdict)
        raw['observations'] = []
        for side in ('parent', 'candidate'):
            eid = side + ':storage'
            pointer = next(pointer for anchor in shown['citation_registry']['evidence'][eid]
                           for pointer in citation_pointers(anchor) if pointer.endswith('/fragment/text'))
            raw['observations'].append(observation(shown, eid, pointer, 'o' + str(len(raw['observations']) + 1)))
        raw['evaluations'] = [dict(id='e1', dimension='Faithfulness', polarity='advantage',
            observation_ids=['o1', 'o2'], uncertainty=['causal_attribution_unproven'])]
        return validate_report(raw, shown)


class AdoptionExperimentTests(unittest.TestCase):
    setUp = fixtures.AcceptancePolicyTests.setUp
    components = fixtures.AcceptancePolicyTests.components
    engine = fixtures.AcceptancePolicyTests.engine
    def b_run(self, *, parent=.5, candidate=.5, verdict='positive', source=FIRST_SOURCE, utility=True, name='b'):
        proposer, evaluator, reviewer = self.components(source=source, parent_score=parent,
            candidate_score=candidate, verdict=verdict)
        if utility:
            reviewer = UtilityReviewer(self.events, verdict=verdict)
        out = self.root / name
        engine = self.engine(proposer, evaluator, reviewer, out=out, policy=B)
        state = engine.run(1)
        trial = RunStore(out).history(state)[0]
        return engine, state, trial, proposer, evaluator, reviewer

    def test_b_accepts_positive_equal_grounded_utility_without_extra_calls(self):
        engine, state, trial, proposer, evaluator, reviewer = self.b_run()
        self.assertTrue(trial['accepted'])
        self.assertEqual(state['current_program'], Program(FIRST_SOURCE).hash)
        self.assertFalse(trial['acceptance_feedback']['strict_reward_branch'])
        self.assertTrue(trial['acceptance_feedback']['equal_utility_branch']['eligible'])
        self.assertEqual(len(proposer.calls), 1)
        self.assertEqual([c['payload']['stage'] for c in reviewer.calls], ['pre', 'post'])
        self.assertFalse(any('confirmation' in c['phase'] for c in evaluator.calls))
        self.assertEqual(engine.run(1), state)
        self.assertEqual(len(proposer.calls), 1)
        self.assertEqual(len(reviewer.calls), 2)
        RunStore(engine.store.root).current()

    def test_b_preserves_a_improvement_even_without_equal_utility(self):
        engine, state, trial, *_ = self.b_run(parent=0, candidate=1, utility=False)
        self.assertTrue(trial['accepted'])
        self.assertTrue(trial['acceptance_feedback']['strict_reward_branch'])
        self.assertFalse(trial['acceptance_feedback']['equal_utility_branch']['eligible'])

    def test_evolver_receives_the_deployment_policy_bound_to_its_arm(self):
        for policy in ('reviewer_and_reward', B):
            with self.subTest(policy=policy):
                proposer, evaluator, reviewer = self.components()
                engine = self.engine(proposer, evaluator, reviewer, out=self.root / policy, policy=policy)
                engine.run(1)
                call = proposer.calls[0]
                evolver = CodeEvolver.__new__(CodeEvolver)
                evolver.tokenizer = CharacterTokenizer()
                evolver.llm = SimpleNamespace(model_name='synthetic-test-model')
                evolver._invoke = Mock()
                evolver(call['payload'], call['seed'])
                system, user = evolver._invoke.call_args.args[0]['messages']
                self.assertIn('deployment_policy', system['content'])
                self.assertNotIn('strictly above the parent', system['content'])
                rule = json.loads(user['content'])['deployment_policy']
                self.assertEqual(rule, dict(name=policy, rule=engine.binding['acceptance']))
                self.assertEqual(rule['rule'], POLICIES[policy])
                self.assertEqual('equal-score branch' in rule['rule'], policy == B)

    def test_b_rejects_zero_regression_negative_and_unbound_quality(self):
        for name, options in [('zero', dict(parent=0,candidate=0)),
                ('regression', dict(parent=1,candidate=.5)),
                ('negative', dict(verdict='negative')), ('unbound',dict(utility=False))]:
            with self.subTest(name=name):
                _, state, trial, *_ = self.b_run(name=name, **options)
                self.assertFalse(trial['accepted'])
                self.assertEqual(state['current_program'], Program(BASELINE_SOURCE).hash)

    def test_invalid_b_does_not_require_missing_current_pairs(self):
        engine, state, trial, *_ = self.b_run(source=fixtures.RUNTIME_FAILURE_SOURCE, utility=False)
        self.assertFalse(trial['accepted'])
        self.assertIsNone(engine.store.adoption_context(state['binding'],trial))
        self.assertEqual(state['revision'],0)

    def test_b_recomputes_current_facts_from_original_native_evidence(self):
        engine, state, trial, *_ = self.b_run()
        receipt = copy.deepcopy(trial['post_review'])
        original = read(engine.store.root / receipt['input_path'])
        trace = next(row['data'] for row in original['evidence']
                     if row['kind']=='execution_trace' and row['id'].startswith('candidate:'))
        trace['score'] = 0
        write(engine.store.root / receipt['input_path'],original)
        receipt['payload_hash'] = digest(original)
        write(engine.store.root / receipt['receipt_path'],receipt)
        changed = dict(trial,post_review=receipt)
        with self.assertRaisesRegex(RuntimeError,'facts|comparison'):
            engine.store.adoption_context(state['binding'],changed)
        with self.assertRaises(RuntimeError):
            engine.store.current()

    def test_b_continual_adoption_preserves_learning_frontier_and_resume(self):
        from phase1_codegen.tests import test_continual_engine as learning
        from phase1_codegen.learning import load_snapshot, snapshot_logs
        class EqualLearningEvaluator(learning.LearningEvaluator):
            def __call__(self, *args, **kwargs):
                rows = super().__call__(*args, **kwargs)
                for row in rows:
                    row.update(score=.5, done=True)
                return rows
        case = learning.ContinualEngineTests()
        case.setUp()
        try:
            evaluator = EqualLearningEvaluator()
            reviewer = UtilityReviewer([])
            engine = case.engine(evaluator, reviewer=reviewer, confirmation_blocks=[], acceptance_policy=B)
            state = engine.run()
            self.assertEqual(state['revision'], 2)
            self.assertEqual([p for p, _ in evaluator.learning_calls],
                             [Program(BASELINE_SOURCE).hash, Program(FIRST_SOURCE).hash])
            snapshot = load_snapshot(case.root, state['task_memory'])
            self.assertEqual([ep['task_id'] for ep in snapshot_logs(snapshot)],
                             ['adaptation-only', 'learn-one', 'learn-two'])
            self.assertEqual(snapshot['program_hash'], state['current_program'])
            self.assertEqual(RunStore(case.root).current().hash, state['current_program'])
            self.assertEqual(engine.run(), state)
            self.assertEqual(len(engine.proposer.calls), 2)
            self.assertEqual(len(reviewer.calls), 4)
        finally:
            case.doCleanups()

    def test_b_feedback_cannot_be_forged_at_commit_boundary(self):
        engine,state,trial,*_=self.b_run()
        changed=copy.deepcopy(trial)
        changed['acceptance_feedback']['equal_utility_branch']['eligible']=False
        with self.assertRaisesRegex(RuntimeError,'feedback'):
            engine.store.validate_trial(state['binding'],changed)


if __name__ == '__main__':
    unittest.main()
