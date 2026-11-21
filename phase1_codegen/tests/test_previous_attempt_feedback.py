"""Iteration feedback contracts; synthetic rewards are not efficacy evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from evomem.common import digest, read, write
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory, public_event
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.proposer import CodeEvolver, PARAMETERS, EVOLVER_PROMPT
from phase1_codegen.protocol import EVENT_CONTRACT, META_CONTEXT_TOKENS
from phase1_codegen.tests.test_engine import (FIRST_SOURCE, SECOND_SOURCE, SyntheticEvaluator,
                                            SyntheticProposer, fixture_log, task)
from phase1_codegen.tests.test_review import CharacterTokenizer
from phase1_codegen.tests.test_review_loop import SyntheticReviewer


class EvidenceTokenizer(CharacterTokenizer):
    """Cheap deterministic fixture, unrelated to the actual model tokenizer."""
    def count(self, text):
        return (len(text) + 3) // 4


BAD_SYNTAX = '''def on_event(state, event, context):
    rows = sorted([], key=lambda row: row)
    return {"state": state, "session": {}, "memory": None}
'''
BAD_ROWS = '''def on_event(state, event, context):
    return {"state": state, "session": {}, "memory": [{"id": "x", "experience": "text", "extra": 1}]}
'''
EMPTY_QUERY = '''def on_event(state, event, context):
    query = context.get("goal", "")
    return {"state": state, "session": {"query": query}, "memory": None}
'''


class PreviousAttemptFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-previous-attempt-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = mock.patch('phase1_codegen.engine.code_binding', return_value={'synthetic': 'fixed'})
        patcher.start(); self.addCleanup(patcher.stop)

    def engine(self, proposer, *, root=None, evaluator=None, reviewer=None):
        return EvolutionEngine(root or self.root, EvidenceTokenizer(), proposer,
            evaluator or SyntheticEvaluator(), EvolutionMemory(),
            {'environment': 'alfworld', 'family': 'synthetic-clean'}, [fixture_log()],
            [task('probe-a'), task('probe-b')],
            [[task('sealed-%d-a' % n), task('sealed-%d-b' % n)] for n in range(2)],
            synthetic=True, acceptance_policy='native_reward', reviewer=reviewer)

    def test_rejected_drafts_survive_exactly_and_next_round_has_one_proposal(self):
        for name, source, parsed in [('syntax', BAD_SYNTAX, False), ('output', BAD_ROWS, True)]:
            with self.subTest(name=name):
                proposer = SyntheticProposer([source, FIRST_SOURCE])
                engine = self.engine(proposer, root=self.root / name)
                state = engine.run(2)
                trials = engine.store.history(state)
                previous = proposer.calls[1]['payload']['previous_attempt']
                self.assertEqual(previous['source'], source)
                self.assertEqual(previous['raw_source_hash'], hashlib.sha256(source.encode()).hexdigest())
                self.assertEqual(previous['parsed_program_hash'] is not None, parsed)
                self.assertEqual(previous['error'], trials[0]['error'])
                self.assertFalse(previous['accepted'])
                self.assertFalse(previous['mechanism_validated'])
                self.assertEqual(proposer.calls[1]['payload']['current_source'], BASELINE_SOURCE)
                self.assertEqual(len(proposer.calls), 2)
                self.assertEqual(state['next_round'], 2)
                self.assertEqual(state['revision'], 1)
                self.assertIn(source.splitlines()[1], trials[0]['evolution_event']['code_diff'])
                self.assertEqual(trials[0]['evolution_event']['after_hash'] is not None, parsed)
                self.assertEqual(trials[0]['evolution_event']['raw_source_hash'], previous['raw_source_hash'])
                self.assertEqual(proposer.calls[0]['payload']['interface']['event_contract'], EVENT_CONTRACT)
                calls = copy.deepcopy(proposer.calls)
                self.assertEqual(engine.run(2), state)
                self.assertEqual(proposer.calls, calls)

    def test_changed_raw_draft_is_detected_before_another_proposal(self):
        proposer = SyntheticProposer([BAD_SYNTAX, FIRST_SOURCE])
        engine = self.engine(proposer)
        engine.run(1)
        path = self.root / 'trials/000/proposal.json'
        receipt = read(path)
        receipt['value']['source'] += '# altered raw candidate\n'
        write(path, receipt)
        with self.assertRaisesRegex(RuntimeError, 'raw proposal source changed'):
            engine.run(2)
        self.assertEqual(len(proposer.calls), 1)
        self.assertEqual(read(self.root / 'state.json')['next_round'], 1)

    def test_invalid_proposal_envelope_retains_source_without_filling_missing_fields(self):
        class MissingRisk(SyntheticProposer):
            def __call__(self, payload, seed):
                value = super().__call__(payload, seed)
                if len(self.calls) == 1:
                    value.pop('risk')
                return value
        proposer = MissingRisk([FIRST_SOURCE, SECOND_SOURCE])
        engine = self.engine(proposer)
        state = engine.run(2)
        trials = engine.store.history(state)
        previous = proposer.calls[1]['payload']['previous_attempt']
        self.assertEqual(previous['source'], FIRST_SOURCE)
        self.assertEqual(previous['raw_source_hash'], Program(FIRST_SOURCE).hash)
        self.assertIsNone(previous['parsed_program_hash'])
        self.assertIn('Expected exactly string fields', previous['error'])
        self.assertIsNone(previous['risk'])
        self.assertFalse(previous['mechanism_validated'])
        self.assertFalse(trials[0]['accepted'])
        self.assertEqual(proposer.calls[1]['payload']['current_source'], BASELINE_SOURCE)
        self.assertEqual(trials[0]['evolution_event']['raw_source_hash'], Program(FIRST_SOURCE).hash)
        self.assertIn('Recorded task goal:', trials[0]['evolution_event']['code_diff'])
        self.assertNotIn('risk', read(self.root / 'trials/000/proposal.json')['value'])
        self.assertEqual(len(proposer.calls), 2)

    def test_empty_source_is_candidate_validation_feedback_and_next_round_continues(self):
        for name, source in [('empty', ''), ('whitespace', ' \n  ')]:
            with self.subTest(name=name):
                proposer = SyntheticProposer([source, FIRST_SOURCE])
                reviewer = SyntheticReviewer([])
                engine = self.engine(proposer, root=self.root / name, reviewer=reviewer)
                state = engine.run(2)
                trials = engine.store.history(state)
                self.assertNotIn('review_error', trials[0])
                self.assertFalse(trials[0]['mechanism_validated'])
                self.assertEqual(trials[0]['raw_source_hash'], hashlib.sha256(source.encode()).hexdigest())
                self.assertIn('-def ', trials[0]['evolution_event']['code_diff'])
                self.assertEqual(proposer.calls[1]['payload']['previous_attempt']['source'], source)
                post = next(call['payload'] for call in reviewer.calls if call['payload']['stage'] == 'post')
                self.assertNotIn('candidate:program', {row['id'] for row in post['evidence']})
                self.assertIn('candidate:validation', {row['id'] for row in post['evidence']})
                self.assertEqual(trials[0]['post_review']['report']['assessment']['subject'], 'candidate')
                self.assertEqual(state['next_round'], 2)
                self.assertEqual(len(proposer.calls), 2)

    def test_absent_draft_and_empty_draft_have_distinct_evolution_evidence(self):
        trial = dict(public_context={'environment': 'alfworld'}, parent_hash=Program(BASELINE_SOURCE).hash,
                     accepted=False, public_status='invalid', synthetic=True)
        absent = public_event(trial, BASELINE_SOURCE, None)
        empty = public_event(trial, BASELINE_SOURCE, '')
        self.assertIsNone(absent['raw_source_hash'])
        self.assertEqual(absent['code_diff'], '')
        self.assertEqual(empty['raw_source_hash'], hashlib.sha256(b'').hexdigest())
        self.assertIn('-def ', empty['code_diff'])
        self.assertNotEqual(absent['id'], empty['id'])

    def test_runtime_observations_reach_next_author_without_reviewer_citations(self):
        proposer = SyntheticProposer([EMPTY_QUERY, FIRST_SOURCE])
        evaluator = SyntheticEvaluator(sources=(EMPTY_QUERY, FIRST_SOURCE), confirmation='tie')
        engine = self.engine(proposer, evaluator=evaluator)
        state = engine.run(2)
        trial = engine.store.history(state)[0]
        previous = proposer.calls[1]['payload']['previous_attempt']
        facts = previous['execution_facts']
        self.assertIsNone(previous['post_review'])
        self.assertIn('missing_dict_keys', facts['facts'])
        missing = facts['facts']['missing_dict_keys']
        self.assertTrue(any(row['first_missing']['key'] == 'goal' for row in missing))
        self.assertEqual(facts['provenance']['program_hash'], Program(EMPTY_QUERY).hash)
        self.assertEqual(facts['provenance']['evidence_time'], 'historical_previous_attempt')
        self.assertEqual(facts['provenance']['round'], 0)
        behavior = read(self.root / trial['behavior_path'])
        self.assertEqual(facts['original_hash'], digest(behavior['candidate']['diagnostics']))
        self.assertNotIn('runtime_snapshot', previous)

    def test_source_and_input_anchor_are_both_checked(self):
        proposer = SyntheticProposer([BAD_ROWS, SECOND_SOURCE])
        engine = self.engine(proposer)
        state = engine.run(1)
        history = engine.store.history(state)
        original = read(self.root / 'trials/000/proposer_input.json')
        write(self.root / 'trials/000/proposer_input.json', dict(original, context={'changed': True}))
        with self.assertRaisesRegex(RuntimeError, 'input binding changed'):
            engine._previous_attempt(history)

    def test_previous_attempt_cannot_cross_environment(self):
        proposer = SyntheticProposer([BAD_SYNTAX])
        engine = self.engine(proposer)
        state = engine.run(1)
        engine.context = dict(engine.context, environment='textworld')
        with self.assertRaisesRegex(RuntimeError, 'different dataset environment'):
            engine._previous_attempt(engine.store.history(state))

    def test_frozen_older_draft_requires_its_committed_review_source_anchor(self):
        proposer = SyntheticProposer([BAD_SYNTAX])
        engine = self.engine(proposer, reviewer=SyntheticReviewer([]))
        state = engine.run(1)
        original = engine.store.history(state)[0]
        historical = {key: value for key, value in original.items()
                      if key not in ('raw_source_hash', 'proposal_payload_hash')}
        previous = engine._previous_attempt([historical])
        self.assertEqual(previous['source'], BAD_SYNTAX)
        self.assertEqual(previous['provenance']['source_binding'], 'committed_post_review_input')
        self.assertIsNone(previous['parsed_program_hash'])
        self.assertEqual(previous['post_review']['report'], original['post_review']['report']['evidence_contract'])
        self.assertNotIn('summary', previous['post_review']['report'])
        self.assertNotIn('findings', previous['post_review']['report'])
        self.assertNotIn('current_program_hash', previous['post_review'])
        without_anchor = dict(historical)
        without_anchor.pop('post_review')
        with self.assertRaisesRegex(RuntimeError, 'lacks a committed source evidence anchor'):
            engine._previous_attempt([without_anchor])
        path = self.root / 'trials/000/proposal.json'
        receipt = read(path)
        receipt['value']['source'] += '# changed historical draft\n'
        write(path, receipt)
        with self.assertRaisesRegex(RuntimeError, 'differs from its committed review source'):
            engine._previous_attempt([historical])


class ProposalPackingTests(unittest.TestCase):
    def evolver(self):
        evolver = CodeEvolver.__new__(CodeEvolver)
        evolver.tokenizer = CharacterTokenizer()
        evolver.llm = mock.Mock(model_name="synthetic-test-model")
        evolver._invoke = mock.Mock(return_value={})
        return evolver

    def test_complete_previous_source_survives_long_clipped_history(self):
        evolver = self.evolver()
        previous = dict(source=BAD_SYNTAX, raw_source_hash=hashlib.sha256(BAD_SYNTAX.encode()).hexdigest(),
                        error='precise latest validation failure', execution_facts={'available': False})
        payload = dict(context={}, current_source=BASELINE_SOURCE, previous_attempt=previous,
                       own_trial_history=[{'old': 'x' * 30000}], own_exploration=[{'text': 'y' * 30000}],
                       own_adaptation=[{'text': 'z' * 30000}])
        evolver(payload, 17)
        request, tokens, _ = evolver._invoke.call_args.args
        shown = json.loads(request['messages'][1]['content'])
        self.assertEqual(shown['previous_attempt'], previous)
        self.assertEqual(shown['current_source'], BASELINE_SOURCE)
        self.assertTrue(shown['own_trial_history'].get('clipped') or shown['own_trial_history'].get('omitted'))
        self.assertLessEqual(tokens + PARAMETERS['max_tokens'] + 512, META_CONTEXT_TOKENS)
        self.assertEqual(evolver._invoke.call_count, 1)

    def test_mandatory_sources_overflow_without_a_model_call_or_truncation(self):
        evolver = self.evolver()
        payload = dict(context={}, current_source='x' * META_CONTEXT_TOKENS,
                       previous_attempt={'source': 'y' * META_CONTEXT_TOKENS}, own_adaptation=['optional'])
        original = copy.deepcopy(payload)
        with self.assertRaisesRegex(ValueError, 'mandatory evidence exceed model context'):
            evolver(payload, 17)
        self.assertEqual(payload, original)
        evolver._invoke.assert_not_called()

    def test_optional_logs_shrink_before_a_large_complete_draft(self):
        evolver = self.evolver()
        base = dict(context={}, current_source=BASELINE_SOURCE,
                    previous_attempt={'source': BAD_SYNTAX, 'error': 'exact retained error'})
        occupied = evolver.tokenizer.chat_count([
            dict(role='system', content=EVOLVER_PROMPT),
            dict(role='user', content=json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(',', ':')))])
        draft = BAD_SYNTAX + '\n# ' + 'r' * (META_CONTEXT_TOKENS - PARAMETERS['max_tokens'] - 512 - occupied - 2000)
        payload = dict(context={}, current_source=BASELINE_SOURCE,
                       previous_attempt={'source': draft, 'error': 'exact retained error'},
                       own_trial_history=['old ' * 5000], own_exploration=['trace ' * 5000],
                       own_adaptation=['history ' * 5000])
        evolver(payload, 19)
        request, tokens, _ = evolver._invoke.call_args.args
        shown = json.loads(request['messages'][1]['content'])
        self.assertEqual(shown['previous_attempt']['source'], draft)
        self.assertEqual(shown['previous_attempt']['error'], 'exact retained error')
        self.assertEqual(shown['current_source'], BASELINE_SOURCE)
        self.assertTrue(any('omission_reason' in shown[key]
                            for key in ('own_trial_history', 'own_exploration', 'own_adaptation')))
        self.assertLessEqual(tokens + PARAMETERS['max_tokens'] + 512, META_CONTEXT_TOKENS)


if __name__ == '__main__':
    unittest.main()
