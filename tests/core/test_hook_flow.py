"""Synthetic Hook flow contracts; real CPU workers, fake environment/Actor/reviewer.

These tests establish transport and evidence boundaries, never task effectiveness.
No GPU, network request, or native benchmark episode is performed.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from evomem.common import digest, read
from phase1_codegen.engine import EvolutionEngine, own_trace
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.hooks import HookSession, replay_episode
from phase1_codegen.learning import make_snapshot, snapshot_logs
from phase1_codegen.memory import CodeMemory
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.roles.evolver import CodeEvolver, EVOLVER_PROMPT, HOOK_INTERFACE_PROMPT
from phase1_codegen.roles.reviewer import bounded_review_payload
from tests.core.test_engine import TinyTokenizer, fixture_log, task
from tests.core.test_review import CharacterTokenizer
from tests.core.test_review_loop import EventEvaluator, EventProposer, SyntheticReviewer

CONTEXT = dict(environment='alfworld', family='synthetic-clean', memory_scope='environment',
               memory_hook_version='memory-events-v1')
PURE_HOOK_SOURCE = '''def on_event(state, event, context):
    memory = None
    if event["type"] == "before_action":
        memory = state.get("episodes", [])
    if event["type"] == "task_end":
        experience = event["experience"]
        rows = state.get("episodes", [])
        rows.append({"id": experience["task_id"], "experience": experience.get("goal", "")})
        state = {"episodes": rows[-4:]}
    return {"state": state, "session": event["session"], "memory": memory}
'''
HOOK_SOURCE = PURE_HOOK_SOURCE
CHANGED_HOOK_SOURCE = HOOK_SOURCE.replace('memory = state.get("episodes", [])',
                                         'memory = state.get("episodes", [])[-1:]')


class ReviewTokenizer(TinyTokenizer):
    """Synthetic packing only; the production tokenizer has separate tests."""
    def count(self, text): return (len(text) + 3) // 4
    def clip(self, text, count): return text[:max(0, count) * 4]
    def chat_count(self, messages):
        return sum(self.count(message['content']) + 16 for message in messages) + 16


def log():
    return dict(fixture_log(), environment='alfworld')


def make_hook_episode(program, logs, item, phase, *, allow_write=False):
    """Actual sandbox events; Actor delivery and native outcome are test doubles."""
    bank = CodeMemory(program, logs, TinyTokenizer(), CONTEXT)
    session = HookSession(bank, item, allow_write)
    episode = dict(task_id=item['id'], group_id=item['group_id'], room=item['room'],
        environment='alfworld', family=item['family'], goal=item['goal'], history=[dict(
            observation='synthetic initial observation', action='synthetic action',
            next_observation='synthetic next observation')],
        score=float(program.hash != Program(BASELINE_SOURCE).hash), done=True, error=None,
        phase=phase, seed=item['seed'], synthetic=True, native_reward_verified=False)
    session.emit('task_start', dict(step=0, visible=dict(obs='synthetic initial observation', goal=item['goal'])))
    initial = copy.deepcopy(session.current_memory)
    session.emit('before_action', dict(step=0, visible=dict(obs='synthetic initial observation')))
    injected = copy.deepcopy(session.current_memory)
    session.emit('after_observation', dict(step=1, action='synthetic action',
                                         visible=dict(obs='synthetic next observation')))
    session.finish(episode)
    if allow_write:
        session.publish()
    else:
        session.close()
    summary = session.summary()
    summary.update(actor_delivery_verified=True, injections=[dict(memory=injected,
        event_sequence=1, attempt=0, actor_delivery_verified=True,
        actor_request_hash=digest('synthetic Actor request'), system_hash=digest('synthetic system'))])
    episode.update(memory_hook=summary, program_hash=program.hash,
        native_recorder=dict(initial=dict(goal=item['goal'], obs='synthetic initial observation'),
                             memory_retrieved=initial))
    return episode, bank


class HookEvaluator(EventEvaluator):
    fingerprint = 'synthetic-hook-flow-no-real-environment-or-actor'

    def __init__(self, events):
        super().__init__(events)
        self.episodes = []

    def __call__(self, program, logs, tasks, phase):
        rows = super().__call__(program, logs, tasks, phase)
        rows = [make_hook_episode(program, logs, item, phase)[0] for item in tasks]
        self.episodes.extend(copy.deepcopy(rows))
        return rows


class HookFlowTests(unittest.TestCase):
    def test_campaign_uses_one_fixed_hook_workflow(self):
        from phase1_codegen.campaign import workflow_context
        selected = [dict(id='synthetic', family='synthetic-clean')]
        self.assertEqual(workflow_context('alfworld', selected)['memory_hook_version'], 'memory-events-v1')
        with self.assertRaisesRegex(ValueError, 'always uses'):
            workflow_context('alfworld', selected, {'memory_hook_version': None})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-hook-flow-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = []
        self.proposer = EventProposer([PURE_HOOK_SOURCE], self.events)
        self.evaluator = HookEvaluator(self.events)
        self.reviewer = SyntheticReviewer(self.events)
        self.item = dict(task('exploration-hook'), environment='alfworld')

    def engine(self, *, enabled=True, initial=BASELINE_SOURCE, source=PURE_HOOK_SOURCE):
        context = dict(CONTEXT)
        if not enabled:
            context.pop('memory_hook_version')
        self.proposer.sources = [source]
        return EvolutionEngine(self.root, TinyTokenizer(), self.proposer, self.evaluator,
            EvolutionMemory(), context, [log()], [self.item], [], initial_source=initial,
            seed=11, synthetic=True, reviewer=self.reviewer, max_rounds=1)

    def test_candidate_runs_same_reviewer_before_and_after_before_adoption(self):
        engine = self.engine()
        state = engine.run(rounds=1)
        trial = engine.store.history(state)[0]
        self.assertTrue(trial['mechanism_validated'])
        self.assertTrue(trial['accepted'])
        self.assertEqual(trial['acceptance_basis'], 'reviewer_and_reward')
        self.assertNotIn('confirmation', trial)
        self.assertEqual([c['payload']['stage'] for c in self.reviewer.calls], ['pre', 'post'])
        self.assertLess(self.events.index('review:pre'), self.events.index('propose'))
        candidate_event = next(i for i, value in enumerate(self.events) if '/exploration/candidate' in value)
        self.assertLess(self.events.index('propose'), candidate_event)
        self.assertLess(candidate_event, self.events.index('review:post'))
        self.assertEqual(state['current_program'], Program(PURE_HOOK_SOURCE).hash)
        self.assertEqual(Program(PURE_HOOK_SOURCE).entrypoints, frozenset({'on_event'}))
        self.assertIn('on_event', self.proposer.calls[0]['payload']['interface'])
        self.assertEqual(len(self.proposer.calls), 1)
        self.assertEqual(len(self.evaluator.calls), 2)

    def test_omitted_version_normalizes_to_the_same_single_workflow(self):
        engine = self.engine(enabled=False)
        self.assertEqual(engine.context['memory_hook_version'], 'memory-events-v1')
        state = engine.run(rounds=1)
        trial = engine.store.history(state)[0]
        self.assertTrue(trial['mechanism_validated'])
        self.assertTrue(trial['accepted'])
        self.assertIn('on_event', self.proposer.calls[0]['payload']['interface'])
        self.assertEqual([c['payload']['stage'] for c in self.reviewer.calls], ['pre', 'post'])
        self.assertEqual(len(self.evaluator.calls), 2)

    def test_equal_initial_slots_do_not_skip_dynamic_hook_candidate_rollout(self):
        engine = self.engine(initial=HOOK_SOURCE, source=CHANGED_HOOK_SOURCE)
        state = engine.run(rounds=1)
        trial = engine.store.history(state)[0]
        behavior = read(self.root / trial['behavior_path'])
        self.assertEqual(behavior['parent']['input_hashes'], behavior['candidate']['input_hashes'])
        self.assertEqual(behavior['candidate']['readouts'], [{}])
        self.assertEqual(len(self.evaluator.calls), 2)
        self.assertTrue(trial['exploration']['complete'])
        self.assertNotEqual(trial['public_status'], 'no_effect_on_current_exploration_inputs')

    def test_own_trace_retains_exact_hook_and_terminal_identity_for_learning_replay(self):
        episode, bank = make_hook_episode(Program(HOOK_SOURCE), [], self.item, 'synthetic-learning', allow_write=True)
        trace = own_trace(dict(episode, unrelated_private_debug='must be omitted'))
        self.assertEqual(trace['memory_hook'], episode['memory_hook'])
        self.assertEqual(trace['group_id'], episode['group_id'])
        self.assertEqual(trace['room'], episode['room'])
        self.assertNotIn('unrelated_private_debug', trace)
        snapshot = make_snapshot([trace], 'alfworld', memory_snapshot=bank.export_snapshot())
        self.assertEqual(snapshot_logs(snapshot), [trace])
        rebuilt = CodeMemory(Program(HOOK_SOURCE), [], TinyTokenizer(), CONTEXT)
        replay_episode(rebuilt, snapshot_logs(snapshot)[0])
        self.assertEqual(rebuilt.state, bank.state)
        self.assertEqual(rebuilt.fingerprint, bank.fingerprint)

    def test_offline_initial_empty_is_scoped_only_to_task_start(self):
        engine = self.engine()
        candidate = Program(HOOK_SOURCE)
        behavior = engine._signature(candidate, [dict(goal=self.item['goal'], observation='synthetic initial')])
        evidence = engine._mechanism_evidence('candidate', candidate.source, behavior, [])
        readout = next(row['data'] for row in evidence if row['kind'] == 'memory_readout')
        self.assertEqual(readout['replayed_readout'], {})
        self.assertEqual(readout['read_tokens_unit'], 'task_start_readout')
        self.assertIn('task_start only', readout['limitation'])
        self.assertFalse(readout['actual_trace_available'])

    def test_bounded_review_retains_dynamic_slot_and_episode_cost_units(self):
        engine = self.engine()
        engine.run(rounds=1)
        original = self.reviewer.calls[-1]['payload']
        shown = bounded_review_payload(original, ReviewTokenizer())
        dynamic = next(row['data'] for row in shown['evidence'] if row['id'].startswith('candidate:hook_read:'))
        self.assertTrue(dynamic['actual_trace_available'])
        self.assertEqual(dynamic['provenance']['actor_request_hash'], digest('synthetic Actor request'))
        self.assertEqual(dynamic['provenance']['system_hash'], digest('synthetic system'))
        self.assertEqual(dynamic['provenance']['attempt'], 0)
        self.assertEqual(dynamic['read_tokens_unit'], 'one_actor_memory_slot')
        self.assertEqual(dynamic['read_tokens_source'], 'actual_actor_slot')
        self.assertTrue(dynamic['fragments'])
        self.assertIn('clean apple', dynamic['fragments'][0]['fragment']['text'])
        self.assertIn('does not prove', dynamic['limitation'])
        facts = next(row['data'] for row in shown['evidence'] if row['kind'] == 'current_facts')
        current = facts['programs']['candidate']
        hook = current['hook_episodes'][0]
        self.assertEqual(hook['event_count'], 4)
        self.assertEqual(hook['actor_slot_count'], 1)
        self.assertEqual(hook['nonempty_actor_slots'], 1)
        self.assertGreater(hook['usage']['injected_tokens'], 0)
        self.assertEqual(hook['usage']['model_calls'], 0)
        self.assertIn('worker_cpu_seconds', hook['usage'])
        self.assertIn('this native episode', hook['cost_scope'])
        self.assertEqual({row['unit'] for row in current['reads']}, {'task_start_readout', 'one_actor_memory_slot'})
        # An empty initial diagnostic cannot erase the actual subsequent delivery.
        self.assertGreater(sum(row['read_tokens'] for row in current['reads']), 0)

    def test_fixed_prompt_has_one_event_contract_without_design_examples(self):
        evolver = CodeEvolver.__new__(CodeEvolver)
        evolver.tokenizer = CharacterTokenizer()
        evolver.llm = mock.Mock(model_name="synthetic-test-model")
        evolver._invoke = mock.Mock(return_value={'source': HOOK_SOURCE, 'hypothesis': 'synthetic', 'risk': 'synthetic'})
        for enabled in (False, True):
            payload = dict(current_source=BASELINE_SOURCE, context=dict(CONTEXT))
            if not enabled:
                payload['context'].pop('memory_hook_version')
            evolver(payload, 17)
            request = evolver._invoke.call_args.args[0]
            expected = EVOLVER_PROMPT
            self.assertEqual(request['messages'][0]['content'], expected)
            self.assertEqual(json.loads(request['messages'][1]['content']), payload)
            self.assertNotIn(HOOK_SOURCE, request['messages'][0]['content'])
        self.assertIn('No timing policy, representation, model use or algorithm is prescribed.', HOOK_INTERFACE_PROMPT)
        self.assertNotIn('def on_event(', HOOK_INTERFACE_PROMPT)
        self.assertNotIn('supersedes', HOOK_INTERFACE_PROMPT)
        self.assertNotIn('write-only mutation restriction above', HOOK_INTERFACE_PROMPT)
        self.assertEqual(evolver._invoke.call_count, 2)
        with self.assertRaises(ValueError):
            evolver(dict(context={'memory_hook_version': 'unfrozen-new-abi'}), 18)
        self.assertEqual(evolver._invoke.call_count, 2)
