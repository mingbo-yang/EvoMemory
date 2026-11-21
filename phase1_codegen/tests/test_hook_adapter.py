"""CPU-only integration with ALMA's actual workflow source and fake I/O.

The environment/model doubles establish adapter contracts, not task performance.
"""
import asyncio
import copy
import json
from pathlib import Path
import sys
import types
import unittest
import tempfile
from unittest.mock import patch

from phase1_codegen.hook_adapter import HookAdapter, HookExecutionError, active_hook


OFFICIAL = Path(__file__).resolve().parents[2] / 'external/alma/evals/workflows/agent_workflow.py'


class Progress:
    def update(self, *args, **kwargs):
        pass


class Recorder:
    def __init__(self, task):
        self.task = task
        self.init = {'goal': 'goal ' + task, 'obs': 'initial ' + task,
                     'actions_list': ['act'], 'private_environment_data': 'hidden'}
        self.steps = []
        self.reward = 0
        self.memory = None

    async def log_memo_retrieved(self, memory):
        self.memory = copy.deepcopy(memory)


class Environment:
    max_trails = 3

    def __init__(self, order, fail_native_once=False):
        self.order = order
        self.recorders = {}
        self.prompt_calls = []
        self.actions = []
        self.fail_native_once = fail_native_once
        self.failed = set()

    async def set_task_env(self, task):
        recorder = Recorder(task)
        self.recorders[task] = recorder
        await asyncio.sleep(0)
        return copy.deepcopy(recorder.init), recorder

    async def get_prompt(self, game_files, **kwargs):
        self.prompt_calls.append((game_files, copy.deepcopy(kwargs)))
        memo = kwargs.get('memory_retrived', {})
        return [{'role': 'system', 'content': 'OFFICIAL FIXED ACTION POLICY\n' +
                 (json.dumps(memo, sort_keys=True) if memo else '')},
                {'role': 'user', 'content': 'OFFICIAL OBSERVATION\n' + kwargs['obs']}]

    async def run_step(self, action, task, **kwargs):
        self.actions.append((task, action))
        if self.fail_native_once and task not in self.failed:
            self.failed.add(task)
            raise ValueError('native rejected action')
        recorder = self.recorders[task]
        index = len(recorder.steps) + 1
        result = {'obs': 'next ' + task + ' ' + str(index), 'actions_list': ['act'],
                  'scores': index / 2, 'dones': index == 2}
        recorder.steps.append(dict(result, action_took=action))
        self.order.append((task, 'native_step'))
        await asyncio.sleep(0)
        return result, recorder

    async def cal_reward(self, scores, task):
        self.order.append((task, 'native_reward'))
        self.recorders[task].reward = scores
        return self.recorders[task]


class Session:
    def __init__(self, task, order, allow_write=False, fail=None, change_memory=True):
        self.task = task
        self.context = {'environment': 'alfworld', 'task_id': task, 'family': 'synthetic'}
        self.current_memory = {}
        self.trace = []
        self.finished = []
        self.aborted = []
        self.order = order
        self.allow_write = allow_write
        self.commits = 0
        self.fail = fail
        self.change_memory = change_memory

    def emit(self, kind, payload):
        self.trace.append({'type': kind, 'payload': copy.deepcopy(payload)})
        self.order.append((self.task, kind))
        if kind == self.fail:
            raise ValueError('synthetic hook failure: ' + kind)
        if self.change_memory and kind in ('task_start', 'before_action'):
            self.current_memory = {'entries': [{'id': self.task, 'experience':
                self.task + ':' + kind + ':' + str(payload.get('attempt', 0))}]}
        return copy.deepcopy(self.current_memory)

    def finish(self, experience):
        self.emit('task_end', {'experience': experience, 'step': len(experience['history'])})
        self.finished.append(copy.deepcopy(experience))
        self.commits += int(self.allow_write)

    def abort(self, error, payload=None):
        self.aborted.append({'error': error, 'payload': copy.deepcopy(payload)})


class Bridge:
    def __init__(self, order):
        self.order = order
        self.retrievals = []
        self.updates = []

    async def general_retrieve(self, recorder):
        self.retrievals.append(recorder.task)
        session = active_hook.get()
        if session is None:
            return {}
        # This is the trusted native bridge's task-start responsibility.
        self.assert_task(session, recorder)
        return session.emit('task_start', {'step': 0, 'visible': {
            key: recorder.init[key] for key in ('obs', 'goal', 'actions_list')}})

    @staticmethod
    def assert_task(session, recorder):
        if session.task != recorder.task:
            raise AssertionError('cross-episode bridge context')

    async def general_update(self, recorder):
        self.updates.append(recorder.task)
        self.order.append((recorder.task, 'write_intent'))


def load_workflow(requests):
    """Load the pinned upstream file without its model/environment imports."""
    class Agent:
        def __init__(self, model, system_prompt):
            self.model = model
            self.messages = [{'role': 'system', 'content': system_prompt}]

        async def ask(self, user_input, **kwargs):
            requests.append({'messages': copy.deepcopy(self.messages), 'user': user_input,
                             'kwargs': copy.deepcopy(kwargs), 'model': self.model})
            self.messages.append({'role': 'user', 'content': user_input})
            await asyncio.sleep(0)
            answer = '  act  '
            self.messages.append({'role': 'assistant', 'content': answer})
            return answer

    modules = {name: types.ModuleType(name) for name in (
        'utils', 'utils.hire_agent', 'agents', 'agents.memo_structure',
        'eval_envs', 'eval_envs.test_envs', 'logger', 'rich', 'rich.progress')}
    modules['utils.hire_agent'].Agent = Agent
    modules['agents.memo_structure'].MemoStructure = object
    modules['logger'].get_logger = lambda name: types.SimpleNamespace(info=lambda *a: None)
    modules['eval_envs'].test_envs = modules['eval_envs.test_envs']
    for name in ('Progress', 'BarColumn', 'TextColumn', 'TimeElapsedColumn'):
        setattr(modules['rich.progress'], name, object)
    module = types.ModuleType('_synthetic_official_hook_workflow')
    modules[module.__name__] = module
    with patch.dict(sys.modules, modules):
        exec(compile(OFFICIAL.read_text(), str(OFFICIAL), 'exec'), module.__dict__)
    return module


class HookAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.order = []
        self.module = load_workflow(self.requests)
        self.bridge = Bridge(self.order)
        self.workflow = self.module.Agent_Workflow([self.bridge], 'fixed-model/fixed-effort')
        self.workflow.env = Environment(self.order)
        self.workflow.env_type = 'alfworld'
        self.original_code = self.workflow.run_single_task.__func__.__code__
        self.original_agent = self.module.Agent

    @staticmethod
    def task(name='task-a'):
        return {'id': name, 'group_id': name, 'room': 'synthetic', 'family': 'synthetic'}

    async def invoke(self, name='task-a', session=None, eval_type='sequential'):
        session = session or Session(name, self.order)
        adapter = HookAdapter(self.workflow, session, self.task(name), 'synthetic', 17)
        with adapter:
            result = await self.workflow.run_single_task(name, eval_type=eval_type,
                current_state='retrieve', progress=Progress(), task_id=name)
        return result, session, adapter

    def assert_restored(self):
        self.assertIsNone(active_hook.get())
        self.assertNotIn('run_single_task', vars(self.workflow))
        self.assertNotIn('_evomem_event_dispatcher', vars(self.workflow))
        self.assertIs(self.workflow.run_single_task.__func__.__code__, self.original_code)
        self.assertIs(self.module.Agent, self.original_agent)

    async def test_actual_loop_event_order_native_reward_and_single_commit(self):
        session = Session('task-a', self.order, allow_write=True)
        result, session, adapter = await self.invoke(session=session)
        self.assertEqual([row['type'] for row in session.trace],
            ['task_start', 'before_action', 'after_observation',
             'before_action', 'after_observation', 'task_end'])
        self.assertEqual([kind for _, kind in self.order][-3:],
                         ['native_reward', 'write_intent', 'task_end'])
        self.assertEqual(result.reward, 1)
        self.assertEqual(session.commits, 1)
        self.assertEqual(len(session.finished), 1)
        self.assertEqual(session.finished[0]['environment'], 'alfworld')
        self.assertEqual(session.finished[0]['score'], 1)
        self.assertEqual(session.finished[0]['seed'], 17)
        self.assertEqual(adapter.completed_steps, 2)
        self.assertEqual(adapter.attempts, 2)
        for row in session.trace[:-1]:
            self.assertNotIn('experience', row['payload'])
            self.assertNotIn('private_environment_data', row['payload']['visible'])
        self.assertEqual(session.trace[2]['payload']['visible']['scores'], .5)
        self.assert_restored()

    async def test_system_slot_replaced_without_changing_actor_inputs_or_history(self):
        _, session, adapter = await self.invoke()
        self.assertEqual(len(self.requests), 2)
        for index, request in enumerate(self.requests):
            self.assertEqual(request['kwargs'], {'with_history': True, 'reasoning_effort': 'fixed-effort'})
            self.assertEqual(request['model'], 'fixed-model')
            self.assertTrue(request['messages'][0]['content'].startswith('OFFICIAL FIXED ACTION POLICY\n'))
            self.assertIn('task-a:before_action:' + str(index), request['messages'][0]['content'])
            self.assertNotIn('task-a:task_start', request['messages'][0]['content'])
            self.assertEqual(adapter.injections[index]['system_message'], request['messages'][0]['content'])
        self.assertEqual([r['user'] for r in self.requests],
                         ['OFFICIAL OBSERVATION\ninitial task-a', 'OFFICIAL OBSERVATION\nnext task-a 1'])
        self.assertEqual(self.requests[1]['messages'][1:], [
            {'role': 'user', 'content': self.requests[0]['user']},
            {'role': 'assistant', 'content': '  act  '}])
        self.assertEqual(self.workflow.env.actions, [('task-a', 'act'), ('task-a', 'act')])
        self.assertEqual(session.commits, 0)
        self.assert_restored()

    async def test_empty_hook_preserves_exact_original_requests(self):
        await self.workflow.run_single_task('task-a', progress=Progress(), task_id='task-a')
        expected = copy.deepcopy(self.requests)
        self.requests.clear()
        await self.invoke(session=Session('task-a', self.order, change_memory=False))
        self.assertEqual(self.requests, expected)
        self.assert_restored()

    async def test_readonly_batched_path_finishes_without_write_intent(self):
        _, session, _ = await self.invoke(eval_type='batched')
        self.assertEqual(len(session.finished), 1)
        self.assertEqual(session.commits, 0)
        self.assertEqual(self.bridge.updates, [])
        self.assert_restored()

    async def test_fatal_hook_is_not_caught_as_actor_error_or_retried(self):
        for failed_kind, request_count in [('before_action', 0), ('after_observation', 1)]:
            with self.subTest(failed_kind=failed_kind):
                self.requests.clear()
                session = Session('task-a', self.order, allow_write=True, fail=failed_kind)
                with self.assertRaisesRegex(HookExecutionError, failed_kind):
                    await self.invoke(session=session)
                self.assertEqual(len(self.requests), request_count)
                self.assertFalse(session.finished)
                self.assertEqual(session.commits, 0)
                self.assertEqual(len(session.aborted), 1)
                self.assertTrue(all('Please revise' not in r['user'] for r in self.requests))
                self.assert_restored()

    async def test_start_and_terminal_failures_abort_without_commit(self):
        for failed_kind in ('task_start', 'task_end'):
            with self.subTest(failed_kind=failed_kind):
                session = Session('task-a', self.order, allow_write=True, fail=failed_kind)
                with self.assertRaisesRegex(ValueError, failed_kind):
                    await self.invoke(session=session)
                self.assertFalse(session.finished)
                self.assertEqual(session.commits, 0)
                self.assertEqual(len(session.aborted), 1)
                self.assert_restored()

    async def test_native_error_keeps_original_retry_and_exposes_only_real_observations(self):
        self.workflow.env.fail_native_once = True
        _, session, adapter = await self.invoke()
        self.assertEqual(adapter.attempts, 3)
        self.assertEqual(adapter.completed_steps, 2)
        before = [r['payload'] for r in session.trace if r['type'] == 'before_action']
        after = [r['payload'] for r in session.trace if r['type'] == 'after_observation']
        self.assertEqual([r['step'] for r in before], [0, 0, 1])
        self.assertEqual([r['attempt'] for r in before], [0, 1, 2])
        self.assertEqual([r['attempt'] for r in after], [1, 2])
        self.assertNotIn('previous_action_failure', before[0])
        self.assertEqual(before[1]['previous_action_failure'],
                         dict(attempt=0, action='act', reason='step_failed'))
        self.assertEqual(before[1]['visible'], before[0]['visible'])
        self.assertNotIn('previous_action_failure', before[2])
        self.assertNotIn('native rejected action', json.dumps(before))
        self.assertNotIn('Traceback', json.dumps(before))
        self.assertIn('native rejected action', self.requests[1]['user'])
        self.assertIn('Please revise', self.requests[1]['user'])
        self.assert_restored()

    async def test_overlapping_same_workflow_episodes_have_independent_context_and_slots(self):
        first, second = await asyncio.gather(self.invoke('alpha'), self.invoke('beta'))
        for (_, session, adapter), name in [(first, 'alpha'), (second, 'beta')]:
            self.assertEqual(session.finished[0]['task_id'], name)
            self.assertEqual(adapter.completed_steps, 2)
            self.assertEqual(len(adapter.injections), 2)
            for injection in adapter.injections:
                self.assertEqual(injection['task_id'], name)
                self.assertEqual(injection['memory']['entries'][0]['id'], name)
            for event in session.trace[:-1]:
                self.assertIn(name, event['payload']['visible']['obs'])
        self.assert_restored()

    async def test_cancelled_episode_aborts_and_restores_all_wrappers(self):
        started = asyncio.Event()
        original_ask = self.module.Agent.ask

        async def suspended_ask(instance, *args, **kwargs):
            started.set()
            await asyncio.Event().wait()
            return await original_ask(instance, *args, **kwargs)

        session = Session('task-a', self.order)
        with patch.object(self.module.Agent, 'ask', suspended_ask):
            running = asyncio.create_task(self.invoke(session=session))
            await started.wait()
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
        self.assertEqual(len(session.aborted), 1)
        self.assertFalse(session.finished)
        self.assert_restored()

    def staged_native(self, directory, *, allow_write=True, actor_ok=True, prior_actor=False):
        from evomem.common import append, digest
        from phase1_codegen.memory import CodeMemory
        from phase1_codegen.native import EpisodeMemory
        from phase1_codegen.program import Program
        from phase1_codegen.tests.test_engine import TinyTokenizer
        source = '''def on_event(state, event, context):
    if event["type"] == "task_end":
        state["tasks"] = state.get("tasks", 0) + 1
    return {"state": state, "session": {}, "memory": None}
'''
        bank = CodeMemory(Program(source), [], TinyTokenizer(), dict(environment='alfworld',
            memory_scope='environment', memory_hook_version='memory-events-v1'))
        ledger = Path(directory) / 'costs.jsonl'
        actor = dict(role='actor', task_id='task-a', phase='synthetic', ok=actor_ok,
            request_hash='synthetic-request', messages=[
                {'role': 'system', 'content': 'fixed system'}, {'role': 'user', 'content': 'visible observation'}])
        if prior_actor:
            append(ledger, actor)
        transaction = EpisodeMemory(self.workflow, bank, self.task(), 'alfworld', 'synthetic', 17,
                                    ledger, allow_write=allow_write)
        ep = dict(task_id='task-a', group_id='task-a', room='synthetic', family='synthetic',
            environment='alfworld', goal='visible goal', history=[], score=1., done=True,
            error=None, phase='synthetic', seed=17, native_reward_verified=True,
            memory_write_calls=int(allow_write), synthetic=True)
        transaction.hook.emit('task_start', {'step': 0})
        transaction.hook.finish(ep)
        transaction.adapter.injections.append(dict(task_id='task-a', attempt=0, step=0,
            memory={}, system_message='fixed system', system_hash=digest('fixed system'),
            original_user_prompt_hash=digest('visible observation'), event_sequence=0))
        append(ledger, actor)
        return bank, transaction, ep

    async def test_shared_native_finalizer_commits_only_after_delivery_and_ignores_old_requests(self):
        from evomem.common import events
        with tempfile.TemporaryDirectory() as directory:
            bank, transaction, ep = self.staged_native(directory, prior_actor=True)
            self.assertEqual(bank.state, {})
            self.assertFalse(transaction.hook.summary()['committed'])
            result = transaction.finalize(ep)
            self.assertIsNone(result['error'])
            self.assertEqual(bank.state, {'tasks': 1})
            self.assertEqual(result['generated_memory_write_calls'], 1)
            self.assertTrue(result['memory_hook']['actor_delivery_verified'])
            self.assertEqual(len(result['memory_hook']['injections']), 1)
            self.assertGreater(result['actor_ledger_start'], 0)
            self.assertTrue(result['memory_hook']['committed'])
            self.assertTrue(result['memory_hook']['closed'])
            self.assertEqual(len(events(Path(directory) / 'canonical_episodes.jsonl')), 1)
            trace = events(Path(directory) / 'code_memory_traces.jsonl')[0]
            self.assertTrue(trace['memory_hook']['committed'])

    async def test_native_post_loop_errors_and_unverified_actor_never_publish(self):
        for error, actor_ok, verified in [('CloseOSError', True, True),
                ('ModelRequestError', True, True), (None, False, True), (None, True, False)]:
            with self.subTest(error=error, actor_ok=actor_ok, verified=verified), tempfile.TemporaryDirectory() as directory:
                bank, transaction, ep = self.staged_native(directory, actor_ok=actor_ok)
                before = bank.export_snapshot()
                ep.update(error=error, native_reward_verified=verified)
                result = transaction.finalize(ep)
                self.assertTrue(result['error'])
                self.assertEqual(bank.export_snapshot(), before)
                self.assertFalse(result['memory_hook']['committed'])
                self.assertTrue(result['memory_hook']['closed'])
                self.assertEqual(sum(r['status'] == 'aborted' for r in transaction.hook.trace), 1)
                self.assertEqual(bank.write_trace, [])

    async def test_readonly_finalization_discards_terminal_mutation_and_records_events(self):
        from evomem.common import events
        with tempfile.TemporaryDirectory() as directory:
            bank, transaction, ep = self.staged_native(directory, allow_write=False)
            before = bank.export_snapshot()
            result = transaction.finalize(ep)
            self.assertIsNone(result['error'])
            self.assertEqual(bank.export_snapshot(), before)
            self.assertFalse(result['memory_hook']['committed'])
            self.assertTrue(result['memory_hook']['closed'])
            trace = events(Path(directory) / 'code_memory_traces.jsonl')[0]
            self.assertEqual([r['event']['type'] for r in trace['memory_hook']['events']],
                             ['task_start', 'task_end'])

    async def test_commit_validation_failures_leave_bank_and_receipts_unchanged(self):
        from phase1_codegen.policy import Rejected
        with tempfile.TemporaryDirectory() as directory:
            bank, transaction, ep = self.staged_native(directory)
            before = bank.export_snapshot()
            for state, native, expected in [({'x': 'x' * 20000}, ep, bank.fingerprint),
                    ({'x': 1}, dict(ep, environment='textworld'), bank.fingerprint),
                    ({'x': 1}, ep, 'wrong-hash')]:
                with self.assertRaises(Rejected):
                    bank.commit_state(state, native, expected, {})
                self.assertEqual(bank.export_snapshot(), before)
                self.assertEqual(bank.write_trace, [])
            transaction.hook.close()


if __name__ == '__main__':
    unittest.main()
