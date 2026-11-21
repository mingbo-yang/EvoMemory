"""Synthetic CPU contracts for within-dataset shared Task Memory.

No native task outcome, model call, or GPU work is performed by these tests.
"""
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from evomem.alma_backend import ALMARunner, EvolvableMemoryStore
from phase1_codegen.memory import CodeMemory
from phase1_codegen.native import CodeALMARunner, code_bridge_class
from phase1_codegen.policy import Rejected
from phase1_codegen.program import BASELINE_SOURCE, Program


class Tokenizer:
    def count(self, text):
        return len(text)

    def clip(self, text, count):
        return text[:max(0, count)]


SOURCE = '''def on_event(state, event, context):
    memory = None
    if event["type"] == "task_end":
        experience = event["experience"]
        rows = state.get("rows", [])
        rows.append({"id": experience["task_id"], "family": context["family"],
                     "environment": context["environment"],
                     "experience_environment": experience["environment"]})
        state = {"rows": rows}
    elif event["type"] == "task_start":
        memory = [{"id": context.get("task_id", "query"),
                   "experience": context["environment"] + "/" + context["family"]
                   + ":" + str(len(state.get("rows", [])))}]
    return {"state": state, "session": event["session"], "memory": memory}
'''



def episode(task_id, family='heat', environment='alfworld'):
    return dict(task_id=task_id, group_id=task_id, room='synthetic', family=family,
                environment=environment, goal='synthetic goal',
                history=[dict(observation='before', action='look', next_observation='after')],
                score=0., done=False, error=None, phase='synthetic', seed=42)


class SharedMemoryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.program = Program(SOURCE)
        self.context = dict(memory_scope='environment', environment='alfworld', family='cool')

    def test_same_environment_multiple_families_share_state_and_current_write_context(self):
        logs = [episode('source', 'heat'), episode('target', 'cool')]
        before = copy.deepcopy(logs)
        memory = CodeMemory(self.program, logs, self.tokenizer, self.context)
        self.assertEqual([row['family'] for row in memory.state['rows']], ['heat', 'cool'])
        self.assertEqual(memory.families, {'heat', 'cool'})
        self.assertIsNone(memory.family)
        self.assertEqual([row['context']['family'] for row in memory.write_trace], ['heat', 'cool'])
        self.assertEqual([row['context']['task_id'] for row in memory.write_trace], ['source', 'target'])
        self.assertTrue(all(row['experience_environment'] == 'alfworld' for row in memory.state['rows']))
        self.assertEqual(logs, before)

    def test_missing_and_foreign_environment_fail_closed(self):
        missing = episode('missing')
        missing.pop('environment')
        for logs in ([missing], [episode('foreign', environment='textworld')],
                     [episode('source'), episode('foreign', environment='textworld')]):
            with self.subTest(logs=logs), self.assertRaises(Rejected):
                CodeMemory(self.program, logs, self.tokenizer, self.context)
        with self.assertRaises(Rejected):
            CodeMemory(self.program, [], self.tokenizer, {'memory_scope': 'environment'})
        memory = CodeMemory(self.program, [episode('source')], self.tokenizer, self.context)
        fingerprint = memory.fingerprint
        with self.assertRaises(Rejected):
            memory.write(episode('foreign', environment='textworld'))
        with self.assertRaises(Rejected):
            memory.read({'environment': 'textworld', 'family': 'treasure_hunter'})
        with self.assertRaises(Rejected):
            memory.for_task({'id': 'foreign', 'family': 'cool'}, 'textworld')
        with self.assertRaises(Rejected):
            memory.for_task({'id': 'foreign', 'family': 'cool', 'dataset': 'textworld'}, 'alfworld')
        self.assertEqual(memory.fingerprint, fingerprint)

    def test_current_task_view_and_read_only_bank_hash(self):
        memory = CodeMemory(self.program, [episode('source', 'heat')], self.tokenizer,
                            dict(self.context, family='heat'))
        before = memory.fingerprint
        task = dict(id='target-query', family='cool')
        view = memory.for_task(task, 'alfworld')
        self.assertEqual(view.context['family'], 'cool')
        store = EvolvableMemoryStore(view, self.tokenizer, allow_write=False)
        result = store.read({'goal': 'target goal', 'obs': 'target observation'})
        self.assertEqual(result['entries'], [{'id': 'target-query', 'experience': 'alfworld/cool:1'}])
        self.assertEqual(memory.read_trace[-1]['context']['family'], 'cool')
        self.assertEqual(memory.read_trace[-1]['context']['task_id'], task['id'])
        self.assertEqual(memory.context['family'], 'heat')
        self.assertEqual(memory.fingerprint, before)
        with self.assertRaises(ValueError):
            store.write(episode('forbidden', 'cool'))
        self.assertEqual(memory.fingerprint, before)

    def test_source_prefix_online_extension_matches_replay_of_shared_history(self):
        source = [episode('source-a', 'heat'), episode('source-b', 'heat')]
        target = [episode('target-a', 'cool'), episode('target-b', 'cool')]
        inherited = CodeMemory(self.program, source, self.tokenizer, self.context)
        prefix_state = inherited.state
        for ep in target:
            inherited.write(ep)
        replayed = CodeMemory(self.program, source + target, self.tokenizer, self.context)
        self.assertEqual(inherited.state, replayed.state)
        self.assertEqual(inherited.fingerprint, replayed.fingerprint)
        self.assertEqual(inherited.state['rows'][:2], prefix_state['rows'])

    def test_explicit_family_scope_still_guards_bank_provenance(self):
        source = episode('source', 'heat')
        memory = CodeMemory(Program(BASELINE_SOURCE), [source], self.tokenizer, {'environment': 'alfworld'})
        self.assertEqual(memory.memory_scope, 'family')
        self.assertEqual(memory.family, 'heat')
        with self.assertRaisesRegex(Rejected, 'task families'):
            memory.write(episode('target', 'cool'))

    def test_native_bridge_attaches_fixed_environment_before_generated_write(self):
        stub = types.ModuleType('agents.memo_structure')
        stub.MemoStructure = type('StubMemoStructure', (), {})
        with patch.dict(sys.modules, {'agents.memo_structure': stub}):
            Bridge = code_bridge_class('textworld')
        context = dict(memory_scope='environment', environment='textworld', family='treasure_hunter')
        memory = CodeMemory(self.program, [], self.tokenizer, context)
        task = dict(id='cooking-source', group_id='cooking-source', room='synthetic',
                    family='the_cooking_game', dataset='textworld')
        view = memory.for_task(task, 'textworld')
        store = EvolvableMemoryStore(view, self.tokenizer, allow_write=True)
        recorder = types.SimpleNamespace(init={'obs': 'native observation', 'goal': 'native goal'},
            steps=[{'action_took': 'look', 'obs': 'native result', 'dones': False}], reward=0.)
        bridge = Bridge(store, task, 'synthetic', 42)
        from phase1_codegen.hooks import HookSession
        from phase1_codegen.hook_adapter import active_hook
        session = HookSession(view, task, True)
        token = active_hook.set(session)
        try:
            asyncio.run(bridge.general_retrieve(recorder))
            asyncio.run(bridge.general_update(recorder))
        finally:
            active_hook.reset(token)
        self.assertEqual(memory.state, {})
        session.finish(store.writes[0])
        session.publish()
        self.assertEqual(store.writes[0]['environment'], 'textworld')
        self.assertEqual(memory.state['rows'][0]['family'], 'the_cooking_game')
        self.assertEqual(memory.state['rows'][0]['environment'], 'textworld')
        self.assertEqual(session.context['family'], 'the_cooking_game')
        self.assertEqual(len(memory.write_trace), 1)
        foreign_task = dict(task, environment='alfworld')
        with self.assertRaises(Rejected):
            asyncio.run(Bridge(store, foreign_task, 'synthetic', 42).general_update(recorder))
        self.assertEqual(len(memory.write_trace), 1)

    def test_alfworld_canonical_episode_has_adapter_environment_provenance(self):
        async def synthetic_parent(self, task, memory, phase, seed, no_memory=False, allow_write=False):
            return dict(task_id=task['id'], phase=phase, score=0., history=[], error=None,
                        synthetic=True, native_recorder={'memory_retrieved': {}})
        with tempfile.TemporaryDirectory() as directory:
            runner = CodeALMARunner.__new__(CodeALMARunner)
            runner.workflow = types.SimpleNamespace()
            runner.layout = Path(directory)
            runner.ledger = Path(directory) / 'costs.jsonl'
            runner.canonical_episodes = Path(directory) / 'canonical_episodes.jsonl'
            runner.trace_path = Path(directory) / 'code_memory_traces.jsonl'
            memory = CodeMemory(self.program, [], self.tokenizer, self.context)
            task = dict(id='alfworld-synthetic', family='heat')
            before = memory.fingerprint
            with patch.object(ALMARunner, '_episode', synthetic_parent):
                result = asyncio.run(runner._episode(task, memory, 'synthetic', 42, no_memory=True))
            self.assertEqual(result['environment'], 'alfworld')
            self.assertEqual(json.loads(runner.canonical_episodes.read_text())['environment'], 'alfworld')
            self.assertEqual(memory.fingerprint, before)


if __name__ == '__main__':
    unittest.main()
