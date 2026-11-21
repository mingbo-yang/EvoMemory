"""Isolated CPU event-program contracts, never task-performance evidence."""
import asyncio
import copy
import json
import sys
import types
import unittest
from unittest.mock import patch

from phase1_codegen.memory import CodeMemory
from phase1_codegen.policy import Rejected
from phase1_codegen.program import BASELINE_SOURCE, Program


class CharacterTokenizer:
    def count(self, text):
        return len(text)

    def clip(self, text, count):
        return text[:max(0, count)]


def fixture(task='room-a', action='take apple 1 from table 1', score=1):
    return dict(task_id=task, group_id=task, family='synthetic-clean', room='synthetic',
                environment='alfworld', goal='clean apple',
                history=[dict(observation='apple on table', action=action, next_observation='apple carried')],
                score=score, done=bool(score), error=None, phase='synthetic', seed=42)


def event_source(ingest='pass', retrieve='memory = []'):
    """Build explicit on_event fixtures; there are no legacy generated entries."""
    indent = lambda body: '\n'.join('        ' + line for line in body.splitlines())
    return ('def on_event(state, event, context):\n    memory = None\n'
        '    if event["type"] == "task_end":\n        experience = event["experience"]\n' + indent(ingest) +
        '\n    elif event["type"] == "task_start":\n        visible = event.get("visible", {})\n' + indent(retrieve) +
        '\n    return {"state": state, "session": event["session"], "memory": memory}\n')


NOVEL_SOURCE = '''def verb(action):
    words = action.lower().split()
    return words[0] if words else "empty"

''' + event_source('''graph = state.get("transition_graph", {})
failures = state.get("failure_counts", {})
for step in experience.get("history", []):
    key = verb(step["action"])
    graph.setdefault(key, []).append(step["next_observation"])
    if experience["score"] < 1:
        failures[key] = failures.get(key, 0) + 1
state = {"transition_graph": graph, "failure_counts": failures}''', '''memory = []
for key in sorted(state.get("transition_graph", {}).keys()):
    if key in visible.get("obs", ""):
        text = key + ": " + " | ".join(state["transition_graph"][key])
        memory.append({"id": key, "experience": text + "; failure_count=" + str(state["failure_counts"].get(key, 0))})''')


class ProgramMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.context = dict(environment='alfworld', family='synthetic-clean')

    def memory(self, source, logs=None, **kwargs):
        return CodeMemory(Program(source), logs or [], self.tokenizer, self.context, **kwargs)

    def test_novel_state_and_algorithm_execute_under_kernel_filter(self):
        logs = [fixture(), fixture('room-b', score=0)]
        original = copy.deepcopy(logs)
        baseline, custom = self.memory(BASELINE_SOURCE, logs), self.memory(NOVEL_SOURCE, logs)
        self.assertEqual(set(custom.state), {'transition_graph', 'failure_counts'})
        self.assertEqual(custom.state['failure_counts'], {'take': 1})
        output = custom.retrieve('clean apple', 'take apple now')
        self.assertEqual(output, [{'id': 'take', 'experience': 'take: apple carried | apple carried; failure_count=1'}])
        self.assertNotEqual(output, baseline.retrieve('clean apple', 'take apple now'))
        self.assertTrue(custom.read_trace[-1]['isolation']['seccomp'])
        self.assertTrue(custom.read_trace[-1]['isolation']['no_new_privs'])
        self.assertEqual(logs, original)

    def test_read_mutation_and_state_snapshot_cannot_change_committed_bank(self):
        source = event_source('state = {"items": [experience["task_id"]]}',
            'state["items"].append("worker mutation")\nmemory = [{"id": "snapshot", "experience": str(len(state["items"]))}]')
        memory = self.memory(source, [fixture()])
        before = memory.fingerprint
        snapshot = memory.state
        snapshot['items'].append('caller mutation')
        self.assertEqual(memory.read({})[0]['experience'], '2')
        self.assertEqual(memory.read({})[0]['experience'], '2')
        self.assertEqual(memory.state, {'items': ['room-a']})
        self.assertEqual(memory.fingerprint, before)

    def test_invalid_interfaces_imports_attributes_and_recursion_rejected(self):
        cases = [BASELINE_SOURCE.replace('on_event(state, event, context)', 'on_event(state, event)'),
            'import os\n' + BASELINE_SOURCE,
            BASELINE_SOURCE + '\ndef unsafe(state):\n    return state.__class__\n',
            'def on_event(state, event, context):\n    return on_event(state, event, context)\n',
            'def on_event(state, event, context):\n    return helper(state)\ndef helper(state):\n    return other(state)\ndef other(state):\n    return helper(state)\n',
            'def on_event(state, event, context):\n    fn = on_event\n    return state\n',
            'def on_event(state, event, context):\n    return {"x": open("/etc/passwd")}\n']
        for source in cases:
            with self.subTest(source=source[:80]), self.assertRaises((Rejected, SyntaxError)):
                Program(source)

    def test_failed_write_and_capacity_rejection_leave_bank_unchanged(self):
        source = event_source('''state["transient"] = True
if experience["task_id"] == "oversize":
    state = {"text": "x" * 17000}
elif experience["task_id"] == "crash":
    state = {"bad": 1 / 0}
else:
    state = {"task": experience["task_id"]}''')
        memory = self.memory(source, [fixture()])
        before = memory.fingerprint
        for name in ('oversize', 'crash'):
            with self.subTest(name=name), self.assertRaises(Rejected):
                memory.write(fixture(name))
            self.assertEqual(memory.fingerprint, before)
            self.assertEqual(memory.state, {'task': 'room-a'})
        self.assertEqual(memory.stats['rejected_writes'], 2)

    def test_fixed_adapter_budgets_bound_generated_output(self):
        memory = self.memory(event_source(retrieve='memory = [{"id": str(i), "experience": "long " * 1000} for i in range(6)]'))
        entries = memory.read({})
        self.assertLessEqual(len(entries), 4)
        self.assertLessEqual(self.tokenizer.count(json.dumps({'entries': entries}, ensure_ascii=False, indent=2)), 2048)
        self.assertEqual(len(memory.read_trace[-1]['raw_readout']), 6)

    def test_family_guard_sanitization_and_native_error_handling(self):
        logs = [fixture()]
        logs[0]['native_recorder'] = {'memory_retrieved': {'foreign': 'forbidden'}}
        memory = self.memory(event_source('state = {"keys": sorted(experience.keys())}'), logs)
        self.assertNotIn('native_recorder', memory.state['keys'])
        foreign = dict(fixture('foreign'), family='different-family')
        with self.assertRaises(Rejected):
            memory.write(foreign)
        with self.assertRaises(Rejected):
            self.memory(BASELINE_SOURCE, logs + [foreign])
        before = memory.fingerprint
        memory.write(dict(fixture('native-error'), error='EnvironmentError'))
        self.assertEqual(memory.fingerprint, before)
        self.assertEqual(memory.stats['skipped_error_episodes'], 1)

    def test_invalid_json_output_and_iteration_exhaustion_rejected(self):
        for body in ['memory = [{"id": "one", "experience": 4}]',
                     'memory = [{"id": "one", "experience": "a"}, {"id": "one", "experience": "b"}]',
                     'total = 0\nfor i in range(100001):\n    total += i']:
            with self.subTest(body=body), self.assertRaises(Rejected):
                Program(event_source(retrieve=body)).run('event', {}, {'type': 'task_start', 'session': {}}, {})

    def test_ast_identity_and_process_determinism(self):
        program, commented = Program(BASELINE_SOURCE), Program('# comment\n' + BASELINE_SOURCE)
        self.assertNotEqual(program.hash, commented.hash)
        self.assertEqual(program.identity_hash, commented.identity_hash)
        source = event_source(retrieve='words = set(visible["obs"].split())\nmemory = [{"id": word, "experience": word} for word in words]')
        program = Program(source)
        event = dict(type='task_start', session={}, visible={'obs': 'take clean move put heat cool'})
        values = [program.run('event', {}, event, {})['value'] for _ in range(3)]
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])

    def test_native_bridge_records_intent_only_inside_active_event_session(self):
        from evomem.alma_backend import EvolvableMemoryStore
        from phase1_codegen.native import code_bridge_class
        from phase1_codegen.hook_adapter import active_hook
        from phase1_codegen.hooks import HookSession
        module = types.ModuleType('agents.memo_structure')
        module.MemoStructure = type('StubMemoStructure', (), {})
        with patch.dict(sys.modules, {'agents.memo_structure': module}):
            Bridge = code_bridge_class('alfworld')
        memory = self.memory(BASELINE_SOURCE)
        store = EvolvableMemoryStore(memory, self.tokenizer, allow_write=True)
        task = dict(id='synthetic-task', group_id='synthetic-group', room='synthetic', family='synthetic-clean')
        recorder = types.SimpleNamespace(init={'obs': 'apple on table', 'goal': 'clean apple'},
            steps=[{'action_took': 'take apple', 'obs': 'apple carried', 'dones': True}], reward=1)
        bridge = Bridge(store, task, 'synthetic', 42)
        with self.assertRaises(Rejected):
            asyncio.run(bridge.general_update(recorder))
        session = HookSession(memory, task, True)
        token = active_hook.set(session)
        try:
            asyncio.run(bridge.general_retrieve(recorder))
            asyncio.run(bridge.general_update(recorder))
        finally:
            active_hook.reset(token)
        self.assertEqual(len(store.writes), 1)
        self.assertEqual(memory.state, {})
        session.finish(store.writes[0])
        self.assertEqual(memory.state, {})
        session.publish()
        self.assertEqual(memory.stats['write_calls'], 1)
        self.assertEqual(len(memory.state['episodes']), 1)
        self.assertEqual(store.reads, [])


if __name__ == '__main__':
    unittest.main()
