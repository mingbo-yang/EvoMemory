"""Synthetic CPU contracts; no model or native-task performance claims."""
import copy
import json
import unittest
from unittest.mock import patch

from phase1_codegen.behavior_diagnostics import TRACE_BYTES, diagnose
from phase1_codegen.memory import CodeMemory, digest
from phase1_codegen.policy import Rejected
from phase1_codegen.program import Program
from phase1_codegen.tests.test_program_capabilities import FakeBroker
from phase1_codegen.tests.test_program_memory import CharacterTokenizer, fixture


def event_source(helpers):
    """Synthetic algorithms exposed through the sole generated event interface."""
    return helpers + '\n' + """def on_event(state, event, context):
    memory = None
    if event["type"] == "task_end":
        state = update_state(state, event["experience"], context)
    else:
        memory = select_memory(state, context)
    return {"state": state, "session": event.get("session", {}), "memory": memory}
"""


def invoke(program, kind, state, *, experience=None, **kwargs):
    return program.run('event', state, experience=dict(type=kind, session={},
                       experience=experience or {}), **kwargs)


SOURCE = '''def count(state, key):
    state[key] = state.get(key, 0) + 1
    return state[key]

def extract(text):
    pieces = []
    if "marker:" in text:
        pieces = text.split("marker:")[1:]
    return pieces

def update_state(state, experience, context):
    rows = state.get("rows", [])
    rows.append({"id": experience["task_id"], "text": experience["goal"]})
    return {"rows": rows}

def select_memory(state, context):
    counter = {}
    selected = []
    if False and count(counter, "wrong"):
        selected.append("unreachable")
    value = count(counter, "once") if True else count(counter, "wrong")
    ranks = []
    for index, row in enumerate(state.get("rows", [])):
        weight = row.get("weight", 0)
        ranks.append([weight, index])
    discarded = ranks.sort(reverse=True)
    order = sorted(ranks)
    parts = extract(context.get("observation", ""))
    answer = []
    for pair in order:
        row = state["rows"][pair[1]]
        answer.append({"id": row["id"], "experience": row["text"] + str(value) + str(parts) + str(discarded)})
    return answer
'''


class BehaviorDiagnosticTests(unittest.TestCase):
    def test_instrumentation_preserves_outputs_side_effect_count_short_circuit_and_sort_none(self):
        program = Program(event_source(SOURCE))
        state = {'rows': [{'id': 'b', 'text': 'B'}, {'id': 'a', 'text': 'A'}]}
        before = copy.deepcopy(state)
        plain = invoke(program, 'task_start', state, context={'observation': 'nothing'})
        traced = invoke(program, 'task_start', state, context={'observation': 'nothing'}, diagnostics=True)
        self.assertEqual(plain['value'], traced['value'])
        self.assertEqual([row['experience'] for row in traced['value']['memory']], ['B1[]None', 'A1[]None'])
        self.assertEqual(state, before)
        self.assertEqual(plain['isolation']['iterations_used'], traced['isolation']['iterations_used'])
        self.assertNotIn('execution_trace', plain)
        trace = traced['execution_trace']
        events = trace['events']
        missing = next(event for event in events if event['expression'] == 'row.get("weight", 0)')
        self.assertEqual(missing['missing_count'], 2)
        self.assertEqual(missing['samples'][0]['key']['value'], 'weight')
        self.assertFalse(missing['samples'][0]['key_present'])
        self.assertEqual(missing['samples'][0]['result']['value'], 0)
        inplace = next(event for event in events if event['kind'] == 'inplace_sort')
        self.assertIsNone(inplace['samples'][0]['result']['value'])
        self.assertTrue(inplace['samples'][0]['reverse']['value'])
        empty = next(event for event in events if event['function'] == 'extract' and event['kind'] == 'return')
        self.assertEqual(empty['empty_count'], 1)
        self.assertEqual(empty['samples'][0]['value']['length'], 0)

    def test_exceptions_and_argument_order_are_unchanged(self):
        prefix = 'def update_state(state, experience, context):\n    return state\n\ndef select_memory(state, context):\n'
        for expression in ('{}["absent"]', '{}.get([])', '[1].sort(reverse="bad")', '"text".get(1 / 0)'):
            errors = []
            for enabled in (False, True):
                with self.subTest(expression=expression, enabled=enabled), self.assertRaises(Rejected) as caught:
                    invoke(Program(event_source(prefix + '    value = ' + expression + '\n    return []\n')),
                        'task_start', {}, diagnostics=enabled)
                errors.append(str(caught.exception))
            self.assertEqual(errors[0], errors[1])

    def test_tracing_does_not_repeat_model_calls_or_disclose_broker_configuration(self):
        source = '''def update_state(state, experience, context):
    value = model_call("test-model", {"text": experience.get("goal", "")})
    second = model_call("test-model", {"text": experience.get("goal", "")})
    return {"value": value, "second": second}

def select_memory(state, context):
    return []
'''
        receipts = []
        for enabled in (False, True):
            broker = FakeBroker({'text': 'synthetic output'})
            broker.secret_config = 'never put configuration in execution trace'
            receipts.append(invoke(Program(event_source(source)), 'task_end', {}, experience={'goal': 'query'},
                                                capabilities=broker, diagnostics=enabled))
            self.assertEqual(len(broker.calls), 1)
            self.assertEqual(len(receipts[-1]['capability_calls']), 2)
            self.assertEqual(receipts[-1]['worker_invocations'], 2)
        self.assertEqual(receipts[0]['value'], receipts[1]['value'])
        self.assertNotIn('never put configuration', json.dumps(receipts[1]))

    def test_counters_and_sampling_limits_are_disclosed(self):
        source = '''def update_state(state, experience, context):
    return state

def select_memory(state, context):
    total = 0
    for number in range(600):
        total = total + state.get("unavailable", 0)
    return []
'''
        trace = invoke(Program(event_source(source)), 'task_start', {}, diagnostics=True)['execution_trace']
        event = next(item for item in trace['events'] if item['kind'] == 'dict_get')
        self.assertEqual(event['missing_count'], 600)
        self.assertEqual(len(event['samples']), 2)
        self.assertGreater(trace['coverage']['samples_omitted'], 0)
        self.assertLessEqual(len(json.dumps(trace, ensure_ascii=False, separators=(',', ':')).encode()), TRACE_BYTES)

    def test_memory_trace_aggregator_is_observational_and_requires_no_execution(self):
        logs = [dict(fixture('one'), environment='alfworld'), dict(fixture('two'), environment='alfworld')]
        original = copy.deepcopy(logs)
        context = dict(environment='alfworld', memory_scope='environment')
        memory = CodeMemory(Program(event_source(SOURCE)), logs, CharacterTokenizer(), context, diagnostics=True)
        fingerprint = memory.fingerprint
        memory.for_task(dict(id='query', family='new-type'), 'alfworld').retrieve('goal', 'nothing')
        behavior = dict(stored_state=memory.state, memory_hash=memory.fingerprint,
                        write_trace=memory.write_trace, read_trace=memory.read_trace)
        with patch.object(Program, 'run', side_effect=AssertionError('diagnose must not execute')):
            report = diagnose(memory.program, behavior, logs)
        self.assertEqual(report['provenance'], dict(program_hash=memory.program.hash,
                         environment='alfworld', phase='offline_replay', snapshot_hash=fingerprint))
        self.assertTrue(report['facts']['missing_dict_keys'])
        self.assertTrue(report['facts']['sorting'])
        self.assertTrue(report['facts']['empty_returns'])
        self.assertEqual(report['facts']['adapter_counts'][0]['task_id'], 'query')
        self.assertEqual(memory.fingerprint, fingerprint)
        self.assertEqual(logs, original)
        no_trace = diagnose(memory.program, {'read_trace': [{}]}, logs)
        self.assertEqual(no_trace['coverage']['missing_trace_invocations'], 1)
        self.assertEqual(no_trace['facts']['missing_dict_keys'], [])
        with self.assertRaisesRegex(ValueError, 'mix environments'):
            diagnose(memory.program, behavior, logs + [dict(environment='textworld')])

    def test_generated_source_cannot_access_trusted_hooks(self):
        with self.assertRaises(Rejected):
            Program(event_source(SOURCE.replace('return answer', 'return _diag_locals()')))

    def test_loop_trace_never_prefetches_a_generator_after_break(self):
        source = '''def count(state):
    state["count"] = state.get("count", 0) + 1
    return state["count"]

def update_state(state, experience, context):
    return state

def select_memory(state, context):
    counter = {}
    for value in (count(counter) for index in range(10)):
        break
    return [{"id": "count", "experience": str(counter["count"])}]
'''
        results = [invoke(Program(event_source(source)), 'task_start', {}, diagnostics=enabled) for enabled in (False, True)]
        self.assertEqual(results[0]['value'], results[1]['value'])
        self.assertEqual(results[1]['value']['memory'][0]['experience'], '1')
        self.assertEqual(results[0]['isolation']['iterations_used'], results[1]['isolation']['iterations_used'])

    def test_near_limit_valid_output_does_not_fail_due_to_trace_bytes(self):
        source = 'def update_state(state, experience, context):\n    return state\n\ndef select_memory(state, context):\n'
        source += ''.join('    value' + str(index) + ' = "sample"\n' for index in range(50))
        source += '    return [{"id": "large", "experience": "x" * 518000}]\n'
        program = Program(event_source(source))
        plain = invoke(program, 'task_start', {})
        traced = invoke(program, 'task_start', {}, diagnostics=True)
        self.assertEqual(plain['value'], traced['value'])
        self.assertGreater(traced['execution_trace']['coverage']['export_sites_omitted'], 0)

    def test_diagnostic_resource_failure_keeps_plain_valid_result_and_single_model_bill(self):
        from phase1_codegen.program import _invoke_worker
        source = '''def update_state(state, experience, context):
    return {"value": model_call("test-model", {"text": experience.get("goal", "")})}

def select_memory(state, context):
    return []
'''
        broker = FakeBroker({'text': 'the actual model result'})
        def fail_only_instrumented(request, timeout):
            if request.get('diagnostics'):
                raise Rejected('worker exceeded cumulative one-second CPU budget')
            return _invoke_worker(request, timeout)
        with patch('phase1_codegen.program._invoke_worker', side_effect=fail_only_instrumented):
            result = invoke(Program(event_source(source)), 'task_end', {}, experience={'goal': 'query'},
                                         capabilities=broker, diagnostics=True)
        self.assertEqual(result['value']['state'], {'value': {'text': 'the actual model result'}})
        self.assertEqual(len(broker.calls), 1)
        self.assertEqual(len(result['capability_calls']), 1)
        self.assertIsNone(result['execution_trace'])
        self.assertEqual(result['diagnostics']['status'], 'unavailable')
        self.assertEqual(result['diagnostics']['new_model_calls'], 0)
        self.assertIn('CPU budget', result['diagnostics']['error'])

    def test_diagnostic_failure_does_not_reject_bank_and_is_visible_in_coverage(self):
        from phase1_codegen.program import _invoke_worker
        def fail_only_instrumented(request, timeout):
            if request.get('diagnostics'):
                raise Rejected('diagnostic-only wall limit')
            return _invoke_worker(request, timeout)
        logs = [dict(fixture(), environment='alfworld')]
        with patch('phase1_codegen.program._invoke_worker', side_effect=fail_only_instrumented):
            memory = CodeMemory(Program(event_source(SOURCE)), logs, CharacterTokenizer(),
                                dict(environment='alfworld', memory_scope='environment'), diagnostics=True)
            result = memory.read({'observation': 'nothing'})
        self.assertEqual(memory.stats['rejected_writes'], 0)
        self.assertTrue(result)
        report = diagnose(memory.program, dict(memory_hash=memory.fingerprint, stored_state=memory.state,
                          read_trace=memory.read_trace, write_trace=memory.write_trace), logs)
        self.assertEqual(len(report['coverage']['diagnostic_failures']), 2)
        self.assertEqual(report['coverage']['instrumented_invocations'], 0)
        self.assertEqual(report['coverage']['missing_trace_invocations'], 2)

    def test_instrumented_replay_cannot_make_new_model_requests(self):
        original = Program._run
        def altered(self, operation, state, experience=None, context=None, **kwargs):
            if kwargs.get('diagnostics'):
                kwargs['capabilities'].call('unrecorded', {}, operation=operation)
            return original(self, operation, state, experience, context, **kwargs)
        broker = FakeBroker()
        with patch.object(Program, '_run', altered):
            result = invoke(Program(event_source(SOURCE)), 'task_start', {'rows': []}, context={}, capabilities=broker, diagnostics=True)
        self.assertEqual(result['value']['memory'], [])
        self.assertEqual(broker.calls, [])
        self.assertEqual(result['diagnostics']['status'], 'unavailable')
        self.assertIn('unrecorded model result', result['diagnostics']['error'])


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.program = Program(event_source(SOURCE))
        self.context = dict(environment='alfworld', memory_scope='environment')
        self.logs = [dict(fixture('a'), environment='alfworld'),
                     dict(fixture('b'), environment='alfworld', family='other-type')]

    @staticmethod
    def rehash(snapshot):
        snapshot['snapshot_hash'] = digest({key: value for key, value in snapshot.items() if key != 'snapshot_hash'})
        return snapshot

    def test_restore_preserves_actual_order_state_and_calls_nothing(self):
        memory = CodeMemory(self.program, self.logs, self.tokenizer, self.context)
        snapshot = memory.export_snapshot()
        with patch.object(Program, 'run', side_effect=AssertionError('restore must never replay')):
            restored = CodeMemory.from_snapshot(self.program, snapshot, self.tokenizer)
        self.assertEqual(restored.export_snapshot(), snapshot)
        self.assertEqual(restored.state, memory.state)
        self.assertEqual(restored.fingerprint, memory.fingerprint)
        self.assertEqual(restored.stats['write_calls'], 0)
        self.assertEqual(restored.write_trace, [])
        self.assertEqual(restored.read_trace, [])
        self.assertEqual(restored.families, {'synthetic-clean', 'other-type'})
        snapshot['state_wire'] = '{}'
        self.assertNotEqual(restored.state, {})

    def test_snapshot_tampering_program_context_budgets_and_schema_fail_closed(self):
        memory = CodeMemory(self.program, self.logs, self.tokenizer, self.context)
        snapshot = memory.export_snapshot()
        cases = [dict(snapshot, memory_hash='0' * 64),
                 self.rehash(dict(snapshot, capacity=99999)),
                 self.rehash(dict(snapshot, state_wire='[]')),
                 self.rehash(dict(snapshot, state_wire='{"a":1,"a":2}')),
                 self.rehash(dict(snapshot, environment='textworld')),
                 self.rehash(dict(snapshot, stored_tokens=0)),
                 self.rehash(dict(snapshot, family='mislabelled')),
                 self.rehash(dict(snapshot, families=['x', 'x']))]
        for altered in cases:
            with self.subTest(altered=altered), self.assertRaises(Rejected):
                CodeMemory.from_snapshot(self.program, altered, self.tokenizer)
        with self.assertRaisesRegex(Rejected, 'identical program'):
            CodeMemory.from_snapshot(Program(event_source('# edited\n' + SOURCE)), snapshot, self.tokenizer)
        with self.assertRaisesRegex(Rejected, 'context mismatch'):
            CodeMemory.from_snapshot(self.program, snapshot, self.tokenizer,
                                     dict(environment='textworld', memory_scope='environment'))

    def test_actual_model_state_restored_without_new_generation(self):
        class Broker(FakeBroker):
            environment = 'alfworld'
            def describe(self):
                return [{'id': 'test-model', 'kind': 'synthetic'}]
        source = '''def update_state(state, experience, context):
    return {"output": model_call("test-model", {"goal": experience.get("goal", "")})}

def select_memory(state, context):
    return []
'''
        broker = Broker({'actual_state': 'first generation'})
        program = Program(event_source(source))
        memory = CodeMemory(program, self.logs[:1], self.tokenizer, self.context, capabilities=broker)
        snapshot = memory.export_snapshot()
        broker.value = {'actual_state': 'a replay would differ'}
        restored = CodeMemory.from_snapshot(program, snapshot, self.tokenizer, capabilities=broker)
        self.assertEqual(len(broker.calls), 1)
        self.assertEqual(restored.state, {'output': {'actual_state': 'first generation'}})
        self.assertEqual(restored.fingerprint, memory.fingerprint)
        self.assertEqual(restored.read_trace, [])

    def test_snapshot_preserves_dictionary_iteration_order(self):
        source = '''def update_state(state, experience, context):
    return {"z": "last alphabetically", "a": "first alphabetically"}

def select_memory(state, context):
    return [{"id": key, "experience": state[key]} for key in state]
'''
        program = Program(event_source(source))
        memory = CodeMemory(program, self.logs[:1], self.tokenizer, self.context)
        snapshot = memory.export_snapshot()
        restored = CodeMemory.from_snapshot(program, snapshot, self.tokenizer)
        self.assertEqual(list(restored.state), ['z', 'a'])
        self.assertEqual([row['id'] for row in restored.read({})], ['z', 'a'])
        changed = dict(snapshot, state_wire=json.dumps(memory.state, sort_keys=True, separators=(',', ':')))
        with self.assertRaisesRegex(Rejected, 'state binding mismatch'):
            CodeMemory.from_snapshot(program, self.rehash(changed), self.tokenizer)


if __name__ == '__main__':
    unittest.main()
