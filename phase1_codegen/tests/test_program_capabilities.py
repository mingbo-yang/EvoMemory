"""CPU-only trusted capability suspension/replay; all models are fake brokers."""
import copy
import unittest
from unittest.mock import patch

from phase1_codegen.policy import (
    CAPABILITY_JSON_BYTES, ITERATION_BUDGET, CapabilityInfrastructureError, Rejected, capability_request,
)
from phase1_codegen.program import Program


EMPTY = 'def on_event(state, event, context):\n    return {"state": state, "session": {}, "memory": None}\n'


def event_program(body):
    return ('def compute(state, event, context):\n'
            + ''.join('    ' + line + '\n' for line in body.splitlines())
            + '\ndef on_event(state, event, context):\n'
              '    return {"state": compute(state, event, context), "session": {}, "memory": None}\n')


class FakeBroker:
    def __init__(self, value=None, handler=None):
        self.value = {'result': 'synthetic'} if value is None else value
        self.handler = handler
        self.calls = []

    def call(self, model_id, payload, *, operation):
        self.calls.append(dict(model_id=model_id, payload=copy.deepcopy(payload), operation=operation))
        value = self.handler(model_id, payload, operation) if self.handler else copy.deepcopy(self.value)
        return dict(value=value, receipt=dict(cache_key='fake-cache-' + str(len(self.calls)), cache_hit=False))


class ProgramCapabilityTests(unittest.TestCase):
    def test_pure_program_never_invokes_supplied_broker(self):
        broker = FakeBroker()
        value = Program(EMPTY).run('event', {'original': 1}, experience={}, capabilities=broker)
        self.assertEqual(value['value']['state'], {'original': 1})
        self.assertEqual(broker.calls, [])
        self.assertEqual(value['capability_calls'], [])
        self.assertEqual(value['worker_invocations'], 1)
        self.assertTrue(value['isolation']['seccomp'])
        self.assertFalse(value['isolation']['imports_files_network_process_creation'])

    def test_suspension_restarts_original_state_and_preserves_input(self):
        source = event_program('state["writes"] = state.get("writes", 0) + 1\n'
                         'state["result"] = model_call("local-fake", {"text": event.get("text", "")})\n'
                         'return state')
        original = {'writes': 2}
        broker = FakeBroker({'answer': 'data only'})
        result = Program(source).run('event', original, experience={'text': 'hello'}, capabilities=broker)
        self.assertEqual(original, {'writes': 2})
        self.assertEqual(result['value']['state'], {'writes': 3, 'result': {'answer': 'data only'}})
        self.assertEqual(result['worker_invocations'], 2)
        self.assertEqual(broker.calls, [dict(model_id='local-fake', payload={'text': 'hello'}, operation='event')])
        call = result['capability_calls'][0]
        self.assertEqual(call['request_hash'], capability_request('local-fake', {'text': 'hello'}, 'event')['request_hash'])
        self.assertEqual(call['receipt']['cache_key'], 'fake-cache-1')
        self.assertNotIn('payload', call)
        self.assertNotIn('value', call)

    def test_repeated_request_calls_broker_once_and_returns_independent_copies(self):
        source = event_program('first = model_call("local-fake", {"x": 1})\n'
                         'first["result"] = "locally changed"\n'
                         'second = model_call("local-fake", {"x": 1})\n'
                         'return {"first": first, "second": second}')
        broker = FakeBroker()
        result = Program(source).run('event', {}, experience={}, capabilities=broker)
        self.assertEqual(len(broker.calls), 1)
        self.assertEqual(len(result['capability_calls']), 2)
        self.assertEqual(result['value']['state']['first']['result'], 'locally changed')
        self.assertEqual(result['value']['state']['second']['result'], 'synthetic')

    def test_dependent_requests_produce_event_memory_with_bound_broker_identity(self):
        source = ('def on_event(state, event, context):\n'
                  '    first = model_call("first-model", {"text": context.get("goal", "")})\n'
                  '    second = model_call("second-model", {"value": first})\n'
                  '    return {"state": state, "session": {}, "memory": [{"id": "computed", "experience": second["result"]}]}\n')
        broker = FakeBroker(handler=lambda model, data, op: {'result': model + '-output'})
        result = Program(source).run('event', {}, experience={}, context={'goal': 'query'}, capabilities=broker)
        self.assertEqual([call['operation'] for call in broker.calls], ['event', 'event'])
        self.assertEqual(broker.calls[1]['payload'], {'value': {'result': 'first-model-output'}})
        self.assertEqual(result['value']['memory'], [{'id': 'computed', 'experience': 'second-model-output'}])
        self.assertEqual(result['value']['state'], {})
        self.assertEqual(result['worker_invocations'], 3)
        self.assertNotEqual(capability_request('first-model', {}, 'event')['request_hash'],
                            capability_request('second-model', {}, 'event')['request_hash'])

    def test_missing_or_failed_broker_rejects_without_state_commit(self):
        source = event_program('state["tentative"] = True\n'
                         'state["answer"] = model_call("missing", {})\nreturn state')
        state = {}
        with self.assertRaisesRegex(Rejected, 'explicitly configured'):
            Program(source).run('event', state, experience={})
        self.assertEqual(state, {})
        def fail(model, payload, operation):
            raise ValueError('model unavailable')
        broker = FakeBroker(handler=fail)
        with self.assertRaisesRegex(CapabilityInfrastructureError, 'broker failed: model unavailable'):
            Program(source).run('event', state, experience={}, capabilities=broker)
        self.assertEqual(state, {})
        self.assertEqual(len(broker.calls), 1)

    def test_eight_unique_calls_succeed_and_ninth_is_rejected_before_broker(self):
        for count in (8, 9):
            with self.subTest(count=count):
                source = event_program('values = []\nfor index in range(' + str(count) + '):\n'
                                 '    values.append(model_call("fake", {"index": index}))\n'
                                 'return {"values": values}')
                broker = FakeBroker()
                if count == 8:
                    result = Program(source).run('event', {}, experience={}, capabilities=broker)
                    self.assertEqual(len(result['value']['state']['values']), 8)
                    self.assertEqual(result['worker_invocations'], 9)
                else:
                    with self.assertRaisesRegex(Rejected, 'model_call budget exceeded'):
                        Program(source).run('event', {}, experience={}, capabilities=broker)
                self.assertEqual(len(broker.calls), 8)

    def test_call_budget_counts_repeated_cached_calls_too(self):
        source = event_program('values = []\nfor index in range(9):\n'
                         '    values.append(model_call("fake", {}))\nreturn {"values": values}')
        broker = FakeBroker()
        with self.assertRaisesRegex(Rejected, 'model_call budget exceeded'):
            Program(source).run('event', {}, experience={}, capabilities=broker)
        self.assertEqual(len(broker.calls), 1)

    def test_invalid_model_requests_are_rejected_before_broker(self):
        for expression in ('model_call("", {})', 'model_call("fake", [])',
                           'model_call("fake", {"x": set([1])})',
                           'model_call("fake", {"x": "a" * 200000})'):
            with self.subTest(expression=expression):
                broker = FakeBroker()
                with self.assertRaises(Rejected):
                    Program(event_program('return {"value": ' + expression + '}')).run(
                        'event', {}, experience={}, capabilities=broker)
                self.assertEqual(broker.calls, [])

    def test_non_json_or_oversized_broker_results_fail_closed(self):
        source = Program(event_program('return {"value": model_call("fake", {})}'))
        for value in ({'nonfinite': float('nan')}, {'not_json': {1, 2}}, {'large': 'x' * CAPABILITY_JSON_BYTES}):
            with self.subTest(value_type=str(type(value))):
                broker = FakeBroker(value)
                with self.assertRaises(CapabilityInfrastructureError):
                    source.run('event', {}, experience={}, capabilities=broker)
                self.assertEqual(len(broker.calls), 1)

    def test_bad_broker_envelope_is_rejected(self):
        class BadBroker:
            def call(self, model_id, payload, *, operation):
                return {'value': 'x', 'receipt': ['not-an-object']}
        with self.assertRaisesRegex(CapabilityInfrastructureError, 'return value and receipt'):
            Program(event_program('return {"value": model_call("fake", {})}')).run(
                'event', {}, experience={}, capabilities=BadBroker())

    def test_capability_cannot_be_shadowed_aliased_or_used_to_enable_unsafe_syntax(self):
        cases = [event_program('model_call = 1\nreturn {}'),
                 event_program('alias = model_call\nreturn {}'),
                 event_program('import os\nreturn {}'),
                 event_program('return {"x": open("/tmp/no") }'),
                 event_program('return {"x": model_call.__globals__}')]
        for source in cases:
            with self.subTest(source=source), self.assertRaises(Rejected):
                Program(source)
        text = 'import os; os.system("untrusted model data")'
        result = Program(event_program('return {"text": model_call("fake", {})}')).run(
            'event', {}, experience={}, capabilities=FakeBroker(text))
        self.assertEqual(result['value']['state'], {'text': text})

    def test_iteration_budget_is_shared_across_replays(self):
        source = event_program('value = 0\nfor index in range(60000):\n    value += 1\n'
                         'answer = model_call("fake", {})\nreturn {"answer": answer}')
        broker = FakeBroker()
        with self.assertRaisesRegex(Rejected, 'shared iteration budget exceeded'):
            Program(source).run('event', {}, experience={}, capabilities=broker)
        self.assertEqual(len(broker.calls), 1)

    def test_cpu_and_wall_budgets_accumulate_but_host_model_time_is_separate(self):
        pending = capability_request('fake', {}, 'event')
        trace = [{key: pending[key] for key in ('request_hash', 'model_id', 'operation')}]
        def receipt(cpu, suspended=False):
            row = dict(isolation=dict(seccomp=True, no_new_privs=True, iterations_used=3,
                                      generated_cpu_seconds=cpu, worker_cpu_seconds=cpu), capability_calls=trace)
            row.update({'needs_model': pending} if suspended else {'value': {'state': {}, 'session': {}, 'memory': None}})
            return row
        program = Program(EMPTY)
        for first, last, message in [((receipt(.6, True), .1), (receipt(.6), .1), 'CPU budget'),
                                     ((receipt(.1, True), 3.0), (receipt(.1), 3.0), 'wall budget')]:
            with self.subTest(message=message), patch('phase1_codegen.program._invoke_worker', side_effect=[first, last]):
                with self.assertRaisesRegex(Rejected, message):
                    program.run('event', {}, experience={}, capabilities=FakeBroker())
        with patch('phase1_codegen.program._invoke_worker', side_effect=[(receipt(.1, True), .1), (receipt(.1), .1)]), \
                patch('phase1_codegen.program.time.monotonic', side_effect=[0.0, 100.0]):
            result = program.run('event', {}, experience={}, capabilities=FakeBroker())
        self.assertEqual(result['capability_seconds'], 100.0)
        self.assertEqual(result['wall_seconds'], .2)
        self.assertEqual(result['isolation']['iterations_used'], 6)
        self.assertEqual(result['isolation']['iteration_budget'], ITERATION_BUDGET)

    def test_suspension_startup_cost_is_recorded_but_not_generated_cpu(self):
        pending = capability_request('fake', {}, 'event')
        trace = [{key: pending[key] for key in ('request_hash', 'model_id', 'operation')}]
        def row(suspended):
            value = dict(isolation=dict(seccomp=True, no_new_privs=True, iterations_used=3,
                generated_cpu_seconds=.1, worker_cpu_seconds=.7), capability_calls=trace)
            value.update({'needs_model': pending} if suspended else
                         {'value': {'state': {}, 'session': {}, 'memory': None}})
            return value
        with patch('phase1_codegen.program._invoke_worker', side_effect=[(row(True), .1), (row(False), .1)]):
            result = Program(EMPTY).run('event', {}, experience={}, capabilities=FakeBroker())
        self.assertAlmostEqual(result['isolation']['generated_cpu_seconds'], .2)
        self.assertAlmostEqual(result['isolation']['worker_cpu_seconds'], 1.4)
        self.assertEqual(result['worker_invocations'], 2)


if __name__ == '__main__':
    unittest.main()
