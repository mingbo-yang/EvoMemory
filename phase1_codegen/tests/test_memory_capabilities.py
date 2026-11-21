"""Synthetic capability integration and review accounting; no model inference."""
import copy
import tempfile
import unittest
from pathlib import Path

from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.memory import CodeMemory
from phase1_codegen.program import Program
from phase1_codegen.policy import Rejected
from phase1_codegen.tests.test_engine import TinyTokenizer, fixture_log, task
from phase1_codegen.tests.test_review_loop import EventEvaluator, EventProposer, SyntheticReviewer


SOURCE = '''def on_event(state, event, context):
    memory = None
    if event["type"] == "task_end":
        experience = event["experience"]
        first = model_call("fake-model", {"prompt": experience.get("goal", "")})
        second = model_call("fake-model", {"prompt": experience.get("goal", "")})
        state = {"id": experience["task_id"], "value": second["text"]}
    elif event["type"] == "task_start":
        result = model_call("fake-model", {"prompt": event.get("visible", {}).get("goal", "")})
        state["tentative"] = result
        memory = [{"id": state["id"], "experience": result["text"]}]
    return {"state": state, "session": event["session"], "memory": memory}
'''



class SyntheticCapabilities:
    environment = 'alfworld'
    fingerprint = 'synthetic-capabilities'

    def __init__(self):
        self.calls = []

    def describe(self):
        return {'api': 'model_call(model_id, payload)',
                'models': [{'id': 'fake-model', 'kind': 'text_generation'}]}

    def call(self, model_id, payload, *, operation):
        self.calls.append((model_id, copy.deepcopy(payload), operation))
        return {'value': {'text': 'synthetic output'}, 'receipt': {
            'request_id': len(self.calls), 'cache_hit': False, 'physical': True,
            'inference_attempted': True, 'ok': True, 'seconds': 0.5}}


class MemoryCapabilitiesTests(unittest.TestCase):
    def test_model_data_respects_readonly_state_and_same_dataset(self):
        capabilities = SyntheticCapabilities()
        log = dict(fixture_log(), environment='alfworld')
        memory = CodeMemory(Program(SOURCE), [log], TinyTokenizer(),
                            {'environment': 'alfworld', 'family': log['family']}, capabilities=capabilities)
        before = memory.state
        self.assertEqual(len(capabilities.calls), 1)
        self.assertEqual(len(memory.write_trace[0]['event_trace'][0]['receipt']['capability_calls']), 2)
        self.assertEqual(memory.context['model_catalog'], capabilities.describe())
        output = memory.read({'goal': 'a synthetic query'})
        self.assertEqual(output[0]['experience'], 'synthetic output')
        self.assertEqual(memory.state, before)
        self.assertEqual(capabilities.calls[-1][2], 'event')
        with self.assertRaisesRegex(Rejected, 'dataset'):
            CodeMemory(Program(SOURCE), [], TinyTokenizer(), {'environment': 'textworld'}, capabilities=capabilities)

    def test_reviewer_counts_one_physical_request_for_two_program_calls(self):
        capabilities = SyntheticCapabilities()
        events = []
        with tempfile.TemporaryDirectory() as directory:
            log = dict(fixture_log(), environment='alfworld')
            engine = EvolutionEngine(Path(directory), TinyTokenizer(), EventProposer([SOURCE], events),
                EventEvaluator(events, sources=(SOURCE,)), EvolutionMemory([]),
                {'environment': 'alfworld', 'family': log['family']}, [log], [task('explore')], [],
                reviewer=SyntheticReviewer(events), synthetic=True, max_rounds=1, capabilities=capabilities)
            memory = CodeMemory(Program(SOURCE), [log], TinyTokenizer(), engine.context, capabilities=capabilities)
            before = {'stored_state': memory.state, 'stored_tokens': memory.stats['stored_tokens'],
                      'memory_hash': memory.fingerprint, 'write_trace': memory.write_trace,
                      'read_trace': [], 'readouts': []}
            payload = engine._review_input('pre', Program(SOURCE), before, [])
            costs = next(row['data'] for row in payload['evidence'] if row['id'] == 'parent:model_calls')
            self.assertEqual((costs['program_calls'], costs['broker_requests'], costs['completed_physical']), (2, 1, 1))
            self.assertEqual(costs['seconds'], 0.5)
            self.assertNotIn('model_api', payload['context'])


if __name__ == '__main__':
    unittest.main()
