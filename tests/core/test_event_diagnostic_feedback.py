"""Observed Hook facts must remain useful and faithful after event migration."""
import copy
import unittest

from evomem.common import digest, dumps
from phase1_codegen.behavior_diagnostics import bounded_execution_facts, diagnose
from phase1_codegen.memory import CodeMemory
from phase1_codegen.program import Program
from tests.core.test_program_memory import CharacterTokenizer, fixture


SOURCE = '''def store(state, experience):
    rows = state.get("rows", [])
    rows.append({"id": experience["task_id"], "experience": experience["goal"]})
    return {"rows": rows}

def read_rows(state, context):
    visible = context.get("visible", {})
    goal = context.get("goal", "")
    observation = context.get("observation", "")
    rows = []
    for row in state.get("rows", []):
        rows.append(row)
    return rows

def on_event(state, event, context):
    memory = None
    if event["type"] == "task_end":
        state = store(state, event["experience"])
    else:
        optional = context.get("optional_metadata", "")
        memory = read_rows(state, context)
    return {"state": state, "session": event["session"], "memory": memory}
'''


class EventDiagnosticFeedbackTests(unittest.TestCase):
    def evidence(self):
        logs = [fixture('one'), fixture('two')]
        memory = CodeMemory(Program(SOURCE), logs, CharacterTokenizer(),
                            {'environment': 'alfworld', 'family': 'synthetic-clean'}, diagnostics=True)
        memory.read({})
        behavior = dict(stored_state=memory.state, memory_hash=memory.fingerprint,
                        read_trace=memory.read_trace, write_trace=memory.write_trace)
        return diagnose(memory.program, behavior, logs)

    def test_event_facts_prioritize_actual_entrypoint_and_keep_iteration_samples(self):
        report = self.evidence()
        missing = report['facts']['missing_dict_keys']
        self.assertEqual(missing[0]['function'], 'on_event')
        self.assertEqual([row['first_missing']['key'] for row in missing],
                         ['optional_metadata', 'visible', 'goal', 'observation'])
        self.assertTrue(all(row['operation'] == 'task_start' for row in missing))
        iterations = report['facts']['iteration_prefixes']
        self.assertTrue(iterations)
        self.assertEqual(iterations[0]['operation'], 'task_start')
        self.assertEqual(iterations[0]['values'][0]['complete'], False)
        self.assertIn('do not by themselves establish a defect', report['interpretation_boundary'])

    def test_shared_fact_budget_retains_exact_values_and_original_pointers(self):
        data = self.evidence()
        before = copy.deepcopy(data)
        tokenizer = CharacterTokenizer()
        selected = bounded_execution_facts(data, tokenizer, 6000)
        self.assertIsNotNone(selected)
        self.assertLessEqual(tokenizer.count(dumps(selected)), 6000)
        self.assertEqual(selected['original_hash'], digest(data))
        self.assertEqual(selected['provenance'], data['provenance'])
        for key, value in selected['facts'].items():
            self.assertEqual(value, data['facts'][key][:len(value)] if isinstance(value, list)
                             else data['facts'][key])
        for pointer in selected['omitted_fact_paths']:
            self.assertTrue(pointer.startswith('/facts/'))
        self.assertEqual(data, before)
        self.assertIsNone(bounded_execution_facts(data, tokenizer, 10))


if __name__ == '__main__':
    unittest.main()
