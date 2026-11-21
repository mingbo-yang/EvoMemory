"""Recorded values and event boundaries; no native task or model inference."""
from unittest import TestCase, main
from unittest.mock import patch
from phase1_codegen.program import Program
from phase1_codegen.memory import CodeMemory
from phase1_codegen.behavior_diagnostics import diagnose
from tests.core.test_engine import fixture_log, TinyTokenizer

SOURCE = '''def parse_score(text):
    tail = text.strip()
    if tail.isdigit():
        return float(tail)
    return 0.0

def query_text(context):
    visible = context.get("visible", {})
    goal = visible.get("goal", "")
    obs = visible.get("obs", "")
    query = (goal + " " + obs).lower()
    return query

def on_event(state, event, context):
    if event["type"] == "task_end":
        state = {"score": parse_score(str(event["experience"]["score"]))}
    if event["type"] == "task_start":
        query = query_text(context)
    return {"state": state, "session": {}, "memory": None}
'''


class ExecutionValueWitnessTests(TestCase):
    def test_real_trace_samples_expose_empty_query_and_float_text_without_reexecution(self):
        logs = [dict(fixture_log(), score=1.0)]
        memory = CodeMemory(Program(SOURCE), logs, TinyTokenizer(),
            dict(environment='alfworld', family=logs[0]['family']), diagnostics=True)
        memory.for_task(dict(id='read-task', family=logs[0]['family']), 'alfworld').retrieve('a visible goal', 'a visible observation')
        before = memory.fingerprint
        behavior = dict(stored_state=memory.state, memory_hash=before,
                        read_trace=memory.read_trace, write_trace=memory.write_trace)
        with patch.object(Program, 'run', side_effect=AssertionError('No execution during projection')):
            result = diagnose(memory.program, behavior, logs)
        row = next(x for x in result['facts']['missing_dict_keys'] if x['expression']=='context.get("visible", {})')
        self.assertTrue(row['event_visible_shape']['obs']['nonempty'])
        self.assertTrue(row['event_visible_shape']['goal']['nonempty'])
        query = next(w for w in row['assignment_witnesses'] if w['expression'].startswith('query ='))
        self.assertEqual(query['sample_at_assignment']['value'], ' ')
        self.assertTrue(query['sample_at_assignment']['whitespace_only'])
        branch = next(x for x in result['facts']['branch_counts'] if x['expression']=='tail.isdigit()')
        tail = next(w for w in branch['assignment_witnesses'] if w['expression'].startswith('tail ='))
        self.assertEqual(tail['sample_at_assignment']['value'], '1.0')
        self.assertEqual(branch['true_count'], 0)
        self.assertEqual(branch['false_count'], 1)
        self.assertEqual(memory.fingerprint, before)
        self.assertEqual(memory.state['score'], 0.0)

    def test_optional_field_absence_keeps_nonempty_fallback_witness_without_defect_label(self):
        code = SOURCE.replace('visible = context.get("visible", {})', 'visible = context')
        code = code.replace('query = query_text(context)', 'query = query_text(event.get("visible", {}))')
        memory = CodeMemory(Program(code), [fixture_log()], TinyTokenizer(),
            dict(environment='alfworld', family=fixture_log()['family']), diagnostics=True)
        memory.read({}, visible={'obs': 'a current observation'})
        result = diagnose(memory.program, dict(stored_state=memory.state, memory_hash=memory.fingerprint,
                         read_trace=memory.read_trace, write_trace=memory.write_trace), [fixture_log()])
        self.assertNotIn('supported_defect', str(result))
        for row in result['facts']['missing_dict_keys']:
            self.assertIn('assignment_witnesses', row)
        query = next(w for row in result['facts']['missing_dict_keys'] for w in row['assignment_witnesses']
                     if w['expression'].startswith('query ='))
        self.assertFalse(query['sample_at_assignment']['whitespace_only'])
        self.assertEqual(result['version'], 'behavior-diagnostics-v2')


if __name__=='__main__':
    main()
