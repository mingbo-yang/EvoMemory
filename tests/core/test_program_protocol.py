"""CPU-only event contracts; synthetic programs are not effectiveness results."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from phase1_codegen import policy
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.program import Program, BASELINE_SOURCE
from phase1_codegen.protocol import HOOK_VERSION, WORKFLOW_VERSION, READ_TOKENS, CAPACITY_TOKENS
from phase1_codegen.roles.evolver import CodeEvolver, EVOLVER_PROMPT, HOOK_INTERFACE_PROMPT
from tests.core.test_engine import TinyTokenizer, SyntheticEvaluator, SyntheticProposer, fixture_log, task
from tests.core.test_review import CharacterTokenizer

PURE = '''def on_event(state, event, context):
    return {"state": state, "session": event["session"], "memory": None}
'''
OLD_SOURCE = '''def write(state, experience, context):
    return state

def read(state, context):
    return []
'''


class ProgramProtocolTests(unittest.TestCase):
    def test_only_event_is_a_public_entrypoint_even_with_ordinary_named_helpers(self):
        helpers = '''def read(value):
    return value

def write(value):
    return read(value)
'''
        program = Program(PURE + helpers)
        self.assertEqual(program.entrypoints, frozenset({'on_event'}))
        self.assertFalse(hasattr(program, 'has_write'))
        self.assertFalse(hasattr(program, 'has_read'))
        value = program.run('event', {'existing': 'state'}, {'type': 'task_start', 'session': {}}, {})
        self.assertEqual(value['value'], {'state': {'existing': 'state'}, 'session': {}, 'memory': None})
        self.assertTrue(value['isolation']['seccomp'])
        for operation in ('write', 'read'):
            with self.assertRaisesRegex(policy.Rejected, 'does not implement'):
                program.run(operation, {}, {} if operation == 'write' else None, {})
        with self.assertRaisesRegex(policy.Rejected, 'on_event'):
            Program(OLD_SOURCE)

    def test_bad_signature_unsafe_source_and_extra_output_are_rejected_without_repair(self):
        invalid = [OLD_SOURCE,
                   PURE.replace('on_event(state, event, context)', 'on_event(state, experience, context)'),
                   PURE.replace('return {', 'return open("/tmp/forbidden")\n    return {'),
                   'import os\n' + PURE,
                   PURE.replace('return {', 'return on_event(state, event, context)\n    return {')]
        for source in invalid:
            with self.subTest(source=source), self.assertRaises(policy.Rejected):
                Program(source)
        extra = PURE.replace('"memory": None', '"memory": [{"id": "x", "experience": "actual", "family": "extra"}]')
        with self.assertRaisesRegex(policy.Rejected, 'requires exactly id and experience'):
            Program(extra).run('event', {}, {'type': 'task_start', 'session': {}}, {})

    def test_static_metadata_is_cached_and_worker_keeps_its_boundary_validation(self):
        with mock.patch('phase1_codegen.program.validate_source', wraps=policy.validate_source) as validate:
            program = Program(PURE)
            for _ in range(4):
                self.assertEqual(len(program.hash), 64)
                self.assertEqual(len(program.identity_hash), 64)
                self.assertEqual(program.entrypoints, frozenset({'on_event'}))
            self.assertEqual(validate.call_count, 1)
            receipt = program.run('event', {}, {'type': 'task_start', 'session': {}}, {})
            self.assertEqual(validate.call_count, 1)
            self.assertTrue(receipt['isolation']['seccomp'])
        worker = ast.parse(Path(policy.__file__).with_name('worker.py').read_text())
        calls = [node.func.id for node in ast.walk(worker) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)]
        self.assertIn('validate_source', calls)
        commented = Program('# same implementation\n' + PURE)
        self.assertEqual(program.identity_hash, commented.identity_hash)
        self.assertNotEqual(program.hash, commented.hash)

    def test_one_prompt_and_event_abi_without_legacy_override_or_repair(self):
        self.assertNotIn('write(state, experience, context)', EVOLVER_PROMPT)
        self.assertNotIn('read(state, context)', EVOLVER_PROMPT)
        self.assertNotIn('legacy', HOOK_INTERFACE_PROMPT)
        self.assertIn('single public entrypoint', HOOK_INTERFACE_PROMPT)
        self.assertIn('mode/origin=history_import', HOOK_INTERFACE_PROMPT)
        self.assertIn(str(READ_TOKENS), HOOK_INTERFACE_PROMPT)
        self.assertIn(str(CAPACITY_TOKENS), EVOLVER_PROMPT)
        self.assertNotIn(BASELINE_SOURCE, EVOLVER_PROMPT)
        evolver = CodeEvolver.__new__(CodeEvolver)
        evolver.tokenizer = CharacterTokenizer()
        evolver.llm = mock.Mock(model_name="synthetic-test-model")
        evolver._invoke = mock.Mock(return_value={})
        self.assertFalse(hasattr(evolver, 'repair_interface'))
        for context in ({}, {'memory_hook_version': HOOK_VERSION}):
            evolver({'context': context, 'current_source': PURE}, 1)
            self.assertEqual(evolver._invoke.call_args.args[0]['messages'][0]['content'], EVOLVER_PROMPT)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            evolver({'context': {'memory_hook_version': None}, 'current_source': PURE}, 1)
        self.assertEqual(evolver._invoke.call_count, 2)

    def test_engine_rejects_old_source_without_new_proposal_or_correction(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-protocol-engine-') as tmp:
            proposer = SyntheticProposer([])
            engine = EvolutionEngine(tmp, TinyTokenizer(), proposer, SyntheticEvaluator(), EvolutionMemory(),
                {'family': 'synthetic-clean', 'environment': 'alfworld'}, [fixture_log()],
                [task('explore')], [[task('confirm')]], synthetic=True, acceptance_policy='native_reward')
            prepared = engine._prepare_candidate({'source': OLD_SOURCE},
                [{'goal': 'clean apple', 'observation': 'apple'}], [])
            self.assertIn('on_event', str(prepared['error']))
            self.assertIsNone(prepared['signature'])
            self.assertEqual(proposer.calls, [])
            self.assertEqual(engine.binding['workflow'], WORKFLOW_VERSION)
            self.assertNotIn('interface_repair', engine.binding)
            self.assertFalse(hasattr(engine, 'interface_repair_attempts'))

    def test_baseline_history_ingestion_and_initial_query_preserve_input_state(self):
        program = Program(BASELINE_SOURCE)
        original = {}
        stored = program.run('event', original,
            {'type': 'task_end', 'session': {}, 'experience': fixture_log(), 'origin': 'history_import'}, {})
        self.assertEqual(original, {})
        self.assertEqual(stored['value']['state']['episodes'][0]['id'], 'adaptation-only')
        self.assertIsNone(stored['value']['memory'])
        result = program.run('event', stored['value']['state'],
            {'type': 'task_start', 'session': {}, 'visible': {'obs': 'apple', 'goal': 'clean'}}, {})
        self.assertIn('take apple', result['value']['memory'][0]['experience'])
        self.assertEqual(result['value']['state'], stored['value']['state'])
        self.assertEqual(policy.validate_state({'valid': [1]}), {'valid': [1]})
        with self.assertRaises(policy.Rejected):
            policy.validate_state([])
