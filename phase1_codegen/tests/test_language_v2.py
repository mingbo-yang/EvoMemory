"""Real isolated workers with synthetic capabilities; no model inference."""
import unittest
from phase1_codegen.program import Program
from phase1_codegen.policy import Rejected, SAFE_CALLS, TYPE_NAMES, EXCEPTION_NAMES
from phase1_codegen.proposer import LANGUAGE_RULES_PROMPT
from phase1_codegen.tests.test_memory_capabilities import SyntheticCapabilities

RETURN = '    return {"state": state, "session": {}, "memory": None}\n'


def source(body):
    return 'def on_event(state, event, context):\n' + body + RETURN


class RestrictedLanguageV2Tests(unittest.TestCase):
    def run_program(self, text, **kwargs):
        return Program(text).run('event', {}, {'type': 'task_start', 'session': {}}, {}, **kwargs)

    def test_static_type_checks_and_native_bool_semantics(self):
        text = source('    state = {"dict": isinstance({}, dict), "tuple": isinstance([], (dict, list)), "bool_int": isinstance(True, int)}\n')
        result = self.run_program(text)['value']['state']
        self.assertEqual(result, {'dict': True, 'tuple': True, 'bool_int': True})

    def test_type_objects_aliases_reflection_and_dynamic_types_stay_forbidden(self):
        bodies = ['    test = dict\n', '    state = {"bad": dict}\n',
                  '    state = {"bad": isinstance({}, type({}))}\n',
                  '    state = {"bad": isinstance({}, event["types"])}\n',
                  '    state = {"bad": isinstance(dict, dict)}\n',
                  '    state = {"bad": isinstance({}, (dict, event))}\n',
                  '    state = {"bad": dir(state)}\n']
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(Rejected):
                Program(source(body))

    def test_narrow_data_exceptions_and_diagnostic_equivalence(self):
        text = source('    try:\n        number = float("not a number")\n    except (ValueError, TypeError):\n        number = 7\n    state = {"number": number}\n')
        normal = self.run_program(text)
        diagnostic = self.run_program(text, diagnostics=True)
        self.assertEqual(normal['value'], diagnostic['value'])
        self.assertEqual(normal['value']['state'], {'number': 7})
        self.assertEqual(diagnostic['diagnostics']['status'], 'available')

    def test_broad_handlers_binding_finally_raise_and_exception_aliases_rejected(self):
        bodies = ['    try:\n        number = 1\n    except:\n        number = 0\n',
                  '    try:\n        number = 1\n    except Exception:\n        number = 0\n',
                  '    try:\n        number = 1\n    except BaseException:\n        number = 0\n',
                  '    try:\n        number = 1\n    except ValueError as error:\n        number = 0\n',
                  '    try:\n        number = 1\n    finally:\n        number = 0\n',
                  '    raise ValueError("bad")\n', '    data = ValueError\n']
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(Rejected):
                Program(source(body))

    def test_catching_data_errors_never_catches_or_resets_iteration_budget(self):
        text = source('    try:\n        for number in range(100001):\n            state["count"] = number\n    except (ValueError, TypeError, KeyError, IndexError):\n        state["swallowed"] = True\n')
        with self.assertRaisesRegex(Rejected, 'TrustedTermination.*iteration budget'):
            self.run_program(text)

    def test_repeated_caught_data_errors_still_consume_loop_budget(self):
        text = source('    for number in range(100001):\n        try:\n            item = float("bad")\n        except ValueError:\n            state["count"] = number\n')
        with self.assertRaisesRegex(Rejected, 'iteration budget'):
            self.run_program(text)

    def test_invalid_capability_payload_cannot_be_caught(self):
        text = source('    try:\n        item = model_call("fake-model", [])\n    except (ValueError, TypeError, KeyError, IndexError):\n        state["swallowed"] = True\n')
        with self.assertRaisesRegex(Rejected, 'TrustedTermination.*capability contract'):
            self.run_program(text, capabilities=SyntheticCapabilities())

    def test_model_budget_cannot_be_caught_and_suspend_replay_is_preserved(self):
        caps = SyntheticCapabilities()
        text = source('    try:\n        for number in range(9):\n            item = model_call("fake-model", {"prompt": "test"})\n    except (ValueError, TypeError, KeyError, IndexError):\n        state["swallowed"] = True\n')
        with self.assertRaisesRegex(Rejected, 'TrustedTermination.*model_call budget'):
            self.run_program(text, capabilities=caps)
        self.assertEqual(len(caps.calls), 1)
        valid = source('    try:\n        item = model_call("fake-model", {"prompt": "test"})\n        state["value"] = item["text"]\n    except ValueError:\n        state["swallowed"] = True\n')
        receipt = self.run_program(valid, capabilities=caps)
        self.assertEqual(receipt['value']['state'], {'value': 'synthetic output'})
        self.assertEqual(receipt['worker_invocations'], 2)

    def test_host_model_failure_cannot_be_converted_to_success(self):
        class Broken(SyntheticCapabilities):
            def call(self, *args, **kwargs):
                raise ValueError('synthetic service failure')
        text = source('    try:\n        item = model_call("fake-model", {"prompt": "test"})\n    except ValueError:\n        state["swallowed"] = True\n')
        with self.assertRaisesRegex(ValueError, 'synthetic service failure'):
            self.run_program(text, capabilities=Broken())

    def test_language_description_uses_same_contract_names(self):
        for name in SAFE_CALLS | TYPE_NAMES | EXCEPTION_NAMES:
            self.assertIn(name, LANGUAGE_RULES_PROMPT)
        self.assertIn('cannot be caught', LANGUAGE_RULES_PROMPT)


if __name__ == '__main__':
    unittest.main()
