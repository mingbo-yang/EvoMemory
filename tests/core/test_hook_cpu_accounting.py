"""Resource/transaction tests, not task-effectiveness experiments."""
import copy
import unittest
from unittest.mock import patch

from phase1_codegen.hooks import HookSession
from phase1_codegen.policy import Rejected
from phase1_codegen.protocol import HookLimits
from tests.core import test_hooks as fixtures


class HookCpuAccountingTests(unittest.TestCase):
    def make_hook(self, **limits):
        fixture = fixtures.HookTests()
        bank = fixture.bank()
        return HookSession(bank, fixture.task(), limits=HookLimits(**limits))

    def receipt(self, state, event):
        return dict(value=dict(state=copy.deepcopy(state), session={}, memory=None),
                    isolation=dict(worker_cpu_seconds=0.2, generated_cpu_seconds=0.001),
                    wall_seconds=0.002)

    def test_162_events_do_not_spend_mechanism_cpu_on_trusted_startup(self):
        hook = self.make_hook()
        inputs = []
        def invocation(op, state, event, ctx, **kw):
            inputs.append(copy.deepcopy(event))
            return self.receipt(state, event)
        with patch.object(type(hook.memory.program), 'run',
                          side_effect=invocation):
            hook.emit('task_start', {'step': 0})
            for i in range(160):
                hook.emit('before_action' if i % 2 == 0 else 'after_observation', {'step': i // 2})
            hook.finish(fixtures.HookTests().episode())
        self.assertTrue(hook.summary()['complete'])
        self.assertGreater(hook.usage['worker_cpu_seconds'], 20)
        self.assertAlmostEqual(hook.usage['generated_cpu_seconds'], 0.162)
        self.assertAlmostEqual(inputs[1]['budget_remaining']['generated_cpu_seconds'], 19.999)

    def test_actual_generated_cpu_exhaustion_still_discards_state(self):
        hook = self.make_hook(generated_cpu_seconds=0.0015)
        original = hook.memory.export_snapshot()
        with patch.object(type(hook.memory.program), 'run',
                          side_effect=lambda op, state, event, ctx, **kw: self.receipt(state, event)):
            hook.emit('task_start', {'step': 0})
            with self.assertRaisesRegex(Rejected, 'generated_cpu_seconds budget exceeded'):
                hook.emit('before_action', {'step': 0})
        self.assertTrue(hook.failed)
        self.assertEqual(original, hook.memory.export_snapshot())
        self.assertFalse(hook.summary()['committed'])


if __name__ == '__main__':
    unittest.main()
