"""CPU integration checks for trusted startup vs generated execution budgets."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phase1_codegen import program as runtime
from phase1_codegen.policy import Rejected
from phase1_codegen.program import Program, WorkerStartupError


EMPTY = 'def on_event(state, event, context):\n    return {"state": state, "session": {}, "memory": None}\n'


class WorkerBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-worker-bootstrap-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def delayed_worker(self, seconds):
        path = self.root / 'trusted_delayed_bootstrap.py'
        path.write_text('import runpy, time\ntime.sleep(%r)\nrunpy.run_path(%r, run_name="__main__")\n' %
                        (seconds, str(runtime.WORKER)))
        return path

    def test_slow_trusted_start_does_not_consume_generated_execution_budget(self):
        wrapper = self.delayed_worker(.3)
        with patch.object(runtime, 'WORKER', wrapper), patch.object(runtime, 'TOTAL_WORKER_WALL_SECONDS', .05):
            result = Program(EMPTY).run('event', {}, experience={})
        self.assertEqual(result['value'], {'state': {}, 'session': {}, 'memory': None})
        self.assertGreaterEqual(result['bootstrap_seconds'], .3)
        self.assertLess(result['wall_seconds'], .05)
        self.assertGreater(result['total_process_seconds'], result['wall_seconds'])
        self.assertTrue(result['isolation']['seccomp'])
        self.assertTrue(result['isolation']['no_new_privs'])

    def test_bootstrap_timeout_is_distinct_and_cannot_execute_generated_code(self):
        wrapper = self.delayed_worker(.3)
        with patch.object(runtime, 'WORKER', wrapper), patch.object(runtime, 'WORKER_BOOTSTRAP_WALL_SECONDS', .05):
            with self.assertRaisesRegex(WorkerStartupError, 'before generated execution'):
                Program(EMPTY).run('event', {}, experience={})

    def test_generated_execution_still_obeys_its_own_wall_deadline(self):
        source = ('def on_event(state, event, context):\n    total = 0\n'
                  '    for value in range(100000):\n        total += value\n'
                  '    return {"state": state, "session": {}, "memory": [{"id": "sum", "experience": str(total)}]}\n')
        with patch.object(runtime, 'TOTAL_WORKER_WALL_SECONDS', .0001):
            with self.assertRaisesRegex(Rejected, 'five-second wall budget'):
                Program(source).run('event', {}, experience={})

    def test_invalid_trusted_bootstrap_is_not_a_valid_empty_memory_result(self):
        path = self.root / 'broken_worker.py'
        path.write_text('raise RuntimeError("SyntheticTrustedStartupFailure")\n')
        with patch.object(runtime, 'WORKER', path):
            with self.assertRaises(WorkerStartupError):
                Program(EMPTY).run('event', {}, experience={})
