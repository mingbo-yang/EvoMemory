"""Behavioral regressions for package imports; no model, task, or downloaded assets."""
import subprocess
import sys
import unittest


class ImportBoundaryTests(unittest.TestCase):
    def test_benchmark_imports_share_actor_without_path_or_prompt_mutation(self):
        code = '''
import copy
import importlib
import sys
from benchmarks.math_apps import actor
before_path = list(sys.path)
before_prompts = copy.deepcopy(actor.PROMPTS)
from benchmarks.math_apps import benchmark, apps_tasks
assert benchmark.actor is actor
assert sys.path == before_path
assert actor.PROMPTS == before_prompts
importlib.reload(benchmark)
assert benchmark.actor is actor
assert actor.PROMPTS == before_prompts
assert sys.path == before_path
'''
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
