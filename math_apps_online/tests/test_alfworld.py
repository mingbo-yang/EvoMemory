"""ALFWorld online transaction with an actual local game and a CPU-only Actor transport fixture."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from evomem.common import MODEL_NAME, read
from math_apps_online.alfworld import ALFWorldOnlineEvaluator
from phase1_codegen.program import Program
from phase1_codegen.tests.test_engine import TinyTokenizer
from benchmarks.alfworld.panel import load_panel


class TokenFixture(TinyTokenizer):
    def chat_count(self, messages):
        return 100


SOURCE = '''def on_event(state, event, context):
    if event["type"] == "task_end":
        state = {"z": event["experience"]["task_id"], "a": len(event["experience"]["history"])}
    return {"state": state, "session": {}, "memory": None}
'''


@unittest.skipUnless(importlib.util.find_spec('alfworld') and (Path(__file__).resolve().parents[2] / 'data/alfworld_panel/crossep_emb_manifest_audit.json').is_file(), 'Prepared ALFWorld assets and runtime required')
class ALFWorldTest(unittest.TestCase):
    def _run_native_case(self, name):
        # ALMA imports modules from its staged game directory for the lifetime of an interpreter.
        target = f'math_apps_online.tests.test_alfworld.ALFWorldTest.{name}'
        run = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-q', target],
                             cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        self.assertEqual(run.returncode, 0, run.stdout)

    def test_selected_multistep_state_and_receipts_are_reused_without_actor_or_hook_replay(self):
        self._run_native_case('check_selected_multistep')

    def check_selected_multistep(self):
        import httpx
        task = load_panel()[0]
        calls = []
        async def fixture(transport, request):
            calls.append(1)
            return httpx.Response(200, request=request, json=dict(id='fixture', object='chat.completion',
                model=MODEL_NAME, created=0, choices=[dict(index=0, finish_reason='stop',
                    message=dict(role='assistant', content='look'))],
                usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101)))
        with tempfile.TemporaryDirectory() as tmp, patch.object(httpx.AsyncHTTPTransport,'handle_async_request',fixture):
            e = ALFWorldOnlineEvaluator(Path(tmp), [task], TokenFixture(), 'cpu_fixture',
                dict(environment='alfworld', family=task['family'], memory_scope='family'))
            self.addCleanup(e.close)
            e.begin_item(task)
            p = Program(SOURCE)
            pre = e.snapshot(p, [])
            ep, = e(p, [], [task], 'first', memory_snapshot=pre)
            self.assertIsNone(ep['error'])
            self.assertEqual(ep['native_outcome'], dict(score=0.0, won=False, status='unsolved',
                             completed_steps=30, action_attempts=30, step_limit=30))
            self.assertEqual(len(calls), 30)
            self.assertEqual(e.item_model_costs(task)['actor']['measured_input_tokens'], 3000)
            self.assertEqual(ep['memory_hash_before'], ep['memory_hash_after'])
            with patch.object(Program, 'run', side_effect=AssertionError('Unexpected reexecution')):
                self.assertEqual(e(p, [], [task], 'first', memory_snapshot=pre), [ep])
                snapshot = e.commit(p, [], task, ep, memory_snapshot=pre)
                self.assertEqual(snapshot['state_wire'], json.dumps({'z':task['id'], 'a':30}, separators=(',',':')))
                self.assertEqual(e.snapshot(p, [ep], snapshot), snapshot)
                self.assertEqual(len(calls), 30)
            folder = e._directory(task, p, 'first')
            self.assertEqual(read(folder/'native_reference.json')['start'], ep['actor_ledger_start'])
            with (folder/'canonical_episodes.jsonl').open('ab') as stream:
                stream.write(b'\n')
            with self.assertRaisesRegex(RuntimeError,'evidence changed'):
                e.commit(p, [], task, ep, memory_snapshot=pre)

    def test_failed_hook_before_actor_is_an_unscored_noncommitting_attempt(self):
        self._run_native_case('check_failed_hook')

    def check_failed_hook(self):
        import httpx
        task = load_panel()[0]
        p = Program(SOURCE.replace('"memory": None', '"memory": [1]'))
        with tempfile.TemporaryDirectory() as tmp, patch.object(httpx.AsyncHTTPTransport,'handle_async_request',
                                                               side_effect=AssertionError('Unexpected Actor dispatch')):
            e = ALFWorldOnlineEvaluator(Path(tmp), [task], TokenFixture(), 'cpu_fixture',
                dict(environment='alfworld', family=task['family'], memory_scope='family'))
            self.addCleanup(e.close)
            pre = e.snapshot(p, [])
            ep, = e(p, [], [task], 'round-000/exploration/candidate', memory_snapshot=pre)
            self.assertFalse(ep['native_reward_verified'])
            self.assertTrue(ep['error'])
            self.assertIsNone(ep['score'])
            self.assertEqual(ep['memory_hash_before'], ep['memory_hash_after'])
            self.assertIsNone(ep.get('native_outcome'))
            self.assertEqual(ep['generated_memory_write_calls'], 0)
            with patch.object(Program, 'run', side_effect=AssertionError('Unexpected reexecution')):
                self.assertEqual(e.measured_episode(p, [], task, ep['phase'], memory_snapshot=pre), ep)
                with self.assertRaisesRegex(ValueError,'verified, measured'):
                    e.commit(p, [], task, ep, memory_snapshot=pre)
