"""CPU contracts and explicitly synthetic model transport; no research scores."""
import asyncio
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import httpx

from evomem.alma_backend import ACTOR_PARAMETERS, CONTEXT
from evomem.common import MODEL_NAME, events
from phase1_codegen.policy import Rejected
from phase1_codegen.textworld_native import STEP_LIMIT, VENDOR, TextWorldALMARunner, TextWorldNativeEvaluator
from phase1_codegen.textworld_input import (goal_delivery_receipt, staged_environment_source,
                                           summarize_goal_delivery)
from phase1_codegen.textworld_tasks import (GAMES, asset_seed_receipt, build_manifest, load_tasks,
                                           verify_task_assets)


class Tokenizer:
    def chat_count(self, messages):
        return sum(len(message['content']) for message in messages)

    def count(self, text):
        return len(text)


def seed_test_inventory():
    # This frozen manifest contains only asset identifiers and checksums. The
    # tests authenticate its original bytes without reopening the game archive.
    raw = (GAMES.parent / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != '2e53b3dc328428447d0ca2fe0e430a91c4f977ff00159967f5453e8f8c0895a0':
        raise AssertionError('historical TextWorld manifest changed')
    manifest = json.loads(raw)
    tasks = [t for t in manifest['tasks'] if t['official_split'] == 'train'
             and t['room'] in {'cooking_item5_seed_12274', 'seed_37212'}]
    return dict(archive_sha256=manifest['archive_sha256'], tasks=tasks)


def record_seed_receipts():
    """Exercise the production episode ledger with one local native step and a synthetic Actor."""
    class LookTransport(httpx.AsyncBaseTransport):
        def __init__(self, **kwargs):
            pass

        async def handle_async_request(self, request):
            return httpx.Response(200, request=request, json={
                'id': 'synthetic', 'object': 'chat.completion', 'created': 0, 'model': MODEL_NAME,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'look'},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 12, 'completion_tokens': 1, 'total_tokens': 13}})

    inventory = seed_test_inventory()
    tasks = inventory['tasks']
    with tempfile.TemporaryDirectory() as temporary, patch.object(httpx, 'AsyncHTTPTransport', LookTransport):
        with patch('phase1_codegen.textworld_native.build_manifest', return_value=inventory):
            evaluator = TextWorldNativeEvaluator(temporary, tasks, Tokenizer(), 'synthetic-CPU')
        try:
            evaluator.runner.workflow.env.max_trails = 1
            evaluator.no_memory(tasks)
            rows = events(Path(temporary) / 'native/canonical_episodes.jsonl')
            requests = {r['task_id']: json.dumps(r['messages']) for r in events(evaluator.ledger)
                        if r.get('kind') == 'model'}
            from phase1_codegen.engine import own_trace
            print(json.dumps([dict(asset=row['asset_seeds'], runtime=row['environment_seed_receipt'],
                                   legacy_field='compiled_game_seed' in row, exposed='asset_seeds' in own_trace(row),
                                   actor_exposed=any(key in requests[row['task_id']]
                                                     for key in ('asset_seeds', 'settings_seed', 'sidecar_sha256')),
                                   steps=len(row['history']), error=row['error']) for row in rows]))
        finally:
            evaluator.close()

class TextWorldContractTests(unittest.TestCase):
    def test_authenticated_generation_seeds_and_legacy_task_payload(self):
        from phase1_codegen.textworld_native import prepare_textworld
        options = prepare_textworld().GameOptions()
        cooking = next(t for t in seed_test_inventory()['tasks'] if t['family'] == 'the_cooking_game')
        self.assertEqual(cooking['compiled_game_seed'], 12274)
        self.assertNotIn('asset_seeds', cooking)
        path = verify_task_assets(cooking)
        receipt = asset_seed_receipt(cooking, path)
        options.seeds = receipt['settings_seed']
        self.assertEqual(receipt['components'], options.seeds)
        options.seeds = receipt['filename_seed']
        self.assertNotEqual(receipt['components'], options.seeds)
        self.assertEqual(receipt['sidecar_sha256'], cooking['sidecar_sha256']['.json'])
        forged = dict(cooking, sidecar_sha256={**cooking['sidecar_sha256'], '.json': '0' * 64})
        with self.assertRaisesRegex(Rejected, 'changed before seed provenance'):
            asset_seed_receipt(forged, path)

    def test_native_episode_separates_asset_seeds_from_reset_and_agent_input(self):
        # The upstream workflow installs process-global modules and creates native
        # temporary files. Keep it out of other unit tests and off network storage.
        child = subprocess.run(
            [sys.executable, '-B', '-c',
             'from tests.core.test_textworld_native import record_seed_receipts; record_seed_receipts()'],
            env={**os.environ, 'TMPDIR': '/tmp'}, capture_output=True, text=True, timeout=120)
        self.assertEqual(child.returncode, 0, child.stderr)
        rows = json.loads(child.stdout.strip().splitlines()[-1])
        self.assertEqual(len(rows), 2)
        by_label = {row['asset']['filename_seed']: row for row in rows}
        for row in rows:
            self.assertEqual((row['runtime'], row['steps'], row['error']), ([42], 1, None))
            self.assertFalse(row['legacy_field'] or row['exposed'] or row['actor_exposed'])
            self.assertEqual(set(row['asset']), {'filename_seed', 'settings_seed', 'components', 'sidecar_sha256'})
        self.assertEqual(by_label[12274]['asset']['settings_seed'], 41708)
        self.assertEqual(by_label[12274]['asset']['components'],
                         dict(map=43616, objects=5808, quest=20492, grammar=18647))
        self.assertIsNone(by_label[37212]['asset']['settings_seed'])
        self.assertEqual(by_label[37212]['asset']['components'],
                         dict(map=20788, objects=44935, quest=3985, grammar=45656))

    def test_staging_changes_only_goal_forwarding_and_objective_filter(self):
        original = (VENDOR / 'envs_archive/textworld_envs.py').read_text()
        transformed = staged_environment_source(original)
        before, after = ast.parse(original), ast.parse(transformed)
        changed = []
        for old_class, new_class in zip(before.body, after.body):
            if not isinstance(old_class, ast.ClassDef):
                self.assertEqual(ast.dump(old_class), ast.dump(new_class))
                continue
            for old, new in zip(old_class.body, new_class.body):
                if ast.dump(old) != ast.dump(new):
                    changed.append((old_class.name, old.name))
        self.assertEqual(changed, [('Textworld_Env', 'get_prompt'),
                                   ('TextWorldWrapper', 'filter_objective')])
        with self.assertRaisesRegex(Rejected, 'no longer matches'):
            staged_environment_source(transformed)

    def test_goal_receipt_does_not_credit_memory_or_history(self):
        instruction, goal = 'fixed instructions', 'Find the current target.'
        cases = [
            ('fixed instructions' + goal, True),
            ('fixed instructions' + goal + '\nRelevant Experience: previous goal', True),
            ('fixed instructions\nRelevant Experience: ' + goal, False),
            ('fixed instructionsOther goal.', False),
        ]
        for system, expected in cases:
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': goal}]
            receipt = goal_delivery_receipt(messages, goal, instruction, 'request-hash')
            self.assertEqual(receipt['direct_goal_verified'], expected)
            self.assertEqual(receipt['request_hash'], 'request-hash')
        self.assertFalse(goal_delivery_receipt([], goal, instruction, 'x')['direct_goal_verified'])
        self.assertFalse(goal_delivery_receipt([{'role': 'system', 'content': instruction}],
                                             '', instruction, 'x')['direct_goal_verified'])
        self.assertFalse(summarize_goal_delivery([], goal)['all_actor_requests_verified'])

    def test_goal_delivery_separates_local_rejection_timeout_and_http_retries(self):
        goal = 'Find the target.'
        messages = [dict(role='system', content='instruction' + goal)]
        receipt = goal_delivery_receipt(messages, goal, 'instruction', 'bound')
        common = dict(request_hash='bound', messages=messages, task_input_receipt=receipt)
        success = dict(common, kind='model', ok=True, http_status=200)
        summary = summarize_goal_delivery([success], goal)
        self.assertTrue(summary['all_actor_requests_verified'])
        self.assertFalse(summarize_goal_delivery([dict(success, request_hash='other')], goal)
                         ['all_actor_requests_verified'])
        self.assertFalse(summarize_goal_delivery([dict(success, messages=[])], goal)
                         ['all_actor_requests_verified'])
        rows = [dict(common, kind='model', ok=False, error='ReadTimeout'),
                dict(common, kind='model', ok=False, error='HTTP503'), success,
                dict(common, kind='local_input_rejection', network_request_sent=False, ok=False)]
        summary = summarize_goal_delivery(rows, goal)
        self.assertEqual(summary['constructed_requests'], 4)
        self.assertEqual(summary['constructed_goals_verified'], 4)
        self.assertEqual(summary['local_rejections'], 1)
        self.assertEqual(summary['actor_requests'], 3)
        self.assertEqual(summary['server_responses'], 2)
        self.assertEqual(summary['delivery_unknown'], 1)
        self.assertEqual(summary['successful_actor_responses'], 1)
        self.assertEqual(summary['direct_goal_verified_requests'], 1)
        self.assertFalse(summary['all_actor_requests_verified'])

    def test_full_official_inventory_and_exact_split(self):
        manifest = build_manifest()
        self.assertEqual(manifest['total_games'], 50)
        self.assertEqual(len({task['id'] for task in manifest['tasks']}), 50)
        rng = random.Random(42)
        for category, suffix in [('treasure_hunter', '*.ulx'), ('the_cooking_game', '*.z8')]:
            paths = sorted((GAMES / category).glob(suffix))
            rng.shuffle(paths)
            actual = [task for task in manifest['tasks'] if task['family'] == category]
            self.assertEqual([Path(task['game_path']).name for task in actual], [path.name for path in paths])
            self.assertEqual([task['official_split'] for task in actual], ['train'] * 12 + ['eval_in_distribution'] * 13)
            self.assertEqual(manifest['counts'][category], {'train': 12, 'eval_in_distribution': 13})
            self.assertTrue(all(task['goal'] == '' for task in actual))
            self.assertTrue(all(task['seed'] == 42 for task in actual))

    def test_frozen_task_payload_and_asset_checks_reject_tampering(self):
        tasks = load_tasks('the_cooking_game')
        task = tasks[0]
        self.assertTrue(verify_task_assets(task).is_file())
        altered = copy.deepcopy(task)
        altered['game_sha256'] = '0' * 64
        with self.assertRaisesRegex(Rejected, 'compiled game changed'):
            verify_task_assets(altered)
        altered = copy.deepcopy(task)
        altered['game_path'] = '/tmp/non-official-game.z8'
        with self.assertRaisesRegex(Rejected, 'escaped'):
            verify_task_assets(altered)
        evaluator = TextWorldNativeEvaluator.__new__(TextWorldNativeEvaluator)
        from evomem.common import digest
        evaluator.task_payloads = {task['id']: digest(task)}
        evaluator.check_tasks([task])
        altered = copy.deepcopy(task)
        altered['seed'] += 1
        with self.assertRaisesRegex(Rejected, 'frozen native task set'):
            evaluator.check_tasks([altered])
        with self.assertRaisesRegex(Rejected, 'unique'):
            evaluator.check_tasks([task, task])

    def test_extracted_asset_change_is_not_blessed_by_rebuilding_manifest(self):
        from phase1_codegen import textworld_tasks
        original = textworld_tasks.file_hash
        def modified_hash(path):
            if Path(path).suffix == '.z8':
                return 'f' * 64
            return original(path)
        with patch.object(textworld_tasks, 'file_hash', side_effect=modified_hash):
            with self.assertRaisesRegex(Rejected, 'differs from the frozen official archive'):
                build_manifest()

    def test_actor_routes_to_gpu2_preserving_raw_prompt_and_parameters(self):
        """Synthetic HTTP response only; this test never calls a model or game."""
        captured = []
        class FakeTransport(httpx.AsyncBaseTransport):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            async def handle_async_request(self, request):
                captured.append((str(request.url), json.loads(request.content)))
                if len(captured) == 1:
                    raise httpx.ReadTimeout('simulated response timeout', request=request)
                if len(captured) == 2:
                    return httpx.Response(503, json={'error': 'simulated busy server'}, request=request)
                return httpx.Response(200, json={
                    'id': 'synthetic', 'object': 'chat.completion', 'created': 0, 'model': MODEL_NAME,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'look'},
                                 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 12, 'completion_tokens': 1, 'total_tokens': 13}}, request=request)
            async def aclose(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            runner = TextWorldALMARunner.__new__(TextWorldALMARunner)
            runner.llm = types.SimpleNamespace(base='http://127.0.0.1:8142/v1',
                fingerprint='synthetic-only', model_name=MODEL_NAME)
            runner.task_instructions = {'treasure_hunter': 'unchanged official system'}
            runner.tokenizer = Tokenizer()
            runner.ledger = Path(temporary) / 'costs.jsonl'
            runner.clients = []
            messages = [{'role': 'system', 'content': 'unchanged official systemFind the target.'},
                        {'role': 'user', 'content': 'unchanged native observation'}]
            async def request():
                token = CONTEXT.set(dict(task={'id': 'synthetic-task', 'family': 'treasure_hunter'},
                                        task_goal='Find the target.', phase='synthetic-test',
                                        model_seed=123, model_errors=0, truncated=0))
                try:
                    client = runner._make_client_factory()()
                    result = await client.chat.completions.create(model=MODEL_NAME, messages=messages)
                    self.assertEqual(result.choices[0].message.content, 'look')
                    await client.close()
                finally:
                    CONTEXT.reset(token)
            with patch.object(httpx, 'AsyncHTTPTransport', FakeTransport):
                asyncio.run(request())
            url, body = captured[0]
            self.assertEqual(url, 'http://127.0.0.1:8142/v1/chat/completions')
            self.assertEqual(body['messages'], messages)
            self.assertNotIn('response_format', body)
            for key, value in ACTOR_PARAMETERS.items():
                self.assertEqual(body[key], value)
            self.assertEqual(body['seed'], 123)
            rows = events(runner.ledger)
            self.assertEqual(len(captured), 3)
            self.assertEqual([row['http_status'] for row in rows], [None, 503, 200])
            summary = summarize_goal_delivery(rows, 'Find the target.')
            self.assertEqual(summary['constructed_requests'], 3)
            self.assertEqual(summary['server_responses'], 2)
            self.assertEqual(summary['delivery_unknown'], 1)
            self.assertEqual(summary['direct_goal_verified_requests'], 1)
            self.assertFalse(summary['all_actor_requests_verified'])
            receipt = rows[-1]
            self.assertTrue(receipt['ok'])
            self.assertEqual(receipt['prompt_tokens'], 12)
            self.assertEqual(receipt['answer'], 'look')
            self.assertEqual(receipt['base_url'], 'http://127.0.0.1:8142/v1')
            self.assertTrue(receipt['task_input_receipt']['direct_goal_verified'])
            self.assertEqual(receipt['task_input_receipt']['request_hash'], receipt['request_hash'])
            self.assertEqual(receipt['model_name'], MODEL_NAME)
            self.assertEqual(STEP_LIMIT, 80)


if __name__ == '__main__':
    unittest.main()
