"""CPU checks of real transports and official ALMA exception handling; no model calls."""
import ast
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import traceback
import types
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import patch

import httpx
import openai

from evomem import alma_backend
from evomem.actor_transport import ACTOR_CONTEXT_TOKENS, ACTOR_RESERVED_TOKENS
from evomem.common import events
from phase1_codegen import textworld_native
from phase1_codegen.tests import test_hook_adapter as hook_fixture

ROOT = Path(__file__).resolve().parents[2]


class FixedTokenizer:
    def __init__(self, tokens):
        self.tokens, self.calls = tokens, 0
    def chat_count(self, messages):
        self.calls += 1
        return self.tokens


def transport_class(module, outer, method, owner):
    """Compile the actual nested class, without staging a native environment."""
    tree = ast.parse(Path(module.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == outer)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    node = next(n for n in method.body if isinstance(n, ast.ClassDef) and n.name == 'TrackedTransport')
    scope = dict(vars(module), owner=owner, httpx=httpx)
    exec(compile(ast.Module(body=[node], type_ignores=[]), module.__file__, 'exec'), scope)
    return scope['TrackedTransport']


def official_objects(client, observed_agents):
    agent_path = ROOT/'external/alma/evals/utils/hire_agent.py'
    agent_node = next(n for n in ast.parse(agent_path.read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'Agent')
    scope = dict(asyncio=asyncio, os=os, json=json, List=List, Dict=Dict, Any=Any,
                 Optional=Optional, AsyncOpenAI=lambda **kwargs: client)
    exec(compile(ast.Module(body=[agent_node], type_ignores=[]), str(agent_path), 'exec'), scope)
    def agent(*args, **kwargs):
        value = scope['Agent'](*args, **kwargs)
        observed_agents.append(value)
        return value
    workflow_path = ROOT/'external/alma/evals/workflows/agent_workflow.py'
    workflow = next(n for n in ast.parse(workflow_path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'Agent_Workflow')
    method = next(n for n in workflow.body if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == 'run_single_task')
    loop_scope = dict(Agent=agent, asyncio=asyncio, traceback=traceback)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(workflow_path), 'exec'), loop_scope)
    return loop_scope['run_single_task']


@unittest.skipUnless((ROOT/'external/alma/evals/workflows/agent_workflow.py').is_file(), 'Pinned ALMA checkout required; run python -m scripts.prepare_assets alma')
class ActorContextBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, environment, tokens, *, official_loop=False):
        sent, agents = [], []
        class FakeNetwork(httpx.AsyncBaseTransport):
            def __init__(self, **kwargs): pass
            async def handle_async_request(self, request):
                sent.append(json.loads(request.content))
                return httpx.Response(200, request=request, json={
                    'id':'synthetic', 'object':'chat.completion', 'created':0, 'model':'synthetic',
                    'choices':[{'index':0,'message':{'role':'assistant','content':'look'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':tokens,'completion_tokens':1,'total_tokens':tokens+1}})
            async def aclose(self): pass
        with tempfile.TemporaryDirectory() as temporary:
            owner = types.SimpleNamespace(tokenizer=FixedTokenizer(tokens),
                ledger=Path(temporary)/'actor.jsonl',
                task_instructions={'treasure_hunter':'fixed instructions'},
                llm=types.SimpleNamespace(base='http://never-network/v1',
                                         model_name='synthetic',fingerprint='synthetic'))
            if environment == 'textworld':
                transport = transport_class(textworld_native,'TextWorldALMARunner','_make_client_factory',owner)
            else:
                transport = transport_class(alma_backend,'ALMARunner','__init__',owner)
            context = dict(task={'id':'synthetic-task','family':'treasure_hunter'}, task_goal='Find target.',
                           phase='cpu-boundary',model_seed=17,model_errors=0,truncated=0)
            token = alma_backend.CONTEXT.set(context)
            with patch.object(httpx,'AsyncHTTPTransport',FakeNetwork):
                client = openai.AsyncOpenAI(api_key='synthetic-only',base_url=owner.llm.base,
                    max_retries=2,http_client=httpx.AsyncClient(transport=transport()))
            messages = [{'role':'system','content':'fixed instructionsFind target.'},
                        {'role':'user','content':'native visible input'}]
            error = None
            try:
                if official_loop:
                    class Recorder:
                        async def log_memo_retrieved(self, value): pass
                    recorder=Recorder()
                    class Env:
                        max_trails=80
                        recorders={'synthetic-task':recorder}
                        async def set_task_env(self, task): return {'obs':'native'},recorder
                        async def get_prompt(self, **kwargs): return messages
                        async def run_step(self, *args, **kwargs): raise AssertionError('no action after context rejection')
                        async def cal_reward(self, *args, **kwargs): raise AssertionError('no completed native reward')
                    loop=official_objects(client,agents)
                    workflow=types.SimpleNamespace(model='synthetic',reasoning_effort=None,
                                                    env=Env(),memo=[object()])
                    with contextlib.redirect_stdout(io.StringIO()):
                        await loop(workflow,'synthetic-task',eval_type='batched',current_state='update')
                else:
                    result=await client.chat.completions.create(model='synthetic',messages=messages)
                    self.assertEqual(result.choices[0].message.content,'look')
            except Exception as exc:
                error=exc
            finally:
                await client.close()
                alma_backend.CONTEXT.reset(token)
            return dict(error=error,sent=sent,rows=events(owner.ledger),context=context,
                        tokenizer_calls=owner.tokenizer.calls,agents=agents,messages=messages)

    async def test_exact_context_boundary_keeps_original_request_and_budget(self):
        self.assertEqual((ACTOR_CONTEXT_TOKENS,ACTOR_RESERVED_TOKENS),(32768,1024))
        for environment in ('alfworld','textworld'):
            with self.subTest(environment=environment):
                result=await self.run_case(environment,31744)
                self.assertIsNone(result['error'])
                self.assertEqual(len(result['sent']),1)
                self.assertEqual(result['sent'][0]['messages'],result['messages'])
                self.assertEqual(result['sent'][0]['max_tokens'],512)
                self.assertEqual(result['tokenizer_calls'],1)
                self.assertEqual(result['context']['model_errors'],0)
                self.assertEqual(result['rows'][0]['kind'],'model')

    async def test_overflow_returns_context_error_once_without_sdk_retry_or_network(self):
        for environment in ('alfworld','textworld'):
            with self.subTest(environment=environment):
                result=await self.run_case(environment,31745)
                self.assertIsInstance(result['error'],openai.BadRequestError)
                self.assertIn('maximum context length',str(result['error']))
                self.assertEqual(result['sent'],[])
                self.assertEqual(result['tokenizer_calls'],1)
                self.assertEqual(result['context']['model_errors'],1)
                self.assertEqual(len(result['rows']),1)
                row=result['rows'][0]
                self.assertEqual(row['kind'],'local_input_rejection')
                self.assertFalse(row['network_request_sent'])
                self.assertEqual(row['estimated_prompt_tokens'],31745)
                self.assertNotIn('prompt_tokens',row)
                self.assertNotIn('completion_tokens',row)
                self.assertEqual(row['messages'],result['messages'])
                if environment=='textworld':
                    self.assertTrue(row['task_input_receipt']['direct_goal_verified'])

    async def test_official_loop_terminates_on_first_rejection_without_feedback_growth(self):
        for environment in ('alfworld','textworld'):
            with self.subTest(environment=environment):
                result=await self.run_case(environment,31745,official_loop=True)
                self.assertIsInstance(result['error'],ValueError)
                self.assertEqual(str(result['error']),'Prompt exceeded model context limit. ')
                self.assertEqual(result['tokenizer_calls'],1)
                self.assertEqual(len(result['rows']),1)
                self.assertEqual(result['sent'],[])
                self.assertEqual(len(result['agents'][0].messages),2)
                self.assertEqual(result['agents'][0].messages[-1]['content'],'native visible input')

    async def test_context_failure_does_not_commit_task_memory_or_verify_partial_score(self):
        fixture=hook_fixture.HookAdapterTests();fixture.setUp()
        with tempfile.TemporaryDirectory() as directory:
            bank,transaction,episode=fixture.staged_native(directory,allow_write=True)
            before=bank.export_snapshot()
            episode.update(error='ValueError: Prompt exceeded model context limit. ',
                           native_reward_verified=False,done=False,score=0.)
            result=transaction.finalize(episode)
            self.assertEqual(bank.export_snapshot(),before)
            self.assertFalse(result['memory_hook']['committed'])
            self.assertFalse(result['native_reward_verified'])
            self.assertEqual(result['generated_memory_write_calls'],0)
            self.assertEqual(len(events(Path(directory)/'canonical_episodes.jsonl')),1)


if __name__=='__main__':unittest.main()
