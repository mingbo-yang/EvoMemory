"""ALMA's action loop with a current-action-list instruction and local model transport."""
from __future__ import annotations
import asyncio
from contextvars import ContextVar
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import re
import time

from .common import BASE_URL, MODEL_NAME, append, digest, dumps, events, write
from .actor_transport import context_limit_response

CONTEXT=ContextVar('phase1_alma_episode',default=None)
ACTOR_PARAMETERS=dict(temperature=0,top_p=1,max_tokens=512,
                      chat_template_kwargs={'enable_thinking':False})
ALFWORLD_ACTOR_PARAMETERS = {**ACTOR_PARAMETERS, 'stop': ['\n']}

ALFWORLD_ACTION_POLICY = (
    "Complete the household task described in the observations. "
    "At each step, choose exactly one line from the <Actions> list in the latest observation. "
    "Copy that line verbatim as your entire reply, including its verb and object numbers.\n"
)


def _alfworld_actor_prompt(get_prompt, upstream_policy):
    """Keep ALMA's native observation, command list and rendered memory intact."""
    async def render(*args, **kwargs):
        messages = await get_prompt(*args, **kwargs)
        system = messages[0]
        if system['role'] != 'system' or not system['content'].startswith(upstream_policy):
            raise ValueError('Unexpected ALFWorld system prompt')
        memory_suffix = system['content'][len(upstream_policy):]
        return [dict(system, content=ALFWORLD_ACTION_POLICY + memory_suffix), *messages[1:]]

    return render


def goal_from_initial(initial):
    if initial.get('goal'): return str(initial['goal'])
    obs=str(initial.get('obs',''))
    match=re.search(r'(?:your task is to|your task is|task goal|task)\s*:\s*([^\n]+)',obs,re.I)
    return match.group(1).strip() if match else obs

def public_context(initial):
    return {k:initial[k] for k in ('obs','goal','actions_list','long_term_context','short_term_context','action_dict') if k in initial}

def experience(recorder,task,phase,seed):
    initial=public_context(recorder.init);before=str(initial.get('obs',dumps(initial)))
    available=initial.get('actions_list',[]);history=[]
    for s in recorder.steps:
        after=str(s.get('obs',dumps({k:s[k] for k in ('long_term_context','short_term_context') if k in s})))
        action=s['action_took']
        history.append(dict(observation=before,admissible=list(available),action=action,
                            next_observation=after,listed_exactly=action in available))
        before=after;available=s.get('actions_list',[])
    return dict(task_id=task['id'],group_id=task['group_id'],room=task['room'],family=task['family'],
                goal=goal_from_initial(initial),history=history,score=float(recorder.reward),
                done=bool(recorder.steps[-1].get('dones',False)) if recorder.steps else False,
                error=None,phase=phase,seed=seed)

class EvolvableMemoryStore:
    """Host adapter for ALMA's fixed memory slot; mechanisms run through on_event."""
    def __init__(self,memory,tokenizer,allow_write=False):
        self.memory=memory;self.tokenizer=tokenizer;self.allow_write=allow_write
        self.reads=[];self.writes=[];self.read_tokens=0

    def read(self,initial):
        initial=public_context(initial);goal=goal_from_initial(initial)
        state=str(initial.get('obs',dumps(initial)))
        entries=self.memory.retrieve(goal,state)
        result={'entries':entries} if entries else {}
        # Count the exact pretty-JSON representation used by ALMA's official prompt.
        while entries and self.tokenizer.count(json.dumps(result,ensure_ascii=False,indent=2))>self.memory.read_budget:
            last=entries[-1];n=self.tokenizer.count(last['experience'])
            if n<48: entries.pop()
            else: last['experience']=self.tokenizer.clip(last['experience'],n-32)
            result={'entries':entries} if entries else {}
        count=self.tokenizer.count(json.dumps(result,ensure_ascii=False,indent=2))
        self.read_tokens+=count;self.reads.append(dict(context_hash=digest(initial),ids=[r['id'] for r in entries],tokens=count))
        return result

    def write(self,ep):
        if not self.allow_write: raise ValueError('Task Memory writes forbidden in probe/final')
        # Episode-level ingestion; the next task rebuilds from this same raw experience.
        self.writes.append(copy.deepcopy(ep))


class ALMARunner:
    def __init__(self,llm,tokenizer,ledger,selected_tasks,concurrency=2):
        from scripts.alma_feasibility import stage, VENDOR
        self.llm=llm;self.tokenizer=tokenizer;self.ledger=Path(ledger);self.concurrency=concurrency
        self.layout,_=stage(selected_tasks)
        from workflows.agent_workflow import Agent_Workflow
        from agents.base import init_global_tracker
        import utils.hire_agent as hire
        import httpx
        import openai
        self.clients=[];self.tracker=init_global_tracker()
        owner=self

        class TrackedTransport(httpx.AsyncBaseTransport):
            def __init__(self): self.inner=httpx.AsyncHTTPTransport(trust_env=False)
            async def handle_async_request(self,request):
                ctx=CONTEXT.get()
                if ctx is None: raise RuntimeError('unscoped ALMA request')
                if str(request.url)!=owner.llm.base+'/chat/completions': raise ValueError('nonlocal or unexpected model route')
                body=json.loads(request.content)
                if body.get('model')!=owner.llm.model_name: raise ValueError('unexpected actor model')
                if 'response_format' in body: raise ValueError('ALMA actor must retain raw action output')
                step=sum(m['role']=='user' for m in body['messages'])-1
                body.update(ALFWORLD_ACTOR_PARAMETERS,seed=ctx['model_seed']+step)
                rejected=context_limit_response(request,body,tokenizer=owner.tokenizer,
                    context=ctx,ledger=owner.ledger,model=owner.llm)
                if rejected is not None: return rejected
                modified=httpx.Request(request.method,request.url,
                    headers={k:v for k,v in request.headers.items() if k.lower()!='content-length'},content=dumps(body).encode())
                started=time.monotonic();data=None;ok=False;error=None
                try:
                    response=await self.inner.handle_async_request(modified)
                    raw=await response.aread()
                    try: data=json.loads(raw)
                    except (ValueError,UnicodeError): data=None
                    ok=response.status_code==200 and isinstance(data,dict) and bool(data.get('choices'))
                    if not ok: error='HTTP'+str(response.status_code)
                    if ok and data['choices'][0].get('finish_reason')=='length':
                        ctx['truncated']+=1
                    return response
                except Exception as exc:
                    error=type(exc).__name__;raise
                finally:
                    if not ok: ctx['model_errors']+=1
                    usage=(data or {}).get('usage',{})
                    choices=(data or {}).get('choices') or [{}]
                    append(owner.ledger,dict(kind='model',role='actor',phase=ctx['phase'],task_id=ctx['task']['id'],
                        seed=body['seed'],ok=ok,error=error,elapsed=time.monotonic()-started,
                        prompt_tokens=usage.get('prompt_tokens'),completion_tokens=usage.get('completion_tokens'),
                        request_hash=digest(body),model_fingerprint=owner.llm.fingerprint,model_name=owner.llm.model_name,
                        messages=body['messages'],
                        finish_reason=choices[0].get('finish_reason'),
                        answer=choices[0].get('message',{}).get('content'),
                        original_message_hash=digest(body['messages'])))
            async def aclose(self): await self.inner.aclose()

        def factory(**kwargs):
            client=openai.AsyncOpenAI(api_key='local-not-a-secret',base_url=owner.llm.base,timeout=300,max_retries=2,
                http_client=httpx.AsyncClient(transport=TrackedTransport(),trust_env=False,timeout=300,follow_redirects=False))
            owner.clients.append(client);return client
        # The official Agent class/methods remain unchanged; replace only its client factory.
        self.client_factory=factory
        hire.AsyncOpenAI=factory
        spec=importlib.util.spec_from_file_location('phase1_official_no_memory',VENDOR/'memo_archive/baseline/memo_structure_no_mem.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.NoMemory=module.SimilarityMemory
        self.workflow=Agent_Workflow([self.NoMemory()],self.llm.model_name,train_size=max(1,len(selected_tasks)),status='train',update_task='train')
        self.workflow.init_env('alfworld')
        from eval_envs.prompts.env_prompt import ALFWORLD_BASIC_PROMPT
        self.workflow.env.get_prompt = _alfworld_actor_prompt(self.workflow.env.get_prompt, ALFWORLD_BASIC_PROMPT)
        if self.workflow.env.max_trails!=30: raise ValueError('ALMA episode budget drift')
        upstream_clone=self.workflow.env._clone_env
        def seeded_clone():
            clone=upstream_clone();upstream_init=clone.init_env
            def init(*args,**kwargs):
                native=upstream_init(*args,**kwargs)
                native.seed(CONTEXT.get()['task']['seed'])
                return native
            clone.init_env=init;return clone
        self.workflow.env._clone_env=seeded_clone

    async def _episode(self,task,memory,phase,seed,no_memory=False,allow_write=False):
        from rich.progress import Progress
        import utils.hire_agent as hire
        hire.AsyncOpenAI=self.client_factory
        started=time.monotonic();ctx=dict(task=task,phase=phase,model_seed=seed,model_errors=0,truncated=0)
        token=CONTEXT.set(ctx);path=str(self.layout/'data/alfworld'/task['archive_member'])
        store=EvolvableMemoryStore(memory,self.tokenizer,allow_write)
        snapshot=self.NoMemory() if no_memory else self.Memory(store,task,phase,seed)
        recorder=None;error=None
        try:
            with Progress(disable=True) as progress:
                tid=progress.add_task(task['id'],total=30)
                recorder=await self.workflow.run_single_task(path,eval_type='sequential' if allow_write else 'batched',
                    current_state='retrieve',progress=progress,task_id=tid,memory_snapshot=snapshot)
        except Exception as exc:
            error=type(exc).__name__
            append(self.ledger,dict(kind='episode_error',task_id=task['id'],phase=phase,error=error))
            recorder=self.workflow.env.recorders.get(path)
        finally:
            native=self.workflow.env.envs.pop(path,None)
            if native is not None:
                try: native.close()
                except Exception as exc: error=error or 'Close'+type(exc).__name__
            self.workflow.env.recorders.pop(path,None)
            for client in self.clients:
                if not client.is_closed(): await client.close()
            self.clients.clear();CONTEXT.reset(token)
        if recorder is not None: ep=experience(recorder,task,phase,seed)
        else: ep=dict(task_id=task['id'],group_id=task['group_id'],room=task['room'],family=task['family'],
            goal=task['goal'],history=[],score=None,done=False,error=error,phase=phase,seed=seed)
        ep.update(error=error or ('ModelRequestError' if ctx['model_errors'] else None) or ('OutputTruncated' if ctx['truncated'] else None),
            native_reward_verified=recorder is not None and error is None,
            elapsed=time.monotonic()-started,memory_hash=memory.fingerprint if not no_memory else 'official-no-memory',
            retrieved={i:1 for r in store.reads for i in r['ids']},read_tokens=store.read_tokens,
            memory_read_calls=len(store.reads),memory_write_calls=len(store.writes),
            executor='ALMA.Agent_Workflow.run_single_task',synthetic=False,environment_seed=task['seed'])
        if recorder is not None:
            ep['native_recorder']=dict(initial=public_context(recorder.init),steps=copy.deepcopy(recorder.steps),
                reward=float(recorder.reward),memory_retrieved=copy.deepcopy(getattr(recorder,'memory_retrieved',{})))
        if ep.get('error') or not ep['native_reward_verified']:
            ep.update(score=None, native_reward_verified=False)
        for i in range(len(ep['history'])): append(self.ledger,dict(kind='env_step',phase=phase,task_id=task['id'],step=i))
        append(self.ledger.with_name('episodes.jsonl'),ep)
        print(dumps(dict(episode=task['id'],phase=phase,score=ep['score'],error=ep['error'],
                         steps=len(ep['history']),seconds=round(ep['elapsed'],2))),flush=True)
        return ep

    def episode(self,task,memory,phase,seed=42000,no_memory=False,allow_write=False):
        return asyncio.run(self._episode(task,memory,phase,seed,no_memory,allow_write))


