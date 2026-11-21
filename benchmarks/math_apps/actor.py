"""Single-turn official ALMA Agent.ask with an explicitly bound local transport."""
import json
import os
from pathlib import Path
import sys
import time

import httpx
from openai import AsyncOpenAI

from evomem.common import BASE_URL, MODEL_NAME, digest, write

ROOT = Path(__file__).resolve().parents[2]
URL = BASE_URL
PARAMETERS = dict(temperature=0.0, top_p=1.0, max_tokens=4096,
                  chat_template_kwargs={'enable_thinking': False})
PROMPTS = {
    'math': r'Solve the mathematics problem. Show your reasoning and put the final answer in \boxed{}.',
    'code': 'Solve the programming problem in Python 3. Respect the requested function signature or input/output format. Return only your complete program in a single python code block.',
}

def messages_for(domain, prompt, slot):
    system = PROMPTS[domain] + '\n\nPrevious task memory:\n' + json.dumps(slot, ensure_ascii=False)
    return [dict(role='system',content=system),dict(role='user',content=prompt)]

async def ask(domain, prompt, slot, seed, file, model_hash):
    messages = messages_for(domain, prompt, slot)
    expected = dict(model=MODEL_NAME, messages=messages, seed=seed, **PARAMETERS)
    if file.exists():
        from evomem.common import read
        saved=read(file)
        if saved['request']!=expected or saved['model_hash']!=model_hash or not saved['ok']:
            raise RuntimeError('Saved Actor receipt cannot be replayed')
        return saved
    os.environ['OPENAI_API_KEY']='local-not-a-secret'
    os.environ['OPENAI_BASE_URL']=URL
    os.environ['DOTENV_PATH']='/dev/null'
    vendor=str(ROOT/'external/alma/evals')
    if vendor not in sys.path: sys.path.insert(0,vendor)
    from utils.hire_agent import Agent

    class BoundTransport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            sent=json.loads(request.content)
            if str(request.url)!=URL+'/chat/completions' or sent != dict(model=MODEL_NAME,messages=messages):
                raise RuntimeError('Unexpected ALMA actor endpoint, messages or parameters')
            headers=dict(request.headers)
            headers.pop('content-length',None)
            sent.update(PARAMETERS,seed=seed)
            wire=httpx.Request('POST',request.url,headers=headers,json=sent,extensions=request.extensions)
            started=time.time()
            record=dict(request=sent,request_hash=digest(sent),model_hash=model_hash,endpoint=URL,start=started,ok=False)
            try:
                response=await super().handle_async_request(wire)
                record['http_status']=response.status_code
                raw=json.loads(await response.aread())
                if response.status_code!=200:
                    raise RuntimeError(f'Actor HTTP {response.status_code}')
                choice=raw['choices'][0]
                record.update(answer=choice['message']['content'], finish=choice['finish_reason'],
                              usage=raw['usage'], response_id=raw['id'], served_model=raw['model'])
                record['ok']=isinstance(record['answer'],str) and bool(record['answer'].strip())
                return response
            except BaseException as exc:
                record['error']=f'{type(exc).__name__}: {exc}'
                raise
            finally:
                record['end']=time.time()
                write(file,record)
    agent=Agent(messages[0]['content'],model=MODEL_NAME)
    await agent.client.close()
    agent.client=AsyncOpenAI(api_key='local-not-a-secret',base_url=URL,max_retries=0,
        http_client=httpx.AsyncClient(transport=BoundTransport(),trust_env=False,timeout=600),timeout=600)
    try:
        answer=await agent.ask(prompt,with_history=False)
    finally:
        await agent.client.close()
    from evomem.common import read
    result=read(file)
    if not result['ok'] or result['answer']!=answer or result['request']!=expected:
        raise RuntimeError('ALMA completion does not match HTTP response')
    return result
