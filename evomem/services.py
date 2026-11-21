"""Shared local tokenizer and model transport; role prompts live in phase1_codegen."""
from __future__ import annotations
from functools import lru_cache
import ipaddress
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from .common import BASE_URL, MODEL, MODEL_NAME, append, digest, dumps


class Tokenizer:
    def __init__(self,path=MODEL):
        from transformers import AutoTokenizer
        self.impl=AutoTokenizer.from_pretrained(str(path),local_files_only=True)

    @lru_cache(maxsize=8192)
    def count(self,text): return len(self.impl.encode(text,add_special_tokens=False))

    def clip(self,text,n):
        ids=self.impl.encode(text,add_special_tokens=False)
        if len(ids)<=n: return text
        if n<12: return ''
        marker=self.impl.encode('\n[readout clipped]\n',add_special_tokens=False)
        available=max(0,n-len(marker));a=available//2
        return self.impl.decode(ids[:a]+marker+ids[-(available-a):],skip_special_tokens=True)

    def chat_count(self,messages):
        return len(self.impl.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False))

class LocalLLM:
    def __init__(self,tokenizer,ledger,model_fingerprint,base_url=BASE_URL,*,model_name=MODEL_NAME):
        import httpx
        u=urlsplit(base_url)
        if u.scheme!='http' or not ipaddress.ip_address(u.hostname).is_loopback or u.path!='/v1':
            raise ValueError('only local loopback model allowed')
        if type(model_name) is not str or not model_name.strip():
            raise ValueError('explicit nonempty model name required')
        self.model_name=model_name
        self.client=httpx.Client(timeout=300,trust_env=False,follow_redirects=False)
        self.base=base_url;self.tokenizer=tokenizer;self.ledger=Path(ledger)
        self.fingerprint=model_fingerprint

    def call(self,system,payload,role,seed,phase):
        messages=[dict(role='system',content=system),dict(role='user',content=dumps(payload))]
        out=512 if role=='actor' else 2048
        n=self.tokenizer.chat_count(messages)
        if n+out+512>32768: raise ValueError('input context budget exceeded')
        request=dict(model=self.model_name,messages=messages,temperature=0 if role=='actor' else .7,
                     max_tokens=out,seed=seed,response_format={'type':'json_object'},
                     chat_template_kwargs={'enable_thinking':False})
        started=time.monotonic();result=None;ok=False;error=None;content=None
        try:
            response=self.client.post(self.base+'/chat/completions',json=request)
            response.raise_for_status();result=response.json();choice=result['choices'][0]
            content=choice['message']['content']
            if choice['finish_reason']=='length': raise ValueError('output truncated')
            value=json.loads(content)
            if not isinstance(value,dict): raise ValueError('response is not object')
            ok=True;return value
        except Exception as e:
            error=type(e).__name__;raise
        finally:
            usage=result.get('usage',{}) if result else {}
            append(self.ledger,dict(kind='model',phase=phase,role=role,seed=seed,ok=ok,error=error,
                elapsed=time.monotonic()-started,request_hash=digest(request),
                prompt_tokens=usage.get('prompt_tokens'),completion_tokens=usage.get('completion_tokens'),
                local_prompt_tokens=n,finish_reason=result['choices'][0].get('finish_reason') if result else None,
                content=content if not ok else None,model_fingerprint=self.fingerprint))
