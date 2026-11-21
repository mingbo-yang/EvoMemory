"""Audit pinned ALMA and exercise its REAL native environment, without LLM calls.

No Meta Agent, generated code, fake environments or synthesized task metadata.
The staged layout reproduces ALMA's three environment-file copies performed by
eval_in_container.py. Vendor files remain unchanged.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evomem.common import LEGACY, RUN, BASE_URL, file_hash, read, write

VENDOR=ROOT/'external/alma'
AUDIT=RUN/'alma_audit'
PIN='0a6ecb1f8ffbc704e15ac2c98a6973ea60caf70b'
METADATA_URL='https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip'
METADATA_SIZE=72018818
META=LEGACY/'downloads/json_2.1.1_json.zip'

def pinned():
    actual=subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()
    if actual!=PIN: raise RuntimeError('ALMA revision mismatch')
    dirty=subprocess.check_output(['git','-C',str(VENDOR),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty: raise RuntimeError('ALMA vendor checkout has tracked modifications')

def manifest():
    pinned();packages={}
    for name in ['openai','rich','jsonschema','scipy','python-dotenv','PyYAML','alfworld','textworld','gym','gymnasium','minihack','nle','colorlog']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    files=['evals/workflows/agent_workflow.py','evals/utils/hire_agent.py','evals/agents/base.py',
        'evals/agents/memo_structure.py','evals/launch.py','envs_archive/alfworld_envs.py',
        'envs_archive/textworld_envs.py','envs_archive/babaisai_envs.py','envs_archive/minihack_envs.py',
        'envs_archive/prompts/alfworld_prompt.py','envs_archive/configs/alfworld_config.yaml',
        'memo_archive/baseline/memo_structure_no_mem.py','requirements.txt']
    with zipfile.ZipFile(LEGACY/'downloads/alfworld-tw-0.4.2.zip') as z:
        data_types=dict(Counter(Path(n).name for n in z.namelist()))
    result=dict(commit=PIN,source='https://github.com/zksha/alma',python=sys.version,
        source_hashes={p:file_hash(VENDOR/p) for p in files},installed_packages=packages,
        existing_alfworld_archive_contents=data_types,official_metadata_url=METADATA_URL,
        official_metadata_bytes=METADATA_SIZE,meta_agent_executed=False,llm_calls=0,
        research_status='source audit; real model end-to-end not yet tested')
    write(AUDIT/'source_manifest.json',result);return result

def download_metadata():
    import requests
    META.parent.mkdir(parents=True,exist_ok=True)
    if META.exists():
        if META.stat().st_size!=METADATA_SIZE: raise RuntimeError('existing metadata size mismatch')
        return
    partial=META.with_suffix('.zip.partial')
    if partial.exists(): raise RuntimeError('partial download exists; inspect before retrying')
    start=time.monotonic()
    with requests.get(METADATA_URL,stream=True,timeout=(15,60)) as response:
        response.raise_for_status()
        with partial.open('wb') as f:
            for chunk in response.iter_content(1024*1024):
                f.write(chunk)
    if partial.stat().st_size!=METADATA_SIZE: raise RuntimeError('official asset length mismatch')
    with zipfile.ZipFile(partial) as z:
        names=z.namelist()
        if not any(n.endswith('/traj_data.json') for n in names): raise RuntimeError('missing actual trajectories')
    partial.rename(META)
    write(AUDIT/'metadata_download.json',dict(url=METADATA_URL,bytes=METADATA_SIZE,
        sha256=file_hash(META),elapsed=time.monotonic()-start,source='official ALFWorld release 0.2.2'))
    print(json.dumps(read(AUDIT/'metadata_download.json')),flush=True)

def stage(selected=None):
    pinned()
    if not META.exists(): raise RuntimeError('official traj_data.json missing; download official metadata first')
    target=AUDIT/'alfworld_runtime/evals'
    if not target.exists():
        shutil.copytree(VENDOR/'evals',target,ignore=shutil.ignore_patterns('.env','__pycache__','logs'))
    for src,dst in [('envs_archive/alfworld_envs.py','eval_envs/test_envs.py'),
                    ('envs_archive/prompts/alfworld_prompt.py','eval_envs/prompts/env_prompt.py'),
                    ('envs_archive/configs/alfworld_config.yaml','eval_envs/configs/env_configs.yaml')]:
        destination=target/dst;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and file_hash(destination)==file_hash(VENDOR/src): continue
        shutil.copyfile(VENDOR/src,destination)
    overrides={'eval_envs/test_envs.py','eval_envs/prompts/env_prompt.py'}
    for original in (VENDOR/'evals').rglob('*.py'):
        relative=original.relative_to(VENDOR/'evals')
        if str(relative) not in overrides and file_hash(target/relative)!=file_hash(original):
            raise RuntimeError('staged ALMA source drift: '+str(relative))
    if selected is None: selected=read(ROOT/'data/manifests/light_adaptation.json')[:2]
    data=target/'data/alfworld'
    for split in ('train','valid_seen','valid_unseen'): (data/'json_2.1.1'/split).mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(META) as meta, zipfile.ZipFile(LEGACY/'downloads/alfworld-tw-0.4.2.zip') as games:
        members=set(meta.namelist())
        for t in selected:
            game=t['archive_member'];traj=str(Path(game).with_name('traj_data.json'))
            if traj not in members: raise RuntimeError('no exact matching official trajectory for '+game)
            for name,source in ((game,games),(traj,meta)):
                p=Path(name)
                if p.is_absolute() or '..' in p.parts: raise ValueError('unsafe dataset member')
                destination=data/p;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(source.read(name))
    empty=AUDIT/'empty.env';empty.write_text('')
    os.environ.update(DOTENV_PATH=str(empty),OPENAI_API_KEY='local-not-a-secret',OPENAI_BASE_URL=BASE_URL)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.chdir(target);sys.path.insert(0,str(target))
    return target,selected

async def native_preflight():
    target,selected=stage()
    from eval_envs.test_envs import ALFWorld_Env
    from eval_envs.prompts.env_prompt import get_alfworld_prompt
    from utils.hire_agent import Agent
    from agents.base import init_global_tracker
    import importlib.util
    spec=importlib.util.spec_from_file_location('alma_official_no_memory',VENDOR/'memo_archive/baseline/memo_structure_no_mem.py')
    no_mem=importlib.util.module_from_spec(spec);spec.loader.exec_module(no_mem)
    memory=no_mem.SimilarityMemory();tracker=init_global_tracker()
    client=Agent(system_prompt='',model='qwen3.8-27b-local')
    endpoint=str(client.client.base_url)
    if endpoint.rstrip('/')!=BASE_URL: raise RuntimeError('ALMA client endpoint not loopback')
    await client.client.close()
    env=ALFWorld_Env(train_size=2,train_eval='train',update_task='train')
    checks=[]
    try:
        for t in selected:
            path=str(target/'data/alfworld'/t['archive_member']);initial=[];outcomes=[]
            for _ in range(2):
                state,recorder=await env.set_task_env(path)
                initial.append(state)
                retrieved=await memory.general_retrieve(recorder)
                if retrieved!={}: raise RuntimeError('official no-memory baseline not empty')
                prompt=await env.get_prompt(game_files=path,**state)
                if prompt!=get_alfworld_prompt(**state): raise RuntimeError('official prompt changed')
                if 'look' not in state['actions_list']: raise RuntimeError('expected native look unavailable')
                nxt,recorder=await env.run_step('look',path)
                await env.cal_reward(nxt['scores'],path)
                outcomes.append(dict(action='look',score=float(nxt['scores']),reward=float(recorder.reward),done=bool(nxt['dones'])))
                env.envs.pop(path).close()
            if initial[0]!=initial[1] or outcomes[0]!=outcomes[1]: raise RuntimeError('native reset/replay mismatch')
            checks.append(dict(task_id=t['id'],group_id=t['group_id'],initial_reset_equal=True,
                replay_equal=True,outcomes=outcomes,seed_control='upstream config42; explicit per-task seed hook not yet added'))
    finally:
        for instance in env.envs.values(): instance.close()
    if any(k.startswith('core.meta_agent') for k in sys.modules): raise RuntimeError('Meta Agent unexpectedly imported')
    result=dict(kind='real_native_environment_preflight',synthetic=False,llm_calls=0,
        tasks=len(checks),resets=len(checks)*2,native_actions=len(checks)*2,checks=checks,
        official_prompt_unchanged=True,official_client_endpoint=endpoint,token_tracker=tracker.summary(),
        meta_agent_imported=False,full_rollout_verified=False,success_path_verified=False,
        note='REAL ALMA/native ALFWorld reset, action, scoring; no model generation or task-success claim')
    write(AUDIT/'native_preflight.json',result);print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['manifest','download-metadata','native-preflight'])
    args=p.parse_args()
    if args.command=='manifest': print(json.dumps(manifest()),flush=True)
    elif args.command=='download-metadata': download_metadata()
    else: asyncio.run(native_preflight())

if __name__=='__main__': main()
