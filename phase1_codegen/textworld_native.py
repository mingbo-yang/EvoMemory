"""The pinned ALMA TextWorld workflow with generated event memory.

The official Agent, action loop, prompt formatter, raw action parsing, and
80-step limit are unchanged. The staged environment restores the native visible
goal in its prompt and observations. Local transport, reset seeding, provenance
and the memory interface are adapted. TextWorld 1.6.2 is loaded from an isolated
dependency directory because the official Treasure Hunter assets are Glulx.
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evomem.alma_backend import (ACTOR_PARAMETERS, CONTEXT, EvolvableMemoryStore,
                                experience, public_context)
from evomem.actor_transport import context_limit_response
from evomem.common import ROOT, MODEL_NAME, append, digest, dumps, file_hash, write
from .local_service import make_local_llm
from .memory import CodeMemory
from .native import (code_bridge_class, fixed_task_seed, EpisodeMemory, validate_environment_context,
                     validate_environment_metadata)
from .policy import Rejected, json_copy
from .program import BASELINE_SOURCE, Program
from .textworld_tasks import (ASSET_SEED_POLICY, GAMES, OFFICIAL_COMMIT, asset_seed_receipt,
                              build_manifest, verify_task_assets)
from .textworld_input import TASK_INPUT_VERSION, goal_delivery_receipt, staged_environment_source


VENDOR = ROOT / 'external/alma'
TEXTWORLD16 = ROOT / 'external/textworld16'
STEP_LIMIT = 80
SOURCE_OVERRIDES = {
    'eval_envs/test_envs.py': 'envs_archive/textworld_envs.py',
    'eval_envs/prompts/env_prompt.py': 'envs_archive/prompts/textworld_prompt.py',
    'eval_envs/configs/env_configs.yaml': 'envs_archive/configs/textworld_config.yaml',
}


def prepare_textworld():
    """Select the isolated native dependency before importing official modules."""
    if not (TEXTWORLD16 / 'textworld/__init__.py').is_file():
        raise Rejected('isolated TextWorld 1.6.2 is missing; install the pinned wheel into external/textworld16')
    existing = sys.modules.get('textworld')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(TEXTWORLD16.resolve()):
        raise Rejected('another TextWorld dependency is already imported; use a fresh TextWorld process')
    if str(TEXTWORLD16) not in sys.path:
        sys.path.insert(0, str(TEXTWORLD16))
    textworld = importlib.import_module('textworld')
    importlib.import_module('textworld.gym')
    if textworld.__version__ != '1.6.2':
        raise Rejected('TextWorld runtime version drift')
    return textworld


def stage_textworld(root):
    """Reproduce ALMA's environment-file staging without changing the checkout."""
    prepare_textworld()
    actual_commit = subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != OFFICIAL_COMMIT:
        raise Rejected('official ALMA checkout is not at the pinned commit')
    dirty = subprocess.check_output(['git', '-C', str(VENDOR), 'status', '--porcelain',
                                     '--untracked-files=no'], text=True)
    if dirty.strip():
        raise Rejected('official ALMA tracked files were modified')
    source_hashes = {}
    for source in sorted((VENDOR / 'evals').rglob('*')):
        if source.is_file() and source.suffix in ('.py', '.yaml'):
            source_hashes[str(source.relative_to(VENDOR))] = file_hash(source)
    for source in SOURCE_OVERRIDES.values():
        source_hashes[source] = file_hash(VENDOR / source)
    corrected_environment = staged_environment_source(
        (VENDOR / SOURCE_OVERRIDES['eval_envs/test_envs.py']).read_text())
    loaded = sys.modules.get('eval_envs.test_envs')
    if loaded is not None:
        # Multiple evaluators may share the identical official TextWorld module
        # within one process. Mixing an ALFWorld stage is explicitly forbidden.
        layout = Path(loaded.__file__).resolve().parents[1]
        if not hasattr(loaded, 'Textworld_Env'):
            raise Rejected('ALMA environment modules already belong to another dataset; use separate processes')
        expected = {**SOURCE_OVERRIDES, 'workflows/agent_workflow.py': 'evals/workflows/agent_workflow.py',
                    'utils/hire_agent.py': 'evals/utils/hire_agent.py'}
        for target, source in expected.items():
            actual = (layout / target).read_text()
            expected_text = (corrected_environment if target == 'eval_envs/test_envs.py'
                             else (VENDOR / source).read_text())
            if actual != expected_text:
                raise Rejected('already-loaded ALMA TextWorld code differs from frozen official source')
    else:
        layout = Path(root).resolve() / 'alma_textworld_runtime/evals'
        shutil.copytree(VENDOR / 'evals', layout, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for target, source in SOURCE_OVERRIDES.items():
            destination = layout / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(VENDOR / source, destination)
        (layout / 'eval_envs/test_envs.py').write_text(corrected_environment)
        games_link = layout.parent / 'balrog/tw_games'
        games_link.parent.mkdir(parents=True, exist_ok=True)
        if games_link.is_symlink():
            if games_link.resolve() != GAMES.resolve():
                raise Rejected('TextWorld staging points to a different game inventory')
        elif games_link.exists():
            raise Rejected('TextWorld staging game path is not the expected data symlink')
        else:
            games_link.symlink_to(GAMES.resolve(), target_is_directory=True)
        sys.path.insert(0, str(layout))
    empty_env = Path(root).resolve() / 'empty.env'
    empty_env.parent.mkdir(parents=True, exist_ok=True)
    empty_env.write_text('# Deliberately empty: only the explicit local service is used.\n')
    os.environ['DOTENV_PATH'] = str(empty_env)
    os.environ['OPENAI_API_KEY'] = 'local-not-a-secret'
    return layout, source_hashes


class TextWorldALMARunner:
    def __init__(self, llm, tokenizer, ledger, tasks, root):
        self.llm, self.tokenizer, self.ledger = llm, tokenizer, Path(ledger)
        self.layout, self.source_hashes = stage_textworld(root)
        previous = Path.cwd()
        try:
            os.chdir(self.layout)
            from workflows.agent_workflow import Agent_Workflow
            from agents.base import init_global_tracker
            from eval_envs.prompts.env_prompt import intruction_prompts
            import utils.hire_agent as hire
            self.task_instructions = dict(intruction_prompts)
            self.tracker = init_global_tracker()
            self.clients = []
            self.client_factory = self._make_client_factory()
            hire.AsyncOpenAI = self.client_factory
            spec = importlib.util.spec_from_file_location(
                'phase1_textworld_official_no_memory', VENDOR / 'memo_archive/baseline/memo_structure_no_mem.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.NoMemory = module.SimilarityMemory
            self.Memory = code_bridge_class('textworld')
            self.workflow = Agent_Workflow([self.NoMemory()], self.llm.model_name,
                                           train_size=max(1, len(tasks)), status='train', update_task='train')
            self.workflow.init_env('textworld')
            if self.workflow.env.max_trails != STEP_LIMIT:
                raise Rejected('official TextWorld episode limit drift')
        finally:
            os.chdir(previous)
        self.textworld = prepare_textworld()
        self.runtime_ids = {}
        self.asset_seeds = {}
        for ids in self.workflow.env.category_to_tasks.values():
            for runtime_id in ids:
                files = self.textworld.gym.registry[runtime_id]['kwargs']['gamefiles']
                if len(files) != 1:
                    raise Rejected('official single-game TextWorld registration changed')
                self.runtime_ids[str(Path(files[0]).resolve())] = runtime_id
        for task in tasks:
            if str(verify_task_assets(task)) not in self.runtime_ids:
                raise Rejected('frozen task is not registered in the official environment')
        original_set = self.workflow.env.set_task_env

        async def seeded_set(task_path):
            original_make = self.textworld.gym.make

            def seeded_make(*args, **kwargs):
                native = original_make(*args, **kwargs)
                ctx = CONTEXT.get()
                ctx['environment_seed_receipt'] = native.seed(ctx['task']['seed'])
                return native

            self.textworld.gym.make = seeded_make
            try:
                result = await original_set(task_path)
                goal = result[0].get('goal')
                if not isinstance(goal, str) or not goal.strip():
                    raise Rejected('native TextWorld reset did not provide a nonempty objective')
                CONTEXT.get()['task_goal'] = goal
                return result
            finally:
                self.textworld.gym.make = original_make

        self.workflow.env.set_task_env = seeded_set
        original_reward = self.workflow.env.cal_reward

        async def recorded_reward(*args, **kwargs):
            recorder = await original_reward(*args, **kwargs)
            CONTEXT.get()['native_reward'] = float(recorder.reward)
            return recorder

        self.workflow.env.cal_reward = recorded_reward

    def _make_client_factory(self):
        import httpx
        import openai
        owner = self

        class TrackedTransport(httpx.AsyncBaseTransport):
            def __init__(self):
                self.inner = httpx.AsyncHTTPTransport(trust_env=False)

            async def handle_async_request(self, request):
                ctx = CONTEXT.get()
                if ctx is None:
                    raise RuntimeError('unscoped ALMA TextWorld model request')
                if str(request.url) != owner.llm.base + '/chat/completions':
                    raise ValueError('unexpected TextWorld model endpoint')
                body = json.loads(request.content)
                if body.get('model') != owner.llm.model_name or 'response_format' in body:
                    raise ValueError('ALMA raw-action model contract changed')
                step = sum(message['role'] == 'user' for message in body['messages']) - 1
                body.update(ACTOR_PARAMETERS, seed=ctx['model_seed'] + step)
                input_receipt = goal_delivery_receipt(body['messages'], ctx.get('task_goal'),
                    owner.task_instructions[ctx['task']['family']], digest(body))
                if not input_receipt['direct_goal_verified']:
                    ctx['model_errors'] += 1
                    append(owner.ledger, dict(kind='task_input_error', role='actor', phase=ctx['phase'],
                        task_id=ctx['task']['id'], task_input_receipt=input_receipt,
                        request_hash=digest(body), messages=body['messages'], ok=False,
                        error='CurrentGoalMissingFromDirectActorInput', network_request_sent=False))
                    raise Rejected('current TextWorld goal is missing from direct Actor input')
                rejected = context_limit_response(request, body, tokenizer=owner.tokenizer,
                    context=ctx, ledger=owner.ledger, model=owner.llm,
                    task_input_receipt=input_receipt)
                if rejected is not None:
                    return rejected
                modified = httpx.Request(request.method, request.url,
                    headers={key: value for key, value in request.headers.items() if key.lower() != 'content-length'},
                    content=dumps(body).encode())
                started = time.monotonic()
                data, ok, error, status = {}, False, None, None
                try:
                    response = await self.inner.handle_async_request(modified)
                    status = response.status_code
                    raw = await response.aread()
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            data = parsed
                    except (ValueError, UnicodeError):
                        pass
                    choices = data.get('choices')
                    ok = (status == 200 and isinstance(choices, list) and bool(choices)
                          and isinstance(choices[0], dict) and isinstance(choices[0].get('message'), dict))
                    if not ok:
                        error = 'InvalidCompletionResponse' if status == 200 else 'HTTP' + str(status)
                    if ok and data['choices'][0].get('finish_reason') == 'length':
                        ctx['truncated'] += 1
                    return response
                except BaseException as exc:
                    error = type(exc).__name__
                    raise
                finally:
                    if not ok:
                        ctx['model_errors'] += 1
                    usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
                    choice = data['choices'][0] if ok else {}
                    append(owner.ledger, dict(
                        kind='model', role='actor', phase=ctx['phase'], task_id=ctx['task']['id'],
                        base_url=owner.llm.base, seed=body['seed'], ok=ok, error=error, http_status=status,
                        elapsed=time.monotonic() - started, prompt_tokens=usage.get('prompt_tokens'),
                        completion_tokens=usage.get('completion_tokens'), request_hash=digest(body),
                        model_fingerprint=owner.llm.fingerprint, model_name=owner.llm.model_name,
                        parameters=ACTOR_PARAMETERS,
                        messages=body['messages'], finish_reason=choice.get('finish_reason'),
                        task_input_receipt=input_receipt,
                        answer=choice.get('message', {}).get('content'),
                        original_message_hash=digest(body['messages'])))

            async def aclose(self):
                await self.inner.aclose()

        def factory(**kwargs):
            client = openai.AsyncOpenAI(api_key='local-not-a-secret', base_url=owner.llm.base,
                timeout=300, max_retries=2, http_client=httpx.AsyncClient(
                    transport=TrackedTransport(), trust_env=False, timeout=300, follow_redirects=False))
            owner.clients.append(client)
            return client

        return factory

    async def _episode(self, task, memory, phase, seed, no_memory=False, allow_write=False):
        from rich.progress import Progress
        import utils.hire_agent as hire
        if no_memory and allow_write:
            raise Rejected('official no-memory baseline cannot collect generated writes')
        validate_environment_metadata(task, 'textworld')
        game = verify_task_assets(task)
        path = self.runtime_ids[str(game)]
        if game not in self.asset_seeds:
            self.asset_seeds[game] = asset_seed_receipt(task, game)
        hire.AsyncOpenAI = self.client_factory
        started = time.monotonic()
        transaction = EpisodeMemory(self.workflow, memory, task, 'textworld', phase, seed,
                                    self.ledger, no_memory=no_memory, allow_write=allow_write)
        ctx = dict(task=task, phase=phase, model_seed=seed, model_errors=0, truncated=0)
        token = CONTEXT.set(ctx)
        store = EvolvableMemoryStore(transaction.view, self.tokenizer, allow_write)
        snapshot = self.NoMemory() if no_memory else self.Memory(store, task, phase, seed)
        recorder, error = None, None
        prior_directory = Path.cwd()
        try:
            os.chdir(self.layout)
            with transaction.adapter, Progress(disable=True) as progress:
                progress_id = progress.add_task(task['id'], total=STEP_LIMIT)
                recorder = await self.workflow.run_single_task(
                    path, eval_type='sequential' if allow_write else 'batched', current_state='retrieve',
                    progress=progress, task_id=progress_id, memory_snapshot=snapshot)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            append(self.ledger, dict(kind='episode_error', task_id=task['id'], phase=phase, error=error))
            recorder = self.workflow.env.recorders.get(path)
        except BaseException as exc:
            transaction.abort(str(exc))
            raise
        finally:
            native = self.workflow.env.envs.pop(path, None)
            if native is not None:
                try:
                    native.close()
                except Exception as exc:
                    error = error or 'Close' + type(exc).__name__
            self.workflow.env.recorders.pop(path, None)
            for client in self.clients:
                if not client.is_closed():
                    try:
                        await client.close()
                    except Exception as exc:
                        error = error or 'ClientClose' + type(exc).__name__
            self.clients.clear()
            CONTEXT.reset(token)
            os.chdir(prior_directory)
        if recorder is not None:
            ep = experience(recorder, task, phase, seed)
        else:
            ep = dict(task_id=task['id'], group_id=task['group_id'], room=task['room'], family=task['family'],
                      goal='', history=[], score=None, done=False, phase=phase, seed=seed)
        error = error or ('ModelRequestError' if ctx['model_errors'] else None)
        error = error or ('OutputTruncated' if ctx['truncated'] else None)
        reward_verified = ('native_reward' in ctx and math.isfinite(ctx['native_reward'])
                           and 0 <= ctx['native_reward'] <= 1 and ctx['native_reward'] == ep['score'])
        if error or not reward_verified:
            ep['score'] = None
            reward_verified = False
        ep.update(error=error, native_reward_verified=reward_verified, elapsed=time.monotonic() - started,
                  retrieved={entry: 1 for read in store.reads for entry in read['ids']},
                  read_tokens=store.read_tokens, memory_read_calls=len(store.reads),
                  memory_write_calls=len(store.writes), executor='ALMA.Agent_Workflow.run_single_task',
                  synthetic=False, environment='textworld', environment_seed=task['seed'],
                  environment_seed_receipt=ctx.get('environment_seed_receipt'),
                  asset_seeds=self.asset_seeds[game], game_sha256=task['game_sha256'],
                  step_limit=STEP_LIMIT, official_split=task['official_split'])
        if recorder is not None:
            ep['native_recorder'] = dict(initial=public_context(recorder.init), steps=copy.deepcopy(recorder.steps),
                reward=float(recorder.reward), memory_retrieved=copy.deepcopy(getattr(recorder, 'memory_retrieved', {})))
        for index in range(len(ep['history'])):
            append(self.ledger, dict(kind='env_step', phase=phase, task_id=task['id'], step=index))
        ep = transaction.finalize(ep, task_goal=ctx.get('task_goal'))
        print(dumps(dict(episode=task['id'], phase=phase, score=ep['score'], error=ep.get('error'),
                         steps=len(ep['history']), seconds=round(ep['elapsed'], 2))), flush=True)
        return ep

    def episode(self, task, memory, phase, seed, no_memory=False, allow_write=False):
        return asyncio.run(self._episode(task, memory, phase, seed, no_memory, allow_write))


class TextWorldNativeEvaluator:
    step_limit = STEP_LIMIT

    def __init__(self, root, tasks, tokenizer, model_fingerprint, context=None,
                 base_url='http://127.0.0.1:8142/v1', *, capabilities=None, model_name=MODEL_NAME):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks = json_copy(tasks)
        self.task_payloads = {task['id']: digest(task) for task in self.tasks}
        if len(self.task_payloads) != len(self.tasks):
            raise Rejected('native task IDs must be unique')
        inventory = build_manifest()
        official_tasks = {task['id']: digest(task) for task in inventory['tasks']}
        for task in self.tasks:
            validate_environment_metadata(task, 'textworld')
            if official_tasks.get(task['id']) != digest(task):
                raise Rejected('task is not an unchanged official TextWorld manifest entry')
        self.tokenizer = tokenizer
        self.capabilities = capabilities
        self.context = json_copy({} if context is None else context)
        validate_environment_context(self.context, 'textworld')
        self.ledger = self.root / 'native/costs.jsonl'
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.llm = make_local_llm(tokenizer, self.ledger, model_fingerprint, base_url=base_url,
                                  model_name=model_name)
        try:
            self.runner = TextWorldALMARunner(self.llm, tokenizer, self.ledger, self.tasks, self.root)
        except Exception:
            self.llm.client.close()
            raise
        versions = {name: importlib.metadata.version(name) for name in
                    ('gym', 'jericho', 'numpy', 'openai', 'httpx', 'rich')}
        versions['textworld'] = self.runner.textworld.__version__
        provenance = dict(
            environment='textworld', official_commit=OFFICIAL_COMMIT,
            official_sources=self.runner.source_hashes, environment_versions=versions,
            actor_parameters=ACTOR_PARAMETERS, model_fingerprint=model_fingerprint,
            model_name=model_name, base_url=base_url,
            python=sys.version, context=self.context, environment_step_limit=STEP_LIMIT,
            dataset_archive_sha256=inventory['archive_sha256'], task_manifest_hash=digest(self.tasks),
            isolated_textworld_wheel_sha256=file_hash(ROOT / 'data/textworld_balrog/downloads'
                / 'textworld-1.6.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'),
            runtime_hashes={name: file_hash(Path(__file__).with_name(name)) for name in
                            ('textworld_native.py', 'textworld_input.py', 'textworld_tasks.py', 'native.py', 'memory.py',
                             'program.py', 'worker.py', 'policy.py', 'isolation.py', 'local_service.py', 'hooks.py', 'hook_adapter.py', 'protocol.py')},
            frozen_backend_hash=file_hash(ROOT / 'evomem/alma_backend.py'),
            isolated_textworld_source_hashes={str(path.relative_to(TEXTWORLD16)): file_hash(path)
                for path in sorted((TEXTWORLD16 / 'textworld').rglob('*'))
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'},
            task_seed_policy='42000 + int(sha256(canonical_json(task_id))[:6],16) % 10000000',
            reset_seed_policy='native.seed(task.seed) before official reset; asset generation is separate',
            asset_seed_policy=ASSET_SEED_POLICY,
            task_input_correction=dict(version=TASK_INPUT_VERSION,
                official_environment_sha256=self.runner.source_hashes[SOURCE_OVERRIDES['eval_envs/test_envs.py']],
                staged_environment_sha256=file_hash(self.runner.layout / 'eval_envs/test_envs.py'),
                goal_forwarded_to_official_formatter=True, native_observation_text_preserved=True),
        )
        if capabilities is not None:
            provenance['model_capabilities'] = capabilities.fingerprint
        self.fingerprint = digest(provenance)
        write(self.root / 'native/evaluator_manifest.json', dict(fingerprint=self.fingerprint, **provenance))
        write(self.root / 'native/task_inventory.json', inventory)

    def check_tasks(self, tasks):
        if len({task['id'] for task in tasks}) != len(tasks):
            raise Rejected('evaluation tasks must be unique')
        for task in tasks:
            validate_environment_metadata(task, 'textworld')
            if self.task_payloads.get(task['id']) != digest(task):
                raise Rejected('task payload was not part of the frozen native task set')
            verify_task_assets(task)

    def __call__(self, program, logs, tasks, phase, *, memory_snapshot=None):
        self.check_tasks(tasks)
        for ep in logs:
            validate_environment_metadata(ep, 'textworld')
        families = {task['family'] for task in tasks}
        memory = (CodeMemory.from_snapshot(program, memory_snapshot, self.tokenizer, capabilities=self.capabilities)
                  if memory_snapshot is not None else
                  CodeMemory(program, logs, self.tokenizer, self.context, capabilities=self.capabilities))
        if memory.memory_scope == 'family' and (len(families) > 1 or (
                memory.family is not None and families and families != {memory.family})):
            raise Rejected('evaluation Task Memory must stay within its task family')
        append(self.ledger, dict(kind='memory_build', phase=phase, program_hash=program.hash,
                                log_hash=memory.source_log_hash, memory_hash=memory.fingerprint, **memory.stats))
        return [self.runner.episode(task, memory, phase, fixed_task_seed(task)) for task in tasks]

    def learn(self, program, logs, task, phase, *, memory_snapshot):
        """One deployed-policy learning task with an uncommitted temporary bank."""
        self.check_tasks([task])
        for episode in logs:
            validate_environment_metadata(episode, 'textworld')
        memory = CodeMemory.from_snapshot(program, memory_snapshot, self.tokenizer, capabilities=self.capabilities)
        if memory.environment != 'textworld':
            raise Rejected('Learning snapshot belongs to a different dataset')
        episode = self.runner.episode(task, memory, phase, fixed_task_seed(task), allow_write=True)
        memory.source_log_hash = digest(logs + [episode])
        return dict(episode=episode, memory_snapshot=memory.export_snapshot())

    def collect(self, program, tasks, phase='code_adaptation_collection'):
        self.check_tasks(tasks)
        if self.context.get('memory_scope', 'family') == 'family' and len({task['family'] for task in tasks}) > 1:
            raise Rejected('collection Task Memory must stay within one task family')
        memory = CodeMemory(program, [], self.tokenizer, self.context, capabilities=self.capabilities)
        return [self.runner.episode(task, memory, phase, fixed_task_seed(task), allow_write=True) for task in tasks]

    def no_memory(self, tasks, phase='no-memory-baseline'):
        self.check_tasks(tasks)
        memory = CodeMemory(Program(BASELINE_SOURCE), [], self.tokenizer, self.context)
        return [self.runner.episode(task, memory, phase, fixed_task_seed(task), no_memory=True) for task in tasks]

    def close(self):
        self.llm.client.close()


NativeEvaluator = TextWorldNativeEvaluator
