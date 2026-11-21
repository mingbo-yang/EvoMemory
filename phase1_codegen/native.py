"""Generated Task Memory inside the pinned, unchanged ALMA action workflow.

Importing this module performs no staging, native task, or model call. Runtime
construction stages the official adapter; evaluation is explicitly requested.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

from evomem.alma_backend import ACTOR_PARAMETERS, ALMARunner, experience
from evomem.common import BASE_URL, MODEL_NAME, ROOT, RUN as OLD, append, digest, file_hash, write
from .outcomes import reported_score
from .memory import CodeMemory
from .policy import Rejected, json_copy
from .program import BASELINE_SOURCE, Program, WorkerStartupError


def validate_environment_metadata(value, environment):
    """Reject explicit foreign provenance; official assets verify missing tags."""
    for key in ('environment', 'dataset'):
        if value.get(key) is not None and value[key] != environment:
            raise Rejected('task or experience declares a different environment')


def validate_environment_context(context, environment):
    if context.get('environment', environment) != environment:
        raise Rejected('memory context does not belong to the native environment')
    if context.get('memory_scope', 'family') == 'environment' and context.get('environment') != environment:
        raise Rejected('shared Task Memory requires explicit native environment context')


def hook_evidence(session, adapter, actor_rows):
    """Bind assembled slots to the actual local Actor request ledger."""
    from evomem.common import digest
    value = session.summary()
    value['injections'] = []
    for index, row in enumerate(adapter.injections):
        item = {key: val for key, val in row.items() if key != 'system_message'}
        actual = actor_rows[index] if index < len(actor_rows) else {}
        messages = actual.get('messages', [])
        users = [message.get('content') for message in messages if message.get('role') == 'user']
        item['actor_request_hash'] = actual.get('request_hash')
        item['actor_delivery_verified'] = bool(messages and users and actual.get('ok')
            and digest(messages[0].get('content')) == row['system_hash']
            and digest(users[-1]) == row['original_user_prompt_hash'])
        value['injections'].append(item)
    value['actor_delivery_verified'] = (len(actor_rows) == len(adapter.injections)
        and all(row['actor_delivery_verified'] for row in value['injections']))
    return value


class EpisodeMemory:
    """Shared native memory boundary: stage, validate delivery, then publish."""
    def __init__(self, workflow, memory, task, environment, phase, seed, ledger,
                 *, no_memory=False, allow_write=False):
        if no_memory and allow_write:
            raise Rejected('official no-memory baseline cannot collect generated writes')
        self.memory, self.task = memory, task
        self.environment, self.phase = environment, phase
        self.no_memory, self.allow_write = no_memory, allow_write
        self.ledger = Path(ledger)
        self.ledger_start = self.ledger.stat().st_size if self.ledger.exists() else 0
        self.before = memory.fingerprint
        self.read_start, self.write_start = len(memory.read_trace), len(memory.write_trace)
        self.view = memory.for_task(task, environment)
        self.hook = None
        self.adapter = nullcontext()
        if not no_memory:
            from .hooks import HookSession
            from .hook_adapter import HookAdapter
            self.hook = HookSession(self.view, task, allow_write)
            self.adapter = HookAdapter(workflow, self.hook, task, phase, seed)

    def abort(self, error):
        if self.hook is not None:
            self.hook.abort(error)

    def actor_rows(self):
        if not self.ledger.exists():
            return []
        if self.ledger.stat().st_size < self.ledger_start:
            raise Rejected('Actor ledger was truncated during its episode')
        rows = []
        with self.ledger.open('rb') as stream:
            stream.seek(self.ledger_start)
            for line in stream:
                row = json.loads(line)
                if (row.get('role') == 'actor' and row.get('task_id') == self.task['id']
                        and row.get('phase') == self.phase):
                    rows.append(row)
        return rows

    def finalize(self, episode, *, task_goal=None):
        """Called after native close/transport checks, once for either environment."""
        result = copy.deepcopy(episode)
        memory = self.memory
        try:
            actor_rows = self.actor_rows()
            if self.environment == 'textworld':
                from .textworld_input import summarize_goal_delivery
                receipt = summarize_goal_delivery(actor_rows, task_goal)
                result['task_input_receipt'] = receipt
                if not receipt['all_actor_requests_verified'] and not result.get('error'):
                    result['error'] = 'CurrentGoalDeliveryUnverified'
        except BaseException as exc:
            self.abort(str(exc))
            raise
        if self.hook is not None:
            evidence = hook_evidence(self.hook, self.adapter, actor_rows)
            if not result.get('error'):
                if not evidence['complete']:
                    result['error'] = 'HookEpisodeIncomplete'
                elif not result.get('native_reward_verified'):
                    result['error'] = 'NativeRewardUnverified'
                elif not evidence['actor_delivery_verified']:
                    result['error'] = 'HookActorDeliveryUnverified'
                elif memory.fingerprint != self.before:
                    result['error'] = 'MemoryChangedBeforeValidation'
                elif len(memory.write_trace) != self.write_start:
                    result['error'] = 'MemoryWriteBeforeValidation'
                elif result.get('memory_write_calls', 0) != int(self.allow_write):
                    result['error'] = 'OriginalWriteIntentTimingChanged'
            if result.get('error'):
                self.hook.abort(result['error'])
            else:
                try:
                    if self.allow_write:
                        self.hook.publish()
                    else:
                        self.hook.close()
                except Exception as exc:
                    result['error'] = type(exc).__name__ + ': ' + str(exc)
                    self.hook.abort(result['error'])
            evidence.update(self.hook.summary())
            result['memory_hook'] = evidence
            result['memory_read_calls'] = sum(row.get('response') is not None for row in self.hook.trace)
            result['read_tokens'] = self.hook.usage['injected_tokens']
            if result.get('error'):
                result['native_reward_verified'] = False
        after = memory.fingerprint
        committed = sum(row.get('status') == 'committed' for row in memory.write_trace[self.write_start:])
        if not self.allow_write and self.before != after:
            result['error'] = result.get('error') or 'ReadOnlyMemoryChanged'
        if not self.no_memory and not result.get('error'):
            if self.allow_write and committed != 1:
                result['error'] = 'GeneratedWriteDidNotCommitExactlyOnce'
        result.update(environment=self.environment,
            memory_hash='official-no-memory' if self.no_memory else self.before,
            memory_hash_before=self.before, memory_hash_after=after,
            program_hash=memory.program.hash, program_identity_hash=memory.program.identity_hash,
            generated_memory_write_calls=committed, read_only_memory=not self.allow_write,
            actor_ledger_start=self.ledger_start,
            actor_ledger_end=self.ledger.stat().st_size if self.ledger.exists() else 0)
        startup_failures = [row for row in (result.get('memory_hook') or {}).get('events', [])
                            if row.get('status') == 'rejected' and row.get('error_type') == 'WorkerStartupError']
        if startup_failures:
            failed = startup_failures[0]
            result['infrastructure_failure'] = dict(kind='trusted_worker_bootstrap',
                generated_execution_started=False, event_sequence=failed['sequence'],
                details=failed.get('infrastructure_details', {}), error=failed['error'])
        result['score'] = reported_score(result)
        if result['score'] is None:
            result['native_reward_verified'] = False
        append(self.ledger.with_name('canonical_episodes.jsonl'), result)
        native = result.get('native_recorder', {})
        append(self.ledger.with_name('code_memory_traces.jsonl'), dict(
            kind='generated_memory_native_episode', synthetic=result.get('synthetic', False),
            phase=self.phase, task_id=self.task['id'], environment=self.environment,
            mechanism=asdict(memory.mechanism), memory_hash_before=self.before, memory_hash_after=after,
            source_log_hash=memory.source_log_hash,
            read_calls=copy.deepcopy(memory.read_trace[self.read_start:]),
            write_calls=copy.deepcopy(memory.write_trace[self.write_start:]),
            actual_injected_memory=copy.deepcopy(native.get('memory_retrieved', {})),
            actual_system_context=actor_rows[0]['messages'][0]['content'] if actor_rows else None,
            actions=[row['action'] for row in result.get('history', [])],
            native_score=result.get('score'), error=result.get('error'),
            stored_tokens=memory.stats['stored_tokens'], memory_hook=result.get('memory_hook')))
        if startup_failures:
            # The failed episode and all incurred Actor/Hook costs exist first.
            # This host failure must stop search, not become a candidate verdict.
            raise WorkerStartupError(result['error'], details=result['infrastructure_failure'])
        return result


def code_bridge_class(environment):
    """Only memory update semantics differ from the original frozen bridge."""
    from agents.memo_structure import MemoStructure

    class GeneratedMemory(MemoStructure):
        def __init__(self, store, task, phase, seed):
            super().__init__()
            self.store = store
            self.task = task
            self.phase = phase
            self.seed = seed

        async def general_retrieve(self, recorder):
            from .hook_adapter import active_hook
            session = active_hook.get()
            if session is None or session.task['id'] != self.task['id']:
                raise Rejected('native memory retrieval requires its active event session')
            from evomem.alma_backend import public_context
            session.emit('task_start', dict(step=0, visible=public_context(recorder.init)))
            return session.current_memory

        async def general_update(self, recorder):
            ep = experience(recorder, self.task, self.phase, self.seed)
            validate_environment_metadata(self.task, environment)
            ep['environment'] = environment
            # Original store checks allow_write and records the attempted write.
            # CodeMemory commits only after isolated execution and validation.
            from .hook_adapter import active_hook
            session = active_hook.get()
            if session is None or session.task['id'] != self.task['id']:
                raise Rejected('native memory update requires its active event session')
            self.store.write(ep)  # Native finalization publishes after all checks.

    return GeneratedMemory


class CodeALMARunner(ALMARunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Memory = code_bridge_class('alfworld')
        self.canonical_episodes = self.ledger.with_name('canonical_episodes.jsonl')
        self.trace_path = self.ledger.with_name('code_memory_traces.jsonl')

    async def _episode(self, task, memory, phase, seed, no_memory=False, allow_write=False):
        validate_environment_metadata(task, 'alfworld')
        transaction = EpisodeMemory(self.workflow, memory, task, 'alfworld', phase, seed,
                                    self.ledger, no_memory=no_memory, allow_write=allow_write)
        prior_directory = Path.cwd()
        try:
            # Preserve official relative assets and its entire native lifecycle.
            os.chdir(self.layout)
            with transaction.adapter:
                episode = await super()._episode(task, transaction.view, phase, seed, no_memory, allow_write)
        except BaseException as exc:
            transaction.abort(str(exc))
            raise
        finally:
            os.chdir(prior_directory)
        return transaction.finalize(episode)


def isolate_stage(root):
    spec = importlib.util.spec_from_file_location('scripts._codegen_isolated_stage', ROOT / 'scripts/alma_feasibility.py')
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    stage.AUDIT = Path(root).resolve() / 'alma_audit'
    stage.META = OLD / 'alma_audit/downloads/json_2.1.1_json.zip'
    return stage


def fixed_task_seed(task):
    return 42000 + int(digest(task['id'])[:6], 16) % 10000000


class NativeEvaluator:
    """A callable evaluator of complete programs with native ALFWorld rewards.

    Each call rebuilds from precisely the supplied target adaptation logs. All
    evaluation tasks use read-only banks. ``collect`` is the separate sequential
    route for obtaining adaptation experiences using generated online writes.
    """
    step_limit = 30

    def __init__(self, root, tasks, tokenizer, model_fingerprint, context=None, *, capabilities=None,
                 base_url=BASE_URL, model_name=MODEL_NAME):
        import scripts.alma_feasibility as dependency
        from .local_service import make_local_llm

        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks = json_copy(tasks)
        self.task_payloads = {task['id']: digest(task) for task in self.tasks}
        if len(self.task_payloads) != len(self.tasks):
            raise Rejected('native task IDs must be unique')
        self.tokenizer = tokenizer
        self.capabilities = capabilities
        self.context = json_copy({} if context is None else context)
        validate_environment_context(self.context, 'alfworld')
        for task in self.tasks:
            self._validate_task_environment(task)
        self.ledger = self.root / 'native/costs.jsonl'
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.llm = make_local_llm(tokenizer, self.ledger, model_fingerprint, base_url=base_url,
                                  model_name=model_name)
        stage = isolate_stage(self.root)
        prior_stage, prior_directory = dependency.stage, Path.cwd()
        try:
            dependency.stage = stage.stage
            self.runner = CodeALMARunner(self.llm, tokenizer, self.ledger, self.tasks)
            for task in self.tasks:
                staged_game = self.runner.layout / 'data/alfworld' / task['archive_member']
                if file_hash(staged_game) != task['game_sha256']:
                    raise Rejected('ALFWorld staged official game differs from the frozen task manifest')
        except Exception:
            self.llm.client.close()
            raise
        finally:
            dependency.stage = prior_stage
            os.chdir(prior_directory)
        manifest = stage.manifest()
        # This is provenance, not a claim of a completed rollout.
        manifest['research_status'] = 'pinned native evaluator constructed; task outcomes recorded separately'
        write(stage.AUDIT / 'source_manifest.json', manifest)
        own_files = ['native.py', 'program.py', 'policy.py', 'worker.py', 'memory.py', 'hooks.py', 'hook_adapter.py', 'protocol.py']
        provenance = dict(
            environment='alfworld',
            actor_parameters=ACTOR_PARAMETERS,
            model_fingerprint=model_fingerprint,
            model_name=model_name,
            model_endpoint=base_url,
            official_commit=manifest['commit'],
            official_sources=manifest['source_hashes'],
            environment_versions=manifest['installed_packages'],
            python=sys.version,
            backend_hash=file_hash(ROOT / 'evomem/alma_backend.py'),
            generated_memory_runtime={name: file_hash(Path(__file__).with_name(name)) for name in own_files},
            trusted_isolation_runtime={'isolation.py': file_hash(Path(__file__).with_name('isolation.py'))},
            task_seed_policy='42000 + int(sha256(canonical_json(task_id))[:6],16) % 10000000',
            environment_step_limit=30,
            task_manifest_hash=digest(self.tasks),
            context=self.context,
        )
        if capabilities is not None:
            provenance['model_capabilities'] = capabilities.fingerprint
        self.fingerprint = digest(provenance)
        write(self.root / 'native/evaluator_manifest.json', dict(fingerprint=self.fingerprint, **provenance))

    @staticmethod
    def _validate_task_environment(task):
        validate_environment_metadata(task, 'alfworld')
        member = Path(task['archive_member'])
        if (member.is_absolute() or '..' in member.parts or len(member.parts) < 3
                or member.parts[0] != 'json_2.1.1'
                or member.parts[1] not in ('train', 'valid_train', 'valid_seen', 'valid_unseen')
                or member.name != 'game.tw-pddl'):
            raise Rejected('task is not an official ALFWorld game member')

    def check_tasks(self, tasks):
        if len({task['id'] for task in tasks}) != len(tasks):
            raise Rejected('evaluation tasks must be unique')
        for task in tasks:
            self._validate_task_environment(task)
            if self.task_payloads.get(task['id']) != digest(task):
                raise Rejected('task payload was not part of the staged frozen task set')

    def __call__(self, program, logs, tasks, phase, *, memory_snapshot=None):
        self.check_tasks(tasks)
        for ep in logs:
            validate_environment_metadata(ep, 'alfworld')
        families = {task.get('family') for task in tasks}
        memory = (CodeMemory.from_snapshot(program, memory_snapshot, self.tokenizer, capabilities=self.capabilities)
                  if memory_snapshot is not None else
                  CodeMemory(program, logs, self.tokenizer, self.context, capabilities=self.capabilities))
        if memory.memory_scope == 'family' and (len(families) > 1 or (
                memory.family is not None and families and families != {memory.family})):
            raise Rejected('evaluation Task Memory must stay within the target task family')
        append(self.ledger, dict(kind='memory_build', phase=phase, program_hash=program.hash,
                                 log_hash=memory.source_log_hash, memory_hash=memory.fingerprint,
                                 **memory.stats))
        return [self.runner.episode(task, memory, phase, fixed_task_seed(task), allow_write=False)
                for task in tasks]

    def learn(self, program, logs, task, phase, *, memory_snapshot):
        """Run one learning task in a disposable bank; caller validates then commits."""
        self.check_tasks([task])
        for episode in logs:
            validate_environment_metadata(episode, 'alfworld')
        memory = CodeMemory.from_snapshot(program, memory_snapshot, self.tokenizer, capabilities=self.capabilities)
        if memory.environment != 'alfworld':
            raise Rejected('Learning snapshot belongs to a different dataset')
        episode = self.runner.episode(task, memory, phase, fixed_task_seed(task), allow_write=True)
        # Even an error episode may have reached ALMA general_update. Return a
        # temporary snapshot only; the engine never publishes it until all native
        # reward, request, close, seed and write-count checks have passed.
        memory.source_log_hash = digest(logs + [episode])
        return dict(episode=episode, memory_snapshot=memory.export_snapshot())

    def collect(self, program, tasks, phase='code_adaptation_collection'):
        self.check_tasks(tasks)
        if self.context.get('memory_scope', 'family') == 'family' and len({task.get('family') for task in tasks}) > 1:
            raise Rejected('collection Task Memory must stay within one task family')
        memory = CodeMemory(program, [], self.tokenizer, self.context, capabilities=self.capabilities)
        episodes = []
        for task in tasks:
            episodes.append(self.runner.episode(task, memory, phase, fixed_task_seed(task), allow_write=True))
        return episodes

    def no_memory(self, tasks, phase='no-memory-baseline'):
        self.check_tasks(tasks)
        memory = CodeMemory(Program(BASELINE_SOURCE), [], self.tokenizer, self.context)
        return [self.runner.episode(task, memory, phase, fixed_task_seed(task), no_memory=True)
                for task in tasks]

    def close(self):
        self.llm.client.close()
