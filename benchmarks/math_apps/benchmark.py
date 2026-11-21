"""One-turn benchmark adapter for the production evolution engine."""
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PILOT = HERE
from .paths import ASSETS
sys.path.insert(0, str(PROJECT))
# Only the pinned one-turn Actor transport is shared with calibration.
from importlib.util import module_from_spec, spec_from_file_location
spec = spec_from_file_location('single_turn_actor', PILOT / 'actor.py')
actor = module_from_spec(spec)
spec.loader.exec_module(actor)

from evomem.common import MODEL_NAME, digest, file_hash, read, write
from phase1_codegen.campaign import bind
from phase1_codegen.hooks import HookSession
from phase1_codegen.memory import CodeMemory
from phase1_codegen.native import fixed_task_seed
from phase1_codegen.policy import Rejected
from phase1_codegen.program import WorkerStartupError

from phase1_codegen.protocol import ONE_TURN_EXECUTOR as EXECUTOR

WORKERS = 2
# The isolated APPS judge's language surface is public; freeze it for both arms.
actor.PROMPTS['code'] += (' Use built-ins or these supported Python modules: array, bisect, collections, copy,'
                          ' decimal, fractions, functools, heapq, io, itertools, json, math, operator, random,'
                          ' re, statistics, string, sys, typing. Read and write through standard streams'
                          ' for standard-I/O submissions.')


class BenchmarkEvaluator:
    step_limit = 1

    def __init__(self, root, domain, adapter, tasks, tokenizer, model_hash, context, capabilities, run_root):
        self.root, self.domain = Path(root), domain
        self.adapter, self.tokens, self.model_hash = adapter, tokenizer, model_hash
        self.context, self.caps = context, capabilities
        self.task_bindings = {t['id']: digest(t) for t in tasks}
        source = {r['id']: r for r in adapter.load_public() if r['id'] in self.task_bindings}
        if len(tasks) != len(self.task_bindings) or set(self.task_bindings) != set(source) or context['environment'] != domain:
            raise ValueError('Duplicate or missing task, or mismatched environment')
        self.public = {}
        for task in tasks:
            row = source[task['id']]
            if (task['environment'] != domain or task['split'] != row['split'] or task['family'] != row['family']
                    or task['prompt_hash'] != digest(row['prompt'])):
                raise ValueError('Task manifest and published benchmark identity differ')
            self.public[task['id']] = row['prompt']
        self.prompt_domain = 'math' if domain == 'math' else 'code'
        self.cache = Path(run_root) / 'requests' / domain
        self.fingerprint = digest(dict(executor=EXECUTOR, domain=domain, context=context,
            manifest=self.task_bindings, adapter=file_hash(Path(adapter.__file__)), model=model_hash,
            actor=file_hash(PILOT/'actor.py'), implementation=file_hash(Path(__file__)),
            capabilities=getattr(capabilities, 'fingerprint', None)))

    def bank(self, program, logs, snapshot):
        if snapshot is None:
            return CodeMemory(program, logs, self.tokens, self.context, capabilities=self.caps)
        bank = CodeMemory.from_snapshot(program, snapshot, self.tokens, capabilities=self.caps)
        if bank.environment != self.domain or bank.source_log_hash != digest(logs):
            raise ValueError('Evaluation history or domain differs from supplied snapshot')
        return bank

    async def answer(self, task, prompt, slot, directory):
        messages = actor.messages_for(self.prompt_domain, prompt, slot)
        seed = fixed_task_seed(task)
        body = dict(model=MODEL_NAME, messages=messages, seed=seed, **actor.PARAMETERS)
        key = digest(dict(task=task['id'], dataset=self.domain, request=body,
                          endpoint=actor.URL, model_hash=self.model_hash))
        physical = self.cache / key
        output = physical/'actor.json'
        if not output.exists():
            intent = physical/'intent.json'
            if intent.exists():
                raise RuntimeError('Unresolved physical Actor submission: ' + str(intent))
            bind(intent, dict(task_id=task['id'], wire_hash=digest(body), dataset=self.domain,
                              checkpoint=self.model_hash))
        reply = await actor.ask(self.prompt_domain, prompt, slot, seed, output, self.model_hash)
        if (reply['request'] != body or reply['served_model'] != MODEL_NAME
                or reply['http_status'] != 200 or not reply['usage'] or not reply['response_id']):
            raise RuntimeError('Actor response identity/usage/delivery mismatch')
        bind(directory/'actor_reference.json', dict(id=key, hash=file_hash(output), finish=reply['finish']))
        return reply

    async def episode(self, program, bank, task, phase, learn):
        if self.task_bindings.get(task['id']) != digest(task) or task['environment'] != self.domain:
            raise ValueError('Unregistered task')
        if learn and task['split'] != 'train':
            raise ValueError('Learning requires official train data')
        directory = self.root/'episodes'/phase/program.hash/digest(task['id'])
        before = bank.fingerprint
        prompt = self.public[task['id']]
        seed = fixed_task_seed(task)
        visible = dict(goal=prompt, obs=prompt)
        hook = HookSession(bank.for_task(task,self.domain), task, allow_write=learn)
        ep = dict(task_id=task['id'], group_id=task['group_id'], family=task['family'],
            room=task['room'], environment=self.domain, phase=phase, goal=prompt, seed=seed,
            environment_seed=task['seed'], synthetic=False, executor=EXECUTOR, history=[],
            score=None, done=False, error=None, native_reward_verified=False, program_hash=program.hash,
            program_identity_hash=program.identity_hash, memory_hash_before=before, read_only_memory=not learn)
        injected = []
        slot = {}
        try:
            hook.emit('task_start', dict(step=0, visible=visible))
            hook.emit('before_action', dict(step=0, attempt=0, visible=visible))
            slot = hook.current_memory
            reply = await self.answer(task, prompt, slot, directory)
            wire = reply['request']['messages']
            injected.append(dict(memory=slot, attempt=0, step=0, event_sequence=1,
                task_id=task['id'], original_user_prompt_hash=digest(prompt), system_hash=digest(wire[0]['content']),
                actor_request_hash=reply['request_hash'],
                actor_delivery_verified=wire==actor.messages_for(self.prompt_domain, prompt, slot)))
            # Identical physical submissions have one immutable grade, including all official cases.
            grade_path = self.cache / read(directory/'actor_reference.json')['id'] / 'grade.json'
            if grade_path.exists():
                grade = read(grade_path)
            else:
                grade = await asyncio.to_thread(self.adapter.grade, task['id'], reply['answer'])
                if grade.get('score') is not None:
                    bind(grade_path, grade)
            bind(directory/'grade.json', grade)
            score = grade['score']
            if type(score) not in (float,int) or score not in (0,1):
                raise RuntimeError('Unverified benchmark grade for ' + task['id'])
            feedback = 'Correct.' if score else 'Incorrect.'
            ep.update(history=[dict(observation=prompt, action=reply['answer'], next_observation=feedback)],
                      done=True, score=score, native_reward_verified=True)
            hook.emit('after_observation', dict(step=1, attempt=0, action=reply['answer'],
                      visible=dict(obs=feedback, scores=score, dones=True)))
            hook.finish(ep)
            if learn:
                hook.publish()
            else:
                hook.close()
        except WorkerStartupError:
            raise
        except Rejected as exc:
            ep.update(error=f'{type(exc).__name__}: {exc}', score=None, native_reward_verified=False)
            hook.abort(ep['error'])
        evidence = hook.summary()
        evidence.update(injections=injected, actor_delivery_verified=bool(injected and injected[0]['actor_delivery_verified']))
        ep.update(memory_hook=evidence, memory_read_calls=sum(e.get('response') is not None for e in hook.trace),
                  read_tokens=hook.usage['injected_tokens'], memory_hash_after=bank.fingerprint,
                  generated_memory_write_calls=int(hook.committed), memory_write_calls=int(hook.committed),
                  native_recorder=dict(initial=visible, reward=ep['score'], memory_retrieved=slot))
        write(directory/'episode.json', ep)
        return ep

    def __call__(self, program, logs, tasks, phase, *, memory_snapshot=None):
        snapshot = self.bank(program, logs, memory_snapshot).export_snapshot()
        async def batch():
            limit = asyncio.Semaphore(WORKERS)
            async def item(task):
                async with limit:
                    bank = CodeMemory.from_snapshot(program, snapshot, self.tokens, capabilities=self.caps)
                    return await self.episode(program, bank, task, phase, False)
            results = await asyncio.gather(*(item(t) for t in tasks), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return results
        return asyncio.run(batch())

    def learn(self, program, logs, task, phase, *, memory_snapshot):
        bank = self.bank(program, logs, memory_snapshot)
        episode = asyncio.run(self.episode(program, bank, task, phase, True))
        if episode['error']:
            return dict(episode=episode, memory_snapshot=memory_snapshot)
        bank.source_log_hash = digest(logs + [episode])
        return dict(episode=episode, memory_snapshot=bank.export_snapshot())
