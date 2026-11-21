"""One-turn native attempts and transactional, type-local Task Memory for online evaluation."""
import asyncio
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from evomem.common import MODEL_NAME, digest, file_hash, read, write
from phase1_codegen.campaign import bind
from phase1_codegen.capabilities import public_usage
from phase1_codegen.learning import ordered_digest, ordered_wire
from phase1_codegen.memory import CodeMemory
from phase1_codegen.native import fixed_task_seed
from benchmarks.math_apps import apps_tasks, benchmark, math_tasks

actor = benchmark.actor
ADAPTERS = {'math': math_tasks, 'apps': apps_tasks}
FAMILIES = {'math': ('algebra', 'number_theory'), 'apps': ('call_based', 'stdio')}
PROJECT = Path(__file__).resolve().parents[1]
ASSETS = benchmark.ASSETS
PILOT = benchmark.PILOT


def adapter_for(domain):
    return ADAPTERS[domain]


def verified_inputs(domain):
    """Freeze the sources and the public/private data actually used by this online adapter."""
    module = adapter_for(domain)
    manifest_path = ASSETS / (domain + '_manifest.json')
    manifest = read(manifest_path)
    public = 'math_public.json' if domain == 'math' else 'apps_public.jsonl'
    expected = {ASSETS / public: manifest['public_sha256']}
    sources = [Path(module.__file__), Path(benchmark.__file__), PILOT/'actor.py', manifest_path]
    if domain == 'math':
        expected[ASSETS/'math_private/gold.json'] = manifest['private_gold_sha256']
    else:
        author = ASSETS/'apps_archive/author_check.json'
        expected.update({
            ASSETS/'apps_archive/offsets.json': manifest['private_offsets_sha256'],
            ASSETS/'apps_archive/test.jsonl': manifest['splits']['test']['raw_sha256'],
            ASSETS/'apps_archive/source_identities.json': read(author)['index_sha256'],
        })
        sources.extend([PILOT/'code_sandbox.py', PILOT/'code_worker.py', author])
    def key(path):
        try: return str(path.relative_to(PROJECT))
        except ValueError: return 'assets/' + str(path.relative_to(ASSETS))
    hashes = {key(path): file_hash(path) for path in (*expected, *sources)}
    for path, registered in expected.items():
        if hashes[key(path)] != registered:
            raise ValueError('Benchmark differs from its published manifest: ' + path.name)
    return hashes


def load_tasks(domain, families=None, *, seed=11, per_family=None):
    """Outcome-blind, family-ordered official test stream, deduplicated before use."""
    families = FAMILIES[domain] if families is None else tuple(families)
    if (not families or len(set(families)) != len(families)
            or not set(families) <= set(FAMILIES[domain])):
        raise ValueError('Unknown or repeated task family')
    if per_family is not None and (type(per_family) is not int or per_family < 1):
        raise ValueError('per_family must be a positive cap')
    public = [r for r in adapter_for(domain).load_public() if r['split'] == 'test']
    if len({r['id'] for r in public}) != len(public):
        raise ValueError('Duplicate public test ID')
    identities = read(ASSETS/'apps_archive/source_identities.json') if domain == 'apps' else {}
    tasks, questions, urls = [], set(), set()
    for family in families:
        pool = sorted((r for r in public if r['family'] == family),
                      key=lambda r: digest([domain, seed, r['id']]))
        picked = 0
        for row in pool:
            identity = identities[row['id']] if identities else {}
            question = identity.get('normalized_question_sha256') or digest(' '.join(row['prompt'].casefold().split()))
            url = identity.get('source_key')
            if question in questions or (url and url in urls):
                continue
            questions.add(question)
            if url:
                urls.add(url)
            tasks.append(dict(id=row['id'], group_id=row['id'], family=family, room=family,
                              environment=domain, split='test', seed=fixed_task_seed(row),
                              prompt_hash=digest(row['prompt'])))
            picked += 1
            if picked == per_family:
                break
        if not picked:
            raise ValueError('No eligible test items for ' + family)
    return tasks


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


class _TerminalState:
    """Observe the already executed Hook before the read-only evaluator discards it."""
    def __init__(self, program):
        self.program = program
        self.hash = program.hash
        self.identity_hash = program.identity_hash
        self.state_wire = None

    def run(self, operation, state, event, context, **kwargs):
        result = self.program.run(operation, state, event, context, **kwargs)
        if event['type'] == 'task_end':
            self.state_wire = ordered_wire(result['value']['state'])
        return result


def cost_totals():
    return dict(logical_calls=0, cache_hits=0, inference_attempts=0, failed_calls=0,
                measured_input_tokens=0, measured_completion_tokens=0,
                unknown_input_usage=0, unknown_completion_usage=0)

def count_cost(totals, hit, attempted, ok, usage):
    totals['logical_calls'] += 1
    totals['cache_hits'] += int(hit)
    totals['failed_calls'] += int(not ok)
    if attempted:
        totals['inference_attempts'] += 1
        for label, value in (('input', usage.get('prompt_tokens', usage.get('input_tokens'))),
                             ('completion', usage.get('completion_tokens'))):
            if type(value) is int and value >= 0:
                totals['measured_'+label+'_tokens'] += value
            else:
                totals['unknown_'+label+'_usage'] += 1


class OnlineTransactions:
    """An isolated family bank and exactly one publishable terminal state per native attempt."""
    def _check_family(self, logs):
        ids = [ep.get('task_id') for ep in logs]
        if (len(set(ids)) != len(ids)
                or any(t not in self.task_bindings for t in ids)
                or any(ep.get('environment') != self.domain or ep.get('family') != self.family
                       or ep.get('native_reward_verified') is not True or ep.get('error') for ep in logs)):
            raise ValueError('Task Memory requires distinct verified items from this registered family')

    def snapshot(self, program, logs, prior_snapshot=None):
        """Restore an identical mechanism exactly; a new schema replays the same prior experiences."""
        self._check_family(logs)
        if prior_snapshot is not None:
            snap = prior_snapshot
            if (snap['snapshot_hash'] != digest({k:v for k,v in snap.items() if k != 'snapshot_hash'})
                    or snap['source_log_hash'] != digest(logs) or snap['environment'] != self.domain
                    or snap['memory_scope'] != 'family'
                    or json.loads(snap['context_wire']).get('family') != self.family):
                raise ValueError('Snapshot does not match this family or experience prefix')
            if snap['program_hash'] == program.hash:
                return CodeMemory.from_snapshot(program, snap, self.tokens, self.context,
                                                capabilities=self.caps).export_snapshot()
        key = digest([self.fingerprint, program.hash, ordered_digest(logs)])
        target = self.root/'rebuilds'/(key+'.json')
        with _lock(target.with_suffix('.lock')):
            if target.exists():
                value = read(target)
                if value['source_log_hash'] != digest(logs):
                    raise ValueError('Rebuilt snapshot belongs to a different history')
                return CodeMemory.from_snapshot(program, value, self.tokens, self.context,
                                                capabilities=self.caps).export_snapshot()
            # Replaying this family's history can already dispatch paid Hook
            # calls before the rebuilt snapshot is available to resume.
            started = target.with_suffix('.started.json')
            if started.exists():
                raise RuntimeError('Unresolved pre-item snapshot rebuild; inspect capability receipts: ' + str(started))
            bind(started, dict(key=key, program_hash=program.hash, history_hash=ordered_digest(logs)))
            bank = CodeMemory(program, logs, self.tokens, self.context, capabilities=self.caps)
            value = bank.export_snapshot()
            bind(target, value)
            return value

    def _directory(self, task, program, phase):
        fragment = Path(phase)
        if not phase or fragment.is_absolute() or '..' in fragment.parts or str(fragment) != phase:
            raise ValueError('Invalid measurement phase')
        if self.task_bindings.get(task['id']) != digest(task):
            raise ValueError('Task differs from registered family stream')
        return self.root/'episodes'/fragment/program.hash/digest(task['id'])

    def _binding(self, program, task, logs, phase, snapshot):
        return dict(evaluator=self.fingerprint, program=program.identity_hash, family=self.family,
                    task=digest(task), history=ordered_digest(logs), pre_item=snapshot['snapshot_hash'], phase=phase)

    def _cost_directory(self, task):
        if self.task_bindings.get(task['id']) != digest(task):
            raise ValueError('Cost accounting requires a registered item')
        return self.root/'item_costs'/digest(task['id'])

    def begin_item(self, task):
        directory = self._cost_directory(task)
        if self.caps is not None:
            # A family reuses one broker and the dataset-wide exact-request cache.
            # Attribute snapshot reconstruction, diagnostics and rollout to the current item.
            self.caps.ledger = directory/'memory_models.jsonl'
            self.caps.phase = 'online/item/'+digest(task)

    def item_model_costs(self, task):
        directory = self._cost_directory(task)
        actor_totals, actor_chain = self._actor_costs(task, directory)
        memory_totals = cost_totals()
        memory_chain = None
        ledger = directory/'memory_models.jsonl'
        if self.caps is not None and (self.caps.ledger != ledger or self.caps.phase != 'online/item/'+digest(task)):
            raise RuntimeError('Memory-model cost ledger is not scoped to the current item')
        if ledger.exists():
            with ledger.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if (self.caps is None or row['environment'] != self.domain or row['phase'] != self.caps.phase
                            or row['kind'] != 'memory_model_call' or row['operation'] != 'event'
                            or row.get('capability_fingerprint', self.caps.fingerprint) != self.caps.fingerprint):
                        raise RuntimeError('Memory-model receipt belongs to another item or broker')
                    if 'cache_key' in row and row.get('cache_path'):
                        key = row['cache_key']
                        path = Path(row['cache_path'])
                        if row.get('failure_kind') == 'infrastructure':
                            if (path.parent != self.caps.root/'failures' or path.suffix != '.json'
                                    or not path.stem.startswith(key+'-') or not path.stem[len(key)+1:].isdigit()
                                    or row['cache_hit'] or row['ok']):
                                raise RuntimeError('Invalid uncached memory-model failure receipt')
                        elif path != self.caps.root/'cache'/(key+'.json'):
                            raise RuntimeError('Invalid memory-model cache reference')
                        saved = read(path)
                        if (saved['key'] != key or saved['capability_fingerprint'] != self.caps.fingerprint
                                or saved['environment'] != self.domain or saved.get('failure_kind') != row.get('failure_kind')
                                or saved['record_hash'] != row['cache_record_hash']
                                or saved['record_hash'] != digest({k:v for k,v in saved.items() if k != 'record_hash'})
                                or saved['model_id'] != row['model_id'] or row['ok'] != saved['ok']
                                or row['inference_attempted'] != (not row['cache_hit'] and saved['inference_attempted'])
                                or row.get('usage') != ({} if row['cache_hit'] else public_usage(saved['usage']))):
                            raise RuntimeError('Memory-model usage differs from its exact-request receipt')
                    count_cost(memory_totals, row['cache_hit'], row.get('inference_attempted', False),
                          row['ok'], row.get('usage') or {})
                    memory_chain = digest([memory_chain, row])
        return dict(actor=actor_totals, memory_models=memory_totals, actor_receipts_hash=actor_chain,
                    memory_model_receipts_hash=memory_chain)

    def _load_measurement(self, directory, binding, task):
        saved = read(directory/'measurement.json')
        ep = json.loads(saved['episode_wire'])
        if (saved['binding'] != binding or saved['episode_hash'] != ordered_digest(ep)
                or read(directory/'episode.json') != ep):
            raise RuntimeError('Saved same-question attempt or pre-item memory differs')
        if ep.get('native_reward_verified'):
            if self._verify_native(directory, ep, task) != ep.get('native_outcome'):
                raise RuntimeError('Native outcome differs from its original receipts')
            if (digest(saved['state_wire']) != saved['state_wire_hash']
                    or digest(json.loads(saved['state_wire'])) != ep['memory_hook']['events'][-1]['state_hash_after']):
                raise RuntimeError('Terminal Hook state differs from the measured event')
        return ep, saved['state_wire']

    def _evaluate_one(self, program, logs, task, phase, snapshot):
        if any(ep['task_id'] == task['id'] for ep in logs):
            raise ValueError('Attempt would read the current item from its own Task Memory')
        directory = self._directory(task, program, phase)
        binding = self._binding(program, task, logs, phase, snapshot)
        with _lock(self.measurement_locks/(digest(task['id'])+'.lock')):
            if (directory/'measurement.json').exists():
                return self._load_measurement(directory, binding, task)[0]
            if (directory/'measurement_started.json').exists():
                raise RuntimeError('Unresolved formal item attempt; inspect native receipts before retrying')
            bind(directory/'measurement_started.json', binding)
            bank = CodeMemory.from_snapshot(program, snapshot, self.tokens, self.context, capabilities=self.caps)
            capture = _TerminalState(program)
            bank.program = capture
            ep = self._run_attempt(program, bank, task, phase, directory)
            if ep['error'] is None:
                if (not ep['native_reward_verified'] or not ep['memory_hook']['actor_delivery_verified']
                        or ep['memory_hash_before'] != snapshot['memory_hash']
                        or ep['memory_hash_after'] != ep['memory_hash_before'] or capture.state_wire is None):
                    raise RuntimeError('Invalid native attempt or read-only bank changed')
                ep['native_outcome'] = self._verify_native(directory, ep, task)
            write(directory/'episode.json', ep)
            bind(directory/'measurement.json', dict(binding=binding, episode_hash=ordered_digest(ep),
                 episode_wire=ordered_wire(ep), state_wire=capture.state_wire, state_wire_hash=digest(capture.state_wire)))
            return self._load_measurement(directory, binding, task)[0]

    def __call__(self, program, logs, tasks, phase, *, memory_snapshot=None):
        snapshot = self.snapshot(program, logs, memory_snapshot)
        return [self._evaluate_one(program, logs, task, phase, snapshot) for task in tasks]

    def _measured(self, program, logs, task, phase, memory_snapshot):
        if not memory_snapshot or memory_snapshot.get('program_hash') != program.hash:
            raise ValueError('Reading a measurement requires its original pre-item program snapshot')
        pre = self.snapshot(program, logs, memory_snapshot)
        directory = self._directory(task, program, phase)
        binding = self._binding(program, task, logs, phase, pre)
        return self._load_measurement(directory, binding, task)

    def measured_episode(self, program, logs, task, phase, *, memory_snapshot):
        """Read the original ordered evidence without rerunning a missing measurement."""
        return self._measured(program, logs, task, phase, memory_snapshot)[0]

    def commit(self, program, logs, task, episode, *, memory_snapshot):
        """Return the selected post-item snapshot, without rerunning the Actor, grader or Hook."""
        self._check_family(logs)
        if episode.get('program_hash') != program.hash or episode.get('task_id') != task['id']:
            raise ValueError('Selected attempt belongs to a different program or item')
        if any(ep['task_id'] == task['id'] for ep in logs):
            raise ValueError('Item already present in Task Memory')
        pre = memory_snapshot
        measured, state_wire = self._measured(program, logs, task, episode['phase'], pre)
        if ordered_digest(measured) != ordered_digest(episode) or not measured['native_reward_verified'] or measured['error']:
            raise ValueError('Only a verified, measured attempt can be selected')
        bank = CodeMemory.from_snapshot(program, pre, self.tokens, self.context, capabilities=self.caps)
        bank.commit_state(json.loads(state_wire), measured, pre['memory_hash'], dict(
            selected_measurement=ordered_digest(measured), phase=episode['phase']))
        bank.source_log_hash = digest(logs + [measured])
        return bank.export_snapshot()


class OnlineEvaluator(OnlineTransactions, benchmark.BenchmarkEvaluator):
    """Native one-turn MATH/APPS execution with exact Actor request reuse."""
    def __init__(self, root, domain, adapter, tasks, tokenizer, model_hash, context, capabilities, run_root):
        family = context['family']
        if (not tasks or context.get('memory_scope') != 'family'
                or any(t['split'] != 'test' or t['family'] != family for t in tasks)):
            raise ValueError('An online evaluator requires official test items of exactly one family')
        super().__init__(root, domain, adapter, tasks, tokenizer, model_hash, context, capabilities, run_root)
        self.measurement_locks = self.cache/'item_locks'
        self.family = family
        self.fingerprint = digest(dict(native=self.fingerprint, online=file_hash(Path(__file__)),
                                       actor_model=model_hash, executor=actor.PARAMETERS))

    def _actor_request(self, task, prompt, slot):
        request = dict(model=MODEL_NAME, messages=actor.messages_for(self.prompt_domain, prompt, slot),
                       seed=fixed_task_seed(task), **actor.PARAMETERS)
        key = digest(dict(task=task['id'], dataset=self.domain, request=request,
                          endpoint=actor.URL, model_hash=self.model_hash))
        return request, key

    async def answer(self, task, prompt, slot, directory):
        # _evaluate_one holds the same task lock around cache lookup and completion.
        request, key = self._actor_request(task, prompt, slot)
        output = self.cache/key/'actor.json'
        hit = output.is_file()
        reply = await super().answer(task, prompt, slot, directory)
        reference = read(directory/'actor_reference.json')
        if (key != reference['id'] or reference['hash'] != file_hash(output)
                or reply['request_hash'] != digest(request)):
            raise RuntimeError('Actor cost cannot be attributed to the measured request')
        logical = str(directory.relative_to(self.root))
        bind(self._cost_directory(task)/'actor'/(digest(logical)+'.json'), dict(
            task_hash=digest(task), logical=logical, reference=reference, cache_hit=hit))
        return reply

    def _verify_native(self, directory, ep, task):
        ref = read(directory/'actor_reference.json')
        physical = self.cache/ref['id']
        reply = read(physical/'actor.json')
        grade = read(directory/'grade.json')
        slot = ep['native_recorder']['memory_retrieved']
        request, key = self._actor_request(task, self.public[task['id']], slot)
        if (key != ref['id'] or file_hash(physical/'actor.json') != ref['hash']
                or ref['finish'] != reply['finish']
                or reply['ok'] is not True or reply['http_status'] != 200 or not reply['response_id']
                or reply['model_hash'] != self.model_hash or reply['endpoint'] != actor.URL
                or reply['served_model'] != MODEL_NAME or reply['request'] != request
                or reply['request_hash'] != digest(request)
                or ep['history'][0]['action'] != reply['answer']
                or grade != read(physical/'grade.json') or grade['score'] != ep['score']
                or ep['native_recorder']['reward'] != ep['score']):
            raise RuntimeError('Online evaluation does not bind the native answer, grade and Actor input')
        return dict(finish=reply['finish'], completion_tokens=reply['usage']['completion_tokens'],
                    max_tokens=request['max_tokens'], **{k:grade[k] for k in
                    ('score','status','tests_total','tests_run','tests_passed','first_failed_index') if k in grade})

    def _actor_costs(self, task, directory):
        actor_totals = cost_totals()
        actor_chain = None
        first_seen = False
        for path in sorted((directory/'actor').glob('*.json')):
            access = read(path)
            ref = access['reference']
            logical = Path(access['logical'])
            output = self.cache/ref['id']/'actor.json'
            if (logical.is_absolute() or '..' in logical.parts or path.stem != digest(str(logical))
                    or access['task_hash'] != digest(task) or type(access['cache_hit']) is not bool
                    or ref != read(self.root/logical/'actor_reference.json') or file_hash(output) != ref['hash']):
                raise RuntimeError('Actor cost receipt does not bind its task and logical attempt')
            count_cost(actor_totals, access['cache_hit'], not access['cache_hit'], True, read(output)['usage'])
            actor_chain = digest([actor_chain, access])
            first_seen |= logical.parts[:2] == ('episodes', 'first')
        if not first_seen:
            raise RuntimeError('Reviewer cost summary requires the first measured Actor request')

        return actor_totals, actor_chain

    def _run_attempt(self, program, bank, task, phase, directory):
        return asyncio.run(self.episode(program, bank, task, phase, False))
