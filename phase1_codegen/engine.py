"""Fixed code-edit search; generated Task Memory is the only evolving component."""
from pathlib import Path
import hashlib
import json
import time

from evomem.common import append, digest, dumps, file_hash, read, write
from evomem.alma_backend import public_context as visible_context
from .experience import EvolutionMemory, public_event, mechanism_query, private_learning_texts, RETRIEVAL_TOKENS
from .outcomes import candidate_hook_failure, comparison_for_feedback, reported_score
from .memory import CodeMemory
from .policy import CapabilityInfrastructureError, Rejected
from .program import BASELINE_SOURCE, Program, WorkerStartupError
from .store import RunStore
from .acceptance import POLICIES, DUAL_POLICIES, REVIEWER_DIRECTED, eligible_for_adoption, adoption_feedback
from .behavior_diagnostics import iter_execution_receipts
from .protocol import (native_executor, HOOK_VERSION, WORKFLOW_VERSION, EVENTS, EVENT_CONTRACT, HookLimits, CAPACITY_TOKENS, READ_TOKENS, READ_TOP_K)


def code_binding():
    root = Path(__file__).resolve().parents[1]
    paths = [p for package in ('phase1_codegen', 'evomem', 'scripts', 'external/alma')
             for p in (root / package).rglob('*.py')
             if '__pycache__' not in p.parts
             and not (package == 'phase1_codegen' and 'tests' in p.relative_to(root / package).parts)]
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(set(paths)) if p.exists()}


class MeasurementFailure(RuntimeError):
    """The evaluator or the host could not attest a native measurement."""


class ProposalCallFailure(RuntimeError):
    """A role call raised before delivering a parsed proposal to the host."""


def public_context(context, logs, episodes):
    def success_fraction(rows):
        scores = [value for row in rows if (value := reported_score(row)) is not None]
        return sum(value >= 1 for value in scores) / len(scores) if scores else None

    return dict(environment=context.get('environment', 'alfworld'), family=context.get('family'),
                memory_scope=context.get('memory_scope', 'family'),
                memory_families=sorted({e['family'] for e in logs if e.get('family')}),
                adaptation_episodes=len(logs),
                adaptation_success=success_fraction(logs),
                mean_steps=sum(len(e.get('history', [])) for e in episodes) / max(1, len(episodes)),
                exploration_success=success_fraction(episodes),
                exploration_errors=sum(bool(e.get('error')) for e in episodes))


def own_trace(ep):
    # Shared-scope adaptation may include other families in the same dataset.
    # Independent confirmation/generalization and foreign datasets remain excluded.
    trace = {k: ep[k] for k in ('task_id', 'group_id', 'room', 'environment', 'family', 'goal', 'history', 'score', 'done', 'error',
                              'phase', 'executor', 'program_hash', 'memory_hash_before', 'memory_hash_after',
                              'read_only_memory', 'synthetic', 'native_reward_verified', 'seed',
                              'environment_seed', 'memory_hook', 'task_input_receipt') if k in ep}
    if 'score' in trace:
        trace['score'] = reported_score(ep)
    return trace


def observed_behavior_comparison(before, after):
    """Compare recorded outputs on aligned probes; never gate execution/adoption."""
    if before is None or after is None:
        return None
    contexts = before.get('context_hashes', [])
    aligned = bool(contexts) and contexts == after.get('context_hashes', [])
    def ordered_equal(left, right):
        return json.dumps(left, ensure_ascii=False, sort_keys=False) == json.dumps(right, ensure_ascii=False, sort_keys=False)
    complete = (aligned and len(before.get('readouts', [])) == len(after.get('readouts', [])) == len(contexts))
    state_wires = [value.get('runtime_snapshot', {}).get('state_wire') for value in (before, after)]
    exact_state = all(isinstance(value, str) for value in state_wires)
    exact_readouts = all(len(value.get('input_hashes', [])) == len(contexts) for value in (before, after))
    return dict(inputs_identical=aligned, input_scope='recorded task_start query contexts only',
                task_count=len(contexts), context_hashes=contexts,
                stored_state_equal=(state_wires[0] == state_wires[1] if exact_state else
                    ordered_equal(before['stored_state'], after['stored_state'])
                    if 'stored_state' in before and 'stored_state' in after else None),
                stored_state_comparison_source='exact_snapshot_state_wire' if exact_state else 'displayed_state_only',
                task_start_readouts_equal=((before['input_hashes'] == after['input_hashes'] if exact_readouts else
                    ordered_equal(before['readouts'], after['readouts'])) if complete else None),
                readout_comparison_source='exact_slot_wire_hashes' if exact_readouts else 'displayed_readouts_only',
                comparison_complete=complete,
                source_evidence_hash=digest(dict(parent=before, candidate=after)),
                boundary='Finite observed post-frontier banks and task_start readouts only; object order is preserved. '
                         'Write histories may differ. This is not all-input semantic equivalence, not a later-event '
                         'comparison, and never skips native validation or changes adoption.')


def review_observation_context(shown, report):
    """Carry exact execution-point/unit metadata with scalar observations.

    A branch-count array index alone does not identify the condition in the
    current code. Only metadata already displayed to the Reviewer is copied;
    there are no inferred explanations or modification suggestions here.
    """
    rows = {row['id']: row for row in shown['evidence']}
    contexts = {}
    fields = ('function', 'line', 'column', 'expression', 'operation', 'unit',
              'stored_tokens_unit', 'read_tokens_unit', 'score_subject')
    for obs in report.get('evidence_contract', {}).get('observations', []):
        row = rows[obs['evidence_id']]
        node, path = row['data'], ''
        for part in obs['pointer'].split('/')[1:]:
            if isinstance(node, dict):
                metadata = {key: node[key] for key in fields if key in node
                            and type(node[key]) in (str, int, float, bool)}
                if metadata:
                    key = (row['id'], path)
                    entry = contexts.setdefault(key, dict(evidence_id=row['id'], pointer=path,
                                                          observation_ids=[], metadata=metadata))
                    if obs['id'] not in entry['observation_ids']:
                        entry['observation_ids'].append(obs['id'])
            key = part.replace('~1', '/').replace('~0', '~')
            node = node[int(key)] if isinstance(node, list) else node[key]
            path += '/' + part
    return list(contexts.values())


class EvolutionEngine:
    def __init__(self, out, tokenizer, proposer, evaluator, evolution_memory, context, logs,
                 exploration_tasks, confirmation_blocks, initial_source=BASELINE_SOURCE,
                 seed=11, synthetic=False, reviewer=None, acceptance_policy='reviewer_and_reward', max_rounds=None,
                 capabilities=None, learning_blocks=None, initial_memory_snapshot=None):
        self.store = RunStore(out)
        self.tokenizer, self.proposer, self.evaluator = tokenizer, proposer, evaluator
        self.reviewer = reviewer
        self.capabilities = capabilities
        if reviewer is not None and reviewer is proposer:
            raise ValueError('Reviewer and Evolver must be distinct role objects')
        if acceptance_policy == 'reviewer_opinion' or (not synthetic and acceptance_policy not in DUAL_POLICIES):
            raise ValueError('New runs require a dual-evidence policy; retired policies are for historical validation only')
        if acceptance_policy not in POLICIES:
            raise ValueError('Unknown acceptance_policy')
        if acceptance_policy in DUAL_POLICIES and not getattr(reviewer, 'fingerprint', None):
            raise ValueError('Reviewer-opinion adoption requires the independent fixed reviewer')
        if not synthetic and not getattr(reviewer, 'fingerprint', None):
            raise ValueError('Formal code evolution requires the fixed mechanism reviewer')
        self.memory = evolution_memory
        context = dict(context)
        context.setdefault('memory_hook_version', HOOK_VERSION)
        self.context, self.logs = context, logs
        if context['memory_hook_version'] != HOOK_VERSION:
            raise ValueError('unsupported memory hook version')
        self.initial_logs = logs
        # Task manifests are metadata; canonicalize their key order once so
        # persisted bindings and resumed requests have identical wire identities.
        # Actual experience/state insertion order is preserved separately.
        self.learning_blocks = json.loads(dumps(learning_blocks)) if learning_blocks is not None else None
        self.task_memory_mode = 'continual' if learning_blocks is not None else 'frozen_replay'
        self.initial_memory_snapshot = initial_memory_snapshot
        self.active_snapshot = None
        self.active_snapshot_ref = None
        self._runtime_snapshots = {}
        self._round_index = 0
        self.memory_scope = context.get('memory_scope', 'family')
        if self.memory_scope not in {'family', 'environment'}:
            raise ValueError('memory_scope must be family or environment')
        self.explore, self.blocks = exploration_tasks, confirmation_blocks
        self.acceptance_policy = acceptance_policy
        if acceptance_policy == REVIEWER_DIRECTED:
            if max_rounds is not None or learning_blocks is not None or self.blocks:
                raise ValueError('Reviewer-directed same-item repair uses a fixed pre-item history and no round cap')
            self.max_rounds = None
        else:
            self.max_rounds = len(self.blocks) if max_rounds is None else max_rounds
        self.initial, self.seed, self.synthetic = Program(initial_source), seed, synthetic
        self.environment_step_limit = getattr(evaluator, 'step_limit', 30)
        if type(self.environment_step_limit) is not int or self.environment_step_limit <= 0:
            raise ValueError('Evaluator must declare a positive native environment step limit')
        self._validate_plan()
        self.binding = dict(schema=2, api='code-memory-review-v4', review_contract=2,
                            workflow=WORKFLOW_VERSION,
                            synthetic=synthetic, seed=seed,
                            context=context, logs_hash=digest(logs), exploration_tasks=exploration_tasks,
                            ordered_logs_hash=digest(json.dumps(logs, ensure_ascii=False, sort_keys=False)),
                            ordered_context_hash=digest(json.dumps(context, ensure_ascii=False, sort_keys=False)),
                            confirmation_blocks=confirmation_blocks, source_memory_hash=digest(evolution_memory.records),
                            evaluator=evaluator.fingerprint, proposer=getattr(proposer, 'fingerprint', 'synthetic'),
                            environment_step_limit=self.environment_step_limit,
                            reviewer=getattr(reviewer, 'fingerprint', None), review_required=reviewer is not None,
                            review_schedule='pre-review after learning; one comparative post-review per proposal; reuse only for the same program, bank and corpus',
                            code=code_binding(), storage_tokens=CAPACITY_TOKENS, read_tokens=READ_TOKENS, top_k=READ_TOP_K,
                            evolution_memory_tokens=3072,
                            acceptance_policy=acceptance_policy, max_rounds=self.max_rounds,
                            task_memory_mode=self.task_memory_mode, learning_blocks=self.learning_blocks,
                            acceptance=POLICIES[acceptance_policy])
        if acceptance_policy == REVIEWER_DIRECTED:
            self.binding['review_schedule'] = 'One initial diagnosis; each post decision guides the next edit or ends this item.'
        if capabilities is not None:
            if context.get('environment') != capabilities.environment:
                raise ValueError('Capability pool and evolution environment must match')
            self.binding['model_capabilities'] = capabilities.fingerprint
        if self.task_memory_mode == 'continual':
            self.binding['initial_task_memory_snapshot'] = self._initial_task_memory(initial_memory_snapshot)

    def _initial_task_memory(self, supplied):
        from .learning import make_snapshot, save_snapshot, load_snapshot, snapshot_memory, snapshot_logs, ordered_digest
        request = dict(binding_hash=digest(self.binding), program_hash=self.initial.hash,
                       supplied_snapshot_hash=digest(supplied) if supplied is not None else None)
        anchor_path = self.store.root / 'initial_task_memory.json'
        pending_path = self.store.root / 'initial_task_memory_attempt.json'
        with self.store.lock():
            if anchor_path.exists():
                anchor = read(anchor_path)
                if anchor.get('request') != request:
                    raise RuntimeError('Initial Task Memory inputs or runtime changed; use a new run')
                reference = anchor['snapshot']
                snapshot = load_snapshot(self.store.root, reference)
                if ordered_digest(snapshot_logs(snapshot)) != ordered_digest(self.initial_logs):
                    raise RuntimeError('Initial learning corpus changed during resume')
                memory = CodeMemory.from_snapshot(self.initial, snapshot_memory(snapshot), self.tokenizer,
                                                  capabilities=self.capabilities)
                if memory.environment != self.context['environment']:
                    raise ValueError('Inherited Task Memory belongs to a different environment')
                return reference
            if pending_path.exists():
                raise RuntimeError('Unresolved initial Task Memory attempt; inspect artifacts before retrying')
            if (self.store.root / 'state.json').exists():
                raise RuntimeError('Continual run lacks its initial Task Memory anchor; use a new run')
            # Publish intent before any generated write/model call. An interruption
            # cannot silently repeat a partially completed initial construction.
            write(pending_path, dict(request=request, started=time.time()))
            if supplied is None:
                memory = CodeMemory(self.initial, self.initial_logs, self.tokenizer, self.context,
                                    capabilities=self.capabilities)
            else:
                memory = CodeMemory.from_snapshot(self.initial, supplied, self.tokenizer,
                                                  capabilities=self.capabilities)
            if memory.environment != self.context['environment']:
                raise ValueError('Inherited Task Memory belongs to a different environment')
            snapshot = make_snapshot(self.initial_logs, self.context['environment'],
                                     memory_snapshot=memory.export_snapshot())
            reference = save_snapshot(self.store.root, snapshot)
            write(anchor_path, dict(request=request, snapshot=reference))
            return reference

    def _validate_plan(self):
        learning_blocks = getattr(self, 'learning_blocks', None)
        if (not self.logs and not learning_blocks) or not self.explore or any(not b for b in self.blocks):
            raise ValueError('Nonempty adaptation/exploration and any supplied confirmation blocks required')
        if type(self.max_rounds) is not int or self.max_rounds <= 0:
            raise ValueError('A positive frozen max_rounds budget is required')
        if self.acceptance_policy not in POLICIES:
            raise ValueError('Unknown acceptance_policy')
        if self.acceptance_policy == 'native_reward' and self.max_rounds > len(self.blocks):
            raise ValueError('Native-reward adoption needs a fresh confirmation block per planned round')
        groups, ids = set(), set()
        if learning_blocks is not None:
            if (type(learning_blocks) is not list or len(learning_blocks) != self.max_rounds
                    or any(type(block) is not list for block in learning_blocks)
                    or not any(learning_blocks)):
                raise ValueError('Continual memory requires explicit learning blocks for every round')
            if not self.context.get('environment'):
                raise ValueError('Continual Task Memory requires a dataset environment')
            if self.memory_scope != 'environment':
                raise ValueError('Continual Task Memory uses explicit within-dataset environment scope')
        roles = [self.logs, self.explore] + list(self.blocks) + (learning_blocks or [])
        families = set()
        for block in roles:
            local_ids, local_groups = set(), set()
            for item in block:
                tid = item.get('task_id', item.get('id'))
                group = item.get('group_id')
                if not tid or not group or tid in local_ids:
                    raise ValueError('Every task requires a unique ID and a group ID')
                local_ids.add(tid); local_groups.add(group)
                families.add(item.get('family'))
            if ids & local_ids or groups & local_groups:
                raise ValueError('Adaptation/exploration/fresh confirmation blocks must be group-disjoint')
            ids |= local_ids; groups |= local_groups
        environment = self.context.get('environment')
        if self.memory_scope == 'environment':
            if type(environment) is not str or not environment:
                raise ValueError('Environment-shared memory requires an explicit dataset environment')
            if any(type(family) is not str or not family for family in families):
                raise ValueError('Shared experiences and tasks require explicit families')
            target_families = {item.get('family') for block in [self.explore] + list(self.blocks) + (learning_blocks or []) for item in block}
            if target_families != {self.context.get('family')}:
                raise ValueError('Exploration and confirmation must retain the current target family')
            if any(item.get('environment') != environment for item in self.logs):
                raise ValueError('Shared Task Memory logs require explicit matching dataset provenance')
            if any(record.get('observed_context', {}).get('environment') != environment for record in self.memory.records):
                raise ValueError('Shared Evolution Memory must remain within its dataset environment')
        elif len(families) != 1 or None in families or self.context.get('family') not in families:
            raise ValueError('Task Memory is isolated to one target family')
        # Historical task manifests may omit environment; native adapters verify
        # their official inventory. Explicit foreign provenance is never allowed.
        if environment is not None:
            if any(item.get('environment') not in (None, environment) for block in roles for item in block):
                raise ValueError('Task or experience belongs to a different dataset environment')
            if any(record.get('observed_context', {}).get('environment') not in (None, environment)
                   for record in self.memory.records):
                raise ValueError('Evolution Memory belongs to a different dataset environment')
        sealed = ('final', 'confirm', 'generalization', 'heldout', 'held_out',
                  'eval_in_distribution', 'eval_out_of_distribution')
        if any(any(marker in str(item.get(field, '')).lower() for marker in sealed)
               or (field in ('split', 'official_split') and str(item.get(field, '')).lower() == 'test')
               for item in self.logs + [t for block in (learning_blocks or []) for t in block]
               for field in ('phase', 'split', 'official_split', 'data_role')):
            raise ValueError('Confirmation/final/generalization outcomes cannot become shared adaptation evidence')
        if not self.synthetic:
            if any(r.get('synthetic') is not False or not r.get('native_reward_verified') or r.get('error') for r in self.logs):
                raise ValueError('Formal adaptation must be successful native recordings, not mock data/errors')
            if any(r.get('synthetic') for r in self.memory.records):
                raise ValueError('Synthetic Evolution Memory cannot enter a formal run')
            if not getattr(self.proposer, 'fingerprint', None):
                raise ValueError('Formal proposer requires a frozen fingerprint')

    def _complete(self, episodes, tasks, *, learning=False):
        if not tasks or len(episodes) != len(tasks) or [e.get('task_id') for e in episodes] != [t['id'] for t in tasks]:
            return False
        for ep, task in zip(episodes, tasks):
            score = reported_score(ep)
            if score is None:
                return False
            if ep.get('memory_hook') and (not ep['memory_hook'].get('complete')
                    or not ep['memory_hook'].get('actor_delivery_verified')
                    or ep['memory_hook'].get('committed') is not learning):
                return False
            if not self.synthetic and (ep.get('synthetic') is not False or not ep.get('native_reward_verified')
                                      or ep.get('executor') != native_executor(self.context.get('environment'))
                                      or ep.get('environment_seed') != task['seed']
                                      or ep.get('seed') != 42000 + int(digest(task['id'])[:6], 16) % 10000000
                                      or ep.get('native_recorder', {}).get('reward') != score
                                      or len(ep.get('history', [])) > self.environment_step_limit
                                      or ep.get('memory_write_calls') != int(learning)
                                      or (not learning and (ep.get('generated_memory_write_calls') != 0
                                                           or ep.get('read_only_memory') is not True))
                                      or (learning and (ep.get('generated_memory_write_calls') != 1
                                                       or ep.get('read_only_memory') is not False))):
                return False
        return True

    def _measure(self, program, tasks, phase, memory_snapshot=None):
        if self.task_memory_mode == 'continual' and memory_snapshot is None:
            memory_snapshot = self._runtime_snapshots.get(program.hash)
            if memory_snapshot is None and self.active_snapshot is not None and program.hash == self.active_snapshot['program_hash']:
                from .learning import snapshot_memory
                memory_snapshot = snapshot_memory(self.active_snapshot)
        def validate_bank(episodes):
            if self.synthetic:
                return
            for ep in episodes:
                if ep.get('error'):
                    continue
                if (ep.get('program_hash') != program.hash
                        or ep.get('read_only_memory') is not True
                        or ep.get('memory_write_calls') != 0
                        or ep.get('generated_memory_write_calls') != 0
                        or ep.get('memory_hash_before') != ep.get('memory_hash_after')
                        or (memory_snapshot is not None and ep.get('memory_hash_before') != memory_snapshot['memory_hash'])):
                    raise RuntimeError('Native measurement does not match the fixed read-only program and bank')
        key_data = dict(binding=self.binding, program=program.hash, tasks=tasks)
        if self.active_snapshot_ref is not None:
            key_data['task_memory_snapshot'] = self.active_snapshot_ref['hash']
        if memory_snapshot is not None:
            key_data['runtime_snapshot'] = digest(memory_snapshot)
        key = digest(key_data)
        path = self.store.root / 'measurements' / (key + '.json')
        def verified_episodes(rows):
            if [row.get('task_id') for row in rows] != [task['id'] for task in tasks]:
                raise MeasurementFailure('Evaluator returned missing or mismatched task IDs')
            validate_bank(rows)
            if not self.synthetic:
                for row, task in zip(rows, tasks):
                    if self._complete([row], [task]):
                        continue
                    if phase.endswith('/candidate') and candidate_hook_failure(row, program.hash):
                        continue
                    raise MeasurementFailure('Native measurement could not be verified for %s: %s' %
                                             (task['id'], row.get('error') or 'receipt mismatch'))
            return [dict(row, score=reported_score(row)) for row in rows]

        failure_path = self.store.root / 'measurement_failures' / (key + '.json')
        if failure_path.exists():
            failed = read(failure_path)
            if failed.get('key') != key or failed.get('candidate_measurement_committed') is not False:
                raise MeasurementFailure('Invalid measurement failure receipt')
            message = 'Previously failed measurement: %s: %s' % (failed['error_type'], failed['error'])
            if failed['error_type'] == 'WorkerStartupError':
                raise WorkerStartupError(message, details=failed.get('infrastructure_details'))
            raise MeasurementFailure(message)
        started = time.monotonic()
        episodes = None
        try:
            if path.exists():
                receipt = read(path)
                if receipt['key'] != key or digest(receipt['episodes']) != receipt['episodes_hash']:
                    raise MeasurementFailure('Cached measurement changed')
                episodes = verified_episodes(receipt['episodes'])
                append(self.store.root / 'measurement_access.jsonl', dict(key=key, phase=phase, physical=False))
                return episodes
            options = {}
            if memory_snapshot is not None:
                options['memory_snapshot'] = memory_snapshot
            elif self.active_snapshot is not None and program.hash == self.active_snapshot['program_hash']:
                from .learning import snapshot_memory
                options['memory_snapshot'] = snapshot_memory(self.active_snapshot)
            episodes = self.evaluator(program, self.logs, tasks, phase, **options)
            episodes = verified_episodes(episodes)
        except Exception as exc:
            write(failure_path, dict(
                key=key, phase=phase, program_hash=program.hash, task_ids=[task['id'] for task in tasks],
                error_type=type(exc).__name__, error=str(exc), infrastructure_details=getattr(exc, 'details', {}),
                returned_episode_hash=digest(episodes) if isinstance(episodes, list) else None,
                seconds=time.monotonic() - started, synthetic=self.synthetic, candidate_measurement_committed=False))
            if isinstance(exc, (WorkerStartupError, MeasurementFailure)):
                raise
            raise MeasurementFailure(f'{type(exc).__name__}: {exc}') from exc
        receipt = dict(key=key, episodes=episodes, episodes_hash=digest(episodes), phase=phase,
                       synthetic=self.synthetic, seconds=time.monotonic() - started)
        write(path, receipt)
        append(self.store.root / 'measurement_access.jsonl', dict(key=key, phase=phase, physical=True,
                                                                 episodes=len(episodes), seconds=receipt['seconds']))
        return episodes

    def _pair(self, parent, candidate, tasks, phase, parent_episodes=None, candidate_episodes=None):
        a = self._measure(parent, tasks, phase + '/parent') if parent_episodes is None else parent_episodes
        b = self._measure(candidate, tasks, phase + '/candidate') if candidate_episodes is None else candidate_episodes
        complete = self._complete(a, tasks) and self._complete(b, tasks)
        mean_a = sum(reported_score(e) for e in a) / len(a) if complete else None
        mean_b = sum(reported_score(e) for e in b) / len(b) if complete else None
        return dict(complete=complete, parent=mean_a, candidate=mean_b,
                    delta=mean_b - mean_a if complete else None,
                    task_ids=[t['id'] for t in tasks], parent_evidence=digest(a), candidate_evidence=digest(b))

    def _signature(self, program, initial_contexts):
        signature_key = digest(dict(binding=self.binding, program=program.hash, contexts=initial_contexts,
                                    ordered_logs_hash=self._logs_digest(), snapshot=self.active_snapshot_ref))
        signature_path = self.store.root / 'diagnostic_signatures' / (signature_key + '.json')
        if signature_path.exists():
            receipt = read(signature_path)
            if receipt.get('key') != signature_key or digest(receipt.get('result')) != receipt.get('result_hash'):
                raise RuntimeError('Stored actual diagnostic replay changed')
            result = receipt['result']
            # Validate the actual state before reusing any readouts/measurements.
            CodeMemory.from_snapshot(program, result['runtime_snapshot'], self.tokenizer, capabilities=self.capabilities)
            self._runtime_snapshots[program.hash] = result['runtime_snapshot']
            return result
        if self.active_snapshot is not None and program.hash == self.active_snapshot['program_hash']:
            from .learning import snapshot_memory
            memory = CodeMemory.from_snapshot(program, snapshot_memory(self.active_snapshot), self.tokenizer,
                                              capabilities=self.capabilities, diagnostics=True)
        else:
            memory = CodeMemory(program, self.logs, self.tokenizer, self.context,
                                capabilities=self.capabilities, diagnostics=True)
        # This is exactly the official ALMA prompt representation after adapter clipping.
        if self.memory_scope == 'environment':
            if len(initial_contexts) != len(self.explore):
                raise ValueError('Shared-memory diagnostic contexts must align with the frozen exploration tasks')
            texts = []
            for task, initial in zip(self.explore, initial_contexts):
                view = memory.for_task(task, self.context['environment'])
                entries = view.read({}, visible=visible_context(initial))
                texts.append(json.dumps({'entries': entries} if entries else {}, ensure_ascii=False, indent=2))
        else:
            texts = []
            for initial in initial_contexts:
                entries = memory.read({}, visible=visible_context(initial))
                texts.append(json.dumps({'entries': entries} if entries else {}, ensure_ascii=False, indent=2))
        result = dict(context_hashes=[digest(i) for i in initial_contexts],
                    input_hashes=[digest(text) for text in texts],
                    memory_hash=memory.fingerprint, stored_state=memory.state,
                    stored_tokens=memory.stats['stored_tokens'], readouts=[json.loads(t) for t in texts],
                    write_trace=memory.write_trace, read_trace=memory.read_trace,
                    storage_origin=('actual_committed_learning_snapshot' if self.active_snapshot is not None
                                    and program.hash == self.active_snapshot['program_hash'] else 'replayed_learning_frontier'),
                    write_trace_scope='This diagnostic invocation only; restoring a committed bank does not replay historical writes',
                    runtime_snapshot=memory.export_snapshot())
        from .behavior_diagnostics import diagnose
        result['diagnostics'] = diagnose(program, result, self.logs)
        self._runtime_snapshots[program.hash] = result['runtime_snapshot']
        write(signature_path, dict(key=signature_key, result=result, result_hash=digest(result)))
        return result

    def _mechanism_evidence(self, label, source, behavior, episodes):
        evidence = [dict(id=label + ':program', kind='program', data=source)]
        if behavior is not None:
            evidence.append(dict(id=label + ':storage', kind='memory_storage',
                                 data=dict(state=behavior['stored_state'], stored_tokens=behavior['stored_tokens'],
                                           stored_tokens_unit='whole_bank', environment=self.context.get('environment'),
                                           memory_scope=self.memory_scope,
                                           state_top_level_keys=sorted(behavior['stored_state']),
                                           memory_hash=behavior['memory_hash'],
                                           measurement_origin=behavior.get('storage_origin', 'replayed_learning_frontier'),
                                           task_memory_snapshot_ref=self.active_snapshot_ref)))
            if behavior.get('diagnostics') is not None:
                evidence.append(dict(id=label + ':behavior', kind='behavior_diagnostics',
                                     data=behavior['diagnostics']))
            fields = ('status', 'task_id', 'stored_tokens', 'error', 'experience_hash', 'memory_hash',
                      'capability_calls', 'capability_seconds')
            evidence.append(dict(id=label + ':writes', kind='memory_write',
                                 data=dict(scope=behavior.get('write_trace_scope'),
                                           records=[dict({k: row[k] for k in fields if k in row},
                                                   stored_tokens_unit='bank_after_write') for row in behavior['write_trace']])))
            calls = [call for trace in behavior['write_trace'] + behavior['read_trace']
                     for receipt, _, _ in iter_execution_receipts(trace)
                     for call in receipt.get('capability_calls', [])]
            if calls:
                receipts = list({digest(call.get('receipt', {})): call.get('receipt', {})
                                 for call in calls}.values())
                evidence.append(dict(id=label + ':model_calls', kind='model_call_cost', data=dict(
                    scope='This diagnostic replay only; not total native execution cost',
                    program_calls=len(calls), broker_requests=len(receipts),
                    intra_invocation_reuses=len(calls) - len(receipts),
                    models=sorted({call['model_id'] for call in calls}),
                    cache_hits=sum(bool(r.get('cache_hit')) for r in receipts),
                    inference_attempts=sum(bool(r.get('inference_attempted')) for r in receipts),
                    completed_physical=sum(bool(r.get('physical')) for r in receipts),
                    failures=sum(r.get('ok') is False for r in receipts),
                    seconds=sum(r.get('seconds', 0) for r in receipts),
                    limitation='Calling a model is not evidence of memory quality or task improvement')))
            if not episodes:
                for i, output in enumerate(behavior['readouts']):
                    evidence.append(dict(id=label + ':read:' + str(i), kind='memory_readout', data=dict(
                        replayed_readout=output, query=behavior['read_trace'][i]['context'],
                        actual_trace_available=False, actual_injected_memory=None,
                        read_tokens=behavior['read_trace'][i]['read_tokens'],
                        read_tokens_unit='task_start_readout',
                        read_tokens_source='replayed_readout', replay_matches_actual=None,
                        limitation='Offline task_start only; no dynamic interaction or candidate Actor was executed')))
        for i, ep in enumerate(episodes):
            native = ep.get('native_recorder', {})
            if behavior is not None and i < len(behavior['readouts']):
                evidence.append(dict(id=label + ':read:' + str(i), kind='memory_readout', data=dict(
                    task_id=ep['task_id'], public_initial=native.get('initial'),
                    query={k: ep.get(k, behavior['read_trace'][i]['context'].get(k))
                           for k in ('task_id', 'environment', 'family', 'goal')},
                    native_score=reported_score(ep),
                    score_subject='current_query_execution_not_retrieved_memory_entry',
                    replayed_readout=behavior['readouts'][i],
                    actual_injected_memory=native.get('memory_retrieved'),
                    actual_trace_available='memory_retrieved' in native,
                    read_tokens=behavior['read_trace'][i]['read_tokens'],
                    read_tokens_unit='task_start_readout',
                    read_tokens_source='replayed_readout',
                    replay_matches_actual=(behavior['readouts'][i] == native['memory_retrieved']
                                           if 'memory_retrieved' in native else None),
                    limitation='Only initial task_start; dynamic interaction requires actual memory_hook evidence; replay is not Actor delivery')))
            if ep.get('memory_hook'):
                # Include actual changes as ordinary readout evidence so at
                # least one concrete dynamic memory witness survives budgeting.
                previous = {}
                for j, row in enumerate(ep['memory_hook'].get('injections', [])):
                    current = row.get('memory', {})
                    if current and current != previous:
                        evidence.append(dict(id=label + ':hook_read:' + str(i) + ':' + str(j), kind='memory_readout',
                            data=dict(task_id=ep['task_id'], query=dict(task_id=ep['task_id'],
                                environment=ep.get('environment'), family=ep.get('family'), goal=ep.get('goal')),
                                actual_injected_memory=current, actual_trace_available=row.get('actor_delivery_verified', False),
                                read_tokens=self.tokenizer.count(json.dumps(current, ensure_ascii=False, indent=2)),
                                read_tokens_unit='one_actor_memory_slot',
                                read_tokens_source='actual_actor_slot', replay_matches_actual=None,
                                event_type='before_action', event_index=row.get('event_sequence'),
                                provenance=dict(actor_request_hash=row.get('actor_request_hash'),
                                                system_hash=row.get('system_hash'), attempt=row.get('attempt')),
                                limitation='Actual memory-slot delivery does not prove the Actor used it')))
                    previous = current
            evidence.append(dict(id=label + ':execution:' + str(i), kind='execution_trace', data=own_trace(ep)))
        if any(':hook_read:' in row['id'] for row in evidence):
            # Reviewer retains actual dynamic witnesses before optional initial
            # diagnostics. This does not rank candidate mechanisms or outcomes.
            evidence.sort(key=lambda row: 0 if ':hook_read:' in row['id'] else 1)
        return evidence

    def _memory_context(self):
        return dict(scope=self.memory_scope, environment=self.context.get('environment'),
                    current_target_family=self.context.get('family'),
                    experience_families=sorted({row['family'] for row in self.logs}),
                    starting_program_hash=self.initial.hash,
                    starting_program_is_baseline=self.initial.source == BASELINE_SOURCE,
                    continuity='Continue the supplied current program; do not reset inherited mechanism code',
                    history_boundary=('Task and evolution histories may span task families only within this dataset'
                                      if self.memory_scope == 'environment' else 'Task histories remain within one family'))

    def _review_input(self, stage, parent, before, base_episodes,
                      trial=None, source=None, after=None, candidate_episodes=()):
        evidence = self._mechanism_evidence('parent', parent.source, before, base_episodes)
        evidence += [dict(id='adaptation:' + str(i), kind='adaptation_trace', data=own_trace(ep))
                     for i, ep in enumerate(self.logs)]
        if type(source) is str and source.strip():
            evidence += self._mechanism_evidence('candidate', source, after, candidate_episodes)
        if trial is not None:
            evidence.append(dict(id='candidate:validation', kind='validation', data=dict(
                status=trial['public_status'], error=trial.get('error'),
                mechanism_validated=trial['mechanism_validated'],
                hypothesis=trial.get('hypothesis'), risk=trial.get('risk'),
                candidate_execution_performed=bool(candidate_episodes))))
            if trial.get('exploration') is not None:
                evidence.append(dict(id='comparison:exploration', kind='paired_exploration', data=trial['exploration']))
        # Available tools belong to the designer's input. The Reviewer evaluates
        # the current code and actual call evidence, not possible future tools.
        for row in evidence:
            if isinstance(row['data'], dict):
                label = row['id'].split(':', 1)[0]
                phase = 'historical_adaptation' if label == 'adaptation' else 'current_exploration'
                provenance = dict(environment=self.context.get('environment'), round=self._round_index,
                                  phase=phase, task_id=row['data'].get('task_id'),
                                  program_hash=(parent.hash if label == 'parent' else
                                                trial.get('candidate_hash') if label == 'candidate' and trial else None),
                                  memory_snapshot_hash=(self.active_snapshot_ref or {}).get('hash'),
                                  ordered_logs_hash=self._logs_digest(),
                                  measurement_kind=row['kind'])
                if row['kind'] == 'adaptation_trace':
                    provenance.update(program_hash=row['data'].get('program_hash'),
                                      collection_phase=row['data'].get('phase'),
                                      source='historical learning input; not current program performance')
                elif row['kind'] == 'behavior_diagnostics':
                    provenance.update(row['data'].get('provenance', {}))
                    provenance['round'] = self._round_index
                    provenance['phase'] = 'offline_replay'
                elif row['kind'] == 'memory_readout':
                    # Preserve only fixed Actor-delivery identities; generated
                    # data cannot override current program/frontier provenance.
                    original = row['data'].get('provenance', {})
                    provenance.update({key: original[key] for key in
                        ('actor_request_hash', 'system_hash', 'attempt') if key in original})
                row['data']['provenance'] = provenance
        return dict(stage=stage, review_contract=2, round=self._round_index, phase=stage,
                    ordered_logs_hash=self._logs_digest(),
                    memory_snapshot_hash=(self.active_snapshot_ref or {}).get('hash'),
                    context=dict(self.context), memory_context=self._memory_context(), current_program_hash=parent.hash,
                    candidate_program_hash=trial.get('candidate_hash') if trial else None,
                    behavior_comparison=observed_behavior_comparison(before, after),
                    evidence=evidence,
                    instruction_boundary='Only permitted dataset history and current exploration; no held-out outcomes; model analysis is not reward or causal proof')

    def _logs_digest(self):
        # Generated programs can observe JSON object insertion order.
        return digest(json.dumps(self.logs, ensure_ascii=False, sort_keys=False))

    def _reuse_review(self, receipt, parent, before):
        if not receipt or not receipt.get('ok'):
            return False
        self._check_review_receipt(receipt)
        original = read(self.store.root / receipt['input_path'])
        if original.get('ordered_logs_hash') != self._logs_digest():
            return False
        # A report about a rejected candidate is not a diagnosis of its parent.
        # Reuse only when the assessed implementation is now the deployed one.
        subject = receipt['report']['assessment']['subject']
        label, key = ('candidate', 'candidate_program_hash') if subject == 'candidate' else ('parent', 'current_program_hash')
        labels = [label] if original.get(key) == parent.hash else []
        return any(row['id'] == label + ':storage' and isinstance(row['data'], dict)
                   and row['data'].get('memory_hash') == before['memory_hash']
                   for label in labels for row in original['evidence'])

    def _activate_snapshot(self, reference):
        from .learning import load_snapshot, snapshot_logs
        self.active_snapshot_ref = reference
        self.active_snapshot = load_snapshot(self.store.root, reference)
        self.logs = snapshot_logs(self.active_snapshot)

    def _learn(self, parent, state, index):
        """Commit only completed deployed-policy learning, never candidate evaluations."""
        from .learning import (make_snapshot, save_snapshot, snapshot_memory, learning_key,
                               begin_learning, finish_learning, read_learning, learning_receipt_ref)
        self._activate_snapshot(state['task_memory'])
        references = []
        for task in self.learning_blocks[index]:
            phase = 'round-%03d/learning' % index
            key = learning_key(self.binding, parent.hash, self.active_snapshot_ref['hash'], task, phase)
            request = json.loads(dumps(dict(binding_hash=digest(self.binding), program_hash=parent.hash,
                           snapshot_hash=self.active_snapshot_ref['hash'], input_snapshot_ref=self.active_snapshot_ref,
                           task=task, phase=phase)))
            cached = read_learning(self.store.root, key, request)
            if cached is None:
                begin_learning(self.store.root, key, request)
                episode = None
                try:
                    result = self.evaluator.learn(parent, self.logs, task, phase,
                                                   memory_snapshot=snapshot_memory(self.active_snapshot))
                    episode = result['episode']
                    if not self._complete([episode], [task], learning=True):
                        raise RuntimeError('Incomplete native learning episode; its temporary memory is not committed')
                    runtime_snapshot = result['memory_snapshot']
                    memory = CodeMemory.from_snapshot(parent, runtime_snapshot, self.tokenizer,
                                                       capabilities=self.capabilities)
                    if episode.get('memory_hash_after') != memory.fingerprint:
                        raise RuntimeError('Learning episode and actual written bank disagree')
                    snapshot = make_snapshot(self.logs + [episode], self.context['environment'],
                                             memory_snapshot=runtime_snapshot,
                                             parent_hash=self.active_snapshot_ref['hash'])
                    output_ref = save_snapshot(self.store.root, snapshot)
                    finish_learning(self.store.root, key, request, episode, output_ref)
                except Exception as exc:
                    failure = episode
                    if not isinstance(failure, dict) or failure.get('task_id') != task['id']:
                        failure = dict(task_id=task['id'], synthetic=self.synthetic)
                    failure = dict(failure, error=type(exc).__name__ + ': ' + str(exc))
                    finish_learning(self.store.root, key, request, failure, None)
                    raise
                cached = read_learning(self.store.root, key, request)
            if cached['status'] != 'complete':
                raise RuntimeError('Previously failed learning episode; inspect its preserved receipt')
            references.append(learning_receipt_ref(cached))
            self._activate_snapshot(cached['output_snapshot_ref'])
        return references

    def _review(self, payload, index, directory):
        from phase1_codegen.roles.reviewer import validate_report
        stage = payload['stage']
        path = directory / (stage + '_review.json')
        if path.exists():
            receipt = read(path)
            if receipt['payload_hash'] != digest(payload) or receipt['reviewer_fingerprint'] != self.reviewer.fingerprint:
                raise RuntimeError('Review inputs or reviewer changed during resume')
            self._check_review_receipt(receipt)
            return receipt
        input_path = directory / (stage + '_review_input.json')
        shown_path = directory / (stage + '_review_displayed.json')
        write(input_path, payload)
        started = time.monotonic()
        report, error, shown = None, None, payload
        try:
            report = self.reviewer(payload, self.seed + 20000 + 2 * index + int(stage == 'post'))
            shown = getattr(self.reviewer, 'last_payload', None) or payload
            validate_report(report, shown)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            report = None
            shown = getattr(self.reviewer, 'last_payload', None) or shown
        write(shown_path, shown)
        receipt = dict(stage=stage, ok=error is None, error=error, report=report,
                       report_hash=digest(report), payload_hash=digest(payload), displayed_hash=digest(shown),
                       input_path=str(input_path.relative_to(self.store.root)),
                       displayed_path=str(shown_path.relative_to(self.store.root)),
                       receipt_path=str(path.relative_to(self.store.root)),
                       reviewer_fingerprint=self.reviewer.fingerprint, synthetic=self.synthetic,
                       seconds=time.monotonic() - started)
        write(path, receipt)
        return receipt

    def _check_review_receipt(self, receipt):
        from phase1_codegen.roles.reviewer import validate_report
        if digest(receipt.get('report')) != receipt['report_hash']:
            raise RuntimeError('Review report changed')
        original = read(self.store.root / receipt['input_path'])
        shown = read(self.store.root / receipt['displayed_path'])
        if digest(original) != receipt['payload_hash'] or digest(shown) != receipt['displayed_hash']:
            raise RuntimeError('Review evidence changed')
        if receipt['ok']:
            validate_report(receipt['report'], shown)

    def _review_for_proposer(self, receipt, current_hash):
        if not receipt or not receipt.get('ok'):
            return None
        self._check_review_receipt(receipt)
        shown = read(self.store.root / receipt['displayed_path'])
        original = read(self.store.root / receipt['input_path'])
        ids = {eid for finding in receipt['report']['findings'] for eid in finding['evidence_ids']}
        ids.update(receipt['report']['assessment']['evidence_ids'])
        # Contract 2 already carries the exact cited values and their provenance.
        # Repeating entire evidence rows can hide the report behind a clipped
        # prefix when it is delivered to the Evolver.
        contract = receipt['report'].get('evidence_contract')
        return dict(current_program_hash=current_hash, source_stage=receipt['stage'], report=receipt['report'],
                    observation_context=review_observation_context(shown, receipt['report']) if contract else [],
                    reviewed_programs=dict(parent_hash=original['current_program_hash'],
                                           candidate_hash=original.get('candidate_program_hash')),
                    report_hash=receipt['report_hash'],
                    evidence_refs=[] if contract else [row for row in shown['evidence'] if row['id'] in ids and row['kind'] != 'program'],
                    scope='Assessment of existing mechanism evidence only; the Evolver alone decides changes. Not independently verified causal conclusions.')

    def _execution_facts(self, behavior, *, round_index, historical):
        """Recorded execution facts, independent of Reviewer citation selection."""
        if not behavior or not behavior.get('diagnostics'):
            return None
        from .behavior_diagnostics import bounded_execution_facts
        scope = dict(round=round_index,
                     evidence_time='historical_previous_attempt' if historical else 'current_parent',
                     observation_source='recorded_execution_diagnostics')
        boundary = ('Observed execution facts, not Reviewer opinions or a defect/adoption judgment. '
                    'Historical facts describe their original program and snapshot; they are not current measurements. ')
        allowance = max(0, 2000 - self.tokenizer.count(dumps(scope) + boundary) - 32)
        diagnostics = bounded_execution_facts(behavior['diagnostics'], self.tokenizer, token_budget=allowance)
        if diagnostics is None:
            return dict(available=False, original_hash=digest(behavior['diagnostics']),
                        provenance=behavior['diagnostics'].get('provenance'), round=round_index,
                        evidence_time='historical_previous_attempt' if historical else 'current_parent',
                        reason='Execution diagnostic metadata exceeds the fixed 2000-token evidence allowance')
        diagnostics = json.loads(dumps(diagnostics))
        diagnostics['provenance'].update(scope)
        diagnostics['interpretation_boundary'] = boundary + diagnostics.get('interpretation_boundary', '')
        if self.tokenizer.count(dumps(diagnostics)) > 2000:
            raise ValueError('Complete execution facts and provenance exceed the fixed evidence budget')
        return diagnostics

    def _previous_attempt(self, history):
        """Recover the last committed proposal, including source rejected by the parser.

        The deployed parent remains authoritative. This is iteration feedback,
        not inherited candidate state or another retrieval/memory mechanism.
        """
        if not history:
            return None
        trial = history[-1]
        if trial.get('public_context', {}).get('environment') != self.context.get('environment'):
            raise RuntimeError('Previous proposal belongs to a different dataset environment')
        directory = self.store.root / 'trials' / ('%03d' % trial['round'])
        source, payload_hash, previous_input, review_input, review_shown = None, None, {}, {}, {}
        source_binding = 'committed_raw_source_hash'
        current_receipt = 'raw_source_hash' in trial
        proposal_path = directory / 'proposal.json'
        if proposal_path.exists():
            proposal = read(proposal_path)
            previous_input = read(directory / 'proposer_input.json')
            payload_hash = digest(previous_input)
            if (proposal.get('payload_hash') != payload_hash
                    or (current_receipt and trial.get('proposal_payload_hash') != payload_hash)):
                raise RuntimeError('Previous proposal input binding changed')
            value = proposal.get('value')
            source = value.get('source') if type(value) is dict else None
            source = source if type(source) is str else None
        raw_hash = hashlib.sha256(source.encode()).hexdigest() if source is not None else None
        if current_receipt and raw_hash != trial.get('raw_source_hash'):
            raise RuntimeError('Previous raw proposal source changed')
        if not current_receipt and source is not None:
            # Frozen older receipts did not store a raw hash. Their committed
            # post-review input already binds the entire candidate, even if
            # parsing failed. Require that independent existing anchor; never
            # rewrite old trials or trust an unbound draft during replay.
            receipt = trial.get('post_review')
            if not receipt:
                raise RuntimeError('Historical draft lacks a committed source evidence anchor')
            self._check_review_receipt(receipt)
            review_input = read(self.store.root / receipt['input_path'])
            sources = [row['data'] for row in review_input['evidence']
                       if row['id'] == 'candidate:program' and row['kind'] == 'program']
            if sources != [source]:
                raise RuntimeError('Historical proposal differs from its committed review source')
            source_binding = 'committed_post_review_input'
        parsed_hash = trial.get('candidate_hash')
        if parsed_hash is not None and (parsed_hash != raw_hash
                or self.store.program(parsed_hash).source != source):
            raise RuntimeError('Previous parsed program and raw proposal differ')
        review = self._review_for_proposer(trial.get('post_review'), trial['parent_hash'])
        if trial.get('post_review'):
            self._check_review_receipt(trial['post_review'])
            review_input = read(self.store.root / trial['post_review']['input_path'])
            review_shown = read(self.store.root / trial['post_review']['displayed_path'])
        if review is not None:
            # Only exact verified observations and normalized assessments cross
            # into the author role. Generated review narration remains archived.
            report = review['report']
            review = {key: val for key, val in review.items()
                      if key not in ('report', 'current_program_hash', 'evidence_refs')}
            review['report'] = report.get('evidence_contract')
            if review['report'] is None:
                review = None
        behavior = None
        behavior_comparison = None
        if trial.get('behavior_path'):
            saved = read(self.store.root / trial['behavior_path'])
            if digest(dict(parent=saved['parent'], candidate=saved['candidate'])) != trial['behavior_evidence']:
                raise RuntimeError('Previous execution evidence changed')
            behavior = saved['candidate']
            behavior_comparison = observed_behavior_comparison(saved['parent'], saved['candidate'])
        from phase1_codegen.roles.reviewer import historical_native_execution
        native_execution = historical_native_execution(review_input.get('evidence', []),
                                                      review_shown.get('evidence', []), self.tokenizer)
        snapshot = trial.get('task_memory_evaluation') or {}
        feedback = trial.get('acceptance_feedback')
        if feedback and 'native_comparison' in feedback:
            feedback = dict(feedback, native_comparison=comparison_for_feedback(feedback['native_comparison']))
        return dict(round=trial['round'], environment=self.context.get('environment'),
                    parent_hash=trial['parent_hash'], raw_source_hash=raw_hash,
                    parsed_program_hash=parsed_hash, source=source,
                    hypothesis=trial.get('hypothesis'), risk=trial.get('risk'),
                    status=trial['public_status'], error=trial.get('error'),
                    mechanism_validated=trial['mechanism_validated'], accepted=trial['accepted'],
                    proposal_payload_hash=payload_hash, trial_hash=digest(trial),
                    provenance=dict(phase='historical_previous_attempt',
                                    source_binding=source_binding,
                                    proposal_binding='committed_trial_and_receipt' if current_receipt else 'historical_receipt_input_hash',
                                    memory_snapshot_hash=snapshot.get('hash'),
                                    ordered_logs_hash=review_input.get('ordered_logs_hash'),
                                    observed_context=trial['public_context']),
                    post_review=review, native_execution=native_execution,
                    behavior_comparison=behavior_comparison,
                    acceptance_feedback=feedback,
                    native_validation=comparison_for_feedback(trial.get('exploration')),
                    execution_facts=self._execution_facts(behavior, round_index=trial['round'], historical=True),
                    boundary='Historical draft and evidence only. current_source is the deployed parent; no candidate state is inherited.')

    def _proposal(self, payload, round_index, directory):
        path = directory / 'proposal.json'
        if path.exists():
            receipt = read(path)
            if receipt['payload_hash'] != digest(payload):
                raise RuntimeError('Proposal inputs changed during resume')
        else:
            write(directory / 'proposer_input.json', payload)
            started = time.monotonic()
            try:
                value = self.proposer(payload, self.seed + round_index)
                receipt = dict(payload_hash=digest(payload), value=value, error=None)
            except Exception as exc:
                receipt = dict(payload_hash=digest(payload), value=None, error=type(exc).__name__ + ': ' + str(exc))
            receipt['seconds'] = time.monotonic() - started
            write(path, receipt)
        if receipt['error']:
            if self.acceptance_policy == REVIEWER_DIRECTED:
                raise ProposalCallFailure(receipt['error'])
            raise ValueError(receipt['error'])
        value = receipt['value']
        if not isinstance(value, dict) or set(value) != {'source', 'hypothesis', 'risk'} or any(type(v) is not str for v in value.values()):
            raise Rejected('Expected exactly string fields source, hypothesis, risk')
        return value

    def _prepare_candidate(self, proposal, contexts, history):
        source, candidate = proposal['source'], None
        try:
            candidate = Program(source)
            try:
                self.store.save_program(candidate)
                visited = {self.initial.identity_hash: None}
                for prior in history:
                    if prior.get('candidate_hash'):
                        identity = self.store.program(prior['candidate_hash']).identity_hash
                        visited.setdefault(identity, prior)
            except Rejected as exc:
                raise MeasurementFailure('Stored mechanism evidence could not be verified: '+str(exc)) from exc
            if candidate.identity_hash in visited:
                error = Rejected('Repeated or unchanged program AST')
                if self.acceptance_policy != REVIEWER_DIRECTED:
                    raise error
                prior = visited[candidate.identity_hash]
                reference = dict(ast_hash=candidate.identity_hash,
                                 source='initial_item_program', program_hash=self.initial.hash)
                if prior is not None:
                    comparison = prior.get('exploration') or {}
                    reference.update(source='earlier_draft_on_this_item', attempt=prior['round'] + 1,
                                     program_hash=prior['candidate_hash'], trial_hash=digest(prior),
                                     compared=comparison.get('complete') is True,
                                     candidate_score=comparison.get('candidate') if comparison.get('complete') else None)
                return dict(candidate=candidate, source=source, signature=None, error=error,
                            duplicate_of=reference)
            signature = self._signature(candidate, contexts)
            return dict(candidate=candidate, source=source, signature=signature, error=None)
        except (WorkerStartupError, CapabilityInfrastructureError):
            raise
        except Exception as exc:
            if self.acceptance_policy == REVIEWER_DIRECTED and not isinstance(exc, Rejected):
                raise
            return dict(candidate=candidate, source=source, signature=None, error=exc)

    def _cost_receipts(self, directory):
        return dict(proposal=str((directory / 'proposal.json').relative_to(self.store.root)),
                    measurements='measurement_access.jsonl', native='native/costs.jsonl',
                    reviews='reviewer_costs.jsonl')

    def run(self, rounds=None):
        with self.store.lock():
            state = self.store.initialize(self.initial, self.binding)
            if rounds is not None and (type(rounds) is not int or rounds < 0):
                raise ValueError('rounds must be a nonnegative cumulative total')
            adaptive = self.acceptance_policy == REVIEWER_DIRECTED
            if adaptive and rounds is not None:
                raise ValueError('The post Reviewer controls the end of same-item repair')
            stop = self.max_rounds if rounds is None else min(self.max_rounds, rounds)
            while stop is None or state['next_round'] < stop:
                history = self.store.history(state)
                if adaptive and history and history[-1]['acceptance_feedback']['decision'] != 'continue':
                    break
                index = state['next_round']
                self._round_index = index
                directory = self.store.root / 'trials' / ('%03d' % index)
                trial_path = directory / 'trial.json'
                if trial_path.exists():
                    state = self.store.commit(state, read(trial_path), trial_path)
                    continue
                parent = self.store.program(state['current_program'])
                self._runtime_snapshots = {}
                memory_before = state.get('task_memory')
                learning_receipts = self._learn(parent, state, index) if self.task_memory_mode == 'continual' else []
                base_episodes = self._measure(parent, self.explore, 'round-%03d/exploration/parent' % index)
                context = public_context(self.context, self.logs, base_episodes)
                trial = dict(round=index, parent_hash=parent.hash, accepted=False, synthetic=self.synthetic,
                             public_context=context, public_status='invalid',
                             acceptance_policy=self.acceptance_policy, acceptance_basis=None,
                             mechanism_validated=False,
                             confirmation_block=index if self.acceptance_policy == 'native_reward' else None)
                if self.task_memory_mode == 'continual':
                    trial.update(task_memory_before=memory_before,
                                 task_memory_evaluation=self.active_snapshot_ref,
                                 learning_receipts=learning_receipts)
                candidate, source, before, after = None, None, None, None
                candidate_episodes = []
                if not self._complete(base_episodes, self.explore):
                    raise RuntimeError('Current mechanism failed native exploration; inspect receipts before proceeding')
                previous = [past.get('evolution_event') for past in history]
                # Online item drafts are staged until the item commits. They
                # reach the author through the item history and direct handoff.
                local_records = [] if adaptive else [record for record in previous if record]
                all_records = EvolutionMemory(self.memory.records + local_records)
                contexts = [e.get('native_recorder', {}).get('initial') for e in base_episodes]
                if self.reviewer is not None:
                    before = self._signature(parent, contexts)
                    prior = history[-1].get('post_review') if history else None
                    if (adaptive and history) or self._reuse_review(prior, parent, before):
                        # In a continuing item the previous candidate review guides its next edit.
                        # It keeps its original candidate, round and evidence bindings.
                        self._check_review_receipt(prior)
                        trial['pre_review'] = prior
                    else:
                        review_input = self._review_input('pre', parent, before, base_episodes)
                        trial['pre_review'] = self._review(review_input, index, directory)
                    if not trial['pre_review']['ok']:
                        trial['review_error'] = trial['pre_review']['error']
                        trial['public_status'] = 'pre_review_failed'
                # Current native observations and the independent pre-review
                # precede retrieval. Historical design evidence never enters
                # the Reviewer, and the complete previous attempt is not sent
                # twice through a competing memory slot.
                parent_review = history[0]['pre_review'] if adaptive and history else trial.get('pre_review')
                query = mechanism_query(context, parent.source, before, parent_review,
                                        previous[-1] if previous else None)
                selected, retrieval = all_records.select(query, self.tokenizer,
                    token_budget=RETRIEVAL_TOKENS,
                    exclude_ids=[previous[-1]['id']] if local_records and previous[-1] else [])
                source_ids = {record['id'] for record in self.memory.records}
                retrieval.update(source_records_available=len(self.memory.records),
                                 local_records_available=len(local_records),
                                 selected_ids=[record['id'] for record in selected],
                                 selected_source_ids=[record['id'] for record in selected if record['id'] in source_ids],
                                 selected_local_ids=[record['id'] for record in selected if record['id'] not in source_ids],
                                 evidence_scope='Selection is not proof that the generated edit used an experience')
                if adaptive:
                    retrieval['staged_item_records'] = len(previous)
                write(directory / 'retrieval.json', retrieval)
                trial['evolution_memory_retrieval'] = retrieval
                payload = dict(current_source=parent.source, context=self.context, observed_context=context,
                               deployment_policy=dict(name=self.acceptance_policy,
                                                      rule=POLICIES[self.acceptance_policy]),
                               memory_context=self._memory_context(),
                               own_adaptation=[own_trace(e) for e in self.logs],
                               own_exploration=[own_trace(e) for e in base_episodes],
                               evolution_memory=selected,
                               own_trial_history=[dict(parent_hash=t['parent_hash'], candidate_hash=t.get('candidate_hash'),
                                                       hypothesis=t.get('hypothesis'), status=t['public_status'],
                                                       exploration=comparison_for_feedback(t.get('exploration')), error=t.get('error')) for t in history],
                               previous_attempt=self._previous_attempt(history),
                               current_execution_facts=self._execution_facts(before, round_index=index, historical=False),
                               interface={},
                               limits=dict(storage_tokens=CAPACITY_TOKENS, read_tokens=READ_TOKENS, top_k=READ_TOP_K),
                               mechanism_review=(
                                   dict(source='previous_attempt.post_review', report_hash=prior['report_hash'])
                                   if adaptive and history else self._review_for_proposer(trial.get('pre_review'), parent.hash)))
                if self.capabilities is not None:
                    payload['available_models'] = self.capabilities.describe()
                from dataclasses import asdict
                payload['interface'] = dict(
                    on_event='on_event(state, event, context) -> {state: dict, session: dict, memory: null|list[{id: str, experience: str}]}',
                    events=list(EVENTS), event_contract=EVENT_CONTRACT)
                payload['limits']['hook'] = asdict(HookLimits.for_environment(self.context.get('environment')))
                try:
                    if trial.get('review_error'):
                        raise ValueError('Initial mechanism review failed; no proposal requested')
                    proposal = self._proposal(payload, index, directory)
                    source = proposal['source']
                    trial.update(hypothesis=proposal['hypothesis'], risk=proposal['risk'])
                    if any(not isinstance(c, dict) or not c for c in contexts):
                        raise ValueError('Native initial contexts required for behavioral validation')
                    prepared = self._prepare_candidate(proposal, contexts, history)
                    candidate, source = prepared['candidate'], prepared['source']
                    trial['candidate_hash'] = candidate.hash if candidate is not None else None
                    if 'duplicate_of' in prepared:
                        trial['duplicate_of'] = prepared['duplicate_of']
                    if prepared['error'] is not None:
                        raise prepared['error']
                    before = before if before is not None else self._signature(parent, contexts)
                    after = prepared['signature']
                    write(directory / 'behavior.json', dict(parent=before, candidate=after, synthetic=self.synthetic))
                    trial['behavior_evidence'] = digest(dict(parent=before, candidate=after))
                    trial['behavior_path'] = str((directory / 'behavior.json').relative_to(self.store.root))
                    trial['mechanism_validated'] = True
                    # A task_start slot cannot characterize later event responses.
                    candidate_episodes = self._measure(candidate, self.explore, 'round-%03d/exploration/candidate' % index,
                        memory_snapshot=after['runtime_snapshot'] if self.task_memory_mode == 'continual' else None)
                    trial['exploration'] = self._pair(parent, candidate, self.explore,
                                                    'round-%03d/exploration' % index, base_episodes, candidate_episodes)
                    exploration = trial['exploration']
                    if not exploration['complete']:
                        trial['public_status'] = 'exploration_incomplete'
                    elif self.acceptance_policy in DUAL_POLICIES:
                        trial['public_status'] = 'awaiting_reviewer_assessment'
                    elif exploration['delta'] < 0:
                        trial['public_status'] = 'exploration_rejected'
                    else:
                        trial['public_status'] = 'awaiting_native_confirmation'
                except MeasurementFailure:
                    # A host receipt already binds the failed measurement to this candidate.
                    raise
                except (WorkerStartupError, CapabilityInfrastructureError) as exc:
                    write(directory / 'infrastructure_failure.json', dict(
                        round=index, error=type(exc).__name__ + ': ' + str(exc),
                        infrastructure_details=getattr(exc, 'details', {}),
                        candidate_hash=trial.get('candidate_hash'),
                        generated_execution_started=False if isinstance(exc, WorkerStartupError) else None,
                        task_memory_snapshot=self.active_snapshot_ref))
                    raise
                except Exception as exc:
                    if adaptive and not isinstance(exc, Rejected):
                        raise
                    trial['error'] = type(exc).__name__ + ': ' + str(exc)
                    if not trial.get('review_error'):
                        trial['public_status'] = 'invalid'
                proposal_path = directory / 'proposal.json'
                if proposal_path.exists():
                    proposal_receipt = read(proposal_path)
                    trial['proposal_payload_hash'] = proposal_receipt['payload_hash']
                    # Even a malformed proposal envelope can contain a raw
                    # draft. Keep its identity without calling it valid code.
                    raw_value = proposal_receipt.get('value')
                    if source is None and type(raw_value) is dict and type(raw_value.get('source')) is str:
                        source = raw_value['source']
                trial['raw_source_hash'] = hashlib.sha256(source.encode()).hexdigest() if source is not None else None
                if self.reviewer is not None and not trial.get('review_error'):
                    # Review only exploration evidence. Independent confirmation is still sealed.
                    review_input = self._review_input('post', parent, before, base_episodes,
                                                      trial, source, after, candidate_episodes)
                    trial['post_review'] = self._review(review_input, index, directory)
                    if not trial['post_review']['ok']:
                        trial['review_error'] = trial['post_review']['error']
                        trial['public_status'] = 'post_review_failed'
                if self.acceptance_policy == 'native_reward' and trial['public_status'] == 'awaiting_native_confirmation':
                    trial['confirmation'] = self._pair(parent, candidate, self.blocks[index],
                                                      'round-%03d/confirmation' % index)
                trial['acceptance_feedback'] = adoption_feedback(self.binding, trial,
                    current_facts=self.store.adoption_context(self.binding, trial))
                trial['accepted'] = trial['acceptance_feedback']['eligible']
                if trial['accepted']:
                    trial['acceptance_basis'] = self.acceptance_policy
                    trial['public_status'] = 'accepted'
                elif trial['public_status'] in {'awaiting_native_confirmation', 'awaiting_reviewer_assessment'}:
                    trial['public_status'] = 'not_accepted'
                trial['cost_receipts'] = self._cost_receipts(directory)
                trial['evolution_event'] = public_event(trial, parent.source, source,
                    tokenizer=self.tokenizer, behavior=after, parent_behavior=before,
                    private_texts=private_learning_texts(self.logs))
                if self.task_memory_mode == 'continual':
                    from .learning import make_snapshot, save_snapshot
                    if trial['accepted']:
                        deployed = make_snapshot(self.logs, self.context['environment'],
                                                 memory_snapshot=after['runtime_snapshot'],
                                                 parent_hash=self.active_snapshot_ref['hash'])
                        trial['task_memory_after'] = save_snapshot(self.store.root, deployed)
                    else:
                        trial['task_memory_after'] = self.active_snapshot_ref
                write(trial_path, trial)
                state = self.store.commit(state, trial, trial_path)
                if trial.get('review_error'):
                    break
            if not adaptive:
                write(self.store.root / 'evolution_memory.json',
                      self.memory.records + [t['evolution_event'] for t in self.store.history(state)])
            return state
