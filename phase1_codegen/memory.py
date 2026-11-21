"""Transactional generated memory behind ALMA's unchanged retrieve interface."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time

from .policy import Rejected, archive_copy, json_copy, validate_state
from .program import Program
from .protocol import CAPACITY_TOKENS, HOOK_VERSION, READ_TOKENS, READ_TOP_K, WRITE_FIELDS


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bound_readout(rows, tokenizer, *, top_k=READ_TOP_K, token_budget=READ_TOKENS):
    """One deterministic adapter for the official pretty-JSON memory slot.

    Empty memory has no injected block and consumes zero injection tokens.
    """
    entries = json_copy(rows[:top_k])
    def payload():
        return {'entries': entries} if entries else {}
    def count():
        return tokenizer.count(json.dumps(payload(), ensure_ascii=False, indent=2)) if entries else 0
    while entries and count() > token_budget:
        last = entries[-1]
        size = tokenizer.count(last['experience'])
        clipped = tokenizer.clip(last['experience'], max(0, size - 32)) if size >= 48 else ''
        if size < 48 or tokenizer.count(clipped) >= size:
            entries.pop()
        else:
            last['experience'] = clipped
    return payload(), count()


@dataclass(frozen=True)
class CodeMechanism:
    program_hash: str
    api_version: str = HOOK_VERSION


class _TaskMemoryView:
    """One task's query metadata over the same committed memory bank."""
    def __init__(self, memory, context):
        self._memory = memory
        self._query_context = json_copy(context)

    def __getattr__(self, name):
        return getattr(self._memory, name)

    @property
    def context(self):
        query = self._memory.context
        query.update(self._query_context)
        return query

    def retrieve(self, goal, observation):
        return self._memory.retrieve(goal, observation, context=self._query_context)

    def read(self, context, *, visible=None):
        query = json_copy(context)
        query.update(self._query_context)
        return self._memory.read(query, visible=visible)


class CodeMemory:
    """The on_event program owns its state schema and all memory event behavior.

    A candidate always starts at {} and replays the same target adaptation logs.
    ``records`` is a diagnostic serialization of the whole state, not a claim
    that generated implementations share a common internal record schema.
    Failed writes never commit. Reads receive disposable snapshots and cannot
    alter the bank. The caller supplies the same fixed tokenizer as the Actor.
    """
    capacity = CAPACITY_TOKENS
    read_budget = READ_TOKENS
    top_k = READ_TOP_K

    def __init__(self, program, logs, tokenizer, context=None, *, capabilities=None, diagnostics=False):
        if not isinstance(program, Program):
            raise TypeError('CodeMemory requires a validated Program')
        started = time.monotonic()
        logs = archive_copy(logs)
        if type(logs) is not list or any(type(ep) is not dict for ep in logs):
            raise Rejected('logs must be a JSON list of experiences')
        self.program = program
        self.capabilities = capabilities
        if type(diagnostics) is not bool:
            raise Rejected('diagnostics must be a boolean')
        self.diagnostics = diagnostics
        self.mechanism = CodeMechanism(program.hash)
        self.tokenizer = tokenizer
        self._context = json_copy({} if context is None else context)
        if type(self._context) is not dict:
            raise Rejected('context must be a JSON object')
        self._context.setdefault('memory_hook_version', HOOK_VERSION)
        if self._context['memory_hook_version'] != HOOK_VERSION:
            raise Rejected('on_event requires an explicitly frozen memory_hook_version')
        if capabilities is not None:
            if self._context.get('environment') != capabilities.environment:
                raise Rejected('Model capability pool must match the Task Memory dataset')
            self._context['model_catalog'] = capabilities.describe()
        self.memory_scope = self._context.get('memory_scope', 'family')
        if self.memory_scope not in ('family', 'environment'):
            raise Rejected('memory_scope must be family or environment')
        self.environment = self._context.get('environment')
        if self.memory_scope == 'environment' and (type(self.environment) is not str or not self.environment):
            raise Rejected('environment-scoped Task Memory requires an explicit environment')
        families = {ep.get('family') for ep in logs if ep.get('family') is not None}
        if self.memory_scope == 'family' and len(families) > 1:
            raise Rejected('Task Memory cannot mix task families')
        self.family = next(iter(families), None) if self.memory_scope == 'family' else None
        self.families = set(families)
        self.source_log_hash = digest(logs)
        self._state = {}
        self.read_trace = []
        self.write_trace = []
        self.stats = dict(write_calls=0, skipped_error_episodes=0, rejected_writes=0,
                          stored_tokens=tokenizer.count(canonical({})), build_seconds=0.0,
                          worker_seconds=0.0, diagnostic_seconds=0.0)
        for ep in logs:
            self.write(ep)
        self.stats['build_seconds'] = time.monotonic() - started

    @property
    def state(self):
        return json_copy(self._state)

    def commit_state(self, state, experience, expected_hash, evidence):
        """Trusted atomic publication; all checks/copies precede bank mutation."""
        if self.fingerprint != expected_hash:
            raise Rejected('committed memory changed during transaction')
        state = validate_state(state)
        native = json_copy({key: experience[key] for key in WRITE_FIELDS if key in experience})
        family = native.get('family')
        if self.memory_scope == 'environment':
            if native.get('environment') != self.environment or type(family) is not str or not family:
                raise Rejected('committed experience must match its environment and task family')
        elif self.family is not None and family is not None and family != self.family:
            raise Rejected('Task Memory cannot cross task families')
        if native.get('error'):
            raise Rejected('cannot commit a failed native episode')
        tokens = self.tokenizer.count(canonical(state))
        if tokens > self.capacity:
            raise Rejected('generated state exceeds fixed 16384-token storage budget')
        trace = archive_copy(evidence)
        trace.update(status='committed', task_id=native.get('task_id'),
            experience_hash=digest(native), parent_memory_hash=expected_hash, stored_tokens=tokens)
        # Compute the post-state identity without publishing a transient bank.
        trace['memory_hash'] = self._state_fingerprint(state)
        families = set(self.families)
        if family is not None:
            families.add(family)
        self._state = state
        self.families = families
        if self.memory_scope == 'family' and self.family is None:
            self.family = family
        self.stats['stored_tokens'] = tokens
        self.stats['write_calls'] += 1
        self.write_trace.append(trace)

    @property
    def context(self):
        return json_copy(self._context)

    @property
    def records(self):
        if not self._state:
            return []
        text = canonical(self._state)
        return [dict(id=digest(self._state)[:20], text=text, tokens=self.tokenizer.count(text),
                     kind='whole_generated_state_snapshot')]

    @property
    def fingerprint(self):
        return self._state_fingerprint(self._state)

    def _state_fingerprint(self, state):
        # A generated algorithm may intentionally traverse dictionary insertion
        # order. Include the exact JSON state bytes as well as its schema/value
        # content, so behaviorally distinct orderings never alias in caches.
        wire = json.dumps(state, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        context_wire = json.dumps(self._context, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        return digest(dict(program_hash=self.program.hash, state_wire=wire, context_wire=context_wire,
                           capacity=self.capacity, read_budget=self.read_budget, top_k=self.top_k))

    def export_snapshot(self):
        """Bind actual committed state; restoration never replays model calls."""
        snapshot = dict(version='code-memory-state-v1', program_hash=self.program.hash,
                        program_identity_hash=self.program.identity_hash,
                        state_wire=json.dumps(self._state, ensure_ascii=False, allow_nan=False, separators=(',', ':')),
                        context_wire=json.dumps(self._context, ensure_ascii=False, allow_nan=False, separators=(',', ':')),
                        memory_hash=self.fingerprint, environment=self.environment,
                        memory_scope=self.memory_scope, family=self.family,
                        families=sorted(self.families), source_log_hash=self.source_log_hash,
                        capacity=self.capacity, read_budget=self.read_budget, top_k=self.top_k,
                        stored_tokens=self.tokenizer.count(canonical(self._state)))
        snapshot['snapshot_hash'] = digest(snapshot)
        return snapshot

    @classmethod
    def from_snapshot(cls, program, snapshot, tokenizer, context=None, *, capabilities=None, diagnostics=False):
        """Restore a verified same-program state with zero generated executions.

        The enclosing learning archive binds cumulative raw logs. This inner
        snapshot binds state, insertion order, context and fixed runtime budgets.
        Hashes detect corruption; callers must use their trusted archive chain.
        """
        snapshot = json_copy(snapshot)
        required = {'version', 'program_hash', 'program_identity_hash', 'state_wire', 'context_wire',
                    'memory_hash', 'environment', 'memory_scope', 'family', 'families', 'source_log_hash',
                    'capacity', 'read_budget', 'top_k', 'stored_tokens', 'snapshot_hash'}
        if type(snapshot) is not dict or set(snapshot) != required:
            raise Rejected('invalid committed memory snapshot schema')
        payload = {key: value for key, value in snapshot.items() if key != 'snapshot_hash'}
        if snapshot['version'] != 'code-memory-state-v1' or snapshot['snapshot_hash'] != digest(payload):
            raise Rejected('committed memory snapshot hash mismatch')
        if (not isinstance(program, Program) or snapshot['program_hash'] != program.hash
                or snapshot['program_identity_hash'] != program.identity_hash):
            raise Rejected('committed state can only restore the identical program')
        for key in ('capacity', 'read_budget', 'top_k'):
            if type(snapshot[key]) is not int or snapshot[key] != getattr(cls, key):
                raise Rejected('committed memory snapshot budget mismatch')
        for key in ('source_log_hash', 'memory_hash'):
            value = snapshot[key]
            if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise Rejected('invalid committed memory provenance hash')
        if type(snapshot['state_wire']) is not str or type(snapshot['context_wire']) is not str:
            raise Rejected('committed memory snapshot requires exact JSON wire strings')
        try:
            state = validate_state(json.loads(snapshot['state_wire']))
            restored_context = json.loads(snapshot['context_wire'])
        except (ValueError, TypeError) as exc:
            raise Rejected('invalid committed memory JSON') from exc
        if type(restored_context) is not dict:
            raise Rejected('committed memory context must be an object')
        # Reject alternate encodings and duplicate-key encodings, preserving the
        # precise dictionary iteration order committed by the generated code.
        wire = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if wire(state) != snapshot['state_wire'] or wire(restored_context) != snapshot['context_wire']:
            raise Rejected('committed memory wire encoding is not canonical insertion order')
        memory = cls(program, [], tokenizer, restored_context if context is None else context,
                     capabilities=capabilities, diagnostics=diagnostics)
        if wire(memory._context) != snapshot['context_wire']:
            raise Rejected('committed memory context mismatch')
        if snapshot['memory_scope'] != memory.memory_scope or snapshot['environment'] != memory.environment:
            raise Rejected('committed memory environment or scope mismatch')
        families = snapshot['families']
        if (type(families) is not list or any(type(item) is not str or not item for item in families)
                or families != sorted(set(families))):
            raise Rejected('invalid committed memory families')
        family = snapshot['family']
        if memory.memory_scope == 'environment':
            if family is not None:
                raise Rejected('environment-scoped snapshot cannot bind one task family')
        elif len(families) > 1 or family != next(iter(families), None):
            raise Rejected('invalid family-scoped snapshot')
        tokens = tokenizer.count(canonical(state))
        if type(snapshot['stored_tokens']) is not int or tokens != snapshot['stored_tokens'] or tokens > cls.capacity:
            raise Rejected('committed memory storage budget or tokenizer mismatch')
        memory._state = state
        memory.family = family
        memory.families = set(families)
        memory.source_log_hash = snapshot['source_log_hash']
        memory.stats['stored_tokens'] = tokens
        memory.stats['build_seconds'] = 0.0
        if memory.fingerprint != snapshot['memory_hash']:
            raise Rejected('committed memory state binding mismatch')
        return memory

    @classmethod
    def restore_at_endpoint(cls, program, snapshot, tokenizer, capabilities, source_manifest):
        """Import a committed bank when only its local generation service has moved."""
        from .capabilities import GENERATION_MODEL
        from .local_service import validate_base_url

        memory = cls.from_snapshot(program, snapshot, tokenizer)
        old_catalog = memory.context.get('model_catalog', {})
        if (memory.environment != capabilities.environment or
                old_catalog.get('fingerprint') != digest(source_manifest)):
            raise Rejected('source capability manifest does not match the committed catalog')
        projected_manifest = archive_copy(capabilities.manifest)
        projected_catalog = capabilities.describe()

        def generation_row(rows):
            matches = [row for row in rows if row['id'] == GENERATION_MODEL and row['kind'] == 'text_generation']
            if len(matches) != 1:
                raise Rejected('expected exactly one transport for the local generation model')
            return matches[0]

        source_url = validate_base_url(generation_row(source_manifest['models'])['endpoint'])
        target_url = validate_base_url(capabilities.base_url)
        for representation in (projected_manifest, projected_catalog):
            row = generation_row(representation['models'])
            if row['endpoint'] != target_url:
                raise Rejected('target capability route differs from the live broker')
            row['endpoint'] = source_url
        if projected_manifest != source_manifest:
            raise Rejected('model identity, contracts or capability settings changed during relocation')
        projected_catalog['fingerprint'] = digest(source_manifest)
        # Generated hooks can observe key order. Verify the old visible catalog
        # is byte-for-byte the new catalog with only the transport restored.
        wire = lambda item: json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if wire(projected_catalog) != wire(old_catalog):
            raise Rejected('committed capability catalog differs beyond the transport')
        memory._context['model_catalog'] = capabilities.describe()
        migrated = memory.export_snapshot()
        cls.from_snapshot(program, migrated, tokenizer, capabilities=capabilities)
        return migrated

    def write(self, experience):
        original = archive_copy(experience)
        if type(original) is not dict:
            raise Rejected('experience must be an object')
        family = original.get('family')
        if self.memory_scope == 'environment':
            if original.get('environment') != self.environment:
                raise Rejected('Task Memory experience must declare the same environment')
            if type(family) is not str or not family:
                raise Rejected('environment-scoped experience requires its current task family')
        elif self.family is not None and family is not None and family != self.family:
            raise Rejected('Task Memory cannot cross task families')
        if original.get('error'):
            self.stats['skipped_error_episodes'] += 1
            self.write_trace.append(dict(status='skipped_native_error', task_id=original.get('task_id')))
            return self.state
        from .hooks import ingest_history
        before = self.fingerprint
        try:
            return ingest_history(self, original)
        except Exception as exc:
            self.stats['write_calls'] += 1
            self.stats['rejected_writes'] += 1
            self.write_trace.append(dict(status='rejected', task_id=original.get('task_id'),
                                         parent_memory_hash=before, error=str(exc)))
            raise

    def read(self, context, *, visible=None):
        """Inspect task_start without changing the bank.

        A supplied native ``visible`` mapping is passed through exactly, even
        when empty. With no mapping, this internal diagnostic uses only explicit
        goal/observation query arguments; it does not reconstruct a native event.
        """
        context = json_copy(context)
        if type(context) is not dict:
            raise Rejected('read context must be a JSON object')
        native_visible = visible is not None
        if native_visible:
            visible = json_copy(visible)
            if type(visible) is not dict:
                raise Rejected('read visible input must be a JSON object')
        else:
            visible = {target: context[source] for source, target in
                       (('observation', 'obs'), ('goal', 'goal')) if source in context}
        if self.memory_scope == 'environment':
            if context.get('environment', self.environment) != self.environment:
                raise Rejected('Task Memory query cannot cross environments')
            context.update(environment=self.environment, memory_scope='environment')
        before = self.fingerprint
        from .hooks import HookSession
        task = dict(id=context.get('task_id', 'diagnostic'), family=context.get('family', self.family or 'unknown'),
                    environment=self.environment)
        session = HookSession(self, task)
        try:
            session.emit('task_start', dict(step=0, visible=visible))
            entries = session.current_memory.get('entries', [])
            receipt = session.trace[-1]['receipt']
            self.read_trace.append(dict(program_hash=self.program.hash, memory_hash=before,
                context=json_copy(context), runtime_context=json_copy(session.context),
                event_visible=json_copy(visible),
                input_origin='external_visible' if native_visible else 'explicit_query_diagnostic',
                raw_readout=session.trace[-1]['response'] or [],
                unpacked_readout=json_copy(entries), read_tokens=session.usage['injected_tokens'],
                readout_hash=digest(entries), capability_calls=receipt.get('capability_calls', []),
                isolation=receipt['isolation'], wall_seconds=receipt['wall_seconds'],
                hook_trace=archive_copy(session.trace),
                coverage='task_start only; dynamic interaction requires actual native rollout'))
            if self.diagnostics:
                self.read_trace[-1]['execution_trace'] = receipt.get('execution_trace')
                self.read_trace[-1]['diagnostics'] = receipt.get('diagnostics')
            return entries
        except BaseException as exc:
            session.abort(str(exc), {'phase': 'initial_read_diagnostic'})
            raise
        finally:
            session.close()

    def retrieve(self, goal, observation, context=None):
        query = self.context
        if context is not None:
            override = json_copy(context)
            if type(override) is not dict:
                raise Rejected('query metadata must be an object')
            query.update(override)
        query.update(goal=str(goal), observation=str(observation))
        return self.read(query)

    def for_task(self, task, environment):
        """Keep query family current without changing shared state or binding."""
        task = json_copy(task)
        if type(environment) is not str or not environment:
            raise Rejected('task view requires an explicit environment')
        if self.memory_scope == 'environment' and environment != self.environment:
            raise Rejected('Task Memory task view cannot cross environments')
        for key in ('environment', 'dataset'):
            if task.get(key) is not None and task[key] != environment:
                raise Rejected('task metadata declares a different environment')
        query = dict(environment=environment)
        for source, destination in (('family', 'family'), ('id', 'task_id')):
            if source in task:
                query[destination] = task[source]
        if self.memory_scope == 'environment' and (type(query.get('family')) is not str or not query['family']):
            raise Rejected('environment-scoped task view requires the current task family')
        return _TaskMemoryView(self, query)
