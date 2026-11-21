"""Fixed event transport; generated memory owns all event responses.

No event handler is supplied to the Evolver as an algorithm example. Session
state is scratch space in the same memory mechanism, not another memory layer.
"""
from __future__ import annotations

from dataclasses import asdict

from .memory import bound_readout, canonical, digest
from .policy import GeneratedBudgetExceeded, Rejected, archive_copy, json_copy
from .program import WorkerStartupError
from .protocol import EVENTS, HOOK_VERSION as VERSION, HookLimits, WRITE_FIELDS, HOOK_BUDGET_COUNTERS


class HookSession:
    """One staged episode transaction. Only explicit validated publish commits."""
    def __init__(self, memory, task, allow_write=False, *, limits=None, origin='native'):
        self.memory = getattr(memory, '_memory', memory)
        self.context = memory.context
        self.task = json_copy(task)
        environment = self.memory.environment or self.context.get('environment')
        if task.get('environment', environment) != environment:
            raise Rejected('Hook task belongs to a different environment')
        self.context.update(environment=environment, task_id=task['id'], family=task['family'])
        self.allow_write = bool(allow_write)
        self.limits = limits or HookLimits.for_environment(environment)
        if origin not in ('native', 'history_import'):
            raise Rejected('unknown memory event origin')
        self.origin = origin
        self.state = self.memory.state
        self.session = {}
        self.current_memory = {}
        self.trace = []
        self.finished = False
        self.failed = False
        self.closed = False
        self.committed = False
        self.native = None
        self.before = self.memory.fingerprint
        self.usage = dict(model_calls=0, worker_cpu_seconds=0., generated_cpu_seconds=0.,
                          worker_wall_seconds=0., injected_tokens=0)

    def emit(self, kind, payload):
        if self.closed or self.finished or self.failed or kind not in EVENTS:
            raise Rejected('invalid Hook event lifecycle')
        initial = 'task_end' if self.origin == 'history_import' else 'task_start'
        if (not self.trace and kind != initial) or (self.trace and kind == 'task_start'):
            raise Rejected('Hook must start exactly once')
        if self.origin == 'history_import' and (self.trace or kind != 'task_end'):
            raise Rejected('history import contains one terminal experience only')
        if len(self.trace) >= self.limits.max_events:
            raise Rejected('Hook event budget exceeded')
        payload = json_copy(payload)
        if type(payload) is not dict or set(payload) & {'type', 'session'}:
            raise Rejected('Hook payload cannot override event type/session')
        event = dict(payload, type=kind, session=self.session)
        if self.origin == 'history_import':
            event.update(mode='history_import', origin='history_import')
        event['budget_remaining'] = {
            key: getattr(self.limits, key) - self.usage[key] for key in HOOK_BUDGET_COUNTERS}
        before = digest(self.state)
        row = dict(version=VERSION, sequence=len(self.trace),
                   event=json_copy({key: value for key, value in event.items() if key not in {'session', 'budget_remaining'}}),
                   session_hash_before=digest(self.session), input_hash=digest(event),
                   program_hash=self.memory.program.hash, state_hash_before=before,
                   allow_commit=self.allow_write, status='pending')
        self.trace.append(row)
        owner = self
        class Capabilities:
            def call(self, model_id, data, *, operation):
                if owner.usage['model_calls'] >= owner.limits.model_calls:
                    raise GeneratedBudgetExceeded('cumulative Hook model-call budget exceeded')
                owner.usage['model_calls'] += 1
                if owner.memory.capabilities is None:
                    raise Rejected('no local model capability broker configured')
                return owner.memory.capabilities.call(model_id, data, operation=operation)
        try:
            receipt = self.memory.program.run('event', self.state, event, self.context,
                capabilities=Capabilities(), diagnostics=self.memory.diagnostics)
            value = receipt['value']
            self.usage['worker_cpu_seconds'] += receipt['isolation']['worker_cpu_seconds']
            self.usage['generated_cpu_seconds'] += receipt['isolation']['generated_cpu_seconds']
            self.usage['worker_wall_seconds'] += receipt['wall_seconds']
            for key in ('generated_cpu_seconds', 'worker_wall_seconds'):
                if self.usage[key] > getattr(self.limits, key):
                    raise GeneratedBudgetExceeded('cumulative Hook ' + key + ' budget exceeded')
            if self.memory.tokenizer.count(canonical(value['state'])) > self.memory.capacity:
                raise GeneratedBudgetExceeded('Hook persistent state storage budget exceeded')
            if self.memory.tokenizer.count(canonical(value['session'])) > self.limits.session_tokens:
                raise GeneratedBudgetExceeded('Hook session storage budget exceeded')
            response = None
            # The end event can update storage; it has no following Actor call.
            if value['memory'] is not None and kind != 'task_end':
                proposed, tokens = bound_readout(value['memory'], self.memory.tokenizer,
                    top_k=self.memory.top_k, token_budget=self.memory.read_budget)
                tokens = tokens if proposed else 0
                if proposed != self.current_memory:
                    if self.usage['injected_tokens'] + tokens > self.limits.injected_tokens:
                        raise GeneratedBudgetExceeded('cumulative Hook memory injection budget exceeded')
                    self.usage['injected_tokens'] += tokens
                response = proposed
            self.state, self.session = value['state'], value['session']
            if response is not None:
                self.current_memory = response
            row.update(status='valid', state_hash_after=digest(self.state),
                       response=json_copy(value['memory']), actual_memory=json_copy(self.current_memory),
                       session_hash=digest(self.session),
                       receipt={key: val for key, val in receipt.items() if key != 'value'},
                       cumulative_cost=json_copy(self.usage))
            self.memory.stats['worker_seconds'] += receipt['wall_seconds']
            self.memory.stats['diagnostic_seconds'] += receipt.get('diagnostics', {}).get('wall_seconds', 0.)
            return response
        except Exception as exc:
            row.update(status='rejected', error_type=type(exc).__name__, error=str(exc))
            if isinstance(exc, WorkerStartupError):
                row['infrastructure_details'] = json_copy(exc.details)
            self.failed = True
            raise

    def finish(self, experience):
        if self.finished:
            raise Rejected('Hook task_end was already processed')
        experience = archive_copy(experience)
        if experience.get('error'):
            return self.abort(experience['error'])
        # Deliberately omit diagnostic/full-recorder/model-response fields.
        native = {key: experience[key] for key in WRITE_FIELDS if key in experience}
        for key, expected in (('task_id', self.task['id']), ('family', self.task['family']),
                              ('environment', self.context['environment'])):
            if native.get(key) != expected:
                raise Rejected('terminal experience does not match Hook task provenance')
        self.emit('task_end', dict(experience=native, step=len(experience.get('history', []))))
        self.finished = True
        self.native = native
        if self.memory.fingerprint != self.before:
            raise Rejected('committed memory changed during Hook transaction')
        return self.summary()

    def publish(self):
        """Called once by the trusted validator after the entire episode passes."""
        if not self.allow_write or not self.finished or self.failed or self.closed or self.committed:
            raise Rejected('Hook transaction is not publishable')
        self.memory.commit_state(self.state, self.native, self.before, dict(
            hook_version=VERSION, origin=self.origin, context=self.context, event_trace=self.trace))
        self.committed = True
        self.close()

    def close(self):
        """Discard scratch state; diagnostic partial reads need no fake task_end."""
        self.closed = True
        self.state = {}
        self.session = {}

    def abort(self, error, payload=None):
        # Error termination is a trusted receipt, never a fabricated successful
        # task_end or another invocation of already-failing generated code.
        if self.committed:
            raise Rejected('cannot abort an already validated published transaction')
        if self.closed and self.failed:
            return self.summary()
        self.failed = True
        self.finished = True
        self.trace.append(dict(version=VERSION, status='aborted', error=str(error),
                               payload=json_copy(payload or {}), committed=False))
        self.close()
        return self.summary()

    def summary(self):
        return dict(version=VERSION, complete=self.finished and not self.failed,
                    committed=self.committed, closed=self.closed, origin=self.origin,
                    coverage='terminal_history_import' if self.origin == 'history_import' else 'dynamic_events',
                    limits=asdict(self.limits), usage=json_copy(self.usage),
                    memory_hash_before=self.before, memory_hash_after=self.memory.fingerprint,
                    events=archive_copy(self.trace))


def replay_episode(memory, experience):
    """New hook traces replay exact visible events; older logs use end ingestion.

    Raw histories without complete event streams use explicit terminal-only
    history import; they are never presented as dynamic interaction replay.
    """
    experience = archive_copy(experience)
    recorded = experience.get('memory_hook')
    if recorded is None:
        return None
    if (type(recorded) is not dict or recorded.get('version') != VERSION
            or recorded.get('complete') is not True):
        raise Rejected('cannot rebuild from incomplete Hook event history')
    # Validate the entire external history before any generated execution or
    # model request. A terminal event commits, so it must be unique and last.
    rows = recorded.get('events')
    if type(rows) is not list or len(rows) < 2:
        raise Rejected('Hook history requires a complete start/end sequence')
    events = []
    for row in rows:
        if type(row) is not dict or row.get('status') != 'valid':
            raise Rejected('Hook history contains an invalid event')
        event = row.get('event')
        if type(event) is not dict or event.get('type') not in EVENTS:
            raise Rejected('Hook history contains an unknown event')
        events.append(event)
    kinds = [event['type'] for event in events]
    if (kinds[0] != 'task_start' or kinds[-1] != 'task_end'
            or kinds.count('task_start') != 1 or kinds.count('task_end') != 1):
        raise Rejected('Hook history requires exactly one start and one final task_end')
    for key in ('task_id', 'family', 'environment'):
        if type(experience.get(key)) is not str or not experience[key]:
            raise Rejected('Hook replay requires explicit task/family/environment provenance')
    native = {key: experience[key] for key in WRITE_FIELDS if key in experience}
    terminal = events[-1].get('experience')
    if (type(terminal) is not dict or digest(terminal) != digest(native)
            or native.get('error')):
        raise Rejected('Hook terminal experience disagrees with its outer native episode')
    task = dict(id=experience['task_id'], family=experience['family'], environment=experience['environment'])
    session = HookSession(memory, task, allow_write=True)
    try:
        for event in events:
            if event['type'] == 'task_end':
                session.finish(event['experience'])
            else:
                # Replay external observations only, never another program's scratch
                # state, readouts, model responses, or final answers.
                session.emit(event['type'], {key: value for key, value in event.items() if key not in {'type', 'session'}})
        if not session.finished:
            raise Rejected('Hook history lacks task_end')
        session.publish()
    except BaseException as exc:
        # A failed replay never reaches publication; discard its staged state.
        session.abort(str(exc), {'phase': 'event_replay'})
        raise
    return memory.state


def ingest_history(memory, experience):
    """One compatibility boundary for true event replay and older raw logs."""
    if experience.get('memory_hook') is not None:
        return replay_episode(memory, experience)
    task = dict(id=experience['task_id'], family=experience['family'], environment=experience['environment'])
    session = HookSession(memory, task, allow_write=True, origin='history_import')
    try:
        session.finish(experience)
        session.publish()
    except BaseException as exc:
        session.abort(str(exc), {'phase': 'history_import'})
        raise
    return memory.state
