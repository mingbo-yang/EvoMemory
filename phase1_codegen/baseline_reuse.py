"""Explicit import of an unchanged source baseline's native measurement.

Old evidence is read-only. Only the new measurement cache and its import audit
are written; neither environment execution nor a model call occurs here.
"""
from pathlib import Path
import json
import math

from evomem.common import append, digest, events, file_hash, read, write
from .program import BASELINE_SOURCE, Program


PHASE = 'round-000/exploration/parent'
ORIGIN_FIELDS = {'schema', 'source_run', 'state_sha256', 'measurement_key',
                 'measurement_sha256', 'native_costs_sha256'}
# These fields cannot affect the already-completed initial native measurement.
NON_EXECUTION_BINDINGS = {'code', 'proposer', 'reviewer', 'review_required',
                          'review_schedule', 'acceptance', 'acceptance_policy', 'max_rounds'}
RUNTIME_FILES = tuple('phase1_codegen/' + name for name in (
    'native.py', 'textworld_native.py', 'textworld_tasks.py', 'memory.py',
    'program.py', 'policy.py', 'worker.py', 'tokenization.py', 'local_service.py',
    'capabilities.py', 'multimodal.py', 'multimodal_worker.py',
    'protocol.py', 'hooks.py', 'hook_adapter.py', 'isolation.py'))
RUNTIME_PREFIXES = ('evomem/', 'external/alma/')


def _require(condition, message):
    if not condition:
        raise RuntimeError('Source baseline reuse refused: ' + message)


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _execution_binding(binding):
    return {key: value for key, value in binding.items() if key not in NON_EXECUTION_BINDINGS}


def _runtime_code(binding):
    code = binding.get('code', {})
    _require(all(_sha(code.get(name)) for name in RUNTIME_FILES), 'missing runtime code binding')
    return {name: value for name, value in code.items()
            if name in RUNTIME_FILES or name.startswith(RUNTIME_PREFIXES)}


def _source_directory(path, environment):
    _require(path.name == environment and path.parent.name == 'source',
             'only source/<same environment> directories are eligible')


def _old_costs(path, episodes, old_seconds):
    rows = events(path)
    selected = [row for row in rows if row.get('phase') == PHASE]
    model = [row for row in selected if row.get('kind') == 'model']
    steps = [row for row in selected if row.get('kind') == 'env_step']
    ids = {episode['task_id'] for episode in episodes}
    _require({row.get('task_id') for row in model} == ids, 'native Actor cost task IDs differ')
    _require(all(row.get('role') == 'actor' and row.get('ok') is True and not row.get('error')
                 and row.get('synthetic') is not True for row in model), 'invalid native Actor cost row')
    for key in ('prompt_tokens', 'completion_tokens'):
        _require(all(type(row.get(key)) is int and row[key] >= 0 for row in model),
                 'missing actual native token usage: ' + key)
    _require(all(type(row.get('elapsed')) in (float, int) and math.isfinite(row['elapsed'])
                 and row['elapsed'] >= 0 for row in model), 'invalid native Actor duration')
    _require(all(row.get('task_id') in ids for row in steps), 'foreign task in native step costs')
    for episode in episodes:
        actual = [row.get('step') for row in steps if row['task_id'] == episode['task_id']]
        _require(actual == list(range(len(episode.get('history', [])))),
                 'native environment-step ledger differs from episode')
    return dict(ledger_path=str(path), ledger_sha256=file_hash(path), phase=PHASE,
                selected_rows_hash=digest(selected), selected_rows=len(selected),
                actor_model_requests=len(model), env_steps=len(steps),
                prompt_tokens=sum(row['prompt_tokens'] for row in model),
                completion_tokens=sum(row['completion_tokens'] for row in model),
                actor_seconds=sum(row['elapsed'] for row in model),
                measurement_seconds=old_seconds,
                scope='Historical baseline costs only; not new execution or candidate/Reviewer/Evolver costs')


def seed_source_baseline(engine, origin):
    """Import exactly the explicitly pinned old source baseline, or fail closed.

    Call before ``engine.run`` on a fresh source. Repeating an identical, already
    audited import on resume only verifies it; existing searches are not modified.
    ``origin`` contains schema=1, absolute source_run, and SHA256 pins for the
    state, measurement (selected by measurement_key), and native costs ledger.
    """
    _require(type(origin) is dict and set(origin) == ORIGIN_FIELDS
             and type(origin['schema']) is int and origin['schema'] == 1,
             'invalid explicit origin schema')
    _require(all(_sha(origin[key]) for key in ORIGIN_FIELDS - {'schema', 'source_run'}),
             'origin hashes must be lowercase SHA256')
    _require(type(origin['source_run']) is str and Path(origin['source_run']).is_absolute(),
             'source_run must be an explicit absolute directory')
    old = Path(origin['source_run']).resolve()
    new = engine.store.root.resolve()
    environment = engine.context.get('environment')
    _require(environment in {'alfworld', 'textworld'}, 'unknown native environment')
    _source_directory(old, environment)
    _source_directory(new, environment)
    _require(old != new, 'origin and destination must differ')
    _require(engine.synthetic is False and engine.binding.get('synthetic') is False,
             'synthetic engines are ineligible')
    baseline = Program(BASELINE_SOURCE)
    _require(engine.initial.source == BASELINE_SOURCE and engine.initial.hash == baseline.hash,
             'new initial source is not the unchanged baseline')
    _require(engine.binding.get('source_memory_hash') == digest([]),
             'source baseline must have no imported Evolution Memory')
    actual = dict(context=engine.context, exploration_tasks=engine.explore, seed=engine.seed,
                  evaluator=engine.evaluator.fingerprint, logs_hash=digest(engine.logs),
                  ordered_logs_hash=digest(json.dumps(engine.logs, ensure_ascii=False, sort_keys=False)),
                  ordered_context_hash=digest(json.dumps(engine.context, ensure_ascii=False, sort_keys=False)),
                  environment_step_limit=engine.environment_step_limit)
    _require(all(engine.binding.get(key) == value for key, value in actual.items()),
             'live engine inputs differ from its frozen binding')
    _require(engine.binding.get('model_capabilities') == getattr(engine.capabilities, 'fingerprint', None),
             'live model capability binding differs')

    state_path = old / 'state.json'
    measurement_path = old / 'measurements' / (origin['measurement_key'] + '.json')
    costs_path = old / 'runner/native/costs.jsonl'
    for path, pin in ((state_path, 'state_sha256'), (measurement_path, 'measurement_sha256'),
                      (costs_path, 'native_costs_sha256')):
        _require(path.is_file() and file_hash(path) == origin[pin], 'origin artifact hash differs: ' + pin)
    state = read(state_path)
    binding = state['binding']
    _require(state.get('binding_hash') == digest(binding), 'old state binding hash differs')
    _require(state.get('initial_program') == baseline.hash, 'old initial program differs')
    source_path = old / 'programs' / (baseline.hash + '.py')
    _require(source_path.is_file() and source_path.read_text() == BASELINE_SOURCE,
             'old baseline source differs')
    _require(_execution_binding(binding) == _execution_binding(engine.binding),
             'native execution binding differs')
    _require(_runtime_code(binding) == _runtime_code(engine.binding), 'native execution code differs')
    expected_old_key = digest(dict(binding=binding, program=baseline.hash, tasks=engine.explore))
    measurement = read(measurement_path)
    _require(origin['measurement_key'] == expected_old_key and measurement.get('key') == expected_old_key,
             'old measurement key differs')
    _require(measurement.get('phase') == PHASE and measurement.get('synthetic') is False,
             'only the real initial parent exploration measurement is eligible')
    _require('reuse_provenance' not in measurement and measurement.get('physical', True) is True
             and measurement.get('newly_executed', True) is True, 'chained imports are ineligible')
    episodes = measurement.get('episodes', [])
    _require(measurement.get('episodes_hash') == digest(episodes), 'old episode hash differs')
    _require(engine._complete(episodes, engine.explore), 'incomplete or invalid native episodes')
    for episode, task in zip(episodes, engine.explore):
        _require(episode.get('phase') == PHASE and episode.get('environment') == environment
                 and episode.get('family') == task.get('family')
                 and episode.get('group_id') == task.get('group_id')
                 and episode.get('program_hash') == baseline.hash
                 and episode.get('program_identity_hash') == baseline.identity_hash
                 and episode.get('read_only_memory') is True
                 and episode.get('memory_read_calls') == 1
                 and episode.get('generated_memory_write_calls') == 0
                 and _sha(episode.get('memory_hash_before'))
                 and episode.get('memory_hash_before') == episode.get('memory_hash_after'),
                 'episode is not the unchanged read-only parent mechanism')
        for record in (episode, task):
            for field in ('phase', 'split', 'official_split', 'data_role'):
                value = str(record.get(field, '')).lower()
                _require(not any(marker in value for marker in ('heldout', 'held_out', 'final',
                             'confirm', 'generalization', 'candidate', 'eval_in_distribution',
                             'eval_out_of_distribution')) and value != 'test', 'sealed or candidate evidence')
    seconds = measurement.get('seconds')
    _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
             'invalid historical measurement duration')
    costs = _old_costs(costs_path, episodes, seconds)
    key = digest(dict(binding=engine.binding, program=baseline.hash, tasks=engine.explore))
    provenance = dict(origin=dict(origin), source_run=str(old), source_binding_hash=state['binding_hash'],
                      source_measurement_path=str(measurement_path), source_measurement_key=expected_old_key,
                      source_episodes_hash=measurement['episodes_hash'], source_program_hash=baseline.hash,
                      historical_costs=costs)
    imported = dict(key=key, episodes=episodes, episodes_hash=measurement['episodes_hash'], phase=PHASE,
                    synthetic=False, seconds=0.0, physical=False, newly_executed=False,
                    reuse_provenance=provenance)
    audit = dict(schema=1, kind='source_baseline_import', destination_binding_hash=digest(engine.binding),
                 measurement_key=key, measurement_hash=digest(imported), phase=PHASE, physical=False,
                 newly_executed=False, episodes=len(episodes), reuse_provenance=provenance)
    access = dict(key=key, phase=PHASE, physical=False, operation='import', episodes=len(episodes),
                  seconds=0.0, reuse_receipt='baseline_reuse.json', reuse_receipt_hash=digest(audit))
    cache_path = new / 'measurements' / (key + '.json')
    audit_path = new / 'baseline_reuse.json'
    access_path = new / 'measurement_access.jsonl'
    with engine.store.lock():
        if audit_path.exists():
            _require(read(audit_path) == audit and cache_path.is_file() and read(cache_path) == imported
                     and [row for row in events(access_path) if row.get('operation') == 'import'] == [access],
                     'existing import receipt changed or is incomplete')
            if (new / 'state.json').exists():
                resumed = read(new / 'state.json')
                _require(resumed.get('binding') == engine.binding
                         and resumed.get('binding_hash') == digest(engine.binding)
                         and resumed.get('initial_program') == baseline.hash, 'resumed source binding differs')
        else:
            _require(not (new / 'state.json').exists() and not (new / 'trials').exists(),
                     'destination search has already started')
            _require(not (new / 'runner/native/costs.jsonl').exists(), 'destination has native execution costs')
            _require(not list((new / 'measurements').glob('*.json')) and not access_path.exists(),
                     'destination already contains measurement evidence')
            write(cache_path, imported)
            write(audit_path, audit)
            append(access_path, access)
    return audit
