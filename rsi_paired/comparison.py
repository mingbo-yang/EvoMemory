"""Descriptive differences between scored slots, computed after the batch is sealed."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re

from evomem.common import digest, file_hash
from .witness import _deliveries

_ORIGINS = ('historical_failure', 'historical_success', 'fresh')


@dataclass(frozen=True)
class _Trace:
    task: tuple
    score: int | None
    actions: tuple
    requests: tuple
    counts: dict


def _trace(ep, actors):
    recorder = ep.get('native_recorder') or {}
    steps = recorder.get('steps')
    has_recorder = isinstance(steps, list) and recorder.get('initial') is not None
    if ep['score'] is not None and not has_recorder:
        raise ValueError('A scored native task is missing its recorder')
    histories = [digest(recorder['initial'])] if has_recorder else []
    actions = []
    for step in steps or []:
        if not isinstance(step, dict) or 'action_took' not in step or 'obs' not in step:
            if ep['score'] is not None:
                raise ValueError('A scored native step is missing its action or observation')
            break
        # Different actions or environment feedback invalidate comparisons at later steps.
        if histories:
            histories.append(digest([histories[-1], step]))
        actions.append(step['action_took'])
    delivered, _, _, counts = _deliveries(ep, actors)
    requests = []
    for i in range(counts['actor_attempts']):
        if i not in delivered:
            requests.append(None)
            continue
        item, _ = delivered[i]
        step = item['step']
        if type(step) is not int or not 0 <= step < len(histories):
            if ep['score'] is not None:
                raise ValueError('Actor delivery is outside the recorded native trajectory')
            requests.append(None)
            continue
        requests.append((histories[step], digest(actors[i]['messages'][1:]), digest(item['memory'])))
    return _Trace(task=(ep['task_id'], ep['family'], ep['seed'], ep['environment_seed']),
                  score=ep['score'], actions=tuple(actions),
                  requests=tuple(requests), counts=counts)


def _native_inputs(stage):
    receipts = {}
    for path in (stage / 'native' / 'episodes').glob('*/complete.json'):
        receipt = json.loads(path.read_bytes())
        binding = receipt['binding']
        if binding['scope'] != 'panel':
            continue
        result = receipt['result']
        job = Path(binding['job'])
        if (job.name != 'private' or job.parent.parent.name != 'evaluation' or digest(binding) != path.parent.name
                or digest(result) != receipt['result_hash'] or not re.fullmatch(r'attempt-[0-9]+', result['attempt'])):
            raise ValueError('Invalid candidate native receipt binding')
        key = job.parent.name, binding['index']
        if key in receipts:
            raise ValueError('Duplicate native evaluation task index')
        receipts[key] = (path.parent / result['attempt'], binding['source'], result)
    return receipts


def _load_trace(receipt, row, source):
    directory, program_hash, result = receipt
    if (digest(row) != digest(result['row']) or program_hash != sha256(source.encode()).hexdigest()
            or result.get('episode_wire') != 'episode.json'):
        raise ValueError('Candidate comparison references a different evaluation or program')
    wire = (directory / 'episode.json').read_bytes()
    if sha256(wire).hexdigest() != row['episode_hash'] or file_hash(directory/'actor.jsonl') != result['actor_hash']:
        raise ValueError('Candidate comparison requires intact Actor and episode receipts')
    ep = json.loads(wire)
    score = ep['score']
    if (ep['synthetic'] is not False or ep['program_hash'] != program_hash or ep['task_id'] != row['task_id']
            or ep['family'] != row['family'] or row['native_score'] != score or ep['phase'] != 'panel'
            or (row['status'] == 'method_eval_failure' and score is not None)
            or (row['status'] == 'completed' and (score not in (0, 1) or not ep['native_reward_verified']))
            or row['status'] not in ('completed', 'method_eval_failure')):
        raise ValueError('Candidate comparison native outcome differs from the episode')
    with (directory / 'actor.jsonl').open() as stream:
        actors = [row for line in stream if (row := json.loads(line)).get('role') == 'actor']
    return _trace(ep, actors)


def _compare(left, right):
    """Compare actual deliveries only while both actors have matching decision histories."""
    memory = Counter(different=0, equal_on_all_delivered_attempts=0, inconclusive=0)
    actions = Counter(different=0, equal=0, unavailable=0)
    rewards = Counter(left_wins=0, ties=0, right_wins=0, unavailable=0)
    verified_pairs = differing_pairs = unpaired_left = unpaired_right = 0
    for a, b in zip(left, right, strict=True):
        if a.task != b.task:
            raise ValueError('Compared native tasks differ in order, identity or seeds')
        aligned = differences = 0
        for lreq, rreq in zip(a.requests, b.requests):
            if lreq is None or rreq is None or lreq[:2] != rreq[:2]:
                break
            aligned += 1
            differences += lreq[2] != rreq[2]
        verified_pairs += aligned
        differing_pairs += differences
        unpaired_left += len(a.requests) - aligned
        unpaired_right += len(b.requests) - aligned
        complete = a.score is not None and b.score is not None
        if differences:
            memory['different'] += 1
        elif complete and aligned > 0 and aligned == len(a.requests) == len(b.requests):
            memory['equal_on_all_delivered_attempts'] += 1
        else:
            memory['inconclusive'] += 1
        if complete:
            actions['equal' if a.actions == b.actions else 'different'] += 1
            if a.score == b.score:
                rewards['ties'] += 1
            else:
                rewards['left_wins' if a.score > b.score else 'right_wins'] += 1
        else:
            actions['unavailable'] += 1
            rewards['unavailable'] += 1
    return dict(memory_tasks=dict(memory), aligned_verified_request_pairs=verified_pairs,
                differing_aligned_request_pairs=differing_pairs, unpaired_left_requests=unpaired_left,
                unpaired_right_requests=unpaired_right, action_tasks=dict(actions), native_tasks=dict(rewards))


def _stratified_scores(entries, slots, traces):
    """Report sample counts, native outcomes and matched differences against the current parent."""
    for name, slot in slots.items():
        groups = {origin: Counter(tasks=0, scored=0, successes=0, wins_vs_parent=0, ties_vs_parent=0,
                                  losses_vs_parent=0, unavailable_vs_parent=0) for origin in _ORIGINS}
        for index, entry in enumerate(entries):
            outcome = groups[entry['source']]
            outcome['tasks'] += 1
            score = traces[name][index].score if name in traces else None
            parent = traces['parent'][index].score
            if score is not None:
                outcome['scored'] += 1
                outcome['successes'] += score
            if score is None or parent is None:
                outcome['unavailable_vs_parent'] += 1
            elif score == parent:
                outcome['ties_vs_parent'] += 1
            else:
                outcome['wins_vs_parent' if score > parent else 'losses_vs_parent'] += 1
        slot['native_by_task_origin'] = {k: dict(v, successes=v['successes'] if v['scored'] else None)
                                         for k, v in groups.items()}


def _panel(records, receipts, sampling=None):
    names = list(records)
    candidates = [n for n in names if n not in ('parent', 'retained')]
    slots, traces, cache = {}, {}, {}
    task_ids = [r['task_id'] for r in records['parent']['evaluation']]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError('Candidate report requires a nonempty ordered task panel with unique task IDs')
    if sampling is not None:
        entries = sampling['entries']
        if ([e['task_id'] for e in entries] != task_ids
                or any(e['source'] not in _ORIGINS for e in entries)
                or Counter(e['source'] for e in entries) != Counter(sampling['counts'])):
            raise ValueError('Sampling strata differ from the ordered scored tasks')
    for name, record in records.items():
        source = record.get('source')
        if source is not None and digest(source) != record.get('source_hash'):
            raise ValueError('Candidate source differs from its registered hash')
        last = record.get('drafts', [{}])[-1].get('source') if source is None else source
        slot = slots[name] = dict(valid=record['valid'], source_hash=record.get('source_hash'),
                                  final_draft_hash=digest(last) if isinstance(last, str) else None,
                                  evaluated_tasks=0, native_scored_tasks=0, native_successes=None)
        if 'evaluation' not in record:
            if record.get('success_count') is not None:
                raise ValueError('Unexecuted slot has a score')
            continue
        rows = record['evaluation']
        if [r['task_id'] for r in rows] != task_ids or source is None:
            raise ValueError('Candidate evaluations do not share the same ordered task panel')
        sequence = []
        for i, row in enumerate(rows):
            key = (record['eval_key'], i)
            if key not in receipts:
                raise ValueError(f'Missing sealed native receipt for {name} at panel index {i}')
            if key not in cache:
                cache[key] = _load_trace(receipts[key], row, source)
            elif (row != receipts[key][2]['row'] or sha256(source.encode()).hexdigest() != receipts[key][1]):
                raise ValueError('Aliased slot references a different native receipt')
            sequence.append(cache[key])
        traces[name] = sequence
        scores = [t.score for t in sequence if t.score is not None]
        slot.update(evaluated_tasks=len(rows), native_scored_tasks=len(scores),
                    native_successes=sum(scores) if scores else None,
                    method_eval_failures=len(rows)-len(scores),
                    actor_requests=sum(t.counts['actor_attempts'] for t in sequence),
                    verified_deliveries=sum(t.counts['wire_verified'] for t in sequence),
                    nonempty_deliveries=sum(t.counts['nonempty'] for t in sequence),
                    observed_content_changes=sum(t.counts['observed_content_changes'] for t in sequence))
    if sampling is not None:
        _stratified_scores(entries, slots, traces)
    pairs = []
    for left, right in combinations(names, 2):
        a, b = slots[left]['source_hash'], slots[right]['source_hash']
        pair = dict(left=left, right=right, identical_source=(a == b if a and b else None))
        if left in traces and right in traces:
            pair.update(status='evaluated', **_compare(traces[left], traces[right]))
        else:
            pair['status'] = 'unscored_slot'
        pairs.append(pair)
    vectors = [tuple(t.score for t in traces[n]) for n in candidates if n in traces]
    complete = [v for v in vectors if None not in v]
    valid_sources = [slots[name]['source_hash'] for name in candidates if slots[name]['valid']]
    unique_sources = len(set(valid_sources))
    first_drafts = [records[name]['drafts'][0] for name in candidates if records[name].get('drafts')]
    known_drafts = [d for d in first_drafts if 'error' in d]
    compliant_first = sum(d['error'] is None for d in known_drafts)
    candidate_pairs = [p for p in pairs if p['status'] == 'evaluated'
                       and p['left'] in candidates and p['right'] in candidates]
    return dict(status='scored', ordered_task_ids=task_ids, slots=slots, pairwise=pairs,
                candidate_slots=len(candidates),
                valid_candidates=len(valid_sources), distinct_valid_candidate_sources=unique_sources,
                duplicate_fraction_of_valid=((len(valid_sources) - unique_sources) / len(valid_sources) if valid_sources else None),
                first_drafts_with_receipts=len(known_drafts), compliant_first_drafts=compliant_first,
                first_draft_compliance_rate=compliant_first / len(known_drafts) if known_drafts else None,
                evaluated_candidate_pairs=len(candidate_pairs),
                candidate_pairs_with_observed_delivery_difference=sum(p['memory_tasks']['different'] > 0 for p in candidate_pairs),
                candidate_pairs_with_native_outcome_difference=sum(
                    p['native_tasks']['left_wins'] + p['native_tasks']['right_wins'] > 0 for p in candidate_pairs),
                candidates_with_complete_native_results=len(complete), distinct_complete_native_vectors=len(set(complete)))


def candidate_differences(stage_root):
    """Summarize completed native panels. Use after the batch audit, outside all model requests."""
    stage = Path(stage_root)
    batch = stage / 'batch'
    history = json.loads((batch/'commit.json').read_bytes())['state']['history']
    receipts = _native_inputs(stage)
    resolved = batch/'resolved_panels.json'
    strata = json.loads(resolved.read_bytes())['sampling'] if resolved.is_file() else {}
    primary = _panel(json.loads((batch/'primary_results.json').read_bytes()), receipts, strata.get('primary'))
    meta_file = batch/'meta_results.json'
    meta = (_panel(json.loads(meta_file.read_bytes()), receipts, strata.get('meta')) if meta_file.is_file()
            else dict(status='not_scored', reason=history[-1]['status']))
    return dict(version=1, primary=primary, meta=meta)
