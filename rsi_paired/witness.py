"""Bounded Reviewer excerpts from the ALMA wire and recorded native transitions."""
from __future__ import annotations

from collections import Counter
import json
from os.path import commonprefix
import re
from evomem.common import digest, dumps


# Bytes of canonical UTF-8 JSON. Long strings and JSON escaping count in full.
WITNESS_BYTES = 4096
_WORD = re.compile(r"[^\W_]+")


def _terms(text):
    return set(_WORD.findall(text.casefold()))


def _excerpt(text, limit, query=(), start=0, end=None):
    """Offsets are Unicode code points in the original string, half-open."""
    end = len(text) if end is None else end
    if end - start > limit:
        anchor = next((m.start() for m in _WORD.finditer(text, start, end)
                       if m.group().casefold() in query), start)
        start = min(end - limit, max(start, anchor - limit // 3))
        end = start + limit
    return dict(text=text[start:end], start=start, end=end, total=len(text))


def _units(text):
    """Use whole lines for structured records or sentence spans for prose."""
    boundary = r"\n+" if '\n' in text else r"(?<=[.!?。！？])\s+"
    start = 0
    for match in re.finditer(boundary, text):
        if match.start() > start:
            yield start, match.start()
        start = match.end()
    if start < len(text):
        yield start, len(text)


def _memory_excerpt(rows, action, feedback, limit=300):
    if not rows:
        return None
    action_terms = _terms(action)
    query = action_terms | _terms(feedback)
    units = []
    for entry, row in enumerate(rows):
        text = row['experience']
        units.extend((entry, start, end, _terms(text[start:end])) for start, end in _units(text))
    if not units:
        return None
    frequency = Counter(term for _, _, _, terms in units for term in terms & query)
    def rank(index):
        terms = units[index][3]
        return sum((2 if t in action_terms else 1) / frequency[t]
                   for t in sorted(terms & query)) / max(1, len(terms)) ** 0.5
    best = max(range(len(units)), key=rank)
    entry, start, end, terms = units[best]
    neighbours = [i for i in (best-1, best+1) if 0 <= i < len(units) and units[i][0] == entry
                  and max(end, units[i][2]) - min(start, units[i][1]) <= limit and rank(i)]
    if neighbours:
        _, left, right, _ = units[max(neighbours, key=rank)]
        start, end = min(start, left), max(end, right)
    row = rows[entry]
    return dict(entry_index=entry, entry_id_hash=digest(row['id']),
                basis='lexical_overlap' if terms & query else 'first_available',
                **_excerpt(row['experience'], limit, query, start, end))


def _changed_raw_excerpt(raw, packed, action, feedback):
    """Account for dropped entries and for packers that keep both ends of one entry."""
    by_id = {row['id']: row['experience'] for row in packed}
    fragments, origins = [], []
    for index, row in enumerate(raw):
        text, visible = row['experience'], by_id.get(row['id'], '')
        if text == visible:
            continue
        start = len(commonprefix((text, visible)))
        suffix = len(commonprefix((text[start:][::-1], visible[start:][::-1])))
        end = len(text) - suffix
        if end > start:
            fragments.append(dict(id=row['id'], experience=text[start:end]))
            origins.append((index, start, len(text)))
    span = _memory_excerpt(fragments, action, feedback, limit=160)
    if span is None:
        return None
    index, start, size = origins[span['entry_index']]
    span.update(entry_index=index, start=start+span['start'], end=start+span['end'], total=size,
                visibility='returned_not_necessarily_delivered',
                present_verbatim_in_any_packed_entry=any(span['text'] in value for value in by_id.values()))
    return span


def _actor_slot(item, actor):
    """Verify the official rendered slot against the successful wire receipt."""
    if not (item.get('actor_delivery_verified') and actor.get('kind') == 'model' and actor.get('ok')
            and actor.get('request_hash') and actor['request_hash'] == item.get('actor_request_hash')):
        return False
    messages = actor.get('messages') or []
    if (not messages or messages[0].get('role') != 'system'
            or digest(messages) != actor.get('original_message_hash')):
        return False
    system = messages[0]['content']
    users = [m['content'] for m in messages if m.get('role') == 'user']
    if not users or digest(system) != item.get('system_hash') or digest(users[-1]) != item.get('original_user_prompt_hash'):
        return False
    memory = item['memory']
    return not memory or json.dumps(memory, ensure_ascii=False, indent=2) in system


def _deliveries(ep, actors):
    """Resolve a later before_action with a null response to its actual producing event."""
    hook = ep.get('memory_hook') or {}
    origins, after, producer = {}, {}, None
    raw_events = packing_changes = packing_unknown = 0
    trace = hook.get('events')
    for event in trace or []:
        if event.get('status') != 'valid':
            continue
        payload = event['event']
        if event.get('response') is not None and payload['type'] != 'task_end':
            producer = event
            raw_events += 1
            packed = event.get('actual_memory')
            if isinstance(packed, dict):
                packing_changes += event['response'] != packed.get('entries', [])
            else:
                packing_unknown += 1
        if payload['type'] == 'before_action':
            origins[event['sequence']] = (payload, producer)
        if payload['type'] == 'after_observation':
            after[payload['step'] - 1] = payload

    verified, changes, nonempty = {}, [], 0
    previous = {}
    slots = hook.get('injections', [])
    for index, item in enumerate(slots):
        if (index >= len(actors) or item.get('attempt') != index
                or not ep.get('task_id') or ep['task_id'] != item.get('task_id')
                or ep['task_id'] != actors[index].get('task_id') or not _actor_slot(item, actors[index])):
            previous = None  # A gap breaks continuity.
            continue
        memory = item['memory']
        payload, source = origins.get(item.get('event_sequence'), ({}, None))
        if (payload.get('attempt') != index or payload.get('step') != item['step']
                or source is None or source.get('actual_memory') != memory):
            source = None
        if previous is not None and memory != previous:
            changes.append(index)
        previous = memory
        nonempty += bool(memory.get('entries'))
        verified[index] = item, source
    count = max(len(slots), len(actors))
    summary = dict(actor_attempts=count, wire_verified=len(verified), nonempty=nonempty,
                   unverified=count-len(verified), observed_content_changes=len(changes),
                   raw_readout_events=raw_events if trace is not None else None,
                   packing_changed_events=packing_changes if trace is not None else None,
                   packing_unverified_events=packing_unknown if trace is not None else None)
    return verified, after, changes, summary


def _selected_steps(steps, goal, changed_steps):
    # The first post-start change, when available, and goal-related transitions
    # with complementary terms. These are excerpt selectors, never defect labels.
    selected = []
    later = sorted(i for i in changed_steps if i in steps and i > 0)
    if later:
        selected.append(later[0])
    goal_terms = _terms(goal)
    words = {i: _terms(a) for i, (a, _) in steps.items()}
    counts = Counter(t for terms in words.values() for t in terms)
    while len(selected) < min(2, len(steps)):
        covered = set().union(*(words[i] for i in selected))
        def rank(i):
            relevant = goal_terms & words[i]
            return (sum(1/counts[t] for t in sorted(relevant-covered)),
                    sum(1/counts[t] for t in sorted(relevant)), i)
        selected.append(max((i for i in steps if i not in selected), key=rank))
    return sorted(selected)


def _delivery_excerpt(item, source, action, feedback):
    rows = item['memory'].get('entries', [])
    result = dict(actor_attempt=item['attempt'], request_hash=item['actor_request_hash'],
                  packed_entries=len(rows), packed_hash=digest(rows))
    optional_raw = None
    if source is None:
        result['raw'] = None
    else:
        raw = source['response']
        result['raw'] = dict(event_sequence=source['sequence'], entries=len(raw),
                             hash=digest(raw), changed_in_packing=raw != rows)
        if raw != rows:
            optional_raw = _changed_raw_excerpt(raw, rows, action, feedback)
            result['raw']['excerpt_status'] = ('omitted_for_budget' if optional_raw is not None
                                                else 'no_distinct_raw_text_range')
    excerpt = _memory_excerpt(rows, action, feedback)
    if excerpt is not None:
        if source is not None:
            entry_id = rows[excerpt['entry_index']]['id']
            for index, row in enumerate(raw):
                if row['id'] == entry_id:
                    excerpt['raw_entry_index'] = index
                    excerpt['raw_entry_chars'] = len(row['experience'])
                    break
        result['excerpt'] = excerpt
    return result, optional_raw


def review_witness(ep, actors, *, episode_hash=None, bank=None):
    history = ep.get('history') or []
    native = (ep.get('native_recorder') or {}).get('steps')
    steps = {}
    for index, (row, recorder) in enumerate(zip(history, native or [])):
        action, feedback = row.get('action'), row.get('next_observation')
        if (isinstance(action, str) and isinstance(feedback, str)
                and action == recorder.get('action_took') and feedback == recorder.get('obs')):
            steps[index] = action, feedback
    verified, after, changes, summary = _deliveries(ep, actors)
    if bank is not None:
        summary['slot_entry_limit'] = bank.get('top_k')
        summary['slot_token_limit'] = bank.get('read_budget')
    changed = {verified[attempt][0]['step'] for attempt in changes}
    selected = _selected_steps(steps, ep.get('goal', ''), changed)
    native_count = len(native) if isinstance(native, list) else None
    result = dict(version='actor_native_excerpt_v1', delivery=summary,
                  history_steps=len(history), native_steps=native_count, verified_transitions=len(steps),
                  omitted_transitions=native_count, transitions=[])
    if episode_hash is not None:
        if re.fullmatch(r'[a-f0-9]{64}', episode_hash) is None:
            raise ValueError('Invalid episode archive hash for Reviewer evidence')
        result['episode_hash'] = episode_hash
    raw_excerpts = []
    for index in selected:
        action, feedback = steps[index]
        transition = dict(step=index, action=_excerpt(action, 128), feedback=_excerpt(feedback, 220))
        event = after.get(index, {})
        slot = verified.get(event.get('attempt'))
        transition['delivery'] = None
        if (slot and slot[0]['step'] == index and event.get('action') == action
                and (event.get('visible') or {}).get('obs') == feedback):
            transition['delivery'], raw = _delivery_excerpt(*slot, action, feedback)
            if raw is not None:
                raw_excerpts.append((transition, raw))
        result['transitions'].append(transition)
        result['omitted_transitions'] -= 1
        if len(dumps(result).encode()) > WITNESS_BYTES:
            # Omit the readout before the native transition under high escape cost.
            transition.pop('delivery', None)
            transition['delivery_omitted_for_budget'] = True
        if len(dumps(result).encode()) > WITNESS_BYTES:
            result['transitions'].pop()
            result['omitted_transitions'] += 1
    # Raw material has lower priority than every actual delivery and native step.
    for transition, span in raw_excerpts:
        if transition not in result['transitions'] or not transition.get('delivery'):
            continue
        raw = transition['delivery']['raw']
        raw['excerpt_status'], raw['excerpt'] = 'shown', span
        if len(dumps(result).encode()) > WITNESS_BYTES:
            del raw['excerpt']
            raw['excerpt_status'] = 'omitted_for_budget'
    return result
