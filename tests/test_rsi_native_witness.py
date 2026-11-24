"""Synthetic wire/recorder fixtures for the bounded Reviewer witness."""
import json

import pytest

from evomem.common import digest, dumps
from rsi_paired.witness import WITNESS_BYTES, review_witness


def fixture(actions, entries, *, updates=None):
    # One initial read followed by before_action / after_observation events.
    history, recorder, events, injections, actors = [], [], [], [], []
    slot = dict(entries=entries) if entries else {}

    def event(payload, response=None):
        index = len(events)
        events.append(dict(sequence=index, status='valid', event=payload,
                           response=response, actual_memory=slot))
        return index

    event(dict(type='task_start', step=0), entries)
    for step, (action, feedback) in enumerate(actions):
        raw = None
        if step in (updates or {}):
            raw, packed = updates[step]
            slot = dict(entries=packed) if packed else {}
        seq = event(dict(type='before_action', step=step, attempt=step), raw)
        system = 'Fixed prompt.\n' + (json.dumps(slot, ensure_ascii=False, indent=2) if slot else 'No entries.')
        messages = [dict(role='system', content=system), dict(role='user', content='visible input')]
        request = digest([step, messages])
        actors.append(dict(task_id='fixture', kind='model', ok=True, request_hash=request,
                           original_message_hash=digest(messages), messages=messages))
        injections.append(dict(task_id='fixture', attempt=step, step=step,
                               actor_delivery_verified=True, actor_request_hash=request,
                               original_user_prompt_hash=digest('visible input'), system_hash=digest(system),
                               event_sequence=seq, memory=slot))
        history.append(dict(action=action, next_observation=feedback))
        recorder.append(dict(action_took=action, obs=feedback))
        event(dict(type='after_observation', attempt=step, step=step+1, action=action, visible=dict(obs=feedback)))
    episode = dict(task_id='fixture', goal='calibrate sensor', history=history,
                   memory_hook=dict(events=events, injections=injections), native_recorder=dict(steps=recorder))
    return episode, actors


def test_related_content_and_persistent_delivery_come_from_real_wire():
    paragraph = 'The workshop is lined with instruments. ' * 18
    experience = paragraph + 'Calibrate the sensor after measuring drift. Read the new offset.'
    entry = dict(id='x', experience=experience)
    ep, actors = fixture([('walk', 'arrived'), ('calibrate sensor', 'offset corrected'), ('close', 'closed')], [entry])
    witness = review_witness(ep, actors)
    assert witness['delivery']['wire_verified'] == witness['delivery']['nonempty'] == 3
    assert witness['delivery']['observed_content_changes'] == 1
    assert witness['delivery']['raw_readout_events'] == 1
    selected = next(row for row in witness['transitions'] if row['step'] == 1)
    delivery = selected['delivery']
    excerpt = delivery['excerpt']
    assert delivery['raw']['event_sequence'] == 0
    assert delivery['request_hash'] == actors[1]['request_hash']
    assert excerpt['start'] > 240 and excerpt['text'] == experience[excerpt['start']:excerpt['end']]
    assert 'Calibrate the sensor' in excerpt['text'] and 'Action:' not in excerpt['text']
    assert selected['feedback']['text'] == 'offset corrected'
    assert witness['omitted_transitions'] == 1


def test_changed_raw_and_delivered_entries_and_empty_slots():
    private = {'id':'long', 'experience':'hidden raw material'}
    delivered = {'id':'visible', 'experience':'Dose at noon, no matter the weather.'}
    raw = [private, delivered]
    ep, actors = fixture([('wait','waiting'), ('dose','done')], [], updates={1: (raw,[delivered])})
    witness = review_witness(ep, actors)
    assert witness['delivery']['nonempty'] == 1
    assert witness['delivery']['packing_changed_events'] == 1
    d = next(t['delivery'] for t in witness['transitions'] if t['step'] == 1)
    assert d['raw']['event_sequence'] == 3 and d['raw']['hash'] == digest(raw)
    assert d['raw']['entries'] == 2 and d['raw']['changed_in_packing']
    assert d['packed_hash'] == digest([delivered]) and d['excerpt']['entry_id_hash'] == digest('visible')
    raw_span = d['raw']['excerpt']
    assert raw_span['text'] == private['experience'] and raw_span['entry_index'] == 0
    assert raw_span['visibility'] == 'returned_not_necessarily_delivered'
    assert raw_span['present_verbatim_in_any_packed_entry'] is False
    assert private['experience'] not in dumps(d['excerpt'])
    zero = next(t['delivery'] for t in witness['transitions'] if t['step'] == 0)
    assert zero['packed_entries'] == 0 and 'excerpt' not in zero and 'excerpt' not in zero['raw']


@pytest.mark.parametrize('duplicate_elsewhere', [False, True])
def test_raw_middle_span_clipped_from_its_entry_may_still_be_delivered_elsewhere(duplicate_elsewhere):
    omitted = 'Measure drift before calibration.'
    original = dict(id='a', experience='Header.\n' + omitted + '\nFooter.')
    packed = [dict(id='a', experience='Header.\n[readout clipped]\nFooter.')]
    other = dict(id='b', experience=omitted)
    raw = [original]
    if duplicate_elsewhere:
        raw.append(other)
        packed.append(other)
    ep, actors = fixture([('measure drift', 'drift recorded')], [], updates={0: (raw, packed)})
    delivery = review_witness(ep, actors)['transitions'][0]['delivery']
    span = delivery['raw']['excerpt']
    assert span['entry_index'] == 0 and original['experience'][span['start']:span['end']] == span['text']
    assert span['text'] == omitted and span['present_verbatim_in_any_packed_entry'] is duplicate_elsewhere


def test_raw_excerpt_under_budget_pressure_preserves_native_and_wire_evidence(monkeypatch):
    import rsi_paired.witness as module
    raw = [dict(id='r', experience='private ' * 300)]
    packed = [dict(id='r', experience='private ' * 20)]
    ep, actors = fixture([('check', 'done'), ('check', 'ready')], [], updates={0: (raw, packed)})
    complete = review_witness(ep, actors)
    for transition in complete['transitions']:
        data = transition['delivery']['raw']
        assert data.pop('excerpt')
        data['excerpt_status'] = 'omitted_for_budget'
    monkeypatch.setattr(module, 'WITNESS_BYTES', len(dumps(complete).encode()))
    bounded = review_witness(ep, actors)
    assert bounded == complete


@pytest.mark.parametrize('fault', ['content', 'task', 'request', 'host_slot', 'adapter', 'messages'])
def test_unverified_request_does_not_support_delivery(fault):
    ep, actors = fixture([('read','visible feedback')], [dict(id='m', experience='specific fact')])
    item = ep['memory_hook']['injections'][0]
    if fault == 'content':
        actors[0]['messages'][0]['content'] = 'changed'
    elif fault == 'task':
        actors[0]['task_id'] = 'another task'
    elif fault == 'request':
        item['actor_request_hash'] = 'foreign'
    elif fault == 'host_slot':
        item['memory'] = {'entries':[dict(id='changed', experience='never delivered')]}
    elif fault == 'adapter':
        item['actor_delivery_verified'] = False
    else:
        actors.clear()
    witness = review_witness(ep, actors)
    assert witness['delivery']['wire_verified'] == 0 and witness['delivery']['unverified'] == 1
    assert witness['transitions'][0]['delivery'] is None
    assert 'specific fact' not in dumps(witness)


def test_long_unbroken_content_is_bounded_and_offsets_are_exact():
    content = '界' * 3000 + ' calibrate ' + '界' * 3000
    ep, actors = fixture([('calibrate' + '\x01'*600, 'corrected' + '\x02'*600)] * 2,
                         [{'id':'x','experience':content}])
    witness = review_witness(ep, actors)
    assert len(dumps(witness).encode()) <= WITNESS_BYTES
    assert witness['omitted_transitions'] == 2 - len(witness['transitions'])
    for transition in witness['transitions']:
        row = ep['history'][transition['step']]
        for key, source in [('action', row['action']), ('feedback', row['next_observation'])]:
            span = transition[key]
            assert span['text'] == source[span['start']:span['end']] and span['total'] == len(source)
        if transition.get('delivery'):
            span = transition['delivery']['excerpt']
            assert span['text'] == content[span['start']:span['end']]
            assert span['end']-span['start'] < span['total']


def test_inconsistent_recorder_or_attempt_leaves_unverified_link_empty():
    ep, actors = fixture([('calibrate','changed'), ('finish','finished')], [dict(id='m', experience='e')])
    ep['native_recorder']['steps'][0]['obs'] = 'unrelated recorder'
    ep['memory_hook']['events'][-1]['event']['attempt'] = 0
    witness = review_witness(ep, actors)
    assert witness['verified_transitions'] == 1
    assert witness['omitted_transitions'] == 1
    assert witness['transitions'][0]['step'] == 1 and witness['transitions'][0]['delivery'] is None
    assert ep['history'][0]['next_observation'] not in dumps(witness['transitions'])


def test_wire_can_be_verified_even_when_raw_event_provenance_is_missing():
    ep, actors = fixture([('inspect', 'visible')], [dict(id='m', experience='verified content')])
    del ep['memory_hook']['events'][0]['actual_memory']
    witness = review_witness(ep, actors, episode_hash='a'*64)
    assert witness['episode_hash'] == 'a'*64
    assert witness['delivery']['packing_unverified_events'] == 1
    delivery = witness['transitions'][0]['delivery']
    assert delivery['raw'] is None and delivery['excerpt']['text'] == 'verified content'


def test_missing_native_recorder_is_unknown_even_with_actor_delivery():
    ep, actors = fixture([('inspect', 'reported')], [])
    del ep['native_recorder']
    witness = review_witness(ep, actors)
    assert witness['delivery']['wire_verified'] == 1
    assert witness['history_steps'] == 1 and witness['native_steps'] is None
    assert witness['transitions'] == [] and witness['omitted_transitions'] is None
