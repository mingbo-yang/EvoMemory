"""Synthetic role/contract receipts. No live service is contacted."""
import copy
import json

import pytest

from phase1_codegen.program import BASELINE_SOURCE, Program
from rsi_paired.roles import (
    INITIAL_STRATEGY, MAX_EVIDENCE_REFERENCES, Roles, _compact_experiences, policy_instruction,
)
from rsi_paired.workflow import ContractError, _Batch, _public_record, digest


class Tokens:
    def count(self, text):
        return (len(text) + 2) // 3

    def chat_count(self, messages):
        return self.count(json.dumps(messages))


def roles(tmp_path, value):
    calls = []

    def transport(request):
        calls.append(request)
        return {'model': request['model'], 'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(value)}}], 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}

    return Roles(tmp_path, {'context_tokens': 65536, 'model_name': 'synthetic', 'model_fingerprint': 'sha'},
                 tokenizer=Tokens(), transport=transport), calls


def test_actual_policy_execution_and_equal_public_view(tmp_path):
    r, calls = roles(tmp_path, {'source': BASELINE_SOURCE})
    updated = copy.deepcopy(INITIAL_STRATEGY)
    updated['propose']['code'] = ('def propose(evidence, repairing):\n'
                                  '    if repairing:\n        return "Check the supplied error."\n'
                                  '    return "Reconsider the stated hypothesis."\n')
    packet = {'phase': 'post_primary', 'evidence_ids': ['post.review'], 'review': {'findings': []}}
    left = r.propose(INITIAL_STRATEGY, BASELINE_SOURCE, packet, 17, None, None)
    right = r.propose(updated, BASELINE_SOURCE, packet, 17, None, None)
    assert left['loaded_evolver_hash'] != right['loaded_evolver_hash']
    assert calls[0]['messages'][1] == calls[1]['messages'][1]
    assert left['public_hash'] == right['public_hash'] == digest(packet)
    assert 'Reconsider the stated hypothesis.' in calls[1]['messages'][0]['content']
    assert calls[0]['seed'] == calls[1]['seed'] == 17
    assert len(list((r.root/'requests').glob('*/intent.json'))) == 2
    again = r.propose(updated, BASELINE_SOURCE, packet, 17, None, None)
    assert again == right and len(calls) == 2
    r.propose(updated, BASELINE_SOURCE, packet, 17, 'incomplete_source', 'invalid structure')
    last = calls[-1]['messages']
    assert last[1]['content'].count('incomplete_source') == 1
    assert json.loads(last[1]['content'])['public_evidence'] == packet
    assert 'Check the supplied error.' in last[0]['content']
    assert len((r.root/'role_costs.jsonl').read_text().splitlines()) == 3


def test_only_strategy_package_mutable(tmp_path):
    r, _ = roles(tmp_path, {'prompt': 'Do not change evaluation.', 'code':
                             'def propose(evidence, repairing):\n    return "Check the evidence."'})
    changed = r.revise(INITIAL_STRATEGY, {'phase': 'post_primary'})
    assert policy_instruction(changed, {}, False) == 'Check the evidence.'
    for code in ('def propose(evidence, repairing):\n    return model_call("model", {})',
                 'def on_event(state, event, context):\n    return {}',
                 'import os\ndef propose(evidence, repairing):\n    return "a"'):
        invalid = copy.deepcopy(changed)
        invalid['propose']['code'] = code
        with pytest.raises(ContractError):
            policy_instruction(invalid, {}, False)
    with pytest.raises(ContractError):
        policy_instruction(dict(changed, decide='yes'), {}, False)


def test_invalid_citations_and_real_output_contract(tmp_path):
    r, _ = roles(tmp_path, {'findings': [{'evidence_ids': ['invisible'], 'judgment': 'supported_defect',
                                          'observation': 'A score was present.'}]})
    with pytest.raises(ContractError):
        r.diagnose(BASELINE_SOURCE, {}, [], [])
    r.check(BASELINE_SOURCE)
    with pytest.raises(ContractError, match='event output requires'):
        r.check('def on_event(state, event, context):\n    return {"unknown": 0}')
    with pytest.raises(ContractError, match='on_event'):
        r.check('def x():\n    return 1')
    # Data-dependent logic failures remain native candidate failures, not repair grants.
    r.check('def on_event(state, event, context):\n    return state["missing"]')


def test_repeated_valid_review_citations_preserve_evidence_and_wire(tmp_path):
    citations = ['parent.memory', 'parent.source', 'parent.memory']
    response = {'findings': [{'evidence_ids': citations, 'judgment': 'insufficient_evidence',
                              'observation': 'The bank is empty.'}]}
    r, calls = roles(tmp_path, response)
    review = r.diagnose(BASELINE_SOURCE, {}, [], [])
    assert review['findings'][0]['evidence_ids'] == ['parent.memory', 'parent.source']
    schema = calls[0]['response_format']['json_schema']['schema']
    refs = schema['properties']['findings']['items']['properties']['evidence_ids']
    assert (refs['minItems'], refs['maxItems']) == (1, MAX_EVIDENCE_REFERENCES)
    receipt, = (r.root / 'requests').glob('*/response.json')
    raw = json.loads(receipt.read_text())['choices'][0]['message']['content']
    assert json.loads(raw) == response


def test_pre_review_only_cites_learning_under_current_mechanism(tmp_path):
    program_hash = Program(BASELINE_SOURCE).hash
    ledger = [{'task_id': 'historical', 'program_hash': 'previous-source', 'score': 0},
              {'task_id': 'unbound', 'score': 0},
              {'task_id': 'current', 'program_hash': program_hash, 'score': 1}]
    response = {'findings': [{'evidence_ids': ['learning.2'], 'judgment': 'no_defect_demonstrated',
                              'observation': 'The current task succeeded.'}]}
    r, calls = roles(tmp_path, response)
    r.diagnose(BASELINE_SOURCE, {}, ledger, [])
    evidence = json.loads(calls[0]['messages'][1]['content'])['evidence']
    assert set(evidence) == {'parent.source', 'parent.memory', 'learning.2'}
    assert evidence['learning.2']['program_hash'] == evidence['parent.memory']['program_hash'] == program_hash
    response['findings'][0]['evidence_ids'] = ['learning.0']
    r, _ = roles(tmp_path / 'invalid', response)
    with pytest.raises(ContractError, match='displayed IDs'):
        r.diagnose(BASELINE_SOURCE, {}, ledger, [])


def test_verified_witness_reaches_pre_and_post_reviewer(tmp_path):
    witness = {'transitions': [{'step': 9, 'feedback': 'synthetic verified feedback'}]}
    row = {'task_id': 'test-task', 'episode_hash': 'archive', 'score': 0,
           'program_hash': Program(BASELINE_SOURCE).hash,
           'evidence': {'review_witness': witness, 'injected_readouts': ['obsolete prefix']}}
    response = {'findings': [{'evidence_ids': ['learning.0'], 'judgment': 'insufficient_evidence',
                              'observation': 'The recorded outcome leaves the cause unresolved.'}]}
    r, calls = roles(tmp_path, response)
    r.diagnose(BASELINE_SOURCE, {}, [row], [])
    state = dict(mechanism=BASELINE_SOURCE, evolver=INITIAL_STRATEGY, memory={}, ledger=[], history=[])
    batch = _Batch(tmp_path / 'batch', state, {}, r, 123, False, {})
    record = {'valid': True, 'evaluation': [row], 'success_count': 0, 'source': BASELINE_SOURCE}
    response['findings'][0]['evidence_ids'] = ['primary.P1']
    review = batch._review('primary', {'parent': record, 'P1': record})
    pre, post = [json.loads(c['messages'][1]['content']) for c in calls]
    ref = digest(witness)
    for body in (pre, post):
        assert body['episode_witnesses'] == {ref: witness}
        assert body['omitted_witness_count'] == 0
        assert 'obsolete prefix' not in str(body)
    assert pre['evidence']['learning.0']['evidence']['review_witness_ref'] == ref
    for key in ('primary.parent', 'primary.P1'):
        assert post['evidence'][key]['evaluation'][0]['evidence']['review_witness_ref'] == ref
    assert review['evidence_hash'] == digest(post)
    assert 'synthetic verified feedback' not in str(_public_record(record))
    assert 'review_witness' in row['evidence']


def test_review_context_omits_complete_witnesses_and_keeps_all_scores(tmp_path):
    r, calls = roles(tmp_path, {'findings': [{'evidence_ids': ['primary.P1'],
        'judgment': 'insufficient_evidence', 'observation': 'Some excerpts were omitted.'}]})
    r.context = 6500
    rows = [{'task_id': f'test-{i}', 'score': i % 2, 'episode_hash': f'archive-{i}',
             'evidence': {'review_witness': {'transitions': [{'step': i, 'text': 'x' * 2400}]}}}
            for i in range(8)]
    r.review(BASELINE_SOURCE, {'P1': {'evaluation': rows}})
    body = json.loads(calls[0]['messages'][1]['content'])
    shown = body['episode_witnesses']
    assert 0 < len(shown) < len(rows)
    assert body['omitted_witness_count'] == len(rows) - len(shown)
    for original, actual in zip(rows, body['evidence']['primary.P1']['evaluation'], strict=True):
        ref = actual['evidence']['review_witness_ref']
        assert actual['score'] == original['score'] and ref == digest(original['evidence']['review_witness'])
        if ref in shown:
            assert shown[ref] == original['evidence']['review_witness']


@pytest.mark.parametrize('citations', [[], ['parent.source', 'unavailable'], [None],
                                       ['parent.source'] * (MAX_EVIDENCE_REFERENCES + 1)])
def test_invalid_bound_review_citations_are_rejected(tmp_path, citations):
    r, _ = roles(tmp_path, {'findings': [{'evidence_ids': citations, 'judgment': 'supported_defect',
                                         'observation': 'The finding requires evidence.'}]})
    with pytest.raises(ContractError, match='displayed IDs'):
        r.diagnose(BASELINE_SOURCE, {}, [], [])


def test_retrieval_shows_no_private_literals(tmp_path):
    r, _ = roles(tmp_path, {})
    mods = [{'slot': 'P1', 'diff_excerpt': ['+    value = "an instance-specific answer"',
             '+    kind = event["type"]'], 'slot_ref': 'slots/P1.json', 'success_count': None}]
    records = _compact_experiences([{'modification_records': mods}], r.tokens)
    assert '[literal]' in str(records) and 'instance-specific' not in str(records)
    selected = r._history(records, {'diagnosis': 'error'})
    assert len(selected) == 1 and selected[0]['success_count'] is None


def test_decision_is_fixed_and_must_cite_displayed_evidence(tmp_path):
    value = {'should_update': True, 'reason': 'The comparison supplied a failed submission.',
             'evidence_ids': ['primary.P1'], 'target_components': ['propose'],
             'expected_effect': 'A different proposal may be testable.'}
    r, calls = roles(tmp_path, value)
    packet = {'evidence_ids': ['primary.P1']}
    value['evidence_ids'] = ['primary.P1', 'primary.P1']
    assert r.decide(INITIAL_STRATEGY, packet) == {**value, 'evidence_ids': ['primary.P1']}
    assert 'primary.P1' in str(calls[-1]['response_format'])
    value['target_components'] = ['revise']
    r.set_job(tmp_path/'batch'/'jobs'/'decision'/'1'/'private')
    with pytest.raises(ContractError):
        r.decide_retry(INITIAL_STRATEGY, packet, 'Invalid target component')
    assert json.loads(calls[-1]['messages'][1]['content'])['public_evidence'] == packet
    value['target_components'] = ['propose']
    value['evidence_ids'] = ['unseen']
    r.set_job(tmp_path/'batch'/'jobs'/'decision'/'2'/'private')
    with pytest.raises(ContractError):
        r.decide(INITIAL_STRATEGY, packet)


def test_repair_keeps_malformed_output_and_does_not_send_unbound_evidence(tmp_path):
    r, _ = roles(tmp_path, {'missing_source': 'draft'})
    with pytest.raises(ContractError) as err:
        r.propose(INITIAL_STRATEGY, BASELINE_SOURCE, {'evidence_ids': []}, 0, None, None)
    assert json.loads(err.value.draft) == {'missing_source': 'draft'}
    packet = {'phase': 'post_primary', 'primary': {'P1': {'valid': False, 'source': None, 'drafts': [
            {'source': 'bad 1', 'error': 'E1'}, {'source': 'bad 2', 'error': 'E2'}]}},
            'evidence_ids': ['primary.P1']}
    shown = r.compress(packet)
    assert shown['primary']['P1']['last_unscored_draft'] == 'bad 2'
    assert shown['primary']['P1']['drafts'][0]['error'] == 'E1'


@pytest.mark.parametrize('model,expected', [('qwen3.8-27b-local', {'include_reasoning': False}),
                                           ('gemma4-31b', {}), ('qwen3.5-9b', {}), ('GLM-4.7-Flash', {})])
def test_structured_role_request_matches_registered_model_options(tmp_path, model, expected):
    calls = []
    def transport(request):
        calls.append(request)
        return {'model': model, 'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'source': BASELINE_SOURCE})}}]}

    caller = Roles(tmp_path, {'model_name': model, 'model_fingerprint': 'cpu', 'context_tokens': 32768},
                   tokenizer=Tokens(), transport=transport)
    caller._call('propose', 'Test', {}, 42)
    request, = calls
    assert request['chat_template_kwargs'] == {'enable_thinking': False}
    assert {key: request[key] for key in ('include_reasoning',) if key in request} == expected
    assert request['response_format']['type'] == 'json_schema'
