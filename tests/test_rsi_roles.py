"""Synthetic role/contract receipts. No live service is contacted."""
import copy
import json

import pytest

from phase1_codegen.program import BASELINE_SOURCE
from rsi_paired.roles import INITIAL_STRATEGY, Roles, _compact_experiences, policy_instruction
from rsi_paired.workflow import ContractError, digest


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
    assert r.decide(INITIAL_STRATEGY, packet) == value
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
