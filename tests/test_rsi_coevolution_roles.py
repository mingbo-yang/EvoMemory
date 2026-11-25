"""Offline wire-level tests for the diagnostic/proposal ablation."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from phase1_codegen.program import BASELINE_SOURCE, Program
from rsi_paired.coevolution_roles import CoevolutionRolesMixin, INITIAL_REVIEWER
from rsi_paired.roles import INITIAL_STRATEGY, Roles
from rsi_paired.workflow import ContractError, digest
from tests.test_rsi_roles import Tokens


class PilotRoles(CoevolutionRolesMixin, Roles):
    pass


def roles(tmp_path, handler, context=32768):
    sent = []
    def transport(request):
        sent.append(request)
        output = handler(request['response_format']['json_schema']['name'], request)
        return {'model': 'synthetic', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output)}}]}

    instance = PilotRoles(tmp_path, {'context_tokens': context, 'model_name': 'synthetic', 'model_fingerprint': 'cpu'},
                          tokenizer=Tokens(), transport=transport)
    instance.native = SimpleNamespace(verified_learning_evidence=lambda ledger: ledger)
    return instance, sent


def test_natural_independent_decisions_are_grounded_and_scoped(tmp_path):
    decisions = {'rsi_decide': (True, ['propose']), 'rsi_reviewer_decide': (False, [])}
    def handler(role, request):
        update, targets = decisions[role]
        return dict(should_update=update, reason='Synthetic distinction.', expected_effect='Different proposed edits.',
                    target_components=targets, evidence_ids=['post.review'])

    r, calls = roles(tmp_path, handler)
    packet = {'evidence_ids': ['post.review']}
    assert r.decide_component('evolver', INITIAL_STRATEGY, packet)['should_update']
    assert not r.decide_component('reviewer', INITIAL_REVIEWER, packet)['should_update']
    assert [json.loads(x['messages'][1]['content'])['public_evidence'] for x in calls] == [packet, packet]
    props = calls[1]['response_format']['json_schema']['schema']['properties']
    assert props['target_components']['items']['enum'] == ['diagnose']
    decisions['rsi_reviewer_decide'] = (True, ['propose'])
    with pytest.raises(ContractError, match='reviewer self-revision decision'):
        r.decide_component('reviewer', INITIAL_REVIEWER, packet, 'synthetic invalid response')


def test_changed_diagnosis_reaches_wire_with_identical_verified_originals(tmp_path):
    focus = 'Compare reported actions to feedback before diagnosing.'
    def handler(role, request):
        if role == 'rsi_reviewer_revise':
            return {'prompt': focus}
        return {'findings': [{'evidence_ids': ['learning.0'], 'judgment': 'insufficient_evidence',
                              'observation': 'Synthetic evidence leaves the cause unresolved.'}]}

    r, calls = roles(tmp_path, handler)
    ledger = [{'task_id': 'synthetic_01', 'score': 0}]
    verified = []
    def verify(rows):
        verified.append(rows)
        return [dict(row, program_hash=Program(BASELINE_SOURCE).hash) for row in rows]
    r.native.verified_learning_evidence = verify
    revised = r.revise_component('reviewer', INITIAL_REVIEWER, {'evidence_ids': ['post.review']})
    before, after = [r.diagnose_with_policy(policy, BASELINE_SOURCE, {}, ledger, [])
                     for policy in (INITIAL_REVIEWER, revised)]
    assert verified == [ledger, ledger]
    assert before['evidence_hash'] == after['evidence_hash']
    assert before['reviewer_policy_hash'] == digest(INITIAL_REVIEWER)
    assert after['reviewer_policy_hash'] == digest(revised)
    assert calls[1]['messages'][1] == calls[2]['messages'][1]
    assert focus not in calls[1]['messages'][0]['content'] and focus in calls[2]['messages'][0]['content']
    with pytest.raises(ContractError, match='unchanged'):
        r.revise_component('reviewer', revised, {'evidence_ids': ['post.review']})
    with pytest.raises(ContractError):
        r.diagnose_with_policy(dict(revised, score=1), BASELINE_SOURCE, {}, [], [])


def test_slot_conditioned_policy_generates_from_same_public_packet(tmp_path):
    code = ('def propose(evidence, repairing):\n'
            '    if evidence["proposal_slot"]["index"] == 1:\n'
            '        return "Check recall."\n'
            '    return "Check precision."')
    def handler(role, request):
        if role == 'rsi_revise':
            return {'code': code, 'prompt': 'Design each candidate around the diagnosis.'}
        return {'source': BASELINE_SOURCE}
    r, calls = roles(tmp_path, handler)
    packet = {'evidence_ids': ['pre.diagnosis'], 'diagnosis': {'findings': []}}
    before = deepcopy(packet)
    revised = r.revise_component('evolver', INITIAL_STRATEGY, packet)
    outcomes = [r.propose_slot(revised, BASELINE_SOURCE, packet, 42, None, None, i) for i in (0, 1)]
    assert all(value['public_hash'] == digest(before) for value in outcomes)
    assert packet == before
    first, second = calls[-2:]
    assert 'Check recall.' in first['messages'][0]['content']
    assert 'Check precision.' in second['messages'][0]['content']
    for i, request in enumerate((first, second), 1):
        wire = json.loads(request['messages'][1]['content'])
        assert wire['public_evidence'] == before and wire['proposal_slot'] == {'index': i, 'count': 3}


def test_large_learning_ledger_keeps_every_observed_outcome_and_citation(tmp_path):
    def handle(role, request):
        return {'findings': [{'evidence_ids': ['learning.499'], 'judgment': 'no_defect_demonstrated',
                              'observation': 'Synthetic observation.'}]}
    r, requests = roles(tmp_path, handle, context=23000)
    program_hash = Program(BASELINE_SOURCE).hash
    ledger = [dict(task_id=f'synthetic-{i}', program_hash=program_hash, score=i % 2, phase='learning',
                   evidence={'review_witness': {'note': 'whole synthetic receipt ' + str(i)}}) for i in range(500)]
    r.diagnose_with_policy(INITIAL_REVIEWER, BASELINE_SOURCE, {}, ledger, [])
    wire = json.loads(requests[0]['messages'][1]['content'])
    assert wire['evidence_layout'] == 'learning_table_v1'
    assert wire['learning_program_hash'] == program_hash
    for i, item in enumerate(ledger):
        values = dict(zip(wire['learning_columns'], wire['evidence'][f'learning.{i}'], strict=True))
        assert values['task_id'] == item['task_id'] and values['score'] == item['score']


def test_paired_post_primary_diagnoses_see_same_native_feedback(tmp_path):
    focus = dict(INITIAL_REVIEWER, prompt='Test whether the diagnosis identified a repeated failure.')
    def handle(role, request):
        revised = focus['prompt'] in request['messages'][0]['content']
        return {'findings': [{'evidence_ids': ['primary.P1', 'learning.499'],
                              'judgment': 'insufficient_evidence',
                              'observation': 'Updated synthetic opinion.' if revised else 'Old synthetic opinion.'}]}
    r, calls = roles(tmp_path, handle, context=24500)
    program = Program(BASELINE_SOURCE).hash
    ledger = [dict(task_id=f'S-{i}', score=i % 2, program_hash=program,
                   evidence={'review_witness': {'feedback': 'synthetic LEARNING receipt ' + str(i)}}) for i in range(500)]
    records = {}
    for slot in ('parent', 'P1', 'P2', 'P3'):
        records[slot] = dict(evaluation=[dict(task_id=f'M-{i}', score=i % 2, native_score=i % 2,
            evidence={'review_witness': {'feedback': f'synthetic PRIMARY {slot} {i}'}}) for i in range(50)])
    post = {'findings': [{'observation': 'Synthetic external review.'}]}
    first, second = [r.diagnose_post_primary(x, BASELINE_SOURCE, {}, ledger, [], records, primary_review=post)
                     for x in (INITIAL_REVIEWER, focus)]
    assert first['findings'] != second['findings']
    assert first['evidence_hash'] == second['evidence_hash']
    assert first['raw_evidence_hash'] == second['raw_evidence_hash']
    assert calls[0]['messages'][1] == calls[1]['messages'][1]
    wire = json.loads(calls[0]['messages'][1]['content'])
    assert wire['evidence']['fixed.primary_review'] == post
    assert wire['evidence_layout'] == 'learning_table_v1' and wire['omitted_witness_count'] > 0
    for slot in records:
        displayed = wire['evidence']['primary.' + slot]
        assert [wire['primary_tasks'][row[0]] for row in displayed['evaluation']] == [f'M-{i}' for i in range(50)]
        assert any(w['feedback'].startswith(f'synthetic PRIMARY {slot} ') for w in wire['episode_witnesses'].values())
    assert any(w['feedback'].startswith('synthetic LEARNING') for w in wire['episode_witnesses'].values())


def test_deployed_policies_guide_future_self_modifications(tmp_path):
    reviewer_guidance = 'Check if your previous findings actually predicted the observed effect.'
    evolver_guidance = 'Compare your proposal to its own predicted effect.'
    code = ('def propose(evidence, repairing):\n'
            '    if evidence["role_phase"] == "self_decide":\n'
            '        return "Audit your past choice."\n'
            '    if evidence["role_phase"] == "self_revise":\n'
            '        return "Revise your weakest hypothesis test."\n'
            '    return "Compare proposed interventions."')
    def handle(role, req):
        if role.endswith('decide'):
            return dict(should_update=False, reason='Synthetic', evidence_ids=['post.review'],
                        expected_effect='Synthetic', target_components=[])
        if role == 'rsi_reviewer_revise':
            old = json.loads(req['messages'][1]['content'])['current_strategy']['prompt']
            return {'prompt': reviewer_guidance if not old else reviewer_guidance + ' Check again.'}
        old = json.loads(req['messages'][1]['content'])['current_strategy']['propose']['prompt']
        return {'code': code, 'prompt': evolver_guidance if old != evolver_guidance else evolver_guidance + ' Compare.'}
    r, calls = roles(tmp_path, handle)
    packet = {'evidence_ids': ['post.review']}
    for role, initial, criteria in [('reviewer', INITIAL_REVIEWER, reviewer_guidance),
                                    ('evolver', INITIAL_STRATEGY, evolver_guidance)]:
        adopted = r.revise_component(role, initial, packet)
        r.decide_component(role, adopted, packet)
        r.revise_component(role, adopted, packet)
        for request in calls[-2:]:
            assert criteria in request['messages'][0]['content']
            assert json.loads(request['messages'][1]['content'])['current_strategy'] == adopted
            assert any(json.loads(p.read_text())['policy_hash'] == digest(adopted)
                       for p in (r.root / 'requests').glob('*/intent.json')
                       if json.loads(p.read_text())['request'] == request)
    assert 'Audit your past choice.' in calls[-2]['messages'][0]['content']
    assert 'Revise your weakest hypothesis test.' in calls[-1]['messages'][0]['content']


def test_episode_value_dictionary_is_lossless_for_generated_policies():
    from rsi_paired.roles import _expand_public_rows, _tabulate_public_rows
    packet = {'primary': {'P1': {'evaluation': []}, 'P2': {'evaluation': []}}, 'compact_episode_refs': True}
    for slot in packet['primary'].values():
        slot['evaluation'] = [dict(task_id=f'synthetic-{i}', status='complete', gate_source='fresh',
                                   episode_hash_prefix=digest(i)[:16], score=i % 2) for i in range(3)]
    before = deepcopy(packet)
    _tabulate_public_rows(packet)
    CoevolutionRolesMixin.share_episode_values(packet)
    assert _expand_public_rows(packet) == before
