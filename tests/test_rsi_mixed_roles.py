"""Mixed evidence attribution, family coverage and lossless public packet tests."""
import json
from copy import deepcopy

import pytest

from evomem.common import dumps
from phase1_codegen.program import BASELINE_SOURCE, Program
from rsi_paired.coevolution_roles import INITIAL_REVIEWER, CoevolutionRolesMixin, paired_family_outcomes
from rsi_paired.panel import COEVOLUTION_MIXED_10X60_PROTOCOL, MIXED_FAMILIES
from rsi_paired.roles import (_compact_experiences, _expand_public_rows, _share_public_series,
                              _tabulate_public_rows)
from rsi_paired.workflow import digest
from tests.test_rsi_coevolution_roles import roles
from tests.test_rsi_roles import Tokens


def outcomes(families, n, *, swap=False):
    rows = []
    for i in range(n):
        family = families[i % len(families)]
        task = f'{family}-training-{i}'
        score = (i // len(families)) % 2
        if swap and family == 'cool':
            score = 1
        if swap and family == 'heat':
            score = 0
        transition = dict(step=i, action={'text':'look'}, feedback={'text':f'{task} feedback'},
                          delivery={'request_hash':digest([task,'request']), 'packed_hash':digest([task,'pack']),
                                    'excerpt':{'text':'Verified past interaction', 'start':0, 'end':25}})
        witness = {'episode_hash':digest(task), 'version':'synthetic', 'transitions':[transition]}
        rows.append(dict(task_id=task, family=family, score=score, native_score=float(score),
                         status='completed', program_hash=Program(BASELINE_SOURCE).hash,
                         evidence={'review_witness':witness}, episode_hash=digest(task)))
    return rows


def test_equal_total_preserves_regression_and_rescue_on_individual_families(tmp_path):
    parent = outcomes(MIXED_FAMILIES, 60)
    candidate = outcomes(MIXED_FAMILIES, 60, swap=True)
    effects = paired_family_outcomes(parent, candidate)
    assert sum(e['gained'] - e['lost'] for e in effects.values()) == 0
    assert effects['cool'] == dict(tasks=10, parent_successes=5, successes=10, gained=5, lost=0)
    assert effects['heat'] == dict(tasks=10, parent_successes=5, successes=0, gained=0, lost=5)
    with pytest.raises(ValueError):
        paired_family_outcomes(parent, list(reversed(candidate)))
    with pytest.raises(ValueError):
        paired_family_outcomes(parent[:-1], candidate)
    r, _ = roles(tmp_path, lambda *_: {})
    r.protocol = COEVOLUTION_MIXED_10X60_PROTOCOL
    packed = r.compress(dict(phase='post_primary', primary={'parent':{'evaluation':parent},
        'P1':{'evaluation':candidate}, 'P2':{'valid':False}}, evidence_ids=['primary.P1']))
    assert packed['mixed.family_outcomes']['P1'] == effects
    assert packed['mixed.family_outcomes']['P2'] is None
    assert 'mixed.family_outcomes' in packed['evidence_ids']
    archived = _compact_experiences([{'modification_records':[dict(slot='P1', family_outcomes=effects)]}], Tokens())
    assert archived[0]['family_outcomes'] == effects


def test_mixed_post_primary_can_use_context_headroom_without_changing_evidence(tmp_path):
    r, _ = roles(tmp_path, lambda *_: {})
    r.protocol = COEVOLUTION_MIXED_10X60_PROTOCOL
    rows = outcomes(MIXED_FAMILIES, 12)
    packet = dict(phase='post_primary',
                  primary={'parent': {'evaluation': rows, 'source': 'x' * 9000}},
                  evidence_ids=['primary.parent'])
    r.config['public_token_limit'] = 3300
    packed = r.compress(deepcopy(packet))
    assert 3300 < r.tokens.count(dumps(packed)) <= 6372
    unpacked = _expand_public_rows(packed)
    assert unpacked['primary']['parent']['source'] == packet['primary']['parent']['source']
    assert [row['score'] for row in unpacked['primary']['parent']['evaluation']] == [row['score'] for row in rows]
    r.context = 32768 - 21000
    with pytest.raises(ValueError, match='Mandatory public evidence'):
        r.compress(deepcopy(packet))


def test_old_and_new_diagnose_all_families_from_same_training_witnesses(tmp_path):
    def respond(role, request):
        msg = request['messages'][0]['content']
        return {'findings':[dict(evidence_ids=['mixed.family_outcomes'], judgment='insufficient_evidence',
                      observation='Revised diagnosis.' if 'A new synthetic focus.' in msg else 'Original diagnosis.')]}
    r, calls = roles(tmp_path, respond, context=30000)
    r.protocol = COEVOLUTION_MIXED_10X60_PROTOCOL
    ledger = outcomes(MIXED_FAMILIES, 600)
    slots = {k:{'evaluation':outcomes(MIXED_FAMILIES,60,swap=k!='parent')} for k in ('parent','P1','P2','P3')}
    fixed = {'findings':[{'observation':'Synthetic post-primary observation.'}]}
    old, new = [r.diagnose_post_primary(policy, BASELINE_SOURCE, {}, ledger, [], slots, primary_review=fixed)
                for policy in (INITIAL_REVIEWER,dict(INITIAL_REVIEWER,prompt='A new synthetic focus.'))]
    assert old['raw_evidence_hash'] == new['raw_evidence_hash'] and old['evidence_hash'] == new['evidence_hash']
    assert old['findings'] != new['findings'] and calls[0]['messages'][1] == calls[1]['messages'][1]
    body = json.loads(calls[0]['messages'][1]['content'])
    assert body['evidence']['fixed.primary_review'] == fixed
    assert body['omitted_witness_count'] > 0
    assert 'final' not in body['evidence']
    for family in MIXED_FAMILIES:
        group = body['evidence']['learning.'+family]
        assert len(group['tasks']) == len(group['scores']) == len(group['ledger_indices']) == 100
        for i,j in enumerate(group['ledger_indices']):
            assert int(group['scores'][i]) == ledger[j]['score'] and group['tasks'][i] == ledger[j]['task_id']
    witnessed = set()
    for key, witness in body['episode_witnesses'].items():
        source, pos, index = key.split('/')
        if source.startswith('learning.'):
            family=source.split('.')[1]
            row=ledger[body['evidence'][source]['ledger_indices'][int(pos)]]
        else:
            row=slots[source.split('.')[1]]['evaluation'][int(pos)]
            family=row['family']
        original=row['evidence']['review_witness']
        assert witness['transition'] == original['transitions'][int(index)]
        assert witness['archived_witness_hash'] == digest(original)
        witnessed.add(family)
    assert witnessed == set(MIXED_FAMILIES)
    assert any(k.startswith('learning.') for k in body['episode_witnesses'])
    assert any(k.startswith('primary.') for k in body['episode_witnesses'])
    for idx, row in enumerate(slots['P1']['evaluation']):
        assert body['panel_tasks'][idx] == [row['family'],row['task_id']]
        assert int(body['evidence']['primary.P1']['evaluation']['scores'][idx]) == row['score']


def test_public_column_encoding_is_lossless_across_mixed_schemas_and_numeric_types():
    parent = outcomes(MIXED_FAMILIES,60)
    candidates = deepcopy(parent)
    candidates[1].update(candidate_error={'code':'synthetic_failure'},native_score=None,score=0)
    for i in (7,13):
        candidates[i].update(candidate_error={'code':'synthetic_failure'},native_score=None,score=0)
    packet = {'primary':{'parent':{'evaluation':parent}, 'P1':{'evaluation':candidates}},'compact_episode_refs':True}
    before = dumps(packet)
    _tabulate_public_rows(packet)
    CoevolutionRolesMixin.share_episode_values(packet)
    _share_public_series(packet)
    assert dumps(_expand_public_rows(packet)) == before
