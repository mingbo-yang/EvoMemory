"""Synthetic task streams for post-commit, native-receipt comparison."""
from hashlib import sha256
import json

import pytest

from evomem.common import digest, file_hash
from rsi_paired.comparison import _compare, _trace, candidate_differences
from tests.test_rsi_native_witness import fixture as make_wire


def make_episode(actions, memories, *, updates=None, score=0):
    ep, actors = make_wire(actions, memories, updates=updates)
    ep['native_recorder']['initial'] = {'obs': 'synthetic test room'}
    ep.update(family='simple', seed=42, environment_seed=312, score=score)
    for row in actors:
        row['role'] = 'actor'
    return ep, actors


def test_exact_deliveries_detect_differences_beyond_a_long_shared_prefix():
    steps = [('walk', 'arrived'), ('measure', 'measured')]
    old = [{'id': 'x', 'experience': 'same prefix ' * 80 + 'A'}]
    new = [{'id': 'x', 'experience': 'same prefix ' * 80 + 'B'}]
    a = _trace(*make_episode(steps, old))
    b = _trace(*make_episode(steps, new))
    result = _compare([a], [b])
    assert result['memory_tasks']['different'] == 1
    assert result['differing_aligned_request_pairs'] == result['aligned_verified_request_pairs'] == 2
    assert (a.counts['nonempty'], a.counts['observed_content_changes']) == (2, 1)
    assert result['action_tasks']['equal'] == result['native_tasks']['ties'] == 1


def test_changes_after_policies_diverge_or_stop_have_no_matched_input():
    new = [{'id': 'x', 'experience': 'Later branch information'}]
    a = _trace(*make_episode([('walk', 'left'), ('look', 'observed')], []))
    b = _trace(*make_episode([('turn', 'right'), ('look', 'observed')], [], updates={1: (new, new)}, score=1))
    result = _compare([a], [b])
    assert result['memory_tasks']['inconclusive'] == result['native_tasks']['right_wins'] == 1
    assert result['action_tasks']['different'] == result['aligned_verified_request_pairs'] == 1
    assert result['differing_aligned_request_pairs'] == 0
    assert result['unpaired_left_requests'] == result['unpaired_right_requests'] == 1
    shorter = _trace(*make_episode([('walk', 'left')], []))
    result = _compare([shorter], [a])
    assert result['unpaired_left_requests'] == 0 and result['unpaired_right_requests'] == 1
    assert result['memory_tasks']['inconclusive'] == 1


def test_unverified_delivery_and_missing_native_reward_are_separate():
    ep, actors = make_episode([('look', 'observed')], [], score=0)
    a = _trace(ep, actors)
    ep['memory_hook']['injections'][0]['actor_request_hash'] = 'wrong'
    ep['score'] = None
    b = _trace(ep, actors)
    result = _compare([a], [b])
    assert result['native_tasks'] == dict(left_wins=0, ties=0, right_wins=0, unavailable=1)
    assert result['memory_tasks']['inconclusive'] == result['action_tasks']['unavailable'] == 1
    assert result['aligned_verified_request_pairs'] == 0


@pytest.mark.parametrize('recorder', [None, {}, {'steps':[{}]}, {'initial':{}, 'steps':[{}]}])
def test_pre_environment_or_partial_method_failure_is_inconclusive(recorder):
    good, actors = make_episode([('look','observed')], [])
    valid = _trace(good, actors)
    good['score'], good['native_recorder'] = None, recorder
    missing = _trace(good, [])
    compared = _compare([missing], [valid])
    assert compared['native_tasks']['unavailable'] == compared['memory_tasks']['inconclusive'] == 1
    assert compared['native_tasks']['ties'] == compared['aligned_verified_request_pairs'] == 0
    good['score'] = 0
    with pytest.raises(ValueError, match='scored native'):
        _trace(good, actors)


def test_sealed_panel_report_reuses_native_receipts_and_keeps_invalid_slot_unscored(tmp_path):
    def save(path, value):
        path.write_text(json.dumps(value, sort_keys=True))
    batch = tmp_path/'batch'
    batch.mkdir()
    save(batch/'commit.json', {'state': {'history': [{'status': 'not_triggered'}]}})
    ep, actors = make_episode([('look', 'observed')], [], score=1)
    source = 'synthetic test source'
    program_hash = sha256(source.encode()).hexdigest()
    # Local fake transport fixture supplies the flags expected in an archived receipt.
    ep.update(synthetic=False, phase='panel', program_hash=program_hash, native_reward_verified=True)
    key = 'one-shared-evaluation'
    binding = dict(scope='panel', job=str(batch/'jobs/evaluation'/key/'private'), index=0, source=program_hash)
    path = tmp_path/'native/episodes'/digest(binding)
    attempt = path/'attempt-0'
    attempt.mkdir(parents=True)
    save(attempt/'episode.json', ep)
    (attempt/'actor.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in actors))
    row = dict(task_id=ep['task_id'], family=ep['family'], native_score=1, score=1, status='completed',
               episode_hash=file_hash(attempt/'episode.json'))
    native = dict(row=row, attempt='attempt-0', episode_wire='episode.json', actor_hash=file_hash(attempt/'actor.jsonl'))
    save(path/'complete.json', dict(binding=binding, result=native, result_hash=digest(native)))
    valid = dict(source=source, source_hash=digest(source), evaluation=[row], eval_key=key, success_count=1, valid=True)
    invalid = dict(source=None, source_hash=None, drafts=[{'source': 'malformed candidate', 'error': 'invalid syntax'}], valid=False, success_count=None)
    records = dict(parent=valid, P1=valid, P2=valid, P3=invalid)
    save(batch/'primary_results.json', records)
    sampling = dict(entries=[{'task_id':ep['task_id'], 'source':'historical_failure'}],
                    counts=dict(historical_failure=1, historical_success=0, fresh=0))
    save(batch/'resolved_panels.json', {'sampling': {'primary':sampling}})
    report = candidate_differences(tmp_path)
    assert report['meta'] == {'status':'not_scored', 'reason':'not_triggered'}
    primary = report['primary']
    assert primary['candidate_slots'] == 3 and primary['distinct_valid_candidate_sources'] == 1
    assert primary['duplicate_fraction_of_valid'] == 0.5
    assert primary['first_drafts_with_receipts'] == 1 and primary['first_draft_compliance_rate'] == 0
    assert primary['evaluated_candidate_pairs'] == primary['distinct_complete_native_vectors'] == 1
    assert primary['candidates_with_complete_native_results'] == 2
    assert primary['slots']['P3']['native_successes'] is None
    assert primary['slots']['P2']['native_by_task_origin']['historical_failure']['successes'] == 1
    missing = primary['slots']['P3']['native_by_task_origin']['historical_failure']
    assert missing['unavailable_vs_parent'] == 1 and missing['successes'] is None
    pairs = {(p['left'], p['right']): p for p in primary['pairwise']}
    assert pairs['P1', 'P2']['memory_tasks']['equal_on_all_delivered_attempts'] == 1
    assert pairs['P1', 'P3']['status'] == 'unscored_slot'
    (attempt/'actor.jsonl').write_text('[]\n')
    with pytest.raises(ValueError, match='intact Actor'):
        candidate_differences(tmp_path)
