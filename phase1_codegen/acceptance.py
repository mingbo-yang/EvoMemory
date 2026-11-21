"""Fixed adoption rules, separate from generation and mechanism assessment."""
from .outcomes import comparison_for_feedback

import math
import re

REVIEWER_DIRECTED = 'reviewer_directed'
DUAL_POLICIES = frozenset({"reviewer_and_reward", "reviewer_and_reward_or_equal_utility", REVIEWER_DIRECTED})

STRICT_RULE = ('The mechanism executes successfully, the paired native comparison is complete, '
               'the evidence-linked post Reviewer assessment of the candidate is positive, '
               'and the candidate mean reward is positive and strictly exceeds the parent mean.')

POLICIES = {
    REVIEWER_DIRECTED: ('One initial diagnosis followed by one reviewed decision per submitted edit. '
                        'Deploy only an executable mechanism with a complete same-item native 0 to 1 '
                        'comparison and an evidence-supported adopt decision. Continue or stop according '
                        'to the post Reviewer. There is no proposal count or batch cost ceiling.'),
    'reviewer_and_reward': STRICT_RULE,
    'reviewer_and_reward_or_equal_utility': STRICT_RULE + (
        ' There is also an equal-score branch with the same execution, complete comparison and '
        'positive review requirements: every paired task reward must be identical, the candidate '
        'mean must be positive, the review must ground a specific mechanism advantage in evidence, '
        'and there must be no confirmed unresolved candidate functional defect.'),
    'reviewer_opinion': 'valid executable mechanism and evidence-linked positive candidate assessment',
    'native_reward': 'complete exploration delta >= 0; fresh confirmation delta > 0',
}


def paired_complete(evidence, tasks):
    return (evidence.get('complete') is True
            and type(evidence.get('delta')) in (int, float)
            and math.isfinite(evidence['delta'])
            and evidence.get('task_ids') == [t['id'] for t in tasks])


def eligible_for_adoption(binding, trial, *, current_facts=None):
    """Consume validated review receipts; never invent scores or ask the Evolver."""
    if (trial.get('mechanism_validated') is not True or trial.get('error')
            or trial.get('review_error') or not trial.get('candidate_hash')):
        return False
    exploration = trial.get('exploration')
    if exploration is not None and not paired_complete(exploration, binding['exploration_tasks']):
        return False
    policy = binding['acceptance_policy']
    if policy == 'reviewer_and_reward_or_equal_utility':
        strict = eligible_for_adoption(dict(binding, acceptance_policy='reviewer_and_reward'), trial)
        if strict:
            return True
        facts = current_facts or {}
        return equal_utility_decision(trial, (facts.get('comparison') or {}).get('tasks', []), facts)['eligible']
    if policy in {'reviewer_opinion', 'reviewer_and_reward', REVIEWER_DIRECTED}:
        review = trial.get('post_review') or {}
        report = review.get('report') or {}
        if binding.get('review_contract') == 2 and not report.get('evidence_contract'):
            return False
        assessment = report.get('assessment') or {}
        if policy == REVIEWER_DIRECTED:
            if (not exploration or len(binding['exploration_tasks']) != 1
                    or exploration.get('parent') != 0 or exploration.get('candidate') != 1):
                return False
            approved = assessment.get('decision') == 'adopt'
        else:
            approved = assessment.get('verdict') == 'positive'
        positive = (review.get('ok') is True and review.get('stage') == 'post'
                    and assessment.get('subject') == 'candidate' and approved)
        if policy == 'reviewer_opinion':
            # Frozen historical receipts retain their original meaning only.
            return positive
        return (positive and exploration is not None
                and all(type(exploration.get(k)) in (int, float)
                        and math.isfinite(exploration[k]) for k in ('parent', 'candidate'))
                and exploration['candidate'] > 0
                and exploration['candidate'] > exploration['parent']
                and exploration['delta'] == exploration['candidate'] - exploration['parent'])
    if policy == 'native_reward':
        confirm = trial.get('confirmation') or {}
        return (exploration is not None and exploration['delta'] >= 0
                and paired_complete(confirm, binding['confirmation_blocks'][trial['round']])
                and confirm['delta'] > 0)
    raise ValueError('Unknown frozen acceptance policy')


def adoption_feedback(binding, trial, *, current_facts=None):
    """Deterministic feedback; qualitative review and native outcomes stay separate."""
    evidence = trial.get('exploration') or {}
    review = trial.get('post_review') or {}
    assessment = (review.get('report') or {}).get('assessment') or {}
    feedback = dict(policy=binding['acceptance_policy'], eligible=eligible_for_adoption(binding, trial, current_facts=current_facts),
                    review_complete=review.get('ok') is True, native_comparison=comparison_for_feedback(evidence))
    if binding['acceptance_policy'] == REVIEWER_DIRECTED:
        decision = assessment.get('decision')
        if (not feedback['review_complete'] or decision not in ('adopt', 'continue', 'stop')
                or (decision == 'adopt') != feedback['eligible']):
            raise ValueError('Online review decision disagrees with the verified native evidence')
        return dict(feedback, decision=decision)
    feedback.update(reviewer_verdict=assessment.get('verdict'),
                    continuation='If unaccepted, another candidate may be proposed only within the existing iteration budget; deployed code/state remain authoritative.')

    if binding['acceptance_policy'] == 'reviewer_and_reward_or_equal_utility':
        facts = current_facts or {}
        feedback['strict_reward_branch'] = eligible_for_adoption(dict(binding, acceptance_policy='reviewer_and_reward'), trial)
        feedback['equal_utility_branch'] = equal_utility_decision(trial, (facts.get('comparison') or {}).get('tasks', []), facts)
    return feedback


def content_kind(observation):
    # Both separately cited contiguous excerpts carry content. Range/hash
    # metadata and omitted intervals never constitute utility evidence.
    if not re.fullmatch(r'/fragments/\d+/fragment/(text|suffix)', observation['pointer']):
        return None
    eid = observation['evidence_id']
    if eid.endswith(':storage'):
        return 'stored_content'
    if ':hook_read:' in eid:
        return 'delivered_content'
    return None

def changed_cited_content(evaluation, observations, trial):
    """A code change or an unrelated changed bank is not a cited utility gain."""
    groups = {}
    for oid in evaluation['observation_ids']:
        observation = observations[oid]
        kind = content_kind(observation)
        scope = observation['scope']
        if kind is None or scope['phase'] != 'current_exploration':
            continue
        side = ('parent' if scope['program_hash'] == trial['parent_hash'] else
                'candidate' if scope['program_hash'] == trial['candidate_hash'] else None)
        if side:
            key = (kind, scope['task_id'], observation['pointer'])
            groups.setdefault(key, {}).setdefault(side, []).append(observation['value'])
    return [dict(kind=key[0], task_id=key[1], pointer=key[2]) for key, pair in groups.items()
            if 'parent' in pair and 'candidate' in pair and pair['parent'] != pair['candidate']]

def resource_coordinate(observation, trial, current_facts):
    """Comparable recorded resource units, never missing data as zero."""
    if observation['evidence_id'] != 'current:facts' or observation['scope']['phase'] != 'current_exploration':
        return None
    pointer, scope = observation['pointer'], observation['scope']
    for side in ('parent', 'candidate'):
        if scope['program_hash'] != trial.get(side + '_hash'):
            continue
        prefix = '/programs/' + side + '/'
        if pointer == prefix + 'stored_tokens':
            return side, ('stored_tokens', None, 'whole_bank_tokens')
        match = re.fullmatch(re.escape(prefix) + r'reads/(\d+)/read_tokens', pointer)
        if match and current_facts:
            rows = current_facts.get('programs', {}).get(side, {}).get('reads', [])
            index = int(match.group(1))
            if index >= len(rows):
                return None
            row = rows[index]
            if (not row.get('unit') or row.get('task_id') is None
                    or row['task_id'] != scope.get('task_id')
                    or row.get('read_tokens') != observation['value']
                    or row.get('actual_trace_available') is not True
                    or row.get('missing_token_measurements', 0)):
                return None
            return side, ('read_tokens', row['task_id'], row['unit'])
    return None

def equal_utility_decision(trial, pairs, current_facts=None):
    """Canonical additional B branch; A is a separate, preserved superset branch.

    This predicate only consumes already validated normalized review evidence.
    The caller verifies the saved report/payload binding. A prospective caller
    supplies native pairs directly from the same current comparison.
    """
    reasons, advantages = [], []
    review = trial.get('post_review') or {}
    report = review.get('report') or {}
    contract = report.get('evidence_contract') or {}
    assessment = report.get('assessment') or {}
    comparison = trial.get('exploration') or {}
    if (trial.get('mechanism_validated') is not True or trial.get('error')
            or trial.get('review_error') or not trial.get('candidate_hash')):
        reasons.append('invalid_or_unvalidated_mechanism')
    if (comparison.get('complete') is not True or not pairs
            or [p.get('task_id') for p in pairs] != comparison.get('task_ids')):
        reasons.append('missing_complete_current_pairs')
    scores_valid = bool(pairs) and all(type(p.get(key)) in (int, float)
        and math.isfinite(p[key]) for p in pairs for key in ('parent_score', 'candidate_score'))
    if not scores_valid:
        reasons.append('missing_finite_per_task_scores')
    elif any(p['parent_score'] != p['candidate_score'] for p in pairs):
        reasons.append('per_task_scores_not_identical')
    else:
        mean = sum(p['candidate_score'] for p in pairs) / len(pairs)
        if mean <= 0:
            reasons.append('zero_or_negative_mean')
        if comparison.get('candidate') != mean or comparison.get('parent') != mean or comparison.get('delta') != 0:
            reasons.append('aggregate_not_bound_to_equal_pairs')
    if (review.get('ok') is not True or review.get('stage') != 'post'
            or assessment.get('subject') != 'candidate' or assessment.get('verdict') != 'positive'
            or not contract):
        reasons.append('no_valid_positive_current_candidate_review')
    diagnoses = contract.get('diagnostic_assessments', [])
    defects = [d['group_id'] for d in diagnoses
               if d.get('subject') == 'candidate' and d.get('judgment') == 'supported_defect']
    if defects:
        reasons.append('confirmed_current_candidate_functional_defect')
    observations = {o['id']: o for o in contract.get('observations', [])}
    evaluations = contract.get('evaluations', [])
    for evaluation in evaluations:
        if evaluation['polarity'] != 'advantage':
            continue
        changed = changed_cited_content(evaluation, observations, trial)
        # An explicit content difference bound to a qualitative non-efficiency
        # advantage is the available normalized utility statement. Mere source
        # novelty, raw length and a private Reviewer narrative cannot qualify.
        if evaluation['dimension'] != 'Efficiency' and changed:
            advantages.append(dict(evaluation_id=evaluation['id'], dimension=evaluation['dimension'],
                                   basis='bound_content_change_and_quality_advantage', evidence=changed))
        if evaluation['dimension'] != 'Efficiency':
            continue
        protected = {e['dimension']: e for e in evaluations
                     if e['dimension'] in ('Faithfulness', 'Relevance', 'Actionability')}
        if set(protected) != {'Faithfulness', 'Relevance', 'Actionability'}:
            continue
        if any(e['polarity'] not in ('unchanged', 'advantage') or
               set(e.get('uncertainty', [])) - {'none_observed'} for e in protected.values()):
            continue
        metrics = {}
        for oid in evaluation['observation_ids']:
            observation = observations[oid]
            if type(observation['value']) not in (int, float):
                continue
            coordinate = resource_coordinate(observation, trial, current_facts)
            if coordinate:
                side, key = coordinate
                metrics.setdefault(key, {})[side] = observation['value']
        gains = [dict(metric=key[0], task_id=key[1], unit=key[2], **values) for key, values in metrics.items()
                 if set(values) == {'parent', 'candidate'} and values['candidate'] < values['parent']]
        if gains and all(changed_cited_content(e, observations, trial) or
                         any(content_kind(observations[oid]) for oid in e['observation_ids'])
                         for e in protected.values()):
            advantages.append(dict(evaluation_id=evaluation['id'], dimension='Efficiency',
                                   basis='measured_saving_with_bound_preserved_quality', evidence=gains))
    if not advantages:
        reasons.append('no_bound_specific_utility_advantage')
    return dict(eligible=not reasons, reasons=reasons, advantages=advantages,
                confirmed_candidate_defects=defects,
                defect_scope='displayed current diagnostic groups only; absence is not proof of global correctness')
