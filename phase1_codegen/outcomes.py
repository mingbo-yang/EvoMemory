"""Outcome views shared by measurement, review and historical readback."""
import json
import math


def reported_score(episode):
    """A failed or unverified Episode has no reported reward, including archived zeros."""
    score = episode.get('score', episode.get('reward'))
    if (episode.get('error') or (episode.get('synthetic') is not True and episode.get('native_reward_verified') is False)
            or (episode.get('synthetic') is False and episode.get('native_reward_verified') is not True)
            or type(score) not in (int, float) or not math.isfinite(score)):
        return None
    return score


def comparison_for_feedback(comparison):
    """Hide incomplete aggregate placeholders while preserving task and evidence bindings."""
    if comparison and comparison.get('complete') is False:
        return dict(comparison, parent=None, candidate=None, delta=None)
    return comparison


def candidate_hook_failure(episode, program_hash):
    """A checkable generated-code rejection; an unknown failure belongs to the host."""
    if not episode.get('error') or episode.get('infrastructure_failure') or episode.get('program_hash') != program_hash:
        return False
    rejected = [row for row in (episode.get('memory_hook') or {}).get('events', []) if row.get('status') == 'rejected']
    if len(rejected) != 1:
        return False
    row = rejected[0]
    if row.get('program_hash') != program_hash:
        return False
    if row.get('error_type') in {'GeneratedBudgetExceeded', 'CapabilityContractError'}:
        return True
    if row.get('error_type') != 'Rejected':
        return False
    message = row.get('error', '')
    if message in {
        'worker exceeded cumulative five-second wall budget after trusted startup',
        'worker exceeded cumulative five-second wall budget', 'worker exceeded cumulative one-second CPU budget',
        'cumulative Hook model-call budget exceeded', 'cumulative Hook generated_cpu_seconds budget exceeded',
        'cumulative Hook worker_wall_seconds budget exceeded', 'cumulative Hook memory injection budget exceeded',
        'Hook persistent state storage budget exceeded', 'Hook session storage budget exceeded',
    }:
        return True
    prefix = 'isolated worker rejected (2): '
    if not message.startswith(prefix):
        return False
    try:
        worker, _ = json.JSONDecoder().raw_decode(message[len(prefix):])
    except ValueError:
        return False
    if not isinstance(worker, dict):
        return False
    if worker.get('error') == 'TrustedTermination':
        return str(worker.get('message')).startswith(
            ('shared iteration budget exceeded', 'model_call budget exceeded', 'model capability contract:'))
    return worker.get('error') in {
        'Rejected', 'ValueError', 'TypeError', 'KeyError', 'IndexError', 'AttributeError', 'NameError',
        'UnboundLocalError', 'ZeroDivisionError', 'OverflowError', 'MemoryError',
    }
