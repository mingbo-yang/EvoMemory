"""Fixed task-input correction for the pinned ALMA TextWorld adapter.

Only the staged environment copy changes. The official prompt formatter, Actor,
action loop, parsing, budgets, native state transitions and rewards stay intact.
This is not a candidate-code transformation or an evolution-time repair.
"""
from __future__ import annotations

from evomem.common import digest
from .policy import Rejected


TASK_INPUT_VERSION = 'textworld-native-objective-v1'


def staged_environment_source(source):
    """Apply exactly the two audited input corrections; reject upstream drift."""
    corrections = (
        ('            memory_retrived=memory_retrived,\n            **kargs)',
         '            memory_retrived=memory_retrived,\n            goal=goal,\n            **kargs)'),
        ('    def filter_objective(self, obs, info):\n'
         '        objective = info["objective"]\n'
         '        parts = obs.split(objective)\n'
         '        if len(parts) == 1:\n'
         '            return parts[0].strip()\n'
         '        else:\n'
         '            return parts[-1].strip()',
         '    def filter_objective(self, obs, info):\n'
         '        # Preserve all native visible text, including the goal command.\n'
         '        return obs.strip()'),
    )
    for before, after in corrections:
        if source.count(before) != 1:
            raise Rejected('pinned TextWorld task-input correction no longer matches official source')
        source = source.replace(before, after, 1)
    source = '# Staged task-input correction: ' + TASK_INPUT_VERSION + '\n' + source
    compile(source, '<staged TextWorld environment>', 'exec')
    return source


def goal_delivery_receipt(messages, goal, instruction, request_hash):
    """Verify the native goal at its direct prompt position, before any memory.

    The pinned official formatter constructs ``instruction.strip() + goal``
    followed by its optional memory section. A goal mentioned in past memory or
    in a later user turn therefore cannot satisfy this check.
    """
    prefix = instruction.strip()
    system = (messages[0].get('content', '') if messages
              and messages[0].get('role') == 'system' else '')
    goal_valid = isinstance(goal, str) and bool(goal.strip())
    expected = prefix + goal if goal_valid else prefix
    verified = bool(goal_valid and isinstance(system, str)
                    and system.startswith(expected))
    return dict(version=TASK_INPUT_VERSION, source='native_reset.objective',
                goal_hash=digest(goal), system_hash=digest(system),
                expected_prefix_hash=digest(expected),
                goal_character_range=[len(prefix), len(expected)] if goal_valid else None,
                direct_goal_verified=verified, request_hash=request_hash)


def summarize_goal_delivery(actor_rows, goal):
    """Count transport attempts from the episode ledger; SDK retries are distinct.

    A server response proves dispatch. A connection error without a response
    leaves delivery unknown. A valid constructed prompt proves neither.
    """
    goal_hash = digest(goal)
    counts = dict(constructed_requests=0, constructed_goals_verified=0,
                  local_rejections=0, network_dispatch_attempts=0,
                  server_responses=0, delivery_unknown=0, successful_actor_responses=0,
                  direct_goal_verified_requests=0, unclassified_requests=0)
    evidence = []
    for row in actor_rows:
        receipt = row.get('task_input_receipt') or {}
        request_hash = row.get('request_hash')
        messages = row.get('messages') or []
        first = messages[0] if messages else {}
        system = first.get('content') if first.get('role') == 'system' else None
        span = receipt.get('goal_character_range')
        verified = bool(isinstance(goal, str) and goal.strip() and request_hash
                        and receipt.get('request_hash') == request_hash
                        and receipt.get('goal_hash') == goal_hash
                        and isinstance(system, str) and digest(system) == receipt.get('system_hash')
                        and isinstance(span, list) and len(span) == 2
                        and all(type(n) is int for n in span) and 0 <= span[0] <= span[1] <= len(system)
                        and system[span[0]:span[1]] == goal
                        and digest(system[:span[1]]) == receipt.get('expected_prefix_hash')
                        and receipt.get('direct_goal_verified') is True)
        counts['constructed_requests'] += 1
        counts['constructed_goals_verified'] += int(verified)
        kind = row.get('kind')
        if kind == 'model':
            counts['network_dispatch_attempts'] += 1
            # Successful archived rows and explicit historical HTTP errors
            # imply a response even when they predate the http_status field.
            status = row.get('http_status')
            error = row.get('error')
            legacy_http = (isinstance(error, str) and error.startswith('HTTP')
                           and len(error) == 7 and error[4:].isdigit())
            responded = status is not None or row.get('ok') is True or legacy_http
            counts['server_responses' if responded else 'delivery_unknown'] += 1
            successful = row.get('ok') is True
            counts['successful_actor_responses'] += int(successful)
            counts['direct_goal_verified_requests'] += int(successful and verified)
        elif kind in ('local_input_rejection', 'task_input_error') and row.get('network_request_sent') is False:
            counts['local_rejections'] += 1
        else:
            counts['unclassified_requests'] += 1
        evidence.append([request_hash, kind, row.get('ok'), row.get('error'),
                         row.get('http_status'), row.get('network_request_sent'), receipt])
    return dict(version='textworld-goal-delivery-v2', input_version=TASK_INPUT_VERSION,
                source='native_reset.objective', goal_hash=goal_hash, **counts,
                actor_requests=counts['network_dispatch_attempts'],
                all_actor_requests_verified=bool(actor_rows) and
                    counts['direct_goal_verified_requests'] == counts['constructed_requests'],
                request_receipts_hash=digest(evidence))
