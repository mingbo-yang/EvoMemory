"""Shared contracts for the active memory workflow, with no runtime side effects.

The event ABI and workflow revision are separate: an implementation cleanup may
change the frozen workflow without changing externally visible event names.
Per-invocation sandbox limits remain owned by policy/program; HookLimits bounds
the whole episode and must not replace those smaller limits.
"""
from dataclasses import dataclass

WORKFLOW_VERSION = 'code-memory-hooks-v9'
HOOK_VERSION = 'memory-events-v1'
EVENTS = ('task_start', 'before_action', 'after_observation', 'task_end')
CAPACITY_TOKENS = 16384
READ_TOKENS = 2048
EPISODE_INJECTED_TOKENS = 8192
READ_TOP_K = 4
# Budgeted episode counters. Whole worker CPU is also recorded as actual cost;
# its trusted interpreter/bootstrap portion is not generated mechanism work.
HOOK_BUDGET_COUNTERS = ('model_calls', 'generated_cpu_seconds', 'worker_wall_seconds', 'injected_tokens')
# Meta-role evidence capacity is independent of the unchanged Executor context
# and output budgets. Reserve room for every task in a paired validation batch.
META_CONTEXT_TOKENS = 65536
REVIEW_INPUT_TOKENS = 56000
REVIEW_FEEDBACK_TOKENS = 7000
WRITE_FIELDS = ('task_id', 'group_id', 'room', 'family', 'environment', 'goal',
                'history', 'score', 'done', 'error', 'phase', 'seed')

# Describes the existing transport, not a second validator or a mechanism recipe.
# Model-facing instructions and evidence use this same description. Native field
# absence is meaningful: in particular ALFWorld need not provide a separate goal.
EVENT_CONTRACT = {
    'version': HOOK_VERSION,
    'entrypoint': 'on_event(state, event, context)',
    'inputs': {
        'state': 'JSON object; mechanism-owned persistent schema; candidate replay begins at {}, later calls receive the prior state',
        'event': {
            'always': {
                'type': {'type': 'string', 'enum': list(EVENTS)},
                'step': 'integer; number of completed environment transitions',
                'session': 'JSON object; this episode scratch state from the preceding response, initially {}',
                'budget_remaining': {
                    'model_calls': 'integer',
                    'generated_cpu_seconds': 'number; mechanism execution including output validation, excluding trusted worker bootstrap',
                    'worker_wall_seconds': 'number',
                    'injected_tokens': 'integer',
                },
            },
            'by_type': {
                'task_start': {'visible': 'JSON object; current externally visible fields only'},
                'before_action': {'visible': 'JSON object; latest externally visible fields only',
                                  'attempt': 'integer; zero-based Actor call index',
                                  'optional': {'previous_action_failure': {
                                      'attempt': 'integer; preceding Actor call whose native step raised',
                                      'action': 'string; action submitted on that attempt',
                                      'reason': 'invalid_action for an explicit native rejection, otherwise step_failed'}}},
                'after_observation': {'visible': 'JSON object; returned environment fields only',
                                      'attempt': 'integer; zero-based Actor call index',
                                      'action': 'string; action just executed'},
                'task_end': {'experience': 'JSON object; completed permitted native experience',
                             'optional': {'mode': 'history_import for a log without recorded events',
                                          'origin': 'history_import for a log without recorded events'}},
            },
            'visible_optional_fields': {
                'obs': 'string; observation, including task text when supplied by the environment',
                'goal': 'string; present only when the environment explicitly supplies it',
                'actions_list': 'list of strings; currently available actions if supplied',
                'long_term_context': 'string; public environment context if supplied',
                'short_term_context': 'string; public environment context if supplied',
                'action_dict': 'JSON object; public environment action descriptions if supplied',
                'scores': 'number; latest native feedback if supplied',
                'dones': 'boolean; episode termination feedback if supplied, not necessarily success',
            },
            'experience_fields': {
                'task_id': 'string', 'family': 'string', 'environment': 'string',
                'optional': {
                    'group_id': 'string', 'room': 'string or integer; dataset room identifier', 'goal': 'string',
                    'history': 'list of step objects; observation/action/next_observation strings; optional admissible list and listed_exactly boolean',
                    'score': 'number; native task reward',
                    'done': 'boolean; termination, not necessarily success',
                    'error': 'null for a completed permitted episode',
                    'phase': 'string', 'seed': 'integer',
                },
            },
        },
        'context': {
            'task_id': 'string; current task identity',
            'family': 'string; current task type',
            'environment': 'string; current dataset identity',
            'memory_hook_version': 'string; memory-events-v1',
            'optional': {
                'memory_scope': 'family or environment',
                'model_catalog': 'JSON object; available local model contracts and limits',
            },
            'meaning': 'Runtime metadata only. Current observations belong to event.visible, not context.visible/context.obs/context.observation/context.goal. Completed history belongs to event.experience. A visible field may be absent; absence must not be replaced with invented environment data.',
        },
    },
    'output': {
        'exact_fields': {
            'state': 'JSON object; proposed persistent state',
            'session': 'JSON object; episode scratch state',
            'memory': 'null or list of rows; every row has exactly id (nonempty unique string) and experience (string)',
        },
        'memory_effect': 'null preserves the slot; [] clears it; rows replace it subject to fixed bounds. task_end has no subsequent Actor call.',
    },
    'boundaries': 'This describes fields, not a memory policy. The mechanism decides its response. No hidden state, future observation or held-out result is supplied. Only validated learning episodes publish persistent state; evaluations discard temporary changes.',
}


ONE_TURN_EXECUTOR = 'ALMA.Agent.ask/one_turn_benchmark_v1'


def native_executor(environment):
    """Bind native measurement and committed replay to the actual Actor entry point."""
    if environment in {'math', 'apps'}:
        return ONE_TURN_EXECUTOR
    return 'ALMA.Agent_Workflow.run_single_task'


@dataclass(frozen=True)
class HookLimits:
    max_events: int = 162
    model_calls: int = 8
    generated_cpu_seconds: float = 20.
    worker_wall_seconds: float = 60.
    session_tokens: int = 2048
    # Charge each changed slot in full; this is independent of the single-slot cap.
    injected_tokens: int = EPISODE_INJECTED_TOKENS

    @classmethod
    def for_environment(cls, environment):
        return cls(max_events=62 if environment == 'alfworld' else 162)
