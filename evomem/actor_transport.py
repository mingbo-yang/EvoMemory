"""Shared Actor context preflight at the OpenAI-compatible transport boundary.

A local ValueError would be retried and wrapped as APIConnectionError by the
SDK. Returning the ordinary API error response preserves the context-limit
meaning for the official ALMA loop's existing termination branch.
"""
from __future__ import annotations

import httpx

from .common import append, digest


ACTOR_CONTEXT_TOKENS = 32768
ACTOR_RESERVED_TOKENS = 1024


def context_limit_response(request, body, *, tokenizer, context, ledger, model,
                           task_input_receipt=None):
    """Return None when admitted, or a local HTTP 400 without a network call."""
    tokens = tokenizer.chat_count(body['messages'])
    if tokens + ACTOR_RESERVED_TOKENS <= ACTOR_CONTEXT_TOKENS:
        return None
    context['model_errors'] += 1
    receipt = dict(kind='local_input_rejection', role='actor',
        phase=context['phase'], task_id=context['task']['id'],
        error='ContextLengthExceeded', ok=False, network_request_sent=False,
        estimated_prompt_tokens=tokens, context_limit=ACTOR_CONTEXT_TOKENS,
        reserved_tokens=ACTOR_RESERVED_TOKENS, request_hash=digest(body),
        model_name=model.model_name, model_fingerprint=model.fingerprint,
        seed=body['seed'], messages=body['messages'],
        original_message_hash=digest(body['messages']))
    if task_input_receipt is not None:
        receipt['task_input_receipt'] = task_input_receipt
    append(ledger, receipt)
    return httpx.Response(400, request=request, json={'error': {
        'type': 'invalid_request_error', 'code': 'context_length_exceeded',
        'param': 'messages',
        'message': ('maximum context length exceeded: '
                    f'{tokens} input tokens + {ACTOR_RESERVED_TOKENS} reserved '
                    f'> {ACTOR_CONTEXT_TOKENS}; rejected locally before sending')}})
