"""Correct chat-template token counts without changing frozen text tokenization.

Some Transformers versions return BatchEncoding by default. Its length counts
fields, not tokens. Count the single input_ids sequence explicitly regardless
of whether the tokenizer returns a mapping, list, or CPU tensor/array.
"""
from collections.abc import Mapping, Sequence
import inspect
from numbers import Integral

from evomem.services import Tokenizer as HistoricalTokenizer


def _single_sequence_length(value):
    if isinstance(value, Mapping):
        if 'input_ids' not in value:
            raise ValueError('chat template mapping is missing input_ids')
        value = value['input_ids']
    shape = getattr(value, 'shape', None)
    if shape is not None:
        shape = tuple(shape)
        if len(shape) not in (1, 2) or (len(shape) == 2 and shape[0] != 1):
            raise ValueError('chat template must produce exactly one token sequence')
        if not hasattr(value, 'tolist'):
            raise TypeError('tensor token IDs must provide tolist')
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError('chat template input_ids must be a token sequence')
    if value and isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes)):
        if len(value) != 1:
            raise ValueError('chat template returned more than one token sequence')
        value = value[0]
    if not value:
        raise ValueError('chat template returned an empty token sequence')
    if any(isinstance(token, bool) or not isinstance(token, Integral) or token < 0 for token in value):
        raise ValueError('chat template input_ids must contain nonnegative integer token IDs')
    length = len(value)
    if shape is not None and length != shape[-1]:
        raise ValueError('chat template tensor shape and token sequence length disagree')
    return length


class Tokenizer(HistoricalTokenizer):
    """Only chat_count differs; count and clip are inherited unchanged."""

    def chat_count(self, messages):
        if (isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence)
                or not messages or any(not isinstance(message, Mapping) for message in messages)):
            raise ValueError('chat_count requires one nonempty conversation, not a batch')
        method = self.impl.apply_chat_template
        kwargs = dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        if 'return_dict' in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD
                                               for p in parameters.values()):
            kwargs['return_dict'] = False
        return _single_sequence_length(method(messages, **kwargs))
