"""CPU-only counting contracts and comparison with archived real server usage."""
import json
from importlib.util import find_spec
import types
import unittest

import numpy as np

from evomem.common import ROOT, MODEL
from evomem.services import Tokenizer as HistoricalTokenizer
from phase1_codegen.tokenization import Tokenizer, _single_sequence_length


class TokenSequenceTests(unittest.TestCase):
    def test_gemma_service_renders_text_as_openai_content_parts(self):
        rendered = []
        def render(messages, **kwargs):
            rendered.append(messages)
            return [1, 2]

        tokenizer = Tokenizer.__new__(Tokenizer)
        tokenizer.impl = types.SimpleNamespace(apply_chat_template=render)
        original = [{'role': 'system', 'content': 'Choose one action.'},
                    {'role': 'user', 'content': [{'type': 'text', 'text': 'A room.'}]}]
        tokenizer.model_name = 'gemma4-31b'
        tokenizer.chat_count(original)
        self.assertEqual(rendered[-1], [{'role': 'system', 'content': [{'type': 'text', 'text': 'Choose one action.'}]},
                                        original[1]])
        tokenizer.model_name = 'qwen3.8-27b-local'
        tokenizer.chat_count(original)
        self.assertEqual(rendered[-1][0]['content'], 'Choose one action.')

    def test_mapping_list_and_numpy_count_tokens_not_fields(self):
        for value in ([7, 8, 9], [[7, 8, 9]],
                      {'input_ids': [7, 8, 9], 'attention_mask': [1, 1, 1]},
                      np.array([7, 8, 9]), np.array([[7, 8, 9]])):
            with self.subTest(type=type(value)):
                self.assertEqual(_single_sequence_length(value), 3)

    @unittest.skipUnless(find_spec('transformers'), 'Optional transformers dependency required')
    def test_transformers_batch_encoding_counts_tokens_not_fields(self):
        from transformers import BatchEncoding
        value = BatchEncoding({'input_ids': [7, 8, 9], 'attention_mask': [1, 1, 1]})
        self.assertEqual(_single_sequence_length(value), 3)

    @unittest.skipUnless(find_spec('torch'), 'Optional torch dependency required')
    def test_cpu_tensor_counts_tokens_not_fields(self):
        import torch
        for value in (torch.tensor([7, 8, 9], device='cpu'),
                      {'input_ids': torch.tensor([[7, 8, 9]], device='cpu')}):
            with self.subTest(type=type(value)):
                self.assertEqual(_single_sequence_length(value), 3)

    def test_empty_batched_and_malformed_sequences_are_rejected(self):
        for value in ({'attention_mask': [1, 1]}, [], [[]], [[1], [2]],
                      [[[1]]], '123', [True], [-1], [1.5],
                      np.array([[1, 2], [3, 4]]), np.array(1)):
            with self.subTest(value=str(value)), self.assertRaises((TypeError, ValueError)):
                _single_sequence_length(value)

    def test_return_dict_disabled_when_supported_and_legacy_signature_supported(self):
        requests = []
        def modern(messages, **kwargs):
            requests.append(kwargs)
            return {'input_ids': [1, 2, 3, 4], 'attention_mask': [1, 1, 1, 1]}
        def legacy(messages, tokenize, add_generation_prompt, enable_thinking):
            return [1, 2, 3, 4]
        tokenizer = Tokenizer.__new__(Tokenizer)
        tokenizer.impl = types.SimpleNamespace(apply_chat_template=modern)
        self.assertEqual(tokenizer.chat_count([{'role': 'user', 'content': 'test'}]), 4)
        self.assertIs(requests[0]['return_dict'], False)
        tokenizer.impl = types.SimpleNamespace(apply_chat_template=legacy)
        self.assertEqual(tokenizer.chat_count([{'role': 'user', 'content': 'test'}]), 4)
        for messages in ([], [[{'role': 'user', 'content': 'batched'}]], 'not a conversation'):
            with self.assertRaises(ValueError):
                tokenizer.chat_count(messages)


@unittest.skipUnless((MODEL/'config.json').is_file(), 'Separately installed real tokenizer checkpoint required')
@unittest.skipUnless(find_spec('transformers'), 'Optional transformers dependency required')
class RealLocalTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer()

    def test_real_template_ids_count_and_text_semantics_are_unchanged(self):
        messages = [{'role': 'system', 'content': 'Use exact task observations.'},
                    {'role': 'user', 'content': '检查真实本地 tokenizer 的聊天模板长度。'}]
        ids = self.tokenizer.impl.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        actual = _single_sequence_length(ids)
        self.assertGreater(actual, 2)
        self.assertEqual(self.tokenizer.chat_count(messages), actual)
        self.assertIs(Tokenizer.count, HistoricalTokenizer.count)
        self.assertIs(Tokenizer.clip, HistoricalTokenizer.clip)
        text = 'Count and clip retain the original tokenization semantics. ' * 15
        self.assertEqual(self.tokenizer.count(text), len(self.tokenizer.impl.encode(text, add_special_tokens=False)))
        self.assertEqual(self.tokenizer.clip(text, 32), HistoricalTokenizer.clip(self.tokenizer, text, 32))

    def test_real_pre_review_counts_match_both_archived_server_usage_values(self):
        expected = {'alfworld': 24132, 'textworld': 25078}
        for environment, count in expected.items():
            path = ROOT / 'runs/two_dataset_shared_memory_20261106/source' / environment / 'reviewer_costs.jsonl'
            if not path.exists():
                self.skipTest('Archived real reviewer receipts are not present in this checkout')
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            pre = next(record for record in records if record['stage'] == 'pre')
            with self.subTest(environment=environment):
                self.assertEqual(pre['usage']['prompt_tokens'], count)
                self.assertEqual(pre['local_prompt_tokens'], 2)  # Preserve evidence of the historical bug.
                self.assertEqual(self.tokenizer.chat_count(pre['request']['messages']), count)


if __name__ == '__main__':
    unittest.main()
