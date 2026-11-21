"""Synthetic CPU contracts for local model capabilities; no model or HTTP calls.

Real small-model loading/shape evidence is kept in the separate v3 diagnostics
probe. These tests use tiny manifest files and fake backends, not task evidence.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evomem.common import digest
from phase1_codegen import capabilities
from phase1_codegen.capabilities import GENERATION_MODEL, LocalModelCapabilities
from phase1_codegen.policy import CapabilityInfrastructureError, Rejected


EMBEDDING_ID = 'synthetic-embedding'
ENDPOINT = 'http://127.0.0.1:8140/v1'


class TinyTokenizer:
    def __init__(self, count=20):
        self.count = count
        self.seen = []

    def chat_count(self, messages):
        self.seen.append(copy.deepcopy(messages))
        return self.count


class FakeEmbedding:
    max_seq_length = 8

    def __init__(self, lengths=None, vectors=None):
        self.lengths = lengths
        self.vectors = vectors
        self.tokenizer_calls = []
        self.encode_calls = []

    def tokenizer(self, texts, **kwargs):
        self.tokenizer_calls.append((copy.deepcopy(texts), kwargs))
        lengths = self.lengths or [len(text.split()) + 2 for text in texts]
        return {'input_ids': [[1] * n for n in lengths]}

    def encode(self, texts, **kwargs):
        self.encode_calls.append((copy.deepcopy(texts), kwargs))
        return self.vectors if self.vectors is not None else [[0.6, 0.8, 0.0] for _ in texts]


class FakeResponse:
    def __init__(self, content='Synthetic model text.', finish='stop'):
        self.value = {'choices': [{'message': {'content': content}, 'finish_reason': finish}],
                      'usage': {'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24}}

    def raise_for_status(self):
        return None

    def json(self):
        return copy.deepcopy(self.value)


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.requests = []
        self.closed = False

    def post(self, url, *, json):
        self.requests.append((url, copy.deepcopy(json)))
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-capability-tests-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{"synthetic":true}\n')
        (model / 'model.safetensors').write_bytes(b'synthetic manifest fixture; not weights')
        self.model_dir = model
        self.catalog = {EMBEDDING_ID: (model, 3, 8)}
        self.patch = patch.object(capabilities, 'EMBEDDING_MODELS', self.catalog)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.multimodal_path_patch = patch.object(capabilities, 'MULTIMODAL_PATH', model)
        self.multimodal_path_patch.start()
        self.addCleanup(self.multimodal_path_patch.stop)
        self.multimodal_runtime_patch = patch.object(
            capabilities, 'multimodal_runtime_manifest',
            return_value={'schema': 1, 'status': 'synthetic', 'fingerprint': 'synthetic-runtime'}, create=True)
        self.multimodal_runtime_patch.start()
        self.addCleanup(self.multimodal_runtime_patch.stop)
        self.tokenizer = TinyTokenizer()

    def broker(self, environment='alfworld', *, root=None, ledger=None, endpoint=ENDPOINT):
        base = root or self.root / environment
        value = LocalModelCapabilities(base, environment, ledger or base / 'calls.jsonl',
                                       self.tokenizer, 'synthetic-generation-checkpoint', endpoint,
                                       phase='synthetic-contract-test')
        self.addCleanup(value.close)
        return value

    def rows(self, broker):
        if not broker.ledger.exists():
            return []
        return [json.loads(line) for line in broker.ledger.read_text().splitlines()]

    def install_embedding(self, broker, **kwargs):
        model = FakeEmbedding(**kwargs)
        broker.models[EMBEDDING_ID] = model
        return model

    def test_manifest_freezes_local_files_and_returns_detached_operational_catalog(self):
        broker = self.broker()
        description = broker.describe()
        self.assertEqual(description['fingerprint'], broker.fingerprint)
        embedding = description['models'][0]
        self.assertEqual(embedding['dimensions'], 3)
        self.assertEqual(embedding['max_input_tokens_per_text'], 8)
        self.assertNotIn('model_files', embedding)
        description['models'][0]['dimensions'] = 99
        self.assertEqual(broker.describe()['models'][0]['dimensions'], 3)
        (self.model_dir / 'config.json').write_text('{"synthetic":"changed"}\n')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            self.broker()

    def test_unknown_identifiers_paths_urls_and_nonlocal_endpoints_cannot_call_backends(self):
        broker = self.broker()
        with patch.object(broker, '_embedding') as embedding, patch.object(broker, '_generation') as generation:
            for model_id in ('unknown', str(self.model_dir), '/etc/passwd', '../model',
                             'http://example.org/model', 'Qwen3-Embedding-8B'):
                with self.subTest(model_id=model_id), self.assertRaises(Rejected):
                    broker.call(model_id, {'texts': ['generic text']}, operation='event')
            embedding.assert_not_called()
            generation.assert_not_called()
        for endpoint in ('https://127.0.0.1:8140/v1', 'http://example.org:8140/v1',
                         'http://localhost:8140/v1', 'http://127.0.0.1:8141/v1',
                         'http://127.0.0.1:8140/v1?url=other', 'file:///tmp/model'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                self.broker(endpoint=endpoint)
        with self.assertRaises(ValueError):
            self.broker(environment='undeclared_dataset')

    def test_exact_payload_schema_and_finite_json_reject_before_backend(self):
        broker = self.broker()
        invalid = [(EMBEDDING_ID, {'texts': []}), (EMBEDDING_ID, {'texts': ['']}),
                   (EMBEDDING_ID, {'texts': [' ']}), (EMBEDDING_ID, {'texts': ['x'] * 9}),
                   (EMBEDDING_ID, {'texts': 'x'}), (EMBEDDING_ID, {'texts': [4]}),
                   (EMBEDDING_ID, {'texts': ['x'], 'path': '/tmp/model'}),
                   (EMBEDDING_ID, {'texts': ['x' * 8193]}),
                   (EMBEDDING_ID, {'texts': ['x'], 'extra': float('nan')}),
                   (GENERATION_MODEL, {'prompt': 'x', 'max_tokens': True}),
                   (GENERATION_MODEL, {'prompt': 'x', 'max_tokens': 0}),
                   (GENERATION_MODEL, {'prompt': 'x', 'max_tokens': 257}),
                   (GENERATION_MODEL, {'prompt': 'x', 'temperature': 1}),
                   (GENERATION_MODEL, {'prompt': ''}), (GENERATION_MODEL, {'prompt': 'x' * 8193}),
                   (GENERATION_MODEL, {'prompt': float('inf')})]
        with patch.object(broker, '_embedding') as embedding, patch.object(broker, '_generation') as generation:
            for model_id, payload in invalid:
                with self.subTest(model_id=model_id, payload=repr(payload)[:100]), self.assertRaises(ValueError):
                    broker.call(model_id, payload, operation='event')
            with self.assertRaises(ValueError):
                broker.call(EMBEDDING_ID, {'texts': ['x']}, operation='outside-memory')
            embedding.assert_not_called()
            generation.assert_not_called()
        self.assertEqual(list((broker.root / 'cache').glob('*.json')), [])

    def test_non_event_operations_are_rejected_before_any_model_backend(self):
        broker = self.broker()
        with patch.object(broker, '_embedding') as embedding, \
                patch.object(broker, '_generation') as generation, \
                patch.object(broker, '_multimodal') as multimodal:
            for operation in ('read', 'write', 'outside-memory', '', None):
                with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'on_event'):
                    broker.call(EMBEDDING_ID, {'texts': ['plain text']}, operation=operation)
            embedding.assert_not_called()
            generation.assert_not_called()
            multimodal.assert_not_called()
        self.assertEqual(self.rows(broker), [])
        self.assertEqual(list((broker.root / 'cache').glob('*.json')), [])

    def test_embedding_exact_texts_shape_and_no_implicit_prefix(self):
        broker = self.broker()
        fake = self.install_embedding(broker)
        payload = {'texts': ['plain first text', 'plain second text']}
        original = copy.deepcopy(payload)
        result = broker.call(EMBEDDING_ID, payload, operation='event')
        self.assertEqual(set(result), {'value', 'receipt'})
        self.assertEqual(result['value'], {'embeddings': [[0.6, 0.8, 0.0]] * 2,
                                           'input_tokens': [5, 5]})
        self.assertEqual(payload, original)
        self.assertEqual(fake.tokenizer_calls, [(payload['texts'], {'truncation': False,
                                                                   'add_special_tokens': True})])
        self.assertEqual(fake.encode_calls[0][0], payload['texts'])
        self.assertTrue(fake.encode_calls[0][1]['normalize_embeddings'])
        self.assertFalse(result['receipt']['cache_hit'])
        self.assertTrue(result['receipt']['physical_attempt'])
        self.assertTrue(result['receipt']['inference_attempted'])
        self.assertTrue(result['receipt']['physical'])
        self.assertEqual(result['receipt']['usage']['input_tokens'], 10)

    def test_same_environment_exact_request_reuses_cache_across_broker_instances(self):
        first = self.broker()
        fake = self.install_embedding(first)
        payload = {'texts': ['plain text']}
        a = first.call(EMBEDDING_ID, payload, operation='event')
        second = self.broker(ledger=self.root / 'new-round.jsonl')
        with patch.object(second, '_embedding', side_effect=AssertionError('must reuse cache')) as backend:
            b = second.call(EMBEDDING_ID, payload, operation='event')
        backend.assert_not_called()
        self.assertEqual(len(fake.encode_calls), 1)
        self.assertEqual(a['value'], b['value'])
        self.assertEqual(a['receipt']['cache_key'], b['receipt']['cache_key'])
        self.assertTrue(b['receipt']['cache_hit'])
        self.assertFalse(b['receipt']['physical_attempt'])
        self.assertFalse(b['receipt']['inference_attempted'])
        self.assertFalse(b['receipt']['physical'])
        self.assertEqual(b['receipt']['usage'], {})
        self.assertEqual(b['receipt']['operation'], 'event')
        b['value']['embeddings'][0][0] = -99
        c = first.call(EMBEDDING_ID, payload, operation='event')
        self.assertEqual(c['value']['embeddings'][0][0], 0.6)

    def test_different_dataset_cache_and_fingerprint_are_isolated(self):
        alf, tw = self.broker('alfworld'), self.broker('textworld')
        models = [self.install_embedding(b) for b in (alf, tw)]
        responses = [b.call(EMBEDDING_ID, {'texts': ['same neutral text']}, operation='event')
                     for b in (alf, tw)]
        self.assertNotEqual(alf.fingerprint, tw.fingerprint)
        self.assertNotEqual(responses[0]['receipt']['cache_key'], responses[1]['receipt']['cache_key'])
        self.assertTrue(all(len(m.encode_calls) == 1 for m in models))
        self.assertTrue(all(not r['receipt']['cache_hit'] for r in responses))
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            self.broker('textworld', root=alf.root)

    def test_modified_cached_value_is_rejected_without_backend_retry(self):
        broker = self.broker()
        self.install_embedding(broker)
        payload = {'texts': ['neutral text']}
        result = broker.call(EMBEDDING_ID, payload, operation='event')
        path = Path(result['receipt']['cache_path'])
        cached = json.loads(path.read_text())
        cached['value']['embeddings'][0][0] = -1
        path.write_text(json.dumps(cached))
        with patch.object(broker, '_embedding') as backend, self.assertRaisesRegex(ValueError, 'cache receipt'):
            broker.call(EMBEDDING_ID, payload, operation='event')
        backend.assert_not_called()

    def test_self_consistent_but_wrong_cache_binding_is_rejected(self):
        for field, replacement in [('model_id', 'different'), ('environment', 'textworld'),
                                   ('capability_fingerprint', 'wrong'), ('request', {'texts': ['other']})]:
            with self.subTest(field=field):
                broker = self.broker(root=self.root / field)
                self.install_embedding(broker)
                payload = {'texts': ['neutral text']}
                result = broker.call(EMBEDDING_ID, payload, operation='event')
                path = Path(result['receipt']['cache_path'])
                cached = json.loads(path.read_text())
                cached[field] = replacement
                cached['record_hash'] = digest({k: v for k, v in cached.items() if k != 'record_hash'})
                path.write_text(json.dumps(cached))
                with patch.object(broker, '_embedding') as backend, self.assertRaises(ValueError):
                    broker.call(EMBEDDING_ID, payload, operation='event')
                backend.assert_not_called()

    def test_infrastructure_failure_is_archived_and_does_not_poison_the_request_cache(self):
        broker = self.broker()
        payload = {'texts': ['neutral text']}
        with patch.object(broker, '_embedding', side_effect=RuntimeError('synthetic backend failure')) as backend:
            with self.assertRaisesRegex(CapabilityInfrastructureError, 'synthetic backend failure'):
                broker.call(EMBEDDING_ID, payload, operation='event')
            backend.assert_called_once()
        failed, = self.rows(broker)
        self.assertEqual(failed['failure_kind'], 'infrastructure')
        self.assertEqual(Path(failed['cache_path']).parent, broker.root/'failures')
        self.assertTrue(Path(failed['cache_path']).is_file())
        self.assertFalse(failed['physical'])
        self.install_embedding(broker)
        later = broker.call(EMBEDDING_ID, payload, operation='event')
        self.assertTrue(later['receipt']['ok'])
        self.assertFalse(later['receipt']['cache_hit'])

    def test_embedding_token_limit_rejects_whole_input_without_truncation_or_encode(self):
        broker = self.broker()
        fake = self.install_embedding(broker, lengths=[9])
        for _ in range(2):
            with self.assertRaisesRegex(Rejected, 'no truncation'):
                broker.call(EMBEDDING_ID, {'texts': ['short characters but nine tokens']}, operation='event')
        self.assertEqual(len(fake.tokenizer_calls), 1)
        self.assertEqual(fake.tokenizer_calls[0][1]['truncation'], False)
        self.assertEqual(fake.encode_calls, [])
        self.assertTrue(all(r['physical'] is False for r in self.rows(broker)))
        self.assertTrue(all(r['inference_attempted'] is False for r in self.rows(broker)))

    def test_generation_token_limit_rejects_without_http_and_without_hidden_retry(self):
        self.tokenizer.count = capabilities.LIMITS['max_generation_input_tokens'] + 1
        broker = self.broker()
        client = FakeClient()
        broker.client = client
        for _ in range(2):
            with self.assertRaisesRegex(Rejected, '2048-token budget'):
                broker.call(GENERATION_MODEL, {'prompt': 'bounded characters'}, operation='event')
        self.assertEqual(client.requests, [])
        self.assertEqual(len(self.tokenizer.seen), 1)
        self.assertTrue(all(r['physical'] is False for r in self.rows(broker)))
        self.assertTrue(all(r['inference_attempted'] is False for r in self.rows(broker)))

    def test_failed_http_request_is_an_inference_attempt_but_not_completed_inference(self):
        broker = self.broker()
        broker.client = FakeClient(error=RuntimeError('synthetic HTTP failure'))
        for _ in range(2):
            with self.assertRaisesRegex(CapabilityInfrastructureError, 'synthetic HTTP failure'):
                broker.call(GENERATION_MODEL, {'prompt': 'Neutral request.'}, operation='event')
        self.assertEqual(len(broker.client.requests), 2)
        rows = self.rows(broker)
        self.assertEqual([r['inference_attempted'] for r in rows], [True, True])
        self.assertTrue(all(r['physical'] is False for r in rows))

    def test_generation_exact_fixed_request_and_receipt_do_not_expose_response(self):
        broker = self.broker()
        client = FakeClient(FakeResponse(content='Synthetic response marker.', finish='length'))
        broker.client = client
        result = broker.call(GENERATION_MODEL, {'prompt': 'Exact prompt marker.', 'max_tokens': 7}, operation='event')
        self.assertEqual(result['value'], {'text': 'Synthetic response marker.', 'truncated': True})
        url, request = client.requests[0]
        self.assertEqual(url, ENDPOINT + '/chat/completions')
        self.assertEqual(request['messages'], self.tokenizer.seen[0])
        self.assertEqual(request['messages'][1]['content'], 'Exact prompt marker.')
        self.assertEqual(request['temperature'], 0)
        self.assertEqual(request['seed'], 42)
        self.assertEqual(request['max_tokens'], 7)
        self.assertEqual(request['chat_template_kwargs'], {'enable_thinking': False})
        receipt_text = json.dumps(result['receipt'])
        self.assertNotIn('Exact prompt marker.', receipt_text)
        self.assertNotIn('Synthetic response marker.', receipt_text)
        self.assertNotIn('request', result['receipt']['usage'])
        self.assertNotIn('response', result['receipt']['usage'])
        broker.call(GENERATION_MODEL, {'prompt': 'Exact prompt marker.', 'max_tokens': 7}, operation='event')
        self.assertEqual(len(client.requests), 1)

    def test_default_generation_output_budget_normalizes_to_same_cache_request(self):
        broker = self.broker()
        broker.client = FakeClient()
        a = broker.call(GENERATION_MODEL, {'prompt': 'Plain text.'}, operation='event')
        b = broker.call(GENERATION_MODEL, {'prompt': 'Plain text.', 'max_tokens': 256}, operation='event')
        self.assertEqual(a['receipt']['cache_key'], b['receipt']['cache_key'])
        self.assertTrue(b['receipt']['cache_hit'])
        self.assertEqual(len(broker.client.requests), 1)

    def test_usage_receipt_is_finite_and_excludes_arbitrary_backend_content(self):
        broker = self.broker(root=self.root / 'extra-usage')
        response = FakeResponse()
        response.value['usage']['arbitrary_content'] = 'Must not enter capability receipt.'
        broker.client = FakeClient(response)
        result = broker.call(GENERATION_MODEL, {'prompt': 'Plain text.'}, operation='event')
        self.assertNotIn('arbitrary_content', result['receipt']['usage'])
        self.assertNotIn('Must not enter capability receipt.', json.dumps(result['receipt']))
        for value in (float('nan'), float('inf'), -1, True, '20'):
            with self.subTest(value=value):
                invalid = self.broker(root=self.root / ('invalid-usage-' + str(value)))
                response = FakeResponse()
                response.value['usage']['prompt_tokens'] = value
                invalid.client = FakeClient(response)
                for _ in range(2):
                    with self.assertRaises(CapabilityInfrastructureError):
                        invalid.call(GENERATION_MODEL, {'prompt': 'Plain text.'}, operation='event')
                self.assertEqual(len(invalid.client.requests), 2)
                for row in self.rows(invalid):
                    json.dumps(row, allow_nan=False)

    def test_nonfinite_wrong_shape_or_non_json_backend_output_is_never_returned(self):
        for label, vectors in [('nonfinite', [[float('nan'), 0.0, 1.0]]),
                               ('wrong_dim', [[1.0, 0.0]]), ('wrong_rows', [])]:
            with self.subTest(label=label):
                broker = self.broker(root=self.root / label)
                fake = self.install_embedding(broker, vectors=vectors)
                for _ in range(2):
                    with self.assertRaises(CapabilityInfrastructureError):
                        broker.call(EMBEDDING_ID, {'texts': ['plain text']}, operation='event')
                self.assertEqual(len(fake.encode_calls), 2)
        broker = self.broker(root=self.root / 'nonjson')
        with patch.object(broker, '_embedding', return_value=({'value': float('inf')}, {})) as backend:
            for _ in range(2):
                with self.assertRaises(CapabilityInfrastructureError):
                    broker.call(EMBEDDING_ID, {'texts': ['plain text']}, operation='event')
            self.assertEqual(backend.call_count, 2)


if __name__ == '__main__':
    unittest.main()
