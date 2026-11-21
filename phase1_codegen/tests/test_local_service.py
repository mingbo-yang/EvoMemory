"""CPU-only endpoint routing contracts; fake HTTP, no server or GPU operations."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evomem.common import MODEL, MODEL_NAME, file_hash, write
from phase1_codegen import local_service
from phase1_codegen.local_service import make_local_llm, validate_base_url, verify_local_service
from phase1_codegen.proposer import CodeEvolver, EVOLVER_PROMPT, PARAMETERS as EVOLVER_PARAMETERS
from phase1_codegen.review import (MechanismReviewer, REVIEW_PROMPT, PARAMETERS as REVIEW_PARAMETERS,
                                  bounded_review_payload, validate_report)
from phase1_codegen.tests.test_review import CharacterTokenizer, FakeClient, FakeResponse, payload, report


GPU0 = 'http://127.0.0.1:8140/v1'
GPU1 = 'http://127.0.0.1:8141/v1'
GPU2 = 'http://127.0.0.1:8142/v1'


class ModelsResponse:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        return None

    def json(self):
        return copy.deepcopy(self.value)


class ModelsClient:
    def __init__(self, value):
        self.value = value
        self.requests = []
        self.closed = False

    def get(self, url):
        self.requests.append(url)
        return ModelsResponse(self.value)

    def close(self):
        self.closed = True


class LocalServiceTests(unittest.TestCase):
    def test_explicit_small_model_identity_is_checked_and_used_in_requests(self):
        name, path = 'qwen3.5-4b-local', Path('/local/small-checkpoint')
        client = ModelsClient({'data': [{'id': name, 'root': str(path)}]})
        receipt = verify_local_service(GPU2, client=client, model_name=name, model_path=path)
        self.assertEqual(receipt['model_name'], name)
        with self.assertRaisesRegex(ValueError, 'explicit checkpoint'):
            verify_local_service(GPU2, client=client, model_name=name)
        proposal = dict(source='synthetic source', hypothesis='Synthetic test.', risk='No task evidence.')
        client = FakeClient([FakeResponse(proposal)])
        with patch('httpx.Client', return_value=client):
            evolver = CodeEvolver(self.tokenizer, self.root / 'small.jsonl', 'small-fingerprint',
                                  base_url=GPU2, model_name=name)
        self.assertEqual(evolver({}, 11), proposal)
        self.assertEqual(client.requests[0][1]['model'], name)
        evolver.close()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-local-service-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tokenizer = CharacterTokenizer()

    def test_model_verification_is_filesystem_only_and_retains_metadata_and_size_checks(self):
        model = self.root / 'synthetic-model'
        model.mkdir()
        (model / 'config.json').write_text('{"fixture": true}')
        (model / 'weights.safetensors').write_bytes(b'synthetic weight fixture')
        manifest = dict(model_path=str(model), small_files={'config.json': file_hash(model / 'config.json')},
                        weight_sizes={'weights.safetensors': (model / 'weights.safetensors').stat().st_size})
        write(self.root / 'runs/feasibility_v1/freeze.json', {'model': manifest})
        with patch.object(local_service, 'ROOT', self.root), patch.object(local_service, 'MODEL', model), \
             patch('httpx.Client') as http:
            self.assertEqual(local_service.verify_model(), manifest)
            (model / 'config.json').write_text('{"fixture": false}')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                local_service.verify_model()
            (model / 'config.json').write_text('{"fixture": true}')
            (model / 'weights.safetensors').write_bytes(b'wrong size')
            with self.assertRaisesRegex(ValueError, 'weight size changed'):
                local_service.verify_model()
            changed = dict(manifest, model_path=str(self.root / 'different-model'))
            write(self.root / 'runs/feasibility_v1/freeze.json', {'model': changed})
            with self.assertRaisesRegex(ValueError, 'Frozen model is not'):
                local_service.verify_model()
        http.assert_not_called()

    def test_factory_routes_only_allowed_local_ports_without_connecting(self):
        for endpoint in (GPU0, GPU1, GPU2):
            with self.subTest(endpoint=endpoint):
                client = FakeClient([])
                with patch('httpx.Client', return_value=client) as factory:
                    llm = make_local_llm(self.tokenizer, self.root / 'costs.jsonl', 'synthetic-model', endpoint)
                self.assertEqual(llm.base, endpoint)
                self.assertEqual(llm.fingerprint, 'synthetic-model')
                self.assertIs(llm.client, client)
                self.assertEqual(client.requests, [])
                factory.assert_called_once_with(timeout=300, trust_env=False, follow_redirects=False)
                llm.client.close()

    def test_nonlocal_unknown_or_ambiguous_endpoints_are_rejected_before_client_creation(self):
        invalid = ['https://127.0.0.1:8142/v1', 'http://127.0.0.1:8143/v1',
                   'http://example.com:8142/v1', 'http://localhost:8142/v1',
                   'http://127.0.0.1:8142/v1/', 'http://127.0.0.1:8142/v1?target=other',
                   'http://user:secret@127.0.0.1:8142/v1', 'http://[::1]:8142/v1', None]
        with patch('httpx.Client') as client:
            for endpoint in invalid:
                with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                    make_local_llm(self.tokenizer, self.root / 'costs.jsonl', 'synthetic-model', endpoint)
            client.assert_not_called()

    def test_gpu2_proposer_and_reviewer_http_calls_keep_fixed_prompt_and_parameters(self):
        proposal = dict(source='synthetic source', hypothesis='Synthetic test.', risk='No task evidence.')
        from phase1_codegen.tests.test_review_contract import contract_payload, bundled_raw_report
        review_input = contract_payload('pre')
        displayed = bounded_review_payload(review_input, self.tokenizer)
        raw_review = bundled_raw_report(displayed)
        for role, constructor, value, input_payload, prompt, parameters in [
            ('evolver', CodeEvolver, proposal, {}, EVOLVER_PROMPT, EVOLVER_PARAMETERS),
            ('reviewer', MechanismReviewer, raw_review, review_input, REVIEW_PROMPT, REVIEW_PARAMETERS),
        ]:
            with self.subTest(role=role):
                client = FakeClient([FakeResponse(value)])
                with patch('httpx.Client', return_value=client):
                    instance = constructor(self.tokenizer, self.root / (role + '.jsonl'),
                                           'synthetic-model', base_url=GPU2)
                expected_value = validate_report(raw_review, displayed) if role == 'reviewer' else value
                self.assertEqual(instance(input_payload, 11), expected_value)
                self.assertEqual(len(client.requests), 1)
                endpoint, request = client.requests[0]
                self.assertEqual(endpoint, GPU2 + '/chat/completions')
                self.assertEqual(request['messages'][0]['content'], prompt)
                self.assertEqual(request['model'], MODEL_NAME)
                for key, expected in parameters.items():
                    self.assertEqual(request[key], expected)
                instance.close()
                self.assertTrue(client.closed)

    def test_endpoint_is_part_of_both_fixed_role_fingerprints(self):
        for constructor in (CodeEvolver, MechanismReviewer):
            with self.subTest(role=constructor.__name__):
                instances = []
                for endpoint in (GPU0, GPU2):
                    with patch('httpx.Client', return_value=FakeClient([])):
                        instances.append(constructor(self.tokenizer, self.root / 'ledger.jsonl',
                                                     'same-frozen-model', base_url=endpoint))
                self.assertNotEqual(instances[0].fingerprint, instances[1].fingerprint)
                for instance in instances:
                    instance.close()

    def test_service_identity_checks_expected_model_and_checkpoint_without_owning_client(self):
        for gpu, endpoint in ((0, GPU0), (1, GPU1), (2, GPU2)):
            with self.subTest(endpoint=endpoint):
                client = ModelsClient({'data': [{'id': MODEL_NAME, 'root': str(MODEL)}]})
                receipt = verify_local_service(endpoint, client=client)
                self.assertEqual(client.requests, [endpoint + '/models'])
                self.assertEqual(receipt['base_url'], endpoint)
                self.assertEqual(receipt['assigned_physical_gpu'], gpu)
                self.assertEqual(receipt['model_name'], MODEL_NAME)
                self.assertTrue(receipt['checkpoint_root_verified'])
                self.assertFalse(client.closed)
                self.assertIn('service-advertised', receipt['identity_source'])

    def test_service_identity_rejects_wrong_model_root_duplicate_or_malformed_responses(self):
        invalid = [None, {'data': {}}, {'data': []}, {'data': [{'id': 'wrong-model'}]},
                   {'data': [{'id': MODEL_NAME, 'root': '/unexpected/checkpoint'}]},
                   {'data': [{'id': MODEL_NAME}, {'id': MODEL_NAME}]}]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                verify_local_service(GPU2, client=ModelsClient(response))

    def test_missing_reported_checkpoint_is_not_claimed_as_verified(self):
        client = ModelsClient({'data': [{'id': MODEL_NAME}]})
        receipt = verify_local_service(GPU0, client=client)
        self.assertFalse(receipt['checkpoint_root_verified'])
        self.assertIsNone(receipt['reported_checkpoint'])

    def test_identity_query_owns_only_its_own_client_and_disables_proxy_redirects(self):
        client = ModelsClient({'data': [{'id': MODEL_NAME, 'root': str(MODEL)}]})
        with patch('httpx.Client', return_value=client) as factory:
            verify_local_service(GPU2)
        factory.assert_called_once_with(timeout=10, trust_env=False, follow_redirects=False)
        self.assertTrue(client.closed)


if __name__ == '__main__':
    unittest.main()
