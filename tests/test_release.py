"""Release contracts: checkpoint identity, explicit configuration, and portable adapters."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evomem.common import file_hash
from scripts.register_model import register
from phase1_codegen import local_service


class ReleaseTests(unittest.TestCase):
    def test_configured_model_and_endpoint_import_without_a_service(self):
        env=dict(os.environ,EVOMEMORY_MODEL_NAME='release-fixture-local',
                 EVOMEMORY_BASE_URL='http://127.0.0.1:18410/v1')
        code=('from phase1_codegen.capabilities import GENERATION_MODEL; '
              'from phase1_codegen.local_service import validate_base_url; '
              'from evomem.common import BASE_URL; '
              'assert GENERATION_MODEL == "release-fixture-local"; '
              'assert validate_base_url(BASE_URL) == "http://127.0.0.1:18410/v1"')
        subprocess.run([sys.executable,'-B','-c',code],env=env,check=True,capture_output=True,text=True)

    def test_checkpoint_registration_verifies_bytes_and_never_overwrites(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'checkpoint';model.mkdir()
            (model/'config.json').write_text('{"model_type":"fixture"}')
            (model/'weights.safetensors').write_bytes(b'synthetic fixture only')
            manifest=root/'registration.json'
            value=register(model,manifest)
            self.assertEqual(value['model']['small_files']['config.json'],file_hash(model/'config.json'))
            with patch.dict('os.environ',{'EVOMEMORY_MODEL_MANIFEST':str(manifest)}), patch.object(local_service,'MODEL',model):
                self.assertEqual(local_service.verify_model(),value['model'])
                (model/'config.json').write_text('{"changed":true}')
                with self.assertRaises(ValueError): local_service.verify_model()
            with self.assertRaises(FileExistsError): register(model,manifest)

    def test_benchmark_imports_do_not_download_or_require_private_data(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('unexpected download')):
            for name in ('benchmarks.math_apps.math_tasks','benchmarks.math_apps.apps_tasks',
                         'benchmarks.math_apps.benchmark','benchmarks.alfworld.official_panel'):
                self.assertIsNotNone(importlib.import_module(name))

    def test_scoring_and_prompts_survive_adapter_move(self):
        from benchmarks.math_apps import math_tasks,apps_tasks,actor
        self.assertEqual(math_tasks.boxed_integer(r'Answer: \boxed{-0012}'),'-12')
        self.assertIsNone(math_tasks.boxed_integer(r'\boxed{1/2}'))
        self.assertTrue(apps_tasks.stdio_equal('1 2\n','1 2'))
        self.assertEqual(actor.PARAMETERS['max_tokens'],4096)
        from evomem.common import BASE_URL
        self.assertEqual(actor.URL,BASE_URL)

    def test_moved_judge_runs_in_isolation_and_rejects_prohibited_import(self):
        from benchmarks.math_apps.code_sandbox import grade_stdio
        normal=grade_stdio('print(input())',['synthetic-fixture'],['synthetic-fixture'])
        blocked=grade_stdio('import socket\nprint("forbidden")',[''],['forbidden'])
        self.assertEqual(normal['score'],1)
        self.assertEqual(normal['status'],'passed')
        self.assertEqual(blocked['score'],0)
        self.assertEqual(blocked['status'],'unsupported_python')

if __name__=='__main__': unittest.main()
