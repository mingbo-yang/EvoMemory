"""Synthetic metadata fixtures only; no model, native environment or GPU calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evomem.common import read, write
from phase1_codegen.prepare_continual import ENVIRONMENTS, ROLES, prepare, split_adaptation
from phase1_codegen.protocol import WORKFLOW_VERSION, HOOK_VERSION


class ContinualPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-continual-plan-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.out = self.root / 'source', self.root / 'new-plan'
        write(self.source / 'protocol.json', dict(local_model_capabilities=True,
              collection_origin={'never_copy': 'old results'}, source_baseline_origins={'never_copy': True}))
        self.manifests = {}
        for environment in ENVIRONMENTS:
            for role in ROLES:
                rows = []
                for index in range(7 if role.endswith('adaptation') else 2):
                    identity = environment + '-' + role + '-' + str(index)
                    rows.append(dict(id=identity, group_id='group-' + identity,
                                     family='source-family' if role.startswith('source') else 'target-family',
                                     environment=environment, seed=42, game_sha256='hash-' + identity,
                                     archive_member='json_2.1.1/train/' + identity + '/game.tw-pddl',
                                     official_split='test' if role == 'target_generalization' else 'train'))
                path = self.source / 'datasets' / environment / (role + '.json')
                value = {'tasks': rows, 'metadata': 'synthetic wrapper'} if role == 'target_generalization' else rows
                write(path, value)
                self.manifests[(environment, role)] = rows
        # Traps are deliberately not valid JSON: opening outcomes would fail.
        (self.source / 'generalization').mkdir()
        (self.source / 'generalization' / 'results.json').write_text('DO NOT READ OUTCOMES')
        (self.source / 'state.json').write_text('DO NOT IMPORT OLD STATE')

    def test_preparation_keeps_order_counts_and_byte_identical_assessment_manifests(self):
        before = {str(p.relative_to(self.source)): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        receipt = prepare(self.source, self.out)
        self.assertEqual(receipt['status'], 'metadata_ready_no_execution')
        self.assertEqual(receipt['native_episodes_executed'], 0)
        protocol = read(self.out / 'protocol.json')
        for name, expected in [('workflow_version', WORKFLOW_VERSION), ('memory_hook_version', HOOK_VERSION),
                               ('mechanism_entrypoint', 'on_event(state, event, context)')]:
            self.assertEqual(protocol[name], expected)
            self.assertEqual(receipt[name], expected)
        self.assertEqual(protocol['task_memory_mode'], 'continual')
        self.assertTrue(protocol['local_model_capabilities'])
        self.assertIn('development_reuse', protocol['task_split_status'])
        self.assertNotIn('collection_origin', protocol)
        self.assertNotIn('source_baseline_origins', protocol)
        for environment in ENVIRONMENTS:
            for stage in ('source', 'target'):
                bootstrap = read(self.out / 'datasets' / environment / (stage + '_adaptation.json'))
                paths = protocol['learning_blocks'][environment][stage]
                blocks = [read(self.out / path) for path in paths]
                self.assertEqual([len(block) for block in blocks], [2, 2, 1])
                self.assertEqual(bootstrap + [row for block in blocks for row in block],
                                 self.manifests[(environment, stage + '_adaptation')])
            for role in ('source_exploration', 'target_exploration', 'target_generalization'):
                relative = Path('datasets') / environment / (role + '.json')
                self.assertEqual((self.out / relative).read_bytes(), (self.source / relative).read_bytes())
        self.assertEqual(before, {str(p.relative_to(self.source)): p.read_bytes()
                                  for p in self.source.rglob('*') if p.is_file()})
        self.assertFalse((self.out / 'generalization').exists())
        self.assertFalse((self.out / 'state.json').exists())

    def test_only_explicit_metadata_files_are_read(self):
        original = Path.read_bytes
        allowed = {self.source / 'protocol.json'} | {self.source / 'datasets' / env / (role + '.json')
                                                     for env in ENVIRONMENTS for role in ROLES}
        def guarded(path):
            if path.is_relative_to(self.source) and path not in allowed:
                raise AssertionError('Attempted non-metadata read: ' + str(path))
            return original(path)
        with patch.object(Path, 'read_bytes', guarded):
            prepare(self.source, self.out)
        self.assertFalse(read(self.out / 'metadata_preparation.json')['outcomes_read'])

    def test_existing_or_nested_destinations_are_refused_without_overwriting(self):
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, 'new directory'):
            prepare(self.source, self.out)
        self.assertEqual(list(self.out.iterdir()), [])
        with self.assertRaisesRegex(ValueError, 'outside'):
            prepare(self.source, self.source / 'nested-new-run')
        self.assertFalse((self.source / 'nested-new-run').exists())

    def test_insufficient_adaptation_is_not_silently_reduced_and_publishes_nothing(self):
        path = self.source / 'datasets/alfworld/source_adaptation.json'
        write(path, read(path)[:4])
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            prepare(self.source, self.out)
        self.assertFalse(self.out.exists())
        self.assertEqual(len(read(path)), 4)

    def test_cross_role_overlap_is_rejected_before_destination_is_published(self):
        path = self.source / 'datasets/textworld/target_exploration.json'
        rows = read(path)
        rows[0]['group_id'] = self.manifests[('textworld', 'source_adaptation')][0]['group_id']
        write(path, rows)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            prepare(self.source, self.out)
        self.assertFalse(self.out.exists())

    def test_bootstrap_is_explicit_and_stable_for_minimum_inventory(self):
        rows = [{'id': str(index)} for index in range(4)]
        prefix, blocks = split_adaptation(rows, 1)
        self.assertEqual(prefix + [row for block in blocks for row in block], rows)
        self.assertEqual([len(block) for block in blocks], [1, 1, 1])
        for value in (0, 3, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                split_adaptation(rows, value)


if __name__ == '__main__':
    unittest.main()
