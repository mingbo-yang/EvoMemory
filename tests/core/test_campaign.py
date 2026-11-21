"""CPU-only campaign contracts using mock role/environment objects.

Native-shaped episode fixtures below are explicit test doubles in temporary
folders. They never call models or environments and never establish real results.
Generated-memory retrieval checks use the actual bounded worker.
"""
from __future__ import annotations

import copy
import contextlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from evomem.common import digest, file_hash, read, write
from phase1_codegen import campaign
from phase1_codegen.reporting import campaign as campaign_report
from phase1_codegen.engine import EvolutionEngine
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.memory import CodeMemory
from phase1_codegen.learning import make_snapshot, save_snapshot
from phase1_codegen.store import RunStore
from tests.core.test_engine import FIRST_SOURCE, SECOND_SOURCE, SyntheticEvaluator, SyntheticProposer, TinyTokenizer, fixture_log, task
from tests.core.test_review_loop import SyntheticReviewer


def native_shaped_test_episode(item, phase='test-double', steps=0):
    """Mock-only schema fixture, never a claimed real environment observation."""
    return dict(task_id=item['id'], group_id=item['group_id'], family=item['family'],
                goal=item['goal'], seed=42000 + int(digest(item['id'])[:6], 16) % 10000000,
                environment_seed=item['seed'], phase=phase, score=0.5, error=None,
                history=[{'observation': 'synthetic fixture', 'action': 'look'} for _ in range(steps)],
                synthetic=False, native_reward_verified=True, fixture_origin='mock-only-test',
                executor='ALMA.Agent_Workflow.run_single_task', memory_write_calls=0,
                native_recorder={'reward': 0.5, 'initial': {'goal': item['goal'], 'obs': 'synthetic fixture'},
                                 'memory_retrieved': {}})


def design_record(name, environment='textworld', synthetic=False, family='synthetic-clean'):
    return dict(id=name, kind='code_change_trial', observed_context={'environment': environment,
                'family': family}, hypothesis='A mock design-trial record, not a task answer.',
                accepted=False, status='not_accepted',
                synthetic=synthetic, before_hash='mock-parent', after_hash='mock-child',
                code_diff='- old retrieval\n+ revised retrieval\n', support_hash='mock-support')


class MockNative:
    def __init__(self, environment, out, selected, tokenizer, model_hash, ctx=None):
        self.environment, self.out = environment, Path(out)
        self.selected = copy.deepcopy(selected)
        self.step_limit = 30 if environment == 'alfworld' else 80
        self.fingerprint = 'mock-native-' + environment
        self.calls = []
        self.closed = False

    def no_memory(self, selected, phase):
        self.calls.append(dict(kind='no_memory', tasks=copy.deepcopy(selected), phase=phase))
        return [native_shaped_test_episode(item, phase=phase) for item in selected]

    def __call__(self, program, logs, selected, phase, *, memory_snapshot=None):
        self.calls.append(dict(kind='memory', program_hash=program.hash, logs=copy.deepcopy(logs),
                               tasks=copy.deepcopy(selected), phase=phase))
        episodes = [native_shaped_test_episode(item, phase=phase) for item in selected]
        if memory_snapshot is not None:
            for episode in episodes:
                episode.update(program_hash=program.hash, read_only_memory=True,
                    generated_memory_write_calls=0, memory_hash_before=memory_snapshot['memory_hash'],
                    memory_hash_after=memory_snapshot['memory_hash'])
        return episodes

    def close(self):
        self.closed = True


class MockRole:
    def __init__(self, tokenizer, ledger, model_hash, base_url):
        self.ledger, self.base_url = Path(ledger), base_url
        self.fingerprint = 'mock-independent-role-' + str(self.ledger)
        self.closed = False

    def close(self):
        self.closed = True


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mock-only-intraenv-campaign-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tokenizer = TinyTokenizer()
        self.natives = []
        self.roles = []
        for environment in campaign.ENDPOINTS:
            for role in ('source_adaptation', 'source_exploration', 'target_adaptation',
                         'target_exploration', 'target_generalization'):
                self.add_tasks(environment, role)
        blocks = {}
        for environment in campaign.ENDPOINTS:
            blocks[environment] = {}
            for stage in ('source', 'target'):
                blocks[environment][stage] = []
                for index in range(campaign.ROUNDS):
                    role = stage + '_learning_%03d' % index
                    self.add_tasks(environment, role, count=1)
                    blocks[environment][stage].append(str(campaign.task_file(self.root, environment, role).relative_to(self.root)))
        write(self.root / 'protocol.json', dict(task_memory_mode='continual',
            workflow_version=campaign.WORKFLOW_VERSION, memory_hook_version=campaign.HOOK_VERSION,
            learning_blocks=blocks))

    def update_protocol(self, **fields):
        protocol = read(self.root / 'protocol.json')
        protocol.update(fields)
        write(self.root / 'protocol.json', protocol)

    def test_explicit_round_budget_controls_the_number_of_learning_blocks(self):
        protocol = read(self.root / 'protocol.json')
        for environment in protocol['learning_blocks']:
            for stage, paths in protocol['learning_blocks'][environment].items():
                protocol['learning_blocks'][environment][stage] = paths[:2]
        protocol['rounds_per_search'] = 2
        write(self.root / 'protocol.json', protocol)
        self.assertEqual(campaign.rounds_per_search(self.root), 2)
        self.assertEqual(len(campaign.learning_blocks(self.root, 'alfworld', 'source')), 2)
        self.update_protocol(rounds_per_search=3)
        with self.assertRaisesRegex(ValueError, 'every round'):
            campaign.learning_blocks(self.root, 'alfworld', 'source')

    def test_round_budget_cannot_be_inferred_or_coerced_from_bad_values(self):
        for value in (0, -1, True, '2', None):
            self.update_protocol(rounds_per_search=value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive frozen integer'):
                campaign.rounds_per_search(self.root)

    def test_default_entrypoint_is_the_single_campaign_entrypoint(self):
        from phase1_codegen import __main__ as entrypoint
        self.assertIs(entrypoint.main, campaign.main)
        self.assertFalse(hasattr(entrypoint, 'legacy_main'))
        self.assertFalse(hasattr(entrypoint, 'load_config'))

    def test_retired_commands_are_rejected_before_services(self):
        from phase1_codegen import __main__ as entrypoint
        for command in ('legacy', 'run', 'validate-config', 'inspect'):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()) as stderr, \
                 patch.object(campaign, 'verify_local_service') as service, \
                 patch.object(campaign, 'Tokenizer') as tokenizer, self.assertRaises(SystemExit) as error:
                entrypoint.main([command, '--root', str(self.root)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn('invalid choice', stderr.getvalue())
            service.assert_not_called()
            tokenizer.assert_not_called()

    def test_current_cli_rejects_workflow_drift_before_service_or_tokenizer(self):
        from phase1_codegen.protocol import WORKFLOW_VERSION, HOOK_VERSION
        current = read(self.root / 'protocol.json')
        for name, value in [('workflow_version', 'old-workflow'), ('memory_hook_version', None)]:
            write(self.root / 'protocol.json', dict(current, **{name: value}))
            with self.subTest(name=name, value=value), \
                 patch.object(campaign, 'verify_local_service') as service, \
                 patch.object(campaign, 'Tokenizer') as tokenizer, self.assertRaises(ValueError):
                campaign.main(['source', '--root', str(self.root), '--environment', 'alfworld'])
            service.assert_not_called()
            tokenizer.assert_not_called()
        self.assertFalse((self.root / 'services').exists())

    def test_current_cli_rejects_invalid_splits_before_service_or_tokenizer(self):
        rows = campaign.tasks(self.root, 'alfworld', 'source_exploration')
        rows[0]['split'] = 'test'
        write(campaign.task_file(self.root, 'alfworld', 'source_exploration'), rows)
        with patch.object(campaign, 'verify_local_service') as service, \
             patch.object(campaign, 'Tokenizer') as tokenizer, self.assertRaises(ValueError):
            campaign.main(['source', '--root', str(self.root), '--environment', 'alfworld'])
        service.assert_not_called()
        tokenizer.assert_not_called()

    def test_current_cli_requires_explicit_continual_mode_before_services(self):
        current = read(self.root / 'protocol.json')
        for mode in (None, 'frozen_replay', 'unknown'):
            protocol = dict(current)
            if mode is None:
                protocol.pop('task_memory_mode')
            else:
                protocol['task_memory_mode'] = mode
            write(self.root / 'protocol.json', protocol)
            with self.subTest(mode=mode), patch.object(campaign, 'verify_local_service') as service, \
                 patch.object(campaign, 'Tokenizer') as tokenizer, self.assertRaisesRegex(ValueError, 'continual'):
                campaign.main(['source', '--root', str(self.root), '--environment', 'alfworld'])
            service.assert_not_called()
            tokenizer.assert_not_called()
        (self.root / 'protocol.json').unlink()
        with self.assertRaisesRegex(ValueError, 'explicit continual protocol'):
            campaign.validate_dataset(self.root, 'alfworld')

    def test_declared_learning_blocks_must_all_be_nonempty(self):
        protocol = read(self.root / 'protocol.json')
        block = self.root / protocol['learning_blocks']['alfworld']['source'][1]
        write(block, [])
        with self.assertRaisesRegex(ValueError, 'nonempty learning block'):
            campaign.validate_dataset(self.root, 'alfworld')

    def test_campaign_freeze_records_the_actual_hook_prompt_and_workflow(self):
        from phase1_codegen.protocol import WORKFLOW_VERSION, HOOK_VERSION
        from phase1_codegen.roles.evolver import EVOLVER_PROMPT
        source = self.root / 'fixture.py'
        source.write_text('# Synthetic freeze fixture, never executed.\n')
        with patch.object(campaign, 'ROOT', self.root), \
             patch.object(campaign, 'code_binding', return_value={'fixture.py': file_hash(source)}), \
             patch.object(campaign.importlib.metadata, 'distributions', return_value=[]):
            frozen = campaign.freeze_search(self.root / 'freeze-fixture', {'synthetic': True}, 'mock-model')
        self.assertEqual(frozen['version'], WORKFLOW_VERSION)
        self.assertEqual(frozen['evolver_abi'], HOOK_VERSION)
        self.assertEqual(frozen['evolver_prompt'], EVOLVER_PROMPT)
        self.assertNotIn('optional_hook_contract', frozen)

    def native_factory(self, *args, **kwargs):
        native = MockNative(*args, **kwargs)
        self.natives.append(native)
        return native

    def role_factory(self, *args, **kwargs):
        role = MockRole(*args, **kwargs)
        self.roles.append(role)
        return role

    def add_tasks(self, environment, role, count=2):
        selected = [task(environment + '-' + role + '-' + str(i)) for i in range(count)]
        family = ({'alfworld': 'clean', 'textworld': 'the_cooking_game'} if role.startswith('source_')
                  else {'alfworld': 'two', 'textworld': 'treasure_hunter'})[environment]
        for item in selected:
            item['family'] = family
            split = 'eval_out_of_distribution' if role == 'target_generalization' else 'train'
            item.update(split=split, official_split=split, archive_member='json_2.1.1/' + split + '/' + item['id'],
                        game_sha256=digest(item['id']))
        write(campaign.task_file(self.root, environment, role), selected)
        return selected

    def add_adaptation(self, environment, role):
        selected = self.add_tasks(environment, role + '_adaptation')
        logs = [native_shaped_test_episode(t, role + '_adaptation') for t in selected]
        write(self.root / 'collection' / environment / (role + '_adaptation.json'),
              dict(episodes=logs, complete=True, synthetic=False, fixture_origin='mock-only-test'))
        return logs

    def prepare_source_states(self, synthetic=False, complete=True):
        states, trials = {}, {}
        for environment in campaign.ENDPOINTS:
            folder = self.root / 'source' / environment
            state = dict(next_round=campaign.ROUNDS, fixture_origin='mock-only-test',
                         binding={'synthetic': synthetic, 'workflow': campaign.WORKFLOW_VERSION,
                                  'context': {'environment': environment,
                                  'family': 'clean' if environment == 'alfworld' else 'the_cooking_game'}},
                         environment=environment, current_program=Program(FIRST_SOURCE).hash)
            collection = self.root / 'collection' / environment / 'source_adaptation.json'
            logs = (read(collection)['episodes'] if collection.exists() else
                    [native_shaped_test_episode(t, 'source_adaptation')
                     for t in campaign.tasks(self.root, environment, 'source_adaptation')])
            logs = [dict(e, environment=environment) for e in logs]
            memory = CodeMemory(Program(FIRST_SOURCE), logs, self.tokenizer,
                                campaign.workflow_context(environment, campaign.tasks(self.root, environment, 'source_exploration')))
            snapshot = make_snapshot(logs, environment, memory_snapshot=memory.export_snapshot())
            state['task_memory'] = save_snapshot(folder, snapshot)
            state['binding']['task_memory_mode'] = 'continual'
            write(folder / 'state.json', state)
            write(folder / 'status.json', {'status': 'complete' if complete else 'searching'})
            states[str(folder)] = state
            # Positive and rejected source events must both survive transfer.
            records = [design_record(environment + '-positive', environment, synthetic, state['binding']['context']['family']),
                       design_record(environment + '-negative', environment, synthetic, state['binding']['context']['family'])]
            records[0]['accepted'] = True
            records[0]['status'] = 'accepted'
            trials[str(folder)] = [dict(evolution_event=r,
                private_task_memory={'state': 'must never transfer'},
                history=[{'observation': 'private source task'}]) for r in records]
        class SourceStore:
            def __init__(self, root):
                self.root = Path(root)
            def validate_state(self, state):
                if state != states[str(self.root)]:
                    raise RuntimeError('Mock source state mismatch')
            def history(self, state):
                return copy.deepcopy(trials[str(self.root)])
            def current(self):
                return Program(FIRST_SOURCE)
            def lock(self):
                return contextlib.nullcontext()
        self.source_store = SourceStore
        return states, trials, SourceStore

    def test_dataset_roles_reject_overlapping_ids_groups_or_underlying_games(self):
        for field in ('id', 'group_id', 'game_sha256'):
            with self.subTest(field=field):
                source = campaign.tasks(self.root, 'alfworld', 'source_adaptation')
                target = campaign.tasks(self.root, 'alfworld', 'target_adaptation')
                original = copy.deepcopy(target)
                target[0][field] = source[0][field]
                write(campaign.task_file(self.root, 'alfworld', 'target_adaptation'), target)
                with self.assertRaisesRegex(ValueError, 'disjoint'):
                    campaign.validate_dataset(self.root, 'alfworld')
                write(campaign.task_file(self.root, 'alfworld', 'target_adaptation'), original)

    def test_dataset_requires_distinct_source_and_target_task_types_within_each_environment(self):
        for environment in campaign.ENDPOINTS:
            with self.subTest(environment=environment):
                target_family = campaign.tasks(self.root, environment, 'target_exploration')[0]['family']
                for role in ('source_adaptation', 'source_exploration'):
                    selected = campaign.tasks(self.root, environment, role)
                    for item in selected:
                        item['family'] = target_family
                    write(campaign.task_file(self.root, environment, role), selected)
                with self.assertRaises((ValueError, RuntimeError)):
                    campaign.validate_dataset(self.root, environment)

    def test_target_rejects_other_environment_design_library_before_execution(self):
        self.add_adaptation('alfworld', 'target')
        write(self.root / 'transfer' / 'alfworld.json',
              [design_record('foreign-textworld-record', 'textworld', family='the_cooking_game')])
        with patch.object(campaign, 'evaluator', self.native_factory), \
             patch.object(campaign, 'CodeEvolver') as proposer, \
             patch.object(campaign, 'MechanismReviewer') as reviewer, \
             self.assertRaises((ValueError, RuntimeError)):
            campaign.search(self.root, 'alfworld', 'target', 'cross_type_em', self.tokenizer, 'mock-model')
        self.assertEqual(self.natives, [])
        proposer.assert_not_called()
        reviewer.assert_not_called()

    def test_search_rejects_old_or_unknown_role_condition_combinations(self):
        self.add_adaptation('alfworld', 'source')
        self.add_adaptation('alfworld', 'target')
        invalid = [('source', 'cross_type_em'), ('source', 'no_external_em'),
                   ('target', 'local_only'), ('target', 'cross_em'),
                   ('target', 'unknown_condition'), ('generalization', 'cross_type_em')]
        with patch.object(campaign, 'evaluator', self.native_factory), \
             patch.object(campaign, 'CodeEvolver') as proposer, \
             patch.object(campaign, 'MechanismReviewer') as reviewer:
            for role, condition in invalid:
                with self.subTest(role=role, condition=condition), self.assertRaises(ValueError):
                    campaign.search(self.root, 'alfworld', role, condition, self.tokenizer, 'mock-model')
        self.assertEqual(self.natives, [])
        proposer.assert_not_called()
        reviewer.assert_not_called()

    def test_adaptation_and_exploration_roles_cannot_disagree_on_task_family(self):
        selected = campaign.tasks(self.root, 'textworld', 'source_adaptation')
        for item in selected:
            item['family'] = 'unexpected_source_family'
        write(campaign.task_file(self.root, 'textworld', 'source_adaptation'), selected)
        with self.assertRaises((ValueError, RuntimeError)):
            campaign.validate_dataset(self.root, 'textworld')

    def test_search_metadata_rejects_heldout_tasks_before_any_native_execution(self):
        selected = campaign.tasks(self.root, 'alfworld', 'source_exploration')
        selected[0]['archive_member'] = 'json_2.1.1/eval_out_of_distribution/forbidden-task'
        selected[0]['split'] = 'test'
        write(campaign.task_file(self.root, 'alfworld', 'source_exploration'), selected)
        with patch.object(campaign, 'evaluator', self.native_factory), self.assertRaises(ValueError):
            campaign.collection(self.root, 'alfworld', self.tokenizer, 'mock-model')
        self.assertEqual(self.natives, [])

    def reuse_fixture(self):
        origin = self.root / 'mock-original-collection'
        expected = campaign.tasks(self.root, 'alfworld', 'target_adaptation')
        original = dict(version='real-no-memory-collection-v1', environment='alfworld',
            model_hash='mock-model', endpoint=campaign.ENDPOINTS['alfworld'],
            actor_parameters={'temperature': 0, 'max_tokens': 512},
            own_code={'native.py': 'mock-code-hash'}, executor_bridge='mock-bridge-hash',
            tasks={'target_adaptation': expected})
        episodes = [native_shaped_test_episode(t, 'target_adaptation') for t in expected]
        record = dict(episodes=episodes, complete=True, synthetic=False,
                      collection='official no-memory Executor', fixture_origin='mock-only-test')
        write(origin / 'binding.json', original)
        write(origin / 'target_adaptation.json', record)
        for item, episode in zip(expected, episodes):
            write(origin / 'episodes' / (digest(item) + '.json'),
                  dict(task_hash=digest(item), binding_hash=digest(original), episode=episode))
        spec = dict(collection=str(origin), binding_hash=digest(original),
                    artifact_hash=file_hash(origin / 'target_adaptation.json'))
        return spec, expected, copy.deepcopy(original), record

    def test_valid_reuse_keeps_original_evidence_and_skips_target_native_execution(self):
        origin_root = self.root / 'mock-original-run'
        shutil.copytree(self.root / 'datasets', origin_root / 'datasets')
        write(origin_root / 'protocol.json', read(self.root / 'protocol.json'))
        with patch.object(campaign, 'evaluator', self.native_factory):
            campaign.collection(origin_root, 'alfworld', self.tokenizer, 'mock-model')
            origin = origin_root / 'collection' / 'alfworld'
            original_binding = read(origin / 'binding.json')
            artifact_hash = file_hash(origin / 'target_adaptation.json')
            spec = dict(collection=str(origin), binding_hash=digest(original_binding),
                        artifact_hash=artifact_hash)
            write(self.root / 'collection' / 'alfworld' / 'reuse.json', {'target_adaptation': spec})
            campaign.collection(self.root, 'alfworld', self.tokenizer, 'mock-model')
            campaign.collection(self.root, 'alfworld', self.tokenizer, 'mock-model')
        self.assertEqual(len(self.natives[0].calls), 4)
        self.assertEqual(len(self.natives[1].calls), 2)
        self.assertTrue(all(c['phase'] == 'source_adaptation' for c in self.natives[1].calls))
        self.assertEqual(self.natives[2].calls, [])
        current = read(self.root / 'collection' / 'alfworld' / 'target_adaptation.json')
        self.assertEqual(current['episodes'], read(origin / 'target_adaptation.json')['episodes'])
        self.assertFalse(current['newly_executed'])
        self.assertEqual(current['reuse_provenance'], spec)
        self.assertEqual(file_hash(origin / 'target_adaptation.json'), artifact_hash)
        status = read(self.root / 'collection' / 'alfworld' / 'status.json')
        self.assertEqual(status['reused_episodes'], len(current['episodes']))
        self.assertTrue(all(native.closed for native in self.natives))

    def test_reuse_rejects_changed_tasks_model_actor_or_executed_code(self):
        spec, expected, binding, _ = self.reuse_fixture()
        for field, changed in [('model_hash', 'different-model'), ('own_code', {'native.py': 'changed'}),
                               ('actor_parameters', {'temperature': 1}), ('executor_bridge', 'changed-bridge'),
                               ('endpoint', campaign.ENDPOINTS['textworld'])]:
            with self.subTest(field=field):
                damaged = copy.deepcopy(binding); damaged[field] = changed
                with self.assertRaisesRegex(RuntimeError, 'execution binding differs'):
                    campaign.reused_adaptation(spec, 'target_adaptation', expected, damaged, 30)
        changed_tasks = copy.deepcopy(expected); changed_tasks[0]['seed'] += 1
        with self.assertRaisesRegex(RuntimeError, 'task split differs'):
            campaign.reused_adaptation(spec, 'target_adaptation', changed_tasks, binding, 30)

    def test_reuse_rejects_original_binding_artifact_and_per_episode_cache_drift(self):
        for mutation in ('binding', 'artifact', 'cache_task', 'cache_binding', 'cache_episode'):
            with self.subTest(mutation=mutation):
                spec, expected, binding, record = self.reuse_fixture()
                origin = Path(spec['collection'])
                if mutation == 'binding':
                    changed = read(origin / 'binding.json'); changed['model_hash'] = 'tampered-original'
                    write(origin / 'binding.json', changed)
                elif mutation == 'artifact':
                    changed = copy.deepcopy(record); changed['episodes'][0]['score'] = 0.9
                    write(origin / 'target_adaptation.json', changed)
                else:
                    path = origin / 'episodes' / (digest(expected[0]) + '.json')
                    changed = read(path)
                    if mutation == 'cache_task':
                        changed['task_hash'] = 'wrong-task'
                    elif mutation == 'cache_binding':
                        changed['binding_hash'] = 'wrong-binding'
                    else:
                        changed['episode']['score'] = 0.9
                    write(path, changed)
                with self.assertRaises(RuntimeError):
                    campaign.reused_adaptation(spec, 'target_adaptation', expected, binding, 30)

    def test_hash_consistent_reuse_still_requires_valid_native_episode(self):
        for mutation in ('synthetic', 'native_reward', 'native_error', 'too_many_steps'):
            with self.subTest(mutation=mutation):
                spec, expected, binding, record = self.reuse_fixture()
                episode = record['episodes'][0]
                if mutation == 'synthetic':
                    episode['synthetic'] = True
                elif mutation == 'native_reward':
                    episode['native_reward_verified'] = False
                elif mutation == 'native_error':
                    episode['error'] = 'MockNativeError'
                else:
                    episode['history'] = [{'action': 'look'}] * 31
                origin = Path(spec['collection'])
                artifact = origin / 'target_adaptation.json'
                write(artifact, record)
                spec['artifact_hash'] = file_hash(artifact)
                cache = origin / 'episodes' / (digest(expected[0]) + '.json')
                item = read(cache); item['episode'] = episode; write(cache, item)
                with self.assertRaisesRegex(RuntimeError, 'Incomplete real episode'):
                    campaign.reused_adaptation(spec, 'target_adaptation', expected, binding, 30)

    def test_collection_uses_only_no_memory_executor_and_reuses_its_own_cache(self):
        for role in ('source_adaptation', 'target_adaptation'):
            self.add_tasks('alfworld', role)
        with patch.object(campaign, 'evaluator', self.native_factory), \
             patch.object(campaign, 'CodeEvolver') as proposer, \
             patch.object(campaign, 'MechanismReviewer') as reviewer:
            campaign.collection(self.root, 'alfworld', self.tokenizer, 'mock-model')
            campaign.collection(self.root, 'alfworld', self.tokenizer, 'mock-model')
        proposer.assert_not_called(); reviewer.assert_not_called()
        self.assertEqual(len(self.natives[0].calls), 4)
        self.assertEqual(self.natives[1].calls, [])
        self.assertTrue(all(call['kind'] == 'no_memory' for call in self.natives[0].calls))
        self.assertTrue(all(native.closed for native in self.natives))
        source = campaign.adaptation(self.root, 'alfworld', 'source')
        target = campaign.adaptation(self.root, 'alfworld', 'target')
        self.assertFalse({r['task_id'] for r in source} & {r['task_id'] for r in target})
        self.assertTrue(all('source_adaptation' in r['phase'] for r in source))
        self.assertTrue(all('target_adaptation' in r['phase'] for r in target))

    def test_collection_cache_binding_cannot_be_reused_after_task_or_model_change(self):
        for role in ('source_adaptation', 'target_adaptation'):
            self.add_tasks('textworld', role)
        with patch.object(campaign, 'evaluator', self.native_factory):
            campaign.collection(self.root, 'textworld', self.tokenizer, 'mock-model')
            with self.assertRaisesRegex(RuntimeError, 'Frozen phase inputs changed'):
                campaign.collection(self.root, 'textworld', self.tokenizer, 'changed-model')
        self.assertEqual(len(self.natives), 1)

    def test_collection_binds_protocol_and_rejects_drift_before_native_reuse(self):
        actual_hash = campaign.file_hash
        def changed_protocol(path):
            return '0' * 64 if Path(path).name == 'protocol.py' else actual_hash(path)
        with patch.object(campaign, 'evaluator', self.native_factory):
            campaign.collection(self.root, 'alfworld', self.tokenizer, 'mock-model')
            binding = read(self.root / 'collection/alfworld/binding.json')
            self.assertEqual(binding['own_code']['protocol.py'],
                             actual_hash(Path(campaign.__file__).with_name('protocol.py')))
            with patch.object(campaign, 'file_hash', changed_protocol), \
                 self.assertRaisesRegex(RuntimeError, 'Frozen phase inputs changed'):
                campaign.collection(self.root, 'alfworld', self.tokenizer, 'mock-model')
        self.assertEqual(len(self.natives), 1)

    def test_target_conditions_use_identical_initial_program_logs_tasks_and_seed(self):
        source_logs = self.add_adaptation('alfworld', 'source')
        target_logs = self.add_adaptation('alfworld', 'target')
        logs = [dict(e, environment='alfworld') for e in source_logs + target_logs]
        states, _, mock_store = self.prepare_source_states()
        selected = self.add_tasks('alfworld', 'target_exploration')
        records = [design_record('same-environment-source-design', 'alfworld', family='clean')]
        write(self.root / 'transfer' / 'alfworld.json', records)
        plans, calls = [], []
        def frozen(out, plan, model):
            plans.append(copy.deepcopy(plan))
        class MockEngine:
            def __init__(inner, out, tokenizer, proposer, native, memory, ctx, supplied_logs,
                         exploration, confirmation, **kwargs):
                calls.append(dict(out=Path(out), proposer=proposer, native=native,
                    reviewer=kwargs['reviewer'], records=memory.records, logs=supplied_logs,
                    exploration=exploration, confirmation=confirmation, kwargs=kwargs))
                inner.store = SimpleNamespace(history=lambda state: [])
            def run(inner, rounds):
                return dict(next_round=rounds, revision=0)
        with patch.object(campaign, 'freeze_search', frozen), \
             patch.object(campaign, 'evaluator', self.native_factory), \
             patch.object(campaign, 'CodeEvolver', self.role_factory), \
             patch.object(campaign, 'MechanismReviewer', self.role_factory), \
             patch.object(campaign, 'RunStore', mock_store), \
             patch.object(campaign, 'EvolutionEngine', MockEngine):
            for condition in ('no_external_em', 'cross_type_em'):
                campaign.search(self.root, 'alfworld', 'target', condition, self.tokenizer, 'mock-model')
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['records'], [])
        self.assertEqual(calls[1]['records'], records)
        for call, plan in zip(calls, plans):
            self.assertEqual(call['logs'], logs)
            self.assertEqual(call['exploration'], selected)
            self.assertEqual(call['kwargs']['seed'], 11)
            self.assertEqual(call['kwargs']['max_rounds'], campaign.ROUNDS)
            self.assertEqual(call['kwargs']['acceptance_policy'], 'reviewer_and_reward')
            self.assertNotIn('interface_repair_attempts', call['kwargs'])
            self.assertEqual(plan['context']['memory_hook_version'], campaign.HOOK_VERSION)
            self.assertEqual(call['confirmation'], [])
            self.assertIsNot(call['proposer'], call['reviewer'])
            self.assertEqual(plan['initial_program_hash'], Program(FIRST_SOURCE).hash)
            self.assertNotEqual(plan['initial_program_hash'], Program(BASELINE_SOURCE).hash)
            self.assertEqual(call['kwargs']['initial_source'], FIRST_SOURCE)
            self.assertEqual(plan['adaptation_hash'], digest(logs))
            self.assertEqual(plan['memory_scope'], 'environment')
            self.assertTrue(plan['task_memory_shared_across_types'])
            self.assertEqual(plan['task_memory_families'], ['clean', 'two'])
            self.assertEqual(plan['inherited_checkpoint']['source_state_hash'],
                             digest(states[str(self.root / 'source' / 'alfworld')]))
            self.assertEqual(plan['inherited_checkpoint']['source_adaptation_hash'],
                             digest([dict(e, environment='alfworld') for e in source_logs]))
        self.assertNotEqual(calls[0]['out'], calls[1]['out'])
        self.assertNotEqual(calls[0]['proposer'].ledger, calls[1]['proposer'].ledger)
        self.assertTrue(all(role.closed for role in self.roles))
        self.assertTrue(all(native.closed for native in self.natives))

    def test_continual_search_rejects_frozen_replay_baseline_import_before_execution(self):
        self.add_adaptation('alfworld', 'source')
        origin = {'fixture': 'mock origin; native identity checks tested separately'}
        self.update_protocol(source_baseline_origins={'alfworld': origin})
        with patch.object(campaign, 'freeze_search') as freeze, \
             patch.object(campaign, 'evaluator', self.native_factory), \
             self.assertRaisesRegex(ValueError, 'frozen-replay source baseline'):
            campaign.search(self.root, 'alfworld', 'source', 'local_only', self.tokenizer, 'mock-model')
        freeze.assert_not_called()
        self.assertEqual(self.natives, [])

    def test_transfer_exports_only_same_environment_different_type_events_and_all_signs(self):
        _, source_trials, mock_store = self.prepare_source_states()
        with patch.object(campaign, 'RunStore', mock_store):
            campaign.transfer(self.root)
        for target_environment in campaign.ENDPOINTS:
            source_environment = target_environment
            records = read(self.root / 'transfer' / (target_environment + '.json'))
            expected = [t['evolution_event'] for t in source_trials[str(self.root / 'source' / source_environment)]]
            self.assertEqual(records, expected)
            self.assertEqual({r['accepted'] for r in records}, {True, False})
            serialized = json.dumps(records)
            self.assertNotIn('private_task_memory', serialized)
            self.assertNotIn('private source task', serialized)
            provenance = read(self.root / 'transfer' / (target_environment + '.provenance.json'))
            self.assertEqual(provenance['source_environment'], target_environment)
            self.assertEqual(provenance['target_environment'], target_environment)
            self.assertEqual(provenance['source_family'], 'clean' if target_environment == 'alfworld' else 'the_cooking_game')
            self.assertEqual(provenance['target_family'], 'two' if target_environment == 'alfworld' else 'treasure_hunter')
            self.assertNotEqual(provenance['source_family'], provenance['target_family'])
            self.assertFalse(provenance['includes_task_memory'])
            self.assertFalse(provenance['includes_raw_trajectories'])
            self.assertFalse(provenance['includes_final_answers'])

    def test_transfer_rejects_synthetic_sources_before_publishing_any_bank(self):
        _, _, mock_store = self.prepare_source_states(synthetic=True)
        with patch.object(campaign, 'RunStore', mock_store), self.assertRaises((ValueError, RuntimeError)):
            campaign.transfer(self.root)
        self.assertFalse(list((self.root / 'transfer').glob('*.json')))

    def test_one_environment_can_export_without_other_environment_source_files(self):
        _, source_trials, mock_store = self.prepare_source_states()
        # The two experimental libraries have no cross-environment dependency.
        (self.root / 'source' / 'textworld' / 'state.json').unlink()
        (self.root / 'source' / 'textworld' / 'status.json').unlink()
        with patch.object(campaign, 'RunStore', mock_store):
            campaign.transfer(self.root, 'alfworld')
        records = read(self.root / 'transfer' / 'alfworld.json')
        self.assertEqual(records, [t['evolution_event'] for t in source_trials[str(self.root / 'source' / 'alfworld')]])
        self.assertFalse((self.root / 'transfer' / 'textworld.json').exists())

    def prepare_generalization(self, completed=True):
        self.add_adaptation('alfworld', 'source')
        self.add_adaptation('alfworld', 'target')
        self.prepare_source_states()
        selected = self.add_tasks('alfworld', 'target_generalization')
        with patch.object(campaign, 'RunStore', self.source_store):
            snapshot = campaign.target_memory_checkpoint(self.root, 'alfworld', self.tokenizer, None)
        for condition in ('no_external_em', 'cross_type_em'):
            folder = self.root / 'target' / 'alfworld' / condition
            write(folder / 'state.json', {'task_memory': save_snapshot(folder, snapshot)})
            write(folder / 'status.json',
                  {'status': 'complete' if completed or condition == 'no_external_em' else 'searching'})
        return selected

    def test_generalization_does_not_execute_until_both_target_conditions_finish(self):
        self.prepare_generalization(completed=False)
        mock_store = self.source_store
        with patch.object(campaign, 'RunStore', mock_store), \
             patch.object(campaign, 'evaluator', self.native_factory), \
             self.assertRaisesRegex(RuntimeError, 'All target search versions'):
            campaign.generalization(self.root, 'alfworld', self.tokenizer, 'mock-model')
        self.assertEqual(self.natives, [])
        self.assertFalse((self.root / 'generalization' / 'alfworld' / 'results.json').exists())

    def test_identical_sealed_variants_share_cache_but_never_call_meta_roles(self):
        selected = self.prepare_generalization()
        mock_store = self.source_store
        with patch.object(campaign, 'RunStore', mock_store), \
             patch.object(campaign, 'evaluator', self.native_factory), \
             patch.object(campaign, 'CodeEvolver') as proposer, \
             patch.object(campaign, 'MechanismReviewer') as reviewer:
            campaign.generalization(self.root, 'alfworld', self.tokenizer, 'mock-model')
            campaign.generalization(self.root, 'alfworld', self.tokenizer, 'mock-model')
        proposer.assert_not_called(); reviewer.assert_not_called()
        self.assertEqual(len(self.natives[0].calls), len(selected))
        self.assertEqual(self.natives[1].calls, [])
        self.assertTrue(all(c['phase'].startswith('sealed_generalization/') for c in self.natives[0].calls))
        self.assertTrue(all(native.closed for native in self.natives))
        results = read(self.root / 'generalization' / 'alfworld' / 'results.json')
        self.assertEqual(set(results), {'inherited_no_target_evolution', 'no_external_em', 'cross_type_em'})
        self.assertEqual(results['inherited_no_target_evolution']['program_hash'], Program(FIRST_SOURCE).hash)
        self.assertNotEqual(results['inherited_no_target_evolution']['program_hash'], Program(BASELINE_SOURCE).hash)
        self.assertEqual(len({r['program_hash'] for r in results.values()}), 1)
        self.assertEqual(len({digest(r['episodes']) for r in results.values()}), 1)
        with patch.object(campaign, 'RunStore', self.source_store):
            shared = campaign.shared_logs(self.root, 'alfworld', 'target')
        self.assertTrue(all(call['logs'] == shared for call in self.natives[0].calls))
        binding = read(self.root / 'generalization' / 'alfworld' / 'binding.json')
        self.assertEqual(binding['logs_hash'], digest(shared))
        self.assertTrue(binding['sealed_from_review_and_evolver'])

    def test_shared_logs_preserve_order_and_leave_original_evidence_unchanged(self):
        source = self.add_adaptation('alfworld', 'source')
        target = self.add_adaptation('alfworld', 'target')
        paths = [self.root / 'collection' / 'alfworld' / (r + '_adaptation.json') for r in ('source', 'target')]
        before = [p.read_bytes() for p in paths]
        self.prepare_source_states()
        self.assertEqual(campaign.shared_logs(self.root, 'alfworld', 'source'),
                         [dict(e, environment='alfworld') for e in source])
        with patch.object(campaign, 'RunStore', self.source_store):
            self.assertEqual(campaign.shared_logs(self.root, 'alfworld', 'target'),
                             [dict(e, environment='alfworld') for e in source + target])
        self.assertEqual([p.read_bytes() for p in paths], before)
        value = read(paths[1]); value['episodes'][0]['environment'] = 'textworld'; write(paths[1], value)
        with patch.object(campaign, 'RunStore', self.source_store), self.assertRaisesRegex(RuntimeError, 'provenance differs'):
            campaign.shared_logs(self.root, 'alfworld', 'target')

    def test_target_requires_complete_own_environment_source_checkpoint(self):
        self.add_adaptation('alfworld', 'source')
        self.add_adaptation('alfworld', 'target')
        states, _, mock_store = self.prepare_source_states()
        source = self.root / 'source' / 'alfworld'
        for mutation in ('status', 'rounds', 'environment'):
            with self.subTest(mutation=mutation):
                state = copy.deepcopy(states[str(source)])
                write(source / 'status.json', {'status': 'searching' if mutation == 'status' else 'complete'})
                if mutation == 'rounds':
                    state['next_round'] -= 1
                if mutation == 'environment':
                    state['binding']['context']['environment'] = 'textworld'
                write(source / 'state.json', state)
                with patch.object(campaign, 'RunStore', mock_store), \
                     patch.object(campaign, 'evaluator', self.native_factory), self.assertRaises(RuntimeError):
                    campaign.search(self.root, 'alfworld', 'target', 'no_external_em', self.tokenizer, 'mock-model')
        self.assertEqual(self.natives, [])

    def test_source_checkpoint_rejects_v2_and_unversioned_banks_before_loading_program(self):
        source = self.root / 'source' / 'alfworld'
        write(source / 'status.json', {'status': 'complete'})
        for workflow in ('code-memory-hooks-v2', None):
            binding = {'context': {'environment': 'alfworld'}}
            if workflow is not None:
                binding['workflow'] = workflow
            write(source / 'state.json', dict(next_round=campaign.ROUNDS, binding=binding))
            with self.subTest(workflow=workflow), patch.object(campaign, 'RunStore') as store, \
                 self.assertRaisesRegex(RuntimeError, 'different workflow'):
                campaign.source_checkpoint(self.root, 'alfworld')
            store.assert_not_called()

    def test_generalization_rejects_cache_receipt_with_wrong_key(self):
        self.prepare_generalization()
        mock_store = self.source_store
        with patch.object(campaign, 'RunStore', mock_store), patch.object(campaign, 'evaluator', self.native_factory):
            campaign.generalization(self.root, 'alfworld', self.tokenizer, 'mock-model')
            cache = next((self.root / 'generalization' / 'alfworld' / 'episodes').glob('*.json'))
            item = read(cache); item['key'] = 'different-program-or-task-key'; write(cache, item)
            with self.assertRaises((RuntimeError, ValueError)):
                campaign.generalization(self.root, 'alfworld', self.tokenizer, 'mock-model')

    def report_shared_fixture(self):
        self.add_adaptation('alfworld', 'source')
        self.add_adaptation('alfworld', 'target')
        source_logs = campaign.adaptation(self.root, 'alfworld', 'source')
        target_logs = campaign.adaptation(self.root, 'alfworld', 'target')
        logs = source_logs + target_logs
        states, _, _ = self.prepare_source_states()
        state = states[str(self.root / 'source' / 'alfworld')]
        program = Program(FIRST_SOURCE)
        path = self.root / 'source' / 'alfworld' / 'programs' / (program.hash + '.py')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(program.source)
        checkpoint = dict(source_environment='alfworld', source_state_hash=digest(state),
                          inherited_program_hash=program.hash, source_adaptation_hash=digest(source_logs))
        source = dict(current_program=program.hash, frozen_plan={'adaptation_hash': digest(source_logs)},
                      binding={'logs_hash': digest(source_logs)})
        targets = {}
        for condition in campaign_report.CONDITIONS:
            targets[condition] = dict(initial_program=program.hash, current_program=program.hash,
                frozen_plan=dict(initial_program_hash=program.hash, inherited_checkpoint=checkpoint,
                    adaptation_hash=digest(logs), memory_scope='environment',
                    task_memory_families=['clean', 'two'], task_memory_shared_across_types=True),
                binding=dict(logs_hash=digest(logs), ordered_logs_hash=digest(json.dumps(logs, ensure_ascii=False, sort_keys=False)),
                    source_memory_hash=digest([]), context={'environment': 'alfworld', 'memory_scope': 'environment'}))
        generalization = dict(binding=dict(logs_hash=digest(logs), inherited_checkpoint=checkpoint,
            versions={key: program.hash for key in (campaign_report.INHERITED,) + campaign_report.CONDITIONS},
            memory_scope='environment', task_memory_families=['clean', 'two'], sealed_from_review_and_evolver=True),
            no_heldout_writes=True)
        records = {r: read(self.root / 'collection' / 'alfworld' / (r + '.json'))
                   for r in ('source_adaptation', 'target_adaptation')}
        manifests = {r: campaign.tasks(self.root, 'alfworld', r) for r in campaign_report.ROLES}
        return dict(reader=campaign_report.Reader(self.root), environment='alfworld', manifests=manifests,
                    collection_records=records, source=source, targets=targets, generalization=generalization)

    def test_report_audits_shared_logs_inherited_checkpoint_and_sealed_versions(self):
        fixture = self.report_shared_fixture()
        report = campaign_report.shared_memory_audit(**fixture)
        self.assertTrue(all(value is True for value in report['checks'].values()))
        self.assertTrue(all(value is True for checks in report['targets'].values() for value in checks.values()))
        self.assertEqual(report['shared_log_count'], 4)
        self.assertIn('not total transfer', report['interpretation'])
        changes = [
            ('inheritance', 'inherits_current_source_program'),
            ('checkpoint', 'inherited_checkpoint_matches'),
            ('log_order', 'shared_ordered_logs_match'),
            ('environment', 'environment_memory_scope'),
            ('unexpected_em', 'expected_source_em_matches'),
        ]
        for mutation, check in changes:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(fixture)
                run = bad['targets']['no_external_em']
                if mutation == 'inheritance':
                    run['initial_program'] = Program(BASELINE_SOURCE).hash
                elif mutation == 'checkpoint':
                    run['frozen_plan']['inherited_checkpoint']['source_state_hash'] = 'wrong-source-state'
                elif mutation == 'log_order':
                    run['binding']['ordered_logs_hash'] = 'target-before-source'
                elif mutation == 'environment':
                    run['binding']['context']['environment'] = 'textworld'
                else:
                    run['binding']['source_memory_hash'] = digest([design_record('forbidden-source')])
                self.assertFalse(campaign_report.shared_memory_audit(**bad)['targets']['no_external_em'][check])
        fixture['generalization']['binding']['versions'][campaign_report.INHERITED] = Program(BASELINE_SOURCE).hash
        fixture['generalization']['no_heldout_writes'] = False
        report = campaign_report.shared_memory_audit(**fixture)
        self.assertFalse(report['checks']['generalization_versions_match'])
        self.assertFalse(report['checks']['generalization_no_heldout_writes'])

    def test_report_linked_collection_is_imported_once_and_keeps_original_costs_separate(self):
        with tempfile.TemporaryDirectory(prefix='mock-historical-native-artifacts-') as directory:
            origin_root = Path(directory)
            origin = origin_root / 'collection' / 'alfworld'
            code = {'phase1_codegen/native.py': b'# mock original native code\n',
                    'evomem/alma_backend.py': b'# mock original bridge\n'}
            archive = origin_root / 'snapshot.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name, content in code.items():
                    z.writestr(name, content)
            hashes = {name: campaign_report.hashlib.sha256(content).hexdigest() for name, content in code.items()}
            manifest = origin_root / 'snapshot.json'
            write(manifest, dict(source_hashes=hashes, zip_hash=file_hash(archive)))
            selected = {r: campaign.tasks(self.root, 'alfworld', r) for r in ('source_adaptation', 'target_adaptation')}
            binding = dict(environment='alfworld', tasks=selected,
                           own_code={'native.py': hashes['phase1_codegen/native.py']},
                           executor_bridge=hashes['evomem/alma_backend.py'])
            write(origin / 'binding.json', binding)
            all_episodes, all_costs = [], []
            for role, batch in selected.items():
                episodes = [native_shaped_test_episode(t, role) for t in batch]
                all_episodes += episodes
                write(origin / (role + '.json'), dict(episodes=episodes, complete=True, fixture_origin='mock-only-test'))
                for item, episode in zip(batch, episodes):
                    write(origin / 'episodes' / (digest(item) + '.json'),
                          dict(task_hash=digest(item), binding_hash=digest(binding), episode=episode))
                    all_costs.append(dict(kind='model', role='actor', task_id=item['id'], phase=role,
                                          usage={'prompt_tokens': 10, 'completion_tokens': 2}, fixture_origin='mock-only-test'))
            native = origin / 'runner' / 'native'; native.mkdir(parents=True)
            (native / 'canonical_episodes.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in all_episodes))
            (native / 'costs.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in all_costs))
            link = self.root / 'collection' / 'alfworld'; link.parent.mkdir(parents=True)
            link.symlink_to(origin, target_is_directory=True)
            write(self.root / 'collection_origins.json', {'alfworld': dict(source_collection=str(origin),
                original_code_snapshot=str(archive), code_snapshot_sha256=file_hash(archive),
                snapshot_manifest=str(manifest), new_model_execution=False)})
            report = campaign_report.build_report(self.root)
            self.assertEqual(report['status'], 'in_progress')
            self.assertEqual(report['physical_native']['total']['real_rows'], 0)
            self.assertEqual(report['physical_native']['imported_existing_real_rows'], 4)
            self.assertEqual(report['model_costs']['by_role'], {})
            self.assertEqual(len(report['imported_evidence']), 2)
            self.assertTrue(all(i['historical_costs_for_this_role_only']['actor']['model_attempts'] == 2
                                for i in report['imported_evidence']))
            self.assertEqual(report['integrity_findings'], [])
            text = campaign_report.markdown(report)
            self.assertIn('继承源机制', text)
            self.assertIn('不计为本目录新增执行', text)
            cache = origin / 'episodes' / (digest(selected['source_adaptation'][0]) + '.json')
            changed = read(cache); changed['episode']['score'] = 0.9; write(cache, changed)
            report = campaign_report.build_report(self.root)
            self.assertTrue(any(i['check'] == 'original_episode_caches_match' for i in report['integrity_findings']))


class EnvironmentBudgetAndRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-engine-budget-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_formal_completion_uses_declared_textworld_80_or_alfworld_30_step_budget(self):
        # Formal-shaped validation fixtures only; no evaluator is invoked.
        adaptation = fixture_log()
        adaptation.update(synthetic=False, native_reward_verified=True)
        selected = [task('exploration-a')]
        proposer = SyntheticProposer([FIRST_SOURCE]); proposer.fingerprint = 'mock-formal-proposer'
        reviewer = SyntheticReviewer([])
        for limit, environment in ((80, 'textworld'), (30, 'alfworld')):
            native = SimpleNamespace(fingerprint='mock-formal-native-' + environment, step_limit=limit)
            engine = EvolutionEngine(self.root / environment, TinyTokenizer(), proposer, native,
                EvolutionMemory(), {'environment': environment, 'family': 'synthetic-clean'},
                [dict(adaptation, environment=environment)], selected, [], reviewer=reviewer, synthetic=False, max_rounds=1)
            self.assertEqual(engine.binding['environment_step_limit'], limit)
            def readonly_episode(steps):
                return dict(native_shaped_test_episode(selected[0], steps=steps),
                            generated_memory_write_calls=0, read_only_memory=True)
            self.assertTrue(engine._complete([readonly_episode(limit)], selected))
            self.assertFalse(engine._complete([readonly_episode(limit + 1)], selected))
            self.assertEqual(engine._complete([readonly_episode(40)], selected), limit == 80)
            for field, value in [('generated_memory_write_calls', 1), ('read_only_memory', False)]:
                invalid = dict(readonly_episode(limit), **{field: value})
                self.assertFalse(engine._complete([invalid], selected))

    def test_retrieval_receipts_distinguish_external_design_records_from_local_trials(self):
        class ApproximateTokenizer(TinyTokenizer):
            def count(self, text):
                return (len(text) + 3) // 4
            def clip(self, text, count):
                return text[:max(0, count) * 4]
        source = design_record('external-source-design', environment='alfworld', synthetic=True)
        proposer = SyntheticProposer([FIRST_SOURCE, SECOND_SOURCE])
        reviewer = SyntheticReviewer([])
        engine = EvolutionEngine(self.root, ApproximateTokenizer(), proposer, SyntheticEvaluator(),
            EvolutionMemory([source]), {'environment': 'alfworld', 'family': 'synthetic-clean'}, [fixture_log()],
            [task('exploration-a'), task('exploration-b')], [], reviewer=reviewer,
            synthetic=True, max_rounds=2)
        state = engine.run(2)
        trials = RunStore(self.root).history(state)
        first, second = [t['evolution_memory_retrieval'] for t in trials]
        self.assertEqual(first['source_records_available'], 1)
        self.assertEqual(first['local_records_available'], 0)
        self.assertEqual(first['selected_source_ids'], [source['id']])
        self.assertEqual(first['selected_local_ids'], [])
        self.assertEqual(second['source_records_available'], 1)
        self.assertEqual(second['local_records_available'], 1)
        self.assertIn(source['id'], second['selected_source_ids'])
        self.assertNotIn(trials[0]['evolution_event']['id'], second['selected_local_ids'])
        self.assertTrue(any(row['id'] == trials[0]['evolution_event']['id']
                            and row['reason'] == 'already_provided_as_previous_attempt'
                            for row in second['ranked_records']))
        self.assertFalse(set(second['selected_source_ids']) & set(second['selected_local_ids']))
        self.assertEqual(set(second['selected_ids']), set(second['selected_source_ids']) | set(second['selected_local_ids']))
        self.assertEqual(second['token_budget'], 3072)
        self.assertEqual(read(self.root / 'trials' / '001' / 'retrieval.json'), second)
        self.assertIn('not proof', second['evidence_scope'])


if __name__ == '__main__':
    unittest.main()
