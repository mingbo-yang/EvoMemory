"""Synthetic same-dataset shared-history contracts; no model/environment calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evomem.common import digest, dumps
from phase1_codegen.roles.evolver import CodeEvolver, bounded_shared_adaptation, ADAPTATION_INPUT_POLICY
from tests.core.test_review import CharacterTokenizer, FakeClient, FakeResponse
from phase1_codegen.engine import EvolutionEngine, own_trace, public_context
from phase1_codegen.experience import EvolutionMemory
from phase1_codegen.program import BASELINE_SOURCE, Program
from tests.core.test_engine import FIRST_SOURCE, SECOND_SOURCE, SyntheticEvaluator, SyntheticProposer, TinyTokenizer, fixture_log, task
from tests.core.test_review_loop import SyntheticReviewer


DIAGNOSTIC_SOURCE = '''def on_event(state, event, context):
    memory = None
    if event.get("type") == "task_end":
        families = state.get("families", [])
        families.append(event["experience"]["family"])
        state = {"families": families}
    if event.get("type") == "task_start":
        text = ",".join(state.get("families", [])) + "|" + context.get("family", "missing-family")
        text = text + "|" + context.get("task_id", "missing-task")
        memory = [{"id": "shared-bank", "experience": text}]
    return {"state": state, "session": event.get("session", {}), "memory": memory}
''' 


def shared_log(name, family):
    value = fixture_log()
    value.update(task_id=name, group_id='group-' + name, room='room-' + name,
                 environment='alfworld', family=family)
    return value


def shared_task(name, family='two', environment=None):
    value = task(name)
    value['family'] = family
    if environment is not None:
        value['environment'] = environment
    return value


def same_dataset_record():
    return dict(id='source-mechanism-trial', kind='code_change_trial',
                observed_context={'environment': 'alfworld', 'family': 'heat'},
                hypothesis='Synthetic design record only.', accepted=False, status='not_accepted', synthetic=True)


class SharedScopeEngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-shared-scope-engine-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.logs = [shared_log('source-heat-adaptation', 'heat'), shared_log('target-two-adaptation', 'two')]
        self.explore = [shared_task('target-exploration-a'), shared_task('target-exploration-b')]
        self.context = {'environment': 'alfworld', 'memory_scope': 'environment', 'family': 'two'}
        self.records = [same_dataset_record()]

    def engine(self, *, logs=None, context=None, records=None, explore=None, initial=FIRST_SOURCE,
               proposer=None, evaluator=None, reviewer=None, out=None):
        return EvolutionEngine(
            out=self.root if out is None else out, tokenizer=TinyTokenizer(),
            proposer=proposer if proposer is not None else SyntheticProposer([SECOND_SOURCE]),
            evaluator=evaluator if evaluator is not None else SyntheticEvaluator(),
            reviewer=reviewer if reviewer is not None else SyntheticReviewer([]),
            evolution_memory=EvolutionMemory(self.records if records is None else records),
            context=self.context if context is None else context,
            logs=self.logs if logs is None else logs,
            exploration_tasks=self.explore if explore is None else explore,
            confirmation_blocks=[], initial_source=initial, seed=11, synthetic=True, max_rounds=1,
        )

    def test_explicit_environment_scope_accepts_multiple_families_and_inherited_source(self):
        engine = self.engine()
        state = engine.run(rounds=0)
        self.assertEqual(state['initial_program'], Program(FIRST_SOURCE).hash)
        self.assertEqual(state['current_program'], Program(FIRST_SOURCE).hash)
        self.assertEqual(state['binding']['context']['memory_scope'], 'environment')
        self.assertEqual({r['family'] for r in engine.logs}, {'heat', 'two'})
        self.assertEqual(engine.memory.records[0]['observed_context']['family'], 'heat')
        self.assertEqual(engine.proposer.calls, [])
        self.assertEqual(engine.evaluator.calls, [])

    def test_legacy_family_scope_still_rejects_mixed_task_memory(self):
        context = {'environment': 'alfworld', 'family': 'two'}
        with self.assertRaisesRegex(ValueError, 'one target family'):
            self.engine(context=context)
        single = self.engine(context=context, logs=[self.logs[1]])
        self.assertEqual(single.memory_scope, 'family')

    def test_environment_scope_requires_explicit_dataset_provenance_for_logs_and_records(self):
        missing_context = dict(self.context); missing_context.pop('environment')
        with self.assertRaisesRegex(ValueError, 'explicit dataset'):
            self.engine(context=missing_context)
        missing_log = copy.deepcopy(self.logs); missing_log[0].pop('environment')
        with self.assertRaisesRegex(ValueError, 'explicit matching dataset provenance'):
            self.engine(logs=missing_log)
        missing_record = copy.deepcopy(self.records)
        missing_record[0]['observed_context'].pop('environment')
        with self.assertRaisesRegex(ValueError, 'within its dataset'):
            self.engine(records=missing_record)
        with self.assertRaisesRegex(ValueError, 'memory_scope'):
            self.engine(context=dict(self.context, memory_scope='global'))

    def test_foreign_dataset_logs_evolution_memory_and_tasks_are_rejected(self):
        foreign_logs = copy.deepcopy(self.logs); foreign_logs[0]['environment'] = 'textworld'
        foreign_records = copy.deepcopy(self.records); foreign_records[0]['observed_context']['environment'] = 'textworld'
        foreign_tasks = copy.deepcopy(self.explore); foreign_tasks[0]['environment'] = 'textworld'
        for kwargs in ({'logs': foreign_logs}, {'records': foreign_records}, {'explore': foreign_tasks}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.engine(**kwargs)
        # Frozen historical ALF task manifests can omit environment: native
        # inventory validation supplies their provenance. Explicit foreign tags
        # are never silently overwritten.
        self.assertTrue(all('environment' not in row for row in self.engine().explore))

    def test_target_evaluation_family_stays_fixed_while_shared_history_spans_families(self):
        wrong_tasks = copy.deepcopy(self.explore); wrong_tasks[0]['family'] = 'heat'
        with self.assertRaisesRegex(ValueError, 'current target family'):
            self.engine(explore=wrong_tasks)

    def test_shared_history_cannot_absorb_confirmation_or_generalization_outcomes(self):
        for phase in ('final', 'confirmation', 'sealed_generalization', 'heldout_assessment'):
            with self.subTest(phase=phase):
                logs = copy.deepcopy(self.logs); logs[0]['phase'] = phase
                with self.assertRaisesRegex(ValueError, 'adaptation evidence'):
                    self.engine(logs=logs)

    def test_cross_type_sharing_does_not_relax_task_or_group_disjointness(self):
        for field, value in (('task_id', self.explore[0]['id']), ('group_id', self.explore[0]['group_id'])):
            with self.subTest(field=field):
                logs = copy.deepcopy(self.logs); logs[0][field] = value
                with self.assertRaisesRegex(ValueError, 'group-disjoint'):
                    self.engine(logs=logs)

    def test_shared_diagnostic_reads_use_each_frozen_task_view_without_mutating_the_bank(self):
        engine = self.engine(initial=DIAGNOSTIC_SOURCE)
        contexts = [{'goal': item['goal'], 'obs': 'Synthetic public context'} for item in self.explore]
        signature = engine._signature(Program(DIAGNOSTIC_SOURCE), contexts)
        self.assertEqual(signature['stored_state']['families'], ['heat', 'two'])
        self.assertEqual(len(signature['read_trace']), 2)
        for item, output, trace in zip(self.explore, signature['readouts'], signature['read_trace']):
            self.assertEqual(output['entries'][0]['experience'], 'heat,two|two|' + item['id'])
            self.assertEqual(trace['context']['family'], 'two')
            self.assertEqual(trace['context']['task_id'], item['id'])
            self.assertEqual(trace['context']['environment'], 'alfworld')
            self.assertEqual(trace['memory_hash'], signature['memory_hash'])
        with self.assertRaisesRegex(ValueError, 'align'):
            engine._signature(Program(DIAGNOSTIC_SOURCE), contexts[:1])

    def test_both_roles_see_shared_history_and_authoritative_inherited_mechanism(self):
        proposer = SyntheticProposer([SECOND_SOURCE])
        reviewer = SyntheticReviewer([])
        engine = self.engine(proposer=proposer, reviewer=reviewer)
        state = engine.run(rounds=1)
        self.assertEqual(state['initial_program'], Program(FIRST_SOURCE).hash)
        supplied = proposer.calls[0]['payload']
        self.assertEqual(supplied['current_source'], FIRST_SOURCE)
        self.assertEqual({row['family'] for row in supplied['own_adaptation']}, {'heat', 'two'})
        self.assertTrue(all(row['environment'] == 'alfworld' for row in supplied['own_adaptation']))
        self.assertEqual([r['id'] for r in supplied['evolution_memory']], ['source-mechanism-trial'])
        for call in reviewer.calls:
            self.assertNotIn('evolution_memory', call['payload'])
            self.assertNotIn('previous_post_review', call['payload'])
        for payload in [supplied] + [call['payload'] for call in reviewer.calls]:
            context = payload['memory_context']
            self.assertEqual(context['scope'], 'environment')
            self.assertEqual(context['experience_families'], ['heat', 'two'])
            self.assertEqual(context['starting_program_hash'], Program(FIRST_SOURCE).hash)
            self.assertFalse(context['starting_program_is_baseline'])
        self.assertEqual(state['current_program'], Program(SECOND_SOURCE).hash)

    def test_diagnostic_summary_preserves_environment_scope_and_family_provenance(self):
        observed = public_context(self.context, self.logs, [])
        self.assertEqual(observed['memory_scope'], 'environment')
        self.assertEqual(observed['memory_families'], ['heat', 'two'])
        self.assertEqual(observed['family'], 'two')
        self.assertEqual(own_trace(self.logs[0])['environment'], 'alfworld')


class SharedAdaptationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-shared-adaptation-budget-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tokenizer = CharacterTokenizer()
        source = shared_log('source-history', 'heat')
        source['goal'] = 'SOURCE_TASK_MARKER'
        source['history'] = [{'observation': 'long source context ' * 5000, 'action': 'look'}]
        target = shared_log('target-history', 'two')
        target['goal'] = 'TARGET_TASK_MARKER'
        self.logs = [source, target]

    def captured_request(self, context, logs):
        value = dict(source=BASELINE_SOURCE, hypothesis='Synthetic fixture.', risk='No task evidence.')
        client = FakeClient([FakeResponse(value)])
        original = dict(current_source=BASELINE_SOURCE, context=context, own_adaptation=logs)
        snapshot = copy.deepcopy(original)
        with patch('httpx.Client', return_value=client):
            proposer = CodeEvolver(self.tokenizer, self.root / 'costs.jsonl', 'synthetic-model')
        self.assertEqual(proposer(original, 11), value)
        self.assertEqual(original, snapshot)
        displayed = json.loads(client.requests[0][1]['messages'][1]['content'])
        proposer.close()
        return displayed['own_adaptation']

    def test_long_source_does_not_displace_target_and_total_includes_wrappers(self):
        displayed = self.captured_request(
            {'environment': 'alfworld', 'family': 'two', 'memory_scope': 'environment'}, self.logs)
        self.assertEqual(displayed['selection_policy'], ADAPTATION_INPUT_POLICY)
        self.assertLessEqual(self.tokenizer.count(dumps(displayed)), 4000)
        self.assertEqual(displayed['token_budget'], 4000)
        self.assertEqual([row['family'] for row in displayed['families']], ['heat', 'two'])
        self.assertEqual(len({row['evidence_token_budget'] for row in displayed['families']}), 1)
        source, target = displayed['families']
        self.assertTrue(source['clipped'])
        self.assertFalse(target['clipped'])
        self.assertIn('SOURCE_TASK_MARKER', source['serialized_evidence'])
        self.assertIn('TARGET_TASK_MARKER', target['serialized_evidence'])
        self.assertEqual(source['original_hash'], digest([self.logs[0]]))
        self.assertEqual(target['original_hash'], digest([self.logs[1]]))
        self.assertEqual(displayed['original_hash'], digest(self.logs))

    def test_family_order_is_deterministic_and_within_family_order_is_not_selected_by_reward(self):
        failure = shared_log('first-failed-source', 'heat'); failure['score'] = 0
        success = shared_log('later-successful-source', 'heat'); success['score'] = 1
        target = shared_log('target-history', 'two')
        logs = [target, failure, success]
        snapshot = copy.deepcopy(logs)
        first = bounded_shared_adaptation(logs, self.tokenizer)
        second = bounded_shared_adaptation(logs, self.tokenizer)
        self.assertEqual(first, second)
        self.assertEqual(logs, snapshot)
        self.assertEqual([row['family'] for row in first['families']], ['heat', 'two'])
        source_rows = json.loads(first['families'][0]['serialized_evidence'])
        self.assertEqual([row['task_id'] for row in source_rows], ['first-failed-source', 'later-successful-source'])
        self.assertEqual([row['score'] for row in source_rows], [0, 1])
        self.assertLessEqual(self.tokenizer.count(dumps(first)), 4000)

    def test_legacy_and_single_family_inputs_keep_the_original_clipping_contract(self):
        for context, logs in [({'environment': 'alfworld', 'memory_scope': 'family'}, self.logs),
                              ({'environment': 'alfworld', 'memory_scope': 'environment'}, self.logs[:1])]:
            with self.subTest(context=context, count=len(logs)):
                displayed = self.captured_request(context, logs)
                raw = dumps(logs)
                self.assertEqual(displayed, dict(serialized_evidence=self.tokenizer.clip(raw, 4000),
                    clipped=self.tokenizer.count(raw) > 4000, original_hash=digest(logs)))

    def test_shared_adaptation_selection_policy_is_bound_into_evolver_fingerprint(self):
        with patch('httpx.Client', return_value=FakeClient([])):
            original = CodeEvolver(self.tokenizer, self.root / 'first.jsonl', 'synthetic-model')
        with patch('phase1_codegen.roles.evolver.ADAPTATION_INPUT_POLICY', 'different-fixed-selection-policy'), \
             patch('httpx.Client', return_value=FakeClient([])):
            changed = CodeEvolver(self.tokenizer, self.root / 'second.jsonl', 'synthetic-model')
        self.assertNotEqual(original.fingerprint, changed.fingerprint)
        original.close(); changed.close()


if __name__ == '__main__':
    unittest.main()
