"""Synthetic CPU-only persistence checks; no native execution or model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from evomem.common import digest, read, write
from phase1_codegen.learning import (
    begin_learning, finish_learning, learning_key, learning_receipt_ref, load_snapshot,
    make_snapshot, ordered_digest, ordered_wire, read_learning, save_snapshot,
    snapshot_logs, snapshot_memory, validate_transition,
)


def episode(task_id, *, environment='alfworld', family='heat', phase='learning/round-0'):
    return dict(task_id=task_id, group_id='group-' + task_id, family=family,
                environment=environment, phase=phase, error=None, score=0., synthetic=True,
                history=[{'observation': 'Synthetic observation', 'action': 'look'}])


def runtime(program='parent', *, state=None, environment='alfworld'):
    """Synthetic exported-state fixture, not an execution receipt."""
    state = {'z': 1, 'a': 2} if state is None else state
    context = dict(memory_scope='environment', environment=environment)
    value = dict(version='code-memory-state-v1', program_hash=program,
                 state_wire=ordered_wire(state), context_wire=ordered_wire(context),
                 memory_hash=digest(dict(program=program, state=ordered_wire(state))),
                 environment=environment, memory_scope='environment')
    return dict(value, snapshot_hash=digest(value))


class ContinualLearningPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-continual-memory-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = [episode('bootstrap')]
        self.initial = make_snapshot(self.logs, 'alfworld', memory_snapshot=runtime())
        self.initial_ref = save_snapshot(self.root, self.initial)
        self.task = dict(id='live', group_id='group-live', family='heat', environment='alfworld')
        self.binding = dict(task_memory_mode='continual', synthetic=True, context={'environment': 'alfworld'},
                            learning_blocks=[[self.task]], initial_task_memory_snapshot=self.initial_ref)

    def request(self, *, snapshot=None, phase='learning/round-0'):
        return dict(binding_hash=digest(self.binding), program_hash='parent',
                    snapshot_hash=(snapshot or self.initial_ref)['hash'], task=self.task, phase=phase)

    def record(self):
        request = self.request()
        key = learning_key(self.binding, 'parent', self.initial_ref['hash'], self.task, request['phase'])
        begin_learning(self.root, key, request)
        updated_runtime = runtime(state={'learned': 1})
        ep = dict(episode('live'), program_hash='parent', memory_hash_before=self.initial['memory_hash'],
                  memory_hash_after=updated_runtime['memory_hash'])
        result = make_snapshot(self.logs + [ep], 'alfworld', memory_snapshot=updated_runtime,
                               parent_hash=self.initial_ref['hash'])
        after = save_snapshot(self.root, result)
        ref = finish_learning(self.root, key, request, ep, after)
        trial = dict(round=0, parent_hash='parent', candidate_hash='candidate', accepted=False,
                     task_memory_before=self.initial_ref, task_memory_evaluation=after,
                     task_memory_after=after, learning_receipts=[ref])
        return key, request, ep, after, ref, trial

    def test_snapshot_round_trip_preserves_nested_key_order_and_is_detached(self):
        loaded = load_snapshot(self.root, self.initial_ref)
        self.assertEqual(list(json.loads(snapshot_memory(loaded)['state_wire'])), ['z', 'a'])
        self.assertEqual(ordered_digest(snapshot_logs(loaded)), ordered_digest(self.logs))
        altered = snapshot_logs(loaded)
        altered[0]['history'].append({'action': 'not executed'})
        self.assertEqual(snapshot_logs(load_snapshot(self.root, self.initial_ref)), self.logs)
        reordered = dict(reversed(list(self.logs[0].items())))
        alternative = make_snapshot([reordered], 'alfworld', memory_snapshot=runtime())
        self.assertEqual(alternative['logs_hash'], self.initial['logs_hash'])
        self.assertNotEqual(alternative['ordered_logs_hash'], self.initial['ordered_logs_hash'])

    def test_snapshot_detects_hash_state_and_reference_tampering(self):
        path = self.root / self.initial_ref['path']
        altered = read(path)
        altered['logs_wire'] = ordered_wire([episode('another-task')])
        write(path, altered)
        with self.assertRaises(ValueError):
            load_snapshot(self.root, self.initial_ref)
        write(path, self.initial)
        for field in ('hash', 'logs_hash', 'ordered_logs_hash'):
            changed = dict(self.initial_ref, **{field: 'changed'})
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_snapshot(self.root, changed)
        with self.assertRaises(ValueError):
            load_snapshot(self.root, dict(self.initial_ref, path='../foreign.json'))

    def test_foreign_sealed_duplicate_and_error_experiences_fail_closed(self):
        cases = [[episode('x', environment='textworld')], [episode('x', phase='sealed_generalization')],
                 [episode('x'), episode('x')], [dict(episode('x'), error='RuntimeFailure')]]
        for logs in cases:
            with self.subTest(logs=logs), self.assertRaises(ValueError):
                make_snapshot(logs, 'alfworld', memory_snapshot=runtime())
        bad = runtime()
        bad['state_wire'] = '{"silently_changed":true}'
        with self.assertRaises(ValueError):
            make_snapshot(self.logs, 'alfworld', memory_snapshot=bad)

    def test_complete_receipt_resume_has_stable_ref_and_rejects_duplicate_start(self):
        key, request, ep, after, ref, _ = self.record()
        receipt = read_learning(self.root, key, request)
        self.assertEqual(learning_receipt_ref(receipt), ref)
        self.assertEqual(receipt['episode'], ep)
        self.assertEqual(receipt['output_snapshot_ref'], after)
        self.assertEqual(finish_learning(self.root, key, request, ep, after), ref)
        with self.assertRaisesRegex(RuntimeError, 'already has'):
            begin_learning(self.root, key, request)

    def test_interrupted_attempt_is_not_silently_reexecuted_and_failure_is_preserved(self):
        request = self.request()
        key = learning_key(self.binding, 'parent', self.initial_ref['hash'], self.task, request['phase'])
        self.assertIsNone(read_learning(self.root, key, request))
        begin_learning(self.root, key, request)
        with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
            read_learning(self.root, key, request)
        failed = dict(episode('live'), error='OutputTruncated')
        ref = finish_learning(self.root, key, request, failed, None)
        receipt = read_learning(self.root, key, request)
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['episode']['error'], 'OutputTruncated')
        self.assertIsNone(receipt['episode']['score'])
        self.assertFalse(receipt['episode']['native_reward_verified'])
        self.assertEqual(learning_receipt_ref(receipt), ref)
        self.assertEqual(snapshot_logs(load_snapshot(self.root, self.initial_ref)), self.logs)

    def test_cache_binding_changes_with_ordered_task_snapshot_program_or_mode(self):
        key = learning_key(self.binding, 'parent', self.initial_ref['hash'], self.task, 'learning')
        reordered = dict(reversed(list(self.task.items())))
        changes = [learning_key(self.binding, 'candidate', self.initial_ref['hash'], self.task, 'learning'),
                   learning_key(self.binding, 'parent', 'different-snapshot', self.task, 'learning'),
                   learning_key(self.binding, 'parent', self.initial_ref['hash'], reordered, 'learning'),
                   learning_key(dict(self.binding, task_memory_mode='frozen_replay'), 'parent',
                                self.initial_ref['hash'], self.task, 'learning')]
        self.assertNotIn(key, changes)

    def test_finished_receipt_detects_request_episode_and_pending_drift(self):
        key, request, _, _, ref, _ = self.record()
        with self.assertRaises(ValueError):
            read_learning(self.root, key, dict(request, phase='changed'))
        path = self.root / ref['path']
        original = read(path)
        altered = copy.deepcopy(original)
        altered['episode']['score'] = 1
        write(path, altered)
        with self.assertRaises(ValueError):
            read_learning(self.root, key, request)
        write(path, original)
        pending_path = self.root / 'learning/pending' / (key + '.json')
        pending = read(pending_path)
        pending['timestamp'] += 1
        write(pending_path, pending)
        with self.assertRaises(ValueError):
            read_learning(self.root, key, request)

    def test_transition_accepts_native_zero_score_and_exact_frontier_then_code_handoff(self):
        _, _, _, after, _, trial = self.record()
        self.assertEqual(validate_transition(trial, self.binding, self.initial_ref, root=self.root), after)
        candidate = make_snapshot(snapshot_logs(load_snapshot(self.root, after)), 'alfworld',
                                  memory_snapshot=runtime('candidate', state={'new_schema': 2}),
                                  parent_hash=after['hash'])
        candidate_ref = save_snapshot(self.root, candidate)
        write(self.root / 'behavior.json', dict(parent={'runtime_snapshot': snapshot_memory(load_snapshot(self.root, after))},
                                               candidate={'runtime_snapshot': snapshot_memory(candidate)}))
        accepted = dict(trial, accepted=True, task_memory_after=candidate_ref, behavior_path='behavior.json')
        self.assertEqual(validate_transition(accepted, self.binding, self.initial_ref, root=self.root), candidate_ref)
        changed = make_snapshot(snapshot_logs(candidate), 'alfworld',
                                memory_snapshot=runtime('candidate', state={'different_unassessed_bank': 1}),
                                parent_hash=after['hash'])
        with self.assertRaisesRegex(ValueError, 'exact parent/candidate'):
            validate_transition(dict(accepted, task_memory_after=save_snapshot(self.root, changed)),
                                self.binding, self.initial_ref, root=self.root)

    def test_transition_rejects_omitted_or_reordered_learning_and_wrong_initial_pointer(self):
        _, _, _, after, _, trial = self.record()
        with self.assertRaises(ValueError):
            validate_transition(dict(trial, learning_receipts=[]), self.binding, self.initial_ref, root=self.root)
        with self.assertRaises(ValueError):
            validate_transition(trial, self.binding, after, root=self.root)
        wrong_block = dict(self.binding, learning_blocks=[[dict(self.task, id='not-live')]])
        with self.assertRaises(ValueError):
            validate_transition(trial, wrong_block, self.initial_ref, root=self.root)

    def test_candidate_trajectories_and_unaccepted_state_changes_cannot_enter_history(self):
        _, _, _, after, _, trial = self.record()
        polluted = make_snapshot(snapshot_logs(load_snapshot(self.root, after)) + [episode('candidate-only')],
                                 'alfworld', memory_snapshot=runtime('candidate'), parent_hash=after['hash'])
        bad_ref = save_snapshot(self.root, polluted)
        with self.assertRaises(ValueError):
            validate_transition(dict(trial, accepted=True, task_memory_after=bad_ref), self.binding,
                                self.initial_ref, root=self.root)
        changed = make_snapshot(snapshot_logs(load_snapshot(self.root, after)), 'alfworld',
                                memory_snapshot=runtime(state={'unaccepted': 1}), parent_hash=after['hash'])
        with self.assertRaises(ValueError):
            validate_transition(dict(trial, task_memory_after=save_snapshot(self.root, changed)), self.binding,
                                self.initial_ref, root=self.root)


if __name__ == '__main__':
    unittest.main()
