"""CPU contract checks. Constructed episodes below are fixtures, not research outcomes."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from evomem.common import file_hash
from phase1_codegen.hooks import HookSession
from phase1_codegen.program import Program
from rsi_paired.native import BASELINE_SOURCE, Native, _archive_episode, _evidence, _prefix, _wire
from rsi_paired.panel import COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL, SEQUENTIAL_PROTOCOL
from rsi_paired.workflow import LearningMechanismError, PROTOCOL, _Journal
from phase1_codegen.policy import Rejected


class Tokens:
    def count(self, value):
        return len(value.split())

    def clip(self, value, limit):
        return ' '.join(value.split()[:limit])


class NativeMemoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.native = object.__new__(Native)
        self.native.root = Path(temp.name)
        self.native.tokenizer = Tokens()
        self.native.capabilities = None
        self.native.fingerprint = 'fixture'
        self.native.engine = SimpleNamespace(check_tasks=lambda tasks: None)

    def test_original_order_episode_bytes_survive_scoring_and_resume(self):
        ep = {'score': 0, 'goal': 'synthetic', 'info': {'z': 'first', 'a': 'second'}}
        task = {'id': 'fixture', 'family': 'cool'}
        program = Program(BASELINE_SOURCE)
        self.native.job = 'synthetic-job'
        def execute(directory, program, bank, before, task, seed, label):
            attempt = directory / 'attempt-0'
            attempt.mkdir()
            (attempt / 'canonical_episodes.jsonl').write_text(json.dumps(ep, sort_keys=True) + '\n')
            (attempt / 'actor.jsonl').touch()
            identity = _archive_episode(attempt, ep, self.native.root / 'learning')
            self.assertEqual((attempt / 'episode.json').read_bytes(), _wire(ep))
            self.assertEqual((self.native.root / 'learning' / (identity + '.json')).read_bytes(), _wire(ep))
            self.assertNotEqual(_wire(ep), _wire(json.loads((attempt / 'canonical_episodes.jsonl').read_text())))
            return dict(row=dict(episode_hash=identity, status='completed'), attempt=attempt.name,
                        snapshot=before, canonical_hash=file_hash(attempt / 'canonical_episodes.jsonl'),
                        actor_hash=file_hash(attempt / 'actor.jsonl'), episode_wire='episode.json')
        self.native._execute = Mock(side_effect=execute)
        args = (program, {}, task, 42, 'panel', 'test-panel', 0)
        first = self.native._episode(*args)
        self.assertEqual(self.native._episode(*args), first)
        wire, = (self.native.root / 'episodes').glob('*/attempt-0/episode.json')
        wire.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'receipt contents changed'):
            self.native._episode(*args)
        self.assertEqual(self.native._execute.call_count, 1)

    def archive_learning(self, family, task_id):
        task = {'id': task_id, 'family': family, 'environment': 'alfworld'}
        bank = self.native._bank(Program(BASELINE_SOURCE), {}, family)
        session = HookSession(bank, task, allow_write=True)
        ep = dict(task_id=task_id, family=family, environment='alfworld', phase='learning',
                  native_reward_verified=True, score=0.0, error=None, goal='fixture observation',
                  seed=42, done=False, history=[], program_hash=Program(BASELINE_SOURCE).hash)
        session.emit('task_start', dict(step=0, visible={'obs': 'visible fixture'}))
        session.finish(ep)
        session.publish()
        ep['memory_hook'] = session.summary()
        raw = _wire(ep)
        identity = hashlib.sha256(raw).hexdigest()
        path = self.native.root/'learning'/f'{identity}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        bank.source_log_hash = _prefix(bank.source_log_hash, identity)
        row = dict(task_id=task_id, family=family, score=0, phase='learning', episode_hash=identity)
        return bank.export_snapshot(), row

    def test_learning_evidence_uses_program_from_verified_archive(self):
        _, row = self.archive_learning('cool', 'cool/a')
        claimed = dict(row, program_hash='wrong-source')
        evidence, = self.native.verified_learning_evidence([claimed])
        self.assertEqual(evidence, dict(row, program_hash=Program(BASELINE_SOURCE).hash))
        self.assertEqual(claimed['program_hash'], 'wrong-source')

    def test_identical_source_reuses_and_changed_source_replays_same_family_prefix(self):
        ice, a = self.archive_learning('cool', 'cool/a')
        heat, b = self.archive_learning('heat', 'heat/b')
        memory = {'heat': heat, 'cool': ice}
        self.assertEqual(self.native.prepare(BASELINE_SOURCE, BASELINE_SOURCE, memory, [a, b]), memory)
        candidate = '''def on_event(state, event, context):
    if event["type"] == "task_end":
        state["ids"] = state.get("ids", []) + [event["experience"]["task_id"]]
    return {"state": state, "session": {}, "memory": None}
'''
        prepared = self.native.prepare(candidate, BASELINE_SOURCE, memory, [a, b])
        self.assertEqual(json.loads(prepared['cool']['state_wire']), {'ids': ['cool/a']})
        self.assertEqual(json.loads(prepared['heat']['state_wire']), {'ids': ['heat/b']})
        self.assertEqual(prepared['heat']['source_log_hash'], heat['source_log_hash'])
        self.assertNotEqual(prepared['heat']['program_hash'], heat['program_hash'])
        with self.assertRaisesRegex(ValueError, 'non-learning'):
            self.native.prepare(candidate, BASELINE_SOURCE, memory, [dict(a, phase='panel'), b])
        with self.assertRaisesRegex(ValueError, 'learning prefix'):
            self.native.prepare(candidate, BASELINE_SOURCE, memory, [dict(a, episode_hash='0'*64), b])

    def test_panel_updates_private_bank_in_order_and_does_not_modify_input(self):
        initial = {'cool': {'value': 8}}
        calls = []

        def episode(program, bank, task, seed, phase, panel, index):
            self.assertEqual(phase, 'panel')
            calls.append((bank['cool']['value'], panel, seed))
            return {'cool': {'value': bank['cool']['value']+1}}, {'task_id': task['id'], 'score': 0}

        self.native._episode = episode
        rows = self.native.evaluate(BASELINE_SOURCE, initial, [{'id': 'a'}, {'id': 'b'}], [40, 42])
        self.assertEqual([r['task_id'] for r in rows], ['a', 'b'])
        self.assertEqual([(x[0], x[2]) for x in calls], [(8, 40), (9, 42)])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(initial['cool']['value'], 8)

    def test_deployed_learning_hook_has_a_durable_unscored_failure_without_retry(self):
        self.assert_learning_failure(SEQUENTIAL_PROTOCOL)

    def test_six_round_coevolution_deployed_learning_failure_is_also_unscored(self):
        self.assert_learning_failure(COEVOLUTION_6X30_PROTOCOL)

    def test_heat_coevolution_deployed_learning_failure_is_also_unscored(self):
        self.assert_learning_failure(COEVOLUTION_HEAT_5X23_PROTOCOL)

    def assert_learning_failure(self, protocol):
        # The environment shell is a fixture. HookSession executes the bad
        # deployed source in the real isolated worker and produces the rejection.
        task = dict(id='fixture', family='heat' if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL else 'cool',
                    environment='alfworld', seed=7,
                    archive_member='json_2.1.1/train/group/trial/game.tw-pddl')
        source = 'def on_event(state, event, context):\n    return state["absent"]\n'
        attempts = []
        def run(task, bank, phase, seed, allow_write):
            attempts.append(phase)
            hook = HookSession(bank, task, allow_write)
            with self.assertRaises(Rejected) as rejected:
                hook.emit('task_start', {'visible': {}, 'step': 0})
            ep = dict(task_id=task['id'], environment='alfworld', family=task['family'], seed=seed, environment_seed=7,
                      phase=phase, program_hash=bank.program.hash, memory_hash_before=bank.fingerprint,
                      memory_hash_after=bank.fingerprint, memory_hook=hook.summary(), actor_ledger_start=0,
                      actor_ledger_end=0, score=None, error=str(rejected.exception), generated_memory_write_calls=0,
                      executor='ALMA.Agent_Workflow.run_single_task', synthetic=False, native_reward_verified=False,
                      history=[], native_recorder={'steps': []})
            self.native.engine.runner.ledger.with_name('canonical_episodes.jsonl').write_text(json.dumps(ep) + '\n')
            return ep
        self.native.engine.runner = SimpleNamespace(episode=run)
        self.native.protocol = protocol
        journal = _Journal(self.native.root / 'batch', self.native, 2)
        for _ in range(2):
            with self.assertRaises(LearningMechanismError) as failure:
                journal.job('learning/000', task, lambda: self.native.learn(source, {}, task, 11))
        self.assertEqual(attempts, ['learning'])
        # If the process died before journaling the error, native recovery also
        # returns the same attested failure, without another environment execution.
        with self.assertRaises(LearningMechanismError) as recovered:
            self.native.learn(source, {}, task, 11)
        self.assertEqual(recovered.exception.evidence, failure.exception.evidence)
        self.assertEqual(attempts, ['learning'])
        directory, = (self.native.root / 'episodes').iterdir()
        result = json.loads((directory / 'learning_failure.json').read_text())['result']
        self.assertIsNone(result['row']['score'])
        self.assertIsNone(result['row']['native_score'])
        self.assertTrue((directory / 'attempt-0/episode.json').is_file())
        self.assertTrue((directory / 'attempt-0/actor.jsonl').is_file())
        self.assertFalse((directory / 'attempt-0/infrastructure.json').exists())
        self.assertFalse((directory / 'complete.json').exists())
        self.assertFalse((self.native.root / 'learning').exists())
        # The archived protocol keeps its original treatment of a learning error.
        self.native.protocol = PROTOCOL
        self.native.set_job(self.native.root / 'legacy/private')
        with self.assertRaisesRegex(RuntimeError, 'Native task incomplete'):
            self.native.learn(source, {}, task, 11)
        self.assertEqual(attempts, ['learning', 'learning'])

    def test_witness_requires_real_delivery_and_tracks_emptying_memory(self):
        bank = {'state_wire': '{}', 'stored_tokens': 1}
        delivered = {'entries': [{'id': 'h', 'experience': 'visible evidence'}]}
        ep = {'history': [], 'memory_hook': {'injections': [
            {'actor_delivery_verified': False, 'memory': delivered},
            {'actor_delivery_verified': True, 'memory': delivered, 'step': 1},
            {'actor_delivery_verified': True, 'memory': delivered, 'step': 2},
            {'actor_delivery_verified': True, 'memory': {}, 'step': 3}]}}
        evidence, cost = _evidence(ep, bank, bank, [])
        self.assertEqual(evidence['exposed_memory_changes'], 2)
        self.assertEqual([s['sample'] for s in evidence['injected_readouts']], ['visible evidence', ''])
        self.assertEqual(cost['actor_calls'], 0)
