"""Trusted host failures stop search after receipts; no native/model calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evomem.common import events,read
from phase1_codegen import program as runtime
from phase1_codegen.program import Program,WorkerStartupError
from phase1_codegen.tests import test_acceptance_policy as policy_fixture
from phase1_codegen.tests import test_hook_adapter as hook_fixture

EMPTY='def on_event(state,event,context):\n    return {"state":state,"session":{},"memory":None}\n'

class StartupReceiptTests(unittest.TestCase):
    def test_premature_exit_keeps_output_and_actual_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'broken.py'
            path.write_text('import sys\nprint("trusted bootstrap failed",file=sys.stderr)\nraise SystemExit(7)\n')
            with patch.object(runtime,'WORKER',path),self.assertRaises(WorkerStartupError) as caught:
                Program(EMPTY).run('event',{},experience={})
            details=caught.exception.details
            self.assertEqual(details['readiness'],'eof_before_ready')
            self.assertEqual(details['returncode_after_cleanup'],7)
            self.assertIn('trusted bootstrap failed',details['stderr_excerpt'])
            self.assertIs(details['generated_execution_started'],False)
    def test_readiness_timeout_is_distinct_from_premature_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'slow.py';path.write_text('import time\ntime.sleep(10)\n')
            with patch.object(runtime,'WORKER',path),patch.object(runtime,'WORKER_BOOTSTRAP_WALL_SECONDS',.1),self.assertRaises(WorkerStartupError) as caught:
                Program(EMPTY).run('event',{},experience={})
            details=caught.exception.details
            self.assertEqual(details['readiness'],'timeout')
            self.assertIsNone(details['returncode_before_cleanup'])
            self.assertLess(details['returncode_after_cleanup'],0)

class InfrastructureFlowTests(unittest.TestCase):
    setUp=policy_fixture.AcceptancePolicyTests.setUp
    components=policy_fixture.AcceptancePolicyTests.components
    engine=policy_fixture.AcceptancePolicyTests.engine
    def test_measurement_does_not_turn_host_failure_into_synthetic_zero_rows(self):
        proposer,evaluator,reviewer=self.components()
        engine=self.engine(proposer,evaluator,reviewer)
        with patch.object(type(evaluator),'__call__',side_effect=WorkerStartupError('synthetic startup',details={'readiness':'timeout'})):
            with self.assertRaises(WorkerStartupError):engine.run(1)
        self.assertEqual(proposer.calls,[])
        self.assertEqual(reviewer.calls,[])
        failures=list((self.root/'measurement_failures').glob('*.json'))
        self.assertEqual(len(failures),1)
        self.assertFalse(read(failures[0])['candidate_measurement_committed'])
        self.assertEqual(list((self.root/'measurements').glob('*.json')),[])
        state=read(self.root/'state.json')
        self.assertEqual(state['next_round'],0)
        self.assertEqual(state['revision'],0)

class NativeFailureReceiptTests(unittest.TestCase):
    def test_native_failure_receipts_exist_before_host_error_propagates(self):
        case=hook_fixture.HookAdapterTests();case.setUp()
        with tempfile.TemporaryDirectory() as directory:
            bank,transaction,ep=case.staged_native(directory,allow_write=False)
            before=bank.export_snapshot()
            transaction.hook.finished=False
            with patch.object(Program,'run',side_effect=WorkerStartupError('synthetic bootstrap failure',details={'readiness':'eof_before_ready'})):
                with self.assertRaises(WorkerStartupError):
                    transaction.hook.emit('after_observation',{'step':1,'visible':{'obs':'visible'}})
            ep['error']='HookExecutionError: synthetic bootstrap failure'
            with self.assertRaises(WorkerStartupError):transaction.finalize(ep)
            rows=events(Path(directory)/'canonical_episodes.jsonl')
            self.assertEqual(len(rows),1)
            self.assertIs(rows[0]['native_reward_verified'],False)
            self.assertIsNone(rows[0]['score'])
            self.assertEqual(rows[0]['infrastructure_failure']['details']['readiness'],'eof_before_ready')
            self.assertEqual(len(events(Path(directory)/'code_memory_traces.jsonl')),1)
            self.assertEqual(bank.export_snapshot(),before)
            self.assertFalse(rows[0]['memory_hook']['committed'])

if __name__=='__main__':unittest.main()
