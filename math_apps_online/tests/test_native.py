"""CPU contract tests with a recorded fake transport, isolated generated Hooks and local fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evomem.common import MODEL_NAME, digest, read, write
from math_apps_online.native import OnlineEvaluator, actor
from phase1_codegen.native import fixed_task_seed
from phase1_codegen.program import Program
from phase1_codegen.tests.test_engine import TinyTokenizer


SOURCE = '''def on_event(state, event, context):
    memory = None
    if event.get("type") == "before_action":
        memory = state.get("rows", [])
    if event.get("type") == "task_end":
        experience = event["experience"]
        state = {"z": 8, "a": 3, "rows": [{"id": experience["task_id"], "experience": experience["goal"]}]}
    return {"state": state, "memory": memory, "session": {}}
'''


class NativeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(id=f'math/algebra/test/{idx}', split='test', family='algebra',
                          prompt=f'Fixture question {idx}') for idx in (1,2)]
        self.tasks = [dict(id=r['id'], group_id=r['id'], family=r['family'], room=r['family'],
                          environment='math', split='test', seed=fixed_task_seed(r),
                          prompt_hash=digest(r['prompt'])) for r in self.rows]
        self.calls = []
        self.grades = []
        def grade(task_id, ans):
            self.grades.append(task_id)
            return dict(score=int(ans.endswith('2')), status='fixture')
        self.adapter = SimpleNamespace(__file__=__file__, load_public=lambda:self.rows, grade=grade)
        self.context = dict(environment='math', family='algebra', memory_scope='family')
        self.evaluator = self.open()
        self.program = Program(SOURCE)
        async def ask(domain, prompt, slot, seed, file, model_hash):
            if file.exists():
                return read(file)
            self.calls.append((prompt,slot))
            query = dict(model=MODEL_NAME, messages=actor.messages_for(domain,prompt,slot),
                         seed=seed, **actor.PARAMETERS)
            response = dict(request=query, request_hash=digest(query), model_hash=model_hash,
                            http_status=200, response_id='fixture'+str(len(self.calls)), ok=True,
                            answer=prompt, finish='stop', served_model=MODEL_NAME, endpoint=actor.URL,
                            usage={'completion_tokens':2})
            write(file,response)
            return response
        self.addCleanup(patch.stopall)
        patch.object(actor,'ask',ask).start()

    def open(self):
        return OnlineEvaluator(self.root/'native', 'math', self.adapter, self.tasks, TinyTokenizer(),
                               'fixture-model', self.context, None, self.root/'wire')

    def test_equal_ast_snapshots_remain_bound_to_their_own_exact_source(self):
        other = Program(SOURCE + '\n# Alternative source formatting\n')
        self.assertEqual(self.program.identity_hash, other.identity_hash)
        for program in (self.program, other, self.program):
            self.assertEqual(self.evaluator.snapshot(program, [])['program_hash'], program.hash)

    def test_interrupted_snapshot_rebuild_is_not_replayed(self):
        with patch('math_apps_online.native.CodeMemory', side_effect=OSError('interrupted')) as build:
            with self.assertRaises(OSError):
                self.evaluator.snapshot(self.program, [])
            with self.assertRaisesRegex(RuntimeError, 'Unresolved pre-item snapshot rebuild'):
                self.open().snapshot(self.program, [])
            self.assertEqual(build.call_count, 1)

    def test_stage_restart_and_commit_no_new_production_calls(self):
        e, prog = self.evaluator, self.program
        task, second = self.tasks
        before = e.snapshot(prog,[])
        ep, = e(prog,[],[task],'first',memory_snapshot=before)
        self.assertTrue(ep['read_only_memory'])
        self.assertEqual(ep['score'],0)
        self.assertEqual(ep['native_outcome']['completion_tokens'],2)
        with patch.object(Program,'run',side_effect=AssertionError('Hook reran')):
            fresh = self.open()
            recovered = fresh.measured_episode(prog,[],task,'first',memory_snapshot=before)
            self.assertEqual(recovered,ep)
            after = fresh.commit(prog,[],task,recovered,memory_snapshot=before)
            self.assertEqual(after,fresh.commit(prog,[],task,recovered,memory_snapshot=before))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(len(self.grades),1)
        self.assertEqual(list(json.loads(after['state_wire'])),['z','a','rows'])
        with self.assertRaises(ValueError):
            e(prog,[ep],[task],'again',memory_snapshot=after)
        later, = e(prog,[ep],[second],'first',memory_snapshot=after)
        self.assertEqual(self.calls[-1][1]['entries'][0]['id'],task['id'])
        self.assertEqual(later['score'],1)
        self.assertEqual(before['state_wire'],'{}')
        # A second logical attempt with the same first Actor input is a cache hit.
        e(prog, [], [task], 'candidate', memory_snapshot=before)
        totals = e.item_model_costs(task)['actor']
        self.assertEqual((totals['logical_calls'], totals['cache_hits'], totals['inference_attempts']), (2,1,1))
        self.assertEqual((totals['measured_completion_tokens'], totals['unknown_input_usage']), (2,1))
        self.assertEqual(e.item_model_costs(second)['actor']['logical_calls'], 1)

    def test_candidate_rebuild_uses_only_same_family_prior_experience(self):
        e, prog = self.evaluator, self.program
        a,b = self.tasks
        pre = e.snapshot(prog,[])
        ep, = e(prog,[],[a],'first',memory_snapshot=pre)
        previous = e.commit(prog,[],a,ep,memory_snapshot=pre)
        candidate = Program(SOURCE.replace('"z": 8','"z": 11'))
        fresh = e.snapshot(candidate,[ep],previous)
        self.assertEqual(json.loads(fresh['state_wire'])['z'],11)
        self.assertEqual(fresh['program_hash'], candidate.hash)
        self.assertEqual(fresh['source_log_hash'], digest([ep]))
        res, = e(candidate,[ep],[b],'attempt/one',memory_snapshot=fresh)
        self.assertEqual(self.calls[-1][1]['entries'][0]['experience'], self.rows[0]['prompt'])
        with self.assertRaisesRegex(ValueError,'family'):
            e.snapshot(candidate,[dict(ep,family='number_theory')],previous)
        with self.assertRaises(ValueError):
            e.commit(prog,[ep],b,res,memory_snapshot=previous)
        with self.assertRaisesRegex(ValueError,'requires official test items'):
            OnlineEvaluator(self.root/'bad','math',self.adapter,[dict(a,split='train')],TinyTokenizer(),
                            'fixture-model',self.context,None,self.root/'wire')

    def test_unresolved_or_mutated_measurement_cannot_be_an_answer(self):
        e, prog = self.evaluator, self.program
        task = self.tasks[0]
        pre = e.snapshot(prog,[])
        ep, = e(prog,[],[task],'first',memory_snapshot=pre)
        folder = e._directory(task,prog,'first')
        state = read(folder/'measurement.json')
        state['state_wire'] = '{"wrong":true}'
        write(folder/'measurement.json',state)
        with self.assertRaisesRegex(RuntimeError,'Terminal Hook state'):
            e.commit(prog,[],task,ep,memory_snapshot=pre)
        (folder/'measurement.json').unlink()
        with self.assertRaises(FileNotFoundError):
            e.measured_episode(prog,[],task,'first',memory_snapshot=pre)
        with self.assertRaisesRegex(RuntimeError,'Unresolved'):
            e(prog,[],[task],'first',memory_snapshot=pre)
        self.assertEqual(len(self.calls),1)


if __name__ == '__main__':
    unittest.main()
