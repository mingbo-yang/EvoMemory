"""CPU execution of the real role contracts with deterministic score/role fixtures."""
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from evomem.common import append, digest, read, write
from math_apps_online.search import ItemSearch, repair_item
from phase1_codegen.engine import MeasurementFailure
from phase1_codegen.memory import CodeMemory
from phase1_codegen.policy import Rejected
from phase1_codegen.program import BASELINE_SOURCE, Program
from phase1_codegen.roles.reviewer import bounded_review_payload, validate_report
from tests.core.test_engine import FIRST_SOURCE, SECOND_SOURCE, SyntheticProposer, custom_source
from tests.core.test_compact_experience import QuarterTokenizer
from tests.core.test_review import ApproximateTokenizer


class FixtureReviewer:
    """Script only the candidate decisions; the normal wire and evidence checks still apply."""
    fingerprint = 'online-reviewer-schema7-cpu-fixture'

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def __call__(self, payload, seed):
        self.last_payload = shown = bounded_review_payload(payload, ApproximateTokenizer())
        stage = shown['stage']
        self.calls.append(stage)
        reason = 'Fixture assessment of the displayed task and validation evidence.'
        assessment = dict(subject='current' if stage == 'pre' else 'candidate', reason=reason)
        polarity = 'uncertain'
        pointer = '/programs/parent/tasks/0/score'
        if stage == 'post':
            decision = next(self.decisions)
            assessment.update(decision=decision, next_check=(
                'Compare the next validation outcome with this one.' if decision == 'continue' else ''))
            pointer = '/comparison/candidate_mean' if decision == 'adopt' else '/validation/mechanism_validated'
            if decision == 'adopt':
                polarity = 'advantage'
        groups = shown['citation_registry']['evidence']['current:facts']
        citation = next(cid for group in groups for cid, path in group['pointers'].items() if path == pointer)
        raw = dict(schema=7, assessment=assessment, focus_evidence_ids=[citation],
                   evaluations=[dict(dimension='Efficiency', polarity=polarity, uncertainty=['limited_tasks'])],
                   diagnostic_assessments={group['group_id']: dict(judgment='insufficient_evidence',
                                           uncertainty=['limited_tasks']) for group in shown['diagnostic_groups']})
        return validate_report(raw, shown)


class ItemEvaluator:
    fingerprint = 'cpu-online-native-fixture'
    step_limit = 1
    context = dict(environment='math', family='number_theory', memory_scope='family', memory_hook_version='memory-events-v1')

    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = []

    def __call__(self, program, logs, tasks, phase, *, memory_snapshot):
        assert phase.endswith('/candidate') and not logs
        task, = tasks
        self.calls.append(phase)
        return [make_episode(program, task, memory_snapshot, next(self.scores))]


def make_episode(program, task, snapshot, score):
    return dict(task_id=task['id'], group_id=task['id'], family=task['family'], environment=task['environment'],
                score=score, synthetic=True, done=True, error=None, program_hash=program.hash, phase='online',
                native_outcome=dict(finish='stop', completion_tokens=3, max_tokens=4096,
                                    status='correct' if score else 'incorrect', score=score),
                memory_hash_before=snapshot['memory_hash'], memory_hash_after=snapshot['memory_hash'],
                read_only_memory=True, memory_write_calls=0, generated_memory_write_calls=0, history=[],
                native_recorder=dict(initial={'obs':'Question fixture', 'goal':'Question fixture'}))


class ItemSearchTests(unittest.TestCase):
    def test_multistep_review_shows_native_action_counts_and_score(self):
        from types import SimpleNamespace
        from math_apps_online.online import registered_test_item
        task = dict(environment='alfworld', family='cool', split='valid_train', benchmark_split='test')
        self.assertTrue(registered_test_item(task, 'alfworld'))
        self.assertFalse(registered_test_item(dict(task, benchmark_split='train'), 'alfworld'))
        self.assertFalse(registered_test_item(dict(task, split='test'), 'alfworld'))
        outcome = dict(status='unsolved', score=0, completed_steps=3, action_attempts=4, step_limit=30, won=False)
        recorder = dict(initial=dict(actions_list=['go to cabinet 1']), steps=[
            dict(action_took='go to cabinet 1', obs='The cabinet is closed.', actions_list=['open cabinet 1']),
            dict(action_took='look', obs='Nothing happens.', actions_list=['look']),
            dict(action_took='look', obs='The sign reads Nothing happens.', actions_list=['look'])])
        ep = dict(task, task_id='test-item', program_hash='program', memory_hash_before='bank',
                  phase='first', history=[{}]*3, native_reward_verified=True, score=0,
                  native_outcome=outcome, native_recorder=recorder)
        engine = SimpleNamespace(environment_step_limit=30, synthetic=False)
        self.assertEqual(ItemSearch._native_outcome(engine, ep)['observation'],
                         dict(outcome, listed_action_steps=2, nothing_happens_steps=1,
                              action_sequence_hash=digest(['go to cabinet 1', 'look', 'look'])))
        with self.assertRaises(MeasurementFailure):
            ItemSearch._native_outcome(engine, dict(ep, native_outcome=dict(outcome, completed_steps=30)))
        with self.assertRaises(MeasurementFailure):
            ItemSearch._native_outcome(engine, dict(ep, native_recorder=dict(recorder, initial={})))
        with self.assertRaises(MeasurementFailure):
            ItemSearch._native_outcome(engine, dict(ep, native_recorder=dict(recorder, steps=[])))
        empty = dict(ep, history=[], native_outcome=dict(outcome, completed_steps=0, action_attempts=0),
                     native_recorder=dict(recorder, steps=[]))
        self.assertEqual(ItemSearch._native_outcome(engine, empty)['observation']['action_sequence_hash'], digest([]))

    def inputs(self, root, scores, sources, decisions=('adopt',)):
        ev=ItemEvaluator(scores)
        p=Program(BASELINE_SOURCE)
        tok=QuarterTokenizer()
        snap=CodeMemory(p, [], tok, ev.context).export_snapshot()
        task=dict(id='math/number_theory/test/engineering-fixture',group_id='test-fixture', environment='math',
                  family='number_theory',split='test',seed=23)
        return dict(out=Path(root), tokenizer=tok, proposer=SyntheticProposer(sources),
                    reviewer=FixtureReviewer(decisions), evaluator=ev, parent=p, parent_snapshot=snap,
                    first_episode=make_episode(p,task,snap,0), logs=[], task=task, records=[], synthetic=True)

    def test_fourth_attempt_adoption_and_exact_handoff_without_repeated_pre_review(self):
        with tempfile.TemporaryDirectory() as td:
            drafts = ['not python!', FIRST_SOURCE, custom_source('Intermediate: '), SECOND_SOURCE]
            args=self.inputs(td,[0,0,1],drafts, ['continue', 'continue', 'continue', 'adopt'])
            r=repair_item(**args)
            self.assertEqual([a['candidate_score'] for a in r['attempts']],[None,0,0,1])
            self.assertEqual([a['decision'] for a in r['attempts']],['continue']*3 + ['adopt'])
            self.assertEqual(args['reviewer'].calls, ['pre'] + ['post']*4)
            self.assertEqual(r['source'], SECOND_SOURCE)
            self.assertEqual(len(args['evaluator'].calls),3)
            for index in range(1, 4):
                prior = read(Path(td)/'trials'/f'{index-1:03d}'/'post_review.json')
                trial = read(Path(td)/'trials'/f'{index:03d}'/'trial.json')
                self.assertEqual(trial['pre_review'], prior)
                payload = args['proposer'].calls[index]['payload']
                self.assertEqual(payload['evolution_memory'], [])
                self.assertEqual(trial['evolution_memory_retrieval']['staged_item_records'], index)
                previous = payload['previous_attempt']
                self.assertEqual(previous['source'], drafts[index-1])
                self.assertEqual(previous['post_review']['report'], prior['report']['evidence_contract'])
                self.assertEqual(payload['mechanism_review']['report_hash'], prior['report_hash'])
                progress = read(Path(td)/'trials'/f'{index:03d}'/'post_review_input.json')['context']['item_progress']
                self.assertEqual(progress['proposals_submitted'], index + 1)
                self.assertEqual(progress['recent_completed_attempts'][-1]['report_hash'], prior['report_hash'])
                self.assertEqual(progress['completed_candidate_native_comparisons'], index)
            self.assertEqual(len(r['events']), 4)
            self.assertEqual(r, repair_item(**args))
            self.assertEqual(len(args['evaluator'].calls),3)
            self.assertEqual(len(args['proposer'].calls),4)
            self.assertEqual(args['reviewer'].calls, ['pre'] + ['post']*4)

    def test_non_json_completion_reaches_post_and_next_author_only_as_item_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            text = 'Incomplete candidate reply: {"source": "def on_event('
            args = self.inputs(root, [], [], ['continue', 'stop'])

            class LedgerAuthor:
                ledger = root/'evolver.jsonl'

                def __call__(self, payload, seed):
                    request = dict(seed=seed, payload=digest(payload))
                    content = text if seed == 11 else json.dumps(dict(source='bad syntax!',
                        hypothesis='Fixture', risk='Fixture'))
                    choice = dict(finish_reason='length' if seed == 11 else 'stop',
                                  message=dict(content=content))
                    append(self.ledger, dict(role='code_evolver', seed=seed, request=request,
                           request_hash=digest(request), response=dict(choices=[choice])))
                    return json.loads(content)

            args['proposer'] = LedgerAuthor()
            result = repair_item(**args)
            post = read(root/'trials/000/post_review.json')
            shown = read(root/post['displayed_path'])['context']['invalid_proposal_response']
            self.assertEqual((shown['content'], shown['finish_reason']), (text, 'length'))
            previous = read(root/'trials/001/proposer_input.json')['previous_attempt']
            self.assertEqual(previous['invalid_proposal_response'], shown)
            self.assertIsNone(previous['source'])
            self.assertNotIn('Incomplete candidate reply', json.dumps(result['events']))

    def test_completed_diagnostic_rejection_is_replayed_without_worker_or_model(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.inputs(td, [], [])
            engine = ItemSearch(**args, seed=11, capabilities=None)
            program = Program('def on_event(state, event, context):\n'
                              '    return {"state": state, "session": {}, "memory": [{"id": "x"}]}\n')
            contexts = [args['first_episode']['native_recorder']['initial']]
            with self.assertRaises(Rejected) as original:
                engine._signature(program, contexts)
            with patch.object(engine, '_diagnostic_result', side_effect=AssertionError('reran diagnosed code')):
                with self.assertRaises(Rejected) as replay:
                    engine._signature(program, contexts)
            self.assertEqual(str(original.exception), str(replay.exception))
            self.assertIn('experience', str(replay.exception))

    def test_host_diagnostic_value_error_does_not_become_an_invalid_candidate(self):
        original = ItemSearch._diagnostic_result
        def broken_diagnostic(engine, program, task, initial):
            if program.hash != engine.initial.hash:
                raise ValueError('corrupt host diagnostic evidence')
            return original(engine, program, task, initial)

        with tempfile.TemporaryDirectory() as td:
            args = self.inputs(td, [], [FIRST_SOURCE], ['stop'])
            with patch.object(ItemSearch, '_diagnostic_result', broken_diagnostic):
                with self.assertRaisesRegex(ValueError, 'corrupt host diagnostic evidence'):
                    repair_item(**args)
            self.assertEqual(args['reviewer'].calls, ['pre'])
            self.assertFalse((Path(td)/'trials/000/trial.json').exists())

    def test_valid_model_reply_followed_by_host_failure_is_infrastructure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self.inputs(root, [], [], ['stop'])

            class BrokenAuthor:
                ledger = root/'proposer.jsonl'

                def __call__(self, payload, seed):
                    request = dict(seed=seed, payload=digest(payload))
                    content = json.dumps(dict(source=FIRST_SOURCE, hypothesis='Fixture', risk='Fixture'))
                    append(self.ledger, dict(role='code_evolver', seed=seed, request=request,
                        request_hash=digest(request), response=dict(choices=[
                            dict(finish_reason='stop', message=dict(content=content))])))
                    raise ValueError('host failed after receiving a valid reply')

            args['proposer'] = BrokenAuthor()
            with self.assertRaisesRegex(MeasurementFailure, 'not explained by the model reply'):
                repair_item(**args)
            self.assertEqual(args['reviewer'].calls, ['pre'])
            self.assertFalse((root/'trials/000/trial.json').exists())

    def test_correct_answer_with_stop_keeps_parent_and_records_solution(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.inputs(td,[1],[FIRST_SOURCE], ['stop'])
            r=repair_item(**args)
            self.assertFalse(r['accepted']); self.assertTrue(r['solved'])
            self.assertEqual((r['source'],r['episode'],r['snapshot']),
                             (args['parent'].source,args['first_episode'],args['parent_snapshot']))
            shown = args['reviewer'].last_payload
            row, = [r for r in shown['evidence'] if r['id'] == 'item:progress']
            self.assertEqual(row['data']['proposals_submitted'], 1)
            group, = shown['citation_registry']['evidence']['item:progress']
            scope = shown['citation_registry']['scopes'][group['scope_id']]
            self.assertEqual(scope['task_id'], args['task']['id'])
            self.assertIsNone(scope['program_hash'])
            self.assertEqual(group['qualification'], 'item_progress')
            self.assertIn('/completed_candidate_native_comparisons', group['pointers'].values())

    def test_invalid_draft_can_stop_with_validation_error_and_no_native_score(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.inputs(td,[],['not python!'], ['stop'])
            result = repair_item(**args)
            self.assertEqual(args['reviewer'].calls, ['pre', 'post'])
            self.assertEqual(len(args['proposer'].calls), 1)
            self.assertFalse(args['evaluator'].calls)
            self.assertEqual(result['attempts'][0]['decision'], 'stop')
            self.assertIsNone(result['attempts'][0]['candidate_score'])
            self.assertFalse(result['solved'])
            self.assertFalse(result['accepted'])

    def test_equivalent_draft_refers_to_the_measured_original_without_copying_its_score(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            drafts = [FIRST_SOURCE, FIRST_SOURCE + '\n# same program\n', BASELINE_SOURCE]
            args = self.inputs(root, [0], drafts, ['continue', 'continue', 'stop'])
            result = repair_item(**args)
            self.assertEqual([a['candidate_score'] for a in result['attempts']], [0, None, None])
            def progress(index):
                return read(root/'trials'/f'{index:03d}'/'post_review_input.json')['context']['item_progress']
            match = progress(1)['current_attempt']['duplicate_of']
            self.assertEqual((match['source'], match['attempt'], match['candidate_score'], match['compared']),
                             ('earlier_draft_on_this_item', 1, 0, True))
            self.assertEqual(match['trial_hash'], digest(read(root/'trials/000/trial.json')))
            self.assertEqual(read(root/'trials/002/proposer_input.json')['previous_attempt']['duplicate_of'], match)
            self.assertEqual(progress(2)['duplicate_submissions'], 2)
            self.assertEqual(progress(2)['current_attempt']['duplicate_of']['source'], 'initial_item_program')
            self.assertEqual(len(args['evaluator'].calls), 1)

    def test_early_stop_and_no_free_repeat_after_role_crash(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.inputs(td,[1],[FIRST_SOURCE])
            result=repair_item(**args)
            self.assertEqual(len(result['attempts']),1)
            self.assertTrue(result['accepted'])
            # Simulate a durable intent but no result on a fresh study directory.
            with tempfile.TemporaryDirectory() as other:
                missing=self.inputs(other,[1],[FIRST_SOURCE]); folder=Path(other)/'trials/000'
                folder.mkdir(parents=True)
                write(folder/'pre_review_intent.json',{'payload':'unknown','proposal':0})
                with self.assertRaises(MeasurementFailure): repair_item(**missing)
                self.assertEqual(len(missing['reviewer'].calls),0)

    def test_no_cross_family_log_and_no_repair_after_first_success(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.inputs(td,[],[])
            args['first_episode']['score']=1
            with self.assertRaises(ValueError): repair_item(**args)
            args['first_episode']['score']=0
            args['logs']=[dict(task_id='other',environment='math',family='algebra')]
            with self.assertRaises(ValueError): repair_item(**args)
            self.assertEqual(len(args['proposer'].calls),0)


class NativeReadback(unittest.TestCase):
    def test_real_evaluator_commits_ordered_answer_and_binds_outcomes_to_role_inputs(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from evomem.common import MODEL_NAME, digest
        from math_apps_online.native import OnlineEvaluator, actor
        from phase1_codegen.native import fixed_task_seed

        row=dict(id='math/algebra/test/integration-fixture',prompt='Offline CPU integration question',split='test',family='algebra')
        task=dict(id=row['id'],group_id=row['id'],family='algebra',environment='math',split='test',
                  room='algebra',seed=fixed_task_seed(row),prompt_hash=digest(row['prompt']))
        adapter=SimpleNamespace(__file__=__file__,load_public=lambda:[row],
                                grade=lambda _,a:dict(score=int(a=='correct'), status='fixture-grader'))
        called=[]
        async def transport(domain, prompt, slot, seed, path, model_hash):
            if path.exists():
                return read(path)
            called.append(slot)
            req=dict(model=MODEL_NAME,messages=actor.messages_for(domain,prompt,slot),seed=seed,**actor.PARAMETERS)
            reply=dict(request=req,request_hash=digest(req),model_hash=model_hash,http_status=200,
                       response_id='fixture'+str(len(called)),ok=True,answer='correct' if slot else 'wrong',
                       finish='stop',served_model=MODEL_NAME,endpoint=actor.URL,usage={'completion_tokens':3})
            write(path,reply)
            return reply

        changed='''def on_event(state, event, context):
    memory = None
    if event.get("type") == "before_action":
        memory = [{"id":"format", "experience":"Fixture cue"}]
    if event.get("type") == "task_end":
        state = {"z": 1, "a": 2}
    return {"state":state,"memory":memory,"session":{}}
'''
        with tempfile.TemporaryDirectory() as td, patch.object(actor,'ask',transport):
            root=Path(td); tok=QuarterTokenizer(); p=Program(BASELINE_SOURCE)
            native=OnlineEvaluator(root/'native','math',adapter,[task],tok,'fixture-cpu',
                                   dict(environment='math',family='algebra',memory_scope='family'),None,root/'wire')
            prior=native.snapshot(p, [])
            first,=native(p,[],[task],'first',memory_snapshot=prior)
            proposer=SyntheticProposer([changed]); proposer.fingerprint='CPU fixture author'
            reviewer=FixtureReviewer(['adopt'])
            args=dict(out=root/'search',tokenizer=tok,proposer=proposer,reviewer=reviewer,evaluator=native,
                      parent=p,parent_snapshot=prior,first_episode=first,logs=[],task=task,records=[])
            result=repair_item(**args)
            self.assertTrue(result['accepted'])
            after=native.commit(Program(result['source']),[],task,result['episode'],memory_snapshot=result['snapshot'])
            self.assertEqual(after['state_wire'],'{"z":1,"a":2}')
            self.assertEqual(result['episode']['native_outcome']['score'],1)
            self.assertEqual(proposer.calls[0]['payload']['online_native']['first_attempt']['observation']['score'],0)
            self.assertEqual(reviewer.last_payload['context']['online_native']['candidate']['observation']['score'],1)
            self.assertEqual(len(called),2)
            self.assertEqual(result,repair_item(**args))
            self.assertEqual(len(called),2)


class ScopeTransport(unittest.TestCase):
    def test_online_evolver_wire_and_paid_ledger_use_independent_memory_scopes(self):
        """Synthetic token/HTTP fixture; real tokenizer accounting is tested separately."""
        import json
        import httpx
        from evomem.common import digest
        from math_apps_online.search import ONLINE_EVOLVER_PROMPT, OnlineEvolver
        from phase1_codegen.roles.evolver import EVOLVER_PROMPT, LANGUAGE_RULES_PROMPT
        from phase1_codegen.roles.reviewer import REVIEW_PROMPT

        captured=[]
        tok=ApproximateTokenizer()
        payload=dict(context={'memory_scope':'family'},evolution_memory=[],current_source=BASELINE_SOURCE)
        def endpoint(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content='{}'))]))
        with tempfile.TemporaryDirectory() as td:
            ledger=Path(td)/'evolver.jsonl'
            author=OnlineEvolver(tok,ledger,'CPU fixture model',base_url='http://127.0.0.1:8141/v1')
            author.llm.client.close()
            author.llm.client=httpx.Client(transport=httpx.MockTransport(endpoint))
            try:
                self.assertEqual(author(payload,11),{})
            finally:
                author.close()
            measured=json.loads(ledger.read_text())
            wire=captured[0]
            self.assertEqual(measured['request'],wire)
            self.assertEqual(measured['request_hash'],digest(wire))
            self.assertEqual(measured['proposer_fingerprint'],author.fingerprint)
            self.assertEqual(wire['messages'][0]['content'],ONLINE_EVOLVER_PROMPT)
            self.assertEqual(measured['local_prompt_tokens'],tok.chat_count(wire['messages']))
            self.assertNotIn('retain the single-family scope',ONLINE_EVOLVER_PROMPT)
            self.assertIn('retain the single-family scope',EVOLVER_PROMPT)
            self.assertIn(LANGUAGE_RULES_PROMPT,ONLINE_EVOLVER_PROMPT)
            self.assertNotIn('retain the single-family scope',REVIEW_PROMPT)
