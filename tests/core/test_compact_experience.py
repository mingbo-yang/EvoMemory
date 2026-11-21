"""Projection/retrieval boundaries, without model or native environment calls."""
import copy
import unittest
from evomem.common import digest, dumps
from phase1_codegen.experience import (EvolutionMemory, SharedCodeView, public_event, mechanism_query,
                                       RECORD_TOKENS, execution_problems, _matching)
from tests.core.test_engine import FIRST_SOURCE, SECOND_SOURCE, TinyTokenizer


class QuarterTokenizer(TinyTokenizer):
    def count(self, text):
        return (len(text.encode()) + 3) // 4


class CompactExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = QuarterTokenizer()
        self.trial = dict(public_context={'environment': 'alfworld', 'family': 'heat'},
                          parent_hash='p' * 64, candidate_hash=None, public_status='invalid',
                          accepted=False, synthetic=True, hypothesis='unverified private task prose')

    def record(self, **changes):
        return public_event(dict(self.trial, **changes), FIRST_SOURCE, SECOND_SOURCE,
                            tokenizer=self.tokenizer)

    def test_preserves_missing_comparison_and_failure_without_zero_imputation(self):
        row = self.record(error='Rejected: unknown call: dir (line 5, column 2)')
        self.assertIsNone(row['exploration'])
        self.assertIsNone(row['after_hash'])
        self.assertEqual(row['error']['symbols'], ['dir'])
        self.assertEqual(row['error']['line'], 5)
        self.assertNotIn(self.trial['hypothesis'], dumps(row))

    def test_typed_outcomes_preserve_zero_and_false(self):
        row = self.record(exploration=dict(complete=False, parent=0, candidate=0.0,
                                          delta=0.0, task_ids=['private-task']))
        self.assertIs(row['exploration']['complete'], False)
        for key in ('parent', 'candidate', 'delta'):
            self.assertIsNone(row['exploration'][key])
        self.assertNotIn('private-task', dumps(row))
        complete = self.record(exploration=dict(complete=True, parent=0, candidate=0.0, delta=0.0))
        self.assertEqual(complete['exploration']['candidate'], 0.0)

    def test_historical_incomplete_view_retains_archive_id_and_binds_both_views(self):
        archived = dict(self.record(), exploration=dict(complete=False, parent=.5, candidate=.5, delta=0))
        memory = EvolutionMemory([archived])
        query = mechanism_query(self.trial['public_context'], FIRST_SOURCE)
        selected, receipt = memory.select(query, self.tokenizer)
        self.assertEqual(memory.records, [archived])
        self.assertEqual(archived['exploration']['parent'], .5)
        self.assertEqual(selected[0]['id'], archived['id'])
        self.assertEqual(selected[0]['exploration'], dict(complete=False, parent=None, candidate=None, delta=None))
        ranked = receipt['ranked_records'][0]
        self.assertEqual(ranked['archived_record_hash'], digest(archived))
        self.assertEqual(ranked['displayed_record_hash'], digest(selected[0]))
        self.assertEqual(ranked['tokens'], self.tokenizer.count(dumps(selected[0])))

    def test_opinion_matching_binds_polarity_to_criterion_and_accepts_archived_opinions(self):
        def opinion(dimension, polarity):
            return dict(dimension=dimension, polarity=polarity, uncertainty=['clipped_evidence'])

        good = opinion('Actionability', 'advantage')
        bad = opinion('Actionability', 'disadvantage')
        base = self.record()
        records = [dict(base, id='a', review_opinions=[bad, opinion('Efficiency', 'advantage')]),
                   dict(base, id='b', review_opinions=[good, opinion('Efficiency', 'disadvantage')])]
        memory = EvolutionMemory(records)
        for current, expected in [(good, 'b'), (bad, 'a')]:
            report = dict(report=dict(assessment=dict(subject='current'),
                                      evidence_contract=dict(evaluations=[current])))
            query = mechanism_query(self.trial['public_context'], FIRST_SOURCE, review=report)
            selection, _ = memory.select(query, self.tokenizer)
            self.assertEqual(selection[0]['id'], expected)
            self.assertIn('current__review__Actionability__', _matching(query))
            self.assertNotIn('current__', _matching(records[0]))

    def test_problem_features_prioritize_differences_at_edits_and_reconcile_both_subjects(self):
        def diagnostic(rows):
            return dict(diagnostics=dict(facts=dict(branch_counts=[
                dict(line=line, expression=expr, calls=n, function='on_event')
                for line, expr, n in rows])))
        common = [(2, 'context.get("goal")', 4)] * 10
        parent = diagnostic(common + [(9, 'state.get("memory")', 5), (6, 'state.get("memory")', 3)])
        candidate = diagnostic(common + [(9, 'state.get("memory")', 8), (6, 'state.get("memory")', 7)])
        for new, cases in [(candidate, (12, 12)), (parent, (12, 12)), (None, (12, 0))]:
            with self.subTest(cases=cases):
                row = public_event(self.trial, FIRST_SOURCE, SECOND_SOURCE, tokenizer=self.tokenizer,
                                   parent_behavior=parent, behavior=new)
                for side, total in zip(('parent', 'candidate'), cases):
                    shown = [x for x in row['problem_features'] if x['subject'] == side]
                    self.assertEqual(total, len(shown) + row['omissions']['problem_features'][side])
                    self.assertEqual(bool(shown), bool(total))
                    if new == candidate:
                        self.assertEqual(shown[0]['line'], 6)

    def test_private_literal_removed_but_normal_state_keys_remain(self):
        sentinel = 'EXACT_PRIVATE_TASK_ANSWER_SENTINEL'
        source = SECOND_SOURCE.replace('Adaptation goal summary: ', sentinel)
        row = public_event(self.trial, FIRST_SOURCE, source, tokenizer=self.tokenizer,
                           private_texts=[sentinel])
        self.assertNotIn(sentinel, dumps(row))
        self.assertIn('private literal sha256', row['code_diff'])
        self.assertIn('events', row['code_diff'])

    def test_online_structural_diff_hides_comments_answers_and_prior_attempt_identifiers(self):
        parent = 'def on_event(state, event, context):\n    return state\n'
        candidate = (
            'def on_event(state, event, context):\n'
            '    OldAttemptName = """memo: copied answer\nsecond line of answer"""  # private explanation\n'
            '    if event.get("type") == "task_end":\n'
            '        state["unshared_key"] = 741853\n'
            '    return {"state": state, "session": {}, "memory": None}\n')
        error = ("Rejected: event memory row 0 requires exactly id and experience; path=/memory/0; "
                 "actual=['id', 'unshared_key', 'prefix copied answer suffix']")
        names = SharedCodeView.private_identifiers(['Earlier rejected answer used OldAttemptName.'])
        row = public_event(dict(self.trial, error=error), parent, candidate, tokenizer=self.tokenizer,
                           shared_code=SharedCodeView(names))
        for text in ('OldAttemptName', 'copied answer', 'second line of answer', 'private explanation',
                     'unshared_key', '741853'):
            self.assertNotIn(text, dumps(row))
        for text in ('on_event', 'event.get', 'task_end', 'number_', 'session'):
            self.assertIn(text, row['code_diff'])
        self.assertIn(row['error']['actual_fields'][1], row['code_diff'])
        self.assertIn('Structure only', row['omissions']['code_view'])

    def test_interpolated_or_unparsed_online_drafts_omit_raw_text(self):
        for draft in ('def f():\n    return f"private prose {value}"\n', 'return (  # private prose'):
            row = public_event(self.trial, FIRST_SOURCE, draft, tokenizer=self.tokenizer,
                               shared_code=SharedCodeView(set()))
            self.assertNotIn('private prose', dumps(row))
            self.assertTrue(row['raw_source_hash'])
        error = 'PrivateValue: unknown call: secret_symbol (line 1, column 2)'
        row = public_event(dict(self.trial, error=error), FIRST_SOURCE, None, tokenizer=self.tokenizer,
                           shared_code=SharedCodeView(set()))
        self.assertEqual(row['error'], dict(kind='proposal_without_submitted_source', original_hash=digest(error)))
        self.assertIsNone(row['raw_source_hash'])

    def test_huge_diff_is_bounded_with_explicit_omissions(self):
        source = 'def on_event(state, event, context):\n' + ''.join(
            '    number = ' + str(n) + '\n' for n in range(1000))
        source += '    return {"state": state, "session": {}, "memory": None}\n'
        row = public_event(self.trial, FIRST_SOURCE, source, tokenizer=self.tokenizer)
        self.assertLessEqual(self.tokenizer.count(dumps(row)), RECORD_TOKENS)
        self.assertGreater(row['omissions']['code_diff_lines'], 0)
        self.assertIn('raw_diff_hash', row['omissions'])

    def test_large_rewrite_keeps_the_actual_late_error_site(self):
        source = 'def on_event(state, event, context):\n' + ''.join(
            '    number = ' + str(n) + '\n' for n in range(1000))
        lines = source.splitlines(True)
        lines[904] = '    number = dir(state)\n'
        row = public_event(dict(self.trial, error='Rejected: unknown call: dir (line 905, column 14)'),
            FIRST_SOURCE, ''.join(lines), tokenizer=self.tokenizer)
        self.assertIn('dir(state)', row['code_diff'])
        self.assertIn('@@', row['code_diff'])
        self.assertEqual(row['omissions']['code_scope']['candidate'], 1001)
        self.assertGreater(row['omissions']['code_diff_lines'], 0)

    def test_matching_ignores_task_family_reward_and_acceptance(self):
        records = [dict(self.record(), id='a', accepted=False), dict(self.record(), id='b', accepted=True)]
        q = mechanism_query({'environment': 'alfworld', 'family': 'two'}, FIRST_SOURCE)
        a, receipt = EvolutionMemory(records).select(q, self.tokenizer)
        records[0]['observed_context']['family'] = 'two'
        records[1]['observed_context']['family'] = 'heat'
        for row in records:
            row['accepted'] = not row['accepted']
        b, _ = EvolutionMemory(records).select(q, self.tokenizer)
        self.assertEqual([r['id'] for r in a], [r['id'] for r in b])
        self.assertNotIn('two', dumps(q))
        self.assertEqual([r['rank'] for r in receipt['ranked_records']], [1, 2])

    def test_previous_attempt_excluded_with_explicit_reason(self):
        row = self.record()
        q = mechanism_query({'environment': 'alfworld'}, FIRST_SOURCE)
        selected, receipt = EvolutionMemory([row]).select(q, self.tokenizer, exclude_ids=[row['id']])
        self.assertEqual(selected, [])
        self.assertEqual(receipt['ranked_records'][0]['reason'], 'already_provided_as_previous_attempt')

    def test_structure_fallback_matches_code_when_no_diagnostic_exists(self):
        numeric = FIRST_SOURCE.replace('memory = None', 'memory = None\n    number = float(1)')
        row = public_event(self.trial, FIRST_SOURCE, numeric, tokenizer=self.tokenizer)
        other = self.record()
        query = mechanism_query({'environment': 'alfworld'}, numeric)
        selected, _ = EvolutionMemory([other, row]).select(query, self.tokenizer)
        self.assertEqual(selected[0]['id'], row['id'])

    def test_oversized_or_foreign_records_fail_explicitly(self):
        row = self.record()
        q = mechanism_query({'environment': 'alfworld'}, FIRST_SOURCE)
        huge = dict(row, code_diff='x' * 10000)
        with self.assertRaisesRegex(ValueError, 'Unbounded'):
            EvolutionMemory([huge]).select(q, self.tokenizer)
        foreign = copy.deepcopy(row)
        foreign['observed_context']['environment'] = 'textworld'
        with self.assertRaisesRegex(ValueError, 'dataset'):
            EvolutionMemory([foreign]).select(q, self.tokenizer)

    def test_observed_missing_field_not_relabelled_as_defect_or_leaked_text(self):
        behavior = {'diagnostics': {'facts': {'missing_dict_keys': [dict(
            function='retrieve', line=12, operation='task_start', calls=14,
            expression='context.get("visible", {})', first_missing={'result': 'TASK_ANSWER'})]}}}
        facts = execution_problems(behavior)
        self.assertEqual(facts[0]['calls'], 14)
        self.assertNotIn('TASK_ANSWER', dumps(facts))
        self.assertNotIn('supported_defect', dumps(facts))
        self.assertIn('visible', facts[0]['terms'])

    def test_two_line_replacement_exposes_new_return_despite_parent_diagnostics(self):
        parent = 'def on_event(state, event, context):\n' + ''.join(
            '    number = ' + str(n) + '\n' for n in range(500))
        parent += '    return {"state": state, "session": {}, "memory": []}\n'
        candidate = 'def on_event(state, event, context):\n    return {"state": state, "session": {}, "memory": None}\n'
        behavior = {'diagnostics': {'facts': {'missing_dict_keys': [dict(function='on_event',
            line=n + 2, calls=100, missing_count=100, expression='context.get("goal", "")') for n in range(80)]}}}
        row = public_event(self.trial, parent, candidate, tokenizer=self.tokenizer, parent_behavior=behavior)
        self.assertIn('+    return {"state": state, "session": {}, "memory": None}', row['code_diff'])
        self.assertIn('def on_event', row['code_diff'])
        self.assertGreater(row['omissions']['code_diff_lines'], 0)
        self.assertLessEqual(self.tokenizer.count(dumps(row)), RECORD_TOKENS)

    def test_worker_output_error_retains_exact_contract_and_actual_fields(self):
        for field in ('score', 'keywords'):
            error = ('Rejected: isolated worker rejected (2): {"error":"Rejected","message":'
                     '"event memory row 0 requires exactly id and experience; path=/memory/0; '
                     "actual=['experience', 'id', '" + field + "']" + '"}')
            row = self.record(error=error)
            self.assertEqual(row['error']['expected'], 'id and experience')
            self.assertEqual(row['error']['actual_fields'], ['experience', 'id', field])
            self.assertEqual(row['error']['field_path'], '/memory/0')
            self.assertIsNone(row['exploration'])

    def test_wrong_output_type_retains_constraint(self):
        row = self.record(error='Rejected: event session must be a JSON object; path=/session; actual=list')
        self.assertEqual(row['error']['constraint'], 'event session must be a JSON object')
        self.assertEqual(row['error']['actual_type'], 'list')

    def test_required_error_evidence_cannot_be_silently_clipped(self):
        fields = ['field_' + str(n) for n in range(1000)]
        error = 'Rejected: event output requires exactly state, session, memory; path=/; actual=' + repr(fields)
        with self.assertRaisesRegex(ValueError, 'Required mechanism experience'):
            self.record(error=error)

    def test_required_changed_line_cannot_be_replaced_by_metadata_only(self):
        candidate = 'def on_event(state, event, context):\n    return ' + repr(list(range(3000))) + '\n'
        with self.assertRaisesRegex(ValueError, 'Required mechanism change/error evidence'):
            public_event(self.trial, FIRST_SOURCE, candidate, tokenizer=self.tokenizer)

    def test_unchanged_return_is_optional_when_the_actual_edit_fits(self):
        row = public_event(self.trial, FIRST_SOURCE, SECOND_SOURCE, tokenizer=TinyTokenizer())
        self.assertLessEqual(TinyTokenizer().count(dumps(row)), RECORD_TOKENS)
        self.assertIn('+', row['code_diff'])
        self.assertIn('Adaptation goal summary:', row['code_diff'])

    def test_matching_keeps_current_parent_distinct_from_failed_historical_candidate(self):
        fact = dict(kind='missing_dict_keys', terms='context goal')
        current = _matching(dict(current_parent=dict(problems=[fact])))
        historical = _matching(dict(historical_candidate=dict(problems=[fact])))
        self.assertNotEqual(current, historical)
        self.assertIn('parent__goal', current)
        self.assertIn('candidate__goal', historical)
        self.assertEqual(_matching(dict(problem_features=[dict(subject='parent', **fact)])), current)

    def test_error_field_names_cannot_reintroduce_private_task_text(self):
        secret = 'EXACT_PRIVATE_TASK_ANSWER_SENTINEL'
        error = 'Rejected: event memory row 0 requires exactly id and experience; path=/memory/0; actual=' + repr(['id', 'experience', secret])
        row = public_event(dict(self.trial, error=error), FIRST_SOURCE, SECOND_SOURCE,
                           tokenizer=self.tokenizer, private_texts=[secret])
        self.assertNotIn(secret, dumps(row))
        self.assertIn('private field sha256', dumps(row['error']))


if __name__ == '__main__':
    unittest.main()
