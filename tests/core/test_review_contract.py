"""Synthetic evidence-contract checks; no model, GPU or benchmark execution."""
import copy
import json
import unittest

from phase1_codegen.roles.reviewer import bounded_review_payload, validate_report, _messages, MAX_INPUT_TOKENS, citation_pointers
from tests.core.test_review import ApproximateTokenizer, SOURCE


def contract_payload(stage='post'):
    rows = []
    for side in ('parent', 'candidate') if stage == 'post' else ('parent',):
        content = 'recorded map observation' if side == 'parent' else 'recorded map observation with provenance'
        rows.extend([
            dict(id=side + ':program', kind='program', data=SOURCE + '# ' + side),
            dict(id=side + ':storage', kind='memory_storage', data=dict(state={'episodes': [
                {'id': 'stored-1', 'experience': content}]}, stored_tokens=21 if side == 'parent' else 19)),
            dict(id=side + ':read:0', kind='memory_readout', data=dict(task_id='task-a',
                query={'task_id': 'task-a', 'goal': 'recorded map', 'environment': 'synthetic'},
                read_tokens=30, actual_trace_available=True, replay_matches_actual=True,
                actual_injected_memory={'entries': [{'id': 'stored-1', 'experience': content}]},
                replayed_readout={'entries': [{'id': 'stored-1', 'experience': content}]})),
            dict(id=side + ':execution:0', kind='execution_trace', data=dict(task_id='task-a', score=0,
                done=True, error=None, history=[{'action': 'look'}])),
        ])
    if stage == 'post':
        rows.append(dict(id='candidate:validation', kind='validation', data=dict(
            status='awaiting_reviewer_assessment', mechanism_validated=True,
            candidate_execution_performed=True, error=None)))
    return dict(review_contract=2, stage=stage, round=2, current_program_hash='parent-hash',
        candidate_program_hash='candidate-hash' if stage == 'post' else None,
        memory_snapshot_hash='snapshot-2', context={'environment': 'synthetic', 'family': 'synthetic'},
        evidence=rows, evolution_memory=[], previous_post_review=None)


def observation(shown, evidence_id, pointer, oid='o1', excerpt=None):
    registry = shown['citation_registry']
    anchor = next(a for a in registry['evidence'][evidence_id] if pointer in citation_pointers(a))
    from phase1_codegen.roles.reviewer import _pointer
    row = next(r for r in shown['evidence'] if r['id'] == evidence_id)
    value = _pointer(row['data'], pointer)
    return dict(id=oid, evidence_id=evidence_id, pointer=pointer,
                match='excerpt' if excerpt is not None else 'value', value=value if excerpt is None else excerpt,
                scope=copy.deepcopy(registry['scopes'][anchor['scope_id']]))


def synthetic_raw_report(shown, verdict='neutral', analysis='Synthetic fixture; no research result.'):
    observations = []
    facts = next(row['data'] for row in shown['evidence'] if row['id'] == 'current:facts')
    for side in ('parent', 'candidate'):
        if facts['programs'].get(side, {}).get('stored_tokens') is not None:
            observations.append(observation(shown, 'current:facts', '/programs/' + side + '/stored_tokens',
                                            'o' + str(len(observations) + 1)))
    if not observations:
        for eid, anchors in shown['citation_registry']['evidence'].items():
            if anchors:
                observations.append(observation(shown, eid, citation_pointers(anchors[0])[0], 'o1', excerpt=None))
                break
    result = dict(schema=2, observations=observations,
        evaluations=[dict(id='e1', dimension='Efficiency', polarity='advantage' if verdict == 'positive' else 'uncertain',
            observation_ids=[o['id'] for o in observations], uncertainty=['causal_attribution_unproven'])],
        assessment=dict(subject=shown['expected_assessment_subject'], verdict=verdict, evaluation_ids=['e1']), analysis=analysis)
    if shown.get('review_output_schema') == 6:
        result['diagnostic_assessments'] = [dict(copy.deepcopy(group),
            judgment='insufficient_evidence', uncertainty=['causal_attribution_unproven'])
            for group in shown.get('diagnostic_groups', [])]
    return result


def indexed_raw_report(shown, original=None, verdict='neutral'):
    original = original or synthetic_raw_report(shown, verdict)
    observations = {row['id']: row for row in original['observations']}
    result = dict(schema=3, evaluations=[], assessment=copy.deepcopy(original['assessment']), analysis=original['analysis'])
    for item in original['evaluations']:
        evidence = []
        for oid in item['observation_ids']:
            obs = observations[oid]
            cid = next(cid for group in shown['citation_registry']['evidence'][obs['evidence_id']]
                       for cid, pointer in group['pointers'].items() if pointer == obs['pointer'])
            evidence.append(dict(citation_id=cid, match=obs['match'], value=obs['value']))
        result['evaluations'].append(dict(id=item['id'], dimension=item['dimension'], polarity=item['polarity'],
                                         evidence=evidence, uncertainty=item['uncertainty']))
    return result


def bundled_raw_report(shown, original=None, verdict='neutral'):
    original = original or synthetic_raw_report(shown, verdict)
    result = dict(schema=shown.get('review_output_schema', 5),
        evaluations=[dict(dimension=item['dimension'], polarity=item['polarity'],
                          uncertainty=copy.deepcopy(item['uncertainty'])) for item in original['evaluations']],
        focus_evidence_ids=[], assessment={k: original['assessment'][k] for k in ('subject', 'verdict')},
        analysis=original['analysis'])
    if result['schema'] == 6:
        result['diagnostic_assessments'] = {group['group_id']: dict(
            judgment='insufficient_evidence', uncertainty=['causal_attribution_unproven'])
            for group in shown.get('diagnostic_groups', [])}
    return result


class ReviewContractTests(unittest.TestCase):
    def test_native_trace_witness_retains_terminal_evidence_and_marks_clipping(self):
        from phase1_codegen.roles.reviewer import native_trace_witness
        from evomem.common import digest
        data = dict(task_id='trace-task', score=0, done=True, error=None,
                    provenance=dict(program_hash='candidate-hash', round=3, phase='current_exploration'),
                    history=[dict(action='action-%d' % i, observation='initial-%d ' % i + 'x' * 5000,
                                  next_observation='feedback-%d ' % i + 'y' * 5000) for i in range(80)])
        shown = native_trace_witness(data, self.tokenizer)
        self.assertEqual(shown['original_hash'], digest(data))
        self.assertEqual(shown['score'], 0)
        self.assertEqual(shown['provenance'], data['provenance'])
        self.assertEqual(shown['step_count'], 80)
        self.assertEqual([s['index'] for s in shown['steps']], [0, 78, 79])
        self.assertEqual(shown['omitted_step_count'], 77)
        self.assertTrue(shown['clipped'])
        self.assertTrue(shown['steps'][-1]['fields']['next_observation']['text'].startswith('feedback-79'))
        self.assertLessEqual(self.tokenizer.count(json.dumps(shown, ensure_ascii=False, sort_keys=True)), 1536)

    def test_trace_citations_survive_sorted_disk_round_trip(self):
        source = contract_payload()
        source['evidence'][3]['data']['history'][0].update(
            observation='current native observation', next_observation='current native feedback')
        shown = bounded_review_payload(source, self.tokenizer)
        raw = synthetic_raw_report(shown, 'neutral')
        raw['observations'].append(observation(shown, 'parent:execution:0', '/steps/0/fields/action/text', 'o3'))
        raw['evaluations'][0]['observation_ids'].append('o3')
        report = validate_report(raw, shown)
        self.assertEqual(validate_report(report, json.loads(json.dumps(shown, sort_keys=True))), report)

    def setUp(self):
        self.tokenizer = ApproximateTokenizer()
        self.payload = contract_payload()
        self.shown = bounded_review_payload(self.payload, self.tokenizer)

    def test_exact_bindings_and_free_analysis_quarantine_preserve_positive_without_reward_gain(self):
        raw = synthetic_raw_report(self.shown, 'positive', 'Missing maps! Add an imaginary architecture.')
        report = validate_report(raw, self.shown)
        self.assertEqual(report['assessment']['verdict'], 'positive')
        self.assertEqual(validate_report(report, self.shown), report)
        self.assertNotIn('Missing maps', json.dumps(report))
        self.assertNotIn('imaginary architecture', json.dumps(report))
        self.assertEqual(self.shown['evidence'][2]['data']['comparison']['delta'], 0)
        changed = copy.deepcopy(report); changed['assessment']['rationale'] += ' invented narrative'
        with self.assertRaisesRegex(ValueError, 'rendered|Rendered'):
            validate_report(changed, self.shown)

    def test_real_old_missing_maps_style_report_cannot_become_trusted_observation(self):
        old = dict(summary='The memory lacks environmental maps.', findings=[], limitations=[],
                   assessment=dict(subject='candidate', verdict='positive', rationale='Needs maps', evidence_ids=['candidate:program']))
        with self.assertRaisesRegex(ValueError, 'contract 2'):
            validate_report(old, self.shown)
        raw = synthetic_raw_report(self.shown)
        raw['observations'][0].update(pointer='/missing_map', value='absent')
        with self.assertRaisesRegex(ValueError, 'not eligible'):
            validate_report(raw, self.shown)
        # Existing map content remains quotable: no architecture-name blacklist.
        raw = synthetic_raw_report(self.shown)
        raw['observations'] = [observation(self.shown, 'parent:read:0', '/fragments/0/fragment/text', excerpt='map')]
        raw['evaluations'][0]['observation_ids'] = ['o1']
        self.assertIn('map', json.dumps(validate_report(raw, self.shown)))

    def test_wrong_scalar_task_program_snapshot_round_or_phase_is_rejected(self):
        for field, value in [('value', 999), ('task_id', 'other'), ('program_hash', 'other'),
                             ('memory_snapshot_hash', 'other'), ('round', 99), ('phase', 'other')]:
            raw = synthetic_raw_report(self.shown)
            if field == 'value': raw['observations'][0][field] = value
            else: raw['observations'][0]['scope'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_report(raw, self.shown)

    def test_positive_content_opinion_cannot_use_validation_or_program_metadata(self):
        raw = synthetic_raw_report(self.shown, 'positive')
        raw['evaluations'][0]['dimension'] = 'Relevance'
        with self.assertRaisesRegex(ValueError, 'eligible current'):
            validate_report(raw, self.shown)
        raw['observations'] = [observation(self.shown, side + ':read:0', '/fragments/0/fragment/text',
                                          'o' + str(i + 1), excerpt='recorded map observation')
                               for i, side in enumerate(('parent', 'candidate'))]
        raw['observations'].append(observation(self.shown, 'candidate:read:0', '/query/goal', 'o3'))
        raw['evaluations'][0]['observation_ids'] = ['o1', 'o2', 'o3']
        self.assertEqual(validate_report(raw, self.shown)['assessment']['verdict'], 'positive')

    def test_current_facts_and_memory_witnesses_precede_long_history(self):
        source = copy.deepcopy(self.payload)
        source['evolution_memory'] = [{'id': 'old', 'hypothesis': 'x' * 1500}]
        source['previous_post_review'] = {'summary': 'hypothetical architecture ' * 500}
        source['evidence'].extend(dict(id='trace-%s' % i, kind='execution_trace',
            data={'history': 'large ' * 10000}) for i in range(12))
        shown = bounded_review_payload(source, self.tokenizer)
        for eid in ('parent:storage', 'candidate:storage', 'parent:read:0', 'candidate:read:0'):
            row = next(r for r in shown['evidence'] if r['id'] == eid)
            self.assertTrue(row['data']['fragments'])
        facts = next(r['data'] for r in shown['evidence'] if r['id'] == 'current:facts')
        self.assertEqual(facts['comparison']['tasks'], [{'task_id': 'task-a', 'parent_score': 0, 'candidate_score': 0, 'delta': 0}])
        self.assertLessEqual(self.tokenizer.chat_count(_messages(shown)), MAX_INPUT_TOKENS)
        self.assertNotIn('hypothetical architecture', json.dumps(shown['previous_post_review']))
        self.assertEqual(source['previous_post_review']['summary'], 'hypothetical architecture ' * 500)

    def test_bound_report_survives_sorted_disk_round_trip_without_scope_id_drift(self):
        raw = synthetic_raw_report(self.shown, 'positive')
        report = validate_report(raw, self.shown)
        persisted = json.loads(json.dumps(self.shown, sort_keys=True))
        self.assertEqual(validate_report(report, persisted), report)

    def test_scope_id_is_exactly_bound_then_normalized_and_reverified(self):
        raw = synthetic_raw_report(self.shown, 'positive')
        expected = validate_report(raw, self.shown)
        for obs in raw['observations']:
            groups = self.shown['citation_registry']['evidence'][obs['evidence_id']]
            obs['scope_id'] = next(g['scope_id'] for g in groups if obs['pointer'] in citation_pointers(g))
            obs.pop('scope')
        result = validate_report(raw, self.shown)
        self.assertEqual(result, expected)
        self.assertEqual(validate_report(result, self.shown), result)
        for obs in result['evidence_contract']['observations']:
            self.assertIn('scope', obs)
            self.assertNotIn('scope_id', obs)
        raw['observations'][0]['scope_id'] = raw['observations'][1]['scope_id']
        with self.assertRaisesRegex(ValueError, 'scope_id'):
            validate_report(raw, self.shown)

    def test_selected_whole_em_and_minimum_parent_candidate_witnesses_are_reserved(self):
        for history_id in ('source-type-record', 'local-evolution-record'):
            source = contract_payload()
            source['evolution_memory'] = [{'id': history_id, 'historical_context': 'e' * 2400}]
            for row in source['evidence']:
                if row['kind'] == 'memory_storage':
                    row['data']['state']['episodes'] = [{'id': str(i), 'experience': 'memory ' * 1800} for i in range(20)]
                elif row['kind'] == 'memory_readout':
                    row['data']['actual_injected_memory'] = row['data']['replayed_readout'] = {
                        'entries': [{'id': str(i), 'experience': 'read ' * 1800} for i in range(20)]}
            source['evidence'].append(dict(id='candidate:behavior', kind='behavior_diagnostics', data=dict(
                facts={'summary': {'calls': 28}, 'branch_counts': [dict(line=i, true_count=i, false_count=10)
                                                                  for i in range(40)]})))
            with self.subTest(history_id=history_id):
                shown = bounded_review_payload(source, self.tokenizer)
                self.assertEqual(shown['evolution_memory'], source['evolution_memory'])
                for eid in ('parent:storage', 'parent:read:0', 'candidate:storage', 'candidate:read:0'):
                    row = next(r for r in shown['evidence'] if r['id'] == eid)
                    self.assertTrue(row['data']['fragments'][0]['fragment']['text'])
                self.assertLessEqual(self.tokenizer.chat_count(_messages(shown)), MAX_INPUT_TOKENS)

    def test_advantages_require_matching_task_and_explicit_runtime_even_with_neutral_verdict(self):
        changed = copy.deepcopy(self.payload)
        row = next(r for r in changed['evidence'] if r['id'] == 'candidate:read:0')
        row['data']['task_id'] = row['data']['query']['task_id'] = 'different-task'
        shown = bounded_review_payload(changed, self.tokenizer)
        raw = synthetic_raw_report(shown)
        raw['observations'] = [observation(shown, side + ':read:0', '/fragments/0/fragment/text',
                                          'o' + str(i + 1), excerpt='recorded map observation')
                               for i, side in enumerate(('parent', 'candidate'))]
        raw['observations'].append(observation(shown, 'candidate:read:0', '/query/goal', 'o3'))
        raw['evaluations'][0].update(dimension='Relevance', polarity='advantage', observation_ids=['o1', 'o2', 'o3'])
        with self.assertRaisesRegex(ValueError, 'eligible current'):
            validate_report(raw, shown)
        absent = copy.deepcopy(self.payload)
        next(r for r in absent['evidence'] if r['id'] == 'candidate:validation')['data']['mechanism_validated'] = None
        shown = bounded_review_payload(absent, self.tokenizer)
        with self.assertRaisesRegex(ValueError, 'explicit current runtime'):
            validate_report(synthetic_raw_report(shown, 'positive'), shown)

    def test_structured_executed_facts_precede_history_and_keep_exact_original_pointers(self):
        source = contract_payload('pre')
        source['evidence'].extend(dict(id='adaptation:' + str(i), kind='adaptation_trace',
                                      data={'history': 'long ' * 2000}) for i in range(10))
        diagnostic = dict(provenance=dict(program_hash='parent-hash', environment='synthetic',
                                           phase='offline_replay', snapshot_hash='executed-bank'),
            coverage={'instrumented_invocations': 3, 'missing_trace_invocations': 0},
            interpretation_boundary='Executed values are not proof of quality.',
            facts={'summary': {'recorded_calls': 7}, 'missing_dict_keys': [
                {'line': 5, 'first_missing': {'key': 'score', 'result': 0}},
                {'line': 6, 'first_missing': {'key': 'long', 'result': 'x' * 9000}}],
                'sorting': [{'line': 9, 'first_sample': {'input': [1, 3], 'result': [1, 3]}}]})
        source['evidence'].append(dict(id='parent:behavior', kind='behavior_diagnostics', data=diagnostic))
        original = copy.deepcopy(source)
        shown = bounded_review_payload(source, self.tokenizer)
        packed = next(row['data'] for row in shown['evidence'] if row['id'] == 'parent:behavior')
        self.assertTrue(packed['clipped'])
        self.assertEqual(packed['facts']['missing_dict_keys'], diagnostic['facts']['missing_dict_keys'][:1])
        self.assertIn('/facts/missing_dict_keys/1', packed['omitted_fact_paths'])
        self.assertEqual(packed['coverage'], diagnostic['coverage'])
        obs = observation(shown, 'parent:behavior', '/facts/missing_dict_keys/0/first_missing/result')
        self.assertEqual(obs['scope']['phase'], 'offline_replay')
        self.assertEqual(obs['scope']['memory_snapshot_hash'], 'executed-bank')
        raw = synthetic_raw_report(shown); raw['observations'] = [obs]
        raw['evaluations'][0]['observation_ids'] = ['o1']
        self.assertEqual(validate_report(raw, shown)['evidence_contract']['observations'][0]['value'], 0)
        raw['observations'][0]['pointer'] = '/facts/missing_dict_keys/1/first_missing/result'
        with self.assertRaisesRegex(ValueError, 'not eligible'):
            validate_report(raw, shown)
        self.assertEqual(source, original)


if __name__ == '__main__': unittest.main()
