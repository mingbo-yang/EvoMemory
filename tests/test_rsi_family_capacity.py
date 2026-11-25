"""Public evidence packing for a 30-task family gate. No model calls."""
from copy import deepcopy

import pytest

from evomem.common import dumps
from rsi_paired.roles import Roles
from rsi_paired.workflow import digest


class Characters:
    def count(self, text):
        return len(text)


def roles(tmp_path, limit):
    return Roles(tmp_path, {'model_name': 'cpu', 'model_fingerprint': 'offline',
                            'context_tokens': 262144, 'public_token_limit': limit},
                 transport=True, tokenizer=Characters())


def table(distinct):
    panel = {}
    for arm in ('parent', 'P1', 'P2', 'P3'):
        label = arm if distinct else 'same'
        source = '# ' + (label + ' mechanism code ') * 75
        rows = []
        for i in range(30):
            task = f'item-{i}'
            origin = 'historical_success' if i < 6 else 'historical_failure' if i < 12 else 'fresh'
            score = None if i == 1 else i % 2
            rows.append({'task_id': task, 'gate_source': origin, 'episode_hash': digest([task, label]),
                         'native_score': score, 'score': score, 'status': 'failed' if score is None else 'completed',
                         'evidence': {'steps': i, 'goal': f'{task}/{label} ' + 'evidence ' * 60}})
        panel[arm] = {'source': source, 'source_hash': digest(source), 'evaluation': rows,
                      'success_count': sum(row['score'] or 0 for row in rows)}
    return {'phase': 'post_primary', 'pre': {'recent_learning': []}, 'primary': panel}


def lookup(packet, pointer):
    value = packet
    for name in pointer.strip('/').split('/'):
        value = value[int(name)] if isinstance(value, list) else value[name]
    return value


def test_duplicate_sources_and_witnesses_still_resolve_to_full_values(tmp_path):
    public = table(False)
    original = deepcopy(public)
    packet = roles(tmp_path, len(dumps(public)) * 3 // 4).compress(public)
    assert 'omission' not in packet
    for slot, record in packet['primary'].items():
        src = record.get('source') or lookup(packet, record['source_ref'])
        assert src == original['primary'][slot]['source']
        for i, row in enumerate(record['evaluation']):
            facts = row.get('evidence') or lookup(packet, row['evidence_ref'])
            prior = original['primary'][slot]['evaluation'][i]
            assert facts == prior['evidence']
            assert {k: row[k] for k in prior if k != 'evidence'} == {k: v for k, v in prior.items() if k != 'evidence'}
    assert sum('source' in r for r in packet['primary'].values()) == 1
    assert public == original


def test_distinct_witnesses_keep_native_outcomes_counts_and_archive_bindings(tmp_path):
    public = table(True)
    packet = roles(tmp_path, len(dumps(public)) // 2).compress(public)
    assert packet['omission']['episode_witnesses_reduced'] == 120
    for slot, record in packet['primary'].items():
        assert record['source'] == public['primary'][slot]['source']
        for row, original in zip(record['evaluation'], public['primary'][slot]['evaluation'], strict=True):
            assert row == {**original, 'evidence': {'steps': original['evidence']['steps']}}
    with pytest.raises(ValueError, match='Mandatory public evidence'):
        roles(tmp_path, 100).compress(public)


def test_evolved_strategy_reads_sources_and_episode_evidence_as_before_packing(tmp_path):
    from rsi_paired.roles import INITIAL_STRATEGY, _policy_evidence, policy_instruction
    from phase1_codegen.program import BASELINE_SOURCE

    full = table(False)
    proposer = roles(tmp_path, len(dumps(full)) * 3 // 4)
    compact = proposer.compress(full)
    original_hash = digest(compact)
    assert 'source_ref' in compact['primary']['P2']
    assert 'evidence_ref' in compact['primary']['P2']['evaluation'][4]
    strategy = deepcopy(INITIAL_STRATEGY)
    strategy['propose']['code'] = (
        'def propose(evidence, repairing):\n'
        '    a = evidence["primary"]["P1"]\n'
        '    b = evidence["primary"]["P2"]\n'
        '    if a["source"] == b["source"] and b["evaluation"][4]["evidence"]["steps"] == 4:\n'
        '        return "Read the supplied evidence."\n'
        '    return ""')
    assert policy_instruction(strategy, compact, True) == policy_instruction(strategy, full, True)
    captured = []

    def capture(role, system, body, seed, **kwargs):
        assert role == 'propose'
        captured.append((system, deepcopy(body)))
        return {'source': BASELINE_SOURCE}, digest(body['public_evidence'])

    proposer._call = capture
    for version in (INITIAL_STRATEGY, strategy):
        assert proposer.propose(version, BASELINE_SOURCE, compact, 10, None, None)['public_hash'] == original_hash
    assert captured[0][1] == captured[1][1] and captured[1][1]['public_evidence'] == compact
    assert 'Read the supplied evidence.' in captured[1][0]
    assert digest(compact) == original_hash
    for path in ['/pre/recent_learning/0/evidence', '../../raw', '/primary/P2/source']:
        bad = deepcopy(compact)
        bad['primary']['P2']['source_ref'] = path
        with pytest.raises(ValueError, match='Invalid displayed'):
            _policy_evidence(bad)
    shortened = deepcopy(compact)
    ref = shortened['primary']['P2']['evaluation'][4]['evidence_ref']
    shown = lookup(shortened, ref)
    shown.pop('goal')
    assert 'goal' not in _policy_evidence(shortened)['primary']['P2']['evaluation'][4]['evidence']


def test_meta_review_correction_binds_same_public_evidence_and_previous_response(tmp_path):
    import json
    from rsi_paired.roles import Roles
    from rsi_paired.workflow import ReviewOutputError

    class Tokens(Characters):
        def chat_count(self, messages):
            return len(dumps(messages)) // 3

    answers = iter(['{"findings":[{"judgment":"supported_defect","evidence_ids":["hidden"],"observation":"Wrong ID."}]}',
                    '{"findings":[{"judgment":"insufficient_evidence","evidence_ids":["meta.N1"],"observation":"The episode is insufficient."}]}'])
    requests, responses = [], []

    def transport(request):
        requests.append(request)
        result = dict(model=request['model'], choices=[dict(finish_reason='stop', message=dict(content=next(answers)))])
        responses.append(result)
        return result

    r = Roles(tmp_path, dict(model_name='cpu', model_fingerprint='offline', context_tokens=24000),
              tokenizer=Tokens(), transport=transport)
    row = {'task_id': 't', 'score': 0, 'native_score': 0, 'evidence': {'review_witness': {'native_steps': 2}}}
    records = {k: {'source': 'code', 'evaluation': [row]} for k in ('retained', 'N1')}
    with pytest.raises(ReviewOutputError, match='displayed IDs') as failed:
        r.review('code', records)
    previous = {'error': str(failed.value), 'previous_response': failed.value.draft}
    ref = previous['previous_response']
    assert ref['response_hash'] == digest(responses[0])
    assert ref['output_hash'] == digest(responses[0]['choices'][0]['message']['content'])
    assert (r.root / 'requests' / ref['request_key'] / 'response.json').is_file()
    with pytest.raises(ValueError, match='original displayed'):
        r.review_retry('changed code', records, previous)
    corrected = r.review_retry('code', records, previous)
    original, retry = [json.loads(x['messages'][1]['content']) for x in requests]
    assert retry.pop('format_correction') == previous and retry == original
    assert corrected['evidence_hash'] == ref['displayed_evidence_hash'] == digest(original)
    assert requests[0]['messages'][0] == requests[1]['messages'][0]
    assert corrected['findings'][0]['evidence_ids'] == ['meta.N1']


def test_missing_meta_review_is_transferable_as_status_without_model_prose():
    from rsi_paired.roles import _compact_experiences

    event = {'meta_review': {'status': 'missing', 'records_hash': digest('sealed meta'),
                             'findings': [], 'attempt_errors': [
                                 {'error': 'sensitive invalid claim', 'previous_response': {'output_hash': 'abc'}}]},
             'modification_records': [{'slot': 'N1', 'success_count': 3, 'task_count': 30}]}
    row, = _compact_experiences([event], Characters())
    assert row['review_status'] == 'missing' and row['review_format_failures'] == 1
    assert row['review_records_hash'] == event['meta_review']['records_hash']
    assert row['reviewer_judgments'] == [] and row['success_count'] == 3
    assert 'sensitive invalid claim' not in str(row)


def test_public_table_preserves_each_native_outcome_and_authenticates_omissions(tmp_path):
    from rsi_paired.audit import Audit
    from rsi_paired.roles import _expand_public_rows

    packet = table(True)
    packed = roles(tmp_path, 20000).compress(packet)
    assert len(dumps(packed)) <= 20000
    assert packed['omission']['layout'] == 'archive_refs_v1'
    assert 'episode_table_format' in packed
    expanded = _expand_public_rows(packed)
    for slot, original in packet['primary'].items():
        for prior, actual in zip(original['evaluation'], expanded['primary'][slot]['evaluation'], strict=True):
            assert actual['gate_source'] == prior['gate_source']
    audit = Audit(tmp_path)
    audit.batch.mkdir()
    (audit.batch / 'H_pre.json').write_text(dumps(packet['pre']))
    audit.report['panel_sampling'] = {'primary': {'entries': [
        {'source': row['gate_source']} for row in packet['primary']['parent']['evaluation']]}}
    audit.family_feedback(packed, packet['primary'], [])
    assert not audit.errors
    table_view = packed['primary']['parent']['evaluation']
    first = table_view['rows'][0]
    columns = table_view['columns'][first[0]]
    first[1 + columns.index('episode_hash_prefix')] = digest('different episode')[:16]
    audit.family_feedback(packed, packet['primary'], [])
    assert any('identity or native result' in error for error in audit.errors)


def test_repeated_candidate_failures_fit_and_resolve_to_exact_errors(tmp_path):
    from rsi_paired.roles import _expand_public_rows

    packet = table(True)
    original = packet['primary']['P1']['evaluation']
    for i, row in enumerate(original):
        row.update(score=0, native_score=None, status='method_eval_failure', candidate_error={
            'message': 'synthetic rejected event ' * 15, 'error_hash': digest(i % 2)})
    packed = roles(tmp_path, 21000).compress(packet)
    wire = packed['primary']['P1']['evaluation']
    assert len(wire['candidate_errors']) == 2
    expanded = _expand_public_rows(packed)
    for stored, actual in zip(original, expanded['primary']['P1']['evaluation'], strict=True):
        assert actual['candidate_error'] == stored['candidate_error']
        assert (actual['score'], actual['native_score'], actual['status']) == (0, None, 'method_eval_failure')
    row = wire['rows'][0]
    row[wire['columns'][row[0]].index('candidate_error_index') + 1] = -1
    with pytest.raises(ValueError, match='candidate error reference'):
        _expand_public_rows(packed)


def test_compact_reviewer_request_keeps_per_task_outcomes_and_whole_witnesses(tmp_path):
    import json
    from rsi_paired.roles import _review_layout, _review_witness_index, _score_view

    class TokenCounter(Characters):
        def chat_count(self, messages):
            return len(json.dumps(messages)) // 3

    requests = []
    def transport(request):
        requests.append(request)
        return {'model': request['model'], 'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'findings': [{'evidence_ids': ['primary.P1'],
                'observation': 'Synthetic result.', 'judgment': 'insufficient_evidence'}]})}}]}

    reviewer = Roles(tmp_path, {'model_name': 'cpu', 'model_fingerprint': 'offline', 'context_tokens': 9000},
                  tokenizer=TokenCounter(), transport=transport)
    records = table(True)['primary']
    for rec in records.values():
        for row in rec['evaluation']:
            row['evidence']['review_witness'] = {'text': 'complete synthetic witness ' + row['task_id']}
    reviewer.review('parent text', records)
    request, = requests
    body = json.loads(request['messages'][1]['content'])
    assert reviewer.tokens.chat_count(request['messages']) + request['max_tokens'] <= reviewer.context
    assert body['evidence_layout'] == 'table_v1'
    indexed = {'parent.source': 'parent text', **{'primary.' + k: _score_view(v) for k, v in records.items()}}
    witnesses = _review_witness_index(indexed)
    expected, abbreviated = _review_layout(indexed, witnesses, 'table_v1')
    assert expected == body['evidence'] and digest(indexed) == body['evidence_source_hash']
    assert all(value == abbreviated[ref] for ref, value in body['episode_witnesses'].items())
    for slot, rec in records.items():
        shown = body['evidence']['primary.' + slot]
        for row, values in zip(rec['evaluation'], shown['evaluation'], strict=True):
            decoded = dict(zip(shown['evaluation_columns'], values, strict=True))
            assert all(decoded[key] == row[key] for key in ('task_id', 'score', 'native_score', 'status', 'gate_source'))


def test_default_roles_count_token_ids_in_mapping_outputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase1_codegen.tokenization import Tokenizer

    ids = list(range(153))
    def render(messages, **options):
        assert options['add_generation_prompt'] is True and options['enable_thinking'] is False
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}

    def init(tokenizer, path, *, model_name):
        tokenizer.model_name = model_name
        tokenizer.impl = SimpleNamespace(apply_chat_template=render)

    monkeypatch.setattr(Tokenizer, '__init__', init)
    reviewer = Roles(tmp_path, {'model_name': 'cpu', 'model_fingerprint': 'offline',
                                'context_tokens': 32768, 'tokenizer_path': 'synthetic'}, transport=True)
    assert reviewer.tokens.chat_count([{'role': 'user', 'content': 'A synthetic request.'}]) == len(ids)
