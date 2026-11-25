from copy import deepcopy

import pytest

from rsi_paired.coevolution_selection import ARMS, MODELS, rank_arms, select_winner, verify_winner
from rsi_paired.panel import COEVOLUTION_PROTOCOL, COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL
from rsi_paired.workflow import _write, digest
from scripts.run_coevolution_pilot import prior_completed


def test_both_clients_must_exit_successfully_and_publish_their_receipts(tmp_path):
    for model in MODELS:
        for receipt in ('complete.json', 'resident-after.json'):
            _write(tmp_path / model / receipt, {})
    assert prior_completed(tmp_path, lambda _: 'inactive')
    assert not prior_completed(tmp_path, lambda u: 'active' if MODELS[1] in u else 'inactive')
    assert not prior_completed(tmp_path, lambda _: 'failed')
    (tmp_path / MODELS[0] / 'resident-after.json').unlink()
    assert not prior_completed(tmp_path, lambda _: 'inactive')


def test_one_winner_across_models_and_predeclared_ties():
    scores = {MODELS[0]: dict(A=10, B=25, C=20, D=28),
              MODELS[1]: dict(A=10, B=22, C=27, D=15)}
    assert rank_arms(scores) == ('B', dict(A=20, B=47, C=47, D=43))
    scores[MODELS[0]]['A'] = 37
    assert rank_arms(scores)[0] == 'A'
    del scores[MODELS[1]]['B']
    with pytest.raises(ValueError):
        rank_arms(scores)


@pytest.mark.parametrize('size', [30, 50])
def test_final_rejects_changed_or_missing_selection(tmp_path, monkeypatch, size):
    from rsi_paired import coevolution_campaign

    manifest = {'selection': [{'id': i} for i in range(size)]}
    configs = {m: {'model': m} for m in MODELS}
    for model, cfg in configs.items():
        _write(tmp_path / 'configs' / (model + '.json'), cfg)
    mock = {}
    for m in MODELS:
        for arm in ARMS:
            mock[m, arm] = dict(arm=arm, tasks=size, panel_hash=digest(manifest['selection']),
                                registration_hash=digest(manifest), model_config_hash=digest(configs[m]),
                                successes=20 + int(arm == 'D'))
    monkeypatch.setattr(coevolution_campaign, 'validate_arm_selection',
                        lambda path, _m, _cfg, arm: deepcopy(mock[path.name, arm]))
    winner = select_winner(tmp_path, manifest)
    assert verify_winner(winner, manifest, tmp_path / MODELS[0], configs[MODELS[0]]) == 'D'
    mock[MODELS[1], 'A']['successes'] += 1
    with pytest.raises(ValueError, match='selection results'):
        verify_winner(winner, manifest, tmp_path / MODELS[0], configs[MODELS[0]])


@pytest.mark.parametrize('size,heat_protocol,two_protocol', [
    (30, COEVOLUTION_HEAT_5X23_PROTOCOL, COEVOLUTION_6X30_PROTOCOL),
    (30, COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_6X30_PROTOCOL),
    (50, COEVOLUTION_PROTOCOL, COEVOLUTION_PROTOCOL),
])
def test_one_global_arm_on_all_families_and_models(tmp_path, monkeypatch, size, heat_protocol, two_protocol):
    from rsi_paired import coevolution_campaign
    from rsi_paired.coevolution_selection import register_suite, select_suite_winner, suite_runs
    from rsi_paired.workflow import _read

    suite = tmp_path / 'study'
    roots = {f: suite / f for f in ('heat', 'two')}
    configs = {m: {'model': m} for m in MODELS}
    mock = {}
    for family, root in roots.items():
        protocol = heat_protocol if family == 'heat' else two_protocol
        final_size = 50 if protocol == COEVOLUTION_HEAT_5X23_PROTOCOL else size
        manifest = {'family': family, 'protocol': protocol, 'seed': 42, 'games_sha256': 'games',
                    'metadata_sha256': 'meta', 'selection': [{'id': f'{family}-{i}'} for i in range(size)],
                    'final': [{'id': f'held-out-{family}-{i}'} for i in range(final_size)]}
        _write(root / 'panel.json', manifest)
        for model, cfg in configs.items():
            _write(root / 'configs' / (model + '.json'), cfg)
            for arm in ARMS:
                by_arm = {'A': 0, 'B': 21, 'C': 25, 'D': 24} if family == 'heat' else {
                    'A': 0, 'B': 26, 'C': 18, 'D': 24}
                mock[family, model, arm] = dict(arm=arm, tasks=size, panel_hash=digest(manifest['selection']),
                    registration_hash=digest(manifest), model_config_hash=digest(cfg), successes=by_arm[arm])
    monkeypatch.setattr(coevolution_campaign, 'validate_arm_selection',
                        lambda model_root, _man, _cfg, arm: deepcopy(mock[model_root.parent.name, model_root.name, arm]))
    plan = register_suite(suite, roots)
    assert plan == register_suite(suite, roots)
    assert suite_runs(suite) == roots
    first = roots['heat']
    with pytest.raises(ValueError, match='Only the suite-wide'):
        select_winner(first, _read(first / 'panel.json'))
    missing = mock.pop(('two', MODELS[1], 'D'))
    with pytest.raises(KeyError):
        select_suite_winner(suite)
    assert all(not (p / 'winner.json').exists() for p in (suite, *roots.values()))
    mock['two', MODELS[1], 'D'] = missing
    selection = _read(select_suite_winner(suite))
    assert selection['arm'] == 'D'
    assert selection['totals'] == dict(A=0, B=94, C=86, D=96)
    assert selection['successes_possible_per_arm'] == 4 * size
    for family, root in roots.items():
        manifest = _read(root / 'panel.json')
        ref = _read(root / 'winner.json')
        assert ref['arm'] == 'D' and ref['winner_hash'] == digest(selection)
        assert ref['family'] == family and ref['study_hash'] == digest(_read(plan))
        for model in MODELS:
            assert verify_winner(root / 'winner.json', manifest, root / model, configs[model]) == 'D'
    # A valid-looking local recommendation cannot override another family's score.
    _write(first / 'winner.json', dict(_read(first / 'winner.json'), arm='C'))
    with pytest.raises(ValueError, match='suite selection'):
        verify_winner(first / 'winner.json', _read(first / 'panel.json'), first / MODELS[0], configs[MODELS[0]])
    (first / 'winner.json').unlink()
    select_suite_winner(suite)
    mock['two', MODELS[0], 'A']['successes'] = 1
    with pytest.raises(ValueError, match='Suite winner differs'):
        verify_winner(first / 'winner.json', _read(first / 'panel.json'), first / MODELS[0], configs[MODELS[0]])


def test_suite_enrollment_precedes_results_and_requires_original_manifests(tmp_path):
    from rsi_paired.coevolution_selection import register_suite, select_suite_winner, suite_runs
    from rsi_paired.workflow import _read
    suite = tmp_path / 'pilot'
    roots = {f: suite / f for f in ('heat', 'two')}
    for family, root in roots.items():
        _write(root / 'panel.json', {'family': family, 'protocol': COEVOLUTION_6X30_PROTOCOL,
                                    'selection': [{'id': family}]})
    path = roots['heat'] / MODELS[0] / 'A/selection/complete.json'
    _write(path, {})
    with pytest.raises(ValueError, match='precede'):
        register_suite(suite, roots)
    assert not (suite / 'study.json').exists()
    path.unlink()
    register_suite(suite, roots)
    _write(roots['two'] / 'panel.json', {'family': 'two', 'protocol': COEVOLUTION_6X30_PROTOCOL,
                                         'selection': [{'id': 'other'}]})
    with pytest.raises(ValueError, match='registration differs'):
        suite_runs(suite)
    with pytest.raises(ValueError):
        select_suite_winner(suite)
    (roots['heat'] / 'suite.json').unlink()
    with pytest.raises(ValueError, match='Missing frozen'):
        select_winner(roots['heat'], _read(roots['heat'] / 'panel.json'))


def test_registered_30_task_counts_check_upper_bound():
    scores = {m: {a: 30 for a in ARMS} for m in MODELS}
    assert rank_arms(scores, 30)[0] == 'A'
    scores[MODELS[0]]['D'] = 31
    with pytest.raises(ValueError, match='panel size'):
        rank_arms(scores, 30)


@pytest.mark.parametrize('heat,two', [
    (COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL),
    (COEVOLUTION_HEAT_5X23_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL),
    (COEVOLUTION_HEAT_5X23_PROTOCOL, COEVOLUTION_PROTOCOL),
    ('unregistered', 'unregistered'),
])
def test_suite_rejects_swapped_or_unregistered_protocols(tmp_path, heat, two):
    from rsi_paired.coevolution_selection import register_suite
    roots = {family: tmp_path / family for family in ('heat', 'two')}
    for family, protocol in [('heat', heat), ('two', two)]:
        _write(roots[family] / 'panel.json', dict(family=family, protocol=protocol,
               seed=42, games_sha256='games', metadata_sha256='meta', selection=[{'id': i} for i in range(30)]))
    with pytest.raises(ValueError, match='protocol pairing'):
        register_suite(tmp_path, roots)
    assert not (tmp_path / 'study.json').exists()
