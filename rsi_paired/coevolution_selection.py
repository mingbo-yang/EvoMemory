"""Choose one treatment for both resident models using only the selection panel."""
from contextlib import ExitStack, contextmanager
import fcntl
from pathlib import Path

from .panel import save_registration
from .workflow import _read, digest


ARMS = ('A', 'B', 'C', 'D')
MODELS = ('gemma-gpu1', 'qwen-gpu3')
FAMILIES = ('heat', 'two')


def rank_arms(scores, task_count=None):
    """Equal model weights; ties prefer fewer adaptive roles, then alphabetical order."""
    if set(scores) != set(MODELS):
        raise ValueError('Both preregistered model results are required')
    if task_count is not None and (type(task_count) is not int or task_count < 1):
        raise ValueError('Expected a positive registered selection size')
    for by_arm in scores.values():
        if set(by_arm) != set(ARMS) or any(type(n) is not int or n < 0 or
                                          (task_count is not None and n > task_count) for n in by_arm.values()):
            raise ValueError('Four valid selection counts within the registered panel size are required')
    totals = {arm: sum(scores[m][arm] for m in MODELS) for arm in ARMS}
    # A has zero adaptive roles, B and C one, D two.
    return max(ARMS, key=totals.get), totals


def winner_record(root, manifest):
    from .coevolution_campaign import validate_arm_selection

    root = Path(root)
    task_count = len(manifest['selection'])
    scores, hashes = {}, {}
    for name in MODELS:
        cfg = _read(root / 'configs' / f'{name}.json')
        scores[name], hashes[name] = {}, {}
        for arm in ARMS:
            summary = validate_arm_selection(root / name, manifest, cfg, arm)
            if (summary['arm'] != arm or summary['tasks'] != task_count
                    or summary['panel_hash'] != digest(manifest['selection'])
                    or summary['model_config_hash'] != digest(cfg)
                    or summary['registration_hash'] != digest(manifest)):
                raise ValueError(f'Incorrect selection binding: {name}/{arm}')
            scores[name][arm] = summary['successes']
            hashes[name][arm] = digest(summary)
    winner, totals = rank_arms(scores, task_count)
    return dict(arm=winner, totals=totals, successes_by_model=scores, selection_hashes=hashes,
                registration_hash=digest(manifest), selection_panel_hash=digest(manifest['selection']),
                tie_order=list(ARMS), successes_possible_per_arm=len(MODELS) * task_count)


@contextmanager
def _selection_lock(root):
    # A standalone decision and suite enrollment must never race.
    with (root / '.selection.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _plan(family_roots):
    from .panel import COEVOLUTION_PROTOCOL, COEVOLUTION_6X30_PROTOCOL, COEVOLUTION_HEAT_5X23_PROTOCOL

    if set(family_roots) != set(FAMILIES):
        raise ValueError('The pilot requires the heat and two-object families')
    entries, protocols = {}, {}
    comparison = None
    for family in FAMILIES:
        root = Path(family_roots[family]).resolve()
        manifest = _read(root / 'panel.json')
        if manifest.get('family') != family or not manifest['selection']:
            raise ValueError(f'Incorrect selection family or empty panel: {family}')
        protocols[family] = manifest.get('protocol')
        common = [len(manifest['selection']), *(manifest.get(k) for k in
                  ('seed', 'games_sha256', 'metadata_sha256'))]
        if comparison is not None and common != comparison:
            raise ValueError('Suite families must share selection panel size, seed and dataset')
        comparison = common
        entries[family] = dict(run_root=str(root), manifest_hash=digest(manifest))
    legacy_pair = protocols['heat'] == protocols['two'] and protocols['heat'] in (
        COEVOLUTION_PROTOCOL, COEVOLUTION_6X30_PROTOCOL)
    current_pair = protocols == {'heat': COEVOLUTION_HEAT_5X23_PROTOCOL, 'two': COEVOLUTION_6X30_PROTOCOL}
    if not (legacy_pair or current_pair):
        raise ValueError('Unsupported registered family protocol pairing')
    return dict(schema='v4-suite-v1', families=entries)


def _binding(suite_root, plan, family):
    return dict(schema='v4-suite-family-v1', suite_root=str(suite_root),
                study_hash=digest(plan), family=family,
                registration_hash=plan['families'][family]['manifest_hash'])


def register_suite(suite_root, family_roots):
    suite_root = Path(suite_root).resolve()
    plan = _plan(family_roots)
    roots = {f: Path(v['run_root']) for f, v in plan['families'].items()}
    if len(set(roots.values()) | {suite_root}) != len(FAMILIES) + 1:
        raise ValueError('Suite and family run directories must be distinct')
    path = suite_root / 'study.json'
    with ExitStack() as stack:
        for root in sorted(roots.values()):
            stack.enter_context(_selection_lock(root))
        for family, root in roots.items():
            marker = root / 'suite.json'
            if marker.exists():
                if _read(marker) != _binding(suite_root, plan, family):
                    raise ValueError('Family already registered in another suite')
            elif ((root / 'winner.json').exists() or
                  any((root / m / arm / 'selection/complete.json').exists() for m in MODELS for arm in ARMS)):
                raise ValueError('Suite enrollment must precede selection results')
        save_registration(path, plan)
        for family, root in roots.items():
            save_registration(root / 'suite.json', _binding(suite_root, plan, family))
    return path


def suite_runs(suite_root):
    """Return pinned run directories only after validating both registrations."""
    suite_root = Path(suite_root).resolve()
    plan = _read(suite_root / 'study.json')
    roots = {f: Path(entry['run_root']) for f, entry in plan['families'].items()}
    if plan != _plan(roots) or suite_root in roots.values():
        raise ValueError('Suite registration differs from its frozen manifests')
    for f, root in roots.items():
        if _read(root / 'suite.json') != _binding(suite_root, plan, f):
            raise ValueError('Family does not match its frozen suite enrollment')
    return roots


def _suite_record(suite_root, roots):
    records = {f: winner_record(root, _read(root / 'panel.json')) for f, root in roots.items()}
    first, second = (roots[f] for f in FAMILIES)
    for model in MODELS:
        if _read(first / 'configs' / f'{model}.json') != _read(second / 'configs' / f'{model}.json'):
            raise ValueError('Different registered model configurations in the same suite')
    totals = {arm: sum(r['totals'][arm] for r in records.values()) for arm in ARMS}
    return dict(schema='v4-suite-winner-v1', study_hash=digest(_read(suite_root / 'study.json')),
                arm=max(ARMS, key=totals.get), totals=totals, tie_order=list(ARMS),
                successes_possible_per_arm=sum(r['successes_possible_per_arm'] for r in records.values()),
                successes_by_family_model={f: r['successes_by_model'] for f, r in records.items()},
                selection_hashes={f: r['selection_hashes'] for f, r in records.items()})


def _winner_ref(marker, decision):
    return {**marker, 'schema': 'v4-suite-winner-ref-v1', 'arm': decision['arm'],
            'winner_hash': digest(decision)}


def select_suite_winner(suite_root):
    suite_root = Path(suite_root).resolve()
    roots = suite_runs(suite_root)
    decision = _suite_record(suite_root, roots)
    path = suite_root / 'winner.json'
    save_registration(path, decision)
    for root in roots.values():
        save_registration(root / 'winner.json', _winner_ref(_read(root / 'suite.json'), decision))
    return path


def _verify_suite_family(root, manifest):
    marker = _read(root / 'suite.json')
    suite_root = Path(marker['suite_root']).resolve()
    roots = suite_runs(suite_root)
    if roots.get(marker['family']) != root or digest(manifest) != marker['registration_hash']:
        raise ValueError('Final selection is bound to a different suite family')
    decision = _suite_record(suite_root, roots)
    if _read(suite_root / 'winner.json') != decision:
        raise ValueError('Suite winner differs from the bound selection results')
    for family_root in roots.values():
        ref = _winner_ref(_read(family_root / 'suite.json'), decision)
        if _read(family_root / 'winner.json') != ref:
            raise ValueError('Family winner differs from the suite selection results')
    return decision['arm']


def _in_suite(root):
    if (root / 'suite.json').exists():
        return True
    # A missing binding in the standard suite/family layout cannot silently
    # authorize standalone selection.
    parent_plan = root.parent / 'study.json'
    if parent_plan.exists():
        entries = _read(parent_plan).get('families', {}).values()
        if any(entry.get('run_root') == str(root) for entry in entries):
            raise ValueError('Missing frozen family enrollment')
    return False


def select_winner(root, manifest):
    root = Path(root).resolve()
    path = root / 'winner.json'
    with _selection_lock(root):
        if _in_suite(root):
            if not path.exists():
                raise ValueError('Only the suite-wide selection can register this family winner')
            _verify_suite_family(root, manifest)
        else:
            save_registration(path, winner_record(root, manifest))
    return path


def verify_winner(winner_path, manifest, local_model_root, cfg):
    """Rebuild the decision from all registered receipts before final inference."""
    local_model_root = Path(local_model_root).resolve()
    root = local_model_root.parent
    if (local_model_root.name not in MODELS or Path(winner_path).resolve() != root / 'winner.json'
            or _read(root / 'configs' / f'{local_model_root.name}.json') != cfg):
        raise ValueError('Final evaluation requested outside the registered model pair')
    if _in_suite(root):
        return _verify_suite_family(root, manifest)
    stored = _read(root / 'winner.json')
    if stored != winner_record(root, manifest):
        raise ValueError('Global winner differs from the bound selection results')
    return stored['arm']
