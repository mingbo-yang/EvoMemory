"""One first answer followed by Reviewer-directed same-question repair, in stream order."""
from collections import defaultdict
import fcntl
import hashlib
from itertools import accumulate
import json
import os
from pathlib import Path

from evomem.common import digest, read
from phase1_codegen.acceptance import REVIEWER_DIRECTED
from phase1_codegen.program import BASELINE_SOURCE, Program

PROTOCOL = 'online-same-question-v2'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def registered_test_item(task, domain):
    if task.get('environment') != domain or not task.get('family'):
        return False
    if domain == 'alfworld':
        # CrossEp-Emb publishes a test mapping into ALFWorld's original asset split.
        return task.get('benchmark_split') == 'test' and task.get('split') == 'valid_train'
    return domain in ('math', 'apps') and task.get('split') == 'test'


def wire_hash(value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def publish(path, value):
    """Publish atomically without sorting nested mechanism-visible mappings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checked_episode(ep, task, source):
    require(ep['task_id'] == task['id'] and ep['family'] == task['family']
            and ep['environment'] == task['environment'] and ep['program_hash'] == Program(source).hash,
            'Episode belongs to a different task, family, or mechanism')
    score = ep.get('score')
    require(ep.get('native_reward_verified') is True and not ep.get('error')
            and type(score) in (float, int) and score in (0, 1), 'Missing verified binary native score')
    return int(score)


def check_snapshot(snapshot, source, logs, task):
    require(snapshot['program_hash'] == Program(source).hash and snapshot['environment'] == task['environment']
            and snapshot['memory_scope'] == 'family' and snapshot['source_log_hash'] == digest(logs)
            and snapshot.get('family') in (None, task['family'])
            and all(f == task['family'] for f in snapshot.get('families', [])),
            'Snapshot does not correspond to this family, program, and history')


def check_repair(result, task, old_source, first_episode, first_snapshot, logs):
    attempts = result['attempts']
    require(attempts and len(attempts) == len(result['events']),
            'A first failure requires at least one proposal and a record for each attempt')
    for i, row in enumerate(attempts, 1):
        score = row['candidate_score']
        require(score is None or (type(score) in (float, int) and score in (0, 1)), 'Invalid candidate score')
        decision = row['decision']
        require(decision in ('continue', 'adopt', 'stop') and row['attempt'] == i
                and isinstance(row.get('reason'), str) and row['reason'].strip()
                and row['accepted'] is (decision == 'adopt')
                and result['events'][i-1]['accepted'] is row['accepted'],
                'Inconsistent Reviewer decision or evidence record')
        require((decision == 'continue') == (i < len(attempts)), 'Repair must end at the first terminal decision')
        require(decision != 'adopt' or score == 1, 'Adoption requires a correct verified candidate')
    require(result['accepted'] is attempts[-1]['accepted'], 'Deployment differs from final decision')
    require(result['solved'] is any(a['candidate_score'] == 1 for a in attempts),
            'Solved and adopted require separate accounting')
    if result['accepted']:
        require(result['source'] != old_source and checked_episode(result['episode'], task, result['source']) == 1,
                'An accepted replacement needs its own correct native episode')
    else:
        require(result['source'] == old_source and wire_hash(result['episode']) == wire_hash(first_episode)
                and wire_hash(result['snapshot']) == wire_hash(first_snapshot), 'Unaccepted proposal changed deployment')
    check_snapshot(result['snapshot'], result['source'], logs, task)
    for event in result['events']:
        require(event['observed_context']['environment'] == task['environment']
                and event['observed_context']['family'] == task['family'], 'Foreign evolution record')


def execute_item(task, source, logs, snapshot, records, directory, evaluate, repair, commit):
    ep, before = evaluate(task=task, source=source, logs=logs, snapshot=snapshot, directory=directory)
    first_score = checked_episode(ep, task, source)
    check_snapshot(before, source, logs, task)
    if first_score:
        result = dict(source=source, episode=ep, snapshot=before, events=[], attempts=[], accepted=False, solved=True)
    else:
        result = repair(task=task, source=source, logs=logs, snapshot=before, first_episode=ep,
                        records=records, directory=directory)
        check_repair(result, task, source, ep, before, logs)
    after = commit(task=task, source=result['source'], logs=logs, snapshot=result['snapshot'],
                   episode=result['episode'], directory=directory)
    check_snapshot(after, result['source'], [*logs, result['episode']], task)
    return dict(result, snapshot=after, task_id=task['id'], family=task['family'], first_score=first_score,
                first_episode_hash=wire_hash(ep), first_snapshot_hash=wire_hash(before))


def added_to_summary(summary, item):
    summary['completed'] += 1
    summary['first_correct'] += item['first_score']
    summary['solved'] += int(item['solved'])
    summary['adopted'] += int(item['accepted'])
    attempts = item['attempts']
    summary['proposals'] += len(attempts)
    summary['solved_on_retry_without_adoption'] += int(bool(attempts) and item['solved'] and not item['accepted'])
    for key in ('proposals_by_attempt', 'native_comparisons_by_attempt', 'adoptions_by_attempt', 'stops_by_attempt'):
        summary[key].extend([0] * (len(attempts) - len(summary[key])))
    # Count the first success at each depth; prefix sums after the stream give cumulative solves.
    summary['correct_by_attempt'].extend([0] * (len(attempts) + 1 - len(summary['correct_by_attempt'])))
    for i, attempt in enumerate(attempts):
        summary['proposals_by_attempt'][i] += 1
        summary['native_comparisons_by_attempt'][i] += int(attempt['candidate_score'] is not None)
        summary['adoptions_by_attempt'][i] += int(attempt['decision'] == 'adopt')
        summary['stops_by_attempt'][i] += int(attempt['decision'] == 'stop')
    first_win = 0 if item['first_score'] else next(
        (a['attempt'] for a in attempts if a['candidate_score'] == 1), None)
    if first_win is not None:
        summary['correct_by_attempt'][first_win] += 1


def run_stream(*, root, domain, tasks, binding, evaluate, repair, commit, initial_source=BASELINE_SOURCE,
               condition='cross_type_em'):
    require(condition in ('cross_type_em', 'no_external_em'), 'Unknown evolution memory condition')
    require(tasks and len({t['id'] for t in tasks}) == len(tasks)
            and all(registered_test_item(t, domain) for t in tasks),
            'Expected unique, ordered official test tasks of a single dataset')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = dict(protocol=PROTOCOL, domain=domain, condition=condition, repair_policy=REVIEWER_DIRECTED,
                initial_source=initial_source, tasks=tasks, binding=binding)
    with (root/'.stream.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan_path = root/'online_plan.json'
        if plan_path.exists():
            require(wire_hash(read(plan_path)) == wire_hash(plan), 'Changed registered stream or configuration')
        else:
            require(not any(p.name != '.stream.lock' for p in root.iterdir()), 'New protocol needs a fresh output directory')
            publish(plan_path, plan)
        item_root = root/'items'
        item_receipts = {p.name: (p/'item.json').is_file() for p in item_root.glob('*') if p.is_dir()}
        committed = sum(item_receipts.values())
        require(len(item_receipts) <= len(tasks) and len(item_receipts) - committed <= 1
                and item_receipts == {f'{i:06d}': i < committed for i in range(len(item_receipts))},
                'Expected a committed prefix of the registered stream followed by at most one unfinished item')
        summary_path = root/'summary.json'
        saved_summary = read(summary_path) if summary_path.exists() else None
        if summary_path.exists():
            require(isinstance(saved_summary, dict) and committed == len(tasks)
                    and saved_summary.get('completed') == committed,
                    'Final stream summary requires the complete set of committed item receipts')
        previous = wire_hash(plan)
        source = initial_source
        logs, snapshots, events, event_ids = defaultdict(list), {}, [], set()
        summary = dict(completed=0, first_correct=0, solved=0, adopted=0, proposals=0,
                       solved_on_retry_without_adoption=0, proposals_by_attempt=[], native_comparisons_by_attempt=[],
                       adoptions_by_attempt=[], stops_by_attempt=[], correct_by_attempt=[0])
        for index, task in enumerate(tasks):
            family = task['family']
            directory = root/'items'/f'{index:06d}'
            path = directory/'item.json'
            if index < committed:
                item = read(path)
                stored = item.pop('receipt_hash')
                require(stored == wire_hash(item) and item['previous'] == previous
                        and item['task_id'] == task['id'] and item['family'] == family, 'Committed stream receipt changed')
            else:
                directory.mkdir(parents=True, exist_ok=True)
                eligible = events if condition == 'cross_type_em' else [
                    e for e in events if e['observed_context']['family'] == family]
                prior = snapshots.get(family)
                if prior is not None and prior['program_hash'] != Program(source).hash:
                    prior = None  # Reconstruct only this family's earlier experiences under the inherited code.
                try:
                    item = execute_item(task, source, logs[family], prior, list(eligible), directory,
                                        evaluate, repair, commit)
                except Exception as exc:
                    # No item receipt exists yet. Record the interruption independently of the
                    # two memory histories and leave the durable prefix at the preceding item.
                    pause = dict(task_id=task['id'], family=family, previous=previous,
                                 deployed_program=Program(source).hash,
                                 error_type=type(exc).__name__, error=str(exc))
                    publish(directory/'interruptions'/(wire_hash(pause)+'.json'), pause)
                    raise
                item['previous'] = previous
                stored = wire_hash(item)
            ids = [e['id'] for e in item['events']]
            require(len(set(ids)) == len(ids) and not event_ids.intersection(ids), 'Duplicate evolution record')
            if index >= committed:
                publish(path, dict(item, receipt_hash=stored))
            previous = stored
            source = item['source']
            logs[family].append(item['episode'])
            snapshots[family] = item['snapshot']
            events.extend(item['events'])
            event_ids.update(ids)
            added_to_summary(summary, item)
        summary['correct_by_attempt'] = list(accumulate(summary['correct_by_attempt']))
        summary.update(domain=domain, condition=condition, repair_policy=REVIEWER_DIRECTED, final_receipt=previous)
        if saved_summary is None:
            publish(summary_path, summary)
        else:
            require(summary == saved_summary, 'Final summary differs from its committed stream receipt chain')
        return summary
