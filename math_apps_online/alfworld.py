"""Transactional same-item evaluation using the official ALMA ALFWorld action loop."""
import hashlib
import json
from pathlib import Path

from evomem.common import BASE_URL, MODEL_NAME, digest, file_hash, read
from phase1_codegen.campaign import bind
from phase1_codegen.native import NativeEvaluator, fixed_task_seed
from .native import OnlineTransactions, cost_totals, count_cost


def ledger_slice(path, start, end):
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= path.stat().st_size:
        raise RuntimeError('Invalid native ledger interval')
    with path.open('rb') as stream:
        stream.seek(start)
        payload = stream.read(end - start)
    rows = [json.loads(row) for row in payload.splitlines()]
    return rows, hashlib.sha256(payload).hexdigest()


class ALFWorldOnlineEvaluator(OnlineTransactions, NativeEvaluator):
    """Keep each native attempt read-only. Publish the selected terminal Hook state on item close."""
    def __init__(self, root, tasks, tokenizer, model_fingerprint, context, *, capabilities=None,
                 base_url=BASE_URL, model_name=MODEL_NAME):
        self.family = context.get('family')
        self.domain = 'alfworld'
        if (not self.family or context.get('memory_scope') != 'family' or context.get('environment') != self.domain
                or not tasks or any(t.get('family') != self.family or t.get('environment') != self.domain
                    or t.get('benchmark_split') != 'test' or t.get('split') != Path(t['archive_member']).parts[1]
                    for t in tasks)):
            raise ValueError('Expected a frozen ALFWorld evaluation panel from one task family')
        super().__init__(root, tasks, tokenizer, model_fingerprint, context, capabilities=capabilities,
                         base_url=base_url, model_name=model_name)
        self.tokens, self.caps = self.tokenizer, self.capabilities
        self.task_bindings = self.task_payloads
        self.measurement_locks = self.root/'item_locks'
        self.fingerprint = digest(dict(native=self.fingerprint, online=file_hash(Path(__file__)),
                                       transactions=file_hash(Path(__file__).with_name('native.py'))))

    def _run_attempt(self, program, bank, task, phase, directory):
        self.check_tasks([task])
        self.begin_item(task)
        # Native ALMA transport and finalization share this episode-local ledger.
        # A bounded reference binds even an interrupted or rejected native attempt.
        ledger = directory/'native_costs.jsonl'
        self.runner.ledger = ledger
        ep = self.runner.episode(task, bank, phase, fixed_task_seed(task), allow_write=False)
        rows, evidence_hash = ledger_slice(ledger, ep['actor_ledger_start'], ep['actor_ledger_end'])
        if any(r.get('task_id') != task['id'] or r.get('phase') != phase for r in rows):
            raise RuntimeError('ALMA ledger spans more than one task or attempt')
        canonical = directory/'canonical_episodes.jsonl'
        with canonical.open() as stream:
            if json.loads(stream.readline()) != ep or stream.readline():
                raise RuntimeError('Native runner returned a different episode from its ledger')
        bind(directory/'native_reference.json', dict(canonical_hash=file_hash(canonical),
             ledger_hash=evidence_hash, start=ep['actor_ledger_start'], end=ep['actor_ledger_end']))
        # All Actor accesses, including failed requests, can be attributed without scanning other items.
        relative = str(directory.relative_to(self.root))
        bind(self._cost_directory(task)/'actor'/(digest(relative)+'.json'),
             dict(logical=relative, task_hash=digest(task), receipt=file_hash(directory/'native_reference.json')))
        return ep

    def _actor_rows(self, directory, task):
        ref = read(directory/'native_reference.json')
        rows, ledger_hash = ledger_slice(directory/'native_costs.jsonl', ref['start'], ref['end'])
        if ledger_hash != ref['ledger_hash'] or file_hash(directory/'canonical_episodes.jsonl') != ref['canonical_hash']:
            raise RuntimeError('ALMA native episode or Actor evidence changed')
        actor_rows = [r for r in rows if r.get('role') == 'actor']
        for row in actor_rows:
            if (row['task_id'] != task['id'] or row['kind'] not in ('model', 'local_input_rejection')
                    or row['model_fingerprint'] != self.llm.fingerprint or row['model_name'] != self.llm.model_name
                    or row['original_message_hash'] != digest(row['messages'])):
                raise RuntimeError('ALMA Actor receipt does not match the registered request')
        return actor_rows

    def _actor_costs(self, task, directory):
        totals, chain = cost_totals(), None
        first_seen = False
        for path in sorted((directory/'actor').glob('*.json')):
            access = read(path)
            relative = Path(access['logical'])
            if (relative.is_absolute() or '..' in relative.parts or digest(str(relative)) != path.stem
                    or access['task_hash'] != digest(task)
                    or file_hash(self.root/relative/'native_reference.json') != access['receipt']):
                raise RuntimeError('ALMA Actor costs do not bind this item and attempt')
            for row in self._actor_rows(self.root/relative, task):
                count_cost(totals, False, row['kind'] == 'model', row['ok'], row)
                chain = digest([chain, access['logical'], row])
                first_seen |= row['phase'] == 'first'
        if not first_seen:
            raise RuntimeError('Reviewer cost summary requires the first measured Actor request')
        return totals, chain

    def _verify_native(self, directory, ep, task):
        rows = self._actor_rows(directory, task)
        with (directory/'canonical_episodes.jsonl').open() as stream:
            native = json.loads(stream.readline())
        if native != {k:v for k,v in ep.items() if k != 'native_outcome'}:
            raise RuntimeError('Online episode differs from the native ALMA recorder')
        steps = native['native_recorder']['steps']
        score = native['native_recorder']['reward']
        injections = native['memory_hook']['injections']
        if (native['executor'] != 'ALMA.Agent_Workflow.run_single_task' or native['synthetic'] is not False
                or native['seed'] != fixed_task_seed(task) or native['environment_seed'] != task['seed']
                or native['score'] != score or score not in (0, 1)
                or score != (steps[-1]['scores'] if steps else 0)
                or [r['action'] for r in ep['history']] != [s['action_took'] for s in steps]
                or not 0 <= len(steps) <= len(rows) <= self.step_limit or len(rows) != len(injections)
                or any(not row['ok'] or not inj['actor_delivery_verified'] or inj['actor_request_hash'] != row['request_hash']
                       or row['phase'] != ep['phase'] for row, inj in zip(rows, injections))):
            raise RuntimeError('Invalid multi-step ALFWorld outcome or Actor delivery')
        return dict(score=score, status='solved' if score == 1 else 'unsolved', won=score == 1,
                    completed_steps=len(steps), action_attempts=len(rows), step_limit=self.step_limit)
