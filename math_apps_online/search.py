"""Same-question mechanism repair with one diagnosis and a Reviewer decision on every draft."""
import hashlib
import json
import os
from pathlib import Path

from evomem.alma_backend import public_context as visible_context
from evomem.common import digest, file_hash, read, write
from phase1_codegen.acceptance import REVIEWER_DIRECTED
from phase1_codegen.behavior_diagnostics import diagnose
from phase1_codegen.engine import EvolutionEngine, MeasurementFailure, ProposalCallFailure
from phase1_codegen.experience import EvolutionMemory, SharedCodeView, private_learning_texts, public_event
from phase1_codegen.memory import CodeMemory
from phase1_codegen.outcomes import reported_score
from phase1_codegen.policy import Rejected
from phase1_codegen.program import WorkerStartupError
from phase1_codegen.proposer import CodeEvolver, EVOLVER_PROMPT
from .online import registered_test_item


def _online_prompt():
    start = EVOLVER_PROMPT.index('When context.memory_scope is environment,')
    end = EVOLVER_PROMPT.index('Do not hard-code any episode', start)
    return (EVOLVER_PROMPT[:start] +
            'Task Memory and the adaptation history are restricted to the current task family. '
            'The supplied mechanism-change experiences may span families within this dataset. '
            'Use only the records provided for this condition. Every draft is tested on this '
            'same item, goal and seed with the same pre-item family history. On a retry, compare '
            'your proposed change with the supplied earlier draft, its validation error or '
            'measured behavior, and the Reviewer feedback. Submit a change whose effect can be '
            'tested here. Re-emitting a failed draft, including changes only to its formatting '
            'or commentary, cannot test a new hypothesis. Do not rely on changing the goal, '
            'running a different task or adding experiences for the test. '
            'Write the hypothesis, risk, any '
            'code comments or docstrings, and all instructions the mechanism sends to other models '
            'in English. Preserve task data and API identifiers verbatim.\n'
            + EVOLVER_PROMPT[end:])


ONLINE_EVOLVER_PROMPT = _online_prompt()


class OnlineEvolver(CodeEvolver):
    """Reuse role, language, budgets, evidence packing and transport with split memory scopes."""

    system_prompt = ONLINE_EVOLVER_PROMPT


def _alfworld_action_facts(episode):
    """Count action-list matches before a step and exact simulator feedback after it."""
    recorder = episode.get('native_recorder')
    if not isinstance(recorder, dict) or not isinstance(recorder.get('initial'), dict):
        raise MeasurementFailure('Missing native ALFWorld recorder')
    steps = recorder.get('steps')
    if not isinstance(steps, list) or len(steps) != episode['native_outcome']['completed_steps']:
        raise MeasurementFailure('Native ALFWorld step count differs from its recorder')
    available = recorder['initial'].get('actions_list')
    actions = []
    listed = nothing_happens = 0
    for step in steps:
        if not isinstance(step, dict):
            raise MeasurementFailure('Malformed native ALFWorld step')
        action = step.get('action_took')
        observation = step.get('obs')
        if (not isinstance(available, list) or not all(isinstance(value, str) for value in available)
                or not isinstance(action, str) or not isinstance(observation, str)):
            raise MeasurementFailure('Missing native ALFWorld actions, chosen action or observation')
        actions.append(action)
        listed += action in available
        nothing_happens += observation == 'Nothing happens.'
        available = step.get('actions_list')
    return dict(listed_action_steps=listed, nothing_happens_steps=nothing_happens,
                action_sequence_hash=digest(actions))


class ItemSearch(EvolutionEngine):
    """A proposal's parent measurement is the original answer to this item."""

    def __init__(self, *, out, tokenizer, proposer, reviewer, evaluator, task, parent, logs,
                 parent_snapshot, first_episode, records, seed, capabilities, synthetic):
        self.first_episode = first_episode
        self.parent_snapshot = parent_snapshot
        super().__init__(out, tokenizer, proposer, evaluator, EvolutionMemory(records), evaluator.context,
                         logs, [task], [], initial_source=parent.source, seed=seed, reviewer=reviewer,
                         acceptance_policy=REVIEWER_DIRECTED, capabilities=capabilities, synthetic=synthetic)
        self.active_snapshot_ref = {'hash': parent_snapshot['snapshot_hash']}
        self.binding.update(item_search='online-same-item-v2', implementation=file_hash(Path(__file__)),
                            review_implementation=file_hash(Path(__file__).with_name('decision.py')),
                            first_answer=digest(first_episode), pre_item_snapshot=parent_snapshot['snapshot_hash'])

    def _validate_plan(self):
        task, = self.explore
        family = task['family']
        domain = task['environment']
        if (not registered_test_item(task, domain) or self.context.get('environment') != domain
                or self.context.get('family') != family or self.memory_scope != 'family'
                or self.max_rounds is not None):
            raise ValueError('Online repair requires a registered single-family test item and Reviewer-directed stopping')
        ids = set()
        for ep in self.logs:
            if (ep.get('environment') != domain or ep.get('family') != family or ep.get('error')
                    or not ep.get('task_id') or ep['task_id'] in ids or ep['task_id'] == task['id']
                    or not self.synthetic and (ep.get('synthetic') is not False or not ep.get('native_reward_verified'))):
                raise ValueError('Expected only earlier, unique, verified same-family experiences')
            ids.add(ep['task_id'])
        if any(r['observed_context'].get('environment') != domain or r.get('synthetic') is not self.synthetic
               for r in self.memory.records):
            raise ValueError('Evolution experience crosses the dataset or verification boundary')
        if not self.synthetic and not getattr(self.proposer, 'fingerprint', None):
            raise ValueError('Formal online search requires a bound Evolver')
        bank = CodeMemory.from_snapshot(self.initial, self.parent_snapshot, self.tokenizer,
                                        context=self.context, capabilities=self.capabilities)
        if (bank.source_log_hash != digest(self.logs) or bank.memory_scope != 'family'
                or bank.families - {family} or self.first_episode.get('program_hash') != self.initial.hash
                or self.first_episode.get('memory_hash_before') != bank.fingerprint
                or self.first_episode.get('memory_hash_after') != bank.fingerprint
                or self.first_episode.get('family') != family or self.first_episode.get('environment') != domain
                or not self._complete([self.first_episode], self.explore) or reported_score(self.first_episode) != 0):
            raise ValueError('Repair requires the verified first failure against the exact pre-item family bank')
        if not self.synthetic:
            original = self.evaluator.measured_episode(self.initial, self.logs, task, self.first_episode['phase'],
                                                        memory_snapshot=self.parent_snapshot)
            if digest(original) != digest(self.first_episode):
                raise MeasurementFailure('Submitted first answer differs from its native evidence')

    def _memory_context(self):
        context = super()._memory_context()
        context.update(task_memory_scope='current family only', evolution_memory_scope='current dataset',
                       history_boundary='Task experiences are restricted to this family. The supplied mechanism-change records may come from other families of this dataset.')
        return context

    def _native_outcome(self, ep):
        """Only the native adapter's verified public status/counters enter role inputs."""
        outcome = ep.get('native_outcome')
        if outcome is None:
            if self.synthetic or ep.get('error'):
                return None
            raise MeasurementFailure('Missing verified online native outcome')
        if ep.get('environment') == 'alfworld':
            keys = required = ('status', 'score', 'completed_steps', 'action_attempts', 'step_limit', 'won')
            if (not isinstance(outcome, dict) or set(outcome) != set(keys)
                    or not all(type(outcome[k]) is int for k in ('completed_steps','action_attempts','step_limit'))
                    or not 0 <= outcome['completed_steps'] <= outcome['action_attempts'] <= outcome['step_limit']
                    or outcome['step_limit'] != self.environment_step_limit
                    or outcome['completed_steps'] != len(ep['history'])
                    or outcome['won'] is not (outcome['score'] == 1)
                    or outcome['status'] != ('solved' if outcome['won'] else 'unsolved')):
                raise MeasurementFailure('Malformed native ALFWorld outcome')
        else:
            required = ('finish', 'completion_tokens', 'max_tokens', 'status', 'score')
            keys = (*required, 'tests_total', 'tests_run', 'tests_passed', 'first_failed_index')
        if (not isinstance(outcome, dict) or not set(required) <= outcome.keys()
                or set(outcome) - set(keys) or outcome['score'] != reported_score(ep)):
            raise MeasurementFailure('Malformed native outcome for online repair')
        observation = {k:outcome[k] for k in keys if k in outcome}
        if ep.get('environment') == 'alfworld':
            observation.update(_alfworld_action_facts(ep))
        return dict(task_id=ep['task_id'], program_hash=ep['program_hash'],
                    pre_item_memory_hash=ep['memory_hash_before'], phase=ep['phase'],
                    measurement_hash=digest(ep), observation=observation)

    def _role_usage(self, totals, role, stem, index, seed):
        """Charge one completed role invocation at its immutable ledger offset."""
        tally = totals.setdefault('evolver' if stem == 'proposal' else 'reviewer', dict(
            ledger_requests=0, prompt_tokens=0, completion_tokens=0, missing_usage=0, unmetered_calls=0))
        ledger = getattr(role, 'ledger', None)
        if ledger is None:
            # Scripted engineering roles may have no physical model transport.
            tally['unmetered_calls'] += 1
            return
        directory = self.store.root/'trials'/f'{index:03d}'
        intent = read(directory/(stem+'_intent.json'))
        entry = read_role_response(ledger, intent['ledger_start'], stem, seed)
        tally['ledger_requests'] += 1
        usage = entry.get('usage') or {}
        if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
            for key in ('prompt_tokens', 'completion_tokens'):
                tally[key] += usage[key]
        else:
            tally['missing_usage'] += 1

    def _item_progress(self, trial, outcome, episodes):
        """Roll forward committed item evidence. The reviewer sees a bounded window, not a round limit."""
        index = trial['round']
        previous = {}
        recent = []
        previous_edit = None
        if index:
            directory = self.store.root/'trials'/f'{index-1:03d}'
            prior_trial = read(directory/'trial.json')
            prior = prior_trial['post_review']
            prior_input = read(self.store.root/prior['input_path'])
            if (prior_trial['round'] != index-1 or prior_input['round'] != index-1
                    or digest(prior_input) != prior['payload_hash']
                    or prior['report']['assessment']['decision'] != 'continue'):
                raise MeasurementFailure('Missing preceding same-item continuation evidence')
            previous = prior_input['context']['item_progress']
            review = prior['report']['assessment']
            recent = [*previous['recent_completed_attempts'], dict(
                previous['current_attempt'], decision=review['decision'], reason=review['reason'],
                next_check=review['next_check'], report_hash=prior['report_hash'])][-4:]
            change = prior_trial['evolution_event']
            previous_edit = dict(attempt=index, parent_hash=prior_trial['parent_hash'],
                                 draft_hash=prior_trial.get('raw_source_hash'), code_diff=change['code_diff'],
                                 omitted_diff_lines=change['omissions']['code_diff_lines'])

        # Count actual role receipts already available to this decision. The
        # current post call itself can be charged only after its response.
        role_usage = previous.get('role_usage_before_this_post', {})
        if index == 0:
            self._role_usage(role_usage, self.reviewer, 'pre_review', 0, self.seed + 20000)
        else:
            self._role_usage(role_usage, self.reviewer, 'post_review', index-1, self.seed + 20000 + 2*(index-1) + 1)
        self._role_usage(role_usage, self.proposer, 'proposal', index, self.seed + index)

        comparison = trial.get('exploration') or {}
        score = comparison.get('candidate')
        observation = outcome['observation'] if outcome else {}
        error = trial.get('error') or (episodes[0].get('error') if episodes else None)
        signals = {key: observation[key] for key in (
            'finish', 'status', 'tests_run', 'tests_total', 'tests_passed', 'first_failed_index',
            'completed_steps', 'action_attempts', 'listed_action_steps',
            'nothing_happens_steps', 'action_sequence_hash') if key in observation}
        signals.update(score=score, validation_error_hash=digest(error) if error else None)
        prev_attempt = previous.get('current_attempt', {})
        streak = previous.get('consecutive_matching_outcomes', 0) if signals == prev_attempt.get('outcome') else 0
        item_costs = getattr(self.evaluator, 'item_model_costs', None)
        return dict(current_attempt=dict(attempt=index+1, draft_hash=trial.get('raw_source_hash'),
                                         validated=trial['mechanism_validated'], outcome=signals,
                                         duplicate_of=trial.get('duplicate_of')),
                    proposals_submitted=index+1, recent_completed_attempts=recent, previous_edit=previous_edit,
                    duplicate_submissions=previous.get('duplicate_submissions', 0) + int('duplicate_of' in trial),
                    earlier_attempts_omitted=index-len(recent),
                    completed_candidate_native_comparisons=previous.get('completed_candidate_native_comparisons', 0)
                                                           + int(comparison.get('complete') is True),
                    candidates_native_correct=previous.get('candidates_native_correct', 0) + int(score == 1),
                    consecutive_matching_outcomes=streak+1, role_usage_before_this_post=role_usage,
                    execution_model_usage=item_costs(self.explore[0]) if item_costs else None,
                    scope='Recent entries are historical candidates and their Reviewer opinions. All drafts and '
                          'evidence remain in the item trial receipts. Matching observed outcomes alone do not establish '
                          'identical causes. Native comparisons count logical evaluations; cached Actor answers may recur. '
                          'Execution model totals include the first Actor call and optional memory-model calls for '
                          'this item through this decision. Cache hits add no inference tokens. Unknown usage is '
                          'counted separately. A null execution total means this evaluator has no usage ledger.')

    def _review_input(self, stage, parent, before, base_episodes, trial=None, source=None,
                      after=None, candidate_episodes=()):
        result = super()._review_input(stage, parent, before, base_episodes, trial, source, after, candidate_episodes)
        result['review_policy'] = REVIEWER_DIRECTED
        result['context']['online_native'] = dict(parent=self._native_outcome(base_episodes[0]),
            candidate=self._native_outcome(candidate_episodes[0]) if candidate_episodes else None)
        if stage == 'post':
            result['context']['item_progress'] = self._item_progress(trial, result['context']['online_native']['candidate'],
                                                                     candidate_episodes)
            ledger = getattr(self.proposer, 'ledger', None)
            if source is None and ledger is not None:
                # Even a truncated or non-JSON reply is a paid attempt. Keep
                # the actual model output as item-local evidence, never as code.
                index = trial['round']
                intent = read(self.store.root/'trials'/f'{index:03d}'/'proposal_intent.json')
                entry = read_role_response(ledger, intent['ledger_start'], 'proposal', self.seed + index)
                choice = entry['response']['choices'][0]
                result['context']['invalid_proposal_response'] = dict(
                    content=choice['message']['content'], finish_reason=choice['finish_reason'],
                    request_hash=entry['request_hash'],
                    scope='Complete recorded model reply with no usable source field. Untrusted text for this item only.')
        result['instruction_boundary'] = 'Only this arrived test-stream item and earlier own-family history. No reference solutions or future task results.'
        return result

    def _previous_attempt(self, history):
        previous = super()._previous_attempt(history)
        if previous is not None and 'duplicate_of' in history[-1]:
            previous['duplicate_of'] = history[-1]['duplicate_of']
        if previous is not None and previous['source'] is None:
            # The parent method has verified this post-review receipt and the
            # hashes of both the original and displayed evidence.
            post = history[-1]['post_review']
            response = read(self.store.root/post['displayed_path'])['context'].get('invalid_proposal_response')
            if response is not None:
                previous['invalid_proposal_response'] = response
        return previous

    def _measure(self, program, tasks, phase, memory_snapshot=None):
        if phase.endswith('/parent'):
            if tasks != self.explore or program.hash != self.initial.hash:
                raise MeasurementFailure('Parent comparison differs from the first attempt')
            return [self.first_episode]
        return super()._measure(program, tasks, phase, memory_snapshot=self._runtime_snapshots[program.hash])

    def _signature(self, program, initial_contexts):
        # An exact restore retains the actual parent state; a changed mechanism may
        # have a different state schema and replays precisely the same earlier history.
        key = digest(dict(binding=self.binding, program=program.hash, initial=initial_contexts))
        path = self.store.root/'diagnostic_signatures'/(key+'.json')
        intent = path.with_suffix('.started.json')
        task, = self.explore
        initial, = initial_contexts
        if path.exists():
            receipt = read(path)
            outcome = receipt['outcome']
            if read(intent) != {'key': key} or receipt['hash'] != digest([key, outcome]):
                raise MeasurementFailure('Online diagnostic receipt changed')
        else:
            if intent.exists():
                raise MeasurementFailure('Unresolved diagnostic execution for this item: '+str(intent))
            write(intent, {'key': key})
            try:
                outcome = dict(result=self._diagnostic_result(program, task, initial))
            except WorkerStartupError:
                raise
            except Rejected as exc:
                outcome = dict(rejection=dict(kind=type(exc).__name__, message=str(exc)))
            write(path, dict(outcome=outcome, hash=digest([key, outcome])))
        if 'rejection' in outcome:
            raise Rejected(outcome['rejection']['message'])
        result = outcome['result']
        try:
            CodeMemory.from_snapshot(program, result['runtime_snapshot'], self.tokenizer,
                                     context=self.context, capabilities=self.capabilities)
        except Rejected as exc:
            raise MeasurementFailure('Host diagnostic snapshot failed verification: '+str(exc)) from exc
        self._runtime_snapshots[program.hash] = result['runtime_snapshot']
        return result

    def _diagnostic_result(self, program, task, initial):
        same_program = program.hash == self.initial.hash
        bank = (CodeMemory.from_snapshot(program, self.parent_snapshot, self.tokenizer, context=self.context,
                                         capabilities=self.capabilities, diagnostics=True)
                if same_program else CodeMemory(program, self.logs, self.tokenizer, self.context,
                                                capabilities=self.capabilities, diagnostics=True))
        rows = bank.for_task(task, task['environment']).read({}, visible=visible_context(initial))
        slot = {'entries': rows} if rows else {}
        wire = json.dumps(slot, ensure_ascii=False, indent=2)
        result = dict(context_hashes=[digest(initial)], input_hashes=[digest(wire)], readouts=[slot],
                      memory_hash=bank.fingerprint, stored_state=bank.state, stored_tokens=bank.stats['stored_tokens'],
                      write_trace=bank.write_trace, read_trace=bank.read_trace, runtime_snapshot=bank.export_snapshot(),
                      storage_origin=('actual_pre_item_snapshot' if same_program else 'replayed_pre_item_family_history'),
                      write_trace_scope='This signature invocation only')
        result['diagnostics'] = diagnose(program, result, self.logs)
        return result

    def _role_intent(self, directory, stem, payload):
        # A crash after dispatch may already have billed a call. Do not turn it
        # into a free second call, even if the role receipt was never published.
        receipt = directory/(stem+'.json')
        intent = directory/(stem+'_intent.json')
        identity = dict(payload=digest(payload), proposal=self._round_index)
        if intent.exists():
            record = read(intent)
            if record.get('identity') != identity or not receipt.exists():
                raise MeasurementFailure('Unresolved or mismatched online role attempt: '+str(intent))
        else:
            if receipt.exists():
                raise MeasurementFailure('Unregistered online role receipt: '+str(receipt))
            ledger = getattr(self.proposer if stem == 'proposal' else self.reviewer, 'ledger', None)
            offset = Path(ledger).stat().st_size if ledger is not None and Path(ledger).exists() else 0
            record = dict(identity=identity, ledger_start=offset)
            write(intent, record)
        return record

    def _proposal(self, payload, round_index, directory):
        payload['online_native'] = {'first_attempt': self._native_outcome(self.first_episode)}
        if round_index:
            prior_dir = self.store.root/'trials'/f'{round_index-1:03d}'
            prior = read(prior_dir/'post_review_input.json')
            payload['online_native']['previous_comparison'] = prior['context']['online_native']
        intent = self._role_intent(directory, 'proposal', payload)
        try:
            return super()._proposal(payload, round_index, directory)
        except (ProposalCallFailure, Rejected) as exc:
            ledger = getattr(self.proposer, 'ledger', None)
            if self.synthetic and ledger is None and isinstance(exc, Rejected):
                raise  # The scripted author returned a value with an invalid envelope.
            entry = read_role_response(ledger, intent['ledger_start'], 'proposal', self.seed + round_index)
            choice = entry['response']['choices'][0]
            if choice['finish_reason'] == 'stop':
                try:
                    submitted = json.loads(choice['message']['content'])
                except ValueError:
                    if isinstance(exc, Rejected):
                        raise MeasurementFailure('Parsed proposal receipt differs from the model reply') from exc
                else:
                    if (isinstance(exc, ProposalCallFailure)
                            or digest(submitted) != digest(read(directory/'proposal.json')['value'])):
                        raise MeasurementFailure('Proposal failure is not explained by the model reply') from exc
            elif isinstance(exc, Rejected):
                raise MeasurementFailure('Parsed proposal receipt conflicts with a truncated model reply') from exc
            raise Rejected(str(exc)) from exc

    def _review(self, payload, index, directory):
        self._role_intent(directory, payload['stage']+'_review', payload)
        result = super()._review(payload, index, directory)
        if not result['ok']:
            raise MeasurementFailure('Online Reviewer did not produce a valid judgment: '+str(result['error']))
        return result

    def _cost_receipts(self, directory):
        paths = dict(proposal=directory/'proposal.json', measurements=self.store.root/'measurement_access.jsonl',
                     evolver=getattr(self.proposer, 'ledger', None), reviewer=getattr(self.reviewer, 'ledger', None),
                     actor_requests=getattr(self.evaluator, 'cache', None),
                     capability_requests=getattr(self.capabilities, 'ledger', None))
        if getattr(self.evaluator, 'root', None):
            paths['actor_references'] = Path(self.evaluator.root)/'episodes'
        return {key: os.path.relpath(path, self.store.root) for key, path in paths.items()
                if path is not None and Path(path).exists()}


def read_role_response(path, offset, stem, seed):
    """Attribute a paid attempt only to the registered request with an actual completion."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(offset)
            entry = json.loads(stream.readline())
        request = entry['request']
        choice = entry['response']['choices'][0]
        role = 'code_evolver' if stem == 'proposal' else 'mechanism_reviewer'
        if (entry['role'] != role or entry['seed'] != seed or request['seed'] != seed
                or (stem != 'proposal' and entry['stage'] != stem.removesuffix('_review'))
                or entry['request_hash'] != digest(request) or choice['finish_reason'] not in ('stop', 'length')
                or not isinstance(choice['message']['content'], str)):
            raise ValueError('Mismatched request or missing completion')
    except (OSError, TypeError, ValueError, IndexError, KeyError) as exc:
        raise MeasurementFailure('No verified model response for '+stem) from exc
    return entry


def repair_item(*, out, tokenizer, proposer, reviewer, evaluator, task, parent, logs, parent_snapshot,
                first_episode, records, seed=11, capabilities=None, synthetic=False):
    engine = ItemSearch(out=out, tokenizer=tokenizer, proposer=proposer, reviewer=reviewer, evaluator=evaluator,
                        task=task, parent=parent, logs=logs, parent_snapshot=parent_snapshot,
                        first_episode=first_episode, records=records, seed=seed,
                        capabilities=capabilities, synthetic=synthetic)
    state = engine.run()
    trials = engine.store.history(state)
    if not trials or any(t.get('review_error') for t in trials):
        raise MeasurementFailure('Unresolved online review; no completed item decision')
    measurements = {}
    accesses = engine.store.root/'measurement_access.jsonl'
    if accesses.is_file():
        with accesses.open() as stream:
            for line in stream:
                access = json.loads(line)
                if measurements.setdefault(access['phase'], access['key']) != access['key']:
                    raise MeasurementFailure('Conflicting measurement receipts for one online attempt')
    attempts, events = [], []
    private_names = SharedCodeView.private_identifiers(private_learning_texts([*logs, first_episode]))
    source, snapshot, episode = parent.source, parent_snapshot, first_episode
    for trial in trials:
        index = trial['round']
        directory = engine.store.root/'trials'/f'{index:03d}'
        comparison = trial.get('exploration') or {}
        native = None
        if comparison:
            phase = f'round-{index:03d}/exploration/candidate'
            key = measurements.get(phase)
            if key is None or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                raise MeasurementFailure('Missing bound online candidate episode')
            measurement = read(engine.store.root/'measurements'/(key+'.json'))
            if (measurement['key'] != key or measurement['phase'] != phase
                    or measurement['episodes_hash'] != comparison['candidate_evidence']
                    or digest(measurement['episodes']) != measurement['episodes_hash']):
                raise MeasurementFailure('Candidate evidence differs from its native episode')
            native, = measurement['episodes']
        assessment = trial['post_review']['report']['assessment']
        decision = assessment['decision']
        if ((decision == 'continue') != (index < len(trials)-1)
                or decision not in {'continue', 'adopt', 'stop'} or (decision == 'adopt') != trial['accepted']):
            raise MeasurementFailure('Item history differs from its recorded Reviewer decisions')
        attempts.append(dict(attempt=index+1, accepted=trial['accepted'], candidate_score=comparison.get('candidate'),
                             decision=decision, reason=assessment['reason'], status=trial['public_status'],
                             trial_hash=digest(trial)))
        # A draft can refer to any earlier attempt of this item, including a
        # rejected one. Only the structural projection leaves the item.
        if native is not None:
            private_names.update(SharedCodeView.private_identifiers(private_learning_texts([native])))
        behavior = read(directory/'behavior.json') if trial.get('behavior_path') else {}
        candidate_source = None
        if trial['raw_source_hash'] is not None:
            proposal = read(directory/'proposal.json').get('value')
            candidate_source = proposal.get('source') if isinstance(proposal, dict) else None
            if (not isinstance(candidate_source, str)
                    or hashlib.sha256(candidate_source.encode()).hexdigest() != trial['raw_source_hash']):
                raise MeasurementFailure('Archived draft differs from the code actually submitted for assessment')
        facts = {k:v for k,v in trial.items() if k != 'evolution_event'}
        event = public_event(facts, parent.source, candidate_source, tokenizer=tokenizer,
                             behavior=behavior.get('candidate'), parent_behavior=behavior.get('parent'),
                             shared_code=SharedCodeView(private_names))
        events.append(event)
        if trial['accepted']:
            if native is None or reported_score(native) != 1:
                raise MeasurementFailure('Adopted repair lacks verified same-item native success')
            source = engine.store.program(trial['candidate_hash']).source
            snapshot = behavior['candidate']['runtime_snapshot']
            episode = native
            if not synthetic:
                # Comparison receipts canonicalize object keys. The online native
                # receipt also binds their original order, required by Hook commit.
                bound_episode = evaluator.measured_episode(engine.store.program(trial['candidate_hash']), logs,
                                                           task, native['phase'], memory_snapshot=snapshot)
                if digest(bound_episode) != digest(native):
                    raise MeasurementFailure('Native readback differs from accepted candidate evidence')
                episode = bound_episode
    return dict(source=source, snapshot=snapshot, episode=episode, events=events, attempts=attempts,
                accepted=trials[-1]['accepted'], solved=any(a['candidate_score'] == 1 for a in attempts))
