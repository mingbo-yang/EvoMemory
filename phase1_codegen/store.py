"""Immutable programs and trial evidence; one atomic authoritative current pointer."""
import contextlib
import fcntl
import json
from pathlib import Path
from evomem.common import digest, read, write
from .program import Program
from .outcomes import comparison_for_feedback
from .acceptance import POLICIES, DUAL_POLICIES, REVIEWER_DIRECTED, eligible_for_adoption, adoption_feedback

class RunStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def lock(self):
        with (self.root / '.evolution.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another evolution process owns this run') from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def save_program(self, program):
        path = self.root / 'programs' / (program.hash + '.py')
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            if path.read_text() != program.source:
                raise RuntimeError('Immutable program hash collision or drift')
        else:
            with path.open('x') as stream:
                stream.write(program.source)
        return program.hash

    def program(self, key):
        program = Program((self.root / 'programs' / (key + '.py')).read_text())
        if program.hash != key:
            raise RuntimeError('Stored program content changed')
        return program

    def initialize(self, initial, binding):
        self.save_program(initial)
        path = self.root / 'state.json'
        if path.exists():
            state = read(path)
            if state['binding_hash'] != digest(binding) or state['initial_program'] != initial.hash:
                raise RuntimeError('Run inputs, runtime or initial program changed; use a new run')
            self.validate_state(state)
            return state
        state = dict(schema=1, revision=0, current_program=initial.hash, initial_program=initial.hash,
                     next_round=0, history=[], binding_hash=digest(binding), binding=binding)
        if binding.get('task_memory_mode') == 'continual':
            state['task_memory'] = binding['initial_task_memory_snapshot']
        write(path, state)
        return state

    def commit(self, expected, trial, trial_path):
        """Evidence exists first; advancing this file is the sole deployment commit."""
        actual = read(self.root / 'state.json')
        if actual != expected:
            raise RuntimeError('Concurrent/stale current mechanism update')
        if trial['round'] != actual['next_round'] or trial['parent_hash'] != actual['current_program']:
            raise RuntimeError('Trial is not based on the authoritative current mechanism')
        if read(trial_path) != trial:
            raise RuntimeError('Persist complete trial evidence before committing')
        previous = self.validate_state(actual)
        self.check_review_handoff(actual['binding'], trial, previous)
        self.validate_trial(actual['binding'], trial)
        if actual['binding'].get('task_memory_mode') == 'continual':
            from .learning import validate_transition
            validate_transition(trial, actual['binding'], actual['task_memory'], root=self.root)
        candidate = trial.get('candidate_hash')
        if trial['accepted']:
            self.program(candidate)
        updated = dict(actual, next_round=actual['next_round'] + 1,
                       revision=actual['revision'] + int(trial['accepted']),
                       current_program=candidate if trial['accepted'] else actual['current_program'],
                       history=actual['history'] + [dict(path=str(Path(trial_path).relative_to(self.root)),
                                                        hash=digest(trial))])
        if actual['binding'].get('task_memory_mode') == 'continual':
            updated['task_memory'] = trial['task_memory_after']
        write(self.root / 'state.json', updated)
        return updated

    def validate_trial(self, binding, trial):
        if type(trial.get('accepted')) is not bool or trial.get('synthetic') is not binding['synthetic']:
            raise RuntimeError('Invalid acceptance value or synthetic/real boundary mismatch')
        index = trial['round']
        policy = binding.get('acceptance_policy')
        limit = binding['max_rounds']
        if (policy not in POLICIES or trial.get('acceptance_policy') != policy or type(index) is not int or index < 0
                or (limit is None and policy != REVIEWER_DIRECTED)
                or (limit is not None and (policy == REVIEWER_DIRECTED or index >= limit))):
            raise RuntimeError('Trial does not match its frozen policy and round budget')
        expected_block = index if policy == 'native_reward' else None
        if trial.get('confirmation_block') != expected_block:
            raise RuntimeError('Trial does not match its preallocated confirmation policy')
        if trial.get('acceptance_basis') != (policy if trial['accepted'] else None):
            raise RuntimeError('Adoption basis differs from the frozen policy')
        facts = self.adoption_context(binding, trial)
        if policy in DUAL_POLICIES:
            feedback = adoption_feedback(binding, trial, current_facts=facts)
            stored_feedback = trial.get('acceptance_feedback')
            # Older receipts can contain numeric placeholders copied from the same incomplete trial.
            exploration = trial.get('exploration')
            if (stored_feedback and isinstance(exploration, dict) and exploration.get('complete') is False
                    and stored_feedback.get('native_comparison') == exploration):
                stored_feedback = dict(stored_feedback, native_comparison=comparison_for_feedback(exploration))
            if stored_feedback != feedback or trial['accepted'] != feedback['eligible']:
                raise RuntimeError('Adoption or iteration feedback differs from verified dual conditions')
        if trial['accepted']:
            if not eligible_for_adoption(binding, trial, current_facts=facts):
                raise RuntimeError('Invalid adoption evidence for the frozen acceptance policy')
            behavior = read(self.root / trial['behavior_path'])
            if (behavior.get('synthetic') is not binding['synthetic']
                    or digest(dict(parent=behavior['parent'], candidate=behavior['candidate'])) != trial['behavior_evidence']):
                raise RuntimeError('Executable mechanism validation evidence changed')
            if trial.get('exploration') is None and behavior['parent']['input_hashes'] != behavior['candidate']['input_hashes']:
                raise RuntimeError('Changed Executor inputs require complete exploration evidence')
        if policy == REVIEWER_DIRECTED or (trial['accepted'] and binding.get('review_required')):
            self.validated_reviews(binding, trial)

    @staticmethod
    def check_review_handoff(binding, trial, previous):
        if binding.get('acceptance_policy') != REVIEWER_DIRECTED:
            return
        if previous is None:
            if (trial['round'] != 0 or (trial.get('pre_review') or {}).get('stage') != 'pre'):
                raise RuntimeError('The first edit requires the initial same-item diagnosis')
        elif (previous['acceptance_feedback']['decision'] != 'continue'
              or trial.get('pre_review') != previous['post_review']):
            raise RuntimeError('A subsequent edit requires the exact preceding continue decision')

    def validated_reviews(self, binding, trial):
        """One receipt verifier for adoption and experimental evidence extraction."""
        from .review import validate_report
        verified = {}
        if trial.get('review_error'):
            raise RuntimeError('Cannot deploy a trial with a failed mechanism review')
        for key in ('pre_review', 'post_review'):
            receipt = trial.get(key, {})
            if (receipt.get('ok') is not True or receipt.get('reviewer_fingerprint') != binding['reviewer']
                    or receipt.get('synthetic') is not binding['synthetic']
                    or digest(receipt.get('report')) != receipt.get('report_hash')):
                raise RuntimeError('Missing or invalid fixed reviewer receipt: ' + key)
            if read(self.root / receipt['receipt_path']) != receipt:
                raise RuntimeError('Persisted mechanism review receipt changed')
            shown = read(self.root / receipt['displayed_path'])
            original = read(self.root / receipt['input_path'])
            if binding.get('review_contract') == 2 and (shown.get('review_contract') != 2
                    or not receipt['report'].get('evidence_contract')):
                raise RuntimeError('New adoption requires the grounded Reviewer evidence contract')
            if digest(shown) != receipt['displayed_hash'] or digest(original) != receipt['payload_hash']:
                raise RuntimeError('Mechanism review evidence changed')
            if binding.get('acceptance_policy') == REVIEWER_DIRECTED:
                if (shown.get('review_policy') != REVIEWER_DIRECTED or original.get('review_policy') != REVIEWER_DIRECTED
                        or original['current_program_hash'] != trial['parent_hash']
                        or (key == 'post_review' and original['round'] != trial['round'])):
                    raise RuntimeError('Decision is not bound to the registered item and proposal')
            if key == 'post_review' and (receipt['stage'] != 'post'
                    or original['current_program_hash'] != trial['parent_hash']
                    or original['candidate_program_hash'] != trial.get('candidate_hash')):
                raise RuntimeError('Post review did not analyze this candidate and parent')
            if key == 'post_review' and binding.get('acceptance_policy') in DUAL_POLICIES:
                comparisons = [row['data'] for row in original['evidence'] if row['kind'] == 'paired_exploration']
                expected = [trial['exploration']] if trial.get('exploration') is not None else []
                if comparisons != expected:
                    raise RuntimeError('Native adoption comparison differs from post-review evidence')
            validate_report(receipt['report'], shown)
            verified[key] = (shown, original)
        return verified

    def adoption_context(self, binding, trial):
        """Return only receipt-verified current facts, never Reviewer arithmetic."""
        if binding.get('acceptance_policy') != 'reviewer_and_reward_or_equal_utility':
            return None
        exploration = trial.get('exploration') or {}
        review = trial.get('post_review') or {}
        if (trial.get('mechanism_validated') is not True or trial.get('error')
                or trial.get('review_error') or not trial.get('candidate_hash')
                or exploration.get('complete') is not True or review.get('ok') is not True):
            return None
        if binding.get('review_contract') != 2 or binding.get('review_required') is not True:
            raise RuntimeError('Experimental adoption requires the grounded fixed Reviewer')
        from .review import _computed_current_facts
        shown, original = self.validated_reviews(binding, trial)['post_review']
        rows = [row['data'] for row in shown['evidence'] if row['id'] == 'current:facts']
        facts = _computed_current_facts(original)
        if rows != [facts]:
            raise RuntimeError('Displayed current facts differ from original execution evidence')
        comparison = facts.get('comparison') or {}
        if (comparison.get('complete') is not True
                or [row.get('task_id') for row in comparison.get('tasks', [])] != exploration.get('task_ids')
                or any(comparison.get(source) != exploration.get(target) for source, target in
                       [('parent_mean', 'parent'), ('candidate_mean', 'candidate'), ('delta', 'delta')])):
            raise RuntimeError('Experimental adoption facts differ from the native trial comparison')
        return facts

    def validate_state(self, state):
        if digest(state['binding']) != state['binding_hash']:
            raise RuntimeError('Frozen run binding changed')
        current, revision = state['initial_program'], 0
        memory_ref = state['binding'].get('initial_task_memory_snapshot')
        if state['binding'].get('task_memory_mode') == 'continual':
            from .learning import load_snapshot
            initial_memory = load_snapshot(self.root, memory_ref)
            if initial_memory['program_hash'] != current:
                raise RuntimeError('Initial Task Memory and mechanism differ')
        self.program(current)
        history = self.history(state)
        previous = None
        for index, trial in enumerate(history):
            self.check_review_handoff(state['binding'], trial, previous)
            self.validate_trial(state['binding'], trial)
            if state['binding'].get('task_memory_mode') == 'continual':
                from .learning import validate_transition
                validate_transition(trial, state['binding'], memory_ref, root=self.root)
                memory_ref = trial['task_memory_after']
            if trial['round'] != index or trial['parent_hash'] != current:
                raise RuntimeError('Committed program ancestry changed')
            if trial['accepted']:
                current = trial['candidate_hash']
                self.program(current)
                revision += 1
            previous = trial
        if state['current_program'] != current or state['revision'] != revision or state['next_round'] != len(history):
            raise RuntimeError('Current mechanism pointer disagrees with committed trial history')
        if state['binding'].get('task_memory_mode') == 'continual' and state.get('task_memory') != memory_ref:
            raise RuntimeError('Committed Task Memory pointer disagrees with the learning history')
        return previous

    def current(self):
        """Executor consumers resolve the authoritative, evidence-checked program."""
        state = read(self.root / 'state.json')
        self.validate_state(state)
        return self.program(state['current_program'])

    def history(self, state):
        rows = []
        for receipt in state['history']:
            value = read(self.root / receipt['path'])
            if digest(value) != receipt['hash']:
                raise RuntimeError('Committed trial evidence changed')
            rows.append(value)
        return rows
