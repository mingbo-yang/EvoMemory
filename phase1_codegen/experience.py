"""Fixed, bounded mechanism-change evidence and problem-matched retrieval.

Raw trials remain local. Transfer contains code fragments and typed observations,
never serialized Task Memory or trajectories. Opinions are not execution facts.
"""
import ast
from collections import Counter
import copy
import difflib
import hashlib
import io
import json
from itertools import zip_longest
import keyword
import re
import tokenize
from evomem.common import digest, dumps
from evomem.memory import rank
from phase1_codegen.outcomes import comparison_for_feedback
from phase1_codegen.policy import SAFE_CALLS, SAFE_METHODS, EXCEPTION_NAMES
from phase1_codegen.protocol import EVENTS, HOOK_BUDGET_COUNTERS, WRITE_FIELDS

EXPERIENCE_VERSION = 2
RECORD_TOKENS = 1400
RETRIEVAL_TOKENS = 3072
RETRIEVAL_LIMIT = 4
DIMENSIONS = {'Faithfulness', 'Relevance', 'Actionability', 'Applicability', 'Efficiency'}
FACT_KINDS = ('missing_dict_keys', 'branch_counts', 'empty_returns', 'sorting')
ABI_FIELDS = {'state', 'event', 'context', 'visible', 'goal', 'obs', 'experience',
              'history', 'score', 'family', 'environment', 'session', 'type',
              'memory', 'task_id', 'action', 'next_observation', 'budget_remaining'}


def _count(tokenizer, value):
    text = dumps(value)
    return tokenizer.count(text) if tokenizer is not None else len(text.encode())


def _error(error, private_texts=()):
    if not error:
        return None
    text = str(error)
    symbols = re.findall(r'(?:unknown call|forbidden syntax):\s*([A-Za-z][A-Za-z0-9_]*)', text)
    path = re.search(r'path=([/A-Za-z0-9_-]+)', text)
    point = re.search(r'line (\d+), column (\d+)', text)
    value = dict(kind=text.split(':', 1)[0][:64], symbols=list(dict.fromkeys(symbols)),
                field_path=path.group(1) if path else None,
                line=int(point.group(1)) if point else None, original_hash=digest(text))
    # Worker envelopes are transport, not the contract violation. Preserve the
    # expected and actual *field names*, never arbitrary runtime values/Task Memory.
    message = text
    start = text.find('{')
    if start >= 0:
        try:
            envelope, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(envelope, dict) and isinstance(envelope.get('message'), str):
                message = envelope['message']
        except (ValueError, TypeError):
            pass
    expected = re.search(r'requires exactly ([A-Za-z_, ]+?)(?:;|$)', message)
    actual = re.search(r'actual=(\[[^\]\n]*\])', message)
    if expected:
        value['expected'] = expected.group(1)
    else:
        constraint = message.removeprefix('Rejected: ').split(';', 1)[0]
        if constraint.startswith(('event output ', 'event session ', 'event memory ')):
            value['constraint'] = constraint
    if actual:
        try:
            keys = ast.literal_eval(actual.group(1))
            if isinstance(keys, list) and all(isinstance(key, str) for key in keys):
                value['actual_fields'] = [key if key not in private_texts and len(key) <= 256 else
                    '[private field sha256:' + digest(key)[:16] + ']' for key in keys]
        except (ValueError, SyntaxError):
            pass
    actual_type = re.search(r"actual='?(dict|list|tuple|str|int|float|bool|NoneType)'?(?:$|[;\s])", message)
    actual_length = re.search(r'actual=list length (\d+)', message)
    if actual_type:
        value['actual_type'] = actual_type.group(1)
    if actual_length:
        value['actual_length'] = int(actual_length.group(1))
    return value


def _expression(text):
    try:
        tree = ast.parse(str(text), mode='eval')
    except (SyntaxError, ValueError):
        return ''
    parts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in ABI_FIELDS:
                parts.append(node.value)
    return ' '.join(parts)


def execution_problems(behavior):
    facts = ((behavior or {}).get('diagnostics') or {}).get('facts', {})
    output = []
    for kind in FACT_KINDS:
        for row in facts.get(kind, []):
            value = dict(kind=kind, function=row.get('function'), line=row.get('line'),
                         operation=row.get('operation'), unit='site_executions',
                         terms=_expression(row.get('expression', '')))
            for key in ('calls', 'missing_count', 'true_count', 'false_count'):
                if type(row.get(key)) in (int, float):
                    value[key] = row[key]
            output.append(value)
    return output


def _prioritize_problems(parent, candidate, before, after):
    """Order observations near edits and differing observations first on each side.

    Comparison is for evidence selection only. A relocated site can have the same
    recorded counts. A row without a counterpart does not establish a defect.
    """
    edits = {'parent': set(), 'candidate': set()}
    for tag, i, j, k, l in difflib.SequenceMatcher(
            None, before.splitlines(), (after or '').splitlines()).get_opcodes():
        if tag != 'equal':
            edits['parent'].update(range(i + 1, j + 1))
            edits['candidate'].update(range(k + 1, l + 1))
    def signature(row):
        return tuple((k, v) for k, v in row.items() if k != 'line')

    def prioritize(rows, other, side):
        matched = {signature(row) for row in other}
        def priority(row):
            return (signature(row) in matched, row['line'] not in edits[side])
        return [dict(subject=side, **row) for row in sorted(rows, key=priority)]

    # Interleave so that a long list of unchanged observations on either side
    # cannot consume the entire comparative evidence allowance.
    ordered = zip_longest(prioritize(candidate, parent, 'candidate'),
                          prioritize(parent, candidate, 'parent'))
    return [row for pair in ordered for row in pair if row is not None]


def _opinions(trial):
    report = ((trial.get('post_review') or {}).get('report') or {})
    contract = report.get('evidence_contract') or {}
    subject = (report.get('assessment') or {}).get('subject')
    opinions = [dict(dimension=item['dimension'], polarity=item.get('polarity'),
                     **({'assessment_subject': subject} if subject in ('current', 'candidate') else {}),
                     uncertainty=item.get('uncertainty', []))
                for item in contract.get('evaluations', []) if item.get('dimension') in DIMENSIONS]
    assessments = [dict(group_id=item.get('group_id'), judgment=item.get('judgment'),
                        subject=item.get('subject'), kind=item.get('kind'),
                        function=item.get('function'), line=item.get('line'),
                        terms=_expression(item.get('expression', '')),
                        uncertainty=item.get('uncertainty', []))
                   for item in contract.get('diagnostic_assessments', [])]
    return opinions, assessments


def _structure(source):
    try:
        return sorted({node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                            for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
                            and isinstance(node.func, (ast.Name, ast.Attribute))})
    except SyntaxError:
        return []


def mechanism_query(context, source, behavior=None, review=None, previous=None):
    opinions, assessments = _opinions(dict(post_review=review))
    historical = {} if not previous else dict(error=previous.get('error'),
        problems=previous.get('problem_features', []), review=previous.get('review_opinions', []))
    return dict(environment=context.get('environment'),
                current_parent=dict(structure=_structure(source), problems=execution_problems(behavior),
                                    review=opinions, diagnostic_assessments=assessments),
                historical_candidate=historical)


def _matching(value, scope=''):
    """Match problem semantics with parent/candidate roles intact.

    The role describes the observed program, not source/local pool membership.
    Neither acceptance nor task family contributes ranking terms.
    """
    if isinstance(value, list):
        return ' '.join(_matching(item, scope) for item in value)
    if not isinstance(value, dict):
        words = re.findall(r'[A-Za-z0-9_]+', value) if isinstance(value, str) else []
        return ' '.join((scope + '__' if scope else '') + word for word in words)
    # Review dimensions are common criteria across a current diagnosis and an
    # archived comparison. Bind polarity to the same dimension; preserve the
    # original subject when observed. Old opinions without a subject match on
    # the criterion only, never on an invented subject.
    if value.get('dimension') in DIMENSIONS and isinstance(value.get('polarity'), str):
        criterion = 'review__' + value['dimension']
        verdict = criterion + '__' + value['polarity']
        subject = value.get('assessment_subject')
        return ' '.join([criterion, verdict] + (
            [subject + '__' + verdict] if subject in ('current', 'candidate') else []))
    scope = value.get('subject') if value.get('subject') in ('parent', 'candidate') else scope
    keys = ('kind', 'symbols', 'field_path', 'terms', 'structure', 'dimension', 'polarity',
            'judgment', 'problems', 'review', 'error', 'current_parent', 'historical_candidate',
            'problem_features', 'review_opinions', 'diagnostic_assessments', 'expected', 'actual_fields',
            'constraint', 'actual_type')
    scopes = {'current_parent': 'parent', 'historical_candidate': 'candidate', 'error': 'candidate',
              'structure': ''}
    return ' '.join(_matching(value[key], scopes.get(key, scope)) for key in keys if key in value)


class SharedCodeView:
    """An online record-local structural view of a draft, never executable source."""
    protocol = ABI_FIELDS | set(EVENTS + HOOK_BUDGET_COUNTERS + WRITE_FIELDS) | {
        'id', 'step', 'attempt', 'entries', 'observation', 'history_import', 'mode', 'origin'}
    language = protocol | SAFE_CALLS | SAFE_METHODS | EXCEPTION_NAMES | {'on_event'}

    def __init__(self, private_names):
        self.private_names = private_names
        self.aliases = {}

    @staticmethod
    def private_identifiers(texts):
        return {name for text in texts if isinstance(text, str)
                for name in re.findall(r'[^\W\d]\w*', text)}

    def alias(self, value, kind):
        key = (kind, type(value), value)
        return self.aliases.setdefault(key, f'{kind}_{len(self.aliases) + 1}')

    def name(self, name):
        if name in self.private_names and name not in self.language and not keyword.iskeyword(name):
            return self.alias(name, 'identifier')
        return name

    def field(self, value):
        return value if value in self.protocol else self.alias(value, 'field')

    def source(self, source):
        """Keep original line coordinates and syntax, dropping prose and value literals."""
        try:
            tree = ast.parse(source)
            if any(isinstance(node, ast.JoinedStr) for node in ast.walk(tree)):
                # Interpolation is outside the Hook language. Body tokenization
                # differs between Python versions and may include private text.
                return '[interpolated draft omitted; see raw_source_hash]'
            reader = io.StringIO(source, newline=None)
            lines = reader.readlines()
            reader.seek(0)
            edits = []
            for token in tokenize.generate_tokens(reader.readline):
                if token.type == tokenize.COMMENT:
                    replacement = ''
                elif token.type == tokenize.STRING:
                    literal = ast.literal_eval(token.string)
                    if literal == '' or literal in self.protocol:
                        # Normalize escapes and quoting; the original spelling can contain prose.
                        replacement = repr(literal)
                    else:
                        replacement = repr(self.alias(literal, 'field'))
                elif token.type == tokenize.NUMBER:
                    replacement = self.alias(ast.literal_eval(token.string), 'number')
                elif token.type == tokenize.NAME:
                    replacement = self.name(token.string)
                else:
                    continue
                edits.append((token.start, token.end, replacement))
        except (SyntaxError, ValueError, tokenize.TokenError):
            return '[unparsed source; see raw_source_hash]'
        for (start, left), (end, right), replacement in reversed(edits):
            if start == end:
                lines[start-1] = lines[start-1][:left] + replacement + lines[start-1][right:]
            else:
                lines[start-1] = lines[start-1][:left] + replacement + '\n'
                for index in range(start, end-1):
                    lines[index] = '\n'
                lines[end-1] = lines[end-1][right:]
        return ''.join(lines)


def _code_lines(parent, candidate, private_texts, shared=None):
    private = {text for text in private_texts if isinstance(text, str) and len(text) >= 8}
    def redact(source):
        if source is None:
            return ''
        if shared is not None:
            return shared.source(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return '[unparsed source; see raw_source_hash]'
        raw = source.encode()
        starts, offset = [], 0
        for line in raw.splitlines(True):
            starts.append(offset)
            offset += len(line)
        edits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in private or len(node.value) > 256:
                    edits.append((starts[node.lineno - 1] + node.col_offset,
                                  starts[node.end_lineno - 1] + node.end_col_offset,
                                  repr('[private literal sha256:' + digest(node.value)[:16] + ']').encode()
                                  + b'\n' * (node.end_lineno - node.lineno)))
        for begin, end, replacement in sorted(edits, reverse=True):
            raw = raw[:begin] + replacement + raw[end:]
        return raw.decode()
    return list(difflib.unified_diff(redact(parent).splitlines(), redact(candidate).splitlines(),
                                   fromfile='parent.py', tofile='candidate.py', lineterm=''))


def _fragments(lines, points):
    """Prioritize exact parent/candidate locations, not the start of a large hunk.

    Each returned fragment has its own accurate unified-diff range. Omitted
    lines are never joined as if they were adjacent source lines.
    """
    rows, old, new = [], 0, 0
    for line in lines:
        if line.startswith('@@'):
            match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)', line)
            old, new = map(int, match.groups())
        elif not line.startswith(('---', '+++')):
            kind = line[:1]
            rows.append((line, old, new, kind != '+', kind != '-'))
            old += kind != '+'
            new += kind != '-'
    centers = []
    for subject, point in points:
        for index, row in enumerate(rows):
            if (subject == 'parent' and row[3] and row[1] == point
                    or subject == 'candidate' and row[4] and row[2] == point):
                centers.append(index)
    used, fragments = set(), []
    def add(begin, end):
        # Preserve disjoint ranges explicitly; no duplicate code consumes tokens.
        current = []
        for index in range(begin, end):
            if index in used:
                if current:
                    fragments.append(current)
                    current = []
            else:
                current.append(rows[index])
                used.add(index)
        if current:
            fragments.append(current)
    # Diagnostic sites come first, then candidate additions, then deletions.
    # A large deleted parent must not hide a two-line replacement implementation.
    exact_centers = list(dict.fromkeys(centers))
    additions = [index for index, row in enumerate(rows) if row[0].startswith('+')]
    if not any(rows[index][0].startswith('+') for index in exact_centers):
        # Keep a real change individually addressable even without a diagnostic
        # on an added line. Its unchanged context can then use remaining space.
        if additions:
            exact_centers.append(additions[0])
        elif rows:
            deletions = (i for i, row in enumerate(rows) if row[0].startswith('-'))
            first_deletion = next(deletions, None)
            if first_deletion is not None:
                exact_centers.append(first_deletion)
    centers += additions
    # Exact sites form the minimum semantic record. Surrounding context remains
    # optional, so an oversized context window cannot suppress a small change.
    for center in exact_centers:
        add(center, center + 1)
    for center in centers:
        add(max(0, center - 2), min(len(rows), center + 3))
    for start in range(0, len(rows), 3):
        add(start, min(len(rows), start + 3))
    output = []
    for chunk in fragments:
        # Rows separated by unchanged code in distinct hunks need separate headers.
        segments, segment = [], []
        for row in chunk:
            if segment:
                previous = segment[-1]
                if row[1] != previous[1] + previous[3] or row[2] != previous[2] + previous[4]:
                    segments.append(segment)
                    segment = []
            segment.append(row)
        if segment:
            segments.append(segment)
        for segment in segments:
            old_count, new_count = sum(r[3] for r in segment), sum(r[4] for r in segment)
            header = '@@ -%s,%s +%s,%s @@' % (segment[0][1], old_count, segment[0][2], new_count)
            output.append([header] + [r[0] for r in segment])
    return output, len(rows)


def private_learning_texts(logs):
    return [text for log in logs for text in
            [log.get('goal', '')] + [step.get(key, '') for step in log.get('history', [])
                                    for key in ('observation', 'action', 'next_observation')]]


def public_event(trial, parent_source, candidate_source, *, tokenizer=None,
                 behavior=None, parent_behavior=None, private_texts=(), shared_code=None):
    opinions, assessments = _opinions(trial)
    raw = comparison_for_feedback(trial.get('exploration'))
    comparison = None if raw is None else {key: raw.get(key) for key in ('complete', 'parent', 'candidate', 'delta')}
    if comparison is not None:
        comparison['tasks'] = len(raw.get('task_ids', []))
    context = trial['public_context']
    assessment = (((trial.get('post_review') or {}).get('report') or {}).get('assessment') or {})
    review_metadata = (dict(decision=assessment['decision']) if 'decision' in assessment
                       else dict(post_assessment=assessment.get('verdict')))
    value = dict(kind='code_change_trial', schema=EXPERIENCE_VERSION,
                 observed_context={key: context.get(key) for key in ('environment', 'family')},
                 before_hash=trial['parent_hash'], after_hash=trial.get('candidate_hash'),
                 raw_source_hash=trial.get('raw_source_hash', hashlib.sha256(candidate_source.encode()).hexdigest()
                                           if candidate_source is not None else None),
                 support_hash=digest(trial), status=trial['public_status'], accepted=trial['accepted'],
                 synthetic=trial['synthetic'], error=_error(trial.get('error'), () if shared_code else private_texts),
                 exploration=comparison,
                 structure=_structure(candidate_source if candidate_source is not None else parent_source),
                 hypothesis=dict(status='unvalidated_archived_at_support_hash'),
                 problem_features=[], review_opinions=[], diagnostic_assessments=[], code_diff='', omissions={},
                 review_metadata=review_metadata)
    shared = shared_code
    if shared and candidate_source is None and value['error']:
        # Arbitrary response text in malformed envelopes is private item evidence.
        value['error'] = dict(kind='proposal_without_submitted_source',
                              original_hash=value['error']['original_hash'])
    actual_fields = set((value['error'] or {}).get('actual_fields', []))
    if shared:
        value['structure'] = [shared.name(name) for name in value['structure']]
        if value['error'] and candidate_source is not None:
            error = value['error']
            error['kind'] = shared.name(error['kind']) if error['kind'].isidentifier() else 'candidate_error'
            error['symbols'] = [shared.name(name) for name in error['symbols']]
            if 'actual_fields' in error:
                error['actual_fields'] = [shared.field(key) for key in error['actual_fields']]
            if error['field_path'] and not re.fullmatch(r'/(?:state|session|memory(?:/\d+)?)?', error['field_path']):
                error['field_path'] = None
            if error.get('expected') not in {'id and experience', 'state, session, memory'}:
                error.pop('expected', None)
            # The unbounded native message stays in the local receipt.
            error.pop('constraint', None)
    value['id'] = digest(value)[:24]
    if _count(tokenizer, value) > RECORD_TOKENS:
        raise ValueError('Required mechanism experience exceeds the fixed 1400-token budget')
    problems = _prioritize_problems(execution_problems(parent_behavior), execution_problems(behavior),
                                    parent_source, candidate_source)
    if shared:
        for item in [*problems, *assessments]:
            item['function'] = shared.name(item.get('function'))
            item['terms'] = ' '.join(shared.name(name) for name in item['terms'].split())
    lines = _code_lines(parent_source, candidate_source, private_texts, shared) if candidate_source is not None else []
    points = [('candidate', int(point)) for point in re.findall(r'line (\d+), column', str(trial.get('error', '')))]
    has_error_site = bool(points)
    # Returns expose the new externally observable behavior, especially when a
    # large old implementation is replaced. This is code selection, not a
    # preferred memory policy. Complete source remains bound by its raw hash.
    try:
        candidate_tree = ast.parse(candidate_source or '')
        for node in ast.walk(candidate_tree):
            if isinstance(node, ast.Dict) and actual_fields and actual_fields.issubset({
                    key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}):
                points.append(('candidate', node.lineno))
        has_error_site = bool(points)
        if len((candidate_source or '').splitlines()) <= 8:
            points += [('candidate', line) for line in range(1, len((candidate_source or '').splitlines()) + 1)]
        points += [(item['subject'], item['line']) for item in problems if type(item.get('line')) is int]
        points += [('candidate', node.lineno) for node in ast.walk(candidate_tree) if isinstance(node, ast.Return)]
    except SyntaxError:
        pass
    chunks, total_lines = _fragments(lines, points)
    # Reserve the complete omission metadata before assigning any optional text.
    # In particular byte-counting fixtures must not overflow on the final hash.
    totals = Counter(row['subject'] for row in problems)
    value['omissions'] = dict(problem_features={side: totals[side] for side in ('parent', 'candidate')},
                             review_opinions=len(opinions),
                             diagnostic_assessments=len(assessments), code_diff_lines=total_lines,
                             code_scope=dict(parent=len(parent_source.splitlines()),
                                             candidate=len((candidate_source or '').splitlines())),
                             raw_diff_hash=digest(''.join(difflib.unified_diff(
                                 parent_source.splitlines(True), (candidate_source or '').splitlines(True),
                                 fromfile='parent.py', tofile='candidate.py'))))
    if shared:
        value['omissions']['code_view'] = 'Structure only. Comments omitted; private names and non-protocol literals use record-local aliases.'
    if _count(tokenizer, value) > RECORD_TOKENS:
        raise ValueError('Required mechanism experience exceeds the fixed 1400-token budget')
    # Required evidence must be readable before optional diagnostic arrays can
    # consume the budget. At least one changed candidate line (or deletion for an
    # empty draft) is mandatory; a metadata-only experience is not useful.
    shown_lines, shown_count = [], 0
    if chunks:
        changed = [chunk for chunk in chunks if any(line.startswith(('+', '-')) for line in chunk[1:])]
        required = next((chunk for chunk in changed if any(line.startswith('+') for line in chunk[1:])),
                        changed[0] if changed else chunks[0])
        # An unchanged source site is useful context, not a mandatory modification.
        # Reserve the actual edit and a reported error site, then use remaining
        # space for unchanged context. This matters even for byte-count fixtures.
        required_chunks = ([chunks[0], required] if has_error_site and required is not chunks[0]
                           else [required])
        for chunk in required_chunks:
            shown_lines.extend(chunk)
            shown_count += len(chunk) - 1
        value['code_diff'] = '\n'.join(shown_lines)
        value['omissions']['code_diff_lines'] = total_lines - shown_count
        if _count(tokenizer, value) > RECORD_TOKENS:
            raise ValueError('Required mechanism change/error evidence exceeds the fixed 1400-token budget')
        chunks = [chunk for chunk in chunks if chunk not in required_chunks]

    # Reserve useful code space without allowing repetitive parent diagnostics
    # or full observation hashes to displace the actual modification. Full
    # diagnostic bindings and opinion evidence IDs resolve through support_hash.
    for key, items, allowance in [('problem_features', problems, 180), ('review_opinions', opinions, 180),
                                  ('diagnostic_assessments', assessments, 180)]:
        shown = []
        for item in items:
            value[key] = shown + [item]
            if _count(tokenizer, value[key]) > allowance or _count(tokenizer, value) > RECORD_TOKENS - 220:
                value[key] = shown
                continue
            shown.append(item)
        if key == 'problem_features':
            kept = Counter(row['subject'] for row in shown)
            value['omissions'][key] = {side: totals[side] - kept[side] for side in ('parent', 'candidate')}
        else:
            value['omissions'][key] = len(items) - len(shown)
    for chunk in chunks:
        value['code_diff'] = '\n'.join(shown_lines + chunk)
        if _count(tokenizer, value) > RECORD_TOKENS - 50:
            value['code_diff'] = '\n'.join(shown_lines)
            continue
        shown_lines.extend(chunk)
        shown_count += len(chunk) - 1
    value['omissions']['code_diff_lines'] = total_lines - shown_count
    value['id'] = digest({key: item for key, item in value.items() if key != 'id'})[:24]
    if _count(tokenizer, value) > RECORD_TOKENS:
        raise ValueError('Compact mechanism experience exceeds the fixed 1400-token budget')
    return value


class EvolutionMemory:
    def __init__(self, records=()):
        self.records, ids = [], set()
        allowed = {'id', 'kind', 'schema', 'observed_context', 'before_hash', 'after_hash',
                   'raw_source_hash', 'support_hash', 'status', 'accepted', 'synthetic', 'error',
                   'exploration', 'hypothesis', 'problem_features', 'review_opinions',
                   'diagnostic_assessments', 'code_diff', 'omissions', 'review_metadata', 'structure'}
        for raw in records:
            record = copy.deepcopy(raw)
            if record.get('kind') != 'code_change_trial':
                raise ValueError('Evolution Memory requires code_change_trial records')
            if set(record) - allowed:
                raise ValueError('Unknown fields in transferable design memory')
            if record.get('schema', EXPERIENCE_VERSION) != EXPERIENCE_VERSION:
                raise ValueError('Unsupported mechanism experience schema')
            forbidden = {'episodes', 'history', 'state', 'raw_logs', 'native_recorder',
                         'confirmation', 'final', 'memory_retrieved', 'answer', 'final_answer',
                         'observation', 'next_observation'}
            def check(value):
                if isinstance(value, dict):
                    if forbidden.intersection(value):
                        raise ValueError('Raw task/final data forbidden in transferable memory')
                    for child in value.values():
                        check(child)
                elif isinstance(value, list):
                    for child in value:
                        check(child)
            check(record)
            if record['id'] not in ids:
                ids.add(record['id'])
                self.records.append(record)

    def select(self, query, tokenizer, *, limit=RETRIEVAL_LIMIT, token_budget=RETRIEVAL_TOKENS, exclude_ids=()):
        environment = query.get('environment')
        if any(record.get('observed_context', {}).get('environment') not in (None, environment)
               for record in self.records):
            raise ValueError('Mechanism experience retrieval cannot cross dataset boundaries')
        ordered = sorted(self.records, key=lambda item: item['id'])
        indices = rank(_matching(query), [_matching(record) for record in ordered])
        selected, receipts, excluded = [], [], set(exclude_ids)
        for position, index in enumerate(indices):
            archived = ordered[index]
            record = archived
            projection = comparison_for_feedback(archived.get('exploration'))
            if projection != archived.get('exploration'):
                # Keep the frozen record ID for provenance, exclusions and source
                # attribution. The receipt binds the actual normalized model view.
                record = dict(archived, exploration=projection)
            tokens = tokenizer.count(dumps(record))
            if tokens > RECORD_TOKENS:
                raise ValueError('Unbounded mechanism experience; project it before retrieval')
            reason = ('already_provided_as_previous_attempt' if record['id'] in excluded else
                      'record_limit' if len(selected) >= limit else
                      'remaining_token_budget' if tokenizer.count(dumps(selected + [record])) > token_budget else 'selected')
            receipts.append(dict(id=record['id'], rank=position + 1, tokens=tokens, reason=reason))
            if record is not archived:
                receipts[-1].update(archived_record_hash=digest(archived), displayed_record_hash=digest(record))
            if reason == 'selected':
                selected.append(record)
        return selected, dict(query=query, query_hash=digest(query), ranked_records=receipts,
                              token_budget=token_budget, tokens=tokenizer.count(dumps(selected)))

    def retrieve(self, context, tokenizer, *, limit=RETRIEVAL_LIMIT, token_budget=RETRIEVAL_TOKENS):
        return self.select(context, tokenizer, limit=limit, token_budget=token_budget)[0]
