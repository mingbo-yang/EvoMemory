"""JSON contracts and restricted Python syntax for entire memory implementations.

The language permits new state schemas and algorithms.  It is not an operator
menu.  Syntax validation is one layer; generated code also executes in a fresh
resource-limited process after the existing kernel syscall filter is installed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math

SOURCE_BYTES = 32768
INPUT_BYTES = 2 * 1024 * 1024
OUTPUT_BYTES = 512 * 1024
STATE_BYTES = 256 * 1024
ITERATION_BUDGET = 100000
MAX_MODEL_CALLS = 8
LANGUAGE_VERSION = 'restricted-python-v2'
TYPE_NAMES = frozenset({'dict', 'list', 'str', 'int', 'float', 'bool', 'tuple', 'set'})
EXCEPTION_NAMES = frozenset({'ValueError', 'TypeError', 'KeyError', 'IndexError'})
CAPABILITY_JSON_BYTES = 128 * 1024
CAPABILITY_RECEIPT_BYTES = 16 * 1024
CAPABILITY_CALLS = frozenset({'model_call'})
SAFE_CALLS = frozenset({
    'abs', 'min', 'max', 'sum', 'len', 'round', 'range', 'sorted', 'str',
    'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'enumerate',
    'zip', 'reversed', 'any', 'all', 'isinstance',
}) | CAPABILITY_CALLS
SAFE_METHODS = frozenset({
    'lower', 'upper', 'casefold', 'strip', 'lstrip', 'rstrip', 'split',
    'rsplit', 'splitlines', 'join', 'replace', 'startswith', 'endswith',
    'count', 'index', 'find', 'rfind', 'isalnum', 'isalpha', 'isdigit',
    'get', 'items', 'keys', 'values', 'copy', 'setdefault', 'update',
    'append', 'extend', 'insert', 'pop', 'remove', 'clear', 'reverse',
    'sort', 'add', 'discard', 'union', 'intersection', 'difference',
})
ALLOWED = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.Assign, ast.AugAssign, ast.Expr, ast.If, ast.IfExp, ast.For,
    ast.Break, ast.Continue, ast.Pass, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript,
    ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.Attribute, ast.keyword, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.comprehension, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
    ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.BitOr, ast.BitAnd,
    ast.Try, ast.ExceptHandler,
)


class Rejected(ValueError):
    """A generated implementation or its output violated a fixed contract."""


class CapabilityContractError(Rejected):
    """The mechanism's model request violated the advertised input contract."""


class CapabilityInfrastructureError(RuntimeError):
    """The trusted model service, its transport or its evidence failed."""


class GeneratedBudgetExceeded(Rejected):
    """A trusted counter for generated work exceeded its declared limit."""


def _static_names(node, allowed):
    return (isinstance(node, ast.Name) and node.id in allowed or
            isinstance(node, ast.Tuple) and bool(node.elts)
            and all(isinstance(item, ast.Name) and item.id in allowed for item in node.elts))


def _type_position(node, parents):
    parent = parents.get(node)
    container = parent if isinstance(parent, ast.Tuple) else node
    owner = parents.get(container)
    if (isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name)
            and owner.func.id == 'isinstance' and len(owner.args) == 2
            and owner.args[1] is container and _static_names(container, TYPE_NAMES)):
        return node.id in TYPE_NAMES
    return (isinstance(owner, ast.ExceptHandler) and owner.type is container
            and _static_names(container, EXCEPTION_NAMES) and node.id in EXCEPTION_NAMES)


def _source_error(message, node=None):
    """Locate a language violation without modifying the proposed source."""
    if len(message) > 256:
        message = message[:256] + ' [%d diagnostic characters omitted]' % (len(message) - 256)
    if node is not None and getattr(node, 'lineno', None) is not None:
        message += ' (line %d, column %d)' % (node.lineno, node.col_offset + 1)
    return message


def validate_source(source):
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > SOURCE_BYTES:
        raise Rejected('source must be nonempty Python within 32768 bytes')
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise Rejected('invalid Python source: %s (line %s, column %s)' %
                       (exc.msg, exc.lineno, exc.offset)) from exc
    except RecursionError as exc:
        raise Rejected('invalid Python source: parser recursion limit exceeded') from exc
    if not 1 <= len(tree.body) <= 16 or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        raise Rejected('top level must contain on_event and optional helper functions only')
    names = [n.name for n in tree.body]
    if len(set(names)) != len(names) or 'on_event' not in names:
        raise Rejected('unique on_event(state, event, context) function required')
    if any(n.startswith('_') or n in SAFE_CALLS | EXCEPTION_NAMES for n in names):
        raise Rejected('private names and builtin shadowing forbidden')
    functions = set(names)
    for fn in tree.body:
        args = fn.args
        actual = [a.arg for a in args.args]
        expected = ['state', 'event', 'context'] if fn.name == 'on_event' else None
        if (expected is not None and actual != expected) or len(actual) > 8:
            message = ('fixed interface for ' + fn.name + ': ' + fn.name + '(' + ', '.join(expected) + ')'
                       if expected is not None else 'helper accepts at most eight positional arguments')
            raise Rejected(_source_error(message + '; actual arguments: ' + repr(actual), fn))
        if (args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or
                args.defaults or args.kw_defaults or fn.decorator_list or fn.returns or
                any(a.annotation or a.arg.startswith('_') or a.arg in SAFE_CALLS | EXCEPTION_NAMES | functions for a in args.args)):
            raise Rejected(_source_error(
                'plain functions without defaults, annotations, decorators, or reserved arguments required', fn))
    nodes = list(ast.walk(tree))
    if len(nodes) > 6000 or sum(isinstance(n, ast.FunctionDef) for n in nodes) != len(tree.body):
        raise Rejected('AST size exceeded or nested function forbidden')
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    errors = []
    error_count = 0
    def reject(message, node):
        nonlocal error_count
        error_count += 1
        if len(errors) < 12:
            errors.append(_source_error(message, node))
    for node in nodes:
        if not isinstance(node, ALLOWED):
            reject('forbidden syntax: ' + type(node).__name__, node)
        if isinstance(node, ast.Name):
            if node.id.startswith('_'):
                reject('private and introspection names forbidden: ' + node.id, node)
            if node.id in SAFE_CALLS | EXCEPTION_NAMES | functions:
                parent = parents.get(node)
                direct = isinstance(parent, ast.Call) and parent.func is node and node.id not in EXCEPTION_NAMES
                if not isinstance(node.ctx, ast.Load) or not (direct or _type_position(node, parents)):
                    reject('functions cannot be shadowed, aliased, or used as data: ' + node.id, node)
        if isinstance(node, ast.Try) and (node.finalbody or not node.handlers):
            reject('try requires narrow handlers and forbids finally', node)
        if isinstance(node, ast.ExceptHandler) and (node.name is not None or not _static_names(node.type, EXCEPTION_NAMES)):
            reject('except requires static permitted data exceptions without object binding', node)
        if isinstance(node, ast.Attribute):
            parent = parents.get(node)
            if (node.attr not in SAFE_METHODS or not isinstance(parent, ast.Call) or
                    parent.func is not node or not isinstance(node.ctx, ast.Load)):
                reject('only direct calls to approved JSON container/string methods allowed: ' + node.attr, node)
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and node.func.id == 'isinstance'
                    and (len(node.args) != 2 or node.keywords or not _static_names(node.args[1], TYPE_NAMES))):
                reject('isinstance requires a value and static permitted type names', node)
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_CALLS | functions:
                    reject('unknown call: ' + node.func.id, node)
            elif not isinstance(node.func, ast.Attribute):
                reject('indirect function calls forbidden', node)
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg.startswith('_') or keyword.arg == 'key':
                    reject('keyword unpacking, private keywords, and callbacks forbidden; actual keyword: '
                           + repr(keyword.arg), keyword)
        if isinstance(node, ast.Dict) and any(k is None for k in node.keys):
            reject('dictionary unpacking forbidden', node)
        if isinstance(node, ast.comprehension) and node.is_async:
            reject('asynchronous comprehensions forbidden', node)
        if isinstance(node, ast.Constant):
            v = node.value
            if type(v) not in (type(None), bool, int, float, str):
                reject('only JSON-like constants allowed; actual type: ' + type(v).__name__, node)
            if type(v) in (int, float) and (abs(v) > 1000000 or not math.isfinite(v)):
                reject('numeric constant budget exceeded', node)
            if isinstance(v, str) and len(v.encode()) > 8192:
                reject('string constant budget exceeded', node)
    if errors:
        if error_count > len(errors):
            errors.append('%d additional source violations omitted from diagnostics' % (error_count - len(errors)))
        raise Rejected('\n'.join(errors))
    # Aliases/callbacks are forbidden above, so this direct call graph covers
    # both immediate recursion and mutual/helper recursion.
    graph = {fn.name: {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id in functions}
             for fn in tree.body}
    visiting, finished = set(), set()
    def visit(name):
        if name in visiting:
            raise Rejected('recursive call graph forbidden')
        if name in finished:
            return
        visiting.add(name)
        for child in graph[name]:
            visit(child)
        visiting.remove(name)
        finished.add(name)
    for name in graph:
        visit(name)
    return tree


def _json_copy(value, byte_budget, node_budget):
    """Shared finite-JSON validation; resource limits belong to the call boundary."""
    count = [0]
    def check(item, depth, path=''):
        count[0] += 1
        if (node_budget is not None and count[0] > node_budget) or depth > 40:
            raise Rejected('JSON depth/node budget exceeded; path=' + (path or '/'))
        kind = type(item)
        if kind in (type(None), bool, str):
            return
        if kind in (int, float):
            if not math.isfinite(item):
                raise Rejected('JSON numbers must be finite; path=' + (path or '/'))
            return
        if kind is list:
            for index, child in enumerate(item):
                check(child, depth + 1, path + '/' + str(index))
            return
        if kind is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise Rejected('JSON object keys must be strings; path=' + (path or '/')
                                   + '; actual key type=' + type(key).__name__)
                check(child, depth + 1, path + '/' + key.replace('~', '~0').replace('/', '~1'))
            return
        raise Rejected('non-JSON value: ' + kind.__name__ + '; path=' + (path or '/'))
    try:
        check(value, 0)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except (RecursionError, OverflowError) as exc:
        raise Rejected('invalid recursive or numeric JSON value') from exc
    if byte_budget is not None and len(encoded.encode()) > byte_budget:
        raise Rejected('JSON byte budget exceeded')
    return json.loads(encoded)


def json_copy(value, byte_budget=INPUT_BYTES):
    """Copy one bounded worker/state payload as an actual finite JSON tree."""
    return _json_copy(value, byte_budget, 100000)


def archive_copy(value):
    """Copy host-owned cumulative evidence, independently of worker I/O quotas.

    Archives include receipts and repeated memory witnesses. They never enter
    generated code wholesale: event payloads, state and outputs retain their
    own fixed limits through json_copy/validate_state at every invocation.
    """
    return _json_copy(value, None, None)


def validate_state(state):
    if type(state) is not dict:
        raise Rejected('state must be a JSON object')
    return json_copy(state, STATE_BYTES)


def validate_request(operation, state, experience, context):
    if operation != 'event' or type(context) is not dict:
        raise Rejected('operation must be event; context must be an object')
    if type(experience) is not dict:
        raise Rejected('event must be an object')
    validate_state(state)
    return json_copy(dict(operation=operation, state=state, experience=experience, context=context))


def capability_request(model_id, payload, operation):
    """Trusted deterministic request identity; generated code sees only the result."""
    if type(model_id) is not str or not 0 < len(model_id) <= 128:
        raise Rejected('model_call requires a nonempty model_id of at most 128 characters')
    if type(payload) is not dict or operation != 'event':
        raise Rejected('model_call payload must be a JSON object within an event operation')
    value = json_copy(dict(model_id=model_id, payload=payload, operation=operation), CAPABILITY_JSON_BYTES)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(',', ':')).encode()
    return dict(value, request_hash=hashlib.sha256(encoded).hexdigest())


def validate_output(operation, value):
    if operation != 'event':
        raise Rejected('operation must be event')
    value = json_copy(value, OUTPUT_BYTES - 4096)
    if type(value) is not dict or set(value) != {'state', 'session', 'memory'}:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise Rejected('event output requires exactly state, session, memory; path=/; actual=' + repr(actual))
    try:
        validate_state(value['state'])
    except Rejected as exc:
        raise Rejected(str(exc) + '; path=/state') from exc
    if type(value['session']) is not dict:
        raise Rejected('event session must be a JSON object; path=/session; actual=' + type(value['session']).__name__)
    try:
        json_copy(value['session'], STATE_BYTES)
    except Rejected as exc:
        raise Rejected(str(exc) + '; path=/session') from exc
    rows = value['memory']
    if rows is None:
        return value
    if type(rows) is not list or len(rows) > 256:
        actual = 'list length ' + str(len(rows)) if type(rows) is list else type(rows).__name__
        raise Rejected('event memory must be null or at most 256 candidates before fixed top-4 bounding; '
                       'path=/memory; actual=' + actual)
    ids = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {'id', 'experience'}:
            actual = sorted(row) if type(row) is dict else type(row).__name__
            raise Rejected('event memory row ' + str(index) + ' requires exactly id and experience; '
                           'path=/memory/' + str(index) + '; actual=' + repr(actual))
        if (type(row['id']) is not str or not 0 < len(row['id']) <= 256
                or type(row['experience']) is not str or row['id'] in ids):
            raise Rejected('event memory rows require unique nonempty string ids and string experience; '
                           'path=/memory/' + str(index) + '; actual id type=' + type(row['id']).__name__
                           + ', experience type=' + type(row['experience']).__name__
                           + ', id length=' + str(len(row['id']) if type(row['id']) is str else None)
                           + ', duplicate id=' + str(type(row['id']) is str and row['id'] in ids))
        ids.add(row['id'])
    return value


class BudgetedLoops(ast.NodeTransformer):
    """Insert trusted shared iteration accounting after validating source."""
    def visit_For(self, node):
        self.generic_visit(node)
        node.iter = ast.Call(func=ast.Name(id='_bounded_iter', ctx=ast.Load()), args=[node.iter], keywords=[])
        return node

    def visit_comprehension(self, node):
        self.generic_visit(node)
        node.iter = ast.Call(func=ast.Name(id='_bounded_iter', ctx=ast.Load()), args=[node.iter], keywords=[])
        return node
