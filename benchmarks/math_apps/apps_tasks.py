"""APPS public problem adapter and CPU scoring; private tests stay outside the submission process."""
import ast
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

from .paths import ASSETS
HERE=ASSETS
ARCHIVE=HERE/'apps_archive'
PILOT=Path(__file__).resolve().parent
if str(PILOT) not in sys.path:sys.path.append(str(PILOT))
from .code_sandbox import grade_stdio, source_of


def load_public():
    """Return only indexable, gradeable problems with original public question and starter."""
    with (HERE/'apps_public.jsonl').open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


@lru_cache(maxsize=1)
def _index():
    return json.loads((ARCHIVE/'offsets.json').read_bytes())


def _record(task_id):
    entry=_index()[task_id]
    with (ARCHIVE/(entry['split']+'.jsonl')).open('rb') as f:
        f.seek(entry['offset'])
        wire=f.read(entry['bytes'])
    from hashlib import sha256
    if sha256(wire).hexdigest()!=entry['sha256']:raise ValueError('Immutable original APPS row changed')
    return json.loads(wire)


def _text(value):
    return value if isinstance(value,str) else '\n'.join(value)


def stdio_equal(actual, expected):
    """APPS testing_util.py lines 323–456, stdio path after joining list-valued ground truth."""
    output=actual.splitlines()
    if '\n'.join(output).strip()==expected.strip() or '\n'.join(s.strip() for s in output).strip()==expected.strip():
        return True
    target=[x.strip() for x in expected.split('\n') if x]
    if output==[expected] or output==target:return True
    output=[x for x in output if x]
    try:
        # numpy.allclose default uses the magnitude of the second operand.
        def close(x,y):
            return x==y or (math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-8+1e-5*abs(y))
        if len(output)==len(target) and all(close(float(x),float(y)) for x,y in zip(output,target)):
            return True
    except (ValueError,OverflowError):pass
    left=[set(s.split()) for s in output if s.split()]
    right=[set(s.split()) for s in target]
    if {frozenset(s) for s in left}=={frozenset(s) for s in right}:return True
    try:
        def rounded(rows):return {frozenset(round(float(x),3) for x in group) for group in rows}
        return rounded(left)==rounded(right)
    except (ValueError, OverflowError):return False


def _int_dict(value):
    return {int(k):v for k,v in value.items()}


def call_equal(actual, serialized_expected):
    """Use Python return-value equality, including the author's singleton-wrapper and tuple cases."""
    marker='APPS_RETURN_362aedc3 '
    lines=actual.splitlines()
    if not lines or not lines[-1].startswith(marker):return False
    try:result=ast.literal_eval(lines[-1][len(marker):])
    except (SyntaxError,ValueError,MemoryError,RecursionError):return False
    expected=json.loads(serialized_expected)
    try:
        if isinstance(expected,dict):expected=[_int_dict(expected)]
        if isinstance(expected,list) and expected and isinstance(expected[0],dict):expected=[_int_dict(expected[0])]
    except (ValueError,TypeError):pass
    if isinstance(result,tuple):result=list(result)
    if result==expected or isinstance(expected,list) and expected and result==expected[0]:return True
    return bool(isinstance(result,list) and result and isinstance(result[0],tuple) and isinstance(expected,list)
                and expected and [list(row) for row in result]==expected[0])


PRELUDE='''import sys, itertools, collections, math, fractions, random, heapq
from itertools import accumulate, product, permutations, combinations
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
from typing import List, Tuple
from heapq import *
'''


def _with_prelude(source):
    nodes=ast.parse(source).body
    split=0
    for node in nodes:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) and split==0:
            split=node.end_lineno
        elif isinstance(node,ast.ImportFrom) and node.module=='__future__':
            split=node.end_lineno
        else:break
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:split])+'\n'+PRELUDE+''.join(lines[split:])+'\n'


def _call_script(source,name):
    if not isinstance(name,str) or not name.isidentifier() or name.startswith('_'):
        raise ValueError('Unexpected APPS entry point')
    fn='Solution().'+name if 'class Solution' in source else name
    return _with_prelude(source)+f'''
import json as _apps_json, sys as _apps_sys
_apps_args=_apps_json.loads(_apps_sys.stdin.read())
if _apps_args and isinstance(_apps_args[0],dict):
    try:
        _apps_args=[{{int(k):v for k,v in _apps_args[0].items()}}]
    except (TypeError,ValueError):
        pass
_apps_return={fn}(*_apps_args)
print("APPS_RETURN_362aedc3 "+repr(_apps_return))
'''


def grade(task_id,completion):
    raw=_record(task_id)
    try:
        cases=json.loads(raw['input_output'])
    except ValueError:
        return dict(score=None,status='unsupported_original_case_encoding',tests_run=0,tests_passed=0)
    xs,ys=cases['inputs'],cases['outputs']
    n=len(xs)
    if completion is None:
        return dict(score=None,status='missing_completion',tests_total=n,tests_run=0,tests_passed=0)
    try: source=source_of(completion)
    except ValueError as exc:
        return dict(score=0,status=str(exc),tests_total=n,tests_run=0,tests_passed=0)
    name=cases.get('fn_name')
    if name:
        result=grade_stdio(_call_script(source,name),[json.dumps(x) for x in xs],
                           [json.dumps(y) for y in ys],compare=call_equal)
    else:
        result=grade_stdio(_with_prelude(source),[_text(x) for x in xs],[_text(y) for y in ys],compare=stdio_equal)
    # The authors' runtime allows scientific packages that the isolated worker lacks.
    # An unavailable Python dependency does not establish an incorrect algorithm.
    if result['status']=='unsupported_python':
        result['score']=None
    result.update(benchmark='APPS',task_id=task_id,interface='call_based' if name else 'stdio')
    return result
