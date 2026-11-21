"""Grade original standard I/O pairs outside isolated candidate processes."""
import ast
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile


WORKER = Path(__file__).resolve().with_name('code_worker.py')
READY = b'ISOLATED-STDIO-v1\n'
LIMITS = 'CPython -I -S; seccomp KILL_PROCESS default deny; 4s CPU/8s wall, 512 MiB memory, 256 KiB outputs per case'


def source_of(completion):
    if not isinstance(completion, str) or not completion.strip() or len(completion.encode()) > 65536:
        raise ValueError('format_error')
    if '```' in completion:
        fenced = re.fullmatch(r'\s*```(?:python(?:3)?|py)?[ \t]*\r?\n(.*?)\r?\n```\s*', completion, re.I | re.S)
        if not fenced:
            raise ValueError('format_error')
        completion = fenced.group(1)
    try:
        compile(completion, '<submission>', 'exec', flags=ast.PyCF_ONLY_AST, dont_inherit=True)
        compile(completion, '<submission>', 'exec', dont_inherit=True)
    except (ValueError, SyntaxError, RecursionError):
        raise ValueError('syntax_error') from None
    return completion


def _run_case(source_file, case, compare):
    user_input, expected = case
    with (tempfile.TemporaryFile(dir='/tmp') as stdin, tempfile.TemporaryFile(dir='/tmp') as out,
          tempfile.TemporaryFile(dir='/tmp') as err):
        stdin.write(user_input.encode())
        stdin.seek(0)
        try:
            child = subprocess.run([sys.executable, '-I', '-S', '-B', str(WORKER), str(source_file)],
                cwd=source_file.parent, env={'LC_ALL':'C.UTF-8'}, close_fds=True,
                stdin=stdin, stdout=out, stderr=err, timeout=8)
        except subprocess.TimeoutExpired:
            err.seek(0)
            return 'resource_limit' if err.read(len(READY)) == READY else 'infrastructure_error'
        except OSError:
            return 'infrastructure_error'
        err.seek(0)
        if err.read(len(READY)) != READY:
            return 'infrastructure_error'
        if child.returncode:
            if child.returncode in (88, -signal.SIGKILL, -signal.SIGXCPU, -signal.SIGXFSZ):
                return 'resource_limit'
            if child.returncode in (87, -signal.SIGSYS):
                return 'unsupported_python'
            return 'runtime_error'
        out.seek(0)
        try:
            answer = out.read().decode('utf-8')
        except UnicodeDecodeError:
            return 'wrong_answer'
        return 'passed' if compare(answer, expected) else 'wrong_answer'


def token_equal(actual, expected):
    return actual.split() == expected.split()


def grade_stdio(completion, inputs, outputs, *, compare=token_equal):
    if not inputs or len(inputs) != len(outputs) or not all(isinstance(v, str) for v in [*inputs,*outputs]):
        raise ValueError('Grader requires aligned original text I/O cases')
    result = dict(score=None, status='infrastructure_error', tests_total=len(inputs), tests_run=0,
                  tests_passed=0, first_failed_index=None)
    if completion is None:
        return dict(result, status='missing_completion')
    try:
        source = source_of(completion)
    except ValueError as exc:
        return dict(result, score=0, status=str(exc))
    with tempfile.TemporaryDirectory(dir='/tmp', prefix='evomem-stdio-') as tmp:
        file = Path(tmp) / 'submission.py'
        file.write_text(source, encoding='utf-8')
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
            verdicts=list(pool.map(lambda case: _run_case(file, case, compare),zip(inputs,outputs)))
    result.update(tests_run=len(verdicts), tests_passed=verdicts.count('passed'))
    result['first_failed_index'] = next((i for i,v in enumerate(verdicts) if v != 'passed'), None)
    if 'infrastructure_error' in verdicts:
        return result
    status = next((v for v in verdicts if v != 'passed'), 'passed')
    return dict(result, score=int(status=='passed'), status=status)
