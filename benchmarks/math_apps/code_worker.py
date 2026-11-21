"""Single private test input. Only authenticated host code runs before the syscall filter."""
import ast
import builtins
import importlib
import importlib.util
import io
import os
from pathlib import Path
import resource
import sys
import types


LIBRARIES = frozenset('array bisect collections copy decimal fractions functools heapq io itertools json '
                      'math operator random re statistics string sys typing'.split())
READY = b'ISOLATED-STDIO-v1\n'


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (4, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 10, 256 << 10))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    source = Path(sys.argv[1]).read_text(encoding='utf-8')
    tree = ast.parse(source)
    code = compile(tree, '<submission>', 'exec', dont_inherit=True, optimize=0)
    modules = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or '']
        else:
            continue
        for name in names:
            if name in LIBRARIES:
                modules[name] = importlib.import_module(name)

    # Some contest submissions read fd 0 and write fd 1 directly. No filesystem
    # path access is needed. Snapshot the three inherited standard descriptors.
    stats = {i: os.fstat(i) for i in range(3)}
    case_input = io.BytesIO(sys.stdin.buffer.read())
    class ContestInput(io.TextIOWrapper):
        mode = 'r'

        def fileno(self):
            return 0
    # Hold the wrapper until the end, even when a contestant replaces
    # sys.stdin. Its finalizer otherwise closes the shared fd-0 input buffer.
    initial_stdin = ContestInput(case_input, encoding='utf-8')
    sys.stdin = initial_stdin

    def line_input(prompt=''):
        if prompt:
            print(prompt, end='', flush=True)
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.removesuffix('\n')

    def std_read(fd, count):
        if fd != 0:
            raise PermissionError('only standard input is readable')
        return case_input.read(count)
    def std_write(fd, data):
        if fd not in (1, 2):
            raise PermissionError('only standard output/error are writable')
        return os.write(fd, data)
    modules['os'] = types.SimpleNamespace(read=std_read, write=std_write, fstat=stats.__getitem__)

    def allowed_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in modules:
            raise ImportError('unsupported contest module')
        return modules[name]

    trusted_filter = Path(__file__).resolve().parents[2] / 'phase1_codegen/isolation.py'
    spec = importlib.util.spec_from_file_location('trusted_filter', trusted_filter)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    allowed = vars(builtins).copy()
    for name in ('open', 'eval', 'exec', 'compile', 'breakpoint', 'help'):
        allowed.pop(name, None)
    allowed['__import__'] = allowed_import
    allowed['input'] = line_input
    namespace = {'__builtins__': allowed, '__name__': '__main__'}
    finish = os._exit
    guard.install_filter()
    os.write(2, READY)
    try:
        exec(code, namespace)
        sys.stdout.flush()
        finish(0)
    except SystemExit as exc:
        sys.stdout.flush()
        finish(0 if exc.code in (0, None) else 86)
    except ImportError:
        finish(87)
    except (MemoryError, RecursionError):
        finish(88)
    except BaseException:
        finish(86)


if __name__ == '__main__':
    main()
