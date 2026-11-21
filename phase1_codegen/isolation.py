"""Trusted syscall filter shared by all generated event programs."""
import ctypes
import ctypes.util


def install_filter():
    library=ctypes.CDLL(ctypes.util.find_library('seccomp') or 'libseccomp.so.2',use_errno=True)
    library.seccomp_init.argtypes=[ctypes.c_uint32];library.seccomp_init.restype=ctypes.c_void_p
    library.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    library.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];library.seccomp_syscall_resolve_name.restype=ctypes.c_int
    library.seccomp_load.argtypes=[ctypes.c_void_p];library.seccomp_load.restype=ctypes.c_int
    library.seccomp_release.argtypes=[ctypes.c_void_p]
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0)!=0: raise RuntimeError('no_new_privs failed')
    context=library.seccomp_init(0x80000000)  # KILL_PROCESS; current kernel4.18 supports it
    if not context: raise RuntimeError('seccomp_init failed')
    allowed=['read','write','close','brk','mmap','munmap','mprotect','mremap','futex','rt_sigaction',
        'rt_sigprocmask','rt_sigreturn','sigaltstack','exit','exit_group','clock_gettime','gettimeofday','sched_yield']
    for name in allowed:
        number=library.seccomp_syscall_resolve_name(name.encode())
        if number<0 or library.seccomp_rule_add(context,0x7fff0000,number,0)!=0: raise RuntimeError('seccomp allow rule failed')
    if library.seccomp_load(context)!=0: raise RuntimeError('seccomp_load failed; code will not execute')
    library.seccomp_release(context)
    return libc,library
