"""Trusted standalone bootstrap. Candidate code runs only after confinement.

This file must remain runnable with Python -I -S; no project imports. The host
pins its bytes, interpreter and libseccomp. Namespace privilege is unnecessary;
chroot privilege is required and failure is terminal before candidate execution.
"""
import base64
import binascii
import ctypes
import hashlib
import json
import os
import resource
import sys


class Compare(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("a", ctypes.c_uint64), ("b", ctypes.c_uint64)]


def main():
    control_fd, root, library = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    # The parent already bounds transport bytes; also bound this standalone reader.
    raw = sys.stdin.buffer.read(19000000)
    if len(raw) >= 19000000: raise ValueError("oversized bootstrap input")
    request = json.loads(raw)
    limits = request["limits"]
    for kind, bound in ((resource.RLIMIT_AS, limits["memory_bytes"]),
                        (resource.RLIMIT_CPU, limits["cpu_seconds"]),
                        (resource.RLIMIT_CORE, 0), (resource.RLIMIT_FSIZE, 0),
                        (resource.RLIMIT_NPROC, 0)):
        resource.setrlimit(kind, (bound, bound))
    # No compilation executes candidate statements. Its namespace starts empty.
    code = compile(request["source"], "<isolated-candidate>", "exec", dont_inherit=True)
    payload = request["payload"]
    host_context = request.get("host_context")
    if host_context is not None and (set(host_context) != {"context_sha256", "scope", "namespace", "value"}
            or host_context["scope"] not in {"synthetic_authority_contract", "registered_production_runtime"}
            or host_context["namespace"] != "TRUSTED_REGISTRARS"
            or not isinstance(host_context["value"], dict)):
        raise ValueError("unsupported explicit host configuration")
    libc = ctypes.CDLL(None, use_errno=True)
    sec = ctypes.CDLL(library, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                          ctypes.c_uint, ctypes.POINTER(Compare)]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    sec.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    # libseccomp checks the native architecture; other ABIs and unknown calls kill.
    context = sec.seccomp_init(0x80000000)  # SCMP_ACT_KILL_PROCESS
    if not context: raise RuntimeError("seccomp initialization failed")
    if sec.seccomp_attr_set(context, 2, 0x80000000) != 0:  # SCMP_FLTATR_ACT_BADARCH
        raise RuntimeError("seccomp bad-architecture policy failed")
    allow = 0x7fff0000
    for name in ("exit", "exit_group", "brk", "mmap", "mremap", "munmap", "mprotect",
                 "futex", "rt_sigreturn", "rt_sigaction", "rt_sigprocmask", "sigaltstack",
                 "clock_gettime", "getrandom", "restart_syscall", "close"):
        number = sec.seccomp_syscall_resolve_name(name.encode())
        if number < 0 or sec.seccomp_rule_add_array(context, allow, number, 0, None) != 0:
            raise RuntimeError("seccomp rule failed: " + name)
    for fd in (1, 2, control_fd):
        condition = Compare(0, 4, fd, 0)  # SCMP_CMP_EQ
        if sec.seccomp_rule_add_array(context, allow,
                sec.seccomp_syscall_resolve_name(b"write"), 1, ctypes.byref(condition)) != 0:
            raise RuntimeError("seccomp descriptor restriction failed")
    # No candidate sees environment, outside cwd, ambient FDs or root capability.
    os.environ.clear()
    for fd in tuple(int(n) for n in os.listdir("/proc/self/fd")):
        if fd not in (1, 2, control_fd):
            try: os.close(fd)
            except OSError: pass
    os.chroot(root)
    os.chdir("/")
    os.setgroups([])
    os.setresgid(65534, 65534, 65534)
    os.setresuid(65534, 65534, 65534)
    if os.getresuid() != (65534,) * 3 or os.getresgid() != (65534,) * 3:
        raise RuntimeError("credential drop failed")
    if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError("dumpability/no-new-privileges restriction failed")
    if libc.prctl(3, 0, 0, 0, 0) != 0 or libc.prctl(39, 0, 0, 0, 0) != 1:
        raise RuntimeError("process restriction readback failed")
    attestation = json.dumps({**request["binding"], "chroot": True, "uid": 65534, "gid": 65534,
        "no_new_privs": True, "dumpable": False, "seccomp_loaded": True,
        "ambient_fds_closed": True}, sort_keys=True).encode()
    if sec.seccomp_load(context) != 0:
        raise RuntimeError("seccomp installation failed")
    sec.seccomp_release(context)
    os.write(control_fd, attestation)
    os.close(control_fd)
    namespace = {"__name__": "candidate", "__builtins__": __builtins__}
    try:
        exec(code, namespace)
        if host_context is not None:
            # A separate frozen host interface supplies public trust material.
            # Candidate outputs still require the independent host oracle; a
            # Python namespace is not a security boundary against source code.
            namespace[host_context["namespace"]] = host_context["value"]
        output = namespace["resolve"](payload)
        if not isinstance(output, dict): raise ValueError("result must be an object")
        encoded = json.dumps(output, allow_nan=False, separators=(",", ":")).encode()
        # Parent independently enforces stream limits, including hostile direct writes.
        if len(encoded) > limits["output_bytes"]: raise ValueError("result exceeds output bound")
        while encoded:
            encoded = encoded[os.write(1, encoded):]
        os._exit(0)
    except BaseException:
        os.write(2, b"candidate execution or serialization failed")
        os._exit(70)


if __name__ == "__main__":
    main()
