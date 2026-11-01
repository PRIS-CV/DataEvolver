"""Bounded Linux pure-data execution, never an in-process candidate import.

The trusted bootstrap loads its runtime, chroots, drops credentials/capabilities,
closes ambient descriptors and installs a default-kill seccomp policy *before*
executing source. Unsupported hosts fail closed. This is a deliberately narrow
profile, not a general dependency installer or a new experiment controller.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256


@dataclass(frozen=True)
class SandboxLimits:
    source_bytes: int = 131072
    input_bytes: int = 4194304
    output_bytes: int = 1048576
    memory_bytes: int = 268435456
    cpu_seconds: int = 2
    wall_seconds: int = 5

    def __post_init__(self):
        bounds = {"source_bytes": (1, 1048576), "input_bytes": (1, 16777216),
                  "output_bytes": (128, 4194304), "memory_bytes": (67108864, 1073741824),
                  "cpu_seconds": (1, 30), "wall_seconds": (1, 60)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("invalid sandbox bound: " + name)
        if self.cpu_seconds > self.wall_seconds:
            raise ValueError("CPU limit cannot exceed wall limit")


@dataclass(frozen=True)
class ProductionSandboxLimits(SandboxLimits):
    """Explicit larger JSON output envelope for full-resolution evidence.

    Legacy defaults and bounds are unchanged. Any comparison must declare this
    distinct runtime identity; this does not grant model or image authority.
    """
    def __post_init__(self):
        values=asdict(self)
        if type(self.output_bytes) is not int or not 128 <= self.output_bytes <= 16777216:
            raise ValueError('invalid production JSON output bound')
        SandboxLimits(**{**values,'output_bytes':min(self.output_bytes,4194304)})


class LinuxPureDataRunner:
    """Host-configured runner. Neither a package nor model chooses this policy."""
    def __init__(self, *, python=None, seccomp_library="/lib/x86_64-linux-gnu/libseccomp.so.2", limits=None):
        self.python = Path(python or sys.executable).absolute()
        self.library = Path(seccomp_library).resolve(strict=True)
        self.bootstrap = Path(__file__).with_name("candidate_sandbox_child.py").resolve()
        self.limits = limits or SandboxLimits()
        self._identity = self._current_identity()

    def _current_identity(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64" or os.geteuid() != 0:
            raise ValueError("pure-data isolation requires Linux x86_64 and a trusted chroot bootstrap")
        return {"profile": "linux-chroot-uid-seccomp-pure-data-v1", "python": str(self.python),
                "python_sha256": file_sha256(self.python), "seccomp_library": str(self.library),
                "seccomp_sha256": file_sha256(self.library), "bootstrap_sha256": file_sha256(self.bootstrap),
                "runner_sha256": file_sha256(Path(__file__)), "limits": asdict(self.limits),
                "kernel": platform.release(), "uid": 65534, "gid": 65534,
                "preloaded_candidate_modules": ["base64", "hashlib", "json"]}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError("sandbox runtime identity changed")
        return dict(self._identity)

    def run(self, source, payload):
        """Only JSON and source bytes cross into the confined interpreter.

        No host store, path, credential, stdin stream, tool or model endpoint is
        exposed. Each result carries raw bounded streams and trusted identity;
        candidate JSON cannot report its own containment or qualification.
        """
        return self._run(source, payload)

    def _run(self, source, payload, *, host_context=None):
        identity = self.identity()
        if not isinstance(source, str) or len(source.encode()) > self.limits.source_bytes:
            raise ValueError("candidate source exceeds the approved bound")
        payload_bytes = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        if len(payload_bytes) > self.limits.input_bytes:
            raise ValueError("candidate input exceeds the approved bound")
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        binding = {"runtime_sha256": payload_sha256(identity), "source_sha256": source_hash,
                   "input_sha256": hashlib.sha256(payload_bytes).hexdigest()}
        if host_context is not None:
            binding["trusted_context_sha256"] = host_context["context_sha256"]
        read_fd, write_fd = os.pipe()
        started = time.monotonic()
        streams = {"stdout": bytearray(), "stderr": bytearray(), "control": bytearray()}
        observed = {key: 0 for key in streams}
        reason = None
        proc = None
        try:
            with tempfile.TemporaryDirectory(prefix="dataevolver-confined-") as empty_root:
                request = json.dumps({"source": source, "payload": json.loads(payload_bytes),
                    "limits": asdict(self.limits), "binding": binding,
                    "host_context": host_context}, allow_nan=False,separators=(',',':')).encode()
                proc = subprocess.Popen([str(self.python), "-I", "-S", str(self.bootstrap),
                    str(write_fd), empty_root, str(self.library)], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(write_fd,),
                    close_fds=True, start_new_session=True, env={"LANG": "C.UTF-8"}, cwd=empty_root)
                os.close(write_fd)
                write_fd = -1
                selector = selectors.DefaultSelector()
                sent = 0
                try:
                    for obj, name, event in ((proc.stdin, "input", selectors.EVENT_WRITE),
                            (proc.stdout, "stdout", selectors.EVENT_READ),
                            (proc.stderr, "stderr", selectors.EVENT_READ),
                            (read_fd, "control", selectors.EVENT_READ)):
                        os.set_blocking(obj if isinstance(obj, int) else obj.fileno(), False)
                        selector.register(obj, event, name)
                    while selector.get_map():
                        if time.monotonic() - started > self.limits.wall_seconds and reason is None:
                            reason = "WALL_LIMIT"
                        if reason is not None and proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                        for key, _ in selector.select(0.05):
                            obj, name = key.fileobj, key.data
                            fd = obj if isinstance(obj, int) else obj.fileno()
                            if name == "input":
                                try:
                                    sent += os.write(fd, request[sent:sent + 65536])
                                except BrokenPipeError:
                                    sent = len(request)
                                if sent == len(request):
                                    selector.unregister(obj); obj.close()
                            else:
                                block = os.read(fd, 65536)
                                if not block:
                                    selector.unregister(obj)
                                    continue
                                observed[name] += len(block)
                                bound = 8192 if name == "control" else self.limits.output_bytes
                                streams[name].extend(block[:max(0, bound - len(streams[name]))])
                                if observed[name] > bound and reason is None:
                                    reason = "OUTPUT_LIMIT"
                        if proc.poll() is not None:
                            # No child is permitted to fork or retain pipes.
                            if proc.stdin and not proc.stdin.closed:
                                try: selector.unregister(proc.stdin)
                                except KeyError: pass
                                proc.stdin.close()
                    proc.wait(timeout=1)
                finally:
                    selector.close()
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=2)
                    for stream in (proc.stdin, proc.stdout, proc.stderr):
                        if stream and not stream.closed: stream.close()
        finally:
            os.close(read_fd)
            if write_fd >= 0: os.close(write_fd)
        wall = time.monotonic() - started
        control = None
        try: control = json.loads(streams["control"])
        except (ValueError, UnicodeError): pass
        expected = {**binding, "chroot": True, "uid": 65534, "gid": 65534,
                    "no_new_privs": True, "dumpable": False, "seccomp_loaded": True,
                    "ambient_fds_closed": True}
        isolated = control == expected
        output = None
        if reason is None:
            if not isolated: reason = "ISOLATION_FAILED"
            elif proc.returncode != 0: reason = "CANDIDATE_PROCESS_FAILED"
            else:
                try:
                    output = json.loads(streams["stdout"], parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                    if not isinstance(output, dict): raise ValueError("result must be an object")
                except (ValueError, UnicodeError): reason = "INVALID_OUTPUT"
        self.identity()
        return {"schema_version": "dataevolver.isolated_candidate_execution.v2" if host_context else "dataevolver.isolated_candidate_execution.v1",
                "binding": binding, "runtime": identity, "containment": control,
                "isolation_verified": isolated, "status": reason or "COMPLETED",
                "return_code": proc.returncode, "observed_wall_seconds": wall,
                "observed_stream_bytes": observed, "stdout": bytes(streams["stdout"]).decode("utf-8", "replace"),
                "stderr": bytes(streams["stderr"]).decode("utf-8", "replace"), "output": output,
                "new_model_calls": 0}


class TrustedContextRunner(LinuxPureDataRunner):
    """Opt-in v2 interface; v1 run() never accepts payload-supplied bindings."""
    def _current_identity(self):
        identity = super()._current_identity()
        identity.update(profile="linux-chroot-uid-seccomp-trusted-context-v2",
            trust_contract_sha256=file_sha256(Path(__file__).with_name("candidate_trust.py")),
            context_scope="synthetic_authority_contract", configured_namespace="TRUSTED_REGISTRARS")
        return identity

    def run(self, source, payload):
        raise ValueError("the trusted-context profile requires an explicit host context")

    def run_with_context(self, source, payload, context):
        from .candidate_trust import CandidateTrustContext, canonical_digest
        if (not isinstance(context, CandidateTrustContext)
                or hashlib.sha256(source.encode()).hexdigest() != context.implementation_sha256
                or canonical_digest(payload) != context.payload_sha256):
            raise ValueError("host context does not bind these exact source and input bytes")
        return self._run(source, payload, host_context=context.execution_value())


class ProductionContextRunner(LinuxPureDataRunner):
    """Separate registered-data profile; synthetic contexts remain inadmissible."""
    def _current_identity(self):
        identity=super()._current_identity()
        identity.update(profile="linux-chroot-uid-seccomp-production-context-v1",
            trust_contract_sha256=file_sha256(Path(__file__).with_name("candidate_trust.py")),
            context_scope="registered_production_runtime",configured_namespace="TRUSTED_REGISTRARS")
        return identity

    def run(self, source, payload):
        raise ValueError("production execution requires an explicit registered context")

    def run_with_context(self, source, payload, context):
        from .candidate_trust import ProductionCandidateTrustContext, canonical_digest
        if (type(context) is not ProductionCandidateTrustContext
                or hashlib.sha256(source.encode()).hexdigest()!=context.implementation_sha256
                or canonical_digest(payload)!=context.payload_sha256
                or canonical_digest(payload['evidence']['source']['manifest'])
                   !=context.origin_bindings['manifest_sha256']):
            raise ValueError("production context does not bind these exact source and input bytes")
        return self._run(source,payload,host_context=context.execution_value())
