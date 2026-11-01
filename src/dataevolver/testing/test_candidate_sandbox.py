"""Actual Linux kernel/resource negatives, not mocked containment claims."""
import json
import os
import platform
import signal

import pytest

from dataevolver.harness.candidate_sandbox import LinuxPureDataRunner, SandboxLimits


pytestmark = pytest.mark.skipif(platform.system() != "Linux" or os.geteuid() != 0,
                                reason="requires the explicit Linux chroot profile")


@pytest.fixture
def runner():
    return LinuxPureDataRunner()


def test_real_pure_data_and_repeated_identity(runner):
    code = 'import base64,hashlib,json\ndef resolve(p):\n return {"hash":hashlib.sha256(base64.b64decode(p["data"])).hexdigest(),"env":__import__("os").environ.copy()}\n'
    a, b = (runner.run(code, {"data": "YQ=="}) for _ in range(2))
    assert a["status"] == b["status"] == "COMPLETED"
    assert a["isolation_verified"] and b["isolation_verified"]
    assert a["output"] == b["output"] and a["output"]["env"] == {}
    assert a["binding"] == b["binding"]
    assert a["containment"]["uid"] == 65534
    assert a["new_model_calls"] == 0


@pytest.mark.parametrize("number,args", [
    (257, '-100,b"/etc/passwd",0,0'),  # path read
    (257, '-100,b"/created",65,384'),  # path write/create
    (41, '2,1,0'),                    # network socket
    (56, '0,0,0,0,0'),                # clone
    (59, 'b"/bin/true",0,0'),         # exec
    (101, '0,0,0,0'),                 # ptrace
    (62, '0,0'),                      # process signalling, harmless signal 0
    (310, '0,0,0,0,0,0'),            # process memory
    (425, '1,0'),                     # io_uring
    (321, '0,0,0'),                    # BPF
    (157, '38,0,0,0,0'),              # attempts to undo no_new_privs
    (161, 'b"/"'),                    # chroot
    (32, '1'),                        # duplicate ambient descriptor
    (0, '0,0,0'),                     # stdin/ambient descriptor reads
    (1, '0,b"x",1'),                 # writes beyond approved output pipes
    (165, '0,0,0,0,0'),               # mount
])
def test_actual_native_syscall_denials(runner, number, args):
    source = f'import ctypes\ndef resolve(p):\n ctypes.CDLL(None).syscall({number},{args})\n return {{"escape":True}}\n'
    result = runner.run(source, {})
    assert result["isolation_verified"]
    assert result["return_code"] == -signal.SIGSYS
    assert result["status"] == "CANDIDATE_PROCESS_FAILED"
    assert result["output"] is None


def test_runtime_cannot_open_files_even_via_python(runner, tmp_path):
    protected = tmp_path / "protected"
    protected.write_text("sentinel")
    source = 'def resolve(p):\n return {"secret":open(p["path"]).read()}\n'
    result = runner.run(source, {"path": str(protected)})
    assert result["return_code"] == -signal.SIGSYS
    assert protected.read_text() == "sentinel"


def test_control_descriptor_closed_before_candidate(runner):
    source = 'import os\ndef resolve(p):\n os.write(4,b"forged"); return {}\n'
    result = runner.run(source, {})
    assert result["isolation_verified"]
    assert result["status"] != "COMPLETED"


def test_cpu_or_wall_termination_is_bounded():
    runner = LinuxPureDataRunner(limits=SandboxLimits(cpu_seconds=1, wall_seconds=2))
    result = runner.run('def resolve(p):\n while True: pass\n', {})
    assert result["isolation_verified"]
    assert result["return_code"] == -signal.SIGKILL
    assert result["observed_wall_seconds"] < 3


def test_output_flood_is_bounded():
    runner = LinuxPureDataRunner(limits=SandboxLimits(output_bytes=4096))
    source = 'import os\ndef resolve(p):\n while True: os.write(1,b"x"*65536)\n'
    result = runner.run(source, {})
    assert result["isolation_verified"] and result["status"] == "OUTPUT_LIMIT"
    assert len(result["stdout"]) <= 4096 and result["output"] is None


def test_address_space_exhaustion_is_bounded(runner):
    result = runner.run('def resolve(p):\n return {"oversized":bytearray(1073741824)}\n', {})
    assert result["isolation_verified"] and result["return_code"] == 70


def test_invalid_source_never_claims_containment(runner):
    result = runner.run('def resolve(', {})
    assert result["status"] == "ISOLATION_FAILED" and not result["isolation_verified"]


def test_invalid_json_outputs_rejected(runner):
    for source in ('def resolve(p): return []', 'def resolve(p): return {"x":float("nan")}'):
        result = runner.run(source, {})
        assert result["isolation_verified"] and result["status"] != "COMPLETED"


def test_input_and_identity_bounds_fail_before_child(runner):
    with pytest.raises(ValueError): runner.run('x' * 131073, {})
    with pytest.raises(ValueError): runner.run('def resolve(p):return {}', {"x": float("nan")})
    runner._identity = {"changed": True}
    with pytest.raises(ValueError, match="identity changed"):
        runner.run('def resolve(p):return {}', {})


def test_missing_library_never_executes():
    with pytest.raises(FileNotFoundError):
        LinuxPureDataRunner(seccomp_library="/no-such-library")


def test_policy_bounds():
    with pytest.raises(ValueError): SandboxLimits(memory_bytes=2 ** 50)
    with pytest.raises(ValueError): SandboxLimits(cpu_seconds=6, wall_seconds=5)
