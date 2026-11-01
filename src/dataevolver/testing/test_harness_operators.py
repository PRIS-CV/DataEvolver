import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dataevolver.harness.models import (
    EvidenceStatus,
    ExpectedOutputSpec,
    OperatorCall,
    OperatorSpec,
    StageStatus,
)
from dataevolver.harness.operators import CommandOperator, FakeOperator, InjectedOperatorCrash


def operator_spec(tmp_path, *, operator_id="fake.baseline", argv=()):
    return OperatorSpec(
        operator_id=operator_id,
        version="1",
        argv=argv,
        executable_realpath=str(sys.executable) if argv else "",
        cwd=str(tmp_path),
        timeout_seconds=10,
        retry_safe=True,
        allowed_env_keys=(),
        input_artifact_refs=(),
        expected_output_specs=(ExpectedOutputSpec(path="effect.json", minimum_size=1),),
        model_config_hash="model-a",
        code_revision="code-a",
        allowed_write_roots=(str(tmp_path),),
    )


def operator_call(tmp_path, operator_id="fake.baseline"):
    return OperatorCall(
        campaign_id="demo",
        call_id="call-1",
        idempotency_key="call-1",
        operator_id=operator_id,
        operator_version="1",
        round_index=0,
        work_dir=str(tmp_path),
        parameters={},
    )


def test_fake_operator_recovers_effect_after_crash_without_repeating_it(tmp_path):
    operator = FakeOperator(
        operator_spec(tmp_path),
        status=StageStatus.FAILED,
        metrics={"quality": 0.4},
        hard_constraints={"artifact_valid": True},
        failure_code="LOW_QUALITY",
        crash_after_effect=True,
    )
    call = operator_call(tmp_path)

    with pytest.raises(InjectedOperatorCrash):
        operator.execute(call)
    recovered = operator.recover(call)
    verification = operator.verify_effect(call, recovered.receipt)
    receipt = operator.execute(call)

    effect = json.loads((tmp_path / "effect.json").read_text(encoding="utf-8"))
    assert recovered.status == "RECOVERED"
    assert verification.status is EvidenceStatus.VERIFIED
    assert receipt.reused is True
    assert effect["execution_count"] == 1


def test_command_operator_rejects_shell_string_and_unlisted_environment(tmp_path, monkeypatch):
    with pytest.raises(TypeError, match="argv must be a sequence"):
        CommandOperator(operator_spec(tmp_path, argv="echo unsafe"))

    spec = operator_spec(
        tmp_path,
        operator_id="command.test",
        argv=(str(sys.executable), "-c", "import os; print(os.environ.get('HARNESS_ALLOWED')); print(os.environ.get('HARNESS_SECRET'))"),
    )
    spec = spec.with_updates(allowed_env_keys=("HARNESS_ALLOWED",))
    monkeypatch.setenv("HARNESS_ALLOWED", "yes")
    monkeypatch.setenv("HARNESS_SECRET", "no")
    receipt = CommandOperator(spec).execute(operator_call(tmp_path, operator_id="command.test"))

    assert receipt.return_code == 0
    assert receipt.stdout_tail.splitlines() == ["yes", "None"]


def test_subprocess_hard_exit_recovers_materialized_effect(tmp_path):
    spec = operator_spec(tmp_path)
    call = operator_call(tmp_path)
    script = tmp_path / "hard_exit.py"
    script.write_text(
        "import os\n"
        "from dataevolver.harness.models import OperatorCall, OperatorSpec, StageStatus\n"
        "from dataevolver.harness.operators import FakeOperator, InjectedOperatorCrash\n"
        f"spec = OperatorSpec.from_dict({spec.to_dict()!r})\n"
        f"call = OperatorCall.from_dict({call.to_dict()!r})\n"
        "operator = FakeOperator(spec, status=StageStatus.FAILED, metrics={'quality': 0.4}, "
        "hard_constraints={'artifact_valid': True}, failure_code='LOW_QUALITY', crash_after_effect=True)\n"
        "try:\n"
        "    operator.execute(call)\n"
        "except InjectedOperatorCrash:\n"
        "    os._exit(91)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    completed = subprocess.run([sys.executable, str(script)], env=env, check=False)

    operator = FakeOperator(
        spec,
        status=StageStatus.FAILED,
        metrics={"quality": 0.4},
        hard_constraints={"artifact_valid": True},
        failure_code="LOW_QUALITY",
    )
    recovered = operator.recover(call)
    verification = operator.verify_effect(call, recovered.receipt)

    assert completed.returncode == 91
    assert verification.status is EvidenceStatus.VERIFIED
    assert json.loads((tmp_path / "effect.json").read_text(encoding="utf-8"))["execution_count"] == 1
