from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import (
    EffectVerification,
    EvidenceStatus,
    OperatorCall,
    OperatorReceipt,
    OperatorSpec,
    RecoveryResult,
    RecoveryStatus,
    StageStatus,
)
from .state_store import atomic_write_json


class InjectedOperatorCrash(RuntimeError):
    """FakeOperator-only abrupt failure after an effect is materialized."""


@runtime_checkable
class Operator(Protocol):
    spec: OperatorSpec

    def execute(self, call: OperatorCall) -> OperatorReceipt: ...

    def recover(self, call: OperatorCall) -> RecoveryResult: ...

    def verify_effect(self, call: OperatorCall, receipt: OperatorReceipt | None) -> EffectVerification: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_path(call: OperatorCall) -> Path:
    name = hashlib.sha256(call.idempotency_key.encode("utf-8")).hexdigest()[:32]
    return Path(call.work_dir) / ".receipts" / f"{name}.json"


def _load_receipt(path: Path) -> OperatorReceipt | None:
    if not path.exists():
        return None
    return OperatorReceipt.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _write_receipt(path: Path, receipt: OperatorReceipt) -> OperatorReceipt:
    stored = receipt.with_updates(receipt_path=str(path))
    atomic_write_json(path, stored)
    return stored


class OperatorRegistry:
    def __init__(self):
        self._operators: dict[str, Operator] = {}
        self._contracts: dict[str, dict] = {}
        self._steps: dict = {}

    def register_step(self, definition) -> None:
        from .operator_sdk import StepDefinition
        if not isinstance(definition, StepDefinition):
            raise TypeError("node factories require a StepDefinition")
        name = definition.spec.operator_id
        if name in self._operators or name in self._steps:
            raise ValueError(f"duplicate operator: {name}")
        self._steps[name] = definition

    def step(self, operator_id, version=None):
        try:
            definition = self._steps[operator_id]
        except KeyError as exc:
            raise KeyError(f"node factory is not registered: {operator_id}") from exc
        if version is not None and definition.spec.version != version:
            raise ValueError("node version differs from registered implementation")
        definition.contract.bound_to(definition.spec)
        return definition

    def step_specs(self):
        return tuple(self._steps[k].spec for k in sorted(self._steps))

    def register(self, operator: Operator, *, contract=None) -> None:
        if operator.spec.operator_id in self._operators or operator.spec.operator_id in self._steps:
            raise ValueError(f"duplicate operator: {operator.spec.operator_id}")
        if contract is None:
            contract = getattr(operator, "operator_contract", None)
        if contract is not None:
            from .operator_contracts import OperatorContract
            if not isinstance(contract, OperatorContract):
                raise TypeError("operator declaration must be an OperatorContract")
            contract.bound_to(operator.spec)
            self._contracts[operator.spec.operator_id] = contract.to_dict()
        self._operators[operator.spec.operator_id] = operator

    def get(self, operator_id: str) -> Operator:
        try:
            return self._operators[operator_id]
        except KeyError as exc:
            raise KeyError(f"operator is not registered: {operator_id}") from exc

    def specs(self) -> tuple[OperatorSpec, ...]:
        return tuple(self._operators[key].spec for key in sorted(self._operators))

    def contracts(self) -> tuple[dict, ...]:
        from .operator_contracts import OperatorContract
        return tuple(OperatorContract.from_dict(self._contracts[key]).bound_to(
            self.get(key).spec).to_dict() for key in sorted(self._contracts))

    def validate_call(self, call: OperatorCall) -> None:
        from .operator_contracts import OperatorContract
        operator = self.get(call.operator_id)
        if operator.spec.version != call.operator_version:
            raise ValueError("operator call version differs from registered implementation")
        if call.operator_id in self._contracts:
            OperatorContract.from_dict(self._contracts[call.operator_id]).bound_to(
                operator.spec).validate_parameters(call.parameters)

    def catalog(self) -> dict:
        contracts = {c['operator_id']: c for c in self.contracts()}
        result = {"schema_version": "dataevolver.operator_catalog.v1",
                "inference_performed": False, "qualification_performed": False,
                "operators": [{"spec": spec.to_dict(), "contract": contracts.get(spec.operator_id),
                               "declaration_status": "DECLARED" if spec.operator_id in contracts else "UNDECLARED",
                               "production_qualification": "NOT_ASSESSED"}
                              for spec in self.specs()]}
        result["operators"].extend({"spec": d.spec.to_dict(), "contract": d.contract.to_dict(),
            "binding_kind": "node_factory", "declaration_status": "DECLARED",
            "production_qualification": "NOT_ASSESSED"} for d in
            (self._steps[k] for k in sorted(self._steps)))
        return result

    def planner_catalog(self, allowed_operators) -> list[dict]:
        """Describe only authorized entries; executable argv and roots stay out."""
        allowed = set(allowed_operators)
        rows = []
        for row in self.catalog()["operators"]:
            spec = row["spec"]
            if spec["operator_id"] in allowed:
                rows.append({"operator_id": spec["operator_id"], "version": spec["version"],
                             "declaration_status": row["declaration_status"],
                             "contract": row["contract"],
                             "production_qualification": row["production_qualification"]})
        return rows


class CommandOperator:
    def __init__(self, spec: OperatorSpec):
        if isinstance(spec.argv, str) or not isinstance(spec.argv, (list, tuple)):
            raise TypeError("argv must be a sequence of strings, not a shell string")
        if not spec.argv or any(not isinstance(item, str) or not item for item in spec.argv):
            raise ValueError("argv must contain non-empty strings")
        executable = shutil.which(spec.argv[0]) or spec.argv[0]
        resolved = Path(executable).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"operator executable not found: {spec.argv[0]}")
        self.spec = spec.with_updates(argv=tuple(spec.argv), executable_realpath=str(resolved))

    def _validate_call(self, call: OperatorCall) -> None:
        if (call.operator_id, call.operator_version) != (self.spec.operator_id, self.spec.version):
            raise ValueError("operator call does not match registered spec")

    def _environment(self) -> dict[str, str]:
        platform_keys = {"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP"} if os.name == "nt" else {"HOME", "TMPDIR", "LANG", "LC_ALL"}
        keys = platform_keys | set(self.spec.allowed_env_keys)
        return {key: os.environ[key] for key in keys if key in os.environ}

    def execute(self, call: OperatorCall) -> OperatorReceipt:
        self._validate_call(call)
        prior = _load_receipt(_receipt_path(call))
        if prior is not None:
            return prior.with_updates(reused=True)
        started = datetime.now(timezone.utc).isoformat()
        try:
            completed = subprocess.run(
                list(self.spec.argv),
                cwd=self.spec.cwd or call.work_dir,
                env=self._environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.spec.timeout_seconds,
                check=False,
                shell=False,
            )
            status = StageStatus.SUCCESS if completed.returncode == 0 else StageStatus.FAILED
            return_code = completed.returncode
            output = completed.stdout[-4000:]
        except subprocess.TimeoutExpired as exc:
            status = StageStatus.TIMEOUT
            return_code = None
            output = str(exc)[-4000:]
        receipt = OperatorReceipt(
            call_id=call.call_id,
            operator_id=self.spec.operator_id,
            operator_version=self.spec.version,
            status=status,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            return_code=return_code,
            stdout_tail=output,
        )
        return _write_receipt(_receipt_path(call), receipt)

    def recover(self, call: OperatorCall) -> RecoveryResult:
        receipt = _load_receipt(_receipt_path(call))
        if receipt is not None:
            return RecoveryResult(RecoveryStatus.RECOVERED, receipt, "receipt found")
        verification = self.verify_effect(call, None)
        if verification.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, "effect exists without a recoverable receipt")
        if self.spec.retry_safe and verification.status is EvidenceStatus.UNOBSERVED:
            return RecoveryResult(RecoveryStatus.RETRY_SAFE, None, "no verified external effect")
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, verification.reason)

    def verify_effect(self, call: OperatorCall, receipt: OperatorReceipt | None) -> EffectVerification:
        if not self.spec.expected_output_specs:
            return EffectVerification(EvidenceStatus.UNOBSERVED, "operator declares no verifiable outputs")
        hashes: dict[str, str] = {}
        for output in self.spec.expected_output_specs:
            path = Path(call.work_dir) / output.path
            if not path.is_file() or path.stat().st_size < output.minimum_size:
                return EffectVerification(EvidenceStatus.UNOBSERVED, f"expected output is missing: {path}")
            actual = _sha256_file(path)
            if output.sha256 and actual != output.sha256:
                return EffectVerification(EvidenceStatus.CONTRADICTED, f"output hash mismatch: {path}")
            hashes[str(path)] = actual
        return EffectVerification(
            EvidenceStatus.VERIFIED,
            "all declared outputs verified",
            artifact_hashes=hashes,
            receipt_hash=receipt.sha256() if receipt else "",
        )


class FakeOperator:
    def __init__(
        self,
        spec: OperatorSpec,
        *,
        status: StageStatus,
        metrics: dict[str, float | bool | None],
        hard_constraints: dict[str, bool | None],
        failure_code: str = "",
        crash_after_effect: bool = False,
    ):
        self.spec = spec
        self.status = status
        self.metrics = dict(metrics)
        self.hard_constraints = dict(hard_constraints)
        self.failure_code = failure_code
        self.crash_after_effect = crash_after_effect

    def _effect_path(self, call: OperatorCall) -> Path:
        relative = self.spec.expected_output_specs[0].path if self.spec.expected_output_specs else "effect.json"
        return Path(call.work_dir) / relative

    def _receipt_from_effect(self, call: OperatorCall, *, reused: bool) -> OperatorReceipt:
        effect_path = self._effect_path(call)
        effect = json.loads(effect_path.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc).isoformat()
        return OperatorReceipt(
            call_id=call.call_id,
            operator_id=self.spec.operator_id,
            operator_version=self.spec.version,
            status=StageStatus(effect["status"]),
            started_at=str(effect["started_at"]),
            finished_at=now,
            return_code=0 if effect["status"] == StageStatus.SUCCESS.value else 1,
            output_artifacts=({"path": str(effect_path), "sha256": _sha256_file(effect_path)},),
            effect_hash=_sha256_file(effect_path),
            reused=reused,
            stage_result=dict(effect["stage_result"]),
        )

    def execute(self, call: OperatorCall) -> OperatorReceipt:
        if (call.operator_id, call.operator_version) != (self.spec.operator_id, self.spec.version):
            raise ValueError("operator call does not match registered spec")
        receipt_path = _receipt_path(call)
        prior = _load_receipt(receipt_path)
        if prior is not None:
            return prior.with_updates(reused=True)
        effect_path = self._effect_path(call)
        if effect_path.exists():
            return _write_receipt(receipt_path, self._receipt_from_effect(call, reused=True))
        started = datetime.now(timezone.utc).isoformat()
        effect = {
            "execution_count": 1,
            "status": self.status.value,
            "started_at": started,
            "stage_result": {
                "operator_id": self.spec.operator_id,
                "operator_version": self.spec.version,
                "status": self.status.value,
                "metrics": self.metrics,
                "hard_constraints": self.hard_constraints,
                "failure_code": self.failure_code,
                "matched_trial_key": call.matched_trial_key.to_dict() if call.matched_trial_key else None,
            },
        }
        atomic_write_json(effect_path, effect)
        if self.crash_after_effect:
            self.crash_after_effect = False
            raise InjectedOperatorCrash("injected crash after external effect")
        return _write_receipt(receipt_path, self._receipt_from_effect(call, reused=False))

    def recover(self, call: OperatorCall) -> RecoveryResult:
        receipt = _load_receipt(_receipt_path(call))
        if receipt is not None:
            return RecoveryResult(RecoveryStatus.RECOVERED, receipt, "receipt found")
        if self._effect_path(call).exists():
            recovered = _write_receipt(_receipt_path(call), self._receipt_from_effect(call, reused=True))
            return RecoveryResult(RecoveryStatus.RECOVERED, recovered, "effect marker recovered")
        return RecoveryResult(RecoveryStatus.RETRY_SAFE, None, "no effect marker")

    def verify_effect(self, call: OperatorCall, receipt: OperatorReceipt | None) -> EffectVerification:
        path = self._effect_path(call)
        if not path.exists():
            return EffectVerification(EvidenceStatus.UNOBSERVED, "effect marker is absent")
        actual = _sha256_file(path)
        if receipt is not None and receipt.effect_hash and receipt.effect_hash != actual:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "effect marker hash mismatch")
        return EffectVerification(
            EvidenceStatus.VERIFIED,
            "effect marker verified",
            artifact_hashes={str(path): actual},
            receipt_hash=receipt.sha256() if receipt else "",
        )
