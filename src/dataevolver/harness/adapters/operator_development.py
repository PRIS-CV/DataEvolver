"""Registered read-only candidate inspection under EvolutionController.

Candidate programs remain inert. Approved qualification operators execute in
their own frozen campaigns; this operator only re-verifies their committed
evidence. It never registers, activates or promotes a candidate.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactStore, file_sha256, payload_sha256
from .. import operator_development as development
from ..models import (
    EffectVerification, EvidenceStatus, ExpectedOutputSpec, OperatorReceipt,
    OperatorSpec, RecoveryResult, RecoveryStatus, StageStatus,
)
from ..operator_contracts import OperatorContract
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import atomic_write_json


class CandidateInspectionOperator:
    def __init__(self, *, plan, package, store, request_authority, output_root,
                 qualification_authorities=None):
        self.plan = development.read_development_plan(plan.to_dict())
        self.package = development.CandidatePackage.from_dict(package.to_dict())
        self.store = ArtifactStore(store.roots)
        self.request_authority = request_authority
        self.authorities = dict(qualification_authorities or {})
        self.output_root = Path(output_root).resolve()
        for root in self.store.roots.values():
            if self.output_root.is_relative_to(root) or root.is_relative_to(self.output_root):
                raise ValueError("inspection output must be isolated from all artifact inputs")
        authority_bindings = {}
        for name, authority in {"request": request_authority, **self.authorities}.items():
            authority_bindings[name] = authority.identity()
        self.authority_bindings = authority_bindings
        self.spec = OperatorSpec(
            operator_id="evolution.inspect_operator_candidate", version="1", retry_safe=True,
            source_path=str(Path(__file__).resolve()),
            code_revision=payload_sha256({
                "adapter": file_sha256(Path(__file__)),
                "contracts": file_sha256(Path(development.__file__)),
                "plan": self.plan.to_dict(), "package": self.package.to_dict(),
                "stores": {key: str(root) for key, root in self.store.roots.items()},
                "authorities": authority_bindings,
            }), allowed_write_roots=(str(self.output_root),),
            expected_output_specs=(ExpectedOutputSpec("candidate_inspection.json"),))
        self.operator_contract = OperatorContract(
            operator_id=self.spec.operator_id, version="1", spec_sha256=self.spec.sha256(),
            category="evaluation", input_schemas={
                "development_plan": {"type": "object"}, "candidate_package": {"type": "object"}},
            output_schemas={"inspection": {"type": "object"}},
            parameter_schema={"type": "object", "additionalProperties": False, "properties": {}},
            modalities=("json", "source_code"), effect_semantics="pure_data",
            recovery_semantics="verify_then_retry", limitations=(
                "Structure verification does not execute or qualify candidate programs.",
                "Qualification requires host-approved frozen Controller execution and evaluator evidence.",
                "Filesystem boundaries are checked; this does not provide OS isolation.",
                "No candidate activation, promotion, reuse or method benefit follows from inspection.",))

    def _validate(self, call):
        if (call.operator_id, call.operator_version) != (self.spec.operator_id, self.spec.version):
            raise ValueError("candidate inspection call identity changed")
        self.operator_contract.validate_parameters(call.parameters)
        work = Path(call.work_dir)
        if any(p.is_symlink() for p in (work, *work.parents)):
            raise ValueError("inspection work directory cannot contain symlinks")
        work = work.resolve()
        if not work.is_relative_to(self.output_root) or work == self.output_root:
            raise ValueError("inspection call must write to a child of its approved output root")
        if any(path.is_symlink() for path in (work / "candidate_inspection.json", work / ".receipts", _receipt_path(call))):
            raise ValueError("inspection outputs and receipts cannot contain symlinks")
        return work / "candidate_inspection.json"

    def _result(self):
        for name, authority in {"request": self.request_authority, **self.authorities}.items():
            if authority.identity() != self.authority_bindings[name]:
                raise ValueError("candidate inspection authority identity changed")
        return development.assess_candidate(self.plan, self.package, self.store,
                                           self.request_authority, self.authorities)

    def verify_effect(self, call, receipt):
        try:
            path = self._validate(call)
            if not path.exists():
                return EffectVerification(EvidenceStatus.UNOBSERVED, "candidate inspection is absent")
            value = json.loads(path.read_text())
            if (value["call_sha256"] != payload_sha256(call.to_dict())
                    or value["operator_spec_sha256"] != self.spec.sha256()
                    or value["inspection"] != self._result()):
                raise ValueError("candidate inspection inputs or evidence changed")
            digest = file_sha256(path)
            if receipt is not None and (receipt.effect_hash != digest or receipt.call_id != call.call_id
                    or receipt.operator_id != self.spec.operator_id or receipt.operator_version != "1"
                    or receipt.stage_result != self._receipt(call, path, reused=False).stage_result):
                raise ValueError("candidate inspection receipt changed")
            return EffectVerification(EvidenceStatus.VERIFIED, "candidate inventory and original Controller evidence verified",
                                      artifact_hashes={str(path): digest})
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def _receipt(self, call, path, reused):
        value = json.loads(path.read_text())
        return OperatorReceipt(
            call_id=call.call_id, operator_id=self.spec.operator_id, operator_version="1",
            status=StageStatus.SUCCESS, return_code=0, started_at=value["at"], finished_at=value["at"],
            reused=reused, effect_hash=file_sha256(path),
            output_artifacts=({"path": str(path), "sha256": file_sha256(path)},),
            stage_result={"operator_id": self.spec.operator_id, "operator_version": "1", "status": "SUCCESS",
                          "metrics": {"structure_verified": 1.0},
                          "hard_constraints": {"artifact_valid": True},
                          "evidence_refs": [str(path)]})

    def recover(self, call):
        self._validate(call)
        prior = _load_receipt(_receipt_path(call))
        verification = self.verify_effect(call, prior)
        if verification.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(
                _receipt_path(call), self._receipt(call, self._validate(call), reused=True)))
        if verification.status is EvidenceStatus.UNOBSERVED and prior is None:
            return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason="no inspection output or receipt")
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason=verification.reason)

    def execute(self, call):
        path = self._validate(call)
        recovery = self.recover(call)
        if recovery.status is RecoveryStatus.RECOVERED:
            return recovery.receipt
        if recovery.status is not RecoveryStatus.RETRY_SAFE:
            raise ValueError(recovery.reason)
        result = self._result()
        atomic_write_json(path, {"at": datetime.now(timezone.utc).isoformat(),
                                "call_sha256": payload_sha256(call.to_dict()),
                                "operator_spec_sha256": self.spec.sha256(), "inspection": result})
        return _write_receipt(_receipt_path(call), self._receipt(call, path, reused=False))
