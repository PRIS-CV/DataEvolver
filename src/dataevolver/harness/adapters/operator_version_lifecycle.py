"""Registered scope-bound adoption/revocation; all decisions remain in Harness."""
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from ..models import (EffectVerification, EvidenceStatus, ExpectedOutputSpec, OperatorReceipt,
                      OperatorSpec, RecoveryResult, RecoveryStatus, StageStatus)
from ..operator_contracts import OperatorContract
from ..operator_development import DevelopmentBudget
from ..operator_lifecycle import qualification_snapshot, CommittedOperatorAdoptionAuthority
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import atomic_write_json


class OperatorVersionMutation:
    def __init__(self, *, catalog, expected_head, budget, output_root, package=None, store=None,
                 engineering=None, independent=None, runtime=None, adoption=None, reason=None, qualification_authority=None):
        self.catalog, self.expected_head = catalog, expected_head
        self.budget = DevelopmentBudget.from_dict(budget.to_dict())
        self.output_root = Path(output_root).resolve()
        self.adoption, self.reason = adoption, reason
        self.action = "REVOKED" if adoption is not None else "ADOPTED"
        if adoption is not None:
            if not isinstance(adoption, CommittedOperatorAdoptionAuthority) or not isinstance(reason, str) or not reason.strip():
                raise ValueError("revocation needs a committed exact adoption and reason")
            self._snapshot_reader = lambda: adoption.read()["snapshot"]
        else:
            if qualification_authority is not None:
                from ..catalog_candidate import CatalogQualificationAuthority
                from ..selection_candidate import SelectionQualificationAuthority
                from ..policy_qualification import PolicyQualificationAuthority
                from ..repair_qualification import RepairQualificationAuthority
                if not isinstance(qualification_authority, (CatalogQualificationAuthority, SelectionQualificationAuthority,
                                                             PolicyQualificationAuthority, RepairQualificationAuthority)):
                    raise TypeError('unknown restricted qualification authority')
                self._snapshot_reader = qualification_authority.read
            else:
                self._snapshot_reader = lambda: qualification_snapshot(package=package, store=store,
                    engineering=engineering, independent=independent, runtime=runtime)
            if any(catalog.root.is_relative_to(root) or root.is_relative_to(catalog.root)
                   for root in store.roots.values()):
                raise ValueError("operator catalog must be outside candidate and protected input stores")
        self.snapshot = self._snapshot_reader()
        if catalog.head() != expected_head:
            raise ValueError("catalog head was not frozen before mutation")
        if (self.output_root.is_relative_to(catalog.root) or catalog.root.is_relative_to(self.output_root)
                or any(self.budget.prior_usage[k] < v for k, v in self.snapshot["lifecycle_total"].items())
                or self.budget.resource_deadline_epoch != self.snapshot["resource_deadline_epoch"]
                or any(self.budget.limits[k] != self.budget.prior_usage[k]
                       for k in ("model_calls", "planner_calls", "image_experiments"))):
            raise ValueError("lifecycle mutation must retain costs/deadline and grant no model authority")
        reservation = {key: 0 for key in self.budget.prior_usage}
        reservation.update(operator_calls=1, wall_seconds=30, output_bytes=65536)
        self.budget.check_usage((reservation,))
        name = "adopt" if self.action == "ADOPTED" else "revoke"
        self.spec = OperatorSpec(operator_id=f"evolution.{name}_operator_version", version="1", retry_safe=False,
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                "adapter": file_sha256(Path(__file__)),
                "lifecycle": file_sha256(Path(__file__).parents[1]/"operator_lifecycle.py"),
                "snapshot": self.snapshot, "catalog": str(catalog.root), "head": expected_head,
                "budget": self.budget.to_dict(), "reason": reason,
                "adoption": adoption.identity() if adoption else None}),
            allowed_write_roots=(str(self.output_root), str(catalog.root)),
            expected_output_specs=(ExpectedOutputSpec("operator_version.json"),))
        self.operator_contract = OperatorContract(self.spec.operator_id, "1", self.spec.sha256(), "evolution",
            {"qualified_version": {"type": "object"}}, {"version_event": {"type": "object"}},
            {"type": "object", "additionalProperties": False, "properties": {}}, ("json", "source_code"),
            "pure_data", "verify_then_reuse", (
                "Adoption is limited to the declared qualified pure-data scope and exact runtime.",
                "Resolution additionally requires committed frozen Controller and Evaluator evidence.",
                "Revocation preserves history and prevents new execution; interrupted revocations fail closed.",
                "No image, production, K or method-improvement authority is granted."))

    def _path(self, call):
        self.operator_contract.validate_parameters(call.parameters)
        work = Path(call.work_dir).absolute()
        if ((call.operator_id, call.operator_version) != (self.spec.operator_id, "1")
                or not work.is_relative_to(self.output_root) or work == self.output_root
                or any(p.is_symlink() for p in (work, *work.parents, work/"operator_version.json",
                    work/"intent.json", work/".receipts", _receipt_path(call)))):
            raise ValueError("operator lifecycle call identity or output boundary changed")
        return work/"operator_version.json"

    def _owner(self, call):
        return {"campaign_id": call.campaign_id, "call_id": call.call_id,
                "call_sha256": payload_sha256(call.to_dict()), "operator_spec_sha256": self.spec.sha256()}

    def _event(self, call):
        matches = [x for x in self.catalog.events() if x["owner"] == self._owner(call)]
        if len(matches) != 1:
            raise ValueError("exact lifecycle effect is absent or ambiguous")
        event = matches[0]
        if (event["snapshot"] != self.snapshot or event["action"] != self.action
                or event["previous_sha256"] != self.expected_head or event["reason"] != self.reason
                or self._snapshot_reader() != self.snapshot):
            raise ValueError("operator lifecycle evidence changed")
        return event

    def _make_report(self, call, wall_seconds):
        event = self._event(call)
        intent = json.loads((self._path(call).parent/"intent.json").read_text())
        if intent["owner"] != self._owner(call):
            raise ValueError("operator mutation intent changed")
        usage = {key: 0 for key in self.budget.prior_usage}
        usage.update(operator_calls=1, wall_seconds=wall_seconds)
        report = {"at": intent["at"], "event": event, "lifecycle_usage": usage,
                  "lifecycle_total": self.budget.check_usage((usage,)), "new_model_calls": 0}
        # Account catalog and result payloads, excluding the Harness's own receipts.
        for _ in range(8):
            size = len(json.dumps(report, sort_keys=True, ensure_ascii=False).encode()) + len(json.dumps(event).encode())
            if usage["output_bytes"] == size:
                break
            usage["output_bytes"] = size
            report["lifecycle_total"] = self.budget.check_usage((usage,))
        return report

    def _receipt(self, call, path, reused):
        value = json.loads(path.read_text())
        metric = "operator_adopted" if self.action == "ADOPTED" else "operator_revoked"
        return OperatorReceipt(call_id=call.call_id, operator_id=self.spec.operator_id, operator_version="1",
            status=StageStatus.SUCCESS, return_code=0, started_at=value["at"], finished_at=value["at"],
            reused=reused, effect_hash=file_sha256(path),
            output_artifacts=({"path": str(path), "sha256": file_sha256(path)},),
            stage_result={"operator_id": self.spec.operator_id, "operator_version": "1", "status": "SUCCESS",
                "metrics": {metric: 1.0}, "hard_constraints": {"artifact_valid": True}, "evidence_refs": [str(path)]})

    def verify_effect(self, call, receipt):
        try:
            path = self._path(call)
            if not path.exists():
                return EffectVerification(EvidenceStatus.UNOBSERVED, "operator lifecycle report absent")
            report = json.loads(path.read_text())
            if report != self._make_report(call, report["lifecycle_usage"]["wall_seconds"]):
                raise ValueError("operator lifecycle report changed")
            if receipt is not None and (receipt.effect_hash != file_sha256(path)
                    or receipt.call_id != call.call_id or receipt.operator_id != self.spec.operator_id
                    or receipt.operator_version != "1" or receipt.stage_result != self._receipt(call, path, False).stage_result):
                raise ValueError("operator lifecycle receipt changed")
            event_path = self.catalog.root/f'{report["event"]["sequence"]:08d}.json'
            return EffectVerification(EvidenceStatus.VERIFIED, "immutable version event and qualifications verified",
                artifact_hashes={str(path): file_sha256(path), str(event_path): file_sha256(event_path)})
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def recover(self, call):
        path = self._path(call)
        receipt = _load_receipt(_receipt_path(call))
        check = self.verify_effect(call, receipt)
        if check.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(_receipt_path(call), self._receipt(call, path, True)))
        if check.status is EvidenceStatus.UNOBSERVED and receipt is None:
            if not (path.parent/"intent.json").exists() and not any(x["owner"] == self._owner(call) for x in self.catalog.events()):
                return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason="no lifecycle intent or effect")
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason="unfinished lifecycle mutation requires reconciliation")

    def execute(self, call):
        path = self._path(call)
        with self.catalog.lease():
            recovery = self.recover(call)
            if recovery.status is RecoveryStatus.RECOVERED:
                return recovery.receipt
            if recovery.status is not RecoveryStatus.RETRY_SAFE:
                raise ValueError(recovery.reason)
            started = time.monotonic()
            if self._snapshot_reader() != self.snapshot:
                raise ValueError("adoption qualifications changed before write")
            atomic_write_json(path.parent/"intent.json", {"owner": self._owner(call), "at": datetime.now(timezone.utc).isoformat()})
            self.catalog.append(action=self.action, snapshot=self.snapshot, owner=self._owner(call),
                                expected_head=self.expected_head, reason=self.reason)
            atomic_write_json(path, self._make_report(call, max(1, math.ceil(time.monotonic()-started))))
            return _write_receipt(_receipt_path(call), self._receipt(call, path, False))
