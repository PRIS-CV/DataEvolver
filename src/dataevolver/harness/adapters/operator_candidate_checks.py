"""Controller-owned, bounded isolated behavior checks of inert candidates.

This stage records real behavior, including rejected candidates. It does not
replace a design-only production qualification protocol, activate a candidate,
grant model authority or change the original package/development plan.
"""
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from ..candidate_sandbox import LinuxPureDataRunner
from ..operator_development import DevelopmentBudget, inspect_candidate
from ..models import EffectVerification, EvidenceStatus, ExpectedOutputSpec, RecoveryResult, RecoveryStatus
from ..operator_contracts import OperatorContract
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import CampaignLease, atomic_write_json
from .operator_development import CandidateInspectionOperator


class CandidateChecksOperator(CandidateInspectionOperator):
    def __init__(self, *, plan, package, store, request_authority, runner, check_protocol, output_root):
        if not isinstance(runner, LinuxPureDataRunner):
            raise ValueError("only the host-approved pure-data runner is supported")
        super().__init__(plan=plan, package=package, store=store,
                         request_authority=request_authority, output_root=output_root)
        self.runner = runner
        self.protocol = json.loads(json.dumps(check_protocol, allow_nan=False))
        self.runtime = runner.identity()
        self._check_protocol()
        self.spec = replace(self.spec, operator_id="evolution.check_operator_candidate",
            source_path=str(Path(__file__).resolve()), retry_safe=False,
            code_revision=payload_sha256({"inspector": self.spec.code_revision,
                "adapter": file_sha256(Path(__file__)), "runtime": self.runtime, "protocol": self.protocol}),
            expected_output_specs=(ExpectedOutputSpec("candidate_checks.json"),))
        self.operator_contract = OperatorContract(self.spec.operator_id, "1", self.spec.sha256(), "evaluation",
            {"candidate_package": {"type": "object"}, "check_protocol": {"type": "object"}},
            {"behavior_checks": {"type": "object"}},
            {"type": "object", "additionalProperties": False, "properties": {}},
            ("json", "source_code"), "pure_data", "verify_then_reuse", (
                "Checks execute only within the approved Linux pure-data confinement profile.",
                "Negative checks alone cannot establish useful or production-ready capability.",
                "No production qualification, promotion, model inference or image-quality claim is granted.",
                "An interrupted child without a durable result is ambiguous and is never rerun."))

    def _check_protocol(self):
        p = self.protocol
        if (set(p) != {"schema_version", "plan_sha256", "package_sha256", "runtime_sha256", "cases", "budget"}
                or p["schema_version"] != "dataevolver.candidate_behavior_checks.v1"
                or p["plan_sha256"] != self.plan.compute_hash()
                or p["package_sha256"] != self.package.compute_hash()
                or p["runtime_sha256"] != payload_sha256(self.runtime)):
            raise ValueError("behavior protocol binding changed")
        self.budget = DevelopmentBudget.from_dict(p["budget"])
        prior = self.budget.prior_usage
        if (self.budget.resource_deadline_epoch != self.plan.budget["resource_deadline_epoch"]
                or any(prior[key] < self.plan.budget["prior_usage"][key] for key in prior)
                or any(self.budget.limits[key] != prior[key] for key in
                       ("model_calls", "planner_calls", "image_experiments"))):
            raise ValueError("CPU checks must retain prior usage/deadline and add no model authority")
        cases = p["cases"]
        if not isinstance(cases, list) or not 1 <= len(cases) <= 16:
            raise ValueError("a bounded predeclared check set is required")
        ids = set()
        for case in cases:
            if (set(case) != {"case_id", "kind", "input", "expected_subset"}
                    or not isinstance(case["case_id"], str) or not 1 <= len(case["case_id"]) <= 128
                    or case["case_id"] in ids or case["kind"] not in {"positive", "negative"}
                    or not isinstance(case["expected_subset"], dict) or not case["expected_subset"]
                    or len(json.dumps(case["expected_subset"]).encode()) > 8192):
                raise ValueError("invalid behavior case")
            ids.add(case["case_id"])
            ref = ArtifactRef.from_dict(case["input"])
            if ref.store not in self._protected_input_stores():
                raise ValueError("behavior input must belong to a protected host store")
            self.store.resolve(ref)
        planned = {key: 0 for key in prior}
        planned.update(operator_calls=1, wall_seconds=len(cases) * self.runner.limits.wall_seconds)
        # Reserve captured bytes before any child runs, including JSON escaping
        # and bounded runtime/attestation metadata. A small protocol cannot use
        # the runner's larger per-process default to exceed its lifecycle cap.
        metadata_bound = len(json.dumps(self.runtime, ensure_ascii=False).encode()) + 16384
        planned["output_bytes"] = len(cases) * (12 * self.runner.limits.output_bytes + metadata_bound)
        self.budget.check_usage((planned,))

    def _protected_input_stores(self):
        return self.plan.protected_stores

    def _path(self, call):
        path = super()._validate(call).with_name("candidate_checks.json")
        for p in (path, path.parent / "check_intent.json", path.parent / "executions"):
            if p.is_symlink(): raise ValueError("behavior check output cannot be a symlink")
        return path

    def _binding(self, call):
        return {"call_sha256": payload_sha256(call.to_dict()), "operator_spec_sha256": self.spec.sha256(),
                "protocol_sha256": payload_sha256(self.protocol), "runtime_sha256": payload_sha256(self.runtime)}

    def _inputs(self):
        self.runner.identity()
        self._check_protocol()
        inspection = inspect_candidate(self.plan, self.package, self.store, self.request_authority)
        dependencies = json.loads(self.store.resolve(ArtifactRef.from_dict(self.package.dependency_manifest)).read_text())
        if dependencies["dependencies"]:
            raise ValueError("pure-data execution cannot install candidate dependencies")
        source = self.store.resolve(ArtifactRef.from_dict(self.package.implementation)).read_bytes().decode("utf-8")
        return inspection, source

    def _execution_binding(self, case, payload):
        import hashlib
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        return {"runtime_sha256": payload_sha256(self.runtime),
                "source_sha256": self.package.implementation["sha256"],
                "input_sha256": hashlib.sha256(encoded).hexdigest()}

    def _run_case(self, case, source, payload):
        return self.runner.run(source, payload)

    def _report(self, call):
        path = self._path(call)
        inspection, _ = self._inputs()
        intent = json.loads((path.parent / "check_intent.json").read_text())
        if intent["binding"] != self._binding(call): raise ValueError("behavior check intent changed")
        entries, usage = [], {key: 0 for key in self.budget.prior_usage}
        wall = 0.0
        artifacts = {}
        for index, case in enumerate(self.protocol["cases"]):
            file = path.parent / "executions" / (str(index) + ".json")
            if file.is_symlink(): raise ValueError("execution record cannot be a symlink")
            result = json.loads(file.read_text())
            payload = json.loads(self.store.resolve(ArtifactRef.from_dict(case["input"])).read_text())
            binding = self._execution_binding(case, payload)
            if result["binding"] != binding or result["runtime"] != self.runtime or result["new_model_calls"] != 0:
                raise ValueError("isolated execution binding changed")
            if result["status"] == "COMPLETED" and (not result["isolation_verified"]
                    or json.loads(result["stdout"]) != result["output"] or result["return_code"] != 0):
                raise ValueError("completed execution contradicts its captured streams")
            passed = (result["isolation_verified"] and result["status"] == "COMPLETED"
                      and all(result["output"].get(k) == v for k, v in case["expected_subset"].items()))
            entries.append({"case_id": case["case_id"], "kind": case["kind"], "passed": passed,
                            "execution_sha256": file_sha256(file), "status": result["status"]})
            artifacts[str(file)] = file_sha256(file)
            wall += result["observed_wall_seconds"]
            usage["output_bytes"] += file.stat().st_size
        usage.update(operator_calls=1, wall_seconds=math.ceil(wall))
        total = self.budget.check_usage((usage,))
        return {**self._binding(call), "at": intent["at"], "inspection": inspection,
                "cases": entries, "execution_artifacts": artifacts, "lifecycle_usage": usage,
                "lifecycle_total": total, "candidate_code_executed": True,
                "all_checks_passed": all(e["passed"] for e in entries),
                "positive_behavior_verified": any(e["passed"] and e["kind"] == "positive" for e in entries),
                "engineering_qualification": "NOT_ASSESSED", "independent_qualification": "NOT_ASSESSED",
                "promoted": False, "reused": False, "method_improvement_verified": False}

    def _receipt(self, call, path, reused):
        receipt = super()._receipt(call, path, reused)
        report = json.loads(path.read_text())
        return replace(receipt, stage_result={"operator_id": self.spec.operator_id, "operator_version": "1",
            "status": "SUCCESS", "metrics": {"behavior_checks_passed": float(report["all_checks_passed"]),
            "positive_behavior_verified": float(report["positive_behavior_verified"])},
            "hard_constraints": {"artifact_valid": True}, "evidence_refs": [str(path)]})

    def verify_effect(self, call, receipt):
        try:
            path = self._path(call)
            if not path.exists(): return EffectVerification(EvidenceStatus.UNOBSERVED, "behavior check report absent")
            if json.loads(path.read_text()) != self._report(call): raise ValueError("behavior evidence changed")
            if receipt is not None and (receipt.effect_hash != file_sha256(path)
                    or receipt.call_id != call.call_id or receipt.operator_id != self.spec.operator_id
                    or receipt.operator_version != "1" or receipt.stage_result != self._receipt(call, path, False).stage_result):
                raise ValueError("behavior receipt changed")
            return EffectVerification(EvidenceStatus.VERIFIED, "captured isolated behavior and immutable inputs verified",
                artifact_hashes={str(path): file_sha256(path), **json.loads(path.read_text())["execution_artifacts"]})
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def recover(self, call):
        path = self._path(call)
        prior = _load_receipt(_receipt_path(call))
        verification = self.verify_effect(call, prior)
        if verification.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(_receipt_path(call), self._receipt(call, path, True)))
        if verification.status is EvidenceStatus.UNOBSERVED and prior is None:
            if not (path.parent / "check_intent.json").exists():
                return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason="no execution intent or effect")
            try:
                report = self._report(call)  # Complete captured outputs only; no rerun.
                atomic_write_json(path, report)
                return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(_receipt_path(call), self._receipt(call, path, True)))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError): pass
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason="incomplete or contradicted isolated execution; no resend")

    def execute(self, call):
        path = self._path(call)
        self.output_root.mkdir(parents=True, exist_ok=True)
        with CampaignLease(self.output_root / ".candidate-checks.lease", ttl_seconds=300):
            journal = self.output_root / "single-check-call.json"
            binding = self._binding(call)
            if journal.exists() and json.loads(journal.read_text()) != binding:
                raise ValueError("one frozen protocol cannot reset execution allowance with another call")
            recovered = self.recover(call)
            if recovered.status is RecoveryStatus.RECOVERED: return recovered.receipt
            if recovered.status is not RecoveryStatus.RETRY_SAFE: raise ValueError(recovered.reason)
            _, source = self._inputs()
            atomic_write_json(journal, binding)
            atomic_write_json(path.parent / "check_intent.json", {"binding": binding, "at": datetime.now(timezone.utc).isoformat()})
            for index, case in enumerate(self.protocol["cases"]):
                payload = json.loads(self.store.resolve(ArtifactRef.from_dict(case["input"])).read_text())
                result = self._run_case(case, source, payload)
                atomic_write_json(path.parent / "executions" / (str(index) + ".json"), result)
            atomic_write_json(path, self._report(call))
            return _write_receipt(_receipt_path(call), self._receipt(call, path, False))
