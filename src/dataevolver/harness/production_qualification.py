"""Production qualification bindings and committed runtime evidence.

The original design remains immutable. A separate execution contract binds real
registered validators, the host runtime and finite prior costs. Parsing or
inspecting this contract never dispatches work or adopts an operator version.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import ClassVar

from dataevolver.dataset.contracts import ArtifactRef, Contract, identifier, payload_sha256
from .operator_development import (
    ControllerQualificationAuthority, DevelopmentBudget, QualificationProtocol,
    _hash, _usage,
)
from .operator_lifecycle_authority import FrozenOperatorLifecycleAuthority


STAGES = ("engineering", "independent")
EVIDENCE_ROLES = frozenset({"input_closure", "matched_comparison", "quality_checks",
                           "cost_ledger", "native_audit", "independent_load"})
RUNTIME_DIGEST_KEYS = frozenset({"runner_sha256", "model_gateway_sha256", "models_sha256",
                                "r_state_sha256", "k_snapshot_sha256", "source_pool_sha256",
                                "online_budget_sha256", "trust_profile_sha256",
                                "registered_evidence_schema_sha256"})
RUNTIME_KEYS = RUNTIME_DIGEST_KEYS | {"trust_namespace"}


def _placeholder(operator_id):
    return any(part == "pending" or part.startswith("pending_")
               for part in operator_id.split("."))


def inspect_validator_bindings(stages, registry, evaluators):
    """Inspect host instances, never import a candidate or call its verifier."""
    if set(stages) != set(STAGES):
        raise ValueError("both production stages must be declared")
    result = {}
    declarations = {c["operator_id"]: c for c in registry.contracts()}
    for stage in STAGES:
        rule = stages[stage]
        gaps = []
        operator_id = rule["validator_operator_id"]
        if _placeholder(operator_id):
            gaps.append("PLACEHOLDER_VALIDATOR")
        try:
            op = registry.get(operator_id)
        except KeyError:
            gaps.append("VALIDATOR_NOT_REGISTERED")
        else:
            if op.spec.sha256() != rule["validator_spec_sha256"]:
                gaps.append("VALIDATOR_SPEC_MISMATCH")
            declaration = declarations.get(operator_id)
            if declaration is None or declaration["category"] != "evaluation":
                gaps.append("VALIDATOR_EVALUATION_CONTRACT_MISSING")
        evaluator = evaluators.get(stage)
        if evaluator is None:
            gaps.append("EVALUATOR_NOT_BOUND")
        else:
            # Reload the actual file; an old in-memory hash is insufficient.
            from .evaluator import EvaluatorContract
            actual = EvaluatorContract.from_path(evaluator.path)
            if (actual.sha256 != evaluator.sha256 or actual.payload != evaluator.payload
                    or actual.sha256 != rule["evaluator_sha256"]):
                gaps.append("EVALUATOR_MISMATCH")
        result[stage] = {"status": "BOUND" if not gaps else "NOT_READY", "gap_codes": gaps}
    return {"status": "BOUND" if all(not v["gap_codes"] for v in result.values()) else "NOT_READY",
            "stages": result, "qualification_performed": False,
            "production_execution_allowed": False}


@dataclass(frozen=True)
class ProductionQualificationExecution(Contract):
    schema: ClassVar[str] = "dataevolver.production_qualification_execution.v1"
    execution_id: str
    predecessor_authority_sha256: str
    predecessor_summary_sha256: str
    operator_id: str
    operator_version: str
    package_sha256: str
    design_protocol: dict
    matched_context: dict
    runtime: dict
    stages: dict
    budget: dict

    def __post_init__(self):
        for name in ("execution_id", "operator_id", "operator_version"):
            identifier(getattr(self, name))
        for name in ("predecessor_authority_sha256", "predecessor_summary_sha256", "package_sha256"):
            _hash(getattr(self, name))
        design = QualificationProtocol.from_dict(self.design_protocol)
        matched = ArtifactRef.from_dict(self.matched_context)
        if matched.sha256 != design.matched_context_sha256:
            raise ValueError("original matched context cannot be replaced")
        if not isinstance(self.runtime, dict) or set(self.runtime) != RUNTIME_KEYS:
            raise ValueError("production runtime must pin runner, Gateway, models, R/K, sources and online budget")
        if self.runtime["trust_namespace"] != "registered_production_runtime":
            raise ValueError("synthetic trust namespaces cannot authorize production qualification")
        for key in RUNTIME_DIGEST_KEYS:
            _hash(self.runtime[key])
        if not isinstance(self.stages, dict) or set(self.stages) != set(STAGES):
            raise ValueError("both production stages must be bound")
        for stage, rule in self.stages.items():
            if set(rule) != set(design.stages[stage]) | {"validator_contract_sha256"}:
                raise ValueError("production stage must bind spec, declaration, Evaluator and original checks")
            identifier(rule["validator_operator_id"])
            if _placeholder(rule["validator_operator_id"]):
                raise ValueError("placeholder validators cannot become production execution bindings")
            for key in ("validator_spec_sha256", "validator_contract_sha256", "evaluator_sha256"):
                _hash(rule[key])
            if rule["check_ids"] != design.stages[stage]["check_ids"]:
                raise ValueError("original production checks cannot be removed, changed or reordered")
        object.__setattr__(self, "design_protocol", design.to_dict())
        object.__setattr__(self, "matched_context", matched.to_dict())
        object.__setattr__(self, "runtime", copy.deepcopy(self.runtime))
        object.__setattr__(self, "stages", copy.deepcopy(self.stages))
        object.__setattr__(self, "budget", DevelopmentBudget.from_dict(self.budget).to_dict())

    def execution_protocol(self):
        """New stage identities; original design sources/checks remain exact."""
        design = QualificationProtocol.from_dict(self.design_protocol)
        stages = {s: {k: v for k, v in rule.items() if k != "validator_contract_sha256"}
                  for s, rule in self.stages.items()}
        return QualificationProtocol(self.execution_id, design.sources, stages, design.matched_context_sha256)

    def verify_predecessor(self, authority):
        if not isinstance(authority, FrozenOperatorLifecycleAuthority):
            raise ValueError("production lineage requires the host-bound original lifecycle reader")
        identity = authority.identity()
        summary = authority.read()
        budget = DevelopmentBudget.from_dict(self.budget)
        design = QualificationProtocol.from_dict(self.design_protocol)
        required = summary["production_requirements"]
        if (payload_sha256(identity) != self.predecessor_authority_sha256
                or payload_sha256(summary) != self.predecessor_summary_sha256
                or design.compute_hash() != required["design_protocol_sha256"]
                or design.matched_context_sha256 != required["matched_context_sha256"]
                or budget.resource_deadline_epoch != summary["resource_deadline_epoch"]
                or any(budget.prior_usage[k] < n for k, n in _usage(summary["lifecycle_total"]).items())):
            raise ValueError("production lineage, original design, deadline or prior costs changed")
        if self.operator_id != summary["operator_id"]:
            raise ValueError("the original requested operator cannot be renamed to evade version history")
        if (summary["current_status"] == "REVOKED"
                and (self.operator_id, self.operator_version) == (summary["operator_id"], summary["version"])):
            raise ValueError("a revoked ID/version cannot acquire production authority")
        return summary

    def inspect(self, *, authority, registry, evaluators, store, now):
        """Read-only admission diagnostics; BOUND is never execution permission."""
        summary = self.verify_predecessor(authority)
        if type(now) not in (float, int) or not math.isfinite(now) or now <= 0:
            raise ValueError("an explicit finite admission time is required")
        design = QualificationProtocol.from_dict(self.design_protocol)
        store.resolve(ArtifactRef.from_dict(self.matched_context))
        for source in design.sources:
            store.resolve(ArtifactRef.from_dict(source["artifact"]))
        bindings = inspect_validator_bindings(self.stages, registry, evaluators)
        declarations = {c["operator_id"]: c for c in registry.contracts()}
        gaps = []
        if bindings["status"] != "BOUND":
            gaps.append("PRODUCTION_STAGE_BINDINGS_NOT_READY")
        for stage, rule in self.stages.items():
            declaration = declarations.get(rule["validator_operator_id"])
            if declaration is not None and payload_sha256(declaration) != rule["validator_contract_sha256"]:
                gaps.append(stage.upper() + "_VALIDATOR_CONTRACT_CHANGED")
        budget = DevelopmentBudget.from_dict(self.budget)
        if now > budget.resource_deadline_epoch:
            gaps.append("RESOURCE_DEADLINE_PASSED")
        return {"schema_version": "dataevolver.production_qualification_admission.v1",
                "execution_contract_sha256": self.compute_hash(), "status": "BOUND" if not gaps else "NOT_READY",
                "gap_codes": gaps, "bindings": bindings,
                "prior_lifecycle_usage": summary["lifecycle_total"],
                "remaining_budget": {k: budget.limits[k] - budget.prior_usage[k] for k in budget.limits},
                "qualification_performed": False, "production_execution_allowed": False,
                "new_execution_or_budget_authorization": False}


class ControllerProductionQualificationAuthority:
    """Consume registered committed evidence through the existing Controller ABI.

    The host verifier remains responsible for native reconciliation, actual
    matched outcomes and complete costs. These references close the evidence
    inventory; they cannot turn a report or a fixture into that verifier.
    """
    def __init__(self, controller, call_id, *, execution, package, store, predecessor_authority):
        self.base = ControllerQualificationAuthority(controller, call_id)
        self.execution, self.package, self.store = execution, package, store
        self.predecessor_authority = predecessor_authority
        self.execution_sha256 = execution.compute_hash()
        self.identity_at_binding = self.base.identity()

    def verify(self, stage):
        c, execution = self.base.controller, self.execution
        if (stage not in STAGES or execution.compute_hash() != self.execution_sha256
                or self.base.identity() != self.identity_at_binding
                or self.package.compute_hash() != execution.package_sha256
                or c.goal.objective.get("production_qualification_execution") != self.execution_sha256):
            raise ValueError("production qualification was not prebound in this frozen Goal")
        execution.verify_predecessor(self.predecessor_authority)
        from .models import OperatorSpec
        from .operator_contracts import OperatorContract
        for raw in self.package.files:
            self.store.resolve(ArtifactRef.from_dict(raw))
        spec = OperatorSpec.from_dict(json.loads(self.store.resolve(
            ArtifactRef.from_dict(self.package.operator_spec)).read_text()))
        OperatorContract.from_dict(json.loads(self.store.resolve(
            ArtifactRef.from_dict(self.package.operator_contract)).read_text())).bound_to(spec)
        if (spec.operator_id, spec.version) != (execution.operator_id, execution.operator_version):
            raise ValueError("production target differs from the inert candidate package")
        protocol = execution.execution_protocol()
        bindings = inspect_validator_bindings(execution.stages, c.registry,
            {stage: c.evaluator.contract})
        if bindings["stages"][stage]["status"] != "BOUND":
            raise ValueError("production validator or Evaluator is not bound")
        declaration = {d["operator_id"]: d for d in c.registry.contracts()}[
            execution.stages[stage]["validator_operator_id"]]
        if payload_sha256(declaration) != execution.stages[stage]["validator_contract_sha256"]:
            raise ValueError("production validator declaration changed")
        proof = self.base.verify(stage, execution, self.package, protocol)
        event = next(e for e in c.store.events.read() if e.event_type == "ACTION_COMMITTED"
                     and e.payload.get("call_id") == self.base.call_id)
        result = event.payload["receipt"]["stage_result"]
        abi = result.get("production_evidence")
        if not isinstance(abi, dict) or set(abi) != {"schema_version", "execution_contract_sha256",
                "stage", "runtime", "source_artifacts", "artifacts"}:
            raise ValueError("registered production evidence ABI is missing or has unknown fields")
        expected_sources = [s["artifact"] for s in protocol.sources
                            if s["family_id"] in protocol.families(stage)]
        if (abi["schema_version"] != "dataevolver.registered_production_evidence.v1"
                or abi["execution_contract_sha256"] != self.execution_sha256
                or abi["stage"] != stage or abi["runtime"] != execution.runtime
                or abi["source_artifacts"] != expected_sources
                or not isinstance(abi["artifacts"], dict) or set(abi["artifacts"]) != EVIDENCE_ROLES):
            raise ValueError("production runtime, sources, evidence roles or execution identity changed")
        refs = abi["source_artifacts"] + list(abi["artifacts"].values()) + [execution.matched_context]
        hashes = event.payload["effect_verification"]["artifact_hashes"]
        candidate_stores = {raw["store"] for raw in self.package.files}
        for raw in refs:
            ref = ArtifactRef.from_dict(raw)
            if ref.store in candidate_stores:
                raise ValueError("production qualification evidence must be host-owned")
            path = self.store.resolve(ref)
            if hashes.get(str(path)) != ref.sha256:
                raise ValueError("production evidence was not covered by the committed host verifier")
        DevelopmentBudget.from_dict(execution.budget).check_usage([proof["usage"]])
        return {**proof, "execution_contract_sha256": self.execution_sha256,
                "registered_evidence_sha256": payload_sha256(abi), "stage": stage,
                "production_qualification_evidence_verified": True,
                "production_adoption_allowed": False, "new_model_calls": 0}


def assess_production_qualification(execution, authorities):
    """Aggregate distinct actual stages; adoption remains a separate commit."""
    if set(authorities) != set(STAGES) or any(
            not isinstance(a, ControllerProductionQualificationAuthority) for a in authorities.values()):
        raise ValueError("both host-bound production Controller authorities are required")
    proofs = {}
    for stage in STAGES:
        authority = authorities[stage]
        if authority.execution.compute_hash() != execution.compute_hash():
            raise ValueError("qualification stages refer to different execution contracts")
        proofs[stage] = authority.verify(stage)
    if len({p["campaign_id"] for p in proofs.values()}) != len(STAGES):
        raise ValueError("engineering and independent need separate frozen campaigns")
    total = DevelopmentBudget.from_dict(execution.budget).check_usage(p["usage"] for p in proofs.values())
    return {"execution_contract_sha256": execution.compute_hash(), "qualification_evidence": proofs,
            "production_engineering_qualification": "VERIFIED",
            "production_independent_qualification": "VERIFIED", "lifecycle_total": total,
            "production_adoption_allowed": False, "new_model_calls": 0,
            "method_improvement_verified": False, "training_benefit_verified": False}
