"""Inert operator candidates and evidence-bound qualification.

These contracts grant no execution, registration, promotion or budget authority.
Artifact stores and evidence readers are supplied by the host, never a package.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar

from dataevolver.dataset.contracts import (
    ArtifactRef,
    ArtifactStore as ArtifactStore,
    Contract,
    file_sha256,
    identifier,
    positive_int,
)
from .capability_requests import validate_capability_request
from .models import CampaignState, EvidenceStatus, OperatorCall, OperatorReceipt, OperatorSpec, StageStatus, StageResult
from .operator_contracts import OperatorContract


USAGE_KEYS = frozenset({"operator_calls", "planner_calls", "model_calls",
                        "image_experiments", "output_bytes", "wall_seconds"})


def _hash(value):
    ArtifactRef("check", "hash", value)
    return value


def _strings(values, name, *, nonempty=True):
    if (not isinstance(values, (tuple, list)) or (nonempty and not values)
            or any(not isinstance(x, str) or not x for x in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be distinct nonempty strings")
    return tuple(values)


def _usage(value):
    if not isinstance(value, dict) or set(value) != USAGE_KEYS:
        raise ValueError("usage must declare every bounded resource")
    for key, count in value.items():
        positive_int(count, key, zero=True)
    return dict(value)


@dataclass(frozen=True)
class DevelopmentBudget(Contract):
    schema: ClassVar[str] = "dataevolver.operator_development_budget.v1"
    limits: dict[str, int]
    prior_usage: dict[str, int]
    resource_deadline_epoch: float

    def __post_init__(self):
        object.__setattr__(self, "limits", _usage(self.limits))
        object.__setattr__(self, "prior_usage", _usage(self.prior_usage))
        if (type(self.resource_deadline_epoch) not in (int, float)
                or not math.isfinite(self.resource_deadline_epoch) or self.resource_deadline_epoch <= 0):
            raise ValueError("an explicit finite resource deadline is required")
        self.check_usage(())

    def check_usage(self, usages):
        total = dict(self.prior_usage)
        for usage in usages:
            for key, amount in _usage(usage).items():
                total[key] += amount
        if any(total[key] > self.limits[key] for key in total):
            raise ValueError("development lifecycle total budget exceeded")
        return total


@dataclass(frozen=True)
class QualificationProtocol(Contract):
    schema: ClassVar[str] = "dataevolver.operator_qualification_protocol.v1"
    protocol_id: str
    sources: tuple[dict, ...]
    stages: dict[str, dict]
    matched_context_sha256: str

    def __post_init__(self):
        identifier(self.protocol_id)
        _hash(self.matched_context_sha256)
        if not isinstance(self.sources, (list, tuple)) or not self.sources:
            raise ValueError("qualification sources must be predeclared")
        families, hashes, splits = set(), set(), set()
        for source in self.sources:
            if set(source) != {"family_id", "split", "artifact"}:
                raise ValueError("qualification source has unknown fields")
            identifier(source["family_id"])
            ref = ArtifactRef.from_dict(source["artifact"])
            if source["split"] not in {"development", "regression", "independent", "heldout"}:
                raise ValueError("unknown qualification split")
            if source["family_id"] in families or ref.sha256 in hashes:
                raise ValueError("qualification source families and contents must be isolated")
            families.add(source["family_id"])
            hashes.add(ref.sha256)
            splits.add(source["split"])
        if splits != {"development", "regression", "independent", "heldout"} or sum(
                s["split"] == "independent" for s in self.sources) < 2:
            raise ValueError("development, regression, two independent families and heldout are required")
        if not isinstance(self.stages, dict) or set(self.stages) != {"engineering", "independent"}:
            raise ValueError("both qualification stages must be frozen")
        for stage in self.stages.values():
            if set(stage) != {"validator_operator_id", "validator_spec_sha256", "evaluator_sha256", "check_ids"}:
                raise ValueError("qualification stage must bind validator, evaluator and checks")
            identifier(stage["validator_operator_id"])
            _hash(stage["validator_spec_sha256"])
            _hash(stage["evaluator_sha256"])
            _strings(stage["check_ids"], "qualification checks")
        object.__setattr__(self, "sources", tuple(copy.deepcopy(self.sources)))
        object.__setattr__(self, "stages", copy.deepcopy(self.stages))

    def families(self, stage):
        splits = {"development", "regression"} if stage == "engineering" else {"independent"}
        return sorted(s["family_id"] for s in self.sources if s["split"] in splits)


@dataclass(frozen=True)
class OperatorDevelopmentPlan(Contract):
    schema: ClassVar[str] = "dataevolver.operator_development_plan.v1"
    plan_id: str
    capability_request: dict
    request_campaign_id: str
    request_frozen_sha256: str
    qualification_protocol: dict
    target_contract_sha256: str
    candidate_store: str
    protected_stores: tuple[str, ...]
    budget: dict

    def __post_init__(self):
        for value in (self.plan_id, self.request_campaign_id, self.candidate_store):
            identifier(value)
        for name in ("capability_request", "qualification_protocol"):
            ref = ArtifactRef.from_dict(getattr(self, name))
            object.__setattr__(self, name, ref.to_dict())
        _hash(self.request_frozen_sha256)
        _hash(self.target_contract_sha256)
        protected = _strings(self.protected_stores, "protected stores")
        for name in protected:
            identifier(name)
        if self.candidate_store in protected or any(
                getattr(self, name)["store"] not in protected
                for name in ("capability_request", "qualification_protocol")):
            raise ValueError("request and protocol must be protected from candidate writes")
        object.__setattr__(self, "protected_stores", protected)
        object.__setattr__(self, "budget", DevelopmentBudget.from_dict(self.budget).to_dict())


@dataclass(frozen=True)
class OperatorInterface(Contract):
    """A requested interface frozen before implementation bytes exist."""
    schema: ClassVar[str] = "dataevolver.operator_interface.v1"
    declaration: dict
    retry_safe: bool
    timeout_seconds: int

    def __post_init__(self):
        if not isinstance(self.declaration, dict) or "spec_sha256" in self.declaration:
            raise ValueError("an interface cannot predeclare unknown implementation bytes")
        contract = OperatorContract.from_dict({**self.declaration, "spec_sha256": "0" * 64})
        if type(self.retry_safe) is not bool:
            raise ValueError("retry authority must be an explicit boolean")
        positive_int(self.timeout_seconds, "operator timeout")
        if contract.recovery_semantics == "verify_then_retry" and not self.retry_safe:
            raise ValueError("interface cannot grant undeclared retry authority")
        value = contract.to_dict()
        value.pop("spec_sha256")
        object.__setattr__(self, "declaration", value)

    def bind(self, spec):
        if (spec.retry_safe != self.retry_safe or spec.timeout_seconds != self.timeout_seconds
                or spec.argv or spec.executable_realpath or spec.cwd or spec.allowed_env_keys
                or spec.allowed_write_roots or spec.input_artifact_refs or spec.expected_output_specs):
            raise ValueError("inert candidate spec cannot acquire execution authority")
        return OperatorContract.from_dict({**self.declaration, "spec_sha256": spec.sha256()}).bound_to(spec)


@dataclass(frozen=True)
class OperatorDevelopmentPlanV2(Contract):
    schema: ClassVar[str] = "dataevolver.operator_development_plan.v2"
    plan_id: str
    capability_request: dict
    request_campaign_id: str
    request_frozen_sha256: str
    qualification_protocol: dict
    target_interface: dict
    candidate_store: str
    protected_stores: tuple[str, ...]
    budget: dict

    def __post_init__(self):
        # Preserve v1 parsing and its exact-contract meaning. V2 freezes an
        # interface before generation, then binds the actual implementation.
        common = OperatorDevelopmentPlan(self.plan_id, self.capability_request, self.request_campaign_id,
            self.request_frozen_sha256, self.qualification_protocol, "0" * 64,
            self.candidate_store, self.protected_stores, self.budget)
        for name in ("capability_request", "qualification_protocol", "protected_stores", "budget"):
            object.__setattr__(self, name, getattr(common, name))
        ref = ArtifactRef.from_dict(self.target_interface)
        if ref.store not in self.protected_stores:
            raise ValueError("target interface must be protected before generation")
        object.__setattr__(self, "target_interface", ref.to_dict())


def read_development_plan(value):
    cls = OperatorDevelopmentPlanV2 if value.get("schema_version") == OperatorDevelopmentPlanV2.schema else OperatorDevelopmentPlan
    return cls.from_dict(value)


@dataclass(frozen=True)
class CandidatePackage(Contract):
    schema: ClassVar[str] = "dataevolver.operator_candidate_package.v1"
    candidate_id: str
    development_plan_sha256: str
    files: tuple[dict, ...]
    implementation: dict
    operator_spec: dict
    operator_contract: dict
    dependency_manifest: dict
    change_description: dict

    def __post_init__(self):
        identifier(self.candidate_id)
        _hash(self.development_plan_sha256)
        if not isinstance(self.files, (list, tuple)) or not self.files:
            raise ValueError("candidate files must be explicit")
        refs = tuple(ArtifactRef.from_dict(ref).to_dict() for ref in self.files)
        keys = [(ref["store"], ref["path"]) for ref in refs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate candidate file")
        for name in ("implementation", "operator_spec", "operator_contract", "dependency_manifest", "change_description"):
            ref = ArtifactRef.from_dict(getattr(self, name)).to_dict()
            if ref not in refs:
                raise ValueError("candidate metadata and entrypoint must belong to its file inventory")
            object.__setattr__(self, name, ref)
        object.__setattr__(self, "files", refs)


class ControllerRequestAuthority:
    """Host-supplied controller for the original request, never package input.

    Load a historical controller with its original frozen source. A caller using
    newer drifting code must fail rather than waive the original verification.
    """
    def __init__(self, controller):
        self.controller = controller

    def identity(self):
        c = self.controller
        return {"campaign_id": c.goal.campaign_id,
                "frozen_sha256": file_sha256(c.layout.frozen_contract_path)}

    def verify(self, plan, store):
        c = self.controller
        ref = ArtifactRef.from_dict(plan.capability_request)
        path = store.resolve(ref)
        if c.goal.campaign_id != plan.request_campaign_id:
            raise ValueError("capability request campaign changed")
        if file_sha256(c.layout.frozen_contract_path) != plan.request_frozen_sha256:
            raise ValueError("capability request frozen identity changed")
        valid, _ = c._frozen()
        state = CampaignState.from_dict(json.loads(c.layout.state_path.read_text()))
        if not valid or c.replay().to_dict() != state.to_dict() or state.terminal_reason != "CAPABILITY_REQUESTED":
            raise ValueError("capability request origin is not frozen, replayable and terminal")
        binding = {"path": str(path), "sha256": ref.sha256}
        if state.data.get("rsi_capability_request") != binding or not any(
                event.event_type == "RSI_CAPABILITY_REQUESTED"
                and event.payload["state_after"]["data"].get("rsi_capability_request") == binding
                for event in c.store.events.read()):
            raise ValueError("capability request was not committed by the Controller")
        request = json.loads(path.read_text())
        validate_capability_request(request["request"])
        if (request.get("proposal_actor") != "runtime_agent" or request.get("production_effects") != 0
                or request.get("decision", {}).get("action") != "REQUEST_CAPABILITY"):
            raise ValueError("not an inert runtime capability request")
        return {"campaign_id": c.goal.campaign_id, "request_sha256": ref.sha256,
                "frozen_sha256": plan.request_frozen_sha256}


class FrozenSourceRequestAuthority:
    """Verify a historical request using its own frozen Controller source.

    Only host-configured framework modules, interpreter and original env.sh are
    loaded. Candidate files are never placed on PYTHONPATH or executed. This is
    a read-only evidence transport, with no run/resume/tick or model request.
    """
    def __init__(self, *, config_root, python):
        self.config_root = Path(config_root).resolve()
        # Preserve a venv entrypoint: resolving its symlink can select the base
        # interpreter and silently drop the original environment's dependencies.
        self.python = Path(python).absolute()
        self.framework_src = Path(__file__).resolve().parents[2]
        self._identity = self._current_identity()

    def _current_identity(self):
        cfg = self.config_root
        campaign = json.loads((cfg / "goal.json").read_text())["campaign_id"]
        return {"campaign_id": campaign,
                "frozen_sha256": file_sha256(cfg / "campaigns" / campaign / "frozen_contract.json"),
                "env_sha256": file_sha256(cfg / "env.sh"),
                "python": str(self.python), "python_sha256": file_sha256(self.python),
                "reader_sha256": file_sha256(Path(__file__)),
                "declarations_sha256": file_sha256(Path(__file__).with_name("operator_contracts.py"))}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError("historical request reader identity changed")
        return dict(self._identity)

    def verify(self, plan, store):
        identity = self.identity()
        if (plan.request_campaign_id != identity["campaign_id"]
                or plan.request_frozen_sha256 != identity["frozen_sha256"]):
            raise ValueError("historical request plan identity changed")
        script = '''import importlib.util,json,sys
from pathlib import Path
from dataevolver.harness.composition import open_controller
cfg,src=map(Path,sys.argv[1:3])
c=open_controller(cfg/'campaigns',json.loads((cfg/'goal.json').read_text())['campaign_id'])
for name in ('operator_contracts','operator_development'):
 key='dataevolver.harness.'+name
 spec=importlib.util.spec_from_file_location(key,src/'dataevolver/harness'/ (name+'.py'))
 module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
from dataevolver.harness.operator_development import read_development_plan,ControllerRequestAuthority
from dataevolver.dataset.contracts import ArtifactStore
value=json.load(sys.stdin)
print(json.dumps(ControllerRequestAuthority(c).verify(read_development_plan(value['plan']),ArtifactStore(value['stores']))))
'''
        # Positional arguments only: no paths or model text interpolated into shell code.
        shell = ('source "$1/env.sh"; cd "$1/source"; export PYTHONPATH="$PWD/src" '
                 'PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 CUDA_VISIBLE_DEVICES=; '
                 'shift; exec "$@"')
        completed = subprocess.run(["bash", "-ec", shell, "frozen-request-reader", str(self.config_root),
            str(self.python), "-c", script, str(self.config_root), str(self.framework_src)],
            input=json.dumps({"plan": plan.to_dict(), "stores": {k: str(v) for k, v in store.roots.items()}}),
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0"})
        if completed.returncode != 0:
            raise ValueError("original frozen request verification failed: " + completed.stderr[-2000:])
        self.identity()
        value = json.loads(completed.stdout)
        if value != {"campaign_id": plan.request_campaign_id, "request_sha256": plan.capability_request["sha256"],
                     "frozen_sha256": plan.request_frozen_sha256}:
            raise ValueError("historical request verification returned a different identity")
        return value


def inspect_candidate(plan, package, store, request_authority):
    """Only read/hash/parse JSON. Never import, install or execute candidate code."""
    if package.development_plan_sha256 != plan.compute_hash():
        raise ValueError("candidate development plan changed")
    root = store.roots[plan.candidate_store]
    for name, other in store.roots.items():
        if name != plan.candidate_store and (root.is_relative_to(other) or other.is_relative_to(root)):
            raise ValueError("candidate store overlaps another artifact store")
    for name in plan.protected_stores:
        if name not in store.roots:
            raise ValueError("protected artifact store is not registered")
    origin = request_authority.verify(plan, store)
    protocol = QualificationProtocol.from_dict(json.loads(store.resolve(
        ArtifactRef.from_dict(plan.qualification_protocol)).read_text()))
    for source in protocol.sources:
        ref = ArtifactRef.from_dict(source["artifact"])
        if ref.store not in plan.protected_stores:
            raise ValueError("qualification sources must be protected")
        store.resolve(ref)
    inventory, total_bytes = {}, 0
    for raw in package.files:
        ref = ArtifactRef.from_dict(raw)
        if ref.store != plan.candidate_store:
            raise ValueError("candidate files must stay in their isolated store")
        lexical = root / ref.path
        if any(p.is_symlink() for p in (lexical, *lexical.parents)):
            raise ValueError("candidate symlinks are forbidden")
        path = store.resolve(ref)
        if path.stat().st_nlink != 1:
            raise ValueError("candidate hard links are forbidden")
        total_bytes += path.stat().st_size
        inventory[ref.path] = ref.sha256
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("candidate store contains links or special files")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(inventory):
        raise ValueError("candidate file inventory is not closed")
    budget = DevelopmentBudget.from_dict(plan.budget)
    if total_bytes + budget.prior_usage["output_bytes"] > budget.limits["output_bytes"]:
        raise ValueError("candidate exceeds remaining output byte budget")
    def read(name):
        return json.loads(store.resolve(ArtifactRef.from_dict(getattr(package, name))).read_text())
    spec_value = read("operator_spec")
    if set(spec_value) - {f.name for f in fields(OperatorSpec)}:
        raise ValueError("unknown candidate OperatorSpec fields")
    spec = OperatorSpec.from_dict(spec_value)
    contract = OperatorContract.from_dict(read("operator_contract")).bound_to(spec)
    if isinstance(plan, OperatorDevelopmentPlanV2):
        target = OperatorInterface.from_dict(json.loads(store.resolve(ArtifactRef.from_dict(plan.target_interface)).read_text()))
        if target.bind(spec).to_dict() != contract.to_dict() or spec.code_revision != package.implementation["sha256"]:
            raise ValueError("generated candidate implementation or interface changed")
    elif contract.compute_hash() != plan.target_contract_sha256:
        raise ValueError("candidate target contract changed")
    if spec.source_path != package.implementation["path"]:
        raise ValueError("candidate implementation or target contract changed")
    dependencies = read("dependency_manifest")
    if (not isinstance(dependencies, dict) or set(dependencies) != {"schema_version", "dependencies"}
            or dependencies["schema_version"] != "dataevolver.candidate_dependencies.v1"
            or not isinstance(dependencies["dependencies"], list)):
        raise ValueError("a versioned dependency inventory is required")
    for dep in dependencies["dependencies"]:
        if (not isinstance(dep, dict) or set(dep) != {"name", "version", "sha256"}
                or any(not isinstance(dep[k], str) or not dep[k] for k in ("name", "version"))):
            raise ValueError("dependencies must be pinned inert inventory entries")
        _hash(dep["sha256"])
    return {"schema_version": "dataevolver.candidate_inspection.v1", "status": "STRUCTURE_VERIFIED",
            "plan_sha256": plan.compute_hash(), "package_sha256": package.compute_hash(),
            "protocol_sha256": protocol.compute_hash(), "request_origin": origin,
            "files": inventory, "total_bytes": total_bytes, "candidate_code_executed": False,
            "engineering_qualification": "NOT_ASSESSED", "independent_qualification": "NOT_ASSESSED",
            "builder_verified": False, "promoted": False, "reused": False,
            "method_improvement_verified": False}


class ControllerQualificationAuthority:
    """Read committed effects of a host-approved frozen qualification operator.

    Its execute method is never called here. Its trusted verifier must validate
    the underlying checks, comparison, sources, costs and native receipts. A
    package-provided JSON report or operator cannot become this authority.
    """
    def __init__(self, controller, call_id):
        self.controller, self.call_id = controller, call_id

    def identity(self):
        return {"campaign_id": self.controller.goal.campaign_id,
                "frozen_sha256": file_sha256(self.controller.layout.frozen_contract_path), "call_id": self.call_id}

    def verify(self, stage, plan, package, protocol):
        c, rule = self.controller, protocol.stages[stage]
        valid, _ = c._frozen()
        state = CampaignState.from_dict(json.loads(c.layout.state_path.read_text()))
        if (not valid or c.replay().to_dict() != state.to_dict()
                or state.state.value != "COMPLETE" or state.data.get("pending_call")):
            raise ValueError("qualification campaign is not complete and frozen")
        expected = {"plan_sha256": plan.compute_hash(), "package_sha256": package.compute_hash(),
                    "protocol_sha256": protocol.compute_hash(), "stage": stage,
                    "matched_context_sha256": protocol.matched_context_sha256}
        if c.goal.objective.get("operator_qualification") != expected:
            raise ValueError("qualification was not prebound in the frozen Goal")
        events = c.store.events.read()
        planned = [e for e in events if e.event_type == "ACTION_PLANNED" and e.payload.get("call_id") == self.call_id]
        committed = [e for e in events if e.event_type == "ACTION_COMMITTED" and e.payload.get("call_id") == self.call_id]
        if len(planned) != 1 or len(committed) != 1 or planned[0].sequence >= committed[0].sequence:
            raise ValueError("qualification requires one committed Controller call")
        call = OperatorCall.from_dict(planned[0].payload["operator_call"])
        receipt = OperatorReceipt.from_dict(committed[0].payload["receipt"])
        validator = c.registry.get(rule["validator_operator_id"])
        if (call.operator_id != validator.spec.operator_id or call.operator_version != validator.spec.version
                or validator.spec.sha256() != rule["validator_spec_sha256"]
                or c.evaluator.contract.sha256 != rule["evaluator_sha256"]
                or (receipt.call_id, receipt.operator_id, receipt.operator_version) != (
                    call.call_id, call.operator_id, call.operator_version)
                or receipt.status is not StageStatus.SUCCESS or receipt.return_code != 0):
            raise ValueError("qualification validator/evaluator/receipt identity changed or failed")
        verification = validator.verify_effect(call, receipt)
        original = committed[0].payload["effect_verification"]
        if (verification.status is not EvidenceStatus.VERIFIED or original["status"] != "VERIFIED"
                or verification.artifact_hashes != original["artifact_hashes"]):
            raise ValueError("qualification effect no longer verifies")
        result = receipt.stage_result
        checks = result.get("check_results", {})
        if (result.get("qualification_binding") != expected
                or result.get("source_family_ids") != protocol.families(stage)
                or set(checks) != set(rule["check_ids"]) or any(value is not True for value in checks.values())):
            raise ValueError("qualification checks, source families or candidate binding changed")
        report = c.evaluator.evaluate(StageResult.from_dict(result), effect_status=verification.status)
        if not c._goal_satisfied(report):
            raise ValueError("frozen qualification evaluator rejected the result")
        usage = _usage(result.get("lifecycle_usage"))
        return {"campaign_id": c.goal.campaign_id, "call_id": self.call_id,
                "frozen_sha256": file_sha256(c.layout.frozen_contract_path),
                "receipt_sha256": receipt.sha256(), "evaluation_sha256": report.sha256(),
                "usage": usage}


def assess_candidate(plan, package, store, request_authority, qualification_authorities):
    inspection = inspect_candidate(plan, package, store, request_authority)
    protocol = QualificationProtocol.from_dict(json.loads(store.resolve(
        ArtifactRef.from_dict(plan.qualification_protocol)).read_text()))
    package_usage = {key: 0 for key in USAGE_KEYS}
    package_usage["output_bytes"] = inspection["total_bytes"]
    evidence, usages = {}, [package_usage]
    if set(qualification_authorities) - {"engineering", "independent"}:
        raise ValueError("unknown qualification authority")
    if "independent" in qualification_authorities and "engineering" not in qualification_authorities:
        raise ValueError("independent qualification requires engineering evidence first")
    for stage in ("engineering", "independent"):
        if stage not in qualification_authorities:
            continue
        proof = qualification_authorities[stage].verify(stage, plan, package, protocol)
        if any(proof["campaign_id"] == old["campaign_id"] for old in evidence.values()):
            raise ValueError("qualification stages need separate frozen campaigns")
        evidence[stage] = proof
        usages.append(proof["usage"])
    total = DevelopmentBudget.from_dict(plan.budget).check_usage(usages)
    return {**inspection, "qualification_evidence": evidence, "lifecycle_usage": total,
            "engineering_qualification": "VERIFIED" if "engineering" in evidence else "NOT_ASSESSED",
            "independent_qualification": "VERIFIED" if "independent" in evidence else "NOT_ASSESSED"}
