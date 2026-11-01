from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import (
    EvaluationReport,
    EvidenceStatus,
    EvidenceValue,
    GoalSpec,
    StageResult,
    canonical_json,
)
from .operators import OperatorRegistry


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvaluatorContract:
    path: Path
    payload: dict[str, Any]
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> "EvaluatorContract":
        resolved = Path(path).resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "dataevolver.harness.evaluator.v1":
            raise ValueError(f"unsupported evaluator schema: {payload.get('schema_version')}")
        return cls(resolved, payload, sha256_file(resolved))

    @property
    def evaluator_id(self) -> str:
        return str(self.payload["evaluator_id"])

    @property
    def revision(self) -> str:
        return str(self.payload["revision"])


@runtime_checkable
class Evaluator(Protocol):
    contract: EvaluatorContract

    def evaluate(self, result: StageResult, *, effect_status: EvidenceStatus) -> EvaluationReport: ...


class DeterministicEvaluator:
    def __init__(self, contract: EvaluatorContract):
        self.contract = contract

    @staticmethod
    def _evidence(
        metric: str,
        source: dict[str, float | bool | None],
        *,
        required: bool,
        effect_status: EvidenceStatus,
    ) -> EvidenceValue:
        value = source.get(metric)
        if metric not in source or value is None:
            return EvidenceValue(metric, EvidenceStatus.UNOBSERVED, None, required, "metric was not observed")
        status = EvidenceStatus.VERIFIED if effect_status is EvidenceStatus.VERIFIED else effect_status
        return EvidenceValue(metric, status, value, required, "deterministic evaluator projection")

    def evaluate(self, result: StageResult, *, effect_status: EvidenceStatus) -> EvaluationReport:
        hard = {
            name: self._evidence(
                name,
                result.hard_constraints,
                required=bool(spec.get("required", True)),
                effect_status=effect_status,
            )
            for name, spec in self.contract.payload.get("hard_constraints", {}).items()
        }
        targets = {
            name: self._evidence(
                name,
                result.metrics,
                required=bool(spec.get("required", True)),
                effect_status=effect_status,
            )
            for name, spec in self.contract.payload.get("target_metrics", {}).items()
        }
        protected = {
            name: self._evidence(
                name,
                result.metrics,
                required=bool(spec.get("required", True)),
                effect_status=effect_status,
            )
            for name, spec in self.contract.payload.get("protected_metrics", {}).items()
        }
        failures = ()
        if result.failure_code:
            failures = ({"code": result.failure_code, "detail": result.failure_detail, "slice_key": "global"},)
        return EvaluationReport(
            evaluator_id=self.contract.evaluator_id,
            evaluator_revision=self.contract.revision,
            evaluator_hash=self.contract.sha256,
            hard_constraints=hard,
            target_metrics=targets,
            protected_metrics=protected,
            failure_slices=failures,
            matched_trial_key=result.matched_trial_key,
            evidence_refs=result.evidence_refs,
        )


@dataclass(frozen=True)
class DriftReport:
    valid: bool
    changed_fields: tuple[str, ...] = ()


class FrozenExecutionContract:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        goal: GoalSpec,
        registry: OperatorRegistry,
        planner_id: str,
        planner_config: dict[str, Any],
        evaluator_path: str | Path,
        schema_paths: tuple[str | Path, ...],
        execution_paths: tuple[str | Path, ...],
    ):
        self.repo_root = Path(repo_root).resolve()
        self.goal = goal
        self.registry = registry
        self.planner_id = planner_id
        self.planner_config = planner_config
        self.evaluator_path = Path(evaluator_path).resolve()
        self.schema_paths = tuple(Path(path).resolve() for path in schema_paths)
        operator_sources = tuple(Path(spec.source_path).resolve() for spec in registry.specs() if spec.source_path)
        self.execution_paths = tuple(dict.fromkeys((*map(Path, execution_paths), *operator_sources)))

    def _repo_revision(self) -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
                check=True,
                shell=False,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "UNAVAILABLE"

    def _path_hashes(self, paths: tuple[Path, ...]) -> dict[str, str]:
        return {str(path.resolve()): sha256_file(path) for path in sorted(paths, key=lambda item: str(item)) if path.is_file()}

    def _execution_diff_hash(self, inventory: dict[str, str]) -> str:
        relative_paths = []
        for path_text in inventory:
            path = Path(path_text)
            try:
                relative_paths.append(str(path.relative_to(self.repo_root)))
            except ValueError:
                continue
        diff = b""
        if relative_paths:
            try:
                completed = subprocess.run(
                    ["git", "diff", "--binary", "HEAD", "--", *relative_paths],
                    cwd=self.repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                    shell=False,
                )
                diff = completed.stdout
            except (OSError, subprocess.SubprocessError):
                pass
        return hashlib.sha256(diff + canonical_json(inventory).encode("utf-8")).hexdigest()

    def freeze(self) -> dict[str, Any]:
        evaluator = EvaluatorContract.from_path(self.evaluator_path)
        if evaluator.sha256 != self.goal.frozen_evaluator["sha256"]:
            raise ValueError("evaluator hash does not match GoalSpec")
        code_inventory = self._path_hashes(tuple(Path(path).resolve() for path in self.execution_paths))
        schema_hashes = self._path_hashes(self.schema_paths)
        schema_versions: dict[str, str] = {}
        for path in self.schema_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            schema_versions[str(path)] = str(payload.get("schema_version") or payload.get("$id") or "UNVERSIONED")
        specs = [spec.to_dict() for spec in self.registry.specs()]
        operator_sources = {
            spec.operator_id: sha256_file(spec.source_path)
            for spec in self.registry.specs()
            if spec.source_path and Path(spec.source_path).is_file()
        }
        promotion_policy = {
            key: evaluator.payload.get(key, {})
            for key in ("hard_constraints", "target_metrics", "protected_metrics")
        }
        frozen = {
            "schema_version": "dataevolver.harness.frozen_contract.v1",
            "repo_revision": self._repo_revision(),
            "execution_dirty_diff_hash": self._execution_diff_hash(code_inventory),
            "harness_code_inventory_hash": sha256_json(code_inventory),
            "schema_versions": schema_versions,
            "schema_hashes": schema_hashes,
            "operator_registry_hash": sha256_json(specs),
            "operator_versions": {spec.operator_id: spec.version for spec in self.registry.specs()},
            "operator_source_hashes": operator_sources,
            "planner_id": self.planner_id,
            "planner_config_hash": sha256_json(self.planner_config),
            "evaluator_id": evaluator.evaluator_id,
            "evaluator_path": str(evaluator.path),
            "evaluator_hash": evaluator.sha256,
            "promotion_policy_hash": sha256_json(promotion_policy),
            "budget_hash": self.goal.budgets.sha256(),
            "resource_policy_hash": sha256_json(self.goal.resource_policy),
            "goal_hash": self.goal.sha256(),
        }
        contracts = self.registry.contracts()
        if contracts:
            frozen["operator_contracts_hash"] = sha256_json(contracts)
        return frozen

    def verify(self, frozen: dict[str, Any]) -> DriftReport:
        try:
            current = self.freeze()
        except (OSError, ValueError, json.JSONDecodeError):
            return DriftReport(False, ("contract_unreadable",))
        changed = tuple(sorted(key for key in frozen.keys() | current.keys()
                               if key not in current or key not in frozen or current[key] != frozen[key]))
        return DriftReport(not changed, changed)
