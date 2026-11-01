from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _reject_unknown(payload: dict[str, Any], allowed: set[str], model_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {model_name} fields: {', '.join(unknown)}")


class JsonModel:
    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def to_json(self) -> str:
        return canonical_json(self)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def with_updates(self, **changes: Any):
        return dataclasses.replace(self, **changes)


class EvidenceStatus(str, Enum):
    UNOBSERVED = "UNOBSERVED"
    OBSERVED = "OBSERVED"
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    STALE = "STALE"


class ControllerState(str, Enum):
    INIT = "INIT"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    OBSERVE = "OBSERVE"
    DIAGNOSE = "DIAGNOSE"
    PROPOSE = "PROPOSE"
    CANARY = "CANARY"
    EVALUATE = "EVALUATE"
    ADOPT_CANDIDATE = "ADOPT_CANDIDATE"
    ROLLBACK = "ROLLBACK"
    UPDATE = "UPDATE"
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"
    PAUSED = "PAUSED"
    PAUSED_CONTRACT_DRIFT = "PAUSED_CONTRACT_DRIFT"


class StageStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"


class RecoveryStatus(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    RECOVERED = "RECOVERED"
    RETRY_SAFE = "RETRY_SAFE"
    AMBIGUOUS = "AMBIGUOUS"


class InterventionLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


class CapabilityStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    PROMOTED = "PROMOTED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class PlannerDecisionKind(str, Enum):
    RUN_EXPERIMENT = "RUN_EXPERIMENT"
    ACCEPT_ASSET = "ACCEPT_ASSET"
    ABANDON_ASSET = "ABANDON_ASSET"
    STOP_CAMPAIGN = "STOP_CAMPAIGN"
    ABSTAIN = "ABSTAIN"


class ExperimentKind(str, Enum):
    PROMPT_REGENERATE = "PROMPT_REGENERATE"
    RESEGMENT_RECONSTRUCT = "RESEGMENT_RECONSTRUCT"
    RECONSTRUCT_RETRY = "RECONSTRUCT_RETRY"
    SCENE_REFINE = "SCENE_REFINE"


@dataclass(frozen=True)
class EvidenceValue(JsonModel):
    metric: str
    status: EvidenceStatus
    value: float | bool | None
    required: bool = True
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceValue":
        return cls(
            metric=str(payload["metric"]),
            status=EvidenceStatus(payload["status"]),
            value=payload.get("value"),
            required=bool(payload.get("required", True)),
            reason=str(payload.get("reason") or ""),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class BudgetSpec(JsonModel):
    max_rounds: int
    max_operator_calls: int
    max_experiments: int
    max_retries: int
    max_wall_seconds: int
    max_planner_calls: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BudgetSpec":
        values = [
            int(payload[name])
            for name in (
                "max_rounds",
                "max_operator_calls",
                "max_experiments",
                "max_retries",
                "max_wall_seconds",
            )
        ]
        planner_calls = int(payload.get("max_planner_calls", 0))
        if any(value < 0 for value in (*values, planner_calls)):
            raise ValueError("budget values must be non-negative")
        return cls(*values, max_planner_calls=planner_calls)


@dataclass(frozen=True)
class MatchedTrialKey(JsonModel):
    input_artifact_hash: str
    parent_artifact_hash: str
    seed: int
    operator_id: str
    operator_version: str
    model_config_hash: str
    preprocessing_hash: str
    evaluator_hash: str
    environment_contract_hash: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MatchedTrialKey":
        return cls(
            input_artifact_hash=str(payload["input_artifact_hash"]),
            parent_artifact_hash=str(payload["parent_artifact_hash"]),
            seed=int(payload["seed"]),
            operator_id=str(payload["operator_id"]),
            operator_version=str(payload["operator_version"]),
            model_config_hash=str(payload["model_config_hash"]),
            preprocessing_hash=str(payload["preprocessing_hash"]),
            evaluator_hash=str(payload["evaluator_hash"]),
            environment_contract_hash=str(payload["environment_contract_hash"]),
        )

    def digest(self) -> str:
        return self.sha256()


@dataclass(frozen=True)
class ExpectedOutputSpec(JsonModel):
    path: str
    minimum_size: int = 1
    sha256: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExpectedOutputSpec":
        return cls(
            path=str(payload["path"]),
            minimum_size=int(payload.get("minimum_size", 1)),
            sha256=str(payload.get("sha256") or ""),
        )


@dataclass(frozen=True)
class OperatorSpec(JsonModel):
    operator_id: str
    version: str
    argv: tuple[str, ...] | str = ()
    executable_realpath: str = ""
    cwd: str = ""
    timeout_seconds: int = 300
    retry_safe: bool = False
    allowed_env_keys: tuple[str, ...] = ()
    input_artifact_refs: tuple[dict[str, Any], ...] = ()
    expected_output_specs: tuple[ExpectedOutputSpec, ...] = ()
    model_config_hash: str = ""
    code_revision: str = ""
    allowed_write_roots: tuple[str, ...] = ()
    source_path: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperatorSpec":
        raw_argv = payload.get("argv", [])
        argv: tuple[str, ...] | str = raw_argv if isinstance(raw_argv, str) else tuple(str(item) for item in raw_argv)
        return cls(
            operator_id=str(payload["operator_id"]),
            version=str(payload["version"]),
            argv=argv,
            executable_realpath=str(payload.get("executable_realpath") or ""),
            cwd=str(payload.get("cwd") or ""),
            timeout_seconds=int(payload.get("timeout_seconds", 300)),
            retry_safe=bool(payload.get("retry_safe", False)),
            allowed_env_keys=tuple(str(item) for item in payload.get("allowed_env_keys", [])),
            input_artifact_refs=tuple(dict(item) for item in payload.get("input_artifact_refs", [])),
            expected_output_specs=tuple(
                ExpectedOutputSpec.from_dict(item) for item in payload.get("expected_output_specs", [])
            ),
            model_config_hash=str(payload.get("model_config_hash") or ""),
            code_revision=str(payload.get("code_revision") or ""),
            allowed_write_roots=tuple(str(item) for item in payload.get("allowed_write_roots", [])),
            source_path=str(payload.get("source_path") or ""),
        )


@dataclass(frozen=True)
class OperatorCall(JsonModel):
    campaign_id: str
    call_id: str
    idempotency_key: str
    operator_id: str
    operator_version: str
    round_index: int
    work_dir: str
    parameters: dict[str, Any] = field(default_factory=dict)
    input_artifact_refs: tuple[dict[str, Any], ...] = ()
    expected_output_specs: tuple[ExpectedOutputSpec, ...] = ()
    matched_trial_key: MatchedTrialKey | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperatorCall":
        key = payload.get("matched_trial_key")
        return cls(
            campaign_id=str(payload["campaign_id"]),
            call_id=str(payload["call_id"]),
            idempotency_key=str(payload["idempotency_key"]),
            operator_id=str(payload["operator_id"]),
            operator_version=str(payload["operator_version"]),
            round_index=int(payload["round_index"]),
            work_dir=str(payload["work_dir"]),
            parameters=dict(payload.get("parameters") or {}),
            input_artifact_refs=tuple(dict(item) for item in payload.get("input_artifact_refs", [])),
            expected_output_specs=tuple(
                ExpectedOutputSpec.from_dict(item) for item in payload.get("expected_output_specs", [])
            ),
            matched_trial_key=MatchedTrialKey.from_dict(key) if key else None,
        )


@dataclass(frozen=True)
class OperatorReceipt(JsonModel):
    call_id: str
    operator_id: str
    operator_version: str
    status: StageStatus
    started_at: str
    finished_at: str
    return_code: int | None = None
    stdout_tail: str = ""
    output_artifacts: tuple[dict[str, Any], ...] = ()
    effect_hash: str = ""
    receipt_path: str = ""
    reused: bool = False
    stage_result: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperatorReceipt":
        return cls(
            call_id=str(payload["call_id"]),
            operator_id=str(payload["operator_id"]),
            operator_version=str(payload["operator_version"]),
            status=StageStatus(payload["status"]),
            started_at=str(payload["started_at"]),
            finished_at=str(payload["finished_at"]),
            return_code=payload.get("return_code"),
            stdout_tail=str(payload.get("stdout_tail") or ""),
            output_artifacts=tuple(dict(item) for item in payload.get("output_artifacts", [])),
            effect_hash=str(payload.get("effect_hash") or ""),
            receipt_path=str(payload.get("receipt_path") or ""),
            reused=bool(payload.get("reused", False)),
            stage_result=dict(payload.get("stage_result") or {}),
        )


@dataclass(frozen=True)
class RecoveryResult(JsonModel):
    status: RecoveryStatus
    receipt: OperatorReceipt | None = None
    reason: str = ""


@dataclass(frozen=True)
class EffectVerification(JsonModel):
    status: EvidenceStatus
    reason: str = ""
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    receipt_hash: str = ""


@dataclass(frozen=True)
class CampaignState(JsonModel):
    campaign_id: str
    state: ControllerState = ControllerState.INIT
    previous_state: ControllerState | None = None
    last_event_seq: int = 0
    round_index: int = 0
    experiment_index: int = 0
    operator_calls: int = 0
    planner_calls: int = 0
    experiments_used: int = 0
    retries_used: int = 0
    started_at: str = ""
    updated_at: str = ""
    terminal_reason: str = ""
    idempotency_key: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CampaignState":
        previous = payload.get("previous_state")
        return cls(
            campaign_id=str(payload["campaign_id"]),
            state=ControllerState(payload.get("state", "INIT")),
            previous_state=ControllerState(previous) if previous else None,
            last_event_seq=int(payload.get("last_event_seq", 0)),
            round_index=int(payload.get("round_index", 0)),
            experiment_index=int(payload.get("experiment_index", 0)),
            operator_calls=int(payload.get("operator_calls", 0)),
            planner_calls=int(payload.get("planner_calls", 0)),
            experiments_used=int(payload.get("experiments_used", 0)),
            retries_used=int(payload.get("retries_used", 0)),
            started_at=str(payload.get("started_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            terminal_reason=str(payload.get("terminal_reason") or ""),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            data=dict(payload.get("data") or {}),
        )


@dataclass(frozen=True)
class StageResult(JsonModel):
    operator_id: str
    operator_version: str
    status: StageStatus
    metrics: dict[str, float | bool | None] = field(default_factory=dict)
    hard_constraints: dict[str, bool | None] = field(default_factory=dict)
    failure_code: str = ""
    failure_detail: str = ""
    input_artifacts: tuple[dict[str, Any], ...] = ()
    output_artifacts: tuple[dict[str, Any], ...] = ()
    evidence_refs: tuple[str, ...] = ()
    matched_trial_key: MatchedTrialKey | None = None
    retryable: bool = False
    cost: dict[str, float] = field(default_factory=dict)
    stdout_tail: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageResult":
        key = payload.get("matched_trial_key")
        return cls(
            operator_id=str(payload["operator_id"]),
            operator_version=str(payload["operator_version"]),
            status=StageStatus(payload["status"]),
            metrics=dict(payload.get("metrics") or {}),
            hard_constraints=dict(payload.get("hard_constraints") or {}),
            failure_code=str(payload.get("failure_code") or ""),
            failure_detail=str(payload.get("failure_detail") or ""),
            input_artifacts=tuple(dict(item) for item in payload.get("input_artifacts", [])),
            output_artifacts=tuple(dict(item) for item in payload.get("output_artifacts", [])),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", [])),
            matched_trial_key=MatchedTrialKey.from_dict(key) if key else None,
            retryable=bool(payload.get("retryable", False)),
            cost={str(name): float(value) for name, value in (payload.get("cost") or {}).items()},
            stdout_tail=str(payload.get("stdout_tail") or ""),
        )


@dataclass(frozen=True)
class EvaluationReport(JsonModel):
    evaluator_id: str
    evaluator_revision: str
    evaluator_hash: str
    hard_constraints: dict[str, EvidenceValue]
    target_metrics: dict[str, EvidenceValue]
    protected_metrics: dict[str, EvidenceValue]
    failure_slices: tuple[dict[str, Any], ...] = ()
    matched_trial_key: MatchedTrialKey | None = None
    evidence_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationReport":
        def values(name: str) -> dict[str, EvidenceValue]:
            return {key: EvidenceValue.from_dict(value) for key, value in (payload.get(name) or {}).items()}

        key = payload.get("matched_trial_key")
        return cls(
            evaluator_id=str(payload["evaluator_id"]),
            evaluator_revision=str(payload["evaluator_revision"]),
            evaluator_hash=str(payload["evaluator_hash"]),
            hard_constraints=values("hard_constraints"),
            target_metrics=values("target_metrics"),
            protected_metrics=values("protected_metrics"),
            failure_slices=tuple(dict(item) for item in payload.get("failure_slices", [])),
            matched_trial_key=MatchedTrialKey.from_dict(key) if key else None,
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", [])),
        )

    def required_evidence(self) -> tuple[EvidenceValue, ...]:
        values = (*self.hard_constraints.values(), *self.target_metrics.values(), *self.protected_metrics.values())
        return tuple(value for value in values if value.required)

    def all_required_verified(self) -> bool:
        return all(value.status is EvidenceStatus.VERIFIED for value in self.required_evidence())


@dataclass(frozen=True)
class FailureSignature(JsonModel):
    code: str
    slice_key: str = "global"
    detail: str = ""
    evidence_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailureSignature":
        return cls(
            code=str(payload["code"]),
            slice_key=str(payload.get("slice_key") or "global"),
            detail=str(payload.get("detail") or ""),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class StrategyPatch(JsonModel):
    prompt_text: str | None = None
    negative_prompt_append: tuple[str, ...] = ()
    t2i_seed: int | None = None
    reconstruction_seed: int | None = None
    force_resegment: bool = False
    scene_actions: tuple[str, ...] = ()
    restart_stage: str = "render"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyPatch":
        allowed = {
            "prompt_text", "negative_prompt_append", "t2i_seed", "reconstruction_seed",
            "force_resegment", "scene_actions", "restart_stage",
        }
        _reject_unknown(payload, allowed, "StrategyPatch")
        prompt = payload.get("prompt_text")
        if prompt is not None:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("StrategyPatch prompt_text must be a non-empty string")
            if len(prompt) > 2048:
                raise ValueError("StrategyPatch prompt_text exceeds 2048 characters")
            prompt = prompt.strip()
        negative = payload.get("negative_prompt_append") or ()
        if not isinstance(negative, (list, tuple)) or len(negative) > 8:
            raise ValueError("StrategyPatch negative_prompt_append must contain at most 8 strings")
        if any(not isinstance(item, str) or not item.strip() or len(item) > 256 for item in negative):
            raise ValueError("StrategyPatch negative prompts must be non-empty strings up to 256 characters")
        actions = payload.get("scene_actions") or ()
        if not isinstance(actions, (list, tuple)) or len(actions) > 2:
            raise ValueError("StrategyPatch scene_actions must contain at most 2 actions")
        if any(not isinstance(item, str) or not item for item in actions):
            raise ValueError("StrategyPatch scene actions must be non-empty strings")
        force = payload.get("force_resegment", False)
        if not isinstance(force, bool):
            raise ValueError("StrategyPatch force_resegment must be boolean")
        restart = str(payload.get("restart_stage") or "render")
        if restart not in {"t2i", "segment", "reconstruct", "sanitize", "render"}:
            raise ValueError("StrategyPatch restart_stage is not allowed")
        for name in ("t2i_seed", "reconstruction_seed"):
            value = payload.get(name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"StrategyPatch {name} must be an integer")
        return cls(
            prompt_text=prompt,
            negative_prompt_append=tuple(item.strip() for item in negative),
            t2i_seed=payload.get("t2i_seed"),
            reconstruction_seed=payload.get("reconstruction_seed"),
            force_resegment=force,
            scene_actions=tuple(actions),
            restart_stage=restart,
        )


@dataclass(frozen=True)
class PlannerContext(JsonModel):
    goal: dict[str, Any]
    remaining_budget: dict[str, int]
    asset_id: str
    incumbent: dict[str, Any]
    evaluation: dict[str, Any]
    failure_signature: FailureSignature
    evidence: dict[str, Any]
    experiment_history: tuple[dict[str, Any], ...]
    attempted_strategy_hashes: tuple[str, ...]
    capability_memories: tuple[dict[str, Any], ...]
    allowed_experiment_kinds: tuple[ExperimentKind, ...]
    allowed_scene_actions: tuple[str, ...]
    registered_operators: tuple[dict[str, Any], ...]
    compatibility_hash: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannerContext":
        return cls(
            goal=dict(payload.get("goal") or {}),
            remaining_budget={str(key): int(value) for key, value in (payload.get("remaining_budget") or {}).items()},
            asset_id=str(payload["asset_id"]),
            incumbent=dict(payload.get("incumbent") or {}),
            evaluation=dict(payload.get("evaluation") or {}),
            failure_signature=FailureSignature.from_dict(payload["failure_signature"]),
            evidence=dict(payload.get("evidence") or {}),
            experiment_history=tuple(dict(item) for item in payload.get("experiment_history", [])),
            attempted_strategy_hashes=tuple(str(item) for item in payload.get("attempted_strategy_hashes", [])),
            capability_memories=tuple(dict(item) for item in payload.get("capability_memories", [])),
            allowed_experiment_kinds=tuple(ExperimentKind(item) for item in payload.get("allowed_experiment_kinds", [])),
            allowed_scene_actions=tuple(str(item) for item in payload.get("allowed_scene_actions", [])),
            registered_operators=tuple(dict(item) for item in payload.get("registered_operators", [])),
            compatibility_hash=str(payload.get("compatibility_hash") or ""),
        )


@dataclass(frozen=True)
class PlannerDecision(JsonModel):
    decision_id: str
    decision: PlannerDecisionKind
    asset_id: str
    failure_code: str
    experiment_kind: ExperimentKind | None
    hypothesis: str
    strategy_patch: StrategyPatch | None
    expected_effect: dict[str, str]
    required_evidence: tuple[str, ...]
    confidence: str
    rationale_summary: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannerDecision":
        allowed = {
            "decision_id", "decision", "asset_id", "failure_code", "experiment_kind",
            "hypothesis", "strategy_patch", "expected_effect", "required_evidence",
            "confidence", "rationale_summary",
        }
        _reject_unknown(payload, allowed, "PlannerDecision")
        decision = PlannerDecisionKind(payload["decision"])
        raw_kind = payload.get("experiment_kind")
        kind = ExperimentKind(raw_kind) if raw_kind else None
        raw_patch = payload.get("strategy_patch")
        patch = StrategyPatch.from_dict(raw_patch) if raw_patch is not None else None
        if decision is PlannerDecisionKind.RUN_EXPERIMENT and (kind is None or patch is None):
            raise ValueError("RUN_EXPERIMENT requires experiment_kind and strategy_patch")
        if decision is not PlannerDecisionKind.RUN_EXPERIMENT and (kind is not None or patch is not None):
            raise ValueError("non-experiment decisions cannot include experiment_kind or strategy_patch")
        hypothesis = str(payload.get("hypothesis") or "").strip()
        rationale = str(payload.get("rationale_summary") or "").strip()
        if len(hypothesis) > 1024 or len(rationale) > 512:
            raise ValueError("PlannerDecision text exceeds length limit")
        confidence = str(payload.get("confidence") or "low")
        if confidence not in {"low", "medium", "high"}:
            raise ValueError("PlannerDecision confidence is not allowed")
        effect = payload.get("expected_effect") or {}
        if not isinstance(effect, dict) or any(value not in {"improve", "preserve", "observe"} for value in effect.values()):
            raise ValueError("PlannerDecision expected_effect must map metrics to improve/preserve/observe")
        evidence = payload.get("required_evidence") or ()
        if not isinstance(evidence, (list, tuple)) or len(evidence) > 16 or any(not isinstance(item, str) for item in evidence):
            raise ValueError("PlannerDecision required_evidence is invalid")
        return cls(
            decision_id=str(payload.get("decision_id") or "").strip(),
            decision=decision,
            asset_id=str(payload.get("asset_id") or "").strip(),
            failure_code=str(payload.get("failure_code") or "").strip(),
            experiment_kind=kind,
            hypothesis=hypothesis,
            strategy_patch=patch,
            expected_effect={str(key): str(value) for key, value in effect.items()},
            required_evidence=tuple(evidence),
            confidence=confidence,
            rationale_summary=rationale,
        )


@dataclass(frozen=True)
class Intervention(JsonModel):
    intervention_id: str
    failure_code: str
    operator_id: str
    level: InterventionLevel
    parameters: dict[str, Any] = field(default_factory=dict)
    candidate_delta: dict[str, Any] = field(default_factory=dict)
    expected_effect: dict[str, Any] = field(default_factory=dict)
    rollback_plan: dict[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    required_budget: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Intervention":
        return cls(
            intervention_id=str(payload["intervention_id"]),
            failure_code=str(payload["failure_code"]),
            operator_id=str(payload["operator_id"]),
            level=InterventionLevel(payload["level"]),
            parameters=dict(payload.get("parameters") or {}),
            candidate_delta=dict(payload.get("candidate_delta") or {}),
            expected_effect=dict(payload.get("expected_effect") or {}),
            rollback_plan=dict(payload.get("rollback_plan") or {}),
            preconditions=tuple(str(item) for item in payload.get("preconditions", [])),
            required_budget={str(key): float(value) for key, value in (payload.get("required_budget") or {}).items()},
        )


@dataclass(frozen=True)
class ExperimentRecord(JsonModel):
    experiment_id: str
    intervention_id: str
    matched_trial_key: MatchedTrialKey
    baseline_mode: str
    decision: str = ""
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Capability(JsonModel):
    capability_id: str
    version: str
    status: CapabilityStatus
    failure_signature: FailureSignature
    intervention: Intervention
    validation_evidence: tuple[str, ...] = ()
    known_failures: tuple[str, ...] = ()
    context_signature: str = ""
    strategy_patch: StrategyPatch | None = None
    experiment_kind: ExperimentKind | None = None
    metric_delta: dict[str, float] = field(default_factory=dict)
    protected_metric_delta: dict[str, float] = field(default_factory=dict)
    compatibility_hash: str = ""
    source_asset_id: str = ""
    novelty_key: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Capability":
        patch = payload.get("strategy_patch")
        kind = payload.get("experiment_kind")
        return cls(
            capability_id=str(payload["capability_id"]),
            version=str(payload["version"]),
            status=CapabilityStatus(payload["status"]),
            failure_signature=FailureSignature.from_dict(payload["failure_signature"]),
            intervention=Intervention.from_dict(payload["intervention"]),
            validation_evidence=tuple(str(item) for item in payload.get("validation_evidence", [])),
            known_failures=tuple(str(item) for item in payload.get("known_failures", [])),
            context_signature=str(payload.get("context_signature") or ""),
            strategy_patch=StrategyPatch.from_dict(patch) if patch is not None else None,
            experiment_kind=ExperimentKind(kind) if kind else None,
            metric_delta={str(key): float(value) for key, value in (payload.get("metric_delta") or {}).items()},
            protected_metric_delta={
                str(key): float(value) for key, value in (payload.get("protected_metric_delta") or {}).items()
            },
            compatibility_hash=str(payload.get("compatibility_hash") or ""),
            source_asset_id=str(payload.get("source_asset_id") or ""),
            novelty_key=str(payload.get("novelty_key") or ""),
        )


@dataclass(frozen=True)
class StepReceipt(JsonModel):
    campaign_id: str
    previous_state: ControllerState
    new_state: ControllerState
    action_count: int
    effect: str
    status: str
    next_safe_action: str
    budget_remaining: dict[str, int]
    receipt_hash: str = ""

    @classmethod
    def build(
        cls,
        *,
        campaign_id: str,
        previous_state: ControllerState,
        new_state: ControllerState,
        action_count: int,
        effect: str,
        status: str,
        next_safe_action: str,
        budget_remaining: dict[str, int],
    ) -> "StepReceipt":
        receipt = cls(
            campaign_id=campaign_id,
            previous_state=previous_state,
            new_state=new_state,
            action_count=action_count,
            effect=effect,
            status=status,
            next_safe_action=next_safe_action,
            budget_remaining=budget_remaining,
        )
        return receipt.with_updates(receipt_hash=receipt.sha256())


@dataclass(frozen=True)
class GoalSpec(JsonModel):
    schema_version: str
    campaign_id: str
    objective: dict[str, Any]
    initial_operator_id: str
    route_candidates: tuple[str, ...]
    budgets: BudgetSpec
    resource_policy: dict[str, Any]
    frozen_evaluator: dict[str, str]
    planner: dict[str, Any] = field(default_factory=dict)
    initial_operator_parameters: dict[str, Any] = field(default_factory=dict)
    interventions: tuple[dict[str, Any], ...] = ()
    lease_seconds: int = 300

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoalSpec":
        if payload.get("schema_version") != "dataevolver.harness.goal.v1":
            raise ValueError(f"unsupported goal schema: {payload.get('schema_version')}")
        campaign_id = str(payload.get("campaign_id") or "").strip()
        if not campaign_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in campaign_id):
            raise ValueError("campaign_id must use letters, digits, dot, underscore, or dash")
        evaluator = payload.get("frozen_evaluator") or {}
        if not evaluator.get("path") or not evaluator.get("sha256"):
            raise ValueError("frozen_evaluator path and sha256 are required")
        return cls(
            schema_version=str(payload["schema_version"]),
            campaign_id=campaign_id,
            objective=dict(payload.get("objective") or {}),
            initial_operator_id=str(payload["initial_operator_id"]),
            route_candidates=tuple(str(item) for item in payload.get("route_candidates", [])),
            budgets=BudgetSpec.from_dict(payload.get("budgets") or {}),
            resource_policy=dict(payload.get("resource_policy") or {}),
            frozen_evaluator={str(key): str(value) for key, value in evaluator.items()},
            planner=dict(payload.get("planner") or {}),
            initial_operator_parameters=dict(payload.get("initial_operator_parameters") or {}),
            interventions=tuple(dict(item) for item in payload.get("interventions", [])),
            lease_seconds=int(payload.get("lease_seconds", 300)),
        )

    @classmethod
    def from_json(cls, text: str) -> "GoalSpec":
        return cls.from_dict(json.loads(text))
