from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .models import (
    ExperimentKind,
    JsonModel,
    PlannerContext,
    PlannerDecision,
    PlannerDecisionKind,
    StrategyPatch,
    canonical_json,
)


@dataclass(frozen=True)
class CompiledExperiment(JsonModel):
    experiment_id: str
    asset_id: str
    experiment_kind: ExperimentKind
    novelty_key: str
    restart_stage: str
    operator_id: str
    operator_version: str
    operator_parameters: dict[str, Any]
    idempotency_key: str
    required_evidence: tuple[str, ...]


class ExperimentCompiler:
    _RESTART = {
        ExperimentKind.PROMPT_REGENERATE: "t2i",
        ExperimentKind.RESEGMENT_RECONSTRUCT: "segment",
        ExperimentKind.RECONSTRUCT_RETRY: "reconstruct",
        ExperimentKind.SCENE_REFINE: "render",
    }

    def __init__(self, *, operator_id: str, operator_version: str, allowed_scene_actions: set[str]):
        self.operator_id = operator_id
        self.operator_version = operator_version
        self.allowed_scene_actions = frozenset(allowed_scene_actions)

    @staticmethod
    def _changed_fields(patch: StrategyPatch) -> set[str]:
        changed = set()
        if patch.prompt_text is not None:
            changed.add("prompt_text")
        if patch.negative_prompt_append:
            changed.add("negative_prompt_append")
        if patch.t2i_seed is not None:
            changed.add("t2i_seed")
        if patch.reconstruction_seed is not None:
            changed.add("reconstruction_seed")
        if patch.force_resegment:
            changed.add("force_resegment")
        if patch.scene_actions:
            changed.add("scene_actions")
        return changed

    def _validate_shape(self, kind: ExperimentKind, patch: StrategyPatch) -> None:
        restart = self._RESTART[kind]
        if patch.restart_stage != restart:
            raise ValueError(f"restart_stage must be {restart} for {kind.value}")
        changed = self._changed_fields(patch)
        allowed = {
            ExperimentKind.PROMPT_REGENERATE: {"prompt_text", "negative_prompt_append", "t2i_seed"},
            ExperimentKind.RESEGMENT_RECONSTRUCT: {"force_resegment"},
            ExperimentKind.RECONSTRUCT_RETRY: {"reconstruction_seed"},
            ExperimentKind.SCENE_REFINE: {"scene_actions"},
        }[kind]
        disallowed = changed - allowed
        if disallowed:
            raise ValueError(f"{sorted(disallowed)[0]} is not allowed for {kind.value}")
        if not changed:
            raise ValueError(f"{kind.value} requires a material strategy change")
        if kind is ExperimentKind.RESEGMENT_RECONSTRUCT and not patch.force_resegment:
            raise ValueError("RESEGMENT_RECONSTRUCT requires force_resegment=true")
        if kind is ExperimentKind.RECONSTRUCT_RETRY and patch.reconstruction_seed is None:
            raise ValueError("RECONSTRUCT_RETRY requires reconstruction_seed")
        if kind is ExperimentKind.SCENE_REFINE:
            if not patch.scene_actions:
                raise ValueError("SCENE_REFINE requires scene_actions")
            for action in patch.scene_actions:
                if action not in self.allowed_scene_actions:
                    raise ValueError(f"scene action is not registered: {action}")

    def compile(self, decision: PlannerDecision, context: PlannerContext) -> CompiledExperiment:
        if decision.decision is not PlannerDecisionKind.RUN_EXPERIMENT:
            raise ValueError("only RUN_EXPERIMENT decisions can be compiled")
        if decision.asset_id != context.asset_id or decision.failure_code != context.failure_signature.code:
            raise ValueError("PlannerDecision does not match PlannerContext")
        if int(context.remaining_budget.get("experiments", 0)) <= 0:
            raise ValueError("experiment budget is exhausted")
        if int(context.remaining_budget.get("operator_calls", 0)) <= 0:
            raise ValueError("operator call budget is exhausted")
        target_metrics = set((context.evaluation.get("target_metrics") or {}).keys())
        if not decision.required_evidence:
            raise ValueError("RUN_EXPERIMENT requires at least one target metric")
        unknown_evidence = set(decision.required_evidence) - target_metrics
        if unknown_evidence:
            raise ValueError(f"required evidence is not a target metric: {sorted(unknown_evidence)[0]}")
        if set(decision.expected_effect) - set(decision.required_evidence):
            raise ValueError("expected_effect must be limited to required_evidence")
        kind = decision.experiment_kind
        patch = decision.strategy_patch
        self._validate_shape(kind, patch)
        novelty_key = hashlib.sha256(
            canonical_json({"asset_id": context.asset_id, "experiment_kind": kind, "strategy_patch": patch}).encode("utf-8")
        ).hexdigest()
        if novelty_key in set(context.attempted_strategy_hashes):
            raise ValueError("strategy was already attempted")
        idempotency_key = f"{context.asset_id}:{kind.value.lower()}:{novelty_key}"
        return CompiledExperiment(
            experiment_id=decision.decision_id,
            asset_id=context.asset_id,
            experiment_kind=kind,
            novelty_key=novelty_key,
            restart_stage=self._RESTART[kind],
            operator_id=self.operator_id,
            operator_version=self.operator_version,
            operator_parameters={
                "asset_id": context.asset_id,
                "experiment_kind": kind.value,
                "strategy_patch": patch.to_dict(),
                "incumbent": dict(context.incumbent),
                "required_evidence": list(decision.required_evidence),
                "compatibility_hash": context.compatibility_hash,
            },
            idempotency_key=idempotency_key,
            required_evidence=decision.required_evidence,
        )
