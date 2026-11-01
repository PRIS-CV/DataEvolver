"""Final-report projection; persistence and state transitions remain in Controller."""

from __future__ import annotations

import json
from typing import Any

from .models import CampaignState, Capability, ControllerState, EvidenceStatus, GoalSpec
from .operators import OperatorRegistry
from .planner import AgentPlanner
from .state_store import CampaignLayout


def build_final_report(
    state: CampaignState,
    *,
    goal: GoalSpec,
    registry: OperatorRegistry,
    planner: AgentPlanner,
    layout: CampaignLayout,
    capabilities: list[Capability],
) -> dict[str, Any]:
    """Project existing route evidence into the versioned final-report payload."""
    real_canary_evidence = all(
        state.data.get(name) is True
        for name in (
            "real_agent_planner_called",
            "real_3d_experiment_verified",
            "matched_candidate_decision_verified",
            "post_decision_replanned",
        )
    )
    scope_reader = getattr(
        registry.get(goal.initial_operator_id), "verified_capability_level", None
    )
    if callable(scope_reader):
        capability_level = scope_reader(state)
        if capability_level not in {
            "designed",
            "mock_verified",
            "real_execution_verified",
        }:
            raise ValueError(
                "registered execution scope cannot grant training benefit or production qualification"
            )
    elif goal.objective.get("kind") == "agentic_3d_rsi":
        capability_level = (
            "agentic_3d_canary_verified"
            if goal.objective.get("real_canary") is True
            and state.state is ControllerState.COMPLETE
            and real_canary_evidence
            and bool(capabilities)
            else "agentic_3d_mock_verified"
        )
    elif goal.planner.get("kind") == "runtime_rsi":
        history = state.data.get("rsi_history", [])
        workflow = state.data.get("training_workflow", {})
        capability_level = (
            "runtime_recipe_iteration_verified"
            if goal.objective.get("real_data_execution") is True
            and state.state is ControllerState.COMPLETE
            and state.data.get("rsi_observations")
            and any(x["decision"]["decision"] == "ADOPT_CANDIDATE" for x in history)
            else "runtime_rsi_engineering_only"
        )
        if (
            goal.objective.get("real_data_execution") is True
            and state.state is ControllerState.COMPLETE
            and workflow.get("validation_pairs")
            and workflow.get("reuse_pairs")
            and workflow.get("completed", [])[-1].get("kind") == "export"
        ):
            capability_level = "runtime_training_knowledge_reuse_verified"
    elif goal.initial_operator_id == "multimodal.t2i_canary":
        capability_level = "real_t2i_canary_verified"
    elif goal.initial_operator_id == "data.image_recipe":
        capability_level = (
            "real_execution_verified"
            if state.state is ControllerState.COMPLETE
            and state.data.get("local_recipe_models_verified")
            else "mock_verified"
        )
    elif goal.initial_operator_id in {
        "data.image_curation",
        "data.graph",
        "data.catalog_trial",
    }:
        verified = (
            state.data.get("last_effect_verification", {}).get("status")
            == EvidenceStatus.VERIFIED.value
        )
        calls = (
            (state.data.get("last_stage_result") or {})
            .get("metrics", {})
            .get("local_model_calls", 0)
        )
        capability_level = (
            "real_execution_verified"
            if state.state is ControllerState.COMPLETE
            and verified
            and (calls > 0 or goal.objective.get("real_data_execution") is True)
            else "mock_verified"
            if state.state is ControllerState.COMPLETE
            else "designed"
        )
    elif goal.initial_operator_id == "data.visual_g2":
        capability_level = (
            "real_g2_execution_verified"
            if state.state is ControllerState.COMPLETE
            and state.data.get("last_effect_verification", {}).get("status")
            == "VERIFIED"
            and state.data.get("last_stage_result", {})
            .get("metrics", {})
            .get("local_model_calls", 0)
            > 0
            else "designed"
        )
    else:
        capability_level = "mock_verified"
    if state.state is ControllerState.COMPLETE:
        terminal_state = "COMPLETE"
    elif state.terminal_reason == "BUDGET_EXHAUSTED":
        terminal_state = "BUDGET_EXHAUSTED"
    elif state.terminal_reason == "PLANNER_ABSTAINED":
        terminal_state = "ABSTAINED"
    elif state.terminal_reason in {
        "NO_SAFE_ACTION",
        "INVALID_ACCEPT_ASSET",
        "INVALID_ABANDON_ASSET",
    }:
        terminal_state = "NO_VALID_EXPERIMENT"
    else:
        terminal_state = "FAILED"
    return {
        "schema_version": "dataevolver.harness.final_report.v1",
        "campaign_id": state.campaign_id,
        "status": state.state.value,
        "reason": state.terminal_reason,
        "operator_calls": state.operator_calls,
        "experiments_used": state.experiments_used,
        "last_event_seq": state.last_event_seq,
        "capability_level": capability_level,
        "terminal_state": terminal_state,
        "planner_id": planner.planner_id,
        "planner_calls": state.planner_calls,
        "rounds": state.round_index,
        "accepted_assets": list(state.data.get("accepted_assets", [])),
        "abandoned_assets": list(state.data.get("abandoned_assets", [])),
        "capabilities_learned": len(capabilities),
        "final_capability": capabilities[-1].to_dict() if capabilities else None,
        "real_canary_evidence": {
            name: bool(state.data.get(name, False))
            for name in (
                "real_agent_planner_called",
                "real_3d_experiment_verified",
                "matched_candidate_decision_verified",
                "post_decision_replanned",
            )
        },
        "recipe_history": state.data.get("recipe_history", []),
        "incumbent_recipe": state.data.get("incumbent_recipe"),
        "recipe_reuse": state.data.get("recipe_reuse"),
        "recipe_improvement_verified": bool(
            state.data.get("recipe_improvement_verified", False)
        ),
        "runtime_rsi_history": state.data.get("rsi_history", []),
        "runtime_proposal_actor": "runtime_agent"
        if goal.planner.get("kind") == "runtime_rsi"
        else None,
        "training_workflow": state.data.get("training_workflow"),
        "independent_runtime_knowledge_verified": capability_level
        == "runtime_training_knowledge_reuse_verified",
        "recursive_method_improvement_verified": False,
        "training_benefit_verified": False,
        **(
            {
                "runtime_admission": state.data["runtime_admission"],
                "runtime_version_provenance": json.loads(
                    (layout.campaign_dir / "runtime-admission.json").read_text()
                )["provenance"],
            }
            if state.data.get("runtime_admission")
            else {}
        ),
    }
