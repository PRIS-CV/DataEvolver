from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .capabilities import CapabilityStore
from .evaluator import DeterministicEvaluator, EvaluatorContract, FrozenExecutionContract
from .experiments import CandidateDecision, CandidateGate, can_reuse_baseline
from .final_report import build_final_report
from .models import (
    CampaignState,
    Capability,
    CapabilityStatus,
    ControllerState,
    EffectVerification,
    EvaluationReport,
    EvidenceStatus,
    ExperimentKind,
    FailureSignature,
    GoalSpec,
    Intervention,
    InterventionLevel,
    MatchedTrialKey,
    OperatorCall,
    PlannerContext,
    PlannerDecision,
    PlannerDecisionKind,
    StageResult,
    StepReceipt,
    canonical_json,
)
from .operators import OperatorRegistry
from .planner import AgentPlanner
from .state_store import CampaignLayout, CampaignLease, StateStore, atomic_write_json
from .strategy import CompiledExperiment, ExperimentCompiler


TERMINAL_STATES = {ControllerState.COMPLETE, ControllerState.ABSTAIN}


class EffectRecoveryError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvolutionController:
    def __init__(self, *, repo_root: Path, layout: CampaignLayout, goal: GoalSpec, registry: OperatorRegistry,
                 planner: AgentPlanner, evaluator_contract: EvaluatorContract, schema_paths: tuple[Path, ...],
                 execution_paths: tuple[Path, ...], experiment_compiler: ExperimentCompiler | None = None):
        self.repo_root = repo_root
        self.layout = layout
        self.goal = goal
        self.registry = registry
        self.planner = planner
        if hasattr(planner, "bind_operator_registry"):
            planner.bind_operator_registry(registry, goal.resource_policy.get("allowed_operators", []))
        self.evaluator = DeterministicEvaluator(evaluator_contract)
        self.gate = CandidateGate(evaluator_contract)
        self.store = StateStore(layout)
        self.capabilities = CapabilityStore(layout.campaign_dir / "capabilities.jsonl")
        self.experiment_compiler = experiment_compiler
        self._entry_admission = None
        if (goal.planner.get('runtime', {}).get('policy_adoption')
                or goal.planner.get('runtime', {}).get('step_bindings')
                or goal.objective.get('step_bindings')):
            from .runtime_entry import admit_execution
            self._entry_admission = admit_execution
        self.contract_guard = FrozenExecutionContract(
            repo_root=repo_root,
            goal=goal,
            registry=registry,
            planner_id=planner.planner_id,
            planner_config=planner.config() if hasattr(planner, "config") else {},
            evaluator_path=evaluator_contract.path,
            schema_paths=schema_paths,
            execution_paths=execution_paths,
        )

    @classmethod
    def create(cls, *, repo_root, campaigns_root, goal, registry, planner, evaluator_contract,
               schema_paths=(), execution_paths=(), experiment_compiler=None, idempotency_key: str):
        layout = CampaignLayout(campaigns_root, goal.campaign_id)
        controller = cls(
            repo_root=Path(repo_root).resolve(), layout=layout, goal=goal, registry=registry, planner=planner,
            evaluator_contract=evaluator_contract,
            schema_paths=tuple(Path(path).resolve() for path in schema_paths),
            execution_paths=tuple(Path(path).resolve() for path in execution_paths),
            experiment_compiler=experiment_compiler,
        )
        if layout.goal_path.exists():
            existing = GoalSpec.from_json(layout.goal_path.read_text(encoding="utf-8"))
            if existing.sha256() != goal.sha256():
                raise ValueError("campaign already exists with a different GoalSpec")
            return controller
        layout.campaign_dir.mkdir(parents=True, exist_ok=True)
        frozen = controller.contract_guard.freeze()
        atomic_write_json(layout.goal_path, goal)
        atomic_write_json(layout.frozen_contract_path, frozen)
        started = _now()
        controller.store.initialize_state(CampaignState(
            campaign_id=goal.campaign_id, started_at=started, updated_at=started, idempotency_key=idempotency_key
        ))
        return controller

    @classmethod
    def open(cls, *, repo_root, campaigns_root, goal, registry, planner, evaluator_contract,
             schema_paths=(), execution_paths=(), experiment_compiler=None):
        return cls(
            repo_root=Path(repo_root).resolve(), layout=CampaignLayout(campaigns_root, goal.campaign_id),
            goal=goal, registry=registry, planner=planner, evaluator_contract=evaluator_contract,
            schema_paths=tuple(Path(path).resolve() for path in schema_paths),
            execution_paths=tuple(Path(path).resolve() for path in execution_paths),
            experiment_compiler=experiment_compiler,
        )

    def _budget_remaining(self, state):
        planner_limit = self.goal.budgets.max_planner_calls
        return {
            "rounds": max(0, self.goal.budgets.max_rounds - state.round_index),
            "operator_calls": max(0, self.goal.budgets.max_operator_calls - state.operator_calls),
            "experiments": max(0, self.goal.budgets.max_experiments - state.experiments_used),
            "retries": max(0, self.goal.budgets.max_retries - state.retries_used),
            "planner_calls": (
                max(0, planner_limit - state.planner_calls)
                if planner_limit > 0
                else max(0, self.goal.budgets.max_rounds + 1 - state.planner_calls)
            ),
        }

    def _receipt(self, before, state, effect, count=1):
        status = "TERMINAL" if state.state in TERMINAL_STATES else (
            "PAUSED" if state.state in {ControllerState.PAUSED, ControllerState.PAUSED_CONTRACT_DRIFT} else "CONTINUE"
        )
        return StepReceipt.build(
            campaign_id=state.campaign_id, previous_state=before, new_state=state.state, action_count=count,
            effect=effect, status=status, next_safe_action="" if status == "TERMINAL" else state.state.value,
            budget_remaining=self._budget_remaining(state),
        )

    def _commit(self, state, new_state, event_type, effect, *, data=None, terminal_reason=None,
                idempotency_key="", counters=None, event_payload=None):
        updates = {
            "previous_state": state.state, "state": new_state, "updated_at": _now(),
            "data": data if data is not None else state.data,
        }
        if terminal_reason is not None:
            updates["terminal_reason"] = terminal_reason
        updates.update(counters or {})
        return self.store.commit(
            state.with_updates(**updates), event_type, {"effect": effect, **(event_payload or {})}, idempotency_key
        )

    def _frozen(self):
        if hasattr(self.planner, "config"):
            self.contract_guard.planner_config = self.planner.config()
        drift = self.contract_guard.verify(json.loads(self.layout.frozen_contract_path.read_text(encoding="utf-8")))
        return drift.valid, drift.changed_fields

    def commit_domain_transition(self, state, proposal):
        from .domains import TransitionProposal
        if not isinstance(proposal, TransitionProposal):
            raise TypeError("domain must return a TransitionProposal")
        proposal.validate(state)
        if not self._goal_satisfied(EvaluationReport.from_dict(state.data['last_evaluation'])):
            raise ValueError("domain completion requires the frozen evaluation")
        updated = state.with_updates(data=proposal.data)
        if proposal.complete:
            return self._complete(updated), proposal.effect
        return self._commit(state, ControllerState.PLAN, proposal.event_type,
            proposal.effect, data=proposal.data, counters=proposal.counters,
            event_payload={"completion_evidence": proposal.completion_evidence}), proposal.effect

    def _pause_for_drift(self, state, changed):
        return self._commit(
            state, ControllerState.PAUSED_CONTRACT_DRIFT, "CONTRACT_DRIFT_DETECTED", "PAUSED_CONTRACT_DRIFT",
            data={**state.data, "paused_from": state.state.value, "contract_drift": list(changed)},
            idempotency_key=f"contract-drift:{state.last_event_seq + 1}",
        )

    def _write_final(self, state):
        capabilities = self.capabilities.list_capabilities()
        report = build_final_report(
            state,
            goal=self.goal,
            registry=self.registry,
            planner=self.planner,
            layout=self.layout,
            capabilities=capabilities,
        )
        atomic_write_json(self.layout.final_report_path, report)

    def _abstain(self, state, reason):
        terminal = self._commit(
            state, ControllerState.ABSTAIN, "CAMPAIGN_ABSTAINED", reason,
            terminal_reason=reason, idempotency_key=f"terminal:abstain:{reason}",
        )
        self._write_final(terminal)
        return terminal

    def _complete(self, state):
        terminal = self._commit(
            state, ControllerState.COMPLETE, "CAMPAIGN_COMPLETED", "GOAL_SATISFIED",
            terminal_reason="GOAL_SATISFIED", idempotency_key="terminal:complete",
        )
        self._write_final(terminal)
        return terminal

    def _goal_satisfied(self, report):
        if not report.all_required_verified() or any(value.value is not True for value in report.hard_constraints.values()):
            return False
        return all(
            float(report.target_metrics[name].value) >= float(
                spec.get("goal_minimum", spec.get("minimum", float("-inf")))
            )
            for name, spec in self.evaluator.contract.payload.get("target_metrics", {}).items()
        )

    def _trial_key(self, name):
        if self.goal.planner.get('kind')=='runtime_rsi' and hasattr(self.planner.domain,'trial_key'):
            return MatchedTrialKey.from_dict(self.planner.domain.trial_key(name,self.store.load_state()))
        payload = self.goal.objective.get(name)
        if not payload:
            raise ValueError(f"GoalSpec objective is missing {name}")
        return MatchedTrialKey.from_dict(payload)

    def _initial_parameters(self,state):
        if self.goal.planner.get('kind')=='runtime_rsi' and hasattr(self.planner.domain,'initial_parameters'):
            return self.planner.domain.initial_parameters(state)
        return self.goal.initial_operator_parameters

    def _make_call(self, *, call_id, operator_id, work_dir, key, parameters=None):
        spec = self.registry.get(operator_id).spec
        return OperatorCall(
            campaign_id=self.goal.campaign_id, call_id=call_id, idempotency_key=call_id,
            operator_id=operator_id, operator_version=spec.version, round_index=self.store.load_state().round_index,
            work_dir=str(work_dir), parameters=dict(parameters or {}), input_artifact_refs=spec.input_artifact_refs,
            expected_output_specs=spec.expected_output_specs, matched_trial_key=key,
        )

    def _execute_call(self, state, call):
        self.registry.validate_call(call)
        operator = self.registry.get(call.operator_id)
        planned_key, committed_key = f"planned:{call.idempotency_key}", f"committed:{call.idempotency_key}"
        if self.store.events.find_idempotency_key(planned_key) is None:
            state = self.store.commit(
                state.with_updates(data={**state.data, "pending_call": call.to_dict()}),
                "ACTION_PLANNED", {"call_id": call.call_id, "operator_call": call.to_dict()}, planned_key,
            )
            receipt = operator.execute(call)
        else:
            recovery = operator.recover(call)
            if recovery.receipt is not None:
                receipt = recovery.receipt
            elif recovery.status.value == "RETRY_SAFE" and operator.spec.retry_safe:
                receipt = operator.execute(call)
            else:
                raise EffectRecoveryError(f"operator effect recovery is ambiguous: {recovery.reason}")
        verification = operator.verify_effect(call, receipt)
        if verification.status is not EvidenceStatus.VERIFIED:
            raise EffectRecoveryError(f"operator effect is not verified: {verification.reason}")
        payload = receipt.stage_result or {
            "operator_id": receipt.operator_id, "operator_version": receipt.operator_version,
            "status": receipt.status.value, "metrics": {}, "hard_constraints": {},
        }
        result = StageResult.from_dict({
            **payload, "output_artifacts": list(receipt.output_artifacts),
            "matched_trial_key": call.matched_trial_key.to_dict() if call.matched_trial_key else None,
            "stdout_tail": receipt.stdout_tail,
        })
        state = self.store.commit(
            state, "ACTION_COMMITTED",
            {"call_id": call.call_id, "receipt": receipt.to_dict(), "effect_verification": verification.to_dict()},
            committed_key,
        )
        return state, result, verification

    def _is_agentic(self) -> bool:
        return self.goal.planner.get("kind") == "agentic_rsi"

    def _planner_budget_exhausted(self, state: CampaignState) -> bool:
        limit = self.goal.budgets.max_planner_calls
        return limit > 0 and state.planner_calls >= limit

    def _compatibility_hash(self) -> str:
        frozen = json.loads(self.layout.frozen_contract_path.read_text(encoding="utf-8"))
        return hashlib.sha256(canonical_json(frozen).encode("utf-8")).hexdigest()

    @staticmethod
    def _compiled_from_dict(payload: dict[str, Any]) -> CompiledExperiment:
        return CompiledExperiment(
            experiment_id=str(payload["experiment_id"]),
            asset_id=str(payload["asset_id"]),
            experiment_kind=ExperimentKind(payload["experiment_kind"]),
            novelty_key=str(payload["novelty_key"]),
            restart_stage=str(payload["restart_stage"]),
            operator_id=str(payload["operator_id"]),
            operator_version=str(payload["operator_version"]),
            operator_parameters=dict(payload["operator_parameters"]),
            idempotency_key=str(payload["idempotency_key"]),
            required_evidence=tuple(str(item) for item in payload.get("required_evidence", [])),
        )

    def _agentic_failure(self, result: StageResult, report: EvaluationReport) -> FailureSignature:
        if result.failure_code:
            return FailureSignature(result.failure_code, detail=result.failure_detail, evidence_refs=result.evidence_refs)
        failed_hard = [name for name, value in report.hard_constraints.items() if value.value is not True]
        if failed_hard:
            return FailureSignature("HARD_CONSTRAINT_FAILED", detail=",".join(failed_hard))
        below_goal = [
            name
            for name, spec in self.evaluator.contract.payload.get("target_metrics", {}).items()
            if float(report.target_metrics[name].value) < float(
                spec.get("goal_minimum", spec.get("minimum", float("-inf")))
            )
        ]
        return FailureSignature("TARGET_NOT_MET", detail=",".join(below_goal))

    def _context_signature(self, failure: FailureSignature, report: EvaluationReport) -> str:
        payload = {
            "failure_code": failure.code,
            "slice_key": failure.slice_key,
            "failed_hard": sorted(name for name, value in report.hard_constraints.items() if value.value is not True),
            "failure_slices": list(report.failure_slices),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def _agentic_context(
        self,
        state: CampaignState,
        failure: FailureSignature,
        report: EvaluationReport,
    ) -> tuple[PlannerContext, tuple[Capability, ...]]:
        compatibility_hash = self._compatibility_hash()
        context_signature = self._context_signature(failure, report)
        attempted = tuple(str(item) for item in state.data.get("attempted_strategy_hashes", []))
        exact = list(
            self.capabilities.query(
                context_signature,
                failure.code,
                compatibility_hash,
                attempted=attempted,
                limit=5,
            )
        )
        seen = {item.capability_id for item in exact}
        compatible_recent = [
            item
            for item in reversed(self.capabilities.list_capabilities())
            if item.compatibility_hash == compatibility_hash
            and item.capability_id not in seen
        ]
        memories = tuple((exact + compatible_recent)[:5])
        allowed_kinds = tuple(
            ExperimentKind(item)
            for item in self.goal.objective.get("allowed_experiment_kinds", [item.value for item in ExperimentKind])
        )
        context = PlannerContext(
            goal={"objective": self.goal.objective, "frozen_evaluator": self.goal.frozen_evaluator},
            remaining_budget=self._budget_remaining(state),
            asset_id=str(state.data["active_asset_id"]),
            incumbent=dict(state.data.get("incumbent") or {}),
            evaluation=report.to_dict(),
            failure_signature=failure,
            evidence={
                "failure_slices": list(report.failure_slices),
                "evidence_refs": list(report.evidence_refs),
                "last_stage_result": {
                    key: state.data.get("last_stage_result", {}).get(key)
                    for key in ("status", "failure_code", "failure_detail", "metrics", "hard_constraints", "evidence_refs")
                },
            },
            experiment_history=tuple(dict(item) for item in state.data.get("experiment_history", [])),
            attempted_strategy_hashes=attempted,
            capability_memories=tuple(item.to_dict() for item in memories),
            allowed_experiment_kinds=allowed_kinds,
            allowed_scene_actions=tuple(str(item) for item in self.goal.objective.get("allowed_scene_actions", [])),
            registered_operators=tuple(
                {"operator_id": spec.operator_id, "version": spec.version}
                for spec in self.registry.specs()
            ),
            compatibility_hash=compatibility_hash,
        )
        return context, memories

    def _accept_agentic_asset(self, state: CampaignState) -> CampaignState:
        data = dict(state.data)
        asset_id = str(data["active_asset_id"])
        accepted = list(data.get("accepted_assets", []))
        if asset_id not in accepted:
            accepted.append(asset_id)
        data["accepted_assets"] = accepted
        quota = int(self.goal.objective.get("accepted_asset_quota", len(self.goal.objective.get("asset_ids", [])) or 1))
        if len(accepted) >= quota:
            return self._complete(state.with_updates(data=data))
        assets = list(self.goal.objective.get("asset_ids", []))
        abandoned = set(data.get("abandoned_assets", []))
        remaining = [item for item in assets if item not in set(accepted) | abandoned]
        if not remaining:
            return self._abstain(state.with_updates(data=data), "ASSET_QUOTA_UNMET")
        data.update(
            {
                "active_asset_id": remaining[0],
                "incumbent": {},
                "last_stage_result": None,
                "last_evaluation": None,
                "attempted_strategy_hashes": [],
                "experiment_history": [],
            }
        )
        return self._commit(state, ControllerState.PLAN, "ASSET_ACCEPTED", "NEXT_ASSET", data=data)

    def _abandon_agentic_asset(self, state: CampaignState) -> CampaignState:
        state = state.with_updates(round_index=state.round_index + 1)
        data = dict(state.data)
        asset_id = str(data["active_asset_id"])
        abandoned = list(data.get("abandoned_assets", []))
        if asset_id not in abandoned:
            abandoned.append(asset_id)
        data["abandoned_assets"] = abandoned
        assets = list(self.goal.objective.get("asset_ids", []))
        accepted = set(data.get("accepted_assets", []))
        remaining = [item for item in assets if item not in accepted | set(abandoned)]
        if not remaining:
            return self._abstain(state.with_updates(data=data), "ASSET_QUOTA_UNMET")
        data.update(
            {
                "active_asset_id": remaining[0],
                "incumbent": {},
                "last_stage_result": None,
                "last_evaluation": None,
                "attempted_strategy_hashes": [],
                "experiment_history": [],
            }
        )
        return self._commit(state, ControllerState.PLAN, "ASSET_ABANDONED", "NEXT_ASSET", data=data)

    @staticmethod
    def _metric_deltas(baseline: EvaluationReport, candidate: EvaluationReport, field_name: str) -> dict[str, float]:
        before = getattr(baseline, field_name)
        after = getattr(candidate, field_name)
        return {
            name: float(after[name].value) - float(value.value)
            for name, value in before.items()
            if value.value is not None and after[name].value is not None
        }

    def _tick_agentic(self, state: CampaignState):
        if state.state is ControllerState.INIT:
            assets = [str(item) for item in self.goal.objective.get("asset_ids", [])]
            if not assets:
                return self._abstain(state, "NO_ASSETS"), "NO_ASSETS"
            data = {
                **state.data,
                "asset_ids": assets,
                "active_asset_id": assets[0],
                "accepted_assets": [],
                "abandoned_assets": [],
                "incumbent": {},
                "attempted_strategy_hashes": [],
                "experiment_history": [],
            }
            return self._commit(
                state,
                ControllerState.PLAN,
                "CAMPAIGN_INITIALIZED",
                "PLAN_CREATED",
                data=data,
                idempotency_key="campaign:init",
            ), "PLAN_CREATED"

        if state.state is ControllerState.PLAN:
            if state.data.get("incumbent"):
                return self._commit(state, ControllerState.DIAGNOSE, "PLAN_CREATED", "REPLAN_INCUMBENT"), "REPLAN_INCUMBENT"
            if state.operator_calls >= self.goal.budgets.max_operator_calls:
                return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
            asset_id = str(state.data["active_asset_id"])
            work_dir = self.layout.campaign_dir / "rounds" / "r000" / "assets" / asset_id / "candidate"
            parameters = {
                **self.goal.initial_operator_parameters,
                "asset_id": asset_id,
                "experiment_kind": "INITIAL",
            }
            call = self._make_call(
                call_id=f"{state.campaign_id}:{asset_id}:r000:initial",
                operator_id=self.goal.initial_operator_id,
                work_dir=work_dir,
                key=self._trial_key("initial_trial_key"),
                parameters=parameters,
            )
            return self._commit(
                state,
                ControllerState.EXECUTE,
                "PLAN_CREATED",
                "INITIAL_OPERATOR_SELECTED",
                data={**state.data, "pending_call": call.to_dict(), "active_work_dir": str(work_dir)},
            ), "INITIAL_OPERATOR_SELECTED"

        if state.state is ControllerState.EXECUTE:
            call = OperatorCall.from_dict(state.data["pending_call"])
            try:
                state, result, verification = self._execute_call(state, call)
            except EffectRecoveryError:
                return self._abstain(state, "EFFECT_RECOVERY_AMBIGUOUS"), "EFFECT_RECOVERY_AMBIGUOUS"
            work_dir = Path(call.work_dir)
            atomic_write_json(work_dir / "stage_result.json", result)
            incumbent = {
                "work_dir": str(work_dir),
                "stage_result_ref": str(work_dir / "stage_result.json"),
                "output_artifacts": list(result.output_artifacts),
            }
            data = {
                **state.data,
                "pending_call": None,
                "last_stage_result": result.to_dict(),
                "last_effect_verification": verification.to_dict(),
                "incumbent": incumbent,
                "active_work_dir": str(work_dir),
            }
            return self._commit(
                state,
                ControllerState.OBSERVE,
                "EXECUTION_OBSERVED",
                "STAGE_RESULT_RECORDED",
                data=data,
                counters={"operator_calls": state.operator_calls + 1},
            ), "STAGE_RESULT_RECORDED"

        if state.state is ControllerState.OBSERVE:
            result = StageResult.from_dict(state.data["last_stage_result"])
            verification = EffectVerification(
                status=EvidenceStatus(state.data["last_effect_verification"]["status"]),
                reason=state.data["last_effect_verification"].get("reason", ""),
            )
            report = self.evaluator.evaluate(result, effect_status=verification.status)
            work_dir = Path(state.data["active_work_dir"])
            atomic_write_json(work_dir / "evaluation.json", report)
            return self._commit(
                state,
                ControllerState.DIAGNOSE,
                "EVALUATION_RECORDED",
                "EVALUATION_RECORDED",
                data={**state.data, "last_evaluation": report.to_dict()},
            ), "EVALUATION_RECORDED"

        if state.state is ControllerState.DIAGNOSE:
            result = StageResult.from_dict(state.data["last_stage_result"])
            report = EvaluationReport.from_dict(state.data["last_evaluation"])
            goal_satisfied = self._goal_satisfied(report)
            requires_post_replan = bool(state.data.get("requires_post_decision_replan", False))
            if goal_satisfied and not requires_post_replan:
                terminal_or_next = self._accept_agentic_asset(state)
                return terminal_or_next, "ASSET_ACCEPTED"
            if self._planner_budget_exhausted(state):
                return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
            if not goal_satisfied and (
                state.round_index >= self.goal.budgets.max_rounds
                or state.experiments_used >= self.goal.budgets.max_experiments
                or state.operator_calls >= self.goal.budgets.max_operator_calls
            ):
                return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
            failure = FailureSignature("GOAL_SATISFIED") if goal_satisfied else self._agentic_failure(result, report)
            context, memories = self._agentic_context(state, failure, report)
            data = {
                **state.data,
                "failure_signature": failure.to_dict(),
                "planner_context": context.to_dict(),
                "context_signature": self._context_signature(failure, report),
                "compatibility_hash": context.compatibility_hash,
                "goal_satisfied_before_planner": goal_satisfied,
            }
            state = self.store.commit(
                state.with_updates(data=data),
                "CAPABILITY_MEMORY_RETRIEVED",
                {"count": len(memories), "capability_ids": [item.capability_id for item in memories]},
                f"capability-memory:{state.data['active_asset_id']}:{state.round_index}:{failure.code}",
            )
            return self._commit(
                state,
                ControllerState.PROPOSE,
                "PLANNER_CONTEXT_CREATED",
                "PLANNER_CONTEXT_CREATED",
                data=state.data,
                event_payload={"context_hash": context.sha256()},
            ), "PLANNER_CONTEXT_CREATED"

        if state.state is ControllerState.PROPOSE:
            if self._planner_budget_exhausted(state):
                return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
            context = PlannerContext.from_dict(state.data["planner_context"])
            decision = self.planner.plan(context)
            trace = self.planner.last_trace() if hasattr(self.planner, "last_trace") else {}
            planner_dir = self.layout.campaign_dir / "planner" / f"p{state.planner_calls:03d}"
            trace_path = planner_dir / "trace.json"
            atomic_write_json(trace_path, trace)
            trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            if decision.strategy_patch is not None:
                atomic_write_json(planner_dir / "strategy_patch.json", decision.strategy_patch)
            valid_planner_trace = trace.get("outcome") == "VALID_DECISION"
            post_decision_replanned = bool(state.data.get("experiment_history")) and valid_planner_trace
            data = {
                **state.data,
                "planner_decision": decision.to_dict(),
                "planner_trace_refs": [
                    *state.data.get("planner_trace_refs", []),
                    {"path": str(trace_path), "sha256": trace_hash},
                ],
                "requires_post_decision_replan": False,
                "post_decision_replanned": bool(
                    state.data.get("post_decision_replanned", False) or post_decision_replanned
                ),
                "real_agent_planner_called": bool(
                    state.data.get("real_agent_planner_called", False)
                    or (
                        trace.get("backend_id") == "openai-compatible"
                        and bool(trace.get("raw_responses"))
                        and valid_planner_trace
                    )
                ),
            }
            state = self.store.commit(
                state.with_updates(data=data, planner_calls=state.planner_calls + 1),
                "PLANNER_DECISION_RECORDED",
                {
                    "decision": decision.to_dict(),
                    "planner_trace": {"path": str(trace_path), "sha256": trace_hash},
                },
                f"planner-decision:{state.data['active_asset_id']}:p{state.planner_calls:03d}",
            )
            if state.data.get("goal_satisfied_before_planner") is True:
                if decision.decision is not PlannerDecisionKind.ACCEPT_ASSET:
                    state = self.store.commit(
                        state,
                        "PLANNER_DECISION_REJECTED",
                        {"reason": "frozen goal is already satisfied; no further experiment is authorized"},
                        f"planner-rejected:{decision.decision_id}",
                    )
                return self._accept_agentic_asset(state), "ASSET_ACCEPTED"
            if decision.decision is PlannerDecisionKind.ABSTAIN:
                return self._abstain(state, "PLANNER_ABSTAINED"), "PLANNER_ABSTAINED"
            if decision.decision is PlannerDecisionKind.ACCEPT_ASSET:
                report = EvaluationReport.from_dict(state.data["last_evaluation"])
                if self._goal_satisfied(report):
                    return self._accept_agentic_asset(state), "ASSET_ACCEPTED"
                rejected = self.store.commit(
                    state,
                    "PLANNER_DECISION_REJECTED",
                    {"reason": "asset does not satisfy frozen goal"},
                    f"planner-rejected:{decision.decision_id}",
                )
                return self._abstain(rejected, "INVALID_ACCEPT_ASSET"), "INVALID_ACCEPT_ASSET"
            if decision.decision is PlannerDecisionKind.ABANDON_ASSET:
                result = StageResult.from_dict(state.data["last_stage_result"])
                if result.failure_code == "ASSET_ABANDON_RECOMMENDED":
                    return self._abandon_agentic_asset(state), "ASSET_ABANDONED"
                rejected = self.store.commit(
                    state,
                    "PLANNER_DECISION_REJECTED",
                    {"reason": "asset abandonment is not supported by verified evidence"},
                    f"planner-rejected:{decision.decision_id}",
                )
                return self._abstain(rejected, "INVALID_ABANDON_ASSET"), "INVALID_ABANDON_ASSET"
            if decision.decision is PlannerDecisionKind.STOP_CAMPAIGN:
                quota = int(self.goal.objective.get("accepted_asset_quota", 1))
                if len(state.data.get("accepted_assets", [])) >= quota:
                    return self._complete(state), "CAMPAIGN_STOPPED_AFTER_GOAL"
                rejected = self.store.commit(
                    state,
                    "PLANNER_DECISION_REJECTED",
                    {"reason": "campaign goal is not satisfied"},
                    f"planner-rejected:{decision.decision_id}",
                )
                return self._abstain(rejected, "INVALID_STOP_CAMPAIGN"), "INVALID_STOP_CAMPAIGN"
            if decision.decision is not PlannerDecisionKind.RUN_EXPERIMENT or self.experiment_compiler is None:
                rejected = self.store.commit(
                    state,
                    "PLANNER_DECISION_REJECTED",
                    {"reason": "decision is not a compilable experiment"},
                    f"planner-rejected:{decision.decision_id}",
                )
                return self._abstain(rejected, "NO_SAFE_ACTION"), "NO_SAFE_ACTION"
            try:
                compiled = self.experiment_compiler.compile(decision, context)
            except ValueError as exc:
                rejected = self.store.commit(
                    state,
                    "PLANNER_DECISION_REJECTED",
                    {"reason": str(exc)},
                    f"planner-rejected:{decision.decision_id}",
                )
                return self._abstain(rejected, "NO_SAFE_ACTION"), "NO_SAFE_ACTION"
            next_round = state.round_index + 1
            work_dir = (
                self.layout.campaign_dir
                / "rounds"
                / f"r{next_round:03d}"
                / "assets"
                / context.asset_id
                / "candidate"
            )
            data = {
                **state.data,
                "compiled_experiment": compiled.to_dict(),
                "candidate_work_dir": str(work_dir),
            }
            experiment_dir = self.layout.campaign_dir / "experiments" / f"e{state.experiment_index:03d}"
            atomic_write_json(experiment_dir / "compiled_experiment.json", compiled)
            return self._commit(
                state,
                ControllerState.CANARY,
                "EXPERIMENT_COMPILED",
                "EXPERIMENT_COMPILED",
                data=data,
                counters={"round_index": next_round, "experiments_used": state.experiments_used + 1},
                idempotency_key=f"experiment-compiled:{compiled.idempotency_key}",
                event_payload={"compiled_experiment": compiled.to_dict()},
            ), "EXPERIMENT_COMPILED"

        if state.state is ControllerState.CANARY:
            if state.operator_calls >= self.goal.budgets.max_operator_calls:
                return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
            compiled = self._compiled_from_dict(state.data["compiled_experiment"])
            baseline = EvaluationReport.from_dict(state.data["last_evaluation"])
            baseline_result = StageResult.from_dict(state.data["last_stage_result"])
            baseline_mode = "BASELINE_REUSED"
            if (
                baseline.matched_trial_key is None
                or baseline_result.matched_trial_key is None
                or baseline.matched_trial_key.digest() != baseline_result.matched_trial_key.digest()
                or not baseline.all_required_verified()
            ):
                effect_status = EvidenceStatus(state.data["last_effect_verification"]["status"])
                baseline = self.evaluator.evaluate(baseline_result, effect_status=effect_status)
                baseline_mode = "BASELINE_REMATERIALIZED"
            if baseline.matched_trial_key is None or not baseline.all_required_verified():
                return self._abstain(state, "BASELINE_NOT_VERIFIED"), "BASELINE_NOT_VERIFIED"
            experiment_dir = self.layout.campaign_dir / "experiments" / f"e{state.experiment_index:03d}"
            atomic_write_json(experiment_dir / "baseline" / "evaluation.json", baseline)
            atomic_write_json(experiment_dir / "baseline" / "stage_result.json", baseline_result)
            atomic_write_json(experiment_dir / "proposal.json", state.data["planner_decision"])
            atomic_write_json(
                experiment_dir / "matched_trial.json",
                {"mode": baseline_mode, "matched_trial_key": baseline.matched_trial_key.to_dict()},
            )
            call = self._make_call(
                call_id=f"{state.campaign_id}:{compiled.asset_id}:e{state.experiment_index:03d}:{compiled.novelty_key[:12]}",
                operator_id=compiled.operator_id,
                work_dir=Path(state.data["candidate_work_dir"]),
                key=baseline.matched_trial_key,
                parameters=compiled.operator_parameters,
            )
            try:
                state, result, verification = self._execute_call(state, call)
            except EffectRecoveryError:
                return self._abstain(state, "EFFECT_RECOVERY_AMBIGUOUS"), "EFFECT_RECOVERY_AMBIGUOUS"
            candidate = self.evaluator.evaluate(result, effect_status=verification.status)
            atomic_write_json(experiment_dir / "candidate" / "stage_result.json", result)
            atomic_write_json(experiment_dir / "candidate" / "evaluation.json", candidate)
            atomic_write_json(Path(state.data["candidate_work_dir"]) / "stage_result.json", result)
            atomic_write_json(Path(state.data["candidate_work_dir"]) / "evaluation.json", candidate)
            data = {
                **state.data,
                "pending_call": None,
                "experiment_baseline_evaluation": baseline.to_dict(),
                "experiment_candidate_evaluation": candidate.to_dict(),
                "candidate_stage_result": result.to_dict(),
                "candidate_effect_verification": verification.to_dict(),
            }
            return self._commit(
                state,
                ControllerState.EVALUATE,
                "CANDIDATE_CANARY_COMPLETED",
                "CANDIDATE_CANARY_COMPLETED",
                data=data,
                counters={"operator_calls": state.operator_calls + 1},
            ), "CANDIDATE_CANARY_COMPLETED"

        if state.state is ControllerState.EVALUATE:
            compiled = self._compiled_from_dict(state.data["compiled_experiment"])
            baseline = EvaluationReport.from_dict(state.data["experiment_baseline_evaluation"])
            candidate = EvaluationReport.from_dict(state.data["experiment_candidate_evaluation"])
            decision = self.gate.decide(
                baseline,
                candidate,
                targeted_metrics=compiled.required_evidence,
                progressive=True,
            )
            atomic_write_json(
                self.layout.campaign_dir / "experiments" / f"e{state.experiment_index:03d}" / "decision.json",
                decision,
            )
            data = {**state.data, "candidate_decision": decision.to_dict()}
            data["matched_candidate_decision_verified"] = bool(
                state.data.get("matched_candidate_decision_verified", False)
                or decision.decision in {CandidateDecision.ADOPT_CANDIDATE, CandidateDecision.ROLLBACK}
            )
            verification = state.data.get("candidate_effect_verification") or {}
            compiled = self._compiled_from_dict(state.data["compiled_experiment"])
            data["real_3d_experiment_verified"] = bool(
                state.data.get("real_3d_experiment_verified", False)
                or (
                    compiled.operator_id == "dataevolver.3d_asset_evolution"
                    and verification.get("status") == EvidenceStatus.VERIFIED.value
                )
            )
            target = (
                ControllerState.ADOPT_CANDIDATE
                if decision.decision is CandidateDecision.ADOPT_CANDIDATE
                else ControllerState.ROLLBACK
            )
            if decision.decision is CandidateDecision.ABSTAIN:
                return self._abstain(state.with_updates(data=data), "EVIDENCE_NOT_VERIFIED"), "EVIDENCE_NOT_VERIFIED"
            return self._commit(state, target, "CANDIDATE_DECIDED", decision.decision.value, data=data), decision.decision.value

        if state.state is ControllerState.ADOPT_CANDIDATE:
            valid, changed = self._frozen()
            if not valid:
                return self._pause_for_drift(state, changed), "PAUSED_CONTRACT_DRIFT"
            return self._commit(
                state,
                ControllerState.UPDATE,
                "INTERVENTION_ADOPTED",
                "INTERVENTION_ADOPTED",
                data={**state.data, "candidate_outcome": "adopted"},
            ), "INTERVENTION_ADOPTED"

        if state.state is ControllerState.ROLLBACK:
            return self._commit(
                state,
                ControllerState.UPDATE,
                "CANDIDATE_ROLLED_BACK",
                "CANDIDATE_ROLLED_BACK",
                data={**state.data, "candidate_outcome": "rolled_back"},
            ), "CANDIDATE_ROLLED_BACK"

        if state.state is ControllerState.UPDATE:
            compiled = self._compiled_from_dict(state.data["compiled_experiment"])
            planner_decision = PlannerDecision.from_dict(state.data["planner_decision"])
            baseline = EvaluationReport.from_dict(state.data["experiment_baseline_evaluation"])
            candidate = EvaluationReport.from_dict(state.data["experiment_candidate_evaluation"])
            attempted = list(state.data.get("attempted_strategy_hashes", []))
            if compiled.novelty_key not in attempted:
                attempted.append(compiled.novelty_key)
            history = list(state.data.get("experiment_history", []))
            history.append(
                {
                    "experiment_id": compiled.experiment_id,
                    "experiment_kind": compiled.experiment_kind.value,
                    "novelty_key": compiled.novelty_key,
                    "outcome": state.data["candidate_outcome"],
                    "decision": state.data["candidate_decision"],
                }
            )
            data = {
                **state.data,
                "attempted_strategy_hashes": attempted,
                "experiment_history": history,
                "requires_post_decision_replan": True,
            }
            if state.data["candidate_outcome"] == "adopted":
                result = StageResult.from_dict(state.data["candidate_stage_result"])
                work_dir = Path(state.data["candidate_work_dir"])
                data.update(
                    {
                        "last_stage_result": result.to_dict(),
                        "last_evaluation": candidate.to_dict(),
                        "last_effect_verification": state.data["candidate_effect_verification"],
                        "active_work_dir": str(work_dir),
                        "incumbent": {
                            "work_dir": str(work_dir),
                            "stage_result_ref": str(work_dir / "stage_result.json"),
                            "output_artifacts": list(result.output_artifacts),
                        },
                        "active_strategy": planner_decision.strategy_patch.to_dict(),
                    }
                )
                intervention = Intervention(
                    intervention_id=compiled.experiment_id,
                    failure_code=planner_decision.failure_code,
                    operator_id=compiled.operator_id,
                    level=InterventionLevel.L0,
                    parameters=compiled.operator_parameters,
                    candidate_delta=planner_decision.strategy_patch.to_dict(),
                    expected_effect=planner_decision.expected_effect,
                    rollback_plan={"kind": "retain_incumbent"},
                )
                capability = Capability(
                    capability_id=f"{state.campaign_id}:{compiled.novelty_key}",
                    version="1",
                    status=CapabilityStatus.CANDIDATE,
                    failure_signature=FailureSignature.from_dict(state.data["failure_signature"]),
                    intervention=intervention,
                    validation_evidence=(
                        str(self.layout.campaign_dir / "experiments" / f"e{state.experiment_index:03d}" / "decision.json"),
                    ),
                    context_signature=str(state.data["context_signature"]),
                    strategy_patch=planner_decision.strategy_patch,
                    experiment_kind=compiled.experiment_kind,
                    metric_delta=self._metric_deltas(baseline, candidate, "target_metrics"),
                    protected_metric_delta=self._metric_deltas(baseline, candidate, "protected_metrics"),
                    compatibility_hash=str(state.data["compatibility_hash"]),
                    source_asset_id=str(state.data["active_asset_id"]),
                    novelty_key=compiled.novelty_key,
                )
                self.capabilities.append_candidate(capability)
                state = self.store.commit(
                    state.with_updates(data=data),
                    "CAPABILITY_CANDIDATE_CREATED",
                    {"capability": capability.to_dict()},
                    f"capability:{capability.capability_id}",
                )
            else:
                self.capabilities.mark_known_failure(
                    compiled.novelty_key,
                    asset_id=str(state.data["active_asset_id"]),
                    compatibility_hash=str(state.data["compatibility_hash"]),
                    reason="; ".join(state.data["candidate_decision"].get("reasons", [])),
                )
                state = state.with_updates(data=data)
            return self._commit(
                state,
                ControllerState.PLAN,
                "CAMPAIGN_UPDATED",
                "REPLAN",
                data=state.data,
                counters={"experiment_index": state.experiment_index + 1},
                idempotency_key=f"experiment-updated:{compiled.idempotency_key}",
            ), "REPLAN"

        raise RuntimeError(f"unsupported agentic controller state: {state.state}")

    def _finish_recipe_search(self, state):
        """Finish the bounded search using one independent reuse slice.

        Adoption still comes exclusively from CandidateGate. A completed search
        can retain its baseline; completion is not a positive scientific claim.
        """
        if state.operator_calls >= self.goal.budgets.max_operator_calls:
            return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
        recipe = state.data.get("incumbent_recipe", self.goal.initial_operator_parameters["recipe"])
        call = self._make_call(call_id=f"{state.campaign_id}:reuse", operator_id=self.goal.initial_operator_id,
            work_dir=self.layout.campaign_dir/"reuse", key=self._trial_key("reuse_trial_key"),
            parameters={"recipe": recipe, "phase": "reuse"})
        try:
            state, result, verification = self._execute_call(state, call)
        except EffectRecoveryError:
            return self._abstain(state, "EFFECT_RECOVERY_AMBIGUOUS"), "EFFECT_RECOVERY_AMBIGUOUS"
        from .adapters.image_recipe import recipe_reuse_outcome

        data, capability = recipe_reuse_outcome(
            state, result, call,
            evaluator_contract=self.evaluator.contract,
            registry=self.registry,
            operator_id=self.goal.initial_operator_id,
            compatibility_hash=self._compatibility_hash,
        )
        if capability is not None:
            self.capabilities.append_candidate(capability)
        state = self._commit(state, ControllerState.UPDATE, "RECIPE_REUSE_VALIDATED", "RECIPE_SEARCH_COMPLETE",
            data=data, counters={"operator_calls": state.operator_calls+1}, idempotency_key="recipe:reuse-validated")
        return self._complete(state), "RECIPE_SEARCH_COMPLETE"

    def _tick_once(self, state):
        if state.state in TERMINAL_STATES:
            return state, "NO_OP"
        if self.layout.pause_request_path.exists() and state.state not in {ControllerState.PAUSED, ControllerState.PAUSED_CONTRACT_DRIFT}:
            return self._commit(
                state, ControllerState.PAUSED, "CAMPAIGN_PAUSED", "PAUSED",
                data={**state.data, "paused_from": state.state.value}
            ), "PAUSED"
        if state.state in {ControllerState.PAUSED, ControllerState.PAUSED_CONTRACT_DRIFT}:
            return state, "NO_OP"
        valid, changed = self._frozen()
        if not valid:
            return self._pause_for_drift(state, changed), "PAUSED_CONTRACT_DRIFT"
        if self._entry_admission is not None:
            self._entry_admission(self)
            state = self.store.load_state()
        if time.time() - datetime.fromisoformat(state.started_at).timestamp() > self.goal.budgets.max_wall_seconds:
            return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"

        if self._is_agentic():
            return self._tick_agentic(state)

        if state.state is ControllerState.INIT:
            return self._commit(state, ControllerState.PLAN, "CAMPAIGN_INITIALIZED", "PLAN_CREATED", idempotency_key="campaign:init"), "PLAN_CREATED"

        if state.state is ControllerState.PLAN:
            if state.operator_calls >= self.goal.budgets.max_operator_calls:
                return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
            if self.goal.initial_operator_id not in set(self.goal.resource_policy.get("allowed_operators", [])):
                return self._abstain(state, "OPERATOR_NOT_AUTHORIZED"), "OPERATOR_NOT_AUTHORIZED"
            call = self._make_call(
                call_id=f"{state.campaign_id}:r{state.round_index:03d}:initial",
                operator_id=self.goal.initial_operator_id,
                work_dir=self.layout.campaign_dir / "rounds" / f"r{state.round_index:03d}" / "execute",
                key=self._trial_key("initial_trial_key"),
                parameters=self._initial_parameters(state),
            )
            return self._commit(
                state, ControllerState.EXECUTE, "PLAN_CREATED", "INITIAL_OPERATOR_SELECTED",
                data={**state.data, "pending_call": call.to_dict()}
            ), "INITIAL_OPERATOR_SELECTED"

        if state.state is ControllerState.EXECUTE:
            call = OperatorCall.from_dict(state.data["pending_call"])
            try:
                state, result, verification = self._execute_call(state, call)
            except EffectRecoveryError:
                return self._abstain(state, "EFFECT_RECOVERY_AMBIGUOUS"), "EFFECT_RECOVERY_AMBIGUOUS"
            round_dir = self.layout.campaign_dir / "rounds" / f"r{state.round_index:03d}"
            atomic_write_json(round_dir / "stage_result.json", result)
            data = {
                **state.data,
                "pending_call": None,
                "last_stage_result": result.to_dict(),
                "last_effect_verification": verification.to_dict(),
            }
            return self._commit(
                state, ControllerState.OBSERVE, "EXECUTION_OBSERVED", "STAGE_RESULT_RECORDED", data=data,
                counters={"operator_calls": state.operator_calls + 1}
            ), "STAGE_RESULT_RECORDED"

        if state.state is ControllerState.OBSERVE:
            result = StageResult.from_dict(state.data["last_stage_result"])
            verification = EffectVerification(
                status=EvidenceStatus(state.data["last_effect_verification"]["status"]),
                reason=state.data["last_effect_verification"].get("reason", ""),
            )
            report = self.evaluator.evaluate(result, effect_status=verification.status)
            atomic_write_json(self.layout.campaign_dir / "rounds" / f"r{state.round_index:03d}" / "evaluation.json", report)
            return self._commit(
                state, ControllerState.DIAGNOSE, "EVALUATION_RECORDED", "EVALUATION_RECORDED",
                data={**state.data, "last_evaluation": report.to_dict()}
            ), "EVALUATION_RECORDED"

        if state.state is ControllerState.DIAGNOSE:
            result, report = StageResult.from_dict(state.data["last_stage_result"]), EvaluationReport.from_dict(state.data["last_evaluation"])
            recipe_mode = self.goal.planner.get("kind") == "recipe_rsi"
            if recipe_mode and (not report.all_required_verified() or any(v.value is not True for v in report.hard_constraints.values())):
                return self._abstain(state, "BASELINE_NOT_VERIFIED"), "BASELINE_NOT_VERIFIED"
            if not recipe_mode and self.goal.planner.get("kind") != "runtime_rsi" and self._goal_satisfied(report):
                return self._complete(state), "GOAL_SATISFIED"
            failure = self.planner.diagnose(result, report)
            return self._commit(
                state, ControllerState.PROPOSE, "FAILURE_DIAGNOSED", "FAILURE_SIGNATURE_CREATED",
                data={**state.data, "failure_signature": failure.to_dict(),
                      **({"incumbent_recipe": self.goal.initial_operator_parameters["recipe"]} if recipe_mode else {})}
            ), "FAILURE_SIGNATURE_CREATED"

        if state.state is ControllerState.PROPOSE:
            if self.goal.planner.get("kind") == "runtime_rsi":
                from .runtime_rsi import propose
                return propose(self, state)
            failure = FailureSignature.from_dict(state.data["failure_signature"])
            attempted = set(state.data.get("attempted_interventions", []))
            if self.goal.planner.get("kind") == "recipe_rsi":
                if (state.planner_calls >= self.goal.budgets.max_planner_calls or
                        state.experiments_used >= self.goal.budgets.max_experiments):
                    return self._finish_recipe_search(state)
                from dataevolver.adapters.model_gateway import ModelError
                try:
                    intervention = self.planner.propose_recipe(state)
                except (ValueError, ModelError) as exc:
                    state = self._commit(state, ControllerState.PROPOSE, "RECIPE_PROPOSAL_REJECTED", "INVALID_RECIPE",
                        data={**state.data, "last_proposal_error": str(exc)},
                        counters={"planner_calls": state.planner_calls+1, "experiments_used": state.experiments_used+1},
                        idempotency_key=f"recipe:proposal-rejected:{state.planner_calls}")
                    return state, "INVALID_RECIPE"
                if intervention is None:
                    return self._finish_recipe_search(state)
                state = state.with_updates(planner_calls=state.planner_calls+1,
                    data={**state.data, "attempted_rankings": [*state.data.get("attempted_rankings", []), intervention.candidate_delta["ranking"]]})
            else:
                intervention = self.planner.propose(failure, attempted)
            if intervention is None:
                return self._abstain(state, "NO_SAFE_ACTION"), "NO_SAFE_ACTION"
            if intervention.operator_id not in set(self.goal.resource_policy.get("allowed_operators", [])):
                return self._abstain(state, "OPERATOR_NOT_AUTHORIZED"), "OPERATOR_NOT_AUTHORIZED"
            data = {
                **state.data, "proposed_intervention": intervention.to_dict(),
                "attempted_interventions": sorted((*attempted, intervention.intervention_id)), "canary_phase": "baseline",
            }
            return self._commit(state, ControllerState.CANARY, "INTERVENTION_PROPOSED", "INTERVENTION_PROPOSED", data=data), "INTERVENTION_PROPOSED"

        if state.state is ControllerState.CANARY:
            if state.data.get("canary_phase") == "baseline" and state.experiments_used >= self.goal.budgets.max_experiments:
                return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
            experiment_dir = self.layout.campaign_dir / "experiments" / f"e{state.experiment_index:03d}"
            expected_key, intervention = self._trial_key("canary_trial_key"), Intervention.from_dict(state.data["proposed_intervention"])
            if state.data.get("canary_phase") == "baseline":
                initial_report = EvaluationReport.from_dict(state.data["last_evaluation"])
                initial_result = StageResult.from_dict(state.data["last_stage_result"])
                data = dict(state.data)
                if can_reuse_baseline(initial_report, initial_report.matched_trial_key, expected_key):
                    baseline_report, mode = initial_report, "BASELINE_REUSED"
                    atomic_write_json(experiment_dir / "baseline" / "evaluation.json", initial_report)
                    atomic_write_json(experiment_dir / "baseline" / "stage_result.json", initial_result)
                else:
                    if state.operator_calls >= self.goal.budgets.max_operator_calls:
                        return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
                    call = self._make_call(
                        call_id=f"{state.campaign_id}:e{state.experiment_index:03d}:baseline",
                        operator_id=self.goal.initial_operator_id, work_dir=experiment_dir / "baseline", key=expected_key,
                        parameters=self._initial_parameters(state),
                    )
                    try:
                        state, result, verification = self._execute_call(state, call)
                    except EffectRecoveryError:
                        return self._abstain(state, "EFFECT_RECOVERY_AMBIGUOUS"), "EFFECT_RECOVERY_AMBIGUOUS"
                    baseline_report, mode = self.evaluator.evaluate(result, effect_status=verification.status), "BASELINE_REMATERIALIZED"
                    atomic_write_json(experiment_dir / "baseline" / "stage_result.json", result)
                    atomic_write_json(experiment_dir / "baseline" / "evaluation.json", baseline_report)
                    state = state.with_updates(operator_calls=state.operator_calls + 1)
                atomic_write_json(experiment_dir / "matched_trial.json", {
                    "mode": mode, "matched_trial_key": expected_key.to_dict(), "candidate_delta": intervention.candidate_delta,
                })
                atomic_write_json(experiment_dir / "proposal.json", intervention)
                data.update({
                    "pending_call": None,
                    "canary_phase": "candidate",
                    "experiment_baseline_evaluation": baseline_report.to_dict(),
                    "baseline_mode": mode,
                })
                return self._commit(
                    state, ControllerState.CANARY, mode, mode, data=data,
                    counters={"experiments_used": state.experiments_used + 1, "operator_calls": state.operator_calls},
                    idempotency_key=f"{mode.lower()}:{state.experiment_index}"
                ), mode

            if state.operator_calls >= self.goal.budgets.max_operator_calls:
                return self._abstain(state, "BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED"
            call = self._make_call(
                call_id=f"{state.campaign_id}:e{state.experiment_index:03d}:candidate",
                operator_id=intervention.operator_id, work_dir=experiment_dir / "candidate", key=expected_key,
                parameters=intervention.parameters,
            )
            try:
                state, result, verification = self._execute_call(state, call)
            except EffectRecoveryError:
                return self._abstain(state, "EFFECT_RECOVERY_AMBIGUOUS"), "EFFECT_RECOVERY_AMBIGUOUS"
            candidate_report = self.evaluator.evaluate(result, effect_status=verification.status)
            atomic_write_json(experiment_dir / "candidate" / "stage_result.json", result)
            atomic_write_json(experiment_dir / "candidate" / "evaluation.json", candidate_report)
            return self._commit(
                state, ControllerState.EVALUATE, "CANDIDATE_CANARY_COMPLETED", "CANDIDATE_CANARY_COMPLETED",
                data={
                    **state.data,
                    "pending_call": None,
                    "experiment_candidate_evaluation": candidate_report.to_dict(),
                    "experiment_candidate_stage_result": result.to_dict(),
                },
                counters={"operator_calls": state.operator_calls + 1}
            ), "CANDIDATE_CANARY_COMPLETED"

        if state.state is ControllerState.EVALUATE:
            decision = self.gate.decide(
                EvaluationReport.from_dict(state.data["experiment_baseline_evaluation"]),
                EvaluationReport.from_dict(state.data["experiment_candidate_evaluation"]),
            )
            atomic_write_json(self.layout.campaign_dir / "experiments" / f"e{state.experiment_index:03d}" / "decision.json", decision)
            data = {**state.data, "candidate_decision": decision.to_dict()}
            if decision.decision is CandidateDecision.ADOPT_CANDIDATE:
                target = ControllerState.ADOPT_CANDIDATE
            elif decision.decision is CandidateDecision.ROLLBACK:
                target = ControllerState.ROLLBACK
            else:
                return self._abstain(state.with_updates(data=data), "EVIDENCE_NOT_VERIFIED"), "EVIDENCE_NOT_VERIFIED"
            return self._commit(state, target, "CANDIDATE_DECIDED", decision.decision.value, data=data), decision.decision.value

        if state.state is ControllerState.ADOPT_CANDIDATE:
            valid, changed = self._frozen()
            if not valid:
                return self._pause_for_drift(state, changed), "PAUSED_CONTRACT_DRIFT"
            return self._commit(
                state, ControllerState.UPDATE, "INTERVENTION_ADOPTED", "INTERVENTION_ADOPTED",
                data={**state.data, "adopted_intervention": state.data["proposed_intervention"]}
            ), "INTERVENTION_ADOPTED"

        if state.state is ControllerState.ROLLBACK:
            return self._commit(state, ControllerState.UPDATE, "CANDIDATE_ROLLED_BACK", "CANDIDATE_ROLLED_BACK"), "CANDIDATE_ROLLED_BACK"

        if state.state is ControllerState.UPDATE:
            if self.goal.planner.get("kind") == "runtime_rsi":
                from .runtime_rsi import update
                return update(self, state)
            if self.goal.planner.get("kind") == "recipe_rsi":
                intervention = Intervention.from_dict(state.data["proposed_intervention"])
                decision = state.data["candidate_decision"]
                data = {**state.data, "recipe_history": [*state.data.get("recipe_history", []),
                    {"ranking": intervention.candidate_delta["ranking"], "decision": decision["decision"],
                     "reasons": decision["reasons"]}]}
                if decision["decision"] == "ADOPT_CANDIDATE":
                    data.update(incumbent_recipe=intervention.parameters["recipe"],
                                last_evaluation=state.data["experiment_candidate_evaluation"],
                                last_stage_result=state.data["experiment_candidate_stage_result"])
                return self._commit(state, ControllerState.PROPOSE, "RECIPE_COMPARISON_RECORDED", "REPLAN", data=data,
                    counters={"experiment_index": state.experiment_index+1, "round_index": state.round_index+1},
                    idempotency_key=f"recipe:updated:{state.experiment_index}"), "REPLAN"
            if "adopted_intervention" not in state.data:
                return self._abstain(state, "CANDIDATE_REJECTED_NO_SAFE_ACTION"), "CANDIDATE_REJECTED_NO_SAFE_ACTION"
            intervention = Intervention.from_dict(state.data["adopted_intervention"])
            capability = Capability(
                capability_id=f"{state.campaign_id}:{intervention.intervention_id}", version="1",
                status=CapabilityStatus.CANDIDATE,
                failure_signature=FailureSignature.from_dict(state.data["failure_signature"]), intervention=intervention,
                validation_evidence=(str(self.layout.campaign_dir / "experiments" / f"e{state.experiment_index:03d}" / "decision.json"),),
            )
            self.capabilities.append_candidate(capability)
            state = self.store.commit(
                state, "CAPABILITY_CANDIDATE_CREATED", {"capability": capability.to_dict(), "state_after": state.to_dict()},
                f"capability:{capability.capability_id}",
            )
            return self._complete(state), "CAPABILITY_CANDIDATE_CREATED"
        raise RuntimeError(f"unsupported controller state: {state.state}")

    def tick(self, max_actions=1):
        if max_actions < 1:
            raise ValueError("max_actions must be >= 1")
        with CampaignLease(self.layout.lease_path, ttl_seconds=self.goal.lease_seconds):
            initial = self.store.load_state()
            state, effect, count = initial, "NO_OP", 0
            for _ in range(max_actions):
                next_state, effect = self._tick_once(state)
                if next_state == state:
                    break
                state, count = next_state, count + 1
                if state.state in TERMINAL_STATES | {ControllerState.PAUSED, ControllerState.PAUSED_CONTRACT_DRIFT}:
                    break
            return self._receipt(initial.state, state, effect, count)

    def run(self):
        budget=self.goal.budgets
        action_bound=max(1000,12*budget.max_operator_calls+6*budget.max_planner_calls+20*budget.max_experiments)
        for _ in range(min(100000,action_bound)):
            state = self.store.load_state()
            if state.state in TERMINAL_STATES | {ControllerState.PAUSED, ControllerState.PAUSED_CONTRACT_DRIFT}:
                return state
            self.tick(1)
        return self._abstain(self.store.load_state(), "CONTROLLER_STEP_LIMIT")

    def pause(self):
        atomic_write_json(self.layout.pause_request_path, {"requested_at": _now()})
        return self.store.load_state()

    def resume(self):
        state = self.store.load_state()
        if self.layout.pause_request_path.exists():
            self.layout.pause_request_path.unlink()
        if state.state is ControllerState.PAUSED_CONTRACT_DRIFT and not self._frozen()[0]:
            return state
        if state.state in {ControllerState.PAUSED, ControllerState.PAUSED_CONTRACT_DRIFT}:
            target = ControllerState(state.data.get("paused_from", "PLAN"))
            state = self._commit(state, target, "CAMPAIGN_RESUMED", "RESUMED")
        return self.run()

    def status(self):
        return self.store.load_state()

    def replay(self):
        return self.store.replay()

    def explain(self, limit=20):
        return {
            "state": self.store.load_state().to_dict(),
            "events": [event.to_dict() for event in self.store.events.read()[-max(1, limit):]],
        }
