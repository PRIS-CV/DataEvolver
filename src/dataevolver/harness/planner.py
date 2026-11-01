from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import (
    EvaluationReport,
    FailureSignature,
    Intervention,
    PlannerContext,
    PlannerDecision,
    PlannerDecisionKind,
    StageResult,
    canonical_json,
)


@runtime_checkable
class AgentPlanner(Protocol):
    planner_id: str

    def diagnose(self, result: StageResult, report: EvaluationReport) -> FailureSignature: ...

    def propose(self, failure: FailureSignature, attempted: set[str]) -> Intervention | None: ...


@runtime_checkable
class PlannerBackend(Protocol):
    backend_id: str
    model_id: str

    def complete_json(self, messages: list[dict[str, str]], *, schema: dict[str, Any]) -> str: ...


class OpenAICompatiblePlannerBackend:
    backend_id = "openai-compatible"

    def __init__(self, config: dict[str, Any]):
        self.base_url_env = str(config.get("base_url_env") or "DATAEVOLVER_AGENT_BASE_URL")
        self.api_key_env = str(config.get("api_key_env") or "DATAEVOLVER_AGENT_API_KEY")
        self.model_env = str(config.get("model_env") or "DATAEVOLVER_AGENT_MODEL")
        self.timeout_seconds = int(config.get("timeout_seconds", 120))

    @property
    def model_id(self) -> str:
        return os.environ.get(self.model_env, "").strip()

    def complete_json(self, messages: list[dict[str, str]], *, schema: dict[str, Any]) -> str:
        base_url = os.environ.get(self.base_url_env, "").strip()
        api_key = os.environ.get(self.api_key_env, "").strip()
        model = self.model_id
        missing = [
            name for name, value in (
                (self.base_url_env, base_url), (self.api_key_env, api_key), (self.model_env, model)
            ) if not value
        ]
        if missing:
            raise RuntimeError(f"planner backend environment is missing: {', '.join(missing)}")
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        wire_messages = [dict(message) for message in messages]
        if not wire_messages or wire_messages[0]["role"] != "system":
            wire_messages.insert(0, {"role": "system", "content": ""})
        wire_messages[0]["content"] += "\n\n" + canonical_json({"response_schema": schema})
        body = json.dumps(
            {
                "model": model,
                # The prompt supplies the contract; local decision validation remains authoritative.
                "messages": wire_messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"planner backend request failed: {exc}") from exc
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("planner backend response is missing message content") from exc


PLANNER_SYSTEM_PROMPT = """You are DataEvolver's bounded 3D experiment planner.
Return one strict JSON PlannerDecision. Diagnose the earliest failing data-plane stage and choose only an allowed
ExperimentKind plus StrategyPatch. Never emit commands, paths, Python, operator IDs, evaluator changes, budgets,
thresholds, or registry changes. Do not repeat attempted strategy hashes. Prefer a compatible learned capability
only as a hypothesis; every strategy must still pass a matched canary. If no safe novel action exists, ABSTAIN.
The objective is final 3D and representative-view quality, not merely a visually pleasing T2I image.
"""


class AgenticRSIPlanner:
    planner_id = "agentic-rsi-v1"

    def __init__(self, *, backend: PlannerBackend, config: dict[str, Any]):
        self.backend = backend
        self._config = dict(config)
        self._decision_schema = self.decision_schema()
        self._last_trace: dict[str, Any] = {}

    def last_trace(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._last_trace, ensure_ascii=False))

    def config(self) -> dict[str, Any]:
        base_url = ""
        if hasattr(self.backend, "base_url_env"):
            base_url = os.environ.get(self.backend.base_url_env, "").strip()
        return {
            **self._config,
            "backend_id": self.backend.backend_id,
            "model_id": self.backend.model_id,
            "base_url_hash": __import__("hashlib").sha256(base_url.encode("utf-8")).hexdigest() if base_url else "",
            "system_prompt": PLANNER_SYSTEM_PROMPT,
            "decision_schema": json.loads(json.dumps(self._decision_schema)),
        }

    def config_hash(self) -> str:
        import hashlib

        return hashlib.sha256(canonical_json(self.config()).encode("utf-8")).hexdigest()

    @staticmethod
    def decision_schema() -> dict[str, Any]:
        root = Path(__file__).resolve().parents[3] / "configs" / "schemas"
        schema = json.loads((root / "harness_planner_decision.schema.json").read_text(encoding="utf-8"))
        patch_schema = json.loads((root / "harness_strategy_patch.schema.json").read_text(encoding="utf-8"))
        schema["properties"]["strategy_patch"]["oneOf"][0] = patch_schema
        return schema

    @staticmethod
    def _abstain(context: PlannerContext, reason: str) -> PlannerDecision:
        return PlannerDecision(
            decision_id="planner-abstain",
            decision=PlannerDecisionKind.ABSTAIN,
            asset_id=context.asset_id,
            failure_code=context.failure_signature.code,
            experiment_kind=None,
            hypothesis="",
            strategy_patch=None,
            expected_effect={},
            required_evidence=(),
            confidence="low",
            rationale_summary=reason[:512],
        )

    @staticmethod
    def _validate_context(decision: PlannerDecision, context: PlannerContext) -> None:
        if decision.asset_id != context.asset_id:
            raise ValueError("PlannerDecision asset_id does not match context")
        if decision.failure_code != context.failure_signature.code:
            raise ValueError("PlannerDecision failure_code does not match context")
        if decision.decision is PlannerDecisionKind.RUN_EXPERIMENT:
            if decision.experiment_kind not in context.allowed_experiment_kinds:
                raise ValueError("PlannerDecision experiment kind is not allowed")
            if int(context.remaining_budget.get("experiments", 0)) <= 0:
                raise ValueError("PlannerDecision exceeds experiment budget")
            unknown_actions = set(decision.strategy_patch.scene_actions) - set(context.allowed_scene_actions)
            if unknown_actions:
                raise ValueError(f"PlannerDecision scene action is not registered: {sorted(unknown_actions)[0]}")

    def plan(self, context: PlannerContext) -> PlannerDecision:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": canonical_json(context)},
        ]
        schema = json.loads(json.dumps(self._decision_schema))
        trace = {
            "backend_id": self.backend.backend_id,
            "model_id": self.backend.model_id,
            "context_hash": context.sha256(),
            "raw_responses": [],
            "repair_attempted": False,
            "validation_errors": [],
            "outcome": "STARTED",
        }
        try:
            raw = self.backend.complete_json(messages, schema=schema)
        except Exception as exc:
            trace.update({"outcome": "BACKEND_ERROR", "error": str(exc)[:1000]})
            self._last_trace = trace
            return self._abstain(context, str(exc))
        trace["raw_responses"].append(raw)
        for attempt in range(2):
            try:
                decision = PlannerDecision.from_dict(json.loads(raw))
                self._validate_context(decision, context)
                trace["outcome"] = "VALID_DECISION"
                self._last_trace = trace
                return decision
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                trace["validation_errors"].append(str(exc)[:1000])
                if attempt == 1:
                    trace["outcome"] = "INVALID_AFTER_REPAIR"
                    self._last_trace = trace
                    return self._abstain(context, f"planner output invalid after repair: {exc}")
                trace["repair_attempted"] = True
                repair = {
                    "validation_error": str(exc),
                    "instruction": "Return a corrected PlannerDecision JSON object only.",
                }
                messages = [*messages, {"role": "assistant", "content": raw}, {"role": "user", "content": canonical_json(repair)}]
                try:
                    raw = self.backend.complete_json(messages, schema=schema)
                except Exception as repair_exc:
                    trace.update({"outcome": "REPAIR_BACKEND_ERROR", "error": str(repair_exc)[:1000]})
                    self._last_trace = trace
                    return self._abstain(context, str(repair_exc))
                trace["raw_responses"].append(raw)
        trace["outcome"] = "NO_DECISION"
        self._last_trace = trace
        return self._abstain(context, "planner did not return a decision")


class DeterministicPlanner:
    planner_id = "deterministic-v1"

    def __init__(self, interventions):
        self.interventions = tuple(
            item if isinstance(item, Intervention) else Intervention.from_dict(item) for item in interventions
        )

    def config(self) -> dict:
        return {"interventions": [item.to_dict() for item in self.interventions]}

    def config_hash(self) -> str:
        import hashlib

        return hashlib.sha256(canonical_json(self.config()).encode("utf-8")).hexdigest()

    def diagnose(self, result: StageResult, report: EvaluationReport) -> FailureSignature:
        if result.failure_code:
            return FailureSignature(result.failure_code, detail=result.failure_detail, evidence_refs=result.evidence_refs)
        failed_hard = [name for name, value in report.hard_constraints.items() if value.value is not True]
        if failed_hard:
            return FailureSignature("HARD_CONSTRAINT_FAILED", detail=",".join(failed_hard))
        return FailureSignature("TARGET_NOT_MET")

    def propose(self, failure: FailureSignature, attempted: set[str]) -> Intervention | None:
        return next(
            (
                item
                for item in self.interventions
                if item.failure_code == failure.code and item.intervention_id not in attempted
            ),
            None,
        )
