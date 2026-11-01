from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .evaluator import EvaluatorContract
from .models import EvaluationReport, EvidenceStatus, JsonModel, MatchedTrialKey


class CandidateDecision(str, Enum):
    ADOPT_CANDIDATE = "ADOPT_CANDIDATE"
    ROLLBACK = "ROLLBACK"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class DecisionRecord(JsonModel):
    decision: CandidateDecision
    reasons: tuple[str, ...]


def can_reuse_baseline(
    baseline: EvaluationReport,
    baseline_key: MatchedTrialKey,
    expected_key: MatchedTrialKey,
) -> bool:
    return (
        baseline_key.digest() == expected_key.digest()
        and baseline.matched_trial_key is not None
        and baseline.matched_trial_key.digest() == expected_key.digest()
        and baseline.all_required_verified()
    )


class CandidateGate:
    def __init__(self, contract: EvaluatorContract):
        self.contract = contract

    def decide(
        self,
        baseline: EvaluationReport,
        candidate: EvaluationReport,
        *,
        targeted_metrics: tuple[str, ...] = (),
        progressive: bool = False,
    ) -> DecisionRecord:
        if baseline.matched_trial_key is None or candidate.matched_trial_key is None:
            return DecisionRecord(CandidateDecision.ABSTAIN, ("matched trial key is missing",))
        if baseline.matched_trial_key.digest() != candidate.matched_trial_key.digest():
            return DecisionRecord(CandidateDecision.ABSTAIN, ("matched trial key mismatch",))
        for report_name, report in (("baseline", baseline), ("candidate", candidate)):
            unverified = [value.metric for value in report.required_evidence() if value.status is not EvidenceStatus.VERIFIED]
            if unverified:
                return DecisionRecord(
                    CandidateDecision.ABSTAIN,
                    (f"{report_name} required evidence is not VERIFIED: {', '.join(unverified)}",),
                )

        reasons: list[str] = []
        for name, value in candidate.hard_constraints.items():
            if value.value is not True:
                reasons.append(f"hard constraint failed: {name}")
        for name, spec in self.contract.payload.get("target_metrics", {}).items():
            baseline_value = float(baseline.target_metrics[name].value)
            candidate_value = float(candidate.target_metrics[name].value)
            if progressive:
                if not targeted_metrics:
                    reasons.append("progressive adoption requires targeted metrics")
                    continue
                if name in targeted_metrics:
                    improvement = float(spec.get("minimum_improvement", 0.0))
                    if candidate_value - baseline_value < improvement:
                        reasons.append(f"target metric improvement too small: {name}")
                else:
                    tolerance = float(spec.get("max_regression", 0.0))
                    if candidate_value < baseline_value - tolerance:
                        reasons.append(f"non-target metric regressed: {name}")
            else:
                minimum = float(spec.get("minimum", float("-inf")))
                improvement = float(spec.get("minimum_improvement", 0.0))
                if candidate_value < minimum:
                    reasons.append(f"target metric below minimum: {name}")
                if candidate_value - baseline_value < improvement:
                    reasons.append(f"target metric improvement too small: {name}")
        for name, spec in self.contract.payload.get("protected_metrics", {}).items():
            baseline_value = float(baseline.protected_metrics[name].value)
            candidate_value = float(candidate.protected_metrics[name].value)
            tolerance = float(spec.get("max_regression", 0.0))
            if candidate_value < baseline_value - tolerance:
                reasons.append(f"protected metric regressed: {name}")
        if reasons:
            return DecisionRecord(CandidateDecision.ROLLBACK, tuple(reasons))
        return DecisionRecord(CandidateDecision.ADOPT_CANDIDATE, ("all adoption gates passed",))
