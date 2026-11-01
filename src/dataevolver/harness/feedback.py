"""Typed feedback and bounded strategy identities; none grants execution authority."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from dataevolver.dataset.contracts import ArtifactRef, Contract, canonical_json, identifier


@dataclass(frozen=True)
class FeedbackSignal(Contract):
    schema: ClassVar[str] = "dataevolver.feedback_signal.v1"
    kind: str
    evidence: tuple[ArtifactRef, ...]
    scope: dict
    metrics: dict = field(default_factory=dict)
    uncertainty: str = "unknown"

    def __post_init__(self):
        if self.kind not in {"capability_gap", "sample_quality", "dataset_quality", "training_result"}:
            raise ValueError("unknown feedback kind")
        if not self.evidence or any(not isinstance(e, ArtifactRef) for e in self.evidence):
            raise ValueError("feedback requires bound evidence")
        if not self.scope or self.uncertainty not in {"supported", "uncertain", "unknown"}:
            raise ValueError("feedback needs scope and explicit uncertainty")
        canonical_json([self.scope, self.metrics])

    @classmethod
    def from_dict(cls, value):
        values = cls.read_fields(value)
        values['evidence'] = tuple(ArtifactRef.from_dict(x) for x in values['evidence'])
        return cls(**values)

    def validate_evidence(self, store):
        for ref in self.evidence:
            store.resolve(ref)
        return self

    def to_dict(self):
        value = super().to_dict()
        value['evidence'] = [ref.to_dict() for ref in self.evidence]
        return value


@dataclass(frozen=True)
class ChangeTarget(Contract):
    schema: ClassVar[str] = "dataevolver.change_target.v1"
    kind: str
    identity: str
    version: str
    source_sha256: str

    def __post_init__(self):
        if self.kind not in {"recipe", "operator", "policy", "experience"}:
            raise ValueError("unknown change target")
        identifier(self.identity); identifier(self.version)
        if len(self.source_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.source_sha256):
            raise ValueError("change target needs an exact source hash")


@dataclass(frozen=True)
class PolicySpec(Contract):
    schema: ClassVar[str] = "dataevolver.policy_spec.v1"
    policy_id: str
    version: str
    role: str
    config: dict

    def __post_init__(self):
        identifier(self.policy_id); identifier(self.version)
        if self.role not in {"execution", "evolution"}:
            raise ValueError("unknown policy role")
        if set(self.config) - {"observation_order", "candidate_limit", "stop_rule"}:
            raise ValueError("strategy cannot change frozen budgets, evaluator or permissions")
        order = self.config.get('observation_order', [])
        if not isinstance(order, list) or any(not isinstance(s, str) for s in order):
            raise ValueError("observation_order must be a list of operation IDs")
        limit = self.config.get('candidate_limit', 1)
        if type(limit) is not int or limit < 0:
            raise ValueError("candidate_limit must be nonnegative")
        if self.config.get('stop_rule', 'goal_or_budget') not in {'goal_or_budget', 'no_improvement'}:
            raise ValueError("unknown stopping strategy")
        canonical_json(self.config)


class PolicyResolver:
    """Resolve a frozen package using the existing lifecycle authority."""
    def __init__(self, version_store, authority):
        self.store, self.authority = version_store, authority

    def resolve(self, adoption_event, *, scope, expected_head, policy, controller=None):
        snapshot = self.store.active(adoption_event, scope=scope, head=expected_head,
                                     qualification_authority=self.authority)
        self.authority.verify(snapshot)
        value = PolicySpec.from_dict(policy)
        if snapshot.get('policy_sha256') != value.compute_hash():
            raise ValueError("policy version differs from qualified snapshot")
        if 'policy_context_sha256' in snapshot:
            from .policy_qualification import policy_context
            from dataevolver.dataset.contracts import payload_sha256
            if controller is None or not controller._frozen()[0]:
                raise ValueError('qualified policy requires its actual frozen consumer Controller')
            state = controller.status()
            if (state.state.value!='INIT' or state.operator_calls or state.planner_calls or state.experiments_used
                    or controller.planner.policy.to_dict()!=value.to_dict()
                    or snapshot['policy_context_sha256']!=payload_sha256(policy_context(controller))):
                raise ValueError('qualified policy cannot change an active batch or its qualified context')
        return value
