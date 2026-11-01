"""Small domain bindings; transition proposals are committed by the Controller."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Protocol

from dataevolver.dataset.contracts import canonical_json, payload_sha256


class PlanCompiler(Protocol):
    def compile(self, plan: dict, context: Any) -> dict: ...


class DomainRuntime(Protocol):
    """Frozen configuration and bounded observation/intervention compilation."""

    def config(self) -> dict: ...
    def context(self, state) -> dict: ...
    def compile_observation(self, decision: dict, state) -> Any: ...
    def compile_intervention(self, decision: dict, state) -> Any: ...


@dataclass(frozen=True)
class DomainBinding:
    compiler: PlanCompiler
    runtime: DomainRuntime | None = None
    evaluator: Any = None
    delivery: Any = None


class LegacyDomainAdapter:
    """Keep existing optional hooks and frozen config exactly as declared."""
    def __init__(self, domain):
        for name in ("config", "context"):
            if not callable(getattr(domain, name, None)):
                raise TypeError(f"domain is missing {name}")
        self.domain = domain

    def __getattr__(self, name):
        return getattr(self.domain, name)

    def config(self):
        return self.domain.config()

    def context(self, state):
        return self.domain.context(state)

    def compile_observation(self, decision, state):
        if not callable(getattr(self.domain, 'compile_observation', None)):
            raise ValueError('domain has no observation compiler')
        return self.domain.compile_observation(decision, state)

    def compile_intervention(self, decision, state):
        if not callable(getattr(self.domain, 'compile_intervention', None)):
            raise ValueError('domain has no intervention compiler')
        return self.domain.compile_intervention(decision, state)


@dataclass(frozen=True)
class TransitionProposal:
    source_state_sha256: str
    data: dict
    complete: bool = False
    event_type: str = "WORKFLOW_STAGE_ACCEPTED"
    effect: str = "NEXT_FROZEN_TASK"
    counters: dict = field(default_factory=dict)
    completion_evidence: dict = field(default_factory=dict)

    @classmethod
    def from_state(cls, state, data, **kwargs):
        return cls(payload_sha256(state.to_dict()), copy.deepcopy(data), **kwargs)

    def validate(self, state):
        if self.source_state_sha256 != payload_sha256(state.to_dict()):
            raise ValueError("domain transition belongs to a different Controller state")
        if (type(self.complete) is not bool or not self.event_type or not self.effect
                or set(self.counters) - {"round_index"}):
            raise ValueError("domain transition cannot change protected state or budgets")
        if self.counters and self.counters != {"round_index": state.round_index + 1}:
            raise ValueError("domain transition must advance one round")
        if not self.completion_evidence:
            raise ValueError("domain transition needs completion evidence")
        canonical_json([self.data, self.completion_evidence])
