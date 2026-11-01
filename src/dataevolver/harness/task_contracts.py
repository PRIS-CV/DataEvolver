"""General visual-task contracts; execution remains owned by EvolutionController."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from dataevolver.dataset.contracts import ArtifactRef, Contract, canonical_json, identifier, positive_int


PROTECTED_GATES = frozenset({"integrity", "provenance", "license", "privacy", "safety", "duplicates", "leakage"})


def _strings(values, name):
    if not isinstance(values, (list, tuple)) or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"{name} must be a sequence of non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicates")
    return tuple(values)


@dataclass(frozen=True)
class DatasetIntent(Contract):
    schema: ClassVar[str] = "dataevolver.dataset_intent.v1"
    intent_id: str
    mode: str
    task: str
    output_schema: str
    description: str
    sources: tuple[ArtifactRef, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.intent_id, "intent_id")
        identifier(self.task, "task")
        if self.mode not in {"curate", "generate", "hybrid"}:
            raise ValueError("intent mode must be curate, generate or hybrid")
        if not self.description.strip() or not self.output_schema.strip():
            raise ValueError("description and output_schema are required")
        if any(not isinstance(x, ArtifactRef) for x in self.sources):
            raise ValueError("intent sources must be ArtifactRef values")
        if self.mode in {"curate", "hybrid"} and not self.sources:
            raise ValueError("curation and hybrid intents require bound source references")
        if not isinstance(self.parameters, dict):
            raise ValueError("intent parameters must be an object")
        canonical_json(self.parameters)

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["sources"] = tuple(ArtifactRef.from_dict(x) for x in values.get("sources", []))
        return cls(**values)

    def to_dict(self):
        result = super().to_dict()
        result["sources"] = [x.to_dict() for x in self.sources]
        return result


@dataclass(frozen=True)
class QualityPolicy(Contract):
    schema: ClassVar[str] = "dataevolver.quality_policy.v1"
    policy_id: str
    required_gates: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    task_criteria: dict[str, Any] = field(default_factory=dict)
    max_model_calls: int = 1

    def __post_init__(self):
        identifier(self.policy_id, "policy_id")
        object.__setattr__(self, "required_gates", _strings(self.required_gates, "required_gates"))
        object.__setattr__(self, "allowed_operations", _strings(self.allowed_operations, "allowed_operations"))
        if not PROTECTED_GATES.issubset(self.required_gates):
            raise ValueError("protected gates cannot be removed")
        if not self.allowed_operations or not isinstance(self.task_criteria, dict):
            raise ValueError("allowed_operations and task_criteria must be explicit")
        positive_int(self.max_model_calls, "max_model_calls")
        canonical_json(self.task_criteria)


@dataclass(frozen=True)
class NodeSpec(Contract):
    schema: ClassVar[str] = "dataevolver.node_spec.v1"
    node_id: str
    operator: str
    operator_version: str
    role: str
    dependencies: tuple[str, ...] = ()
    inputs: tuple[ArtifactRef, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("node_id", "operator", "operator_version", "role"):
            identifier(getattr(self, name), name)
        object.__setattr__(self, "dependencies", _strings(self.dependencies, "dependencies"))
        if self.node_id in self.dependencies or any(not isinstance(x, ArtifactRef) for x in self.inputs):
            raise ValueError("invalid node dependencies or inputs")
        if not isinstance(self.parameters, dict):
            raise ValueError("node parameters must be an object")
        canonical_json(self.parameters)

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["inputs"] = tuple(ArtifactRef.from_dict(x) for x in values.get("inputs", []))
        return cls(**values)

    def to_dict(self):
        result = super().to_dict()
        result["inputs"] = [x.to_dict() for x in self.inputs]
        return result


@dataclass(frozen=True)
class NodeSpecGraph(Contract):
    schema: ClassVar[str] = "dataevolver.node_graph.v1"
    nodes: tuple[NodeSpec, ...]

    def __post_init__(self):
        if not self.nodes or any(not isinstance(x, NodeSpec) for x in self.nodes):
            raise ValueError("node graph requires typed nodes")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node IDs must be unique")
        if any(set(node.dependencies) - set(ids) for node in self.nodes):
            raise ValueError("node dependency is missing")
        self.ordered()

    def ordered(self) -> tuple[NodeSpec, ...]:
        ordered, pending = [], list(self.nodes)
        while pending:
            ready = [node for node in pending if set(node.dependencies).issubset({x.node_id for x in ordered})]
            if not ready:
                raise ValueError("node graph contains a cycle")
            for node in ready:
                ordered.append(node)
                pending.remove(node)
        return tuple(ordered)

    def validate_policy(self, policy: QualityPolicy, roles: set[str]):
        if len(self.nodes) > policy.max_model_calls:
            raise ValueError("node graph exceeds the approved model-call budget")
        for node in self.nodes:
            if node.operator not in policy.allowed_operations or node.role not in roles:
                raise ValueError("node operator or model role is not approved")

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["nodes"] = tuple(NodeSpec.from_dict(x) for x in values["nodes"])
        return cls(**values)

    def to_dict(self):
        return {"schema_version": self.schema, "nodes": [x.to_dict() for x in self.nodes]}


@dataclass(frozen=True)
class NodeRun(Contract):
    schema: ClassVar[str] = "dataevolver.node_run.v1"
    campaign_id: str
    call_id: str
    node_id: str
    node_spec_hash: str
    request_id: str
    request_hash: str
    receipt_hash: str
    output: ArtifactRef
    status: str = "VERIFIED"

    def __post_init__(self):
        identifier(self.campaign_id, "campaign_id")
        identifier(self.node_id, "node_id")
        identifier(self.request_id, "request_id")
        if self.status != "VERIFIED" or not isinstance(self.output, ArtifactRef):
            raise ValueError("NodeRun requires a verified, bound output")
        for name in ("node_spec_hash", "request_hash", "receipt_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a SHA-256 digest")

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["output"] = ArtifactRef.from_dict(values["output"])
        return cls(**values)

    def to_dict(self):
        result = super().to_dict()
        result["output"] = self.output.to_dict()
        return result


def adapt_raw_image_pair_contracts(intent: Mapping, policy: Mapping, *, source: ArtifactRef):
    """Bridge local legacy intent/policy without importing its autonomous workflow.

    Caller supplies the immutable catalog reference; legacy absolute source_root
    is never persisted as a new framework contract or silently read from disk.
    """
    if intent.get("schema_version") != "raw_image_pairs.dataset_intent.v1":
        raise ValueError("unsupported legacy intent schema")
    if policy.get("schema_version") != "raw_image_pairs.quality_policy.v1":
        raise ValueError("unsupported legacy policy schema")
    if intent.get("license_policy", "require_known") != "require_known" or intent.get("safety_policy", "require_known") != "require_known":
        raise ValueError("legacy warning policies require a separately reviewed policy")
    parameters = {key: intent[key] for key in (
        "task_types", "allowed_changes", "preserved_content", "target_width", "target_height", "requested_task_quotas", "budget"
    ) if key in intent}
    new_intent = DatasetIntent(
        intent_id=intent["dataset_id"], mode="hybrid", task="image_edit",
        output_schema="raw_image_pairs.train_pair.v1", description=intent["training_goal"],
        sources=(source,), parameters=parameters,
    )
    new_policy = QualityPolicy(
        policy_id=f"{intent['dataset_id']}.quality", required_gates=tuple(sorted(PROTECTED_GATES)),
        allowed_operations=("model.invoke",), task_criteria={
            key: value for key, value in policy.items()
            if key not in {"schema_version", "promotion_min_absolute_samples", "promotion_min_rate_delta"}
        },
    )
    return new_intent, new_policy
