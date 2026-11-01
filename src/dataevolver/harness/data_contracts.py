"""Versioned mixed data/model node contracts; P1 contracts remain readable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from dataevolver.dataset.contracts import ArtifactRef, Contract, canonical_json, identifier, positive_int


@dataclass(frozen=True)
class DataInput(Contract):
    schema: ClassVar[str] = "dataevolver.node_input.v2"
    artifact: ArtifactRef | None = None
    node: str = ""
    output: str = ""

    def __post_init__(self):
        if self.artifact is not None:
            if not isinstance(self.artifact, ArtifactRef) or self.node or self.output:
                raise ValueError("input must bind one artifact or one upstream output")
        else:
            identifier(self.node, "upstream node")
            identifier(self.output, "upstream output")

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        if values.get("artifact") is not None:
            values["artifact"] = ArtifactRef.from_dict(values["artifact"])
        return cls(**values)

    def to_dict(self):
        return {"schema_version": self.schema, "artifact": self.artifact.to_dict() if self.artifact else None,
                "node": self.node, "output": self.output}


@dataclass(frozen=True)
class DataRunBudget(Contract):
    schema: ClassVar[str] = "dataevolver.data_budget.v2"
    max_operator_calls: int
    max_model_calls: int
    max_model_images: int
    max_input_rows: int
    max_output_bytes: int
    max_wall_seconds: int

    def __post_init__(self):
        for name in ("max_operator_calls", "max_input_rows", "max_output_bytes", "max_wall_seconds"):
            positive_int(getattr(self, name), name)
        for name in ("max_model_calls", "max_model_images"):
            positive_int(getattr(self, name), name, zero=True)


@dataclass(frozen=True)
class DataNodeSpec(Contract):
    schema: ClassVar[str] = "dataevolver.node_spec.v2"
    node_id: str
    operator: str
    operator_version: str
    kind: str
    inputs: dict[str, DataInput]
    parameters: dict[str, Any] = field(default_factory=dict)
    role: str = ""
    max_model_calls: int = 0
    max_model_images: int = 0

    def __post_init__(self):
        for name in ("node_id", "operator", "operator_version"):
            identifier(getattr(self, name), name)
        if self.kind not in {"data", "model"} or not isinstance(self.inputs, dict):
            raise ValueError("node kind and named inputs are required")
        for name, value in self.inputs.items():
            identifier(name, "input name")
            if not isinstance(value, DataInput) or value.node == self.node_id:
                raise ValueError("invalid or self-dependent node input")
        if not isinstance(self.parameters, dict):
            raise ValueError("node parameters must be an object")
        canonical_json(self.parameters)
        positive_int(self.max_model_calls, "max_model_calls", zero=True)
        positive_int(self.max_model_images, "max_model_images", zero=True)
        if self.kind == "data" and (self.role or self.max_model_calls or self.max_model_images):
            raise ValueError("pure data node cannot acquire model authority or model budget")
        if self.kind == "model":
            identifier(self.role, "model role")
            positive_int(self.max_model_calls, "model node calls")
            positive_int(self.max_model_images, "model node images", zero=self.schema == "dataevolver.node_spec.v3")

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["inputs"] = {k: DataInput.from_dict(v) for k, v in values["inputs"].items()}
        return cls(**values)

    def to_dict(self):
        result = super().to_dict()
        result["inputs"] = {k: v.to_dict() for k, v in self.inputs.items()}
        return result


@dataclass(frozen=True)
class DataNodeGraph(Contract):
    schema: ClassVar[str] = "dataevolver.node_graph.v2"
    nodes: tuple[DataNodeSpec, ...]

    def __post_init__(self):
        if not self.nodes or any(not isinstance(n, DataNodeSpec) for n in self.nodes):
            raise ValueError("graph requires typed nodes")
        names = [n.node_id for n in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("duplicate node ID")
        if any(v.node and v.node not in names for n in self.nodes for v in n.inputs.values()):
            raise ValueError("upstream node is missing")
        self.ordered()

    def ordered(self):
        ordered, pending = [], list(self.nodes)
        while pending:
            ready = [n for n in pending if {v.node for v in n.inputs.values() if v.node}.issubset(
                {x.node_id for x in ordered})]
            if not ready:
                raise ValueError("node graph contains a cycle")
            for node in ready:
                ordered.append(node)
                pending.remove(node)
        return tuple(ordered)

    def validate_policy(self, policy, budget, roles):
        if len(self.nodes) > budget.max_operator_calls:
            raise ValueError("node graph exceeds Operator budget")
        calls = sum(n.max_model_calls for n in self.nodes)
        images = sum(n.max_model_images for n in self.nodes)
        if calls > min(policy.max_model_calls, budget.max_model_calls) or images > budget.max_model_images:
            raise ValueError("node graph exceeds model budget")
        for node in self.nodes:
            if node.operator not in policy.allowed_operations:
                raise ValueError("node Operator is not approved")
            if node.kind == "model" and node.role not in roles:
                raise ValueError("model role is not approved")

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["nodes"] = tuple(DataNodeSpec.from_dict(n) for n in values["nodes"])
        return cls(**values)

    def to_dict(self):
        return {"schema_version": self.schema, "nodes": [n.to_dict() for n in self.nodes]}


@dataclass(frozen=True)
class DataNodeSpecV3(DataNodeSpec):
    """Explicit text-capable version; v2 model nodes still require images."""
    schema: ClassVar[str] = "dataevolver.node_spec.v3"


@dataclass(frozen=True)
class DataNodeGraphV3(DataNodeGraph):
    schema: ClassVar[str] = "dataevolver.node_graph.v3"

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["nodes"] = tuple(DataNodeSpecV3.from_dict(n) for n in values["nodes"])
        return cls(**values)


def read_data_graph(payload):
    cls = DataNodeGraphV3 if payload.get("schema_version") == DataNodeGraphV3.schema else DataNodeGraph
    return cls.from_dict(payload)


@dataclass(frozen=True)
class DataNodeRun(Contract):
    schema: ClassVar[str] = "dataevolver.node_run.v2"
    campaign_id: str
    call_id: str
    node_id: str
    node_spec_hash: str
    operator_revision: str
    inputs: dict[str, ArtifactRef]
    outputs: dict[str, ArtifactRef]
    operator_receipt_hash: str
    model_receipt_hashes: tuple[str, ...] = ()
    status: str = "VERIFIED"

    def __post_init__(self):
        for name in ("campaign_id", "call_id", "node_id"):
            identifier(getattr(self, name), name)
        if self.status != "VERIFIED" or not self.outputs:
            raise ValueError("NodeRun requires verified outputs")
        for values in (self.inputs, self.outputs):
            for name, ref in values.items():
                identifier(name, "artifact port")
                if not isinstance(ref, ArtifactRef):
                    raise ValueError("NodeRun ports require bound artifacts")
        for digest in (self.node_spec_hash, self.operator_revision, self.operator_receipt_hash,
                       *self.model_receipt_hashes):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("NodeRun requires SHA-256 bindings")

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        for name in ("inputs", "outputs"):
            values[name] = {k: ArtifactRef.from_dict(v) for k, v in values[name].items()}
        values["model_receipt_hashes"] = tuple(values.get("model_receipt_hashes", ()))
        return cls(**values)

    def to_dict(self):
        result = super().to_dict()
        for name in ("inputs", "outputs"):
            result[name] = {k: v.to_dict() for k, v in getattr(self, name).items()}
        return result
