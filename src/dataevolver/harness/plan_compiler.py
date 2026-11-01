"""Compile declared plans to the existing graph Operator, without executing it."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import ClassVar

from dataevolver.dataset.contracts import ArtifactRef, Contract, payload_sha256
from .data_contracts import DataInput, DataNodeGraphV3, DataNodeSpecV3, read_data_graph
from .domains import DomainBinding


@dataclass(frozen=True)
class CompilationGap:
    gaps: tuple[dict, ...]

    def to_dict(self):
        return {"compiled": False, "gaps": list(self.gaps), "execution_performed": False}


@dataclass(frozen=True)
class CompileContext:
    registry: object
    domains: dict[str, DomainBinding]
    allowed_operators: tuple[str, ...]
    execution_contract: dict = field(default_factory=dict)
    source_bindings: dict = field(default_factory=dict)
    roots: dict = field(default_factory=dict)
    policies: tuple[dict, ...] = ()


class GraphPlanCompiler:
    """Accept a declared graph or compile training-goal node bindings to it."""
    def compile(self, plan, context):
        if "contract" in plan:
            contract = copy.deepcopy(plan["contract"])
            graph = read_data_graph(contract['graph'])
            contract['graph'] = DataNodeGraphV3(tuple(DataNodeSpecV3(
                n.node_id, n.operator, n.operator_version, n.kind, n.inputs, n.parameters,
                n.role, n.max_model_calls, n.max_model_images) for n in graph.nodes)).to_dict()
            return contract
        contract = copy.deepcopy(context.execution_contract)
        rows = plan.get("dataset_plan", plan).get("nodes", [])
        if not rows:
            raise ValueError("plan has no executable nodes")
        nodes = []
        for row in rows:
            definition = context.registry.step(row["operator"], row.get("operator_version", row.get("version", "1")))
            inputs = {}
            if "input_bindings" in row:
                for binding in row["input_bindings"]:
                    source = binding["source"]
                    if source.startswith("node:"):
                        node, _, output = source[5:].rpartition(".")
                        inputs[binding["port"]] = DataInput(node=node, output=output)
                    else:
                        inputs[binding["port"]] = DataInput(artifact=ArtifactRef.from_dict(context.source_bindings[source]))
            else:
                inputs = {k: DataInput.from_dict(v) for k, v in row.get("inputs", {}).items()}
            parameters = row.get("parameters", json.loads(row.get("parameters_json", "{}")))
            nodes.append(DataNodeSpecV3(row.get("node_id", row.get("id")), definition.spec.operator_id,
                definition.spec.version, definition.kind, inputs, parameters,
                row.get("role", ""), row.get("max_model_calls", 0), row.get("max_model_images", 0)))
        contract["graph"] = DataNodeGraphV3(tuple(nodes)).to_dict()
        return contract


def builtin_domains():
    return {name: DomainBinding(GraphPlanCompiler()) for name in (
        "image_curation", "catalog", "visual_generation", "text_to_image_generation")}


@dataclass(frozen=True)
class CompiledRun(Contract):
    schema: ClassVar[str] = "dataevolver.compiled_run.v1"
    domain: str
    contract: dict
    operator_refs: tuple[dict, ...]
    node_calls: dict
    policies: tuple[dict, ...] = ()

    def __post_init__(self):
        graph = read_data_graph(self.contract["graph"])
        if set(self.node_calls) != {n.node_id for n in graph.nodes}:
            raise ValueError("compiled node/call map differs from graph")
        if not self.operator_refs or len({r['operator_id'] for r in self.operator_refs}) != len(self.operator_refs):
            raise ValueError("compiled run needs one exact binding per operator ID")

    @classmethod
    def from_dict(cls, value):
        fields = cls.read_fields(value)
        for name in ("operator_refs", "policies"):
            fields[name] = tuple(fields.get(name, ()))
        return cls(**fields)

    def bind(self, registry, *, allowed_write_root, roots=None, providers=None,source_gateways=None):
        from .adapters.data_graph import DataGraphOperator
        for ref in self.operator_refs:
            actual = registry.step(ref["operator_id"], ref["version"]).spec
            if actual.sha256() != ref["spec_sha256"]:
                raise ValueError("compiled operator implementation changed")
        return DataGraphOperator(allowed_write_root=allowed_write_root, contract=self.contract,
                                 roots=roots, registry=registry, providers=providers,source_gateways=source_gateways)


def compile_plan(plan, context: CompileContext):
    """Return all discoverable binding gaps before model loading or effects."""
    from jsonschema.exceptions import ValidationError
    domain = plan.get("domain", plan.get("goal", {}).get("task_type", ""))
    if domain not in context.domains:
        return CompilationGap(({"kind": "missing_domain", "domain": domain},))
    try:
        contract = context.domains[domain].compiler.compile(plan, context)
        graph = read_data_graph(contract["graph"])
    except (KeyError, TypeError, ValueError) as exc:
        return CompilationGap(({"kind": "invalid_plan", "reason": str(exc)},))
    gaps, refs, outputs = [], {}, {}
    for node in graph.ordered():
        try:
            if node.operator not in context.allowed_operators:
                raise ValueError("operator is outside the authorized catalog")
            definition = context.registry.step(node.operator, node.operator_version)
            definition.validate_node(node)
            outputs[node.node_id] = definition.contract.output_schemas
            for port, value in node.inputs.items():
                if value.node:
                    if value.output not in outputs[value.node]:
                        raise ValueError("upstream output port does not exist")
                    upstream, downstream = outputs[value.node][value.output], definition.contract.input_schemas[port]
                    if upstream.get("type") and downstream.get("type") and upstream['type'] != downstream['type']:
                        raise ValueError("incompatible connected port schemas")
            refs[node.operator] = {"operator_id": node.operator, "version": node.operator_version,
                                    "spec_sha256": definition.spec.sha256()}
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            gaps.append({"kind": "operator_binding", "node": node.node_id, "operator": node.operator, "reason": str(exc)})
    if gaps:
        return CompilationGap(tuple(gaps))
    try:
        from .data_contracts import DataRunBudget
        from .task_contracts import QualityPolicy
        from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
        budget = DataRunBudget.from_dict(contract["budget"])
        quality = QualityPolicy.from_dict(contract["quality_policy"])
        models = contract["models"]
        policy = ModelPolicy.from_dict(models["policy"]) if models else None
        profiles = {p.profile_id: p for p in map(ModelProfile.from_dict, models.get("profiles", []))}
        graph.validate_policy(quality, budget, set(policy.bindings) if policy else set())
        for node in graph.nodes:
            if node.kind == "model":
                profile = profiles[policy.bindings[node.role]]
                if node.max_model_images and "vision" not in profile.capabilities:
                    raise ValueError("image node lacks a vision model binding")
        for row in context.policies:
            from .feedback import PolicySpec
            PolicySpec.from_dict(row)
    except (KeyError, TypeError, ValueError) as exc:
        return CompilationGap(({"kind": "resource_binding", "reason": str(exc)},))
    result = CompiledRun(domain, contract, tuple(refs[k] for k in sorted(refs)),
        {n.node_id: {"operator_id": n.operator, "version": n.operator_version} for n in graph.nodes}, context.policies)
    if context.roots:
        try:
            from dataevolver.dataset.contracts import ArtifactStore
            store = ArtifactStore(context.roots)
            for node in graph.nodes:
                for value in node.inputs.values():
                    if value.artifact:
                        store.resolve(value.artifact)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            return CompilationGap(({"kind": "input_asset", "reason": str(exc)},))
    return result
