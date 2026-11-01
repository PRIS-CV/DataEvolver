"""Thin step SDK over OperatorRegistry and the existing artifact/receipt ABI."""
from __future__ import annotations

import inspect
import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator

from dataevolver.dataset.contracts import ArtifactRef, canonical_json, file_sha256
from .models import OperatorSpec
from .operator_contracts import OperatorContract


@dataclass(frozen=True)
class StepDefinition:
    spec: OperatorSpec
    contract: OperatorContract
    kind: str
    implementation: Callable

    def __post_init__(self):
        self.contract.bound_to(self.spec)
        if self.kind not in {"data", "model"} or not callable(self.implementation):
            raise ValueError("step must bind a callable data/model implementation")
        if (self.kind == "model") != (self.contract.effect_semantics == "model"):
            raise ValueError("step kind differs from effect declaration")

    def validate_node(self, node):
        if (node.operator, node.operator_version) != (self.spec.operator_id, self.spec.version):
            raise ValueError("node version differs from registered implementation")
        if node.kind != self.kind or set(node.inputs) != set(self.contract.input_schemas):
            raise ValueError("node kind or input ports differ from declaration")
        from jsonschema.exceptions import ValidationError
        try:
            self.contract.validate_parameters(node.parameters)
        except ValidationError as exc:
            raise ValueError('node parameters violate the operator contract') from exc

    def validate_ports(self, values, store, *, output=False):
        schemas = self.contract.output_schemas if output else self.contract.input_schemas
        if set(values) != set(schemas):
            raise ValueError("artifact ports differ from declaration")
        for name, ref in values.items():
            if not isinstance(ref, ArtifactRef):
                raise ValueError("ports require ArtifactRef values")
            path = store.resolve(ref)
            if not path.stat().st_size:
                raise ValueError("artifact is empty")
            if ref.media_type == "application/json":
                Draft202012Validator(schemas[name]).validate(json.loads(path.read_text()))


def declare_step(operator_id, implementation, *, inputs, outputs, parameters,
                 version="1", kind="data", category="basic", modalities=("json",)):
    source = Path(inspect.getsourcefile(implementation)).resolve()
    spec = OperatorSpec(operator_id, version, retry_safe=True,
                        source_path=str(source), code_revision=file_sha256(source))
    contract = OperatorContract(operator_id, version, spec.sha256(), category,
        dict(inputs), dict(outputs), parameters, modalities,
        "model" if kind == "model" else "pure_data", "verify_then_reuse",
        ("Declaration does not grant production qualification.",))
    return StepDefinition(spec, contract, kind, implementation)


def builtin_step_registry(*, family_step=None, selection_step=None, repair_step=None):
    """One declaration source for legacy image steps and catalog consumers."""
    from dataevolver.dataset import curation, curation_quality
    from .operators import OperatorRegistry
    registry = OperatorRegistry()
    steps = {**curation.STEPS, **curation_quality.STEPS}
    for name, (kind, inputs, parameters, outputs, implementation) in steps.items():
        if name == 'images.families' and family_step is not None:
            registry.register_step(family_step)
            continue
        if name == 'images.select_balanced' and selection_step is not None:
            registry.register_step(selection_step)
            continue
        registry.register_step(declare_step(name, implementation,
            inputs={p: {} for p in sorted(inputs)}, outputs={p: {} for p in sorted(outputs)},
            parameters={"type": "object", "properties": {p: {} for p in sorted(parameters)},
                        "required": sorted(parameters), "additionalProperties": False},
            kind=kind, modalities=("json", "image")))
    from dataevolver.dataset.catalog_steps import register
    register(registry)
    from dataevolver.dataset.visual_steps import register as register_visual
    register_visual(registry, repair_step=repair_step)
    from dataevolver.dataset.t2i_steps import register as register_t2i
    register_t2i(registry)
    from dataevolver.dataset.family_ledger import register as register_families
    register_families(registry)
    return registry


def check_declaration(spec, contract):
    OperatorContract.from_dict(contract).bound_to(OperatorSpec.from_dict(spec))
    return {"valid": True, "execution_performed": False, "qualification_performed": False}


def attach_development_context(interface, *, public_step, protocol, store, input_port,
                               max_example_bytes=16384):
    """Freeze public contract provenance and development-only JSON in an interface.

    CandidateBuildOperator already forwards the complete OperatorInterface.
    Independent, regression and heldout artifacts are never resolved here.
    This packaging grants no execution, qualification or additional model call.
    """
    from .operator_development import OperatorInterface, QualificationProtocol
    if not isinstance(interface, OperatorInterface) or not isinstance(public_step, StepDefinition):
        raise TypeError('development context needs a declared interface and registered public step')
    public_step.contract.bound_to(public_step.spec)
    if (interface.declaration['operator_id'] != public_step.spec.operator_id or
            interface.declaration['version'] == public_step.spec.version):
        raise ValueError('development context must bind a new version of the public operator')
    if type(max_example_bytes) is not int or not 0 < max_example_bytes <= 65536:
        raise ValueError('development example byte limit is invalid')
    protocol = QualificationProtocol.from_dict(protocol.to_dict())
    examples, size = [], 0
    for source in protocol.sources:
        if source['split'] != 'development':
            continue
        ref = ArtifactRef.from_dict(source['artifact'])
        if ref.media_type != 'application/json':
            raise ValueError('development examples must be explicit JSON artifacts')
        path = store.resolve(ref)
        size += path.stat().st_size
        if size > max_example_bytes:
            raise ValueError('development examples exceed the frozen byte limit')
        examples.append({'split': 'development', 'family_id': source['family_id'],
                         'artifact': ref.to_dict(), 'input': json.loads(path.read_text())})
    if not examples:
        raise ValueError('no declared development example is available')
    declaration = copy.deepcopy(interface.declaration)
    schema = declaration['input_schemas'][input_port]
    field = 'x-dataevolver-development-context'
    if field in schema:
        raise ValueError('an existing frozen development context cannot be overwritten')
    schema[field] = {'public_operator_spec_sha256': public_step.spec.sha256(),
        'public_operator_contract_sha256': public_step.contract.compute_hash(),
        'public_operator_contract': public_step.contract.to_dict(),
        'qualification_protocol_sha256': protocol.compute_hash(), 'development_examples': examples}
    canonical_json(schema[field])
    return OperatorInterface(declaration, interface.retry_safe, interface.timeout_seconds)


def scaffold_operator(directory, *, operator_id, effect="pure_data"):
    """Developer scaffold; never installs, registers or qualifies generated files."""
    from dataevolver.dataset.contracts import identifier
    identifier(operator_id)
    if effect not in {"pure_data", "model", "external", "composite"}:
        raise ValueError("unsupported operator template")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    (root / "operator.py").write_text(
        '"""Implement execute/recover/verify_effect using the declared effect contract."""\n'
        'from dataevolver.harness.operators import Operator\n\n'
        'class Implementation:\n'
        '    def __init__(self, spec): self.spec = spec\n'
        '    def execute(self, call): raise NotImplementedError\n'
        '    def recover(self, call): raise NotImplementedError\n'
        '    def verify_effect(self, call, receipt): raise NotImplementedError\n')
    (root / "declaration.json").write_text(json.dumps({"operator_id": operator_id,
        "version": "1", "effect_semantics": effect, "status": "NEEDS_IMPLEMENTATION"}, indent=2)+"\n")
    return root
