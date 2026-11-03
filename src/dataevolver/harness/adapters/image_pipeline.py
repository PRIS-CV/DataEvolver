"""Collection v1 and layout curation v3 reuse the existing data graph controller."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

from dataevolver.dataset import collection, curation_layout
from dataevolver.dataset.collection_spec import CollectionSpec, load_source_bundle, resolve_hf_endpoint
from dataevolver.dataset.contracts import ArtifactRef, canonical_json, payload_sha256
from dataevolver.dataset.layout_quality import compile_criteria
from ..models import ControllerState, EvidenceStatus
from ..operator_sdk import declare_step
from ..operators import OperatorRegistry
from .data_graph import DataGraphOperator


def step_registry(steps):
    registry = OperatorRegistry()
    for name, (kind, inputs, parameters, outputs, implementation) in steps.items():
        registry.register_step(declare_step(name, implementation,
            inputs={name: {} for name in sorted(inputs)}, outputs={name: {} for name in sorted(outputs)},
            parameters={"type": "object", "properties": {name: {} for name in sorted(parameters)},
                        "required": sorted(parameters), "additionalProperties": False},
            kind=kind, modalities=("json", "image")))
    return registry


class ImageCollectionOperator(DataGraphOperator):
    operator_id = "data.image_collection"
    version = "1"
    steps = collection.STEPS
    validate_step = staticmethod(collection.validate_step)
    allowed_model_backends = frozenset()

    def __init__(self, **kwargs):
        kwargs.setdefault("registry", step_registry(self.steps))
        super().__init__(**kwargs)
        for node in self.graph.nodes:
            self.validate_step(node)
        if len(self.graph.nodes) != 1 or self.profiles:
            raise ValueError("collection requires one pure-data node")
        self.mode = self.intent.parameters.get("execution_mode")
        spec_ref = self.graph.nodes[0].inputs["spec"].artifact
        if spec_ref is None:
            raise ValueError("collection requires an immutable CollectionSpec")
        self.collection_spec = CollectionSpec.from_dict(json.loads(self.store.resolve(spec_ref).read_text()))
        if self.mode != self.collection_spec.mode:
            raise ValueError("collection execution mode differs from source spec")
        self.collection_spec = replace(self.collection_spec, hf_endpoint=resolve_hf_endpoint(self.collection_spec.hf_endpoint))
        self.contract_hash = payload_sha256([self.contract_hash, self.collection_spec.to_dict()])
        self.spec = self.spec.with_updates(model_config_hash=self.contract_hash)

    def _validate(self, call):
        super()._validate(call)
        if self.mode == "server" and sys.platform != "linux":
            raise ValueError("real image collection must run on the Linux server")

    def evidence_level(self):
        return "mock_verified" if self.mode == "fixture" else "real_execution_verified"

    def verified_capability_level(self, state):
        if (state.state is not ControllerState.COMPLETE
                or state.data.get("last_effect_verification", {}).get("status") != EvidenceStatus.VERIFIED.value):
            return "designed"
        return self.evidence_level()


class ImageLayoutOperator(DataGraphOperator):
    version = "3"
    steps = curation_layout.STEPS
    validate_step = staticmethod(curation_layout.validate_step)
    allowed_model_backends = frozenset({"mock", "local_endpoint"})

    def __init__(self, **kwargs):
        kwargs.setdefault("registry", step_registry(self.steps))
        super().__init__(**kwargs)
        for node in self.graph.nodes:
            self.validate_step(node)
        self.criteria = compile_criteria(self.policy.task_criteria)
        if "deletion" not in self.policy.required_gates:
            raise ValueError("layout curation requires the protected deletion-policy gate")
        self.mode = self.criteria["execution_mode"]
        if self.intent.parameters.get("execution_mode") != self.mode or self.intent.output_schema != "curated_images.v2":
            raise ValueError("layout execution mode/output schema differ from intent")
        expected = ["layout.audit", "layout.families", "layout.review", "layout.repair", "layout.rereview", "layout.export"]
        if [node.operator for node in self.graph.ordered()] != expected:
            raise ValueError("v3 requires the registered audit/review/one-repair/rereview/export graph")
        roles = self.model_policy.bindings if self.model_policy else {}
        if set(roles) != {"annotator", "reviewer"}:
            raise ValueError("v3 requires explicit annotator and independent reviewer roles")
        expected_backend = "mock" if self.mode == "fixture" else "local_endpoint"
        if any(profile.backend != expected_backend for profile in self.profiles):
            raise ValueError("model backend differs from fixture/server execution mode")
        approved = {canonical_json(value.to_dict()) for value in self.intent.sources}
        for name in ("calibration", "protected_evidence", "deployments"):
            if self.criteria.get(name) and canonical_json(self.criteria[name]) not in approved:
                raise ValueError("quality evidence is outside DatasetIntent source scope")
        if self.mode == "server":
            self._deployments()

    def _deployments(self):
        if not self.criteria.get("deployments"):
            raise ValueError("real layout processing requires server model deployment evidence")
        deployments = json.loads(self.store.resolve(ArtifactRef.from_dict(self.criteria["deployments"])).read_text())
        if deployments.get("schema_version") != "dataevolver.model_deployments.v1" or deployments.get("host_role") != "server":
            raise ValueError("invalid server deployment record")
        from dataevolver.adapters.model_contracts import origin
        for profile in self.profiles:
            record = deployments.get("profiles", {}).get(profile.profile_id, {})
            endpoint = os.environ.get(profile.endpoint_env, "")
            if (record.get("model") != profile.model or record.get("revision") != profile.revision
                    or profile.revision in {"main", "latest"} or not record.get("deployment_id")
                    or not record.get("model_family") or origin(endpoint) != record.get("endpoint_origin")
                    or origin(endpoint) not in self.model_policy.allowed_local_origins):
                raise ValueError("served model revision/deployment/origin evidence is missing or mismatched")

    def _validate(self, call):
        super()._validate(call)
        if self.mode == "server" and sys.platform != "linux":
            raise ValueError("real layout processing must run on the Linux server")

    def _inspect(self, call):
        verification, artifacts = super()._inspect(call)
        if verification.status is EvidenceStatus.VERIFIED:
            try:
                first = self.graph.ordered()[0]
                load_source_bundle(self.store.resolve(first.inputs["bundle"].artifact))
            except (ValueError, OSError, KeyError) as exc:
                from ..models import EffectVerification
                return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc)), ()
        return verification, artifacts

    def evidence_level(self):
        return "mock_verified" if self.mode == "fixture" else "real_execution_verified"

    verified_capability_level = ImageCollectionOperator.verified_capability_level
