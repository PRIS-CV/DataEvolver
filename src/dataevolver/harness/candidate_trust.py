"""Explicit host-owned trust configuration for synthetic contract qualification.

These artifacts are separate from candidate payloads and generated source. They
grant no production evidence, image submission, model access or activation.
"""
import hashlib
import json
from dataclasses import dataclass, replace
from typing import ClassVar

from dataevolver.dataset.contracts import ArtifactRef, Contract, identifier


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class CapabilityRequestBinding(Contract):
    schema: ClassVar[str] = "dataevolver.capability_request_binding.v1"
    request: dict
    campaign_id: str
    frozen_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "request", ArtifactRef.from_dict(self.request).to_dict())
        identifier(self.campaign_id)
        ArtifactRef("check", "hash", self.frozen_sha256)

    def verification_view(self, original_plan):
        """Use the existing historical request reader, never rebind a package.

        This temporary view is only for request-origin verification. It is not
        a replacement development plan and is never used to inspect a package.
        """
        protected = tuple(dict.fromkeys((*original_plan.protected_stores, self.request["store"])))
        return replace(original_plan, capability_request=self.request, request_campaign_id=self.campaign_id,
                       request_frozen_sha256=self.frozen_sha256, protected_stores=protected)


@dataclass(frozen=True)
class CandidateTrustContext(Contract):
    schema: ClassVar[str] = "dataevolver.candidate_trust_context.v1"
    context_id: str
    package_sha256: str
    implementation_sha256: str
    qualification_request_sha256: str
    input_artifact: dict
    payload_sha256: str
    fixture_family_id: str
    registrars: dict
    scope: str = "synthetic_authority_contract"
    namespace: str = "TRUSTED_REGISTRARS"

    def __post_init__(self):
        identifier(self.context_id)
        identifier(self.fixture_family_id)
        for name in ("package_sha256", "implementation_sha256", "qualification_request_sha256", "payload_sha256"):
            ArtifactRef("check", "hash", getattr(self, name))
        object.__setattr__(self, "input_artifact", ArtifactRef.from_dict(self.input_artifact).to_dict())
        if self.scope != "synthetic_authority_contract" or self.namespace != "TRUSTED_REGISTRARS":
            raise ValueError("only the explicit synthetic registrar interface is supported")
        if not isinstance(self.registrars, dict) or not 1 <= len(self.registrars) <= 8:
            raise ValueError("a bounded independently controlled registrar snapshot is required")
        clean = {}
        for key_id, key in self.registrars.items():
            identifier(key_id)
            if not isinstance(key, dict) or set(key) != {"n_hex", "e", "manifest_sha256"}:
                raise ValueError("registrar key fields changed")
            text = key["n_hex"]
            if (not isinstance(text, str) or not 512 <= len(text) <= 1024
                    or any(c not in "0123456789abcdef" for c in text)
                    or not 2048 <= int(text, 16).bit_length() <= 4096
                    or type(key["e"]) is not int or key["e"] != 65537):
                raise ValueError("only bounded RSA public keys with exponent 65537 are supported")
            ArtifactRef("check", "hash", key["manifest_sha256"])
            clean[key_id] = dict(key)
        object.__setattr__(self, "registrars", clean)

    def namespace_value(self):
        # Only public material crosses the process boundary. Private signing
        # keys stay in the independent fixture producer, outside all payloads.
        return {key_id: {"n": int(key["n_hex"], 16), "e": key["e"],
                         "manifest_sha256": key["manifest_sha256"]}
                for key_id, key in self.registrars.items()}

    def execution_value(self):
        return {"context_sha256": self.compute_hash(), "scope": self.scope,
                "namespace": self.namespace, "value": self.namespace_value()}


@dataclass(frozen=True)
class ProductionCandidateTrustContext(Contract):
    """Host-owned provenance for a confined production-data diagnostic.

    The public registrar ABI is shared with the proposed implementation. Its
    trust scope and required origin bindings are distinct from synthetic tests.
    Authentication establishes origin, never image permission or qualification.
    """
    schema: ClassVar[str] = "dataevolver.production_candidate_trust_context.v1"
    context_id: str
    package_sha256: str
    implementation_sha256: str
    qualification_request_sha256: str
    input_artifact: dict
    payload_sha256: str
    fixture_family_id: str
    registrars: dict
    origin_bindings: dict
    evidence_artifacts: tuple
    scope: str = "registered_production_runtime"
    namespace: str = "TRUSTED_REGISTRARS"

    def __post_init__(self):
        # Reuse key validation only; do not convert a synthetic context into a
        # production one or change its accepted scope.
        key_check = CandidateTrustContext(self.context_id,self.package_sha256,
            self.implementation_sha256,self.qualification_request_sha256,
            self.input_artifact,self.payload_sha256,self.fixture_family_id,self.registrars)
        if self.scope != "registered_production_runtime" or self.namespace != "TRUSTED_REGISTRARS":
            raise ValueError("production context namespace changed")
        if (not isinstance(self.origin_bindings,dict) or set(self.origin_bindings) != {
                "trial_authority_sha256","observation_authority_sha256","request_authority_sha256",
                "trial_sha256","observation_sha256","compiler_sha256","manifest_sha256"}):
            raise ValueError("production context needs all registered origin bindings")
        for value in self.origin_bindings.values():ArtifactRef("check","hash",value)
        refs=tuple(ArtifactRef.from_dict(x).to_dict() for x in self.evidence_artifacts)
        if not 3 <= len(refs) <= 64 or len({canonical_digest(x) for x in refs}) != len(refs):
            raise ValueError("production context needs a finite distinct evidence closure")
        object.__setattr__(self,"evidence_artifacts",refs)
        object.__setattr__(self,"input_artifact",key_check.input_artifact)
        object.__setattr__(self,"registrars",key_check.registrars)

    namespace_value = CandidateTrustContext.namespace_value
    execution_value = CandidateTrustContext.execution_value
