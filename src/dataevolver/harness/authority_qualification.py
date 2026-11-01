"""Host-side oracle for independently specified synthetic authority fixtures.

No candidate import or geometric inference occurs here. Explicit expected pixels
come from a frozen fixture declaration. Signature verification uses the separate
cryptography implementation, rather than the generated source's RSA code.
"""
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dataevolver.dataset.contracts import ArtifactRef, Contract, file_sha256, identifier
from .candidate_trust import canonical_bytes, canonical_digest


def crypto_identity():
    import cryptography
    from cryptography.hazmat.bindings import _rust
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes
    return {"version": cryptography.__version__, "module_sha256": file_sha256(Path(cryptography.__file__)),
            "native_sha256": file_sha256(Path(_rust.__file__)),
            "rsa_sha256": file_sha256(Path(rsa.__file__)), "padding_sha256": file_sha256(Path(padding.__file__)),
            "hashes_sha256": file_sha256(Path(hashes.__file__))}


def authenticated(payload, context):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    try:
        evidence = payload["evidence"]
        source = evidence["source"]
        if set(source) != {"registrar", "signature", "manifest"}: return False
        key = context.registrars[source["registrar"]]
        if canonical_digest(source["manifest"]) != key["manifest_sha256"]: return False
        unsigned = dict(evidence)
        unsigned["source"] = {k: v for k, v in source.items() if k != "signature"}
        public = rsa.RSAPublicNumbers(key["e"], int(key["n_hex"], 16)).public_key()
        public.verify(base64.b64decode(source["signature"], validate=True), canonical_bytes(unsigned),
                      padding.PKCS1v15(), hashes.SHA256())
        return True
    except (InvalidSignature, ValueError, KeyError, TypeError):
        return False


@dataclass(frozen=True)
class AuthorityFixtureExpectation(Contract):
    schema: ClassVar[str] = "dataevolver.authority_fixture_expectation.v1"
    fixture_family_id: str
    payload_sha256: str
    authenticated: bool
    status: str
    resolved_pixels: list
    receipt_required: bool
    added_pixels: list
    removed_pixels: list
    unresolved_reasons: list
    conflict_reasons: list
    operator_id: str = "data.boundary_effect_authority"
    operator_version: str = "1"

    def __post_init__(self):
        identifier(self.fixture_family_id)
        identifier(self.operator_id)
        if not isinstance(self.operator_version,str) or not self.operator_version:
            raise ValueError("an explicit operator receipt version is required")
        ArtifactRef("check", "hash", self.payload_sha256)
        if (type(self.authenticated) is not bool or type(self.receipt_required) is not bool
                or self.status not in {"executable", "unsupported"}):
            raise ValueError("fixture expectation has invalid status/authentication")
        for name in ("resolved_pixels", "added_pixels", "removed_pixels"):
            points = getattr(self, name)
            if (not isinstance(points, list) or len(points) > 4096
                    or any(not isinstance(p, list) or len(p) != 2
                           or any(type(v) is not int or not 0 <= v <= 4096 for v in p) for p in points)
                    or points != sorted(points, key=lambda p: (p[1], p[0]))
                    or len({tuple(p) for p in points}) != len(points)):
                raise ValueError("expected pixels must be bounded, explicit, ordered and distinct")
        for name in ("unresolved_reasons", "conflict_reasons"):
            values = getattr(self, name)
            if (not isinstance(values, list) or len(values) > 32
                    or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values)):
                raise ValueError("expected reasons must be explicit and distinct")
        if self.status == "executable" and (not self.authenticated or not self.receipt_required
                                             or self.unresolved_reasons or self.conflict_reasons):
            raise ValueError("a positive fixture needs authenticated complete coverage")


def validate_fixture(payload, context, expected):
    if (expected.payload_sha256 != canonical_digest(payload)
            or expected.fixture_family_id != context.fixture_family_id
            or authenticated(payload, context) is not expected.authenticated):
        raise ValueError("independently declared fixture authentication or identity is invalid")


def check_authority_output(payload, output, context, expected):
    """Malformed candidate outputs fail checks; invalid fixtures stop execution."""
    validate_fixture(payload, context, expected)
    try:
        return _compare_output(payload, output, context, expected)
    except (AttributeError, ValueError, KeyError, TypeError, IndexError):
        return {"output_structure": False}


def _compare_output(payload, output, context, expected):
    checks = {"status": output.get("status") == expected.status,
              "submission_and_adoption_gates": output.get("image_submission_allowed") is False
                    and output.get("candidate_adoption_allowed") is False,
              "unresolved_reasons": sorted({x.get("reason") for x in output.get("unresolved", [])}) == sorted(expected.unresolved_reasons),
              "conflict_reasons": sorted({x.get("reason") for x in output.get("conflicts", [])}) == sorted(expected.conflict_reasons)}
    support = output.get("authorized_support")
    if expected.status == "unsupported":
        checks["support"] = support == {}
    else:
        crop = payload["evidence"]["source"]["manifest"]["transform"]["crop"]
        cropped = [[x-crop[0], y-crop[1]] for x,y in expected.resolved_pixels]
        checks["support"] = (isinstance(support, dict) and support.get("source_pixels") == expected.resolved_pixels
            and support.get("conditioning") == cropped and support.get("composition") == cropped
            and support.get("inverse_projection") == expected.resolved_pixels and support.get("crop") == crop)
    receipt = output.get("authority_receipt")
    if not expected.receipt_required:
        checks["receipt"] = receipt is None
        return checks
    if not isinstance(receipt, dict):
        checks["receipt"] = False
        return checks
    evidence = payload["evidence"]
    manifest = evidence["source"]["manifest"]
    unsigned = dict(evidence)
    unsigned["source"] = {k:v for k,v in evidence["source"].items() if k != "signature"}
    provenance = [{"observation": x["id"], "kind": x["kind"], "subject": x["subject"],
                   "pixels": sorted(x["pixels"], key=lambda p:(p[1],p[0])), "basis": x["basis"]}
                  for x in evidence["observations"]]
    checks.update({
        "receipt_identity": receipt.get("operator")==expected.operator_id and receipt.get("version")==expected.operator_version,
        "receipt_integrity": output.get("authority_receipt_sha256") == canonical_digest(receipt)
            and (expected.status != "executable" or support.get("receipt_sha256") == canonical_digest(receipt)),
        "manifest_and_frozen_requirements": output.get("frozen_manifest") == manifest
            and receipt.get("manifest_sha256") == canonical_digest(manifest)
            and receipt.get("frozen_sha256") == canonical_digest(manifest["frozen"]),
        "evidence_and_transform": receipt.get("evidence_sha256") == canonical_digest(unsigned)
            and receipt.get("transform_sha256") == canonical_digest(manifest["transform"])
            and receipt.get("registrar") == evidence["source"]["registrar"],
        "support_digest": receipt.get("support_sha256") == canonical_digest(expected.resolved_pixels),
        "prior_support_changes": receipt.get("changes", {}).get("added") == expected.added_pixels
            and receipt.get("changes", {}).get("removed") == expected.removed_pixels,
        "witness_provenance": receipt.get("observations") == provenance and output.get("provenance") == provenance,
        "pending_image_validation": output.get("pending_validation") == {
            "model_checks": manifest["frozen"]["required_model_checks"],
            "independent_runtime_visual_audit": True, "isolated_engineering_validation": True},
    })
    return checks
