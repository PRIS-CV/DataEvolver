from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataevolver.workflows.multimodal.router import build_request
from dataevolver.workflows.multimodal.run_multimodal_dataset import run

from ..models import (
    EffectVerification,
    EvidenceStatus,
    ExpectedOutputSpec,
    OperatorCall,
    OperatorReceipt,
    OperatorSpec,
    RecoveryResult,
    RecoveryStatus,
    StageStatus,
    canonical_json,
)
from ..operators import _load_receipt, _receipt_path, _sha256_file, _write_receipt


OPERATOR_ID = "multimodal.t2i_dryrun"
OPERATOR_VERSION = "1"
ALLOWED_PARAMETERS = {
    "user_request",
    "num_samples",
    "height",
    "width",
    "seed",
    "sample_prefix",
    "skip_existing",
    "validate_outputs",
}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class MultimodalT2IDryRunOperator:
    def __init__(self, *, allowed_write_root: str | Path):
        source = Path(__file__).resolve()
        root = Path(allowed_write_root).resolve()
        self.allowed_write_root = root
        self.spec = OperatorSpec(
            operator_id=OPERATOR_ID,
            version=OPERATOR_VERSION,
            retry_safe=True,
            expected_output_specs=(
                ExpectedOutputSpec("dataset/request.json"),
                ExpectedOutputSpec("dataset/dataset_summary.json"),
                ExpectedOutputSpec("dataset/manifest.jsonl"),
            ),
            model_config_hash="multimodal-t2i-dryrun-v1",
            code_revision=_sha256_file(source),
            allowed_write_roots=(str(root),),
            source_path=str(source),
        )

    def _parameters(self, call: OperatorCall) -> dict[str, Any]:
        if (call.operator_id, call.operator_version) != (OPERATOR_ID, OPERATOR_VERSION):
            raise ValueError("operator call does not match registered T2I dry-run adapter")
        work_dir = Path(call.work_dir).resolve()
        try:
            work_dir.relative_to(self.allowed_write_root)
        except ValueError as exc:
            raise ValueError("OperatorCall work_dir is outside allowed write root") from exc
        unknown = sorted(set(call.parameters) - ALLOWED_PARAMETERS)
        if unknown:
            raise ValueError(f"unsupported parameters: {', '.join(unknown)}")
        parameters = {
            "user_request": call.parameters.get("user_request"),
            "num_samples": call.parameters.get("num_samples", 1),
            "height": call.parameters.get("height", 512),
            "width": call.parameters.get("width", 512),
            "seed": call.parameters.get("seed", 11),
            "sample_prefix": call.parameters.get("sample_prefix", "sample"),
            "skip_existing": call.parameters.get("skip_existing", True),
            "validate_outputs": call.parameters.get("validate_outputs", True),
        }
        if not isinstance(parameters["user_request"], str) or not parameters["user_request"].strip():
            raise ValueError("user_request must be a non-empty string")
        for name in ("num_samples", "height", "width", "seed"):
            if type(parameters[name]) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= parameters["num_samples"] <= 100:
            raise ValueError("num_samples must be in [1, 100]")
        if not 64 <= parameters["height"] <= 2048 or not 64 <= parameters["width"] <= 2048:
            raise ValueError("height and width must be in [64, 2048]")
        if type(parameters["skip_existing"]) is not bool or type(parameters["validate_outputs"]) is not bool:
            raise ValueError("skip_existing and validate_outputs must be booleans")
        if not isinstance(parameters["sample_prefix"], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", parameters["sample_prefix"]
        ):
            raise ValueError("sample_prefix must be path-safe")
        return parameters

    @staticmethod
    def _dataset_root(call: OperatorCall) -> Path:
        return Path(call.work_dir).resolve() / "dataset"

    def _request(self, call: OperatorCall, parameters: dict[str, Any]):
        return build_request(
            route="t2i",
            user_request=parameters["user_request"],
            output_root=str(self._dataset_root(call)),
            dry_run=True,
            allow_model_inference=False,
            num_samples=parameters["num_samples"],
            input_image_dir=None,
            prompt_file=None,
            generator="dryrun",
            model_name=None,
            model_path=None,
            device="cpu",
            height=parameters["height"],
            width=parameters["width"],
            steps=1,
            num_frames=1,
            fps=1,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
            negative_prompt="",
            seed=parameters["seed"],
            sample_prefix=parameters["sample_prefix"],
            skip_existing=parameters["skip_existing"],
            validate_outputs=parameters["validate_outputs"],
            vlm_eval=False,
            vlm_device="cpu",
            vlm_model_path=None,
            vlm_max_samples=None,
            vlm_max_retries=1,
            vlm_review_mode="studio",
            vlm_use_freeform_first=False,
            vlm_enable_thinking=False,
        )

    @staticmethod
    def _effect_hash(hashes: dict[str, str]) -> str:
        return hashlib.sha256(canonical_json(hashes).encode("utf-8")).hexdigest()

    def _inspect(self, call: OperatorCall) -> tuple[EffectVerification, tuple[dict[str, str], ...]]:
        parameters = self._parameters(call)
        root = self._dataset_root(call)
        request_path = root / "request.json"
        summary_path = root / "dataset_summary.json"
        manifest_path = root / "manifest.jsonl"
        required = (request_path, summary_path, manifest_path)
        if any(not path.exists() for path in required):
            return EffectVerification(EvidenceStatus.UNOBSERVED, "required dry-run outputs are missing"), ()
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, f"invalid dry-run metadata: {exc}"), ()
        expected_request = {
            "route": "t2i",
            "dry_run": True,
            "allow_model_inference": False,
            "generator": "dryrun",
            "device": "cpu",
            "vlm_eval": False,
            "num_samples": parameters["num_samples"],
        }
        if any(request.get(key) != value for key, value in expected_request.items()):
            return EffectVerification(EvidenceStatus.CONTRADICTED, "request contract mismatch"), ()
        try:
            if Path(request["output_root"]).resolve() != root:
                raise ValueError("request output_root mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc)), ()
        if (
            summary.get("schema") != "dataevolver_multimodal_dataset_summary_v1"
            or summary.get("route") != "t2i"
            or summary.get("dry_run") is not True
            or summary.get("generator") != "dryrun"
            or summary.get("sample_count") != parameters["num_samples"]
            or len(rows) != parameters["num_samples"]
        ):
            return EffectVerification(EvidenceStatus.CONTRADICTED, "summary or manifest count mismatch"), ()
        paths = [*required]
        for row in rows:
            if row.get("route") != "t2i" or row.get("generator") != "dryrun":
                return EffectVerification(EvidenceStatus.CONTRADICTED, "manifest route or generator mismatch"), ()
            output_value = row.get("output_path")
            if not isinstance(output_value, str) or not output_value:
                return EffectVerification(EvidenceStatus.CONTRADICTED, "manifest output_path is missing"), ()
            output = (root / output_value).resolve()
            try:
                output.relative_to(root)
            except ValueError:
                return EffectVerification(EvidenceStatus.CONTRADICTED, "manifest output_path escapes dataset root"), ()
            if not output.is_file():
                return EffectVerification(EvidenceStatus.CONTRADICTED, "manifest PNG output is invalid"), ()
            with output.open("rb") as handle:
                if handle.read(len(PNG_MAGIC)) != PNG_MAGIC:
                    return EffectVerification(EvidenceStatus.CONTRADICTED, "manifest PNG output is invalid"), ()
            paths.append(output)
        hashes = {str(path): _sha256_file(path) for path in sorted(paths, key=str)}
        artifacts = tuple({"path": path, "sha256": digest} for path, digest in hashes.items())
        return EffectVerification(
            EvidenceStatus.VERIFIED,
            "T2I dry-run artifacts verified",
            artifact_hashes=hashes,
        ), artifacts

    def _receipt_from_outputs(self, call: OperatorCall, *, reused: bool) -> OperatorReceipt:
        verification, artifacts = self._inspect(call)
        if verification.status is not EvidenceStatus.VERIFIED:
            raise RuntimeError(verification.reason)
        now = datetime.now(timezone.utc).isoformat()
        return OperatorReceipt(
            call_id=call.call_id,
            operator_id=OPERATOR_ID,
            operator_version=OPERATOR_VERSION,
            status=StageStatus.SUCCESS,
            started_at=now,
            finished_at=now,
            return_code=0,
            output_artifacts=artifacts,
            effect_hash=self._effect_hash(verification.artifact_hashes),
            reused=reused,
            stage_result={
                "operator_id": OPERATOR_ID,
                "operator_version": OPERATOR_VERSION,
                "status": StageStatus.SUCCESS.value,
                "metrics": {"completion_ratio": 1.0, "resource_policy_respected": 1.0},
                "hard_constraints": {"artifact_valid": True},
                "matched_trial_key": call.matched_trial_key.to_dict() if call.matched_trial_key else None,
            },
        )

    def execute(self, call: OperatorCall) -> OperatorReceipt:
        parameters = self._parameters(call)
        receipt_path = _receipt_path(call)
        prior = _load_receipt(receipt_path)
        if prior is not None:
            verification = self.verify_effect(call, prior)
            if verification.status is not EvidenceStatus.VERIFIED:
                raise RuntimeError(verification.reason)
            return prior.with_updates(reused=True)
        existing, _ = self._inspect(call)
        if existing.status is EvidenceStatus.VERIFIED:
            return _write_receipt(receipt_path, self._receipt_from_outputs(call, reused=True))
        if existing.status is EvidenceStatus.CONTRADICTED:
            raise RuntimeError(existing.reason)
        run(self._request(call, parameters))
        return _write_receipt(receipt_path, self._receipt_from_outputs(call, reused=False))

    def recover(self, call: OperatorCall) -> RecoveryResult:
        self._parameters(call)
        try:
            receipt = _load_receipt(_receipt_path(call))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, f"invalid receipt: {exc}")
        if receipt is not None:
            verification = self.verify_effect(call, receipt)
            if verification.status is EvidenceStatus.VERIFIED:
                return RecoveryResult(RecoveryStatus.RECOVERED, receipt, "receipt and outputs verified")
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, verification.reason)
        verification, _ = self._inspect(call)
        if verification.status is EvidenceStatus.VERIFIED:
            recovered = _write_receipt(_receipt_path(call), self._receipt_from_outputs(call, reused=True))
            return RecoveryResult(RecoveryStatus.RECOVERED, recovered, "outputs verified and receipt rebuilt")
        if verification.status is EvidenceStatus.UNOBSERVED:
            return RecoveryResult(RecoveryStatus.RETRY_SAFE, None, verification.reason)
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, verification.reason)

    def verify_effect(self, call: OperatorCall, receipt: OperatorReceipt | None) -> EffectVerification:
        verification, _ = self._inspect(call)
        if verification.status is not EvidenceStatus.VERIFIED or receipt is None:
            return verification
        actual = self._effect_hash(verification.artifact_hashes)
        if receipt.effect_hash and receipt.effect_hash != actual:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "receipt effect hash mismatch")
        return verification.with_updates(receipt_hash=receipt.sha256())
