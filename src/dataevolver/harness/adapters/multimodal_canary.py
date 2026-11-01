from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from dataevolver.workflows.multimodal.router import build_request
from dataevolver.workflows.multimodal.run_multimodal_dataset import run as run_multimodal

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
from ..state_store import atomic_write_json


OPERATOR_ID = "multimodal.t2i_canary"
OPERATOR_VERSION = "1"
FIXED_PROMPT = "A red ceramic mug on a plain wooden table, studio lighting, centered composition."
FIXED_SEED = 17
FIXED_HEIGHT = 512
FIXED_WIDTH = 512
FIXED_STEPS = 1
FIXED_PARAMETERS = {
    "prompt": FIXED_PROMPT,
    "seed": FIXED_SEED,
    "height": FIXED_HEIGHT,
    "width": FIXED_WIDTH,
    "steps": FIXED_STEPS,
}


class MultimodalT2ICanaryOperator:
    def __init__(self, *, allowed_write_root: str | Path):
        source = Path(__file__).resolve()
        root = Path(allowed_write_root).resolve()
        self.allowed_write_root = root
        self.spec = OperatorSpec(
            operator_id=OPERATOR_ID,
            version=OPERATOR_VERSION,
            retry_safe=True,
            expected_output_specs=(
                ExpectedOutputSpec("canary_prompt.json"),
                ExpectedOutputSpec("dataset/request.json"),
                ExpectedOutputSpec("dataset/dataset_summary.json"),
                ExpectedOutputSpec("dataset/manifest.jsonl"),
            ),
            model_config_hash="qwen-image-2.1-canary-v1",
            code_revision=_sha256_file(source),
            allowed_write_roots=(str(root),),
            source_path=str(source),
        )

    def _validate_call(self, call: OperatorCall) -> None:
        if (call.operator_id, call.operator_version) != (OPERATOR_ID, OPERATOR_VERSION):
            raise ValueError("operator call does not match registered real T2I canary")
        try:
            Path(call.work_dir).resolve().relative_to(self.allowed_write_root)
        except ValueError as exc:
            raise ValueError("OperatorCall work_dir is outside allowed write root") from exc
        if call.parameters != FIXED_PARAMETERS:
            raise ValueError("parameters do not match the fixed canary contract")

    @staticmethod
    def _model_path() -> Path:
        value = os.environ.get("QWEN_IMAGE_MODEL_PATH", "").strip()
        if not value:
            raise ValueError("QWEN_IMAGE_MODEL_PATH is required for the real T2I canary")
        model = Path(value).expanduser().resolve()
        required = (
            next(iter(sorted((model / "transformer").glob("diffusion_pytorch_model*.safetensors"))), None),
            next(iter(sorted((model / "text_encoder").glob("model*.safetensors"))), None),
            model / "vae" / "diffusion_pytorch_model.safetensors",
            model / "tokenizer",
        )
        if required[0] is None or required[1] is None or not required[2].is_file() or not required[3].is_dir():
            raise ValueError("QWEN_IMAGE_MODEL_PATH is incomplete")
        return model

    @staticmethod
    def _prompt_path(call: OperatorCall) -> Path:
        return Path(call.work_dir).resolve() / "canary_prompt.json"

    @staticmethod
    def _dataset_root(call: OperatorCall) -> Path:
        return Path(call.work_dir).resolve() / "dataset"

    @staticmethod
    def _effect_hash(hashes: dict[str, str]) -> str:
        return hashlib.sha256(canonical_json(hashes).encode("utf-8")).hexdigest()

    def _request(self, call: OperatorCall, model: Path):
        return build_request(
            route="t2i",
            user_request=FIXED_PROMPT,
            output_root=str(self._dataset_root(call)),
            dry_run=False,
            allow_model_inference=True,
            num_samples=1,
            input_image_dir=None,
            prompt_file=str(self._prompt_path(call)),
            generator="qwen-image",
            model_name="qwen-image-2.1",
            model_path=str(model),
            device="cuda:0",
            height=FIXED_HEIGHT,
            width=FIXED_WIDTH,
            steps=FIXED_STEPS,
            num_frames=1,
            fps=1,
            true_cfg_scale=7.5,
            guidance_scale=1.0,
            negative_prompt="",
            seed=FIXED_SEED,
            sample_prefix="real_t2i_canary",
            skip_existing=True,
            validate_outputs=True,
            vlm_eval=False,
            vlm_device="cpu",
            vlm_model_path=None,
            vlm_max_samples=None,
            vlm_max_retries=1,
            vlm_review_mode="studio",
            vlm_use_freeform_first=False,
            vlm_enable_thinking=False,
        )

    def _inspect(self, call: OperatorCall) -> tuple[EffectVerification, tuple[dict[str, str], ...]]:
        self._validate_call(call)
        model = self._model_path()
        prompt_path = self._prompt_path(call)
        root = self._dataset_root(call)
        request_path = root / "request.json"
        summary_path = root / "dataset_summary.json"
        manifest_path = root / "manifest.jsonl"
        required = (prompt_path, request_path, summary_path, manifest_path)
        if any(not path.exists() for path in required):
            return EffectVerification(EvidenceStatus.UNOBSERVED, "required real canary outputs are missing"), ()
        try:
            prompt_rows = json.loads(prompt_path.read_text(encoding="utf-8"))
            request = json.loads(request_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, f"invalid canary metadata: {exc}"), ()
        if prompt_rows != [{"id": "real_t2i_canary", "prompt": FIXED_PROMPT}]:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "canary prompt contract mismatch"), ()
        expected_request = {
            "route": "t2i",
            "dry_run": False,
            "allow_model_inference": True,
            "num_samples": 1,
            "generator": "qwen-image",
            "model_name": "qwen-image-2.1",
            "model_path": str(model),
            "device": "cuda:0",
            "height": FIXED_HEIGHT,
            "width": FIXED_WIDTH,
            "steps": FIXED_STEPS,
            "seed": FIXED_SEED,
            "vlm_eval": False,
        }
        if any(request.get(key) != value for key, value in expected_request.items()):
            return EffectVerification(EvidenceStatus.CONTRADICTED, "real canary request contract mismatch"), ()
        try:
            if Path(request["output_root"]).resolve() != root:
                raise ValueError("request output_root mismatch")
            if Path(request["prompt_file"]).resolve() != prompt_path:
                raise ValueError("request prompt_file mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc)), ()
        if (
            summary.get("schema") != "dataevolver_multimodal_dataset_summary_v1"
            or summary.get("route") != "t2i"
            or summary.get("dry_run") is not False
            or summary.get("generator") != "qwen-image"
            or summary.get("sample_count") != 1
            or len(rows) != 1
        ):
            return EffectVerification(EvidenceStatus.CONTRADICTED, "canary summary or manifest mismatch"), ()
        row = rows[0]
        if row.get("sample_id") != "real_t2i_canary" or row.get("generator") != "qwen-image":
            return EffectVerification(EvidenceStatus.CONTRADICTED, "canary manifest identity mismatch"), ()
        output_value = row.get("output_path")
        if not isinstance(output_value, str) or not output_value:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "canary output_path is missing"), ()
        image_path = (root / output_value).resolve()
        try:
            image_path.relative_to(root)
        except ValueError:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "canary image escapes campaign root"), ()
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "canary image is missing or empty"), ()
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.format not in {"PNG", "JPEG"} or image.size != (FIXED_WIDTH, FIXED_HEIGHT):
                    raise ValueError("format or dimensions mismatch")
        except Exception as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, f"canary image is invalid: {exc}"), ()
        paths = (*required, image_path)
        hashes = {str(path): _sha256_file(path) for path in sorted(paths, key=str)}
        artifacts = tuple({"path": path, "sha256": digest} for path, digest in hashes.items())
        return EffectVerification(
            EvidenceStatus.VERIFIED,
            "real T2I canary artifacts verified",
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
        self._validate_call(call)
        model = self._model_path()
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
        atomic_write_json(self._prompt_path(call), [{"id": "real_t2i_canary", "prompt": FIXED_PROMPT}])
        run_multimodal(self._request(call, model))
        return _write_receipt(receipt_path, self._receipt_from_outputs(call, reused=False))

    def recover(self, call: OperatorCall) -> RecoveryResult:
        self._validate_call(call)
        try:
            receipt = _load_receipt(_receipt_path(call))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, f"invalid receipt: {exc}")
        if receipt is not None:
            verification = self.verify_effect(call, receipt)
            if verification.status is EvidenceStatus.VERIFIED:
                return RecoveryResult(RecoveryStatus.RECOVERED, receipt, "receipt and image verified")
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, verification.reason)
        verification, _ = self._inspect(call)
        if verification.status is EvidenceStatus.VERIFIED:
            recovered = _write_receipt(_receipt_path(call), self._receipt_from_outputs(call, reused=True))
            return RecoveryResult(RecoveryStatus.RECOVERED, recovered, "image artifacts verified and receipt rebuilt")
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
