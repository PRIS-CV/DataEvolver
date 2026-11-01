from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from dataevolver.agents.feedback_apply import apply_action, load_action_space
from dataevolver.paths import REPO_ROOT, SRC_ROOT

from ..models import (
    EffectVerification,
    EvidenceStatus,
    ExpectedOutputSpec,
    JsonModel,
    OperatorCall,
    OperatorReceipt,
    OperatorSpec,
    RecoveryResult,
    RecoveryStatus,
    StageStatus,
    StrategyPatch,
    canonical_json,
)
from ..operators import _load_receipt, _receipt_path, _sha256_file, _write_receipt
from ..state_store import atomic_write_json


OPERATOR_ID = "dataevolver.3d_asset_evolution"
OPERATOR_VERSION = "1"
STAGES = ("t2i", "segment", "reconstruct", "sanitize", "render", "review")


@dataclass(frozen=True)
class StageExecution(JsonModel):
    return_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class StageInvocation(JsonModel):
    stage: str
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str]
    timeout_seconds: int
    input_paths: tuple[str, ...]
    output_paths: tuple[str, ...]


class StageRunner(Protocol):
    def run(self, invocation: StageInvocation) -> StageExecution: ...


class SubprocessStageRunner:
    def run(self, invocation: StageInvocation) -> StageExecution:
        completed = subprocess.run(
            list(invocation.argv),
            cwd=invocation.cwd,
            env=dict(invocation.env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=invocation.timeout_seconds,
            check=False,
            shell=False,
        )
        return StageExecution(
            return_code=completed.returncode,
            stdout=(completed.stdout or "")[-4000:],
            stderr=(completed.stderr or "")[-4000:],
        )


class ThreeDAssetEvolutionOperator:
    _CALL_FIELDS = {
        "asset_id",
        "experiment_kind",
        "initial_prompt",
        "t2i_seed",
        "reconstruction_seed",
        "strategy_patch",
        "incumbent",
        "required_evidence",
        "compatibility_hash",
    }

    def __init__(
        self,
        *,
        allowed_write_root: str | Path,
        resource_config: dict[str, Any],
        stage_runner: StageRunner | None = None,
    ):
        self.allowed_write_root = Path(allowed_write_root).resolve()
        self.resource_config = dict(resource_config)
        self.stage_runner = stage_runner or SubprocessStageRunner()
        source = Path(__file__).resolve()
        resource_fingerprint = {
            key: value
            for key, value in self.resource_config.items()
            if key not in {"api_key", "token", "secret"}
        }
        resource_fingerprint["resolved_resources"] = self._resource_fingerprint(resource_fingerprint)
        self._spec = OperatorSpec(
            operator_id=OPERATOR_ID,
            version=OPERATOR_VERSION,
            retry_safe=True,
            expected_output_specs=(
                ExpectedOutputSpec("bundle_manifest.json"),
                ExpectedOutputSpec("stage_result.json"),
                ExpectedOutputSpec("stage_receipts.jsonl"),
            ),
            model_config_hash=hashlib.sha256(canonical_json(resource_fingerprint).encode("utf-8")).hexdigest(),
            code_revision=_sha256_file(source),
            allowed_env_keys=tuple(
                str(item)
                for item in self.resource_config.get(
                    "allowed_env_keys",
                    (
                        "PATH", "PYTHONPATH", "HOME", "TMPDIR", "LD_LIBRARY_PATH", "CUDA_HOME",
                        "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "TRANSFORMERS_CACHE",
                        "DISPLAY", "XDG_RUNTIME_DIR", "CUDA_VISIBLE_DEVICES", "QWEN_IMAGE_MODEL_PATH",
                        "SAM3_CKPT", "SAM3_DIR", "HUNYUAN3D_REPO", "MODEL_HUB",
                        "PAINT_MODEL_HUB", "DINO_MODEL_PATH", "REALESRGAN_CKPT", "VLM_MODEL_PATH",
                        "BLENDER_PYTHON_SITE_PACKAGES",
                    ),
                )
            ),
            allowed_write_roots=(str(self.allowed_write_root),),
            source_path=str(source),
        )

    @property
    def spec(self) -> OperatorSpec:
        resource_fingerprint = {
            key: value
            for key, value in self.resource_config.items()
            if key not in {"api_key", "token", "secret"}
        }
        resource_fingerprint["resolved_resources"] = self._resource_fingerprint(resource_fingerprint)
        digest = hashlib.sha256(canonical_json(resource_fingerprint).encode("utf-8")).hexdigest()
        return self._spec.with_updates(model_config_hash=digest)

    @staticmethod
    def _resource_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
        candidates = {
            "scene_template": config.get("scene_template"),
            "action_space": config.get("action_space"),
            **dict(config.get("resource_paths") or {}),
        }
        fingerprint = {}
        for name, raw in candidates.items():
            if not raw:
                fingerprint[name] = {"status": "UNSET"}
                continue
            path = Path(str(raw)).expanduser().resolve()
            if path.is_file():
                fingerprint[name] = {"kind": "file", "path": str(path), "sha256": _sha256_file(path)}
            elif path.is_dir():
                inventory = []
                for item in sorted(path.rglob("*"), key=lambda value: str(value)):
                    if item.is_file() and item.suffix.lower() in {".json", ".safetensors", ".pt", ".pth"}:
                        inventory.append(
                            {"path": str(item.relative_to(path)), "size": item.stat().st_size}
                        )
                    if len(inventory) >= 256:
                        break
                fingerprint[name] = {"kind": "directory", "path": str(path), "inventory": inventory}
            else:
                fingerprint[name] = {"status": "MISSING", "path": str(path)}
        return fingerprint

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _validate_call(self, call: OperatorCall) -> None:
        if (call.operator_id, call.operator_version) != (OPERATOR_ID, OPERATOR_VERSION):
            raise ValueError("operator call does not match registered 3D asset evolution Operator")
        root = Path(call.work_dir).resolve()
        try:
            root.relative_to(self.allowed_write_root)
        except ValueError as exc:
            raise ValueError("OperatorCall work_dir is outside allowed write root") from exc
        unknown = sorted(set(call.parameters) - self._CALL_FIELDS)
        if unknown:
            raise ValueError(f"unknown OperatorCall parameters: {', '.join(unknown)}")
        asset_id = call.parameters.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in asset_id):
            raise ValueError("asset_id is invalid")
        kind = str(call.parameters.get("experiment_kind") or "")
        if kind not in {"INITIAL", "PROMPT_REGENERATE", "RESEGMENT_RECONSTRUCT", "RECONSTRUCT_RETRY", "SCENE_REFINE"}:
            raise ValueError("experiment_kind is invalid")
        incumbent = call.parameters.get("incumbent") or {}
        if incumbent:
            incumbent_root = Path(str(incumbent.get("work_dir") or "")).resolve()
            try:
                incumbent_root.relative_to(self.allowed_write_root)
            except ValueError as exc:
                raise ValueError("incumbent work_dir is outside allowed write root") from exc

    @staticmethod
    def _paths(root: Path, asset_id: str, round_index: int, views: tuple[tuple[int, int], ...]) -> dict[str, Any]:
        render_dir = root / "renders" / asset_id
        return {
            "strategy": root / "strategy.json",
            "prompt": root / "prompt" / "prompts.json",
            "rgb": root / "images" / f"{asset_id}.png",
            "rgba": root / "images_rgba" / f"{asset_id}.png",
            "mesh_raw": root / "meshes_raw" / f"{asset_id}.glb",
            "reconstruct_summary": root / "meshes_raw" / "stage3_summary.json",
            "mesh": root / "meshes" / f"{asset_id}.glb",
            "mesh_qc": root / "mesh_qc" / f"{asset_id}.json",
            "control_state": root / "control_state.json",
            "scene_template": root / "scene_template.json",
            "render_metadata": render_dir / "metadata.json",
            "renders": tuple(render_dir / f"az{az:03d}_el{el:+03d}.png" for az, el in views),
            "review": root / "reviews" / f"{asset_id}_r{round_index:02d}_agg.json",
            "stage_receipts": root / "stage_receipts.jsonl",
            "bundle": root / "bundle_manifest.json",
            "stage_result": root / "stage_result.json",
        }

    def _views(self) -> tuple[tuple[int, int], ...]:
        raw = self.resource_config.get("representative_views", ((0, 0), (90, 0), (180, 0), (270, 0)))
        views = tuple((int(item[0]), int(item[1])) for item in raw)
        if views != ((0, 0), (90, 0), (180, 0), (270, 0)):
            raise ValueError("representative views must be the frozen four-view contract")
        return views

    def _environment(self) -> dict[str, str]:
        environment = {}
        for key in self.spec.allowed_env_keys:
            if key in os.environ:
                environment[key] = os.environ[key]
        environment.setdefault("PATH", os.environ.get("PATH", ""))
        python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(SRC_ROOT), python_path] if python_path else [str(SRC_ROOT)]
        )
        physical_gpu = self.resource_config.get("physical_gpu_id")
        if physical_gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
        return environment

    def _strategy(self, call: OperatorCall) -> tuple[StrategyPatch, str]:
        kind = str(call.parameters["experiment_kind"])
        if kind == "INITIAL":
            patch = StrategyPatch.from_dict(
                {
                    "prompt_text": str(call.parameters.get("initial_prompt") or "").strip(),
                    "t2i_seed": int(call.parameters.get("t2i_seed", 17)),
                    "reconstruction_seed": int(call.parameters.get("reconstruction_seed", 42)),
                    "restart_stage": "t2i",
                }
            )
        else:
            patch = StrategyPatch.from_dict(dict(call.parameters.get("strategy_patch") or {}))
        restart = {
            "INITIAL": "t2i",
            "PROMPT_REGENERATE": "t2i",
            "RESEGMENT_RECONSTRUCT": "segment",
            "RECONSTRUCT_RETRY": "reconstruct",
            "SCENE_REFINE": "render",
        }[kind]
        if patch.restart_stage != restart:
            raise ValueError("strategy restart_stage does not match experiment_kind")
        return patch, restart

    def _prepare_inputs(self, call: OperatorCall, paths: dict[str, Any], patch: StrategyPatch) -> None:
        root = Path(call.work_dir).resolve()
        asset_id = str(call.parameters["asset_id"])
        incumbent = call.parameters.get("incumbent") or {}
        prompt = patch.prompt_text
        if prompt is None and incumbent:
            previous = Path(str(incumbent["work_dir"])) / "prompt" / "prompts.json"
            if previous.is_file():
                rows = json.loads(previous.read_text(encoding="utf-8"))
                prompt = str(rows[0]["prompt"])
        if not prompt:
            raise ValueError("a prompt is required for 3D asset materialization")
        paths["prompt"].parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            paths["prompt"],
            [{"id": asset_id, "name": asset_id, "prompt": prompt, "features": {}}],
        )
        action_space_path = Path(str(self.resource_config["action_space"])).resolve()
        action_space = load_action_space(str(action_space_path))
        control_state = dict(action_space.get("default_control_state") or {})
        if incumbent:
            previous_control = Path(str(incumbent["work_dir"])) / "control_state.json"
            if previous_control.is_file():
                control_state = json.loads(previous_control.read_text(encoding="utf-8"))
        for action in patch.scene_actions:
            result = apply_action(action, control_state, action_space)
            if not result.applied:
                raise ValueError(f"scene action could not be applied: {action}")
            control_state = result.state
        atomic_write_json(paths["control_state"], control_state)
        template_path = Path(str(self.resource_config["scene_template"])).resolve()
        template = json.loads(template_path.read_text(encoding="utf-8"))
        template["qc_views"] = [list(item) for item in self._views()]
        atomic_write_json(paths["scene_template"], template)
        atomic_write_json(
            paths["strategy"],
            {
                "asset_id": asset_id,
                "experiment_kind": call.parameters["experiment_kind"],
                "strategy_patch": patch.to_dict(),
                "matched_trial_key": call.matched_trial_key.to_dict() if call.matched_trial_key else None,
            },
        )
        root.mkdir(parents=True, exist_ok=True)

    def _invocations(self, call: OperatorCall, paths: dict[str, Any], patch: StrategyPatch) -> dict[str, StageInvocation]:
        asset_id = str(call.parameters["asset_id"])
        python = str(self.resource_config.get("python_executable") or sys.executable)
        blender = str(self.resource_config.get("blender_executable") or os.environ.get("BLENDER_BIN") or "blender")
        device = str(self.resource_config.get("device") or "cuda:0")
        timeout = int(self.resource_config.get("stage_timeout_seconds", 7200))
        environment = self._environment()
        negative = ", ".join(
            (
                str(
                    self.resource_config.get(
                        "base_negative_prompt",
                        "blurry, low quality, distorted, multiple objects, background clutter",
                    )
                ),
                *patch.negative_prompt_append,
            )
        )
        t2i_seed = patch.t2i_seed if patch.t2i_seed is not None else int(call.parameters.get("t2i_seed", 17))
        reconstruction_seed = (
            patch.reconstruction_seed
            if patch.reconstruction_seed is not None
            else int(call.parameters.get("reconstruction_seed", 42))
        )
        render_outputs = (*paths["renders"], paths["render_metadata"])
        return {
            "t2i": StageInvocation(
                "t2i",
                (
                    python, "-m", "dataevolver.workflows.stages.t2i_generate",
                    "--prompts-path", str(paths["prompt"]), "--output-dir", str(paths["rgb"].parent),
                    "--ids", asset_id, "--device", device, "--seed-base", str(t2i_seed),
                    "--height", str(int(self.resource_config.get("t2i_height", 512))),
                    "--width", str(int(self.resource_config.get("t2i_width", 512))),
                    "--steps", str(int(self.resource_config.get("t2i_steps", 20))),
                    "--negative-prompt", negative, "--no-skip",
                ),
                str(REPO_ROOT), environment, timeout, (str(paths["prompt"]),), (str(paths["rgb"]),),
            ),
            "segment": StageInvocation(
                "segment",
                (
                    python, "-m", "dataevolver.workflows.stages.sam_segment", "--ids", asset_id,
                    "--input-dir", str(paths["rgb"].parent), "--output-dir", str(paths["rgba"].parent),
                    "--prompts-path", str(paths["prompt"]), "--device", device, "--no-skip", "--no-fallback",
                ),
                str(REPO_ROOT), environment, timeout, (str(paths["rgb"]),), (str(paths["rgba"]),),
            ),
            "reconstruct": StageInvocation(
                "reconstruct",
                (
                    python, "-m", "dataevolver.workflows.stages.image_to_3d", "--ids", asset_id,
                    "--images-dir", str(paths["rgba"].parent), "--rgb-images-dir", str(paths["rgb"].parent),
                    "--output-dir", str(paths["mesh_raw"].parent), "--device", device,
                    "--seed-base", str(reconstruction_seed), "--no-skip", "--no-shape-only-fallback",
                ),
                str(REPO_ROOT), environment, timeout, (str(paths["rgba"]), str(paths["rgb"])),
                (str(paths["mesh_raw"]), str(paths["reconstruct_summary"])),
            ),
            "sanitize": StageInvocation(
                "sanitize",
                (
                    python, "-m", "dataevolver.workflows.stages.mesh_sanitize", "--ids", asset_id,
                    "--input-dir", str(paths["mesh_raw"].parent), "--output-dir", str(paths["mesh"].parent),
                    "--qc-dir", str(paths["mesh_qc"].parent), "--no-skip",
                ),
                str(REPO_ROOT), environment, timeout, (str(paths["mesh_raw"]),),
                (str(paths["mesh"]), str(paths["mesh_qc"])),
            ),
            "render": StageInvocation(
                "render",
                (
                    blender, "-b", "--python", str(Path(__file__).parents[2] / "workflows" / "stages" / "scene_render.py"),
                    "--", "--input-dir", str(paths["mesh"].parent), "--output-dir", str(paths["render_metadata"].parents[1]),
                    "--obj-id", asset_id, "--scene-template", str(paths["scene_template"]),
                    "--control-state", str(paths["control_state"]), "--resolution",
                    str(int(self.resource_config.get("render_resolution", 512))),
                ),
                str(REPO_ROOT), environment, timeout, (str(paths["mesh"]), str(paths["control_state"])),
                tuple(str(item) for item in render_outputs),
            ),
            "review": StageInvocation(
                "review",
                (
                    python, "-m", "dataevolver.annotation.vlm_review_stage", "--renders-dir",
                    str(paths["render_metadata"].parents[1]), "--obj-id", asset_id, "--output-dir",
                    str(paths["review"].parent), "--round-idx", str(call.round_index), "--device", device,
                    "--review-mode", "scene_insert", "--action-space-path", str(self.resource_config["action_space"]),
                ),
                str(REPO_ROOT), environment, timeout, tuple(str(item) for item in paths["renders"]),
                (str(paths["review"]),),
            ),
        }

    @staticmethod
    def _read_stage_receipts(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        receipts = []
        lines = path.read_bytes().splitlines()
        for index, raw in enumerate(lines):
            if not raw.strip():
                continue
            try:
                receipts.append(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if index == len(lines) - 1:
                    break
                raise ValueError("stage receipt log is malformed")
        return receipts

    @staticmethod
    def _append_stage_receipt(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _verify_receipt_outputs(receipt: dict[str, Any]) -> bool:
        outputs = receipt.get("output_hashes") or {}
        return bool(outputs) and all(
            Path(path).is_file() and _sha256_file(Path(path)) == digest for path, digest in outputs.items()
        )

    def _copy_reused_stage(self, incumbent_root: Path, invocation: StageInvocation) -> None:
        candidate_root = Path(invocation.output_paths[0]).parents[1]
        for output_text in invocation.output_paths:
            output = Path(output_text)
            relative = output.relative_to(candidate_root)
            source = incumbent_root / relative
            if not source.is_file():
                raise RuntimeError(f"incumbent artifact is missing for reuse: {source}")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)

    def _write_failure_bundle(
        self,
        call: OperatorCall,
        paths: dict[str, Any],
        stage: str,
        execution: StageExecution,
    ) -> OperatorReceipt:
        result = {
            "operator_id": OPERATOR_ID,
            "operator_version": OPERATOR_VERSION,
            "status": StageStatus.FAILED.value,
            "metrics": {},
            "hard_constraints": {"artifact_lineage_verified": True},
            "failure_code": f"{stage.upper()}_FAILED",
            "failure_detail": (execution.stderr or execution.stdout)[-4000:],
            "matched_trial_key": call.matched_trial_key.to_dict() if call.matched_trial_key else None,
        }
        atomic_write_json(paths["stage_result"], result)
        artifacts = {
            str(paths["stage_result"]): _sha256_file(paths["stage_result"]),
            str(paths["stage_receipts"]): _sha256_file(paths["stage_receipts"]),
        }
        atomic_write_json(
            paths["bundle"],
            {"schema_version": "dataevolver.3d_asset.bundle.v1", "complete": False, "failure_stage": stage, "artifacts": artifacts},
        )
        return _write_receipt(_receipt_path(call), self._receipt_from_bundle(call, reused=False))

    def _finalize_bundle(self, call: OperatorCall, paths: dict[str, Any]) -> OperatorReceipt:
        review = json.loads(paths["review"].read_text(encoding="utf-8"))
        hard = {
            "rgb_readable": self._valid_image(paths["rgb"], "RGB"),
            "rgba_usable": self._valid_image(paths["rgba"], "RGBA"),
            "glb_readable": paths["mesh"].is_file() and paths["mesh"].stat().st_size > 16,
            "mesh_nonempty": paths["mesh"].is_file() and paths["mesh"].stat().st_size > 16,
            "texture_present_if_required": not bool(self.resource_config.get("require_texture", False))
            or self._textured_summary(paths["reconstruct_summary"]),
            "representative_views_complete": len(paths["renders"]) == 4
            and all(self._valid_image(path, "RGB") for path in paths["renders"]),
            "review_complete": int(review.get("num_views_reviewed", 0)) == 4,
            "structure_not_major_mismatch": review.get("structure_consistency") != "major_mismatch",
            "physics_not_major_issue": review.get("physics_consistency") != "major_issue",
            "asset_not_abandon": review.get("asset_viability") != "abandon",
            "artifact_lineage_verified": True,
        }
        scores = review.get("agg_vlm_scores") or {}
        metrics = {
            "hybrid_quality": float(review.get("hybrid_score", 0.0)),
            "object_integrity": float(scores.get("object_integrity", 0.0)) / 5.0,
            "geometry_structure_consistency": 1.0 if review.get("structure_consistency") == "good" else 0.5,
            "composition_multiview_usability": float(scores.get("composition", 0.0)) / 5.0,
            "render_quality": float(scores.get("render_quality_semantic", 0.0)) / 5.0,
            "prompt_identity_alignment": float(scores.get("object_integrity", 0.0)) / 5.0,
            "geometry_validity": 1.0 if hard["mesh_nonempty"] else 0.0,
            "texture_presence": 1.0 if hard["texture_present_if_required"] else 0.0,
            "critical_physics_consistency": 1.0 if hard["physics_not_major_issue"] else 0.0,
        }
        failure_code = ""
        if not all(hard.values()):
            failure_code = next(name.upper() for name, value in hard.items() if not value)
        elif review.get("issue_tags") and review.get("issue_tags") != ["none"]:
            failure_code = str(review["issue_tags"][0]).upper()
        result = {
            "operator_id": OPERATOR_ID,
            "operator_version": OPERATOR_VERSION,
            "status": StageStatus.SUCCESS.value if all(hard.values()) else StageStatus.FAILED.value,
            "metrics": metrics,
            "hard_constraints": hard,
            "failure_code": failure_code,
            "evidence_refs": [str(paths["review"]), str(paths["mesh_qc"]), str(paths["render_metadata"])],
            "matched_trial_key": call.matched_trial_key.to_dict() if call.matched_trial_key else None,
        }
        atomic_write_json(paths["stage_result"], result)
        artifact_paths = [
            paths["strategy"], paths["prompt"], paths["rgb"], paths["rgba"], paths["mesh_raw"],
            paths["reconstruct_summary"], paths["mesh"], paths["mesh_qc"], paths["control_state"],
            paths["scene_template"], paths["render_metadata"], *paths["renders"], paths["review"],
            paths["stage_receipts"], paths["stage_result"],
        ]
        artifacts = {str(path): _sha256_file(path) for path in artifact_paths}
        atomic_write_json(
            paths["bundle"],
            {
                "schema_version": "dataevolver.3d_asset.bundle.v1",
                "complete": True,
                "asset_id": call.parameters["asset_id"],
                "representative_views": [list(item) for item in self._views()],
                "artifacts": artifacts,
            },
        )
        return _write_receipt(_receipt_path(call), self._receipt_from_bundle(call, reused=False))

    @staticmethod
    def _valid_image(path: Path, mode: str) -> bool:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                return image.mode == mode
        except Exception:
            return False

    @staticmethod
    def _textured_summary(path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return int((payload.get("stats") or {}).get("textured_success_count", 0)) >= 1

    def _inspect_bundle(self, call: OperatorCall) -> tuple[EffectVerification, tuple[dict[str, str], ...]]:
        root = Path(call.work_dir).resolve()
        bundle_path = root / "bundle_manifest.json"
        result_path = root / "stage_result.json"
        if not bundle_path.is_file() or not result_path.is_file():
            return EffectVerification(EvidenceStatus.UNOBSERVED, "bundle or stage result is missing"), ()
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, f"bundle metadata is malformed: {exc}"), ()
        if bundle.get("schema_version") != "dataevolver.3d_asset.bundle.v1":
            return EffectVerification(EvidenceStatus.CONTRADICTED, "bundle schema mismatch"), ()
        hashes = dict(bundle.get("artifacts") or {})
        if not hashes:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "bundle artifact inventory is empty"), ()
        for path_text, expected in hashes.items():
            path = Path(path_text).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                return EffectVerification(EvidenceStatus.CONTRADICTED, "bundle artifact escapes candidate root"), ()
            if not path.is_file() or _sha256_file(path) != expected:
                return EffectVerification(EvidenceStatus.CONTRADICTED, f"bundle artifact hash mismatch: {path}"), ()
        hashes[str(bundle_path)] = _sha256_file(bundle_path)
        artifacts = tuple({"path": path, "sha256": digest} for path, digest in sorted(hashes.items()))
        return EffectVerification(EvidenceStatus.VERIFIED, "3D asset bundle verified", hashes), artifacts

    def _receipt_from_bundle(self, call: OperatorCall, *, reused: bool) -> OperatorReceipt:
        verification, artifacts = self._inspect_bundle(call)
        if verification.status is not EvidenceStatus.VERIFIED:
            raise RuntimeError(verification.reason)
        result = json.loads((Path(call.work_dir) / "stage_result.json").read_text(encoding="utf-8"))
        now = self._now()
        return OperatorReceipt(
            call_id=call.call_id,
            operator_id=OPERATOR_ID,
            operator_version=OPERATOR_VERSION,
            status=StageStatus(result["status"]),
            started_at=now,
            finished_at=now,
            return_code=0 if result["status"] == StageStatus.SUCCESS.value else 1,
            output_artifacts=artifacts,
            effect_hash=hashlib.sha256(canonical_json(verification.artifact_hashes).encode("utf-8")).hexdigest(),
            reused=reused,
            stage_result=result,
        )

    def execute(self, call: OperatorCall) -> OperatorReceipt:
        self._validate_call(call)
        prior = _load_receipt(_receipt_path(call))
        if prior is not None:
            verification = self.verify_effect(call, prior)
            if verification.status is not EvidenceStatus.VERIFIED:
                raise RuntimeError(verification.reason)
            return prior.with_updates(reused=True)
        existing, _ = self._inspect_bundle(call)
        if existing.status is EvidenceStatus.VERIFIED:
            return _write_receipt(_receipt_path(call), self._receipt_from_bundle(call, reused=True))
        if existing.status is EvidenceStatus.CONTRADICTED:
            raise RuntimeError(existing.reason)
        root = Path(call.work_dir).resolve()
        patch, restart = self._strategy(call)
        paths = self._paths(root, str(call.parameters["asset_id"]), call.round_index, self._views())
        self._prepare_inputs(call, paths, patch)
        invocations = self._invocations(call, paths, patch)
        receipts = self._read_stage_receipts(paths["stage_receipts"])
        receipt_by_stage = {item["stage"]: item for item in receipts}
        for stage, receipt in receipt_by_stage.items():
            if stage not in STAGES or not self._verify_receipt_outputs(receipt):
                raise RuntimeError(f"stage receipt is contradicted: {stage}")
        restart_index = STAGES.index(restart)
        incumbent = call.parameters.get("incumbent") or {}
        incumbent_root = Path(str(incumbent.get("work_dir") or "")).resolve() if incumbent else None
        for index, stage in enumerate(STAGES):
            if stage in receipt_by_stage:
                continue
            invocation = invocations[stage]
            started = self._now()
            if index < restart_index:
                if incumbent_root is None:
                    raise RuntimeError(f"no incumbent available to reuse stage: {stage}")
                self._copy_reused_stage(incumbent_root, invocation)
                execution = StageExecution(0, "reused verified incumbent artifacts", "")
                status = "REUSED"
            else:
                execution = self.stage_runner.run(invocation)
                status = "VERIFIED" if execution.return_code == 0 else "FAILED"
            output_hashes = {
                path: _sha256_file(Path(path)) for path in invocation.output_paths if Path(path).is_file()
            }
            receipt_payload = {
                "schema_version": "dataevolver.3d_asset.stage_receipt.v1",
                "stage": stage,
                "status": status,
                "argv": list(invocation.argv),
                "input_hashes": {
                    path: _sha256_file(Path(path)) for path in invocation.input_paths if Path(path).is_file()
                },
                "output_hashes": output_hashes,
                "started_at": started,
                "finished_at": self._now(),
                "return_code": execution.return_code,
                "stdout_tail": execution.stdout[-4000:],
                "stderr_tail": execution.stderr[-4000:],
            }
            self._append_stage_receipt(paths["stage_receipts"], receipt_payload)
            if execution.return_code != 0:
                return self._write_failure_bundle(call, paths, stage, execution)
            if len(output_hashes) != len(invocation.output_paths):
                raise RuntimeError(f"stage did not produce all declared outputs: {stage}")
        return self._finalize_bundle(call, paths)

    def recover(self, call: OperatorCall) -> RecoveryResult:
        try:
            self._validate_call(call)
            receipt = _load_receipt(_receipt_path(call))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, f"invalid receipt or call: {exc}")
        if receipt is not None:
            verification = self.verify_effect(call, receipt)
            if verification.status is EvidenceStatus.VERIFIED:
                return RecoveryResult(RecoveryStatus.RECOVERED, receipt, "receipt and 3D bundle verified")
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, verification.reason)
        verification, _ = self._inspect_bundle(call)
        if verification.status is EvidenceStatus.VERIFIED:
            recovered = _write_receipt(_receipt_path(call), self._receipt_from_bundle(call, reused=True))
            return RecoveryResult(RecoveryStatus.RECOVERED, recovered, "3D bundle verified and receipt rebuilt")
        if verification.status is EvidenceStatus.CONTRADICTED:
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, verification.reason)
        try:
            root = Path(call.work_dir).resolve()
            paths = self._paths(root, str(call.parameters["asset_id"]), call.round_index, self._views())
            receipts = self._read_stage_receipts(paths["stage_receipts"])
            seen = []
            for item in receipts:
                stage = str(item.get("stage") or "")
                if stage not in STAGES or stage in seen or STAGES.index(stage) != len(seen):
                    return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, "stage receipt sequence is invalid")
                if not self._verify_receipt_outputs(item):
                    return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, f"stage receipt is contradicted: {stage}")
                seen.append(stage)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, f"stage recovery failed: {exc}")
        return RecoveryResult(RecoveryStatus.RETRY_SAFE, None, "verified stage prefix can be resumed")

    def verify_effect(self, call: OperatorCall, receipt: OperatorReceipt | None) -> EffectVerification:
        self._validate_call(call)
        verification, _ = self._inspect_bundle(call)
        if verification.status is not EvidenceStatus.VERIFIED or receipt is None:
            return verification
        actual = hashlib.sha256(canonical_json(verification.artifact_hashes).encode("utf-8")).hexdigest()
        if receipt.effect_hash and receipt.effect_hash != actual:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "receipt effect hash mismatch")
        return verification.with_updates(receipt_hash=receipt.sha256())
