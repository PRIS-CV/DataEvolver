import json
from pathlib import Path

import pytest
from PIL import Image

import dataevolver.harness.adapters.multimodal_canary as canary_module
from dataevolver.harness.adapters.multimodal_canary import (
    FIXED_HEIGHT,
    FIXED_PARAMETERS,
    FIXED_PROMPT,
    FIXED_SEED,
    FIXED_STEPS,
    FIXED_WIDTH,
    MultimodalT2ICanaryOperator,
)
from dataevolver.harness.cli import _registry, _validate_creation_mode
from dataevolver.harness.models import (
    BudgetSpec,
    EvidenceStatus,
    GoalSpec,
    OperatorCall,
    RecoveryStatus,
    StageStatus,
)
from dataevolver.workflows.stages.t2i_generate import verify_outputs as verify_stage2_outputs


def make_call(tmp_path, *, parameters=None):
    return OperatorCall(
        campaign_id="real-t2i-canary",
        call_id="real-t2i-canary:r000:initial",
        idempotency_key="real-t2i-canary:r000:initial",
        operator_id="multimodal.t2i_canary",
        operator_version="1",
        round_index=0,
        work_dir=str(tmp_path / "campaign" / "rounds" / "r000" / "execute"),
        parameters=parameters
        or {
            "prompt": FIXED_PROMPT,
            "seed": FIXED_SEED,
            "height": FIXED_HEIGHT,
            "width": FIXED_WIDTH,
            "steps": FIXED_STEPS,
        },
    )


def make_model(tmp_path):
    model = tmp_path / "Qwen-Image-2.1"
    (model / "transformer").mkdir(parents=True)
    (model / "text_encoder").mkdir()
    (model / "vae").mkdir()
    (model / "tokenizer").mkdir()
    (model / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors").write_bytes(b"x")
    (model / "text_encoder" / "model-00001-of-00001.safetensors").write_bytes(b"x")
    (model / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    return model


def fake_real_runner(counter):
    def run(request):
        counter["calls"] += 1
        assert request.route == "t2i"
        assert request.dry_run is False
        assert request.allow_model_inference is True
        assert request.generator == "qwen-image"
        assert request.device == "cuda:0"
        assert request.num_samples == 1
        assert request.seed == FIXED_SEED
        assert request.height == FIXED_HEIGHT
        assert request.width == FIXED_WIDTH
        assert request.steps == FIXED_STEPS
        prompt_rows = json.loads(Path(request.prompt_file).read_text(encoding="utf-8"))
        assert prompt_rows == [{"id": "real_t2i_canary", "prompt": FIXED_PROMPT}]

        root = Path(request.output_root)
        image_path = root / "samples" / "images" / "real_t2i_canary.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (FIXED_WIDTH, FIXED_HEIGHT), (180, 20, 20)).save(image_path)
        (root / "request.json").write_text(json.dumps(request.to_dict()), encoding="utf-8")
        row = {
            "sample_id": "real_t2i_canary",
            "route": "t2i",
            "prompt": FIXED_PROMPT,
            "output_path": "samples/images/real_t2i_canary.png",
            "input_path": None,
            "edit_instruction": None,
            "generator": "qwen-image",
            "status": "generated",
            "metadata": {
                "dry_run": False,
                "backend": "dataevolver.workflows.stages.t2i_generate",
                "model_name": "qwen-image-2.1",
                "model_path": request.model_path,
                "device": "cuda:0",
                "height": FIXED_HEIGHT,
                "width": FIXED_WIDTH,
                "steps": FIXED_STEPS,
            },
            "validation": {
                "status": "pass",
                "path": str(image_path),
                "format": "PNG",
                "width": FIXED_WIDTH,
                "height": FIXED_HEIGHT,
            },
            "vlm_review": {"status": "pending", "enabled": False, "reason": "VLM evaluation is deferred"},
        }
        (root / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (root / "dataset_summary.json").write_text(
            json.dumps(
                {
                    "schema": "dataevolver_multimodal_dataset_summary_v1",
                    "route": "t2i",
                    "dry_run": False,
                    "generator": "qwen-image",
                    "sample_count": 1,
                }
            ),
            encoding="utf-8",
        )
        return []

    return run


def test_real_t2i_canary_receipt_is_idempotent_and_recovers_from_artifacts(tmp_path, monkeypatch):
    model = make_model(tmp_path)
    monkeypatch.setenv("QWEN_IMAGE_MODEL_PATH", str(model))
    calls = {"calls": 0}
    monkeypatch.setattr(canary_module, "run_multimodal", fake_real_runner(calls))
    operator = MultimodalT2ICanaryOperator(allowed_write_root=tmp_path / "campaign")
    call = make_call(tmp_path)

    first = operator.execute(call)
    second = operator.execute(call)
    Path(first.receipt_path).unlink()
    recovered = operator.recover(call)

    assert first.status is StageStatus.SUCCESS
    assert second.reused is True
    assert calls["calls"] == 1
    assert recovered.status is RecoveryStatus.RECOVERED
    assert recovered.receipt is not None and recovered.receipt.reused is True
    assert operator.verify_effect(call, recovered.receipt).status is EvidenceStatus.VERIFIED
    assert first.stage_result["metrics"] == {
        "completion_ratio": 1.0,
        "resource_policy_respected": 1.0,
    }


def test_real_t2i_canary_fails_closed_on_tampered_image_or_changed_request(tmp_path, monkeypatch):
    model = make_model(tmp_path)
    monkeypatch.setenv("QWEN_IMAGE_MODEL_PATH", str(model))
    monkeypatch.setattr(canary_module, "run_multimodal", fake_real_runner({"calls": 0}))
    operator = MultimodalT2ICanaryOperator(allowed_write_root=tmp_path / "campaign")
    call = make_call(tmp_path)
    receipt = operator.execute(call)
    image_path = Path(call.work_dir) / "dataset" / "samples" / "images" / "real_t2i_canary.png"
    image_path.write_bytes(b"not an image")

    assert operator.verify_effect(call, receipt).status is EvidenceStatus.CONTRADICTED
    with pytest.raises(ValueError, match="fixed canary contract"):
        operator.execute(make_call(tmp_path, parameters={"prompt": "different"}))


def test_real_t2i_canary_missing_outputs_are_retry_safe(tmp_path, monkeypatch):
    model = make_model(tmp_path)
    monkeypatch.setenv("QWEN_IMAGE_MODEL_PATH", str(model))
    operator = MultimodalT2ICanaryOperator(allowed_write_root=tmp_path / "campaign")

    recovered = operator.recover(make_call(tmp_path))

    assert recovered.status is RecoveryStatus.RETRY_SAFE
    assert recovered.receipt is None


def test_cli_registry_accepts_only_the_named_real_t2i_canary(tmp_path):
    goal = GoalSpec(
        schema_version="dataevolver.harness.goal.v1",
        campaign_id="real-t2i-canary",
        objective={"kind": "real_t2i_canary"},
        initial_operator_id="multimodal.t2i_canary",
        initial_operator_parameters=FIXED_PARAMETERS,
        route_candidates=("t2i",),
        budgets=BudgetSpec(1, 1, 0, 0, 3600),
        resource_policy={"allow_gpu": True, "allowed_operators": ["multimodal.t2i_canary"]},
        frozen_evaluator={"path": "evaluator.json", "sha256": "a" * 64},
    )

    registry = _registry(goal, tmp_path / "campaigns")

    assert isinstance(registry.get("multimodal.t2i_canary"), MultimodalT2ICanaryOperator)


def test_real_canary_cli_mode_is_explicit_and_not_labeled_dry_run(tmp_path):
    goal = GoalSpec(
        schema_version="dataevolver.harness.goal.v1",
        campaign_id="real-t2i-canary",
        objective={"kind": "real_t2i_canary"},
        initial_operator_id="multimodal.t2i_canary",
        initial_operator_parameters=FIXED_PARAMETERS,
        route_candidates=("t2i",),
        budgets=BudgetSpec(1, 1, 0, 0, 3600),
        resource_policy={"allow_gpu": True, "allowed_operators": ["multimodal.t2i_canary"]},
        frozen_evaluator={"path": "evaluator.json", "sha256": "a" * 64},
    )

    _validate_creation_mode(goal, dry_run=False)
    with pytest.raises(ValueError, match="must not use --dry-run"):
        _validate_creation_mode(goal, dry_run=True)


def test_stage2_accepts_a_valid_png_even_when_highly_compressible(tmp_path):
    image_path = tmp_path / "real_t2i_canary.png"
    Image.new("RGB", (32, 32), (180, 20, 20)).save(image_path, optimize=True, compress_level=9)
    assert image_path.stat().st_size < 1024

    assert verify_stage2_outputs(
        [{"id": "real_t2i_canary"}],
        output_dir=str(tmp_path),
    ) is True
