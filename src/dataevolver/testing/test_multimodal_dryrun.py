import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from dataevolver.workflows.multimodal.generators.qwen_image import QwenImageGenerator
from dataevolver.workflows.multimodal.schemas import DatasetRequest
from dataevolver.workflows.multimodal.router import build_request
from dataevolver.workflows.multimodal.run_multimodal_dataset import run


def make_request(out, *, route="t2i", user_request="generate desk objects", dry_run=True, **kwargs):
    values = DatasetRequest(route=route, user_request=user_request, output_root=str(out),
                            dry_run=dry_run, **kwargs).to_dict()
    del values["status"], values["notes"]
    return build_request(**values)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_t2i_dryrun_creates_manifest_and_placeholder_png(tmp_path):
    out = tmp_path / "t2i"
    run(make_request(out, num_samples=2, model_name="fixture-image", model_path="fixture-checkpoint"))

    summary = json.loads((out / "dataset_summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(out / "manifest.jsonl")
    assert summary["route"] == "t2i"
    assert summary["dry_run"] is True
    assert summary["sample_count"] == 2
    assert len(rows) == 2
    first_output = out / rows[0]["output_path"]
    assert first_output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert rows[0]["metadata"]["intended_model"] == "fixture-image"
    assert rows[0]["metadata"]["intended_model_path"] == "fixture-checkpoint"
    assert rows[0]["validation"]["status"] in {"pass", "warn"}
    assert rows[0]["vlm_review"]["status"] == "pending"


def test_edit_dryrun_copies_input_images_and_records_instruction(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    source = input_dir / "source.jpg"
    source.write_bytes(b"fake-jpeg-for-dryrun")

    out = tmp_path / "edit"
    run(make_request(out, route="edit", user_request="make the main object red",
                     input_image_dir=str(input_dir), model_name="fixture-edit", model_path="fixture-checkpoint"))

    rows = read_jsonl(out / "manifest.jsonl")
    assert len(rows) == 1
    copied = out / rows[0]["output_path"]
    assert copied.read_bytes() == source.read_bytes()
    assert rows[0]["edit_instruction"]
    assert rows[0]["metadata"]["intended_model"] == "fixture-edit"
    assert rows[0]["metadata"]["intended_model_path"] == "fixture-checkpoint"
    assert rows[0]["validation"]["status"] in {"fail", "warn", "pass"}


def test_t2v_dryrun_creates_placeholder_gif_video(tmp_path):
    out = tmp_path / "t2v"
    run(make_request(out, route="t2v", user_request="generate a short product video", model_name="fixture-video"))

    rows = read_jsonl(out / "manifest.jsonl")
    assert rows[0]["status"] == "generated"
    assert rows[0]["output_path"].endswith(".gif")
    assert (out / rows[0]["output_path"]).read_bytes().startswith(b"GIF")
    assert rows[0]["metadata"]["backend"] == "placeholder_gif_video"
    assert rows[0]["metadata"]["intended_model"] == "fixture-video"
    assert rows[0]["validation"]["status"] == "pass"


def test_real_adapter_requires_explicit_inference_flag(tmp_path):
    out = tmp_path / "blocked"
    request = make_request(out, generator="qwen-image", dry_run=False)
    with pytest.raises(SystemExit, match="--allow-model-inference"):
        run(request)


def test_model_resolution_has_no_checkpoint_default(monkeypatch):
    from dataevolver.workflows.multimodal.model_registry import resolve_model
    for route in ("t2i", "edit", "t2v", "blender"):
        prefix = "DATAEVOLVER_" + route.upper()
        monkeypatch.delenv(prefix + "_MODEL", raising=False)
        monkeypatch.delenv(prefix + "_MODEL_PATH", raising=False)
        assert resolve_model(route, None, None)["model_name"] == ""
        assert resolve_model(route, None, None)["model_path"] == ""
    monkeypatch.setenv("DATAEVOLVER_T2I_MODEL", "host-model")
    monkeypatch.setenv("DATAEVOLVER_T2I_MODEL_PATH", "host-checkpoint")
    assert resolve_model("t2i", None, None)["model_name"] == "host-model"
    assert resolve_model("t2i", "explicit-model", "explicit-checkpoint")["model_path"] == "explicit-checkpoint"


def test_prompt_file_and_resume_skip_existing(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("red cube on white background\nblue mug on wood table\n", encoding="utf-8")
    out = tmp_path / "t2i_prompts"
    request = make_request(out, user_request="fallback request", prompt_file=str(prompts),
                           num_samples=2)
    run(request)
    run(request)

    rows = read_jsonl(out / "manifest.jsonl")
    assert len(rows) == 2
    assert rows[0]["prompt"] == "red cube on white background"
    assert rows[0]["status"] == "skipped_existing"
    summary = json.loads((out / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"]["skipped_existing"] == 2


def test_qwen_image_generator_only_passes_supported_stage2_arguments(tmp_path, monkeypatch):
    output_root = tmp_path / "real_t2i"
    request = DatasetRequest(
        route="t2i",
        user_request="A red ceramic mug on a plain wooden table, studio lighting, centered composition.",
        output_root=str(output_root),
        dry_run=False,
        allow_model_inference=True,
        num_samples=1,
        generator="qwen-image",
        model_name="qwen-image-2.1",
        model_path=str(tmp_path / "Qwen-Image-2.1"),
        device="cuda:0",
        height=512,
        width=512,
        steps=1,
        true_cfg_scale=4.0,
        seed=17,
        sample_prefix="canary",
        skip_existing=True,
        validate_outputs=True,
    )

    def fake_stage2(cmd, **kwargs):
        assert "--cfg-scale" not in cmd
        prompt_path = Path(cmd[cmd.index("--prompts-path") + 1])
        image_dir = Path(cmd[cmd.index("--output-dir") + 1])
        image_dir.mkdir(parents=True, exist_ok=True)
        for row in json.loads(prompt_path.read_text(encoding="utf-8")):
            Image.new("RGB", (512, 512), (180, 30, 30)).save(image_dir / f"{row['id']}.png")
        return SimpleNamespace(returncode=0, stdout="stage2 ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_stage2)

    samples = QwenImageGenerator().generate(request, output_root)

    assert len(samples) == 1
    assert samples[0].status == "generated"
    assert (output_root / samples[0].output_path).is_file()
