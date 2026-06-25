import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "pipeline" / "multimodal" / "run_multimodal_dataset.py"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_t2i_dryrun_creates_manifest_and_placeholder_png(tmp_path):
    out = tmp_path / "t2i"
    subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--route",
            "t2i",
            "--user-request",
            "generate desk objects",
            "--output-root",
            str(out),
            "--dry-run",
            "--num-samples",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "dataset_summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(out / "manifest.jsonl")
    assert summary["route"] == "t2i"
    assert summary["dry_run"] is True
    assert summary["sample_count"] == 2
    assert len(rows) == 2
    first_output = out / rows[0]["output_path"]
    assert first_output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert rows[0]["metadata"]["intended_model"] == "qwen-image-2512"
    assert rows[0]["metadata"]["intended_model_path"].endswith("Qwen-Image-2512")
    assert rows[0]["validation"]["status"] in {"pass", "warn"}
    assert rows[0]["vlm_review"]["status"] == "pending"


def test_edit_dryrun_copies_input_images_and_records_instruction(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    source = input_dir / "source.jpg"
    source.write_bytes(b"fake-jpeg-for-dryrun")

    out = tmp_path / "edit"
    subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--route",
            "edit",
            "--user-request",
            "make the main object red",
            "--input-image-dir",
            str(input_dir),
            "--output-root",
            str(out),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
    )

    rows = read_jsonl(out / "manifest.jsonl")
    assert len(rows) == 1
    copied = out / rows[0]["output_path"]
    assert copied.read_bytes() == source.read_bytes()
    assert rows[0]["edit_instruction"]
    assert rows[0]["metadata"]["intended_model"] == "qwen-image-edit-2511"
    assert rows[0]["metadata"]["intended_model_path"].endswith("Qwen-Image-Edit-2511")
    assert rows[0]["validation"]["status"] in {"fail", "warn", "pass"}


def test_t2v_dryrun_creates_placeholder_gif_video(tmp_path):
    out = tmp_path / "t2v"
    subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--route",
            "t2v",
            "--user-request",
            "generate a short product video",
            "--output-root",
            str(out),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
    )

    rows = read_jsonl(out / "manifest.jsonl")
    assert rows[0]["status"] == "generated"
    assert rows[0]["output_path"].endswith(".gif")
    assert (out / rows[0]["output_path"]).read_bytes().startswith(b"GIF")
    assert rows[0]["metadata"]["backend"] == "placeholder_gif_video"
    assert rows[0]["metadata"]["intended_model"] == "wan2.1-t2v-1.3b"
    assert rows[0]["validation"]["status"] == "pass"


def test_real_adapter_requires_explicit_inference_flag(tmp_path):
    out = tmp_path / "blocked"
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--route",
            "t2i",
            "--user-request",
            "generate desk objects",
            "--output-root",
            str(out),
            "--generator",
            "qwen-image",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "--allow-model-inference" in (result.stderr + result.stdout)


def test_prompt_file_and_resume_skip_existing(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("red cube on white background\nblue mug on wood table\n", encoding="utf-8")
    out = tmp_path / "t2i_prompts"
    cmd = [
        sys.executable,
        str(ENTRYPOINT),
        "--route",
        "t2i",
        "--user-request",
        "fallback request",
        "--prompt-file",
        str(prompts),
        "--output-root",
        str(out),
        "--dry-run",
        "--num-samples",
        "2",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

    rows = read_jsonl(out / "manifest.jsonl")
    assert len(rows) == 2
    assert rows[0]["prompt"] == "red cube on white background"
    assert rows[0]["status"] == "skipped_existing"
    summary = json.loads((out / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"]["skipped_existing"] == 2
