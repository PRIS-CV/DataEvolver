import json
from pathlib import Path

import pytest
from PIL import Image

from dataevolver.harness.adapters.three_d_asset import (
    OPERATOR_ID,
    StageExecution,
    ThreeDAssetEvolutionOperator,
)
from dataevolver.harness.models import (
    EvidenceStatus,
    MatchedTrialKey,
    OperatorCall,
    RecoveryStatus,
)


VIEWS = ((0, 0), (90, 0), (180, 0), (270, 0))


class FakeStageRunner:
    def __init__(self, *, fail_once_at=None):
        self.fail_once_at = fail_once_at
        self.calls = []

    def run(self, invocation):
        self.calls.append(invocation.stage)
        if self.fail_once_at == invocation.stage:
            self.fail_once_at = None
            raise RuntimeError(f"injected crash before {invocation.stage}")
        for path_text in invocation.output_paths:
            path = Path(path_text)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".png":
                mode = "RGBA" if "images_rgba" in path.parts else "RGB"
                color = (180, 30, 30, 255) if mode == "RGBA" else (180, 30, 30)
                Image.new(mode, (64, 64), color).save(path)
            elif path.suffix.lower() == ".glb":
                path.write_bytes(b"glTF" + b"0" * 128)
            elif path.name.endswith("_agg.json"):
                path.write_text(
                    json.dumps(
                        {
                            "hybrid_score": 0.86,
                            "agg_vlm_scores": {
                                "object_integrity": 4.5,
                                "composition": 4.2,
                                "render_quality_semantic": 4.3,
                                "overall": 4.4,
                            },
                            "issue_tags": ["none"],
                            "suggested_actions": ["NO_OP"],
                            "lighting_diagnosis": "good",
                            "structure_consistency": "good",
                            "color_consistency": "good",
                            "physics_consistency": "good",
                            "asset_viability": "continue",
                            "num_views_reviewed": 4,
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return StageExecution(return_code=0, stdout=f"{invocation.stage} ok", stderr="")


def trial_key():
    return MatchedTrialKey(
        input_artifact_hash="asset-request",
        parent_artifact_hash="parent",
        seed=17,
        operator_id=OPERATOR_ID,
        operator_version="1",
        model_config_hash="models",
        preprocessing_hash="pre",
        evaluator_hash="eval",
        environment_contract_hash="env",
    )


def make_call(tmp_path, *, work_dir=None, parameters=None, call_id="call-1"):
    return OperatorCall(
        campaign_id="campaign",
        call_id=call_id,
        idempotency_key=call_id,
        operator_id=OPERATOR_ID,
        operator_version="1",
        round_index=0,
        work_dir=str(work_dir or tmp_path / "campaign" / "rounds" / "r000" / "assets" / "mug" / "candidate"),
        parameters=parameters
        or {
            "asset_id": "mug",
            "experiment_kind": "INITIAL",
            "initial_prompt": "A single red ceramic mug.",
            "t2i_seed": 17,
            "reconstruction_seed": 42,
        },
        matched_trial_key=trial_key(),
    )


def operator(tmp_path, runner):
    template = tmp_path / "scene_template.json"
    template.write_text(json.dumps({"blend_path": "fixture.blend", "qc_views": [[0, 0]]}), encoding="utf-8")
    action_space = tmp_path / "scene_action_space.json"
    action_space.write_text(
        json.dumps({"groups": {}, "default_control_state": {}}),
        encoding="utf-8",
    )
    return ThreeDAssetEvolutionOperator(
        allowed_write_root=tmp_path / "campaign",
        stage_runner=runner,
        resource_config={
            "python_executable": "python",
            "blender_executable": "blender",
            "scene_template": str(template),
            "action_space": str(action_space),
            "device": "cuda:0",
            "render_resolution": 64,
            "representative_views": [list(item) for item in VIEWS],
        },
    )


def test_three_d_operator_materializes_and_verifies_complete_bundle(tmp_path):
    runner = FakeStageRunner()
    adapter = operator(tmp_path, runner)
    call = make_call(tmp_path)

    receipt = adapter.execute(call)
    verification = adapter.verify_effect(call, receipt)
    root = Path(call.work_dir)

    assert verification.status is EvidenceStatus.VERIFIED
    assert runner.calls == ["t2i", "segment", "reconstruct", "sanitize", "render", "review"]
    t2i_argv = adapter._invocations(call, adapter._paths(root, "mug", 0, VIEWS), adapter._strategy(call)[0])["t2i"].argv
    assert ("--height", "512") == t2i_argv[t2i_argv.index("--height"):t2i_argv.index("--height") + 2]
    assert ("--width", "512") == t2i_argv[t2i_argv.index("--width"):t2i_argv.index("--width") + 2]
    assert (root / "strategy.json").is_file()
    assert (root / "stage_receipts.jsonl").is_file()
    assert (root / "bundle_manifest.json").is_file()
    assert (root / "stage_result.json").is_file()
    assert len(list((root / "renders" / "mug").glob("az*_el+00.png"))) == 4
    assert receipt.stage_result["hard_constraints"]["representative_views_complete"] is True
    assert receipt.stage_result["metrics"]["hybrid_quality"] == 0.86


def test_three_d_operator_resumes_from_last_verified_stage_after_crash(tmp_path):
    runner = FakeStageRunner(fail_once_at="render")
    adapter = operator(tmp_path, runner)
    call = make_call(tmp_path)

    with pytest.raises(RuntimeError, match="injected crash"):
        adapter.execute(call)
    recovery = adapter.recover(call)
    assert recovery.status is RecoveryStatus.RETRY_SAFE

    receipt = adapter.execute(call)

    assert adapter.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert runner.calls.count("t2i") == 1
    assert runner.calls.count("segment") == 1
    assert runner.calls.count("reconstruct") == 1
    assert runner.calls.count("sanitize") == 1
    assert runner.calls.count("render") == 2
    assert runner.calls.count("review") == 1


def test_three_d_operator_rebuilds_missing_top_level_receipt_from_artifacts(tmp_path):
    adapter = operator(tmp_path, FakeStageRunner())
    call = make_call(tmp_path)
    receipt = adapter.execute(call)
    Path(receipt.receipt_path).unlink()

    recovered = adapter.recover(call)

    assert recovered.status is RecoveryStatus.RECOVERED
    assert recovered.receipt is not None
    assert recovered.receipt.reused is True
    assert adapter.verify_effect(call, recovered.receipt).status is EvidenceStatus.VERIFIED


def test_three_d_operator_fails_closed_on_corruption_unknown_parameter_or_path_escape(tmp_path):
    adapter = operator(tmp_path, FakeStageRunner())
    call = make_call(tmp_path)
    receipt = adapter.execute(call)
    (Path(call.work_dir) / "meshes" / "mug.glb").write_bytes(b"corrupt")

    assert adapter.verify_effect(call, receipt).status is EvidenceStatus.CONTRADICTED
    assert adapter.recover(call).status is RecoveryStatus.AMBIGUOUS

    with pytest.raises(ValueError, match="unknown OperatorCall parameters"):
        adapter.execute(make_call(tmp_path, call_id="unsafe", parameters={**call.parameters, "command": "echo unsafe"}))
    with pytest.raises(ValueError, match="outside allowed write root"):
        adapter.execute(make_call(tmp_path, call_id="escape", work_dir=tmp_path / "escape"))


def test_three_d_operator_resource_contract_changes_when_scene_template_drifts(tmp_path):
    adapter = operator(tmp_path, FakeStageRunner())
    before = adapter.spec.model_config_hash
    template = Path(adapter.resource_config["scene_template"])
    template.write_text(json.dumps({"blend_path": "changed.blend", "qc_views": [[0, 0]]}), encoding="utf-8")

    assert adapter.spec.model_config_hash != before
