"""Contract/recovery fixtures only; these tests do not measure visual benefits."""
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
from dataevolver.adapters.model_gateway import ModelError, ModelGateway
from dataevolver.adapters.visual_worker import VisualLocalProvider
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.dataset.g2_contracts import G2Task, compile_prompt, validate_g2_config, validate_recipe, validate_split_boundaries
from dataevolver.dataset.g2_pipeline import CONSTRAINTS, edit_request, load_pair_manifest
from dataevolver.harness.adapters.visual_g2 import VisualG2Operator
from dataevolver.harness.models import OperatorCall, StageStatus


@pytest.fixture
def setup(tmp_path):
    source, dataset, work = (tmp_path/x for x in ("source", "dataset", "runtime"))
    for p in (source, dataset, work):
        p.mkdir()
    image = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(image).rectangle((128, 128, 383, 383), fill="red")
    image.save(source/"input.png")
    task = G2Task("square", ArtifactRef("source", "input.png", file_sha256(source/"input.png"), "image/png"),
                  "square-family", "pilot", "Make only the red square blue. Preserve shape, other objects and background.",
                  "red square", "blue", (.25, .25, .75, .75), 17, {"kind": "test_fixture"})
    worker = {"kind": "qwen_edit_gpu", "model_path_env": "G2_TEST_MODEL", "python_env": "G2_TEST_PYTHON",
              "file_hashes": {"model_index.json": "1"*64}, "runtime_versions": {"python": "test"}, "threads": 1}
    review_worker = {**worker, "kind": "qwen_vl_cpu"}
    profiles = [ModelProfile(n, "local_python", n, "sha256:"+payload_sha256(worker["file_hashes"]), ("json", "vision")).to_dict()
                for n in ("edit", "review")]
    config = {"schema_version": "dataevolver.g2_config.v1", "tasks": [task.to_dict()],
              "allowed_recipe_values": {"square": {"target_description": ["the red square", "the square"],
                  "spatial_expression": ["in the center", ""], "preservation_instruction": ["Keep everything else unchanged."]}},
              "initial_recipe": {"square": {"target_description": "the red square", "spatial_expression": "",
                  "preservation_instruction": "Keep everything else unchanged."}},
              "inference": {"width": 512, "height": 512, "steps": 1, "true_cfg_scale": 4., "guidance_scale": 1.,
                            "negative_prompt": " ", "timeout_seconds": 10},
              "evaluation": {"max_outside_mae": .01, "min_target_mae": .01, "review_max_tokens": 200,
                             "review_timeout_seconds": 10, "calibration_status": "pilot_uncalibrated"},
              "budget": {"max_model_calls": 2, "max_edit_calls": 1, "max_probe_calls": 0,
                         "max_wall_seconds": 60, "max_output_bytes": 10000000},
              "models": {"profiles": profiles, "workers": {"edit": worker, "review": review_worker},
                         "policy": ModelPolicy({"visual.edit": "edit", "visual.inspect_region": "review", "visual.review_constraints": "review"}, 2).to_dict()},
              "stores": {"source": "G2_TEST_SOURCE", "dataset": "G2_TEST_DATASET"}, "dataset_store": "dataset",
              "method": {"id": "g2-fixed-m-v1", "mutable": False, "knowledge_snapshot": None},
              "split_policy": {"selection_splits": ["pilot", "experience_validation"], "benefit_split": "reuse_test", "freeze_k_during_test": True}}
    call = OperatorCall("campaign", "call", "call", "data.visual_g2", "1", 0, str(work/"execute"), config)
    return SimpleNamespace(source=source, dataset=dataset, work=work, task=task, config=config, call=call)


class FixtureProvider:
    def __init__(self, setup, kind, fail=False, quality_fail=False):
        self.s, self.kind, self.fail, self.quality_fail = setup, kind, fail, quality_fail
        self.effects, self.calls = {}, 0

    def health(self):
        return {"status": "TEST_FIXTURE"}

    def invoke(self, request, request_hash):
        self.calls += 1
        if self.fail:
            raise ModelError("LOCAL_WORKER_FAILED", "fixture process failed")
        if self.kind == "edit":
            image = Image.open(self.s.source/"input.png").convert("RGB")
            ImageDraw.Draw(image).rectangle((128, 128, 383, 383), fill="blue")
            path = self.s.dataset/f"{request_hash}.png"; image.save(path)
            output = {"artifact": ArtifactRef("dataset", path.name, file_sha256(path), "image/png").to_dict(),
                      "actual_parameters": json.loads(request.messages[0]["content"]), "image_size": [512, 512], "runtime": {"fixture": True}}
        else:
            output = {"checks": {k: {"status": "fail" if self.quality_fail and k == "background" else "pass",
                                      "evidence": "test fixture only"} for k in CONSTRAINTS}}
        response = {"model": self.kind, "output": output, "finish_reason": "stop",
                    "usage": {"images": 1, "cost": None}, "provider_request_id": request.request_id}
        self.effects[request_hash] = response
        return response

    def lookup(self, request, request_hash):
        if request_hash not in self.effects:
            raise ModelError("AMBIGUOUS_EFFECT", "fixture failed without committed effect")
        return self.effects[request_hash]


def operator(s, **kwargs):
    edit, review = FixtureProvider(s, "edit", fail=kwargs.get("edit_fail", False)), FixtureProvider(s, "review", fail=kwargs.get("review_fail", False), quality_fail=kwargs.get("quality_fail", False))
    op = VisualG2Operator(allowed_write_root=s.work, contract=s.config,
                         roots={"source": s.source, "dataset": s.dataset}, providers={"edit": lambda: edit, "review": lambda: review})
    return op, edit, review


def test_original_task_and_evaluation_survive_execution_recipe_changes(setup):
    s = setup
    recipe = {**s.config["initial_recipe"]["square"], "spatial_expression": "in the center"}
    prompt = compile_prompt(s.task, recipe, s.config["allowed_recipe_values"]["square"])
    assert "in the center" in prompt and s.task.original_instruction in prompt
    for field in ("color", "original_instruction", "threshold"):
        with pytest.raises(ValueError):
            validate_recipe({**recipe, field: "changed"}, s.config["allowed_recipe_values"]["square"])
    with pytest.raises(ValueError):
        G2Task.from_dict({**s.task.to_dict(), "constraints": ["target_color"]})


def test_split_access_rejects_same_family_or_same_source(setup):
    task = setup.task
    for family in (task.source_family, "different"):
        other = G2Task.from_dict({**task.to_dict(), "task_id": "other", "source_family": family, "split": "reuse_test"})
        with pytest.raises(ValueError):
            validate_split_boundaries((task, other))


def test_changed_parameters_cannot_reuse_stale_output(setup):
    s = setup; op, edit, _ = operator(s)
    gateway = op.gateway(s.call)
    recipe = s.config["initial_recipe"]["square"]
    first = edit_request(s.task, recipe, s.config, "fixed-id")
    gateway.invoke(first)
    changed = edit_request(s.task, {**recipe, "spatial_expression": "in the center"}, s.config, "fixed-id")
    with pytest.raises(ModelError, match="binding|bound|different"):
        gateway.invoke(changed)
    assert edit.calls == 1


def test_completed_batch_recovers_without_resubmission_and_loads_independently(setup):
    s = setup; op, edit, review = operator(s)
    receipt = op.execute(s.call)
    assert receipt.status is StageStatus.SUCCESS
    assert op.verify_effect(s.call, receipt).status.value == "VERIFIED"
    assert op.execute(s.call).reused
    assert (edit.calls, review.calls) == (1, 1)
    pairs = load_pair_manifest(Path(s.call.work_dir)/"pairs.json", op.store)
    assert len(pairs) == 1 and pairs[0]["task"]["original_instruction"] == s.task.original_instruction
    assert json.loads((Path(s.call.work_dir)/"pairs.json").read_text())["usage"]["model_calls"] == 2


@pytest.mark.parametrize("failure", ["edit_fail", "review_fail"])
def test_failure_is_recorded_and_charged_without_false_success(setup, failure):
    s = setup; op, edit, review = operator(s, **{failure: True})
    receipt = op.execute(s.call)
    assert receipt.status is StageStatus.FAILED
    value = json.loads((Path(s.call.work_dir)/"pairs.json").read_text())
    assert value["summary"]["tasks"] == 1
    assert value["summary"]["failed_tasks"] == 1
    assert value["summary"]["calls_per_qualified_pair"] is None
    assert value["usage"]["model_calls"] == edit.calls + review.calls
    assert value["usage"]["attempts"][-1]["error_code"] == "LOCAL_WORKER_FAILED"
    failed_request = value["pairs"][0]["attempts"][-1]
    assert failed_request["request"]["role"] == ("visual.edit" if failure == "edit_fail" else "visual.review_constraints")
    assert failed_request["task_hash"] == s.task.compute_hash()
    assert failed_request["candidate_eligible"] is (failure == "edit_fail")
    assert not load_pair_manifest(Path(s.call.work_dir)/"pairs.json", op.store)
    op.execute(s.call)
    assert edit.calls == 1


def test_failed_visual_constraint_is_excluded_from_training_export(setup):
    s = setup; op, _, _ = operator(s, quality_fail=True)
    op.execute(s.call)
    path = Path(s.call.work_dir)/"pairs.json"
    assert not load_pair_manifest(path, op.store)
    value = json.loads(path.read_text()); value["pairs"][0]["status"] = "pass"; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="falsely"):
        load_pair_manifest(path, op.store)


def test_corrupt_media_invalidates_effect(setup):
    s = setup; op, _, _ = operator(s); receipt = op.execute(s.call)
    next(s.dataset.glob("*.png")).write_bytes(b"corrupt")
    assert op.verify_effect(s.call, receipt).status.value == "CONTRADICTED"


def test_timeout_orphan_image_does_not_trigger_second_submission(setup, monkeypatch):
    import sys
    s = setup
    monkeypatch.setenv("G2_TEST_MODEL", str(s.source))
    monkeypatch.setenv("G2_TEST_PYTHON", sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test-gpu")
    store = ArtifactStore({"source": s.source, "dataset": s.dataset})
    profile = ModelProfile.from_dict(s.config["models"]["profiles"][0])
    provider = VisualLocalProvider(profile, s.config["models"]["workers"]["edit"], store, "dataset", s.work/"native")
    calls = []
    def timeout(*args, **kwargs):
        if args[0][0] == "nvidia-smi":
            return SimpleNamespace(stdout="test-gpu, 200000, 0\n")
        calls.append(args)
        payload = json.loads(Path(args[0][-1]).read_text())
        orphan = Path(payload["image_output"]); orphan.parent.mkdir(parents=True); orphan.write_bytes(b"orphan")
        raise subprocess.TimeoutExpired(args[0], 10)
    monkeypatch.setattr(subprocess, "run", timeout)
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({"visual.edit": "edit"}, 1),
                           receipt_root=s.work/"gateway", artifact_store=store, providers={"edit": lambda: provider})
    request = edit_request(s.task, s.config["initial_recipe"]["square"], s.config, "timeout-id")
    for _ in range(2):
        with pytest.raises(ModelError) as error:
            gateway.invoke(request)
        assert error.value.code == "AMBIGUOUS_EFFECT"
    assert len(calls) == 1 and gateway.audit_usage()["model_calls"] == 1
    assert '"event":"timeout"' in (s.work/"native/timeout-id.log").read_text()


def test_nonzero_exit_with_orphan_file_is_never_a_success(setup, monkeypatch):
    import sys
    s = setup
    monkeypatch.setenv("G2_TEST_MODEL", str(s.source)); monkeypatch.setenv("G2_TEST_PYTHON", sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test-gpu")
    profile = ModelProfile.from_dict(s.config["models"]["profiles"][0])
    provider = VisualLocalProvider(profile, s.config["models"]["workers"]["edit"], ArtifactStore({"source": s.source, "dataset": s.dataset}), "dataset", s.work/"native")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="test-gpu, 200000, 0\n"))
    request = edit_request(s.task, s.config["initial_recipe"]["square"], s.config, "failed-id")
    with pytest.raises(ModelError) as error:
        provider.invoke(request, "2"*64)
    assert error.value.code == "LOCAL_WORKER_FAILED"
    assert not provider._path(request).exists()


def test_task_view_is_built_from_records_including_failed_attempts(setup):
    from dataevolver.harness.g2_view import build_draft
    s = setup; op, _, _ = operator(s, review_fail=True); op.execute(s.call)
    output = s.work/"view.md"
    result = build_draft(Path(s.call.work_dir)/"pairs.json", {"source": s.source, "dataset": s.dataset}, output, s.work/"media")
    draft = output.read_text()
    assert result["panels"] == 2 and "LOCAL_WORKER_FAILED" in draft
    assert "failed" in draft and "模型调用（含失败） | 2" in draft
    assert len(list((s.work/"media").glob("*.png"))) == 2


def test_step1_guard_keeps_p2_gpu_authority_closed(setup):
    from dataevolver.harness.cli import _registry, _validate_creation_mode
    from dataevolver.harness.models import GoalSpec
    s = setup
    payload = {"schema_version": "dataevolver.harness.goal.v1", "campaign_id": "test", "objective": {"real_data_execution": True},
               "initial_operator_id": "data.visual_g2", "budgets": {"max_rounds": 1, "max_operator_calls": 1, "max_experiments": 0,
                    "max_retries": 0, "max_wall_seconds": 60}, "resource_policy": {"allow_gpu": True, "allow_network": False},
               "frozen_evaluator": {"path": "unused", "sha256": "1"*64}, "initial_operator_parameters": s.config}
    goal = GoalSpec.from_dict(payload)
    _validate_creation_mode(goal, dry_run=False)
    with pytest.raises(ValueError):
        _validate_creation_mode(goal, dry_run=True)
    p2 = GoalSpec.from_dict({**payload, "initial_operator_id": "data.image_curation"})
    with pytest.raises(ValueError, match="offline CPU"):
        _registry(p2, s.work)


def test_existing_controller_owns_g2_execution_and_replay(setup):
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.models import GoalSpec
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.planner import DeterministicPlanner
    from dataevolver.paths import REPO_ROOT
    s = setup; op, edit, review = operator(s)
    evaluator_path = s.work/"evaluator.json"
    evaluator_path.write_text(json.dumps({"schema_version": "dataevolver.harness.evaluator.v1", "evaluator_id": "test", "revision": "1",
        "hard_constraints": {"artifact_valid": {"required": True}, "all_tasks_recorded": {"required": True}},
        "target_metrics": {"completion_ratio": {"required": True, "minimum": 1.}}, "protected_metrics": {}}))
    goal = GoalSpec.from_dict({"schema_version": "dataevolver.harness.goal.v1", "campaign_id": "campaign", "objective": {"real_data_execution": True,
        "initial_trial_key": {"input_artifact_hash": s.task.source.sha256, "parent_artifact_hash": "", "seed": 17,
            "operator_id": "data.visual_g2", "operator_version": "1", "model_config_hash": op.spec.model_config_hash,
            "preprocessing_hash": "none", "evaluator_hash": file_sha256(evaluator_path), "environment_contract_hash": "fixture"}},
        "initial_operator_id": "data.visual_g2", "budgets": {"max_rounds": 1, "max_operator_calls": 1, "max_experiments": 0,
        "max_retries": 0, "max_wall_seconds": 60}, "resource_policy": {"allowed_operators": ["data.visual_g2"]},
        "frozen_evaluator": {"path": str(evaluator_path), "sha256": file_sha256(evaluator_path)}, "initial_operator_parameters": s.config})
    registry = OperatorRegistry(); registry.register(op)
    controller = EvolutionController.create(repo_root=REPO_ROOT, campaigns_root=s.work, goal=goal, registry=registry,
        planner=DeterministicPlanner(()), evaluator_contract=EvaluatorContract.from_path(evaluator_path),
        schema_paths=(), execution_paths=(Path(__file__),), idempotency_key="fixture")
    controller.run()
    assert controller.status().state.value == "COMPLETE"
    assert controller.replay().to_dict() == controller.status().to_dict()
    assert (edit.calls, review.calls) == (1, 1)


def test_portable_bundle_keeps_task_hash_and_cannot_hide_corruption(setup):
    from dataevolver.dataset.g2_export import export_pairs, load_bundle
    s = setup; op, _, _ = operator(s); op.execute(s.call)
    manifest = Path(s.call.work_dir)/"pairs.json"
    bundle = s.work/"portable"
    export_pairs(manifest, op.store, bundle)
    assert (bundle/"pairs.json").read_bytes() == manifest.read_bytes()
    assert load_bundle(bundle)[0]["task_hash"] == s.task.compute_hash()
    next((bundle/"stores/dataset").glob("*.png")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(bundle)


def test_completed_runner_reuses_verified_export_without_gpu_or_resubmission(setup, monkeypatch):
    from dataevolver.harness import g2_run
    s = setup
    s.call = s.call.with_updates(work_dir=str(s.work/"campaign/rounds/r000/execute"))
    op, edit, review = operator(s); op.execute(s.call)
    config_dir = s.work/"config"; config_dir.mkdir()
    (config_dir/"config.json").write_text(json.dumps(s.config))
    (config_dir/"goal.json").write_text(json.dumps({"campaign_id": "campaign"}))
    (config_dir/"frozen.json").write_text(json.dumps({"config_sha256": file_sha256(config_dir/"config.json"),
        "goal_sha256": file_sha256(config_dir/"goal.json")}))
    (s.work/"campaign/final_report.json").write_text("{}")
    monkeypatch.setenv("G2_TEST_SOURCE", str(s.source)); monkeypatch.setenv("G2_TEST_DATASET", str(s.dataset))
    monkeypatch.setattr(g2_run, "validate_campaign", lambda *a: {"step1_passed": True})
    def forbidden():
        raise AssertionError("a completed campaign must not wait for or allocate a GPU")
    monkeypatch.setattr(g2_run, "gpu_snapshot", forbidden)
    args = SimpleNamespace(config_dir=str(config_dir), campaign_root=str(s.work),
            export_dir=str(s.work/"portable"), status=str(s.work/"status.json"), validation=str(s.work/"validation.json"))
    assert g2_run.run(args) == 0
    assert g2_run.run(args) == 0
    assert json.loads((s.work/"validation.json").read_text())["portable_export"]["reused"]
    assert (edit.calls, review.calls) == (1, 1)
    (s.work/"portable/pairs.json").write_text("{}")
    assert g2_run.run(args) == 1
    assert (edit.calls, review.calls) == (1, 1)
