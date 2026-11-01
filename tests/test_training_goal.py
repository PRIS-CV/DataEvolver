"""Contract fixtures are engineering evidence, never real planning/quality claims."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from dataevolver.workflows.training_goal.catalog import experience_snapshot, operator_catalog
from dataevolver.workflows.training_goal.contracts import validate_plan
from dataevolver.workflows.training_goal.planner import render_files
from dataevolver.workflows.training_goal.tools import ReadOnlyTools, public_url


@pytest.fixture
def fixture_plan():
    """Minimal, explicitly synthetic planning contract for semantic regressions."""
    request = {"request": "fixture goal", "constraints": [{"id": "c1", "text": "preserve originals"}],
               "materials": [], "task_config": {"models": {"image": "requested-model"}}}
    sources = [{"evidence_id": f"s{i}", "url": f"https://example.org/{i}",
                "resolved_url": f"https://example.org/{i}", "version": f"rev{i}", "ok": True} for i in range(3)]
    node = {"id": "n1", "operator": "background.complete", "status": "needs_implementation",
            "depends_on": [], "input_bindings": [{"port": "request", "source": "goal:request"}],
            "outputs": [{"name": "record", "kind": "training_record", "description": "fixture"}],
            "parameters_json": "{}", "evidence_ids": ["s0"], "experience_ids": [], "rationale": "fixture"}
    example = {"is_placeholder": True, "family_id": "placeholder:family",
               "input": {"background": "placeholder:bg", "instruction": "insert",
                         "subjects": [{"id": "subject1", "reference": "placeholder:ref"}]},
               "target": {"image": "placeholder:target"},
               "annotations": {"target_instances": [{"subject_id": "subject1", "target_instance_id": "instance1"}]}}
    plan = {"goal": {"task_type": "fixture_task", "capability": "fixture",
                    "training_inputs": [{"name": "subjects", "kind": "list", "description": "fixture"}],
                    "training_targets": [{"name": "image", "kind": "image", "description": "fixture"}],
                    "supervision_kind": "paired", "construction_mode": "fixture",
                    "constraints": request["constraints"], "assumptions": [], "material_usage": []},
            "research": {"sources": [{**{k: v for k, v in e.items() if k in ("evidence_id", "url", "version")},
                                       "title": "fixture", "finding": "fixture", "design_impact": "fixture",
                                       "node_ids": ["n1"]} for e in sources],
                         "operator_candidates": [{"operation": "background.complete", "repository": "fixture",
                                                   "path": "fixture.py", "revision": "rev0",
                                                   "input_adaptation": "fixture", "output_adaptation": "fixture",
                                                   "verification_status": "candidate", "evidence_id": "s0",
                                                   "node_ids": ["n1"]}], "revisions": []},
            "dataset_plan": {"nodes": [node], "model_roles": [{"role": "image", "model": "requested-model",
                                                            "deployment_status": "unverified"}],
                             "checks": [], "coverage_json": "{}", "split_strategy": "family",
                             "sample_example_json": json.dumps(example)},
            "todo": [{"id": "t1", "title": "fixture", "depends_on": [], "node_ids": ["n1"],
                      "deliverable": "fixture", "acceptance": "fixture", "status": "todo"}],
            "experiences_used": []}
    catalog = {"operators": [{"operator": "background.complete", "status": "needs_implementation"}]}
    snapshot = {"records": []}
    return plan, request, catalog, snapshot, sources


def test_valid_and_four_exports(fixture_plan, tmp_path):
    assert validate_plan(*fixture_plan)["valid"]
    before = copy.deepcopy(fixture_plan[0])
    render_files(fixture_plan[0], fixture_plan[1], tmp_path)
    assert fixture_plan[0] == before
    assert {p.name for p in tmp_path.iterdir()} == {"goal.json", "research.md", "dataset_plan.yaml", "todo.md"}
    assert json.loads((tmp_path / "dataset_plan.yaml").read_text())["sample_example"]["is_placeholder"]
    assert json.loads((tmp_path / "goal.json").read_text())["original_request"] == "fixture goal"


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["goal"].update(constraints=[]), "constraints"),
    (lambda p: p["dataset_plan"]["nodes"][0].update(status="implemented"), "executable"),
    (lambda p: p["dataset_plan"]["nodes"][0].update(depends_on=["n1"]), "cycle"),
    (lambda p: p["dataset_plan"]["nodes"][0].update(input_bindings=[{"port": "x", "source": "node:missing.output"}]), "upstream"),
    (lambda p: p["dataset_plan"]["nodes"][0].update(input_bindings=[{"port": "x", "source": "config:missing"}]), "missing goal/config"),
    (lambda p: p["research"]["sources"][0].update(version="invented"), "version"),
    (lambda p: p["research"]["sources"][0].update(url="https://example.org/invented"), "fetched"),
    (lambda p: p["research"]["operator_candidates"][0].update(verification_status="real_execution_verified"), "candidate"),
    (lambda p: p.update(todo=[]), "TODO"),
    (lambda p: p["dataset_plan"]["model_roles"][0].update(model="substitute"), "model"),
    (lambda p: p["dataset_plan"]["nodes"][0].update(experience_ids=["invented"]), "experience"),
])
def test_semantic_failures(fixture_plan, mutation, match):
    values = copy.deepcopy(fixture_plan)
    mutation(values[0])
    with pytest.raises(ValueError, match=match):
        validate_plan(*values)


def test_target_side_leak_and_identity(fixture_plan):
    example = json.loads(fixture_plan[0]["dataset_plan"]["sample_example_json"])
    example["input"]["observed_events"] = ["target-derived"]
    fixture_plan[0]["dataset_plan"]["sample_example_json"] = json.dumps(example)
    with pytest.raises(ValueError, match="leaked"):
        validate_plan(*fixture_plan)
    example["input"].pop("observed_events")
    example["annotations"]["target_instances"] = []
    fixture_plan[0]["dataset_plan"]["sample_example_json"] = json.dumps(example)
    with pytest.raises(ValueError, match="bind target"):
        validate_plan(*fixture_plan)
    example["input"]["nested"] = {"target_masks": ["placeholder:mask"]}
    fixture_plan[0]["dataset_plan"]["sample_example_json"] = json.dumps(example)
    with pytest.raises(ValueError, match="leaked"):
        validate_plan(*fixture_plan)


def test_video_action_and_observation(fixture_plan):
    example = {"is_placeholder": True, "family_id": "placeholder:family",
               "input": {"initial_frame": "placeholder:image"},
               "target": {"video": "placeholder:video"},
               "annotations": {"expected_events": [], "observed_events": [], "evidence": []}}
    fixture_plan[0]["dataset_plan"]["sample_example_json"] = json.dumps(example)
    with pytest.raises(ValueError, match="action"):
        validate_plan(*fixture_plan)
    example["input"]["action_condition"] = "push"
    fixture_plan[0]["dataset_plan"]["sample_example_json"] = json.dumps(example)
    assert validate_plan(*fixture_plan)["valid"]
    example["annotations"].pop("evidence")
    example["annotations"]["observed_events"] = [{"status": "uncertain", "evidence": ["placeholder:frame"]}]
    fixture_plan[0]["dataset_plan"]["sample_example_json"] = json.dumps(example)
    assert validate_plan(*fixture_plan)["valid"]
    del example["annotations"]["observed_events"]
    fixture_plan[0]["dataset_plan"]["sample_example_json"] = json.dumps(example)
    with pytest.raises(ValueError, match="expected/observed"):
        validate_plan(*fixture_plan)


def test_material_must_enter_graph(fixture_plan):
    fixture_plan[1]["materials"] = [{"id": "ref", "origin": "user_declared"}]
    fixture_plan[0]["goal"]["material_usage"] = [{"material_id": "ref", "role": "reference", "preservation": "retain"}]
    with pytest.raises(ValueError, match="actually enter"):
        validate_plan(*fixture_plan)
    fixture_plan[0]["dataset_plan"]["nodes"][0]["input_bindings"].append({"port": "reference", "source": "material:ref"})
    assert validate_plan(*fixture_plan)["valid"]


def test_experience_changes_actual_parameters(fixture_plan):
    fixture_plan[3]["records"] = [{"id": "e1", "version": "1",
                                 "scope": {"operator": "background.complete",
                                           "changes": [{"field": "parameters.shadow", "after_json": "true"}]}}]
    fixture_plan[0]["experiences_used"] = [{"id": "e1", "version": "1", "node_id": "n1", "why": "fixture",
                                          "changes": [{"field": "parameters.shadow", "before_json": "false", "after_json": "true"}]}]
    fixture_plan[0]["dataset_plan"]["nodes"][0]["experience_ids"] = ["e1"]
    with pytest.raises(ValueError, match="actual plan"):
        validate_plan(*fixture_plan)
    fixture_plan[0]["dataset_plan"]["nodes"][0]["parameters_json"] = '{"shadow":true}'
    assert validate_plan(*fixture_plan)["valid"]
    fixture_plan[3]["records"] = []
    with pytest.raises(ValueError, match="unavailable"):
        validate_plan(*fixture_plan)


def test_catalog_reads_registered_source():
    package = Path(__file__).resolve().parents[1] / "src/dataevolver"
    catalog = operator_catalog(package)
    by_id = {r["operator"]: r for r in catalog["operators"]}
    assert by_id["images.audit"]["status"] == "implemented"
    assert by_id["subjects.segment"]["status"] == "needs_implementation"
    assert "single" in by_id["subjects.segment"]["adaptation"] or "one best" in by_id["subjects.segment"]["adaptation"]
    assert experience_snapshot(None, "task", {}, catalog)["records"] == []


@pytest.mark.parametrize("url", ["http://example.org", "https://localhost/x", "https://127.0.0.1/x",
                                  "https://user:secret@example.org", "https://example.org:444/x"])
def test_source_permissions(url):
    with pytest.raises(ValueError):
        public_url(url)


def test_fetch_failure_is_recorded(tmp_path):
    (tmp_path / "catalog.json").write_text('{"operators":[]}')
    (tmp_path / "experience-snapshot.json").write_text('{"records":[]}')
    tools = ReadOnlyTools(tmp_path)
    with pytest.raises(ValueError):
        tools.call("read_source", {"url": "http://localhost"})
    trace = json.loads((tmp_path / "tool-trace.jsonl").read_text())
    assert trace["ok"] is False and trace["tool"] == "read_source"


def test_code_candidate_cannot_use_mutable_page_as_commit(fixture_plan):
    fixture_plan[4][0]["resolved_url"] = "https://github.com/fixture/repo/blob/main/fixture.py"
    with pytest.raises(ValueError, match="actually read commit"):
        validate_plan(*fixture_plan)


def test_material_variation_changes_supervision_not_only_title():
    from dataevolver.workflows.training_goal.acceptance import check_material_variation
    baseline = {"goal": {"construction_mode": "scene_decomposition"}, "todo": ["extract"],
                "dataset_plan": {"nodes": [{"id": "extract", "operator": "subjects.extract"}]}}
    changed = {"goal": {"construction_mode": "independent_reference_composition"}, "todo": ["compose"],
               "dataset_plan": {"nodes": [
                   {"id": "prepare", "operator": "materials.prepare", "depends_on": [],
                    "input_bindings": [{"source": "material:bg"}, {"source": "material:ref"}]},
                   {"id": "compose", "operator": "image.compose", "depends_on": ["prepare"], "input_bindings": []}]}}
    request = {"materials": [{"id": "bg"}, {"id": "ref"}]}
    assert check_material_variation(baseline, changed, request)["composition_materials"] == ["bg", "ref"]
    changed["dataset_plan"]["nodes"][0]["input_bindings"].pop()
    with pytest.raises(ValueError, match="all supplied"):
        check_material_variation(baseline, changed, request)
    changed["dataset_plan"]["nodes"][0]["input_bindings"].append({"source": "material:ref"})
    changed["dataset_plan"]["nodes"][0]["operator"] = "subjects.extract"
    with pytest.raises(ValueError, match="target-derived"):
        check_material_variation(baseline, changed, request)


def test_old_stage0_commands_and_new_entry():
    from dataevolver.tools.stage0_web_research import build_parser
    parser = build_parser()
    assert parser.parse_args(["status", "--session-dir", "fixture"]).command == "status"
    parsed = parser.parse_args(["plan-goal", "--goal", "fixture", "--output-dir", "fixture", "--planner-instance", "fixture"])
    assert parsed.goal == "fixture"


@pytest.mark.parametrize("change,reason", [
    ({"status": "CANDIDATE"}, "not_promoted"),
    ({"status": "RETIRED"}, "not_promoted"),
    ({"validation_evidence": []}, "missing_validation_evidence"),
    ({"scope": {"operator_version": "2"}}, "incompatible_operator"),
    ({"scope": {"models": {"image": "other-revision"}}}, "incompatible_or_unspecified_models"),
    ({"scope": {"changes": [{"field": "constraints.ignore", "after_json": "true"}]}}, "unsafe_or_missing_changes"),
    ({}, None),
])
def test_readonly_experience_filter(tmp_path, change, reason):
    """Synthetic capabilities test compatibility; they are not validated methods."""
    scope = {"task": "fixture", "models": {"image": "fixed-revision"},
             "model_versions": {"image": "fixture-revision"},
             "operator": "background.complete", "operator_version": "1",
             "changes": [{"field": "parameters.shadow", "after_json": "true"}]}
    change = copy.deepcopy(change)
    scope.update(change.pop("scope", {}))
    row = {"capability_id": "fixture", "version": "1", "status": "PROMOTED",
           "failure_signature": {"code": "fixture"}, "validation_evidence": ["fixture:evidence"],
           "intervention": {"intervention_id": "fixture", "failure_code": "fixture",
                            "operator_id": "background.complete", "level": "L1",
                            "parameters": {"planning_scope": scope}}, **change}
    path = tmp_path / "capabilities.jsonl"
    path.write_text(json.dumps(row) + "\n")
    before = path.read_bytes()
    result = experience_snapshot(path, "fixture", {"image": "fixed-revision"},
                                 {"operators": [{"operator": "background.complete"}]},
                                 model_versions={"image": "fixture-revision"})
    assert path.read_bytes() == before
    assert result["read_only"] is True
    if reason:
        assert not result["records"] and result["excluded"][0]["reason"] == reason
    else:
        assert result["records"][0]["id"] == "fixture"
        conflicting = experience_snapshot(path, "fixture", {"image": "fixed-revision"},
                    {"operators": [{"operator": "background.complete"}]},
                    model_versions={"image": "fixture-revision"}, constraints=[{"id": "preserve", "text": "preserve"}])
        assert conflicting["excluded"][0]["reason"] == "constraint_compatibility_unknown_or_conflict"
        unfrozen = experience_snapshot(path, "fixture", {"image": "fixed-revision"},
                    {"operators": [{"operator": "background.complete"}]})
        assert unfrozen["excluded"][0]["reason"] == "model_versions_not_frozen_or_incompatible"


def test_legacy_stage0_research_lifecycle(tmp_path, capsys):
    from dataevolver.tools.stage0_web_research import main
    session = str(tmp_path / "legacy")
    assert main(["init", "--query", "fixture research", "--output-dir", session]) == 0
    assert main(["add-evidence", "--session-dir", session, "--url", "https://example.org/fixture",
                 "--title", "fixture source", "--summary", "synthetic regression evidence",
                 "--dataset-workflow", "fixture generation", "--quality-gate", "fixture check"]) == 0
    assert main(["finalize", "--session-dir", session, "--summary", "fixture research only"]) == 0
    assert main(["validate", "--prior-path", str(Path(session) / "research_prior.json")]) == 0
    payload = json.loads((Path(session) / "research_prior.json").read_text())
    assert "example.org" in json.dumps(payload)


def test_bridge_starts_without_pythonpath(tmp_path):
    import os
    import subprocess
    from dataevolver.workflows.training_goal import tools
    (tmp_path / "catalog.json").write_text('{"operators":[]}')
    (tmp_path / "experience-snapshot.json").write_text('{"records":[]}')
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    import sys
    result = subprocess.run([sys.executable, tools.__file__, "--run-dir", str(tmp_path)],
                            input=json.dumps(message) + "\n", text=True, capture_output=True, env=env)
    assert result.returncode == 0
    descriptors = json.loads(result.stdout)["result"]["tools"]
    assert {t["name"] for t in descriptors} == {"read_source", "query_operator_catalog", "query_experiences"}
    assert all(t["annotations"]["readOnlyHint"] and not t["annotations"]["destructiveHint"] for t in descriptors)


@pytest.mark.parametrize("failure", ["login", "malformed", "timeout", "cancel"])
def test_client_failures_retain_evidence(tmp_path, monkeypatch, failure):
    import subprocess
    from dataevolver.workflows.training_goal import planner
    instance = tmp_path / "instance"
    instance.mkdir(); (instance / "config.toml").write_text("")
    output = tmp_path / "output"

    class FixtureProcess:
        pid = 2147483647
        returncode = 1 if failure == "login" else (None if failure in {"timeout", "cancel"} else 0)
        stdin = open(tmp_path / "fixture-stdin", "w")

        def __init__(self, command, **kw):
            kw["stdout"].write(json.dumps({"type": "thread.started", "thread_id": "fixture"}) + "\n")
            kw["stdout"].flush()
            kw["stderr"].write("fixture authentication failure\n" if failure == "login" else "")
            (output / "final-response.json").write_text("invalid fixture JSON")

        def poll(self):
            if failure == "cancel" and not getattr(self, "interrupted", False):
                self.interrupted = True
                raise KeyboardInterrupt()
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", FixtureProcess)
    # Preflight is a distinct protocol-only check, never a model invocation.
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {
        "returncode": 0, "stderr": "", "stdout": '{"result":{}}\n{"result":{"tools":[]}}\n'})())
    expected = KeyboardInterrupt if failure == "cancel" else (TimeoutError if failure == "timeout" else (json.JSONDecodeError if failure == "malformed" else RuntimeError))
    with pytest.raises(expected):
        planner.plan_goal({"request": "fixture failure"}, output, instance, timeout=-1)
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == ("CANCELLED" if failure == "cancel" else "FAILED")
    assert status["monetary_cost"] is None
    assert (output / "attempt-history.jsonl").exists() and (output / "invocation.json").exists()
    assert not (output / "goal.json").exists()


def test_complete_reuse_does_not_invoke_client(fixture_plan, tmp_path, monkeypatch):
    from dataevolver.workflows.training_goal import planner
    import subprocess
    # Build the normal binding without submitting a client: protocol preflight fails.
    instance = tmp_path / "instance"
    instance.mkdir(); (instance / "config.toml").write_text("")
    output = tmp_path / "output"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {
        "returncode": 1, "stderr": "fixture", "stdout": ""})())
    with pytest.raises(RuntimeError, match="preflight"):
        planner.plan_goal(fixture_plan[1], output, instance)
    request, catalog = fixture_plan[1], json.loads((output / "catalog.json").read_text())
    snapshot = json.loads((output / "experience-snapshot.json").read_text())
    binding = hashlib.sha256(json.dumps({"request": request, "catalog": catalog, "experience": snapshot,
        "model": "gpt-6.1-sol", "instructions": planner.INSTRUCTIONS, "schema": planner.PLAN_SCHEMA},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    (output / "plan.json").write_text(json.dumps(fixture_plan[0]))
    (output / "source-evidence.jsonl").write_text("\n".join(json.dumps(e) for e in fixture_plan[4]))
    planner.render_files(fixture_plan[0], request, output)
    files = {n: hashlib.sha256((output / n).read_bytes()).hexdigest() for n in
             ("goal.json", "research.md", "dataset_plan.yaml", "todo.md", "plan.json")}
    planner.write_json(output / "status.json", {"state": "COMPLETE", "binding": binding, "files": files})
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate model invocation"))
    assert planner.plan_goal(request, output, instance)["new_model_invocations"] == 0
    with pytest.raises(ValueError, match="revision needs"):
        planner.plan_goal(request, output, instance, revise=True)
    # Local validation repair reuses the same retained response and tool evidence.
    planner.write_json(output / "status.json", {"state": "FAILED", "binding": binding, "error": "fixture local validation"})
    (output / "final-response.json").write_text(json.dumps(fixture_plan[0]))
    (output / "codex-events.jsonl").write_text(json.dumps({"type": "item.completed", "item": {"type": "web_search"}}) + "\n")
    (output / "tool-trace.jsonl").write_text("\n".join(json.dumps({"tool": t, "ok": True}) for t in
        ("read_source", "query_operator_catalog", "query_experiences")))
    result = planner.plan_goal(request, output, instance, revalidate=True)
    assert result["state"] == "COMPLETE" and result["new_model_invocations"] == 0
    (output / "goal.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        planner.plan_goal(request, output, instance)
