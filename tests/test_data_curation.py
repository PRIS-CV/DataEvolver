import copy
import io
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from dataevolver.dataset.contracts import ArtifactRef, file_sha256
from dataevolver.dataset.curation_loader import load_export
from dataevolver.harness.adapters import data_graph
from dataevolver.harness.adapters.data_graph import DataGraphOperator
from dataevolver.harness.data_budget import DataBudgetExceeded, DataBudgetJournal
from dataevolver.harness.data_contracts import DataInput, DataNodeGraph, DataNodeSpec, DataRunBudget
from dataevolver.harness.models import EvidenceStatus, OperatorCall, RecoveryStatus, StageStatus
from dataevolver.harness.task_contracts import DatasetIntent, PROTECTED_GATES, QualityPolicy


def fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "license.txt").write_text("MIT fixture input. Synthetic images only.")
    evidence = ArtifactRef("raw", "license.txt", file_sha256(raw / "license.txt"), "text/plain")
    sources, first = [], None
    for source in ("a", "b"):
        values = []
        for i in range(24):
            image = Image.frombytes("RGB", (64, 64), random.Random(f"{source}-{i}").randbytes(64 * 64 * 3))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            data = stream.getvalue()
            if source == "a" and i == 0:
                first = data
            if source == "b" and i == 0:
                data = first  # Cross-source duplicate with different provenance.
            values.append({"image": {"bytes": data, "path": f"{source}-{i}.png"}, "label": i % 3,
                           "family": f"leaf-{i // 2}"})
        path = raw / f"{source}.parquet"
        pq.write_table(pa.Table.from_pylist(values), path)
        sources.append({"source_id": source, "dataset": f"fixture/{source}", "revision": "fixture-v1",
                        "source_split": "train", "file": ArtifactRef("raw", path.name, file_sha256(path)).to_dict(),
                        "license": "mit", "license_evidence": [evidence.to_dict()],
                        "image_column": "image", "label_column": "label", "family_column": "family"})
    catalog = raw / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": "dataevolver.image_sources.v1", "sources": sources}))
    source_ref = ArtifactRef("raw", "catalog.json", file_sha256(catalog))
    nodes = []
    params = [("audit", {"max_rows_per_source": 24, "min_side": 16, "max_pixels": 100000}),
              ("families", {"hamming_distance": 0}), ("split", {"ratios": {"train": .6, "dev": .2, "eval": .2}, "seed": 3}),
              ("select", {"per_source": 3, "seed": 4}), ("clean", {"max_side": 64}),
              ("export", {"require_semantic_match": False})]
    prior = None
    for name, parameters in params:
        inputs = {"sources": DataInput(artifact=source_ref)} if prior is None else {"catalog": DataInput(node=prior, output="catalog")}
        nodes.append(DataNodeSpec(name, f"images.{name}", "1", "data", inputs, parameters))
        prior = name
    contract = {"intent": DatasetIntent("fixture", "curate", "image_curation", "curated_images.v1", "Synthetic fixture only", (source_ref,)).to_dict(),
                "quality_policy": QualityPolicy("fixture", tuple(sorted(PROTECTED_GATES)), tuple(n.operator for n in nodes),
                                                {"allowed_licenses": ["mit"]}, 1).to_dict(),
                "graph": DataNodeGraph(tuple(nodes)).to_dict(),
                "budget": DataRunBudget(8, 0, 0, 48, 10000000, 120).to_dict(),
                "stores": {"raw": "FIXTURE_RAW", "dataset": "FIXTURE_DATASET"}, "dataset_store": "dataset", "models": {}}
    return contract, {"raw": raw, "dataset": tmp_path / "dataset"}


def operator_call(tmp_path, contract, roots):
    operator = DataGraphOperator(allowed_write_root=tmp_path / "campaign", contract=contract, roots=roots)
    call = OperatorCall("fixture", "graph-call", "fixture-key", "data.image_curation", "2", 0,
                        str(tmp_path / "campaign" / "execute"), parameters=contract)
    return operator, call


def exported_index(call):
    run = json.loads((Path(call.work_dir) / "node_runs/export/node_run.json").read_text())
    return ArtifactRef.from_dict(run["outputs"]["dataset"])


def test_pure_data_graph_uses_no_model_budget_and_exports_standalone(tmp_path):
    contract, roots = fixture(tmp_path)
    before = {p: file_sha256(p) for p in roots["raw"].iterdir()}
    operator, call = operator_call(tmp_path, contract, roots)
    receipt = operator.execute(call)
    assert operator.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    nodes = [json.loads(p.read_text()) for p in Path(call.work_dir).glob("node_runs/*/node_run.json")]
    assert len(nodes) == 6 and all(not n["model_receipt_hashes"] for n in nodes)
    assert not list(Path(call.work_dir).glob("model_receipts/*.json"))
    index = operator.store.resolve(exported_index(call))
    rows = load_export(index.parent)
    assert len(rows) == 6 and {r["source_id"] for r in rows} == {"a", "b"}
    assert len({r["family_id"] for r in rows}) == 6
    assert all(r["protected_gates"]["privacy"] is None for r in rows)
    assert before == {p: file_sha256(p) for p in roots["raw"].iterdir()}
    journal = json.loads((Path(call.work_dir) / "data_budget.json").read_text())
    assert sum(journal["attempts"].values()) == 6 and sum(journal["rows"].values()) == 48
    assert operator.execute(call).reused
    assert json.loads((Path(call.work_dir) / "data_budget.json").read_text()) == journal


def test_resume_after_step_effect_commit_does_not_repeat_or_reset_budget(tmp_path, monkeypatch):
    contract, roots = fixture(tmp_path)
    operator, call = operator_call(tmp_path, contract, roots)
    original = data_graph._write_receipt
    crashed = []

    def crash_once(path, receipt):
        if receipt.operator_id == "images.clean" and not crashed:
            crashed.append(True)
            raise RuntimeError("interrupted after clean effect")
        return original(path, receipt)

    monkeypatch.setattr(data_graph, "_write_receipt", crash_once)
    with pytest.raises(RuntimeError, match="interrupted"):
        operator.execute(call)
    reopened, _ = operator_call(tmp_path, contract, roots)
    assert reopened.recover(call).status is RecoveryStatus.RETRY_SAFE
    reopened.execute(call)
    journal = json.loads((Path(call.work_dir) / "data_budget.json").read_text())
    assert sum(journal["attempts"].values()) == 6 and all(n == 1 for n in journal["attempts"].values())
    assert len(load_export(reopened.store.resolve(exported_index(call)).parent)) == 6


@pytest.mark.parametrize("limit,match", [("max_input_rows", "rows"), ("max_output_bytes", "files")])
def test_failed_work_consumes_durable_budget(tmp_path, limit, match):
    contract, roots = fixture(tmp_path)
    contract["budget"][limit] = 1
    operator, call = operator_call(tmp_path, contract, roots)
    receipt = operator.execute(call)
    assert receipt.status is StageStatus.FAILED
    assert receipt.stage_result["failure_code"] == "DATA_BUDGET_EXHAUSTED"
    journal = json.loads((Path(call.work_dir) / "data_budget.json").read_text())
    assert journal["attempts"]["audit"] == 1
    assert not list(Path(call.work_dir).glob("node_runs/*/node_run.json"))


def test_missing_budget_and_corrupt_assets_cannot_be_accepted(tmp_path):
    contract, roots = fixture(tmp_path)
    operator, call = operator_call(tmp_path, contract, roots)
    operator.execute(call)
    index = operator.store.resolve(exported_index(call))
    first = load_export(index.parent)[0]
    (index.parent / first["image"]).write_bytes(b"corrupted")
    assert operator.recover(call).status is RecoveryStatus.AMBIGUOUS
    with pytest.raises(ValueError):
        load_export(index.parent)
    (Path(call.work_dir) / "data_budget.json").unlink()
    with pytest.raises(ValueError):
        operator.execute(call)
    assert not (Path(call.work_dir) / "data_budget.json").exists()


def test_invalid_nodes_and_output_bindings_fail_before_writes(tmp_path):
    contract, roots = fixture(tmp_path)
    wrong = copy.deepcopy(contract)
    wrong["graph"]["nodes"][0]["role"] = "reviewer"
    with pytest.raises(ValueError, match="pure data"):
        operator_call(tmp_path, wrong, roots)
    wrong = copy.deepcopy(contract)
    wrong["graph"]["nodes"][1]["inputs"]["catalog"]["output"] = "missing"
    with pytest.raises(ValueError, match="output port"):
        operator_call(tmp_path, wrong, roots)
    wrong = copy.deepcopy(contract)
    wrong["graph"]["nodes"][0]["parameters"]["shell"] = "echo nope"
    with pytest.raises(ValueError, match="contract"):
        operator_call(tmp_path, wrong, roots)
    wrong = copy.deepcopy(contract)
    wrong["budget"]["max_operator_calls"] = 5
    with pytest.raises(ValueError, match="Operator budget"):
        operator_call(tmp_path, wrong, roots)
    assert not roots["dataset"].exists() and not (tmp_path / "campaign").exists()


def test_cycles_and_inherited_model_authority_are_rejected():
    a = DataNodeSpec("a", "images.clean", "1", "data", {"catalog": DataInput(node="b", output="catalog")}, {"max_side": 64})
    b = DataNodeSpec("b", "images.clean", "1", "data", {"catalog": DataInput(node="a", output="catalog")}, {"max_side": 64})
    with pytest.raises(ValueError, match="cycle"):
        DataNodeGraph((a, b))
    with pytest.raises(ValueError, match="pure data"):
        replace(a, max_model_calls=1)


def test_deadline_survives_reopening_and_attempts_include_failures(tmp_path):
    now = [100.]
    budget = DataRunBudget(1, 0, 0, 1, 1, 10)
    path = tmp_path / "budget.json"
    ledger = DataBudgetJournal(path, budget, "bound", now=lambda: now[0])
    ledger.reserve_operator("audit")
    with pytest.raises(DataBudgetExceeded, match="Operator"):
        ledger.reserve_operator("audit")
    now[0] = 111.
    reopened = DataBudgetJournal(path, budget, "bound", now=lambda: now[0])
    with pytest.raises(DataBudgetExceeded, match="wall-time"):
        reopened.check()


def test_source_mutation_and_write_escape_fail_before_processing(tmp_path):
    contract, roots = fixture(tmp_path)
    operator, call = operator_call(tmp_path, contract, roots)
    with pytest.raises(ValueError, match="escapes"):
        operator.execute(replace(call, work_dir=str(tmp_path / "outside")))
    (roots["raw"] / "catalog.json").write_text("{}")
    assert operator.execute(call).stage_result["failure_code"] == "INVALID_DATA_EFFECT"
    assert not roots["dataset"].exists()


def test_cross_source_exact_and_declared_leaf_families_share_splits(tmp_path):
    contract, roots = fixture(tmp_path)
    operator, call = operator_call(tmp_path, contract, roots)
    operator.execute(call)
    run = json.loads((Path(call.work_dir) / "node_runs/split/node_run.json").read_text())
    catalog = json.loads(operator.store.resolve(ArtifactRef.from_dict(run["outputs"]["catalog"])).read_text())
    group = [r for r in catalog["rows"] if r["source_row"] == 0]
    assert len({r["family_id"] for r in group}) == 1 and len({r["split"] for r in group}) == 1
    family_splits = {}
    for row in catalog["rows"]:
        family_splits.setdefault(row["family_id"], set()).add(row["split"])
    assert all(len(v) == 1 for v in family_splits.values())


def test_model_node_recovery_and_complete_cache_load_no_additional_worker(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from dataevolver.adapters import clip_worker
    from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
    from dataevolver.dataset.contracts import payload_sha256

    contract, roots = fixture(tmp_path)
    worker = {"kind": "clip_cpu", "model_path_env": "TEST_MODEL", "python_env": "TEST_MODEL_PYTHON", "threads": 2, "max_images": 6,
              "file_hashes": {n: "0" * 64 for n in ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json",
                                                     "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json")}}
    profile = ModelProfile("clip", "local_python", "fixture-clip", "fixture-revision", ("json", "vision"))
    contract["models"] = {"profiles": [profile.to_dict()], "policy": ModelPolicy({"embedder": "clip"}, 1).to_dict(), "workers": {"clip": worker}}
    contract["budget"].update(max_model_calls=1, max_model_images=6)
    score_node = DataNodeSpec("score", "images.score", "1", "model", {"catalog": DataInput(node="clean", output="catalog")},
                              {"texts": ["leaf", "other"], "min_margin": 0., "timeout_seconds": 60}, "embedder", 1, 6)
    nodes = contract["graph"]["nodes"]
    nodes.insert(-1, score_node.to_dict())
    nodes[-1]["inputs"]["catalog"]["node"] = "score"
    nodes[-1]["parameters"]["require_semantic_match"] = True
    contract["quality_policy"]["allowed_operations"].append("images.score")
    monkeypatch.setenv("TEST_MODEL", str(tmp_path / "fake-weights"))
    monkeypatch.setenv("TEST_MODEL_PYTHON", sys.executable)
    children = []

    def child(argv, **kwargs):
        job = json.loads(Path(argv[-1]).read_text())
        children.append(argv)
        response = {"model": profile.model, "output": {"scores": [{"artifact_sha256": image["sha256"], "similarities": [.8, .1]}
                    for image in job["images"]]}, "finish_reason": "stop", "usage": {"images": len(job["images"]), "output_tokens": 0}}
        data_graph.atomic_write_json(Path(job["output"]), {"binding": job["binding"], "response": response, "sha256": payload_sha256(response)})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(clip_worker.subprocess, "run", child)
    original = data_graph.atomic_write_json
    crashed = []

    def crash(path, value):
        if path.name == "node_run.json" and path.parent.name == "score" and not crashed:
            crashed.append(True)
            raise RuntimeError("crash before model NodeRun")
        return original(path, value)

    monkeypatch.setattr(data_graph, "atomic_write_json", crash)
    operator, call = operator_call(tmp_path, contract, roots)
    with pytest.raises(RuntimeError, match="NodeRun"):
        operator.execute(call)
    operator.execute(call)
    monkeypatch.delenv("TEST_MODEL")
    monkeypatch.delenv("TEST_MODEL_PYTHON")
    assert operator.execute(call).reused and len(children) == 1
    assert json.loads((Path(call.work_dir) / "model_receipts/score.intent.json").read_text())["attempts"] == 1
    rows = load_export(operator.store.resolve(exported_index(call)).parent)
    assert all(r["semantic_evidence"]["matches_task"] and r["protected_gates"]["privacy"] is None for r in rows)
    ledger = json.loads((Path(call.work_dir) / "data_budget.json").read_text())
    assert sum(ledger["attempts"].values()) == 7


@pytest.mark.parametrize("row_limit,terminal", [(48, "COMPLETE"), (1, "ABSTAIN")])
def test_existing_controller_owns_terminal_state_and_replay(tmp_path, monkeypatch, row_limit, terminal):
    from dataevolver.harness.cli import _controller
    from dataevolver.harness.models import ControllerState, GoalSpec

    contract, roots = fixture(tmp_path)
    contract["budget"]["max_input_rows"] = row_limit
    monkeypatch.setenv("FIXTURE_RAW", str(roots["raw"]))
    monkeypatch.setenv("FIXTURE_DATASET", str(roots["dataset"]))
    goal_path = Path(__file__).resolve().parents[1] / "configs/harness/visual_contract_mock.json"
    payload = json.loads(goal_path.read_text())
    payload.update(campaign_id="image-fixture", initial_operator_id="data.image_curation", initial_operator_parameters=contract)
    payload["resource_policy"]["allowed_operators"] = ["data.image_curation"]
    payload["objective"]["real_data_execution"] = True
    registry_operator = DataGraphOperator(allowed_write_root=tmp_path / "campaigns/image-fixture", contract=contract)
    for field in ("initial_trial_key", "canary_trial_key"):
        payload["objective"][field].update(operator_id="data.image_curation", operator_version="2", model_config_hash=registry_operator.spec.model_config_hash)
    goal = GoalSpec.from_dict(payload)
    controller = _controller(goal, tmp_path / "campaigns", create=True, idempotency_key="images")
    assert controller.run().state is ControllerState(terminal)
    if terminal == "ABSTAIN":
        assert controller.status().data["last_stage_result"]["failure_code"] == "DATA_BUDGET_EXHAUSTED"
    assert controller.replay().to_dict() == controller.status().to_dict()
    before = (tmp_path / "campaigns/image-fixture/events.jsonl").read_bytes()
    assert _controller(goal, tmp_path / "campaigns", create=False).run().state is ControllerState(terminal)
    assert (tmp_path / "campaigns/image-fixture/events.jsonl").read_bytes() == before


def test_deleted_budget_after_failed_partial_writes_cannot_restart_accounting(tmp_path):
    contract, roots = fixture(tmp_path)
    contract["budget"]["max_input_rows"] = 1
    operator, call = operator_call(tmp_path, contract, roots)
    assert operator.execute(call).status is StageStatus.FAILED
    (Path(call.work_dir) / "data_budget.json").unlink()
    with pytest.raises(ValueError, match="journal is missing"):
        operator.execute(call)


def test_model_node_budget_and_static_parameters_fail_before_side_effects(tmp_path):
    contract, roots = fixture(tmp_path)
    contract["graph"]["nodes"][2]["parameters"]["ratios"] = {"train": 1, "dev": 1, "eval": 1}
    with pytest.raises(ValueError, match="ratios"):
        operator_call(tmp_path, contract, roots)
    assert not roots["dataset"].exists()


def test_archive_intake_uses_bound_training_index_without_extracting_paths(tmp_path):
    import zipfile

    contract, roots = fixture(tmp_path)
    catalog_path = roots["raw"] / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    source = catalog["sources"][0]
    rows = pq.read_table(roots["raw"] / "a.parquet").to_pylist()
    archive_path, index_path = roots["raw"] / "images.zip", roots["raw"] / "index.json"
    index_rows = []
    with zipfile.ZipFile(archive_path, "w") as archive:
        for i, row in enumerate(rows):
            name = f"raw/color/class-{row['label']}/image-{i}.png"
            archive.writestr(name, row["image"]["bytes"])
            index_rows.append({"archive_path": name, "label": row["label"], "family": row["family"]})
    index_path.write_text(json.dumps({"schema_version": "dataevolver.source_index.v1", "rows": index_rows}))
    source.update(format="zip_indexed", file=ArtifactRef("raw", archive_path.name, file_sha256(archive_path)).to_dict(),
                  index=ArtifactRef("raw", index_path.name, file_sha256(index_path)).to_dict())
    catalog_path.write_text(json.dumps(catalog))
    ref = ArtifactRef("raw", catalog_path.name, file_sha256(catalog_path)).to_dict()
    contract["intent"]["sources"] = [ref]
    contract["graph"]["nodes"][0]["inputs"]["sources"]["artifact"] = ref
    operator, call = operator_call(tmp_path, contract, roots)
    operator.execute(call)
    exported = load_export(operator.store.resolve(exported_index(call)).parent)
    assert any(isinstance(row["source_row"], str) for row in exported)
    assert not (roots["raw"] / "raw/color").exists()


def test_export_decodes_after_source_and_campaign_are_unavailable(tmp_path):
    import shutil

    contract, roots = fixture(tmp_path)
    operator, call = operator_call(tmp_path, contract, roots)
    operator.execute(call)
    source = operator.store.resolve(exported_index(call)).parent
    independent = tmp_path / "independent-export"
    shutil.copytree(source, independent)
    shutil.rmtree(roots["raw"])
    shutil.rmtree(Path(call.work_dir))
    assert len(load_export(independent)) == 6


def test_source_scope_cannot_be_rebound_by_a_graph(tmp_path):
    contract, roots = fixture(tmp_path)
    contract["intent"]["sources"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source scope"):
        operator_call(tmp_path, contract, roots)
    assert not roots["dataset"].exists()


def test_predictable_temporary_symlink_cannot_overwrite_another_file(tmp_path):
    contract, roots = fixture(tmp_path)
    operator, call = operator_call(tmp_path, contract, roots)
    budget = operator.journal(call)
    _, step = operator._step(call, operator.graph.ordered()[0], {}, budget)
    directory = roots["dataset"] / step.ctx.prefix
    directory.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"preserve")
    (directory / "payload.bin.tmp").symlink_to(victim)
    ref = step.ctx.write_bytes("payload.bin", b"inside", "application/octet-stream")
    assert victim.read_bytes() == b"preserve" and operator.store.resolve(ref).read_bytes() == b"inside"
