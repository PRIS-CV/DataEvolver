import copy
import io
import json
import random
import shutil
import socket
import struct
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageOps

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

from dataevolver.adapters.model_gateway import LocalEndpointProvider, ModelError, ModelGateway
from dataevolver.dataset.collection import DownloadLedger, normalize_coco_record, normalize_vg_record
from dataevolver.dataset.collection_spec import CollectionSpec, load_source_bundle, resolve_hf_endpoint
from dataevolver.dataset.contracts import ArtifactRef, canonical_json, file_sha256, payload_sha256
from dataevolver.dataset.curation_loader import load_export
from dataevolver.dataset.image_records import read_parquet
from dataevolver.dataset.layout_quality import (GENERIC_IMAGE_V1, MULTI_SUBJECT_LAYOUT_V1, check_claims, compile_criteria,
    group_and_split, spatial_relations, transform_box_exif, validate_annotation)
from dataevolver.harness.adapters import data_graph
from dataevolver.harness.adapters.image_pipeline import ImageCollectionOperator, ImageLayoutOperator
from dataevolver.harness.cli import _validate_creation_mode, main as harness_main
from dataevolver.harness.data_budget import DataBudgetExceeded, DataBudgetJournal
from dataevolver.harness.data_contracts import DataRunBudget
from dataevolver.harness.models import EvidenceStatus, OperatorCall, RecoveryStatus, StageStatus
from dataevolver.harness.image_pipeline import offline_plan
from dataevolver.tools.image_pipeline_fixture import prepare_fixture, prepare_layout, run_fixture


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("offline tests must not open network connections")
    monkeypatch.setattr(socket.socket, "connect", fail)


def collection_operator(tmp_path, count=4):
    fixture = prepare_fixture(tmp_path/"assets", count_per_source=count)
    contract = fixture["collection_contract"]
    roots = {"raw": fixture["raw"], "bundle": fixture["bundle_root"]}
    operator = ImageCollectionOperator(allowed_write_root=tmp_path/"campaign-collection", contract=contract, roots=roots)
    call = OperatorCall("fixture", "collect", "collect", operator.operator_id, "1", 0,
                        str(operator.root/"execute"), parameters=contract)
    return fixture, operator, call


def rebind_fixture(fixture):
    def refresh(value):
        if isinstance(value, dict):
            if value.get("schema_version") == ArtifactRef.schema:
                return {**value, "sha256": file_sha256(fixture["raw"]/value["path"])}
            return {key: refresh(item) for key, item in value.items()}
        if isinstance(value, list):
            return [refresh(item) for item in value]
        return value
    spec = refresh(fixture["spec"].to_dict())
    fixture["spec"] = CollectionSpec.from_dict(spec)
    path = fixture["raw"]/"collection.json"
    path.write_text(canonical_json(spec))
    reference = ArtifactRef("raw", path.name, file_sha256(path)).to_dict()
    fixture["collection_contract"]["intent"]["sources"] = [reference]
    fixture["collection_contract"]["graph"]["nodes"][0]["inputs"]["spec"]["artifact"] = reference


def collect_changed_fixture(tmp_path, mutate):
    fixture = prepare_fixture(tmp_path/"assets")
    mutate(fixture)
    rebind_fixture(fixture)
    contract = fixture["collection_contract"]
    collection = ImageCollectionOperator(allowed_write_root=tmp_path/"collection", contract=contract,
        roots={"raw": fixture["raw"], "bundle": fixture["bundle_root"]})
    call = OperatorCall("fixture", "collection", "collection", collection.operator_id, "1", 0,
                       str(collection.root/"execute"), parameters=contract)
    assert collection.execute(call).status is StageStatus.SUCCESS
    contract = prepare_layout(fixture, node_output(collection, call, "collect", "source_bundle"))
    layout = ImageLayoutOperator(allowed_write_root=tmp_path/"layout", contract=contract,
        roots={"raw": fixture["raw"], "bundle": fixture["bundle_root"], "dataset": fixture["dataset_root"]})
    call = OperatorCall("fixture", "layout", "layout", layout.operator_id, "3", 0,
                       str(layout.root/"execute"), parameters=contract)
    assert layout.execute(call).status is StageStatus.SUCCESS
    return fixture, layout, call


def node_output(operator, call, node, output):
    run = json.loads((Path(call.work_dir)/"node_runs"/node/"node_run.json").read_text())
    return ArtifactRef.from_dict(run["outputs"][output])


def layout_operator(tmp_path, *, change=None):
    fixture, collection, collection_call = collection_operator(tmp_path)
    assert collection.execute(collection_call).status is StageStatus.SUCCESS
    bundle = node_output(collection, collection_call, "collect", "source_bundle")
    contract = prepare_layout(fixture, bundle)
    if change:
        change(contract)
    roots = {"raw": fixture["raw"], "bundle": fixture["bundle_root"], "dataset": fixture["dataset_root"]}
    operator = ImageLayoutOperator(allowed_write_root=tmp_path/"campaign-layout", contract=contract, roots=roots)
    call = OperatorCall("fixture", "layout", "layout", operator.operator_id, "3", 0,
                        str(operator.root/"execute"), parameters=contract)
    return fixture, operator, call


def export_root(operator, call):
    return operator.store.resolve(node_output(operator, call, "export", "dataset")).parent


def test_two_cli_campaigns_plan_resume_replay_and_portable_export(tmp_path, capsys):
    result = run_fixture(tmp_path/"fixture")
    root = Path(result["dataset_root"])/result["dataset"]["path"]
    rows = load_export(root.parent)
    assert len(rows) == 7
    assert {row["governance"] for row in rows} == {"mock_verified"}
    assert all(row["layout_ready"] and row["verified_caption"] for row in rows)
    for name in ("image-collection-fixture", "image-curation-fixture"):
        final = json.loads((tmp_path/"fixture"/"campaigns"/name/"final_report.json").read_text())
        assert final["capability_level"] == "mock_verified" and final["terminal_state"] == "COMPLETE"
    plan = offline_plan(tmp_path/"fixture"/"curation-goal.json")
    assert plan["operator_version"] == "3" and "review_rubric_hash" in plan
    isolated = tmp_path/"independent-export"
    shutil.copytree(root.parent, isolated)
    shutil.rmtree(tmp_path/"fixture")
    assert load_export(isolated) == rows
    (isolated/rows[0]["image"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="modified"):
        load_export(isolated)


def test_collection_bundle_normalizes_instances_captions_vg_ids_and_relations(tmp_path):
    fixture, operator, call = collection_operator(tmp_path)
    before = {path.name: file_sha256(path) for path in fixture["raw"].iterdir()}
    receipt = operator.execute(call)
    assert operator.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    bundle_path = operator.store.resolve(node_output(operator, call, "collect", "source_bundle"))
    bundle = load_source_bundle(bundle_path)
    rows = [row for shard in bundle.parquet_files for row in read_parquet(bundle_path.parent/shard["path"])]
    assert len(rows) == 8
    assert rows[0]["objects"][0]["id"] == "1" and len(rows[0]["captions"]) == 1
    vg = next(row for row in rows if row["source_id"] == "vg_fixture")
    assert vg["objects"][0]["attributes"] == ["red"]
    assert vg["relations"][0]["relationship_id"] == 9
    assert vg["coco_id"] == rows[0]["coco_id"] == 1
    assert before == {path.name: file_sha256(path) for path in fixture["raw"].iterdir()}
    budget = (Path(call.work_dir)/"data_budget.json").read_bytes()
    assert operator.execute(call).reused
    assert budget == (Path(call.work_dir)/"data_budget.json").read_bytes()
    shard = bundle_path.parent/bundle.parquet_files[0]["path"]
    shard.chmod(0o644)
    shard.write_bytes(b"changed")
    assert operator.recover(call).status is RecoveryStatus.AMBIGUOUS


def test_coco_train_identity_never_inferred_from_hub_train_or_filename():
    raw = {"id": 1, "file_name": "000000000001.jpg", "split": "train", "license": 99}
    row = normalize_coco_record(raw, [], [], {}, {})
    assert row["source_split"] == "train"
    assert row["license"] is None and row["image_license_evidence"]["license_record"] is None
    train = normalize_coco_record({**raw, "coco_url": "http://images.cocodataset.org/train2017/000000000001.jpg"}, [], [], {}, {})
    assert train["source_split"] == "train2017"


def test_vg_excludes_coco_evaluation_ids_and_preserves_unknown_license():
    image = {"image_id": 1, "coco_id": 12, "url": "https://original.example/image.jpg"}
    assert normalize_vg_record(image, [], [], [], {12}) is None
    row = normalize_vg_record(image, [], [], [], {99})
    assert row["image_url"] == image["url"] and row["license"] is None


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_exif_box_transforms_match_actual_lossless_pixels(orientation):
    image = Image.new("RGB", (320, 400), "white")
    image.paste("red", (30, 50, 140, 240))
    exif = image.getexif()
    exif[274] = orientation
    stream = io.BytesIO()
    image.save(stream, format="PNG", exif=exif)
    with Image.open(io.BytesIO(stream.getvalue())) as original:
        actual = ImageOps.exif_transpose(original)
    mask = actual.getchannel("G").point(lambda value: 255 if value == 0 else 0)
    expected, width, height = transform_box_exif([30, 50, 140, 240], orientation, 320, 400)
    assert tuple(expected) == mask.getbbox()
    assert (width, height) == actual.size


@pytest.mark.parametrize("mutation,reason", [
    (lambda values: values[1].update(id="1"), "duplicate"),
    (lambda values: values[0].update(box_xyxy=[-.1, 0, .2, .3]), "invalid"),
    (lambda values: values[0].update(box_xyxy=[0, 0, float("nan"), .3]), "invalid"),
    (lambda values: values[0].update(attributes=[{"value": "red"}]), "attributes"),
])
def test_object_validation_rejects_bad_ids_boxes_and_attributes(mutation, reason):
    values = [{"id": "1", "category": "cat", "box_xyxy": [0, 0, .2, .3], "attributes": []},
              {"id": "2", "category": "dog", "box_xyxy": [.6, .1, .9, .5], "attributes": []}]
    mutation(values)
    assert reason in validate_annotation(values)


def test_spatial_relations_require_full_boxes_and_caption_claims_match_facts():
    objects = [{"id": "1", "category": "cat", "box_xyxy": [0, 0, .4, .5], "attributes": ["red"]},
               {"id": "2", "category": "dog", "box_xyxy": [.6, 0, .9, .5], "attributes": []}]
    relations = spatial_relations(objects)
    row = {"selected_objects": objects, "relations": relations}
    claims = {"object_ids": ["1", "2"], "counts": [], "attributes": [], "relations": [
        {"subject_id": "2", "predicate": "left_of", "object_id": "1"}]}
    assert check_claims(row, claims) == "caption_relation_mismatch"
    claims.update(relations=[], counts=[{"category": "cat", "count": 2}])
    assert check_claims(row, claims) == "caption_count_mismatch"
    claims.update(counts=[], attributes=[{"object_id": "2", "value": "red"}])
    assert check_claims(row, claims) == "caption_attribute_mismatch"
    objects[1]["box_xyxy"] = [.3, 0, .9, .5]
    assert spatial_relations(objects) == []
    assert validate_annotation(objects, relations=[{"subject_id": "missing", "object_id": "1"}])
    assert validate_annotation(objects, attributes=[{"object_id": "missing"}])


def test_duplicate_and_related_groups_have_distinct_semantics_and_scale_to_5000():
    criteria = {**GENERIC_IMAGE_V1, **MULTI_SUBJECT_LAYOUT_V1}
    rng = random.Random(11)
    summaries = [{"id": f"img-{number:05d}", "pixel_sha256": str(number), "dhash": rng.getrandbits(64),
                  "related_keys": []} for number in range(5000)]
    summaries[1].update(pixel_sha256=summaries[0]["pixel_sha256"], dhash=summaries[0]["dhash"])
    summaries[2]["related_keys"] = ["same-source-subject"]
    summaries[3]["related_keys"] = ["same-source-subject"]
    groups = group_and_split(summaries, criteria)
    assert groups[summaries[0]["id"]] == groups[summaries[1]["id"]]
    assert groups[summaries[2]["id"]]["related_group"] == groups[summaries[3]["id"]]["related_group"]
    assert groups[summaries[2]["id"]]["duplicate_group"] != groups[summaries[3]["id"]]["duplicate_group"]
    assert groups[summaries[2]["id"]]["split"] == groups[summaries[3]["id"]]["split"]
    assert {value["split"] for value in groups.values()} == {"train", "dev", "eval"}


@pytest.mark.parametrize("missing", ("calibration", "protected_evidence"))
def test_missing_independent_evidence_stays_unknown(tmp_path, missing):
    def change(contract):
        contract["quality_policy"]["task_criteria"].pop(missing)
    fixture, operator, call = layout_operator(tmp_path, change=change)
    receipt = operator.execute(call)
    assert receipt.status is StageStatus.SUCCESS
    root = export_root(operator, call)
    if missing == "calibration":
        assert not load_export(root)
        report = json.loads((root.parent/"export_report.json").read_text())
        assert report["outcome"] == "NO_QUALIFIED_OUTPUT" and report["gate_counts"]["unknown"]["calibration"] == 8
    else:
        assert all(row["gates"]["privacy"]["status"] == "unknown" and row["gates"]["safety"]["status"] == "unknown" and row["gates"]["deletion"]["status"] == "unknown"
                   for row in load_export(root))


def test_one_repair_then_independent_rereview_rejects_disagreement(tmp_path):
    def change(contract):
        reviewer = next(value for value in contract["models"]["profiles"] if value["profile_id"] == "reviewer")
        reviewer["mock_output"]["criteria"]["caption_faithfulness"]["status"] = "fail"
    fixture, operator, call = layout_operator(tmp_path, change=change)
    assert operator.execute(call).status is StageStatus.SUCCESS
    assert not load_export(export_root(operator, call))
    index_path = operator.store.resolve(node_output(operator, call, "rereview", "catalog"))
    index = json.loads(index_path.read_text())
    rows = [row for shard in index["shards"] for row in read_parquet(operator.store.resolve(ArtifactRef.from_dict(shard)))]
    assert {row["repair_count"] for row in rows} == {1}
    assert all([value["role"] for value in row["reviews"]] == ["reviewer", "annotator", "reviewer"] for row in rows)


@pytest.mark.parametrize("defect", ("blur", "watermark", "noise", "exposure"))
def test_visual_defects_are_quarantined_without_threshold_relaxation(tmp_path, defect):
    def change(contract):
        contract["models"]["profiles"][0]["mock_output"]["criteria"][defect]["status"] = "fail"
    fixture, operator, call = layout_operator(tmp_path, change=change)
    assert operator.execute(call).status is StageStatus.SUCCESS
    assert load_export(export_root(operator, call)) == []


def test_mock_governance_cannot_be_requested_as_an_export_parameter(tmp_path):
    fixture, operator, call = layout_operator(tmp_path)
    changed = copy.deepcopy(operator.contract)
    changed["graph"]["nodes"][-1]["parameters"]["governance"] = "training_candidate"
    with pytest.raises(ValueError, match="operator contract"):
        ImageLayoutOperator(allowed_write_root=operator.root, contract=changed,
            roots={"raw": fixture["raw"], "bundle": fixture["bundle_root"], "dataset": fixture["dataset_root"]})


def test_effect_receipt_loss_recovers_without_reinference(tmp_path, monkeypatch):
    fixture, operator, call = layout_operator(tmp_path)
    original = data_graph._write_receipt
    crashed = []

    def fail_once(path, receipt):
        if receipt.operator_id == "layout.review" and not crashed:
            crashed.append(True)
            raise RuntimeError("interrupted after review effect")
        return original(path, receipt)
    monkeypatch.setattr(data_graph, "_write_receipt", fail_once)
    with pytest.raises(RuntimeError, match="interrupted"):
        operator.execute(call)
    intents = {path.name: file_sha256(path) for path in (Path(call.work_dir)/"model_receipts").glob("*.intent.json")}
    assert operator.recover(call).status is RecoveryStatus.RETRY_SAFE
    assert operator.execute(call).status is StageStatus.SUCCESS
    assert intents == {path.name: file_sha256(path) for path in (Path(call.work_dir)/"model_receipts").glob("*.intent.json")}
    assert len(load_export(export_root(operator, call))) == 7


def test_missing_model_receipt_or_budget_cannot_be_reused(tmp_path):
    fixture, operator, call = layout_operator(tmp_path)
    operator.execute(call)
    path = next((Path(call.work_dir)/"model_receipts").glob("*.receipt.json"))
    path.unlink()
    assert operator.recover(call).status is RecoveryStatus.AMBIGUOUS
    (Path(call.work_dir)/"data_budget.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        operator.execute(call)


def test_no_endpoint_automatic_resend_after_unknown_timeout(tmp_path):
    from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy, ModelRequest
    profile = ModelProfile("reviewer", "local_endpoint", "server-reviewer", "pinned-revision",
                           endpoint_env="IMAGE_TEST_ENDPOINT")
    policy = ModelPolicy({"reviewer": "reviewer"}, 2, ("http://127.0.0.1:9000",))
    previous = __import__("os").environ.get("IMAGE_TEST_ENDPOINT")
    __import__("os").environ["IMAGE_TEST_ENDPOINT"] = "http://127.0.0.1:9000/v1"
    calls = []

    class TimeoutProvider:
        def invoke(self, request, request_hash):
            calls.append(request.request_id)
            raise ModelError("AMBIGUOUS_EFFECT", "unknown timeout")

        def lookup(self, request, request_hash):
            raise ModelError("AMBIGUOUS_EFFECT", "no native lookup")
    try:
        gateway = ModelGateway(profiles=(profile,), policy=policy, receipt_root=tmp_path,
                               providers={"reviewer": TimeoutProvider})
        request = ModelRequest("timeout", "reviewer", ({"role": "user", "content": "review"},), {"type": "object"})
        for attempt in range(2):
            with pytest.raises(ModelError, match="timeout|lookup"):
                gateway.invoke(request)
        assert calls == ["timeout"]
        assert json.loads(next(tmp_path.glob("*.intent.json")).read_text())["attempts"] == 1
    finally:
        if previous is None:
            __import__("os").environ.pop("IMAGE_TEST_ENDPOINT")
        else:
            __import__("os").environ["IMAGE_TEST_ENDPOINT"] = previous


def test_endpoint_precedence_and_protected_profile_overrides(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert resolve_hf_endpoint() == "https://hf-mirror.com"
    monkeypatch.setenv("HF_ENDPOINT", "https://environment.example/")
    assert resolve_hf_endpoint() == "https://environment.example"
    assert resolve_hf_endpoint("https://configured.example/") == "https://configured.example"
    criteria = {"base_profile": "generic_image_v1", "task_profile": "multi_subject_layout_v1",
                "execution_mode": "fixture", "overrides": {"privacy": False}}
    with pytest.raises(ValueError, match="protected"):
        compile_criteria(criteria)


def ledger_fixture(tmp_path):
    fixture = prepare_fixture(tmp_path/"fixture", count_per_source=1)
    sources = tuple({**source, "revision": "a"*40} for source in fixture["spec"].sources)
    spec = replace(fixture["spec"], mode="server", sources=sources)
    budget = DataBudgetJournal(tmp_path/"budget.json", DataRunBudget(1, 0, 0, 2, 10000, 120), "binding")
    context = SimpleNamespace(parent=SimpleNamespace(work_dir=tmp_path), check=budget.check, budget=budget,
                              graph=SimpleNamespace(dataset_root=tmp_path/"cache"))
    return DownloadLedger(context, spec), sources[0]


def test_hf_provider_uses_one_endpoint_and_exact_revision_and_tracks_cache_bytes(tmp_path, monkeypatch):
    import huggingface_hub
    ledger, source = ledger_fixture(tmp_path)
    calls = []

    class Api:
        def __init__(self, **kwargs):
            calls.append(("metadata_endpoint", kwargs["endpoint"]))

        def repo_info(self, repo_id, **kwargs):
            calls.append(("metadata_revision", kwargs["revision"]))
            return SimpleNamespace(sha=kwargs["revision"], siblings=[SimpleNamespace(rfilename="shard.parquet", size=8)])

    def download(**kwargs):
        calls.append(("download_endpoint", kwargs["endpoint"]))
        calls.append(("download_revision", kwargs["revision"]))
        progress = kwargs["tqdm_class"](total=8, disable=True)
        progress.update_transfer(8)
        path = tmp_path/"sdk-file"
        path.write_bytes(b"12345678")
        return str(path)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    ledger.pin(source)
    first = ledger.download(source, {"path": "shard.parquet"})
    assert ledger.download(source, {"path": "shard.parquet"}) == first
    assert ledger.data["transferred_bytes"] == 8
    assert calls == [("metadata_endpoint", "https://hf-mirror.com"), ("metadata_revision", "a"*40),
                     ("download_endpoint", "https://hf-mirror.com"), ("download_revision", "a"*40)]
    reopened = DownloadLedger(ledger.ctx, ledger.spec)
    assert reopened.download(source, {"path": "shard.parquet"}) == first
    assert reopened.data["transferred_bytes"] == 8


def test_unavailable_pinned_revision_never_falls_back(tmp_path, monkeypatch):
    import huggingface_hub
    ledger, source = ledger_fixture(tmp_path)
    calls = []

    class Missing:
        def __init__(self, **kwargs):
            calls.append(kwargs["endpoint"])

        def repo_info(self, *args, **kwargs):
            raise ValueError("revision unavailable")
    monkeypatch.setattr(huggingface_hub, "HfApi", Missing)
    with pytest.raises(ValueError, match="unavailable"):
        ledger.pin(source)
    assert calls == ["https://hf-mirror.com"]


def test_download_attempts_are_bounded_and_budget_failure_is_not_retried(tmp_path, monkeypatch):
    import huggingface_hub
    ledger, source = ledger_fixture(tmp_path)
    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs["revision"])
        raise OSError("temporary outage")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    for attempt in range(2):
        with pytest.raises(ValueError, match="attempts exhausted"):
            ledger.download(source, {"path": "large-shard.parquet"})
    assert len(calls) == 3
    ledger.spec = replace(ledger.spec, max_download_bytes=1)

    def over_budget(**kwargs):
        kwargs["tqdm_class"](total=8, disable=True).update_transfer(8)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", over_budget)
    with pytest.raises(DataBudgetExceeded, match="byte"):
        ledger.download(source, {"path": "different-shard.parquet"})
    item = next(value for value in ledger.data["items"].values() if value["file"]["path"] == "different-shard.parquet")
    assert item["attempts"] == 1


def test_corrupt_pixel_bomb_and_small_images_are_quarantined_and_exif_coordinates_follow_pixels(tmp_path):
    def mutate(fixture):
        path = fixture["raw"]/"coco_fixture-images.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["image"]["bytes"] = b"corrupt-image"
        raw = bytearray(rows[1]["image"]["bytes"])
        raw[16:24] = struct.pack(">II", 9000, 9000)
        raw[29:33] = struct.pack(">I", zlib.crc32(raw[12:29]) & 0xffffffff)
        rows[1]["image"]["bytes"] = bytes(raw)
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(stream, format="PNG")
        rows[2]["image"]["bytes"] = stream.getvalue()
        with Image.open(io.BytesIO(rows[3]["image"]["bytes"])) as image:
            exif = image.getexif()
            exif[274] = 6
            stream = io.BytesIO()
            image.save(stream, format="PNG", exif=exif)
        rows[3]["image"]["bytes"] = stream.getvalue()
        pq.write_table(pa.Table.from_pylist(rows), path)
    fixture, operator, call = collect_changed_fixture(tmp_path, mutate)
    index = json.loads(operator.store.resolve(node_output(operator, call, "audit", "catalog")).read_text())
    rows = [row for shard in index["shards"] for row in read_parquet(operator.store.resolve(ArtifactRef.from_dict(shard)))]
    coco = {row["source_row_id"]: row for row in rows if row["source_id"] == "coco_fixture"}
    assert coco["1"]["gates"]["integrity"]["status"] == "fail"
    assert coco["2"]["gates"]["integrity"]["reason"] == "image exceeds pixel bound"
    assert coco["3"]["gates"]["dimensions"]["status"] == "fail"
    oriented = coco["4"]
    assert oriented["transform"]["exif_orientation"] == 6
    expected, width, height = transform_box_exif([24, 48, 88, 160], 6, 256, 256)
    assert oriented["objects"][0]["box_xyxy"] == [expected[0]/width, expected[1]/height, expected[2]/width, expected[3]/height]


def test_a_source_can_yield_zero_without_dropping_other_sources(tmp_path):
    def mutate(fixture):
        path = fixture["raw"]/"vg_fixture-images.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["image"]["bytes"] = b"corrupt"
        pq.write_table(pa.Table.from_pylist(rows), path)
    fixture, operator, call = collect_changed_fixture(tmp_path, mutate)
    report = json.loads((export_root(operator, call).parent/"export_report.json").read_text())
    assert report["source_counts"] == {"coco_fixture": 4, "vg_fixture": 0}
    assert len(load_export(export_root(operator, call))) == 4


def test_out_of_bounds_boxes_are_recorded_then_repaired_once(tmp_path):
    def mutate(fixture):
        path = fixture["raw"]/"coco_fixture-instances.json"
        rows = json.loads(path.read_text())
        rows[0]["bbox"] = [-100, 48, 200, 112]
        path.write_text(canonical_json(rows))
    fixture, operator, call = collect_changed_fixture(tmp_path, mutate)
    index = json.loads(operator.store.resolve(node_output(operator, call, "audit", "catalog")).read_text())
    rows = [row for shard in index["shards"] for row in read_parquet(operator.store.resolve(ArtifactRef.from_dict(shard)))]
    original = next(row for row in rows if row["source_id"] == "coco_fixture" and row["source_row_id"] == "1")
    assert original["gates"]["boxes"]["status"] == "fail"
    assert original["raw_annotation"]["instances"][0]["bbox"][0] == -100
    repaired_index = json.loads(operator.store.resolve(node_output(operator, call, "rereview", "catalog")).read_text())
    repaired = next(row for shard in repaired_index["shards"] for row in read_parquet(operator.store.resolve(ArtifactRef.from_dict(shard)))
                    if row["id"] == original["id"])
    assert repaired["repair_count"] == 1
    assert repaired["gates"]["boxes"]["status"] == "pass"
    assert repaired["gates"]["independent_review"]["status"] == "pass"


def test_model_cannot_override_protected_evidence_with_extra_output(tmp_path):
    def change(contract):
        contract["models"]["profiles"][0]["mock_output"]["privacy"] = "pass"
    fixture, operator, call = layout_operator(tmp_path, change=change)
    receipt = operator.execute(call)
    assert receipt.status is StageStatus.FAILED
    assert receipt.stage_result["failure_code"] == "MODEL_INVALID_OUTPUT"
    assert not list(fixture["dataset_root"].glob("**/dataset.json"))


def test_same_model_repair_and_review_remain_coupled(tmp_path):
    def change(contract):
        contract["models"]["profiles"][1]["model"] = "fixture-reviewer"
        contract["models"]["profiles"][0]["mock_output"]["claims"]["counts"][0]["count"] = 2
    fixture, operator, call = layout_operator(tmp_path, change=change)
    assert operator.execute(call).status is StageStatus.SUCCESS
    assert not load_export(export_root(operator, call))
    report = json.loads((export_root(operator, call).parent/"export_report.json").read_text())
    assert report["gate_counts"]["unknown"]["independent_review"] == 8


def test_missing_license_is_unknown_and_never_invented(tmp_path):
    def mutate(fixture):
        path = fixture["raw"]/"coco_fixture-licenses.json"
        path.write_text("[]")
    fixture, operator, call = collect_changed_fixture(tmp_path, mutate)
    rows = load_export(export_root(operator, call))
    coco = [row for row in rows if row["source_id"] == "coco_fixture"]
    assert coco and all(row["license"] is None and row["gates"]["license"]["status"] == "unknown" for row in coco)


@pytest.mark.parametrize("corruption", ("document", "record"))
def test_metadata_failure_is_logged_without_blocking_other_records_or_sources(tmp_path, corruption):
    def mutate(fixture):
        path = fixture["raw"]/"coco_fixture-image_info.json"
        if corruption == "document":
            path.write_text("{invalid")
        else:
            rows = json.loads(path.read_text())
            rows[0]["id"] = "invalid-id"
            path.write_text(canonical_json(rows))
    fixture, operator, call = collect_changed_fixture(tmp_path, mutate)
    index = json.loads(operator.store.resolve(node_output(operator, call, "audit", "catalog")).read_text())
    bundle_path = operator.store.resolve(ArtifactRef.from_dict(index["bundle"]))
    bundle = load_source_bundle(bundle_path)
    assert bundle.count == (4 if corruption == "document" else 7)
    downloads = json.loads((bundle_path.parent/"downloads.json").read_text())
    assert any("metadata" in item["reason"] for item in downloads["items"] if item["source_id"] == "coco_fixture")
    assert next(source for source in bundle.sources if source["source_id"] == "vg_fixture")["collected_count"] == 4


def test_server_collection_rejects_unresolved_template_before_network():
    path = Path(__file__).resolve().parents[1]/"configs/harness/image_sources.example.json"
    with pytest.raises(ValueError, match="unresolved __SET_"):
        CollectionSpec.from_dict(json.loads(path.read_text()))


def test_external_range_resume_preserves_original_host_and_records_received_bytes(tmp_path, monkeypatch):
    import urllib.request
    ledger, source = ledger_fixture(tmp_path)
    source = {**source, "allowed_hosts": ["original.example"]}
    requested = []

    class Response:
        status = 200
        headers = {"ETag": '"fixture"', "Content-Length": "8"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            if not getattr(self, "first_read", False):
                self.first_read = True
                return b"1234"
            raise OSError("interrupted transfer")

    class Resumed(Response):
        status = 206
        headers = {"ETag": '"fixture"', "Content-Length": "4", "Content-Range": "bytes 4-7/8"}

        def read(self, size):
            if not getattr(self, "first_read", False):
                self.first_read = True
                return b"5678"
            return b""

    class Opener:
        def open(self, request, **kwargs):
            requested.append((request.full_url, request.get_header("Range")))
            return Response() if len(requested) == 1 else Resumed()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    result = ledger.download(source, {"url": "https://original.example/image"})
    assert result.read_bytes() == b"12345678"
    assert requested == [("https://original.example/image", None), ("https://original.example/image", "bytes=4-")]
    assert ledger.data["transferred_bytes"] == 8


def test_plan_creates_no_campaign_and_dry_run_is_required_for_fixtures(tmp_path, capsys):
    from dataevolver.harness.image_pipeline import image_pipeline_goal
    fixture = prepare_fixture(tmp_path/"fixture", count_per_source=1)
    evaluator = tmp_path/"evaluator.json"
    evaluator.write_text("{}")
    goal = image_pipeline_goal(fixture["collection_contract"], campaign_id="offline-plan", evaluator_path=evaluator)
    path = tmp_path/"goal.json"
    path.write_text(goal.to_json())
    before = {str(value) for value in tmp_path.rglob("*")}
    assert harness_main(["plan", "--goal", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["network_performed"] is False and output["campaign_created"] is False
    assert before == {str(value) for value in tmp_path.rglob("*")}
    with pytest.raises(ValueError, match="dry-run"):
        _validate_creation_mode(goal, dry_run=False)
