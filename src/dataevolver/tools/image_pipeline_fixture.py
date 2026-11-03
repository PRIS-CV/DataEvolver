"""Create and exercise synthetic image contracts without network or model loads."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
from pathlib import Path

from PIL import Image

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
from dataevolver.dataset.collection_spec import CollectionSpec, resolve_hf_endpoint
from dataevolver.dataset.contracts import ArtifactRef, canonical_json, file_sha256, payload_sha256
from dataevolver.dataset.layout_quality import REVIEW_CRITERIA, RUBRIC_HASH, compile_criteria
from dataevolver.harness.data_contracts import DataRunBudget
from dataevolver.harness.image_pipeline import collection_contract, image_pipeline_goal, layout_contract


def prepare_fixture(root, *, count_per_source=4):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    raw = root / "inputs"
    raw.mkdir(parents=True, exist_ok=True)

    def write(name, value, media="application/json"):
        path = raw / name
        path.write_text(canonical_json(value))
        return ArtifactRef("raw", name, file_sha256(path), media)

    evidence = write("fixture-evidence.json", {"evidence_level": "mock_verified",
        "scope": "Synthetic fixture only. This is not approval for real data."})
    sources, originals = [], {}
    for source_id, adapter in (("coco_fixture", "coco"), ("vg_fixture", "visual_genome")):
        tables, images, records = {}, [], {key: [] for key in (
            "image_info", "instances", "captions", "objects", "attributes", "relationships", "categories", "licenses")}
        for number in range(count_per_source):
            image_id = number+1 if adapter == "coco" else 1000+number
            data = random.Random(f"image-{source_id}-{number}").randbytes(256*256*3)
            image = Image.frombytes("RGB", (256, 256), data)
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            encoded = stream.getvalue()
            if adapter == "coco":
                originals[number] = encoded
            elif number == 0:
                encoded = originals[number]
            images.append({"id": image_id, "image": {"bytes": encoded, "path": f"{image_id}.png"}})
            info = {"id": image_id, "width": 256, "height": 256, "license": 1,
                    "source_split": "train2017" if adapter == "coco" else "unspecified"}
            if adapter == "coco":
                info["coco_url"] = f"https://images.cocodataset.org/train2017/{image_id:012d}.jpg"
                records["instances"].extend([
                    {"image_id": image_id, "id": 1, "category_id": 1, "bbox": [24, 48, 64, 112]},
                    {"image_id": image_id, "id": 2, "category_id": 2, "bbox": [160, 48, 72, 112]}])
                records["captions"].append({"image_id": image_id, "caption": "A cat is left of a dog."})
            else:
                info.update(image_id=image_id, url=f"https://images.example.invalid/{image_id}.jpg",
                            coco_id=1 if number == 0 else None, license="fixture_only",
                            license_evidence={"fixture": True}, captions=["A cat is left of a dog."])
                records["objects"].append({"image_id": image_id, "objects": [
                    {"object_id": 1, "names": ["cat"], "x": 24, "y": 48, "w": 64, "h": 112},
                    {"object_id": 2, "names": ["dog"], "x": 160, "y": 48, "w": 72, "h": 112}]})
                records["attributes"].append({"image_id": image_id, "attributes": [{"object_id": 1, "attributes": ["red"]}]})
                records["relationships"].append({"image_id": image_id, "relationships": [
                    {"subject": {"object_id": 1}, "object": {"object_id": 2}, "predicate": "left of", "relationship_id": 9}]})
            records["image_info"].append(info)
        records["categories"] = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
        records["licenses"] = [{"id": 1, "name": "fixture_only", "url": "https://example.invalid/fixture"}]
        roles = ("image_info", "instances", "captions", "categories", "licenses") if adapter == "coco" else (
            "image_info", "objects", "attributes", "relationships")
        for role in roles:
            tables[role] = [{"artifact": write(f"{source_id}-{role}.json", records[role]).to_dict()}]
        path = raw / f"{source_id}-images.parquet"
        pq.write_table(pa.Table.from_pylist(images), path)
        image_ref = ArtifactRef("raw", path.name, file_sha256(path), "application/vnd.apache.parquet")
        sources.append({"source_id": source_id, "adapter": adapter, "repo_id": f"fixture/{source_id}",
            "revision": "fixture-v1", "quota": count_per_source, "tables": tables,
            "image_files": [{"artifact": image_ref.to_dict()}],
            "dataset_license": "fixture_only", "dataset_evidence": [evidence.to_dict()],
            "origin_type": "synthetic_fixture", "origin_evidence": [evidence.to_dict()], "allowed_hosts": []})
    spec = CollectionSpec("synthetic-layout", "fixture", tuple(sources), hf_endpoint=resolve_hf_endpoint(),
                          max_candidates=2*count_per_source)
    spec_ref = write("collection.json", spec.to_dict())
    budget = DataRunBudget(8, 0, 0, 2*count_per_source, 200_000_000, 300)
    contract = collection_contract(spec_ref, mode="fixture", stores={"raw": "IMAGE_FIXTURE_RAW", "bundle": "IMAGE_FIXTURE_BUNDLE"},
                                   dataset_store="bundle", budget=budget)
    return {"root": root, "raw": raw, "bundle_root": root/"source_bundles", "dataset_root": root/"curated",
            "collection_contract": contract, "spec": spec, "evidence": evidence, "candidate_count": 2*count_per_source}


def prepare_layout(fixture, bundle_ref):
    raw = fixture["raw"]
    criteria = {"base_profile": "generic_image_v1", "task_profile": "multi_subject_layout_v1", "execution_mode": "fixture",
                "allowed_image_licenses": ["fixture_only"], "allowed_dataset_licenses": ["fixture_only"]}
    calibration = {"schema_version": "dataevolver.layout_calibration.v1", "authority": "independent_human",
        "evidence_level": "mock_verified", "passed": True, "reviewer": {"model": "fixture-reviewer", "revision": "fixture-v1"},
        "rubric_hash": RUBRIC_HASH, "profile_hash": compile_criteria(criteria)["profile_hash"],
        "thresholds": {name: .9 for name in REVIEW_CRITERIA}, "gold_evidence": [fixture["evidence"].to_dict()],
        "development_ids": ["fixture-development"], "held_out_ids": ["fixture-held-out"]}
    path = raw/"calibration.json"
    path.write_text(canonical_json(calibration))
    criteria["calibration"] = ArtifactRef("raw", path.name, file_sha256(path)).to_dict()
    samples = {}
    for source in fixture["spec"].sources:
        for number in range(source["quota"]):
            identity = str(number+1 if source["adapter"] == "coco" else 1000+number)
            sample_id = "img-"+payload_sha256([source["repo_id"], source["revision"], identity])[:24]
            samples[sample_id] = {name: {"status": "pass", "evidence": [fixture["evidence"].to_dict()]}
                                  for name in ("privacy", "safety", "deletion")}
    path = raw/"protected.json"
    path.write_text(canonical_json({"schema_version": "dataevolver.image_protected_evidence.v1",
        "authority": "independent_human", "evidence_level": "mock_verified", "samples": samples}))
    criteria["protected_evidence"] = ArtifactRef("raw", path.name, file_sha256(path)).to_dict()
    output = {"criteria": {name: {"status": "pass", "score": .99, "evidence": "Synthetic fixture observation."}
                           for name in REVIEW_CRITERIA},
              "claims": {"object_ids": ["1", "2"], "counts": [{"category": "cat", "count": 1}, {"category": "dog", "count": 1}],
                         "attributes": [], "relations": [{"subject_id": "1", "predicate": "left_of", "object_id": "2"}]}}
    repair = {"caption": "A cat is left of a dog.", "objects": [
        {"id": "1", "category": "cat", "box_xyxy": [24/256, 48/256, 88/256, 160/256], "attributes": []},
        {"id": "2", "category": "dog", "box_xyxy": [160/256, 48/256, 232/256, 160/256], "attributes": []}]}
    profiles = [ModelProfile("reviewer", "mock", "fixture-reviewer", "fixture-v1", ("json", "vision"), mock_output=output),
                ModelProfile("annotator", "mock", "fixture-annotator", "fixture-v1", ("json", "vision"), mock_output=repair)]
    count = fixture["candidate_count"]
    models = {"profiles": [value.to_dict() for value in profiles],
              "policy": ModelPolicy({"annotator": "annotator", "reviewer": "reviewer"}, 3*count).to_dict(), "workers": {}}
    return layout_contract(bundle_ref, task_criteria=criteria, models=models,
        stores={"raw": "IMAGE_FIXTURE_RAW", "bundle": "IMAGE_FIXTURE_BUNDLE", "dataset": "IMAGE_FIXTURE_DATASET"},
        dataset_store="dataset", budget=DataRunBudget(10, 3*count, 3*count, count, 200_000_000, 300), candidate_limit=count)


def run_fixture(root):
    from dataevolver.harness.cli import main as harness_main
    fixture = prepare_fixture(root)
    root = fixture["root"]
    evaluator = root/"evaluator.json"
    evaluator.write_text(canonical_json({"schema_version": "dataevolver.harness.evaluator.v1", "evaluator_id": "fixture", "revision": "1",
        "hard_constraints": {"artifact_valid": {"required": True}}, "target_metrics": {
            "completion_ratio": {"required": True, "minimum": 1.0}}, "protected_metrics": {}}))
    bindings = {"IMAGE_FIXTURE_RAW": str(fixture["raw"]), "IMAGE_FIXTURE_BUNDLE": str(fixture["bundle_root"]),
                "IMAGE_FIXTURE_DATASET": str(fixture["dataset_root"])}
    previous = {key: os.environ.get(key) for key in bindings}
    os.environ.update(bindings)
    log = io.StringIO()
    outputs = {}
    try:
        for route in ("collection", "curation"):
            contract = fixture["collection_contract"] if route == "collection" else prepare_layout(fixture, outputs["source_bundle"])
            goal = image_pipeline_goal(contract, campaign_id=f"image-{route}-fixture", evaluator_path=evaluator)
            path = root/f"{route}-goal.json"
            path.write_text(goal.to_json())
            with contextlib.redirect_stdout(log):
                if harness_main(["plan", "--goal", str(path)]) != 0:
                    raise ValueError("fixture plan failed")
                if harness_main(["evolve", "--goal", str(path), "--campaign-root", str(root/"campaigns"),
                                 "--idempotency-key", f"{route}-fixture", "--dry-run"]) != 0:
                    raise ValueError("fixture campaign failed")
                for command in ("resume", "status", "replay"):
                    if harness_main([command, "--campaign", goal.campaign_id, "--campaign-root", str(root/"campaigns")]) != 0:
                        raise ValueError(f"fixture {command} failed")
            node = "collect" if route == "collection" else "export"
            node_run = next((root/"campaigns"/goal.campaign_id).glob(f"**/node_runs/{node}/node_run.json"))
            result = json.loads(node_run.read_text())["outputs"]
            outputs.update({name: ArtifactRef.from_dict(reference) for name, reference in result.items()})
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        (root/"harness_calls.jsonl").write_text(log.getvalue())
    result = {"evidence_level": "mock_verified", "source_bundle": outputs["source_bundle"].to_dict(),
              "dataset": outputs["dataset"].to_dict(), "dataset_root": str(fixture["dataset_root"]),
              "network_performed": False, "model_loads": 0}
    (root/"fixture_result.json").write_text(canonical_json(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(canonical_json(run_fixture(args.root)))


if __name__ == "__main__":
    main()
