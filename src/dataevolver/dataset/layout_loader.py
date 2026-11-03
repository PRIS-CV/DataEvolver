"""Independent portable v2 loader; no Harness, source store or model dependency."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .collection_spec import member_path
from .contracts import file_sha256, payload_sha256
from .layout_quality import check_claims, spatial_relations, valid_box, validate_annotation


def validate_layout_package(root, metadata, *, require_receipt=True):
    root = Path(root).resolve()
    if metadata.get("schema_version") != "dataevolver.curated_images.v2":
        raise ValueError("unsupported layout export schema")
    if metadata.get("governance") not in {"mock_verified", "layout_ready", "training_candidate",
                                         "technical_candidate_pending_independent_review"}:
        raise ValueError("unsupported layout governance")
    members = metadata["members"]
    if metadata.get("evidence_level") == "mock_verified" and metadata["governance"] != "mock_verified":
        raise ValueError("mock evidence cannot be promoted")
    if not {"manifest.jsonl", "sources.json", "quarantine.jsonl", "reviews.jsonl", "filtering_config.json", "DATA_CARD.md"}.issubset(members):
        raise ValueError("layout export is missing evidence")
    for relative, digest in members.items():
        path = member_path(root, relative)
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError("layout member is missing or modified")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual - set(members) - {"dataset.json"}:
        raise ValueError("layout export contains unindexed members")
    with (root / "manifest.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != metadata["count"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("layout export count or identities changed")
    if not rows and metadata.get("outcome") != "NO_QUALIFIED_OUTPUT":
        raise ValueError("empty export must report no qualified output")
    sources = {source["source_id"]: source for source in json.loads((root / "sources.json").read_text())}
    filtering = json.loads((root / "filtering_config.json").read_text())
    criteria = filtering["resolved"]
    duplicate_groups, related_groups = set(), {}
    for row in rows:
        if row["source_id"] not in sources or row["source_revision"] != sources[row["source_id"]]["revision"]:
            raise ValueError("source provenance binding changed")
        if row["governance"] == "training_candidate" and (
                row["evidence_level"] == "mock_verified" or metadata["evidence_level"] == "mock_verified"
                or any(check["status"] != "pass" for check in row["gates"].values())
                or any(value["receipt"]["backend"] == "mock" for value in row["reviews"])):
            raise ValueError("mock/unknown evidence cannot authorize training candidate")
        if metadata["governance"] == "training_candidate" and row["governance"] != "training_candidate":
            raise ValueError("dataset governance exceeds row evidence")
        if metadata["evidence_level"] == "mock_verified" and row["governance"] != "mock_verified":
            raise ValueError("mock export cannot be promoted")
        if row["duplicate_group"] in duplicate_groups:
            raise ValueError("duplicate group has multiple representatives")
        duplicate_groups.add(row["duplicate_group"])
        if row["split"] not in {"train", "dev", "eval"} or related_groups.get(row["related_group"], row["split"]) != row["split"]:
            raise ValueError("related group crosses splits")
        related_groups[row["related_group"]] = row["split"]
        if any(row["lineage"][name] != row[name] for name in ("duplicate_group", "related_group", "split")):
            raise ValueError("lineage binding changed")
        for field in ("image", "original_image"):
            if members.get(row[field]) != row[f"{field}_sha256"]:
                raise ValueError("image membership/hash binding changed")
        if row["original_image_sha256"] != row["original_sha256"]:
            raise ValueError("original image hash changed")
        with Image.open(root / row["image"]) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or list(image.size) != row["output_size"]:
                raise ValueError("normalized PNG dimensions/mode changed")
        if validate_annotation(row["objects"]) or any(not valid_box(value["box_xyxy"]) for value in row["selected_objects"]):
            raise ValueError("invalid exported annotation")
        selected_ids = {value["id"] for value in row["selected_objects"]}
        if (len(selected_ids) != len(row["selected_objects"]) or row["count_scope"]["total"] != len(selected_ids)
                or set(row["count_scope"]["object_ids"]) != selected_ids
                or any(value["subject_id"] not in selected_ids or value["object_id"] not in selected_ids for value in row["relations"])):
            raise ValueError("count or relation reference changed")
        known = {value["id"]: value for value in row["objects"]}
        if any(known.get(value["id"]) != value for value in row["selected_objects"]):
            raise ValueError("selected objects differ from annotation")
        geometric = {(value["subject_id"], value["predicate"], value["object_id"]) for value in spatial_relations(
            row["selected_objects"], criteria["relation_margin"])}
        if any((value["subject_id"], value["predicate"], value["object_id"]) not in geometric for value in row["relations"]):
            raise ValueError("exported directional relation contradicts geometry")
        for evidence in row["reviews"]:
            receipt = {**evidence["receipt"], "reused": False}
            if payload_sha256(receipt) != evidence["receipt_hash"]:
                raise ValueError("model review receipt hash changed")
        if check_claims(row, row["caption_claims"]):
            raise ValueError("exported caption claims contradict facts")
        if row["layout_ready"] and (row["verified_caption"] != row["caption"] or not row["reviews"]):
            raise ValueError("verified caption or independent review is missing")
        for source_key in ("dataset_evidence", "origin_evidence"):
            if any(name not in members for name in sources[row["source_id"]][source_key]):
                raise ValueError("source evidence is missing")

    def check_references(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"} and members.get(value["path"]) != value["sha256"]:
                raise ValueError("portable evidence reference is missing or changed")
            for item in value.values():
                check_references(item)
        elif isinstance(value, list):
            for item in value:
                check_references(item)
    check_references(rows)
    check_references(json.loads((root / "filtering_config.json").read_text()))
    if require_receipt:
        name = metadata["loader_receipt"]
        if name not in members:
            raise ValueError("independent loader receipt is missing")
        receipt = json.loads((root / name).read_text())
        if (receipt.get("independent_loader_verified") is not True or receipt["count"] != len(rows)
                or receipt["manifest_sha256"] != members["manifest.jsonl"]
                or receipt["members_hash"] != payload_sha256({key: value for key, value in members.items() if key != name})):
            raise ValueError("independent loader receipt binding changed")
    return rows


def load_layout_export(root):
    root = Path(root).resolve()
    return validate_layout_package(root, json.loads((root / "dataset.json").read_text()))
