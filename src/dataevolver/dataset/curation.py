"""Bounded image curation steps extracted from Raw Image Pair audit/export ideas.

Inputs and writes are supplied by a Harness node context. Sources are immutable;
semantic scores never grant provenance, license, privacy or safety authority.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath
from itertools import islice

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from .contracts import ArtifactRef, canonical_json, payload_sha256


def _source_rows(ctx, source, path):
    import pyarrow.parquet as pq

    index = None
    if source.get("index"):
        index = json.loads(ctx.store.resolve(ArtifactRef.from_dict(source["index"])).read_text())
        if set(index) != {"schema_version", "rows"} or index["schema_version"] != "dataevolver.source_index.v1":
            raise ValueError("unsupported source row index")
    format_name = source.get("format", "parquet")
    if format_name == "zip_indexed":
        if index is None:
            raise ValueError("archive intake requires an immutable member index")
        if len({r.get("archive_path") for r in index["rows"]}) != len(index["rows"]):
            raise ValueError("archive index contains duplicate source units")
        with zipfile.ZipFile(path) as archive:
            for item in index["rows"]:
                if set(item) != {"archive_path", "label", "family"}:
                    raise ValueError("archive index has unsupported fields")
                name = item["archive_path"]
                member = PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts or str(member) != name or "\\" in name:
                    raise ValueError("archive index member escapes source")
                if archive.getinfo(name).file_size > 32 * 1024 * 1024:
                    raise ValueError("archive image exceeds bounded encoded size")
                yield name, {source["image_column"]: {"bytes": archive.read(name), "path": name},
                             source["label_column"]: item["label"], source["family_column"]: item["family"]}
        return
    if format_name != "parquet":
        raise ValueError("unsupported source format")
    wanted = None
    if index is not None:
        if any(type(i) is not int or i < 0 for i in index["rows"]) or len(set(index["rows"])) != len(index["rows"]):
            raise ValueError("parquet index must contain unique non-negative row offsets")
        wanted = set(index["rows"])
    parquet = pq.ParquetFile(path)
    if wanted and max(wanted) >= parquet.metadata.num_rows:
        raise ValueError("parquet row index exceeds source membership")
    columns = [source["image_column"], source["label_column"]]
    if source["family_column"]:
        columns.append(source["family_column"])
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise ValueError("source columns do not match the frozen catalog")
    offset = 0
    for batch in parquet.iter_batches(batch_size=16, columns=columns):
        for value in batch.to_pylist():
            if wanted is None or offset in wanted:
                yield offset, value
            offset += 1


def audit(ctx):
    catalog = ctx.read("sources")
    if set(catalog) != {"schema_version", "sources"} or catalog["schema_version"] != "dataevolver.image_sources.v1":
        raise ValueError("unsupported source catalog")
    seen, rows, rejected, sources = set(), [], [], []
    limit = ctx.parameters["max_rows_per_source"]
    allowed = ctx.policy.task_criteria.get("allowed_licenses", [])
    for source in catalog["sources"]:
        required = {"source_id", "dataset", "revision", "source_split", "file", "license", "license_evidence",
                    "image_column", "label_column", "family_column"}
        if (not required.issubset(source) or set(source) - required - {"format", "index"} or
                source["source_id"] in seen or not source["revision"]):
            raise ValueError("source identity/fields must be explicit and unique")
        seen.add(source["source_id"])
        if source["source_split"] != "train":
            raise ValueError("P2 curation intake accepts upstream training data only")
        if source["license"] not in allowed or not source["license_evidence"]:
            raise ValueError("source license is unapproved or lacks bound evidence")
        file_ref = ArtifactRef.from_dict(source["file"])
        if file_ref.store == ctx.graph.dataset_store:
            raise ValueError("original sources must use a read-only artifact store")
        path = ctx.store.resolve(file_ref)
        for value in source["license_evidence"]:
            ctx.store.resolve(ArtifactRef.from_dict(value))
        count = 0
        for source_row, value in islice(_source_rows(ctx, source, path), limit):
            ctx.check()
            unit_id = "img-" + payload_sha256([source["dataset"], source["revision"], file_ref.sha256, source_row])[:24]
            ctx.budget.reserve_rows(unit_id, 1)
            count += 1
            image_value = value[source["image_column"]]
            raw = image_value.get("bytes") if isinstance(image_value, dict) else None
            if not isinstance(raw, bytes):
                raise ValueError("intake requires embedded image bytes, never external row paths")
            try:
                with Image.open(io.BytesIO(raw)) as opened:
                    if opened.width * opened.height > ctx.parameters["max_pixels"]:
                        raise ValueError("image pixel limit exceeded")
                    opened.load()
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    if min(image.size) < ctx.parameters["min_side"]:
                        rejected.append({"id": unit_id, "source_id": source["source_id"], "reason": "image_too_small"})
                        continue
                    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                    pixels = small.tobytes()
                    bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1] for y in range(8) for x in range(8)]
                    dhash = sum(int(bit) << i for i, bit in enumerate(bits))
                    pixel_hash = hashlib.sha256(canonical_json(image.size).encode() + image.tobytes()).hexdigest()
                    sharpness = ImageStat.Stat(image.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]
                    size = list(image.size)
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                rejected.append({"id": unit_id, "source_id": source["source_id"], "reason": "decode_failed"})
                continue
            ref = ctx.write_bytes(f"intake/{unit_id}.image", raw, "application/octet-stream")
            family = value.get(source["family_column"]) if source["family_column"] else None
            rows.append({"id": unit_id, "source_id": source["source_id"], "source_dataset": source["dataset"],
                         "source_revision": source["revision"], "source_file_sha256": file_ref.sha256,
                         "source_row": source_row, "source_split": source["source_split"],
                         "source_label": value[source["label_column"]], "license": source["license"],
                         "source_family": str(family) if family is not None else "",
                         "image": ref.to_dict(), "pixel_sha256": pixel_hash, "dhash": f"{dhash:016x}",
                         "width": size[0], "height": size[1], "edge_variance": sharpness,
                         "protected_gates": {"integrity": True, "provenance": True, "license": True,
                                             "privacy": None, "safety": None}})
        ctx.store.resolve(file_ref)  # Recheck the immutable source after reading.
        # Keep the immutable digest as provenance. Recheck the original source
        # catalog at graph verification, rather than rereading large archives
        # through every intermediate catalog.
        recorded = {**source, "rows_inspected": count}
        recorded["file"] = {k: v for k, v in source["file"].items() if k != "schema_version"}
        sources.append(recorded)
    if not rows:
        raise ValueError("source audit found no eligible images")
    return {"catalog": ctx.write_json("audit.json", {"rows": rows, "rejected": rejected, "sources": sources})}


def families(ctx):
    from .families import assign_families, duplicate_method
    catalog = ctx.read("catalog")
    distance = ctx.parameters["hamming_distance"]
    catalog["rows"] = assign_families(catalog["rows"], distance, check=ctx.check)
    catalog["duplicate_method"] = duplicate_method(distance)
    return {"catalog": ctx.write_json("families.json", catalog)}


def split(ctx):
    catalog = ctx.read("catalog")
    groups = sorted({r["family_id"] for r in catalog["rows"]},
                    key=lambda family: payload_sha256([ctx.parameters["seed"], family]))
    ratios = ctx.parameters["ratios"]
    if set(ratios) != {"train", "dev", "eval"} or any(type(v) not in (int, float) or v < 0 for v in ratios.values()):
        raise ValueError("split ratios must specify train/dev/eval")
    if abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("split ratios must sum to one")
    dev = max(1, int(len(groups) * ratios["dev"])) if len(groups) >= 3 and ratios["dev"] else 0
    evaluation = max(1, int(len(groups) * ratios["eval"])) if len(groups) >= 3 and ratios["eval"] else 0
    assignments = {family: ("dev" if i < dev else "eval" if i < dev + evaluation else "train")
                   for i, family in enumerate(groups)}
    for row in catalog["rows"]:
        row["split"] = assignments[row["family_id"]]
        row["protected_gates"]["leakage"] = True
    catalog["split_policy"] = {"ratios": ratios, "seed": ctx.parameters["seed"],
                               "scope": "engineering_slice_not_sealed_benchmark"}
    return {"catalog": ctx.write_json("split.json", catalog)}


def select(ctx):
    catalog = ctx.read("catalog")
    groups = defaultdict(list)
    for row in sorted(catalog["rows"], key=lambda r: payload_sha256([ctx.parameters["seed"], r["id"]])):
        groups[row["source_id"], row["split"]].append(row)
    selected, seen = [], set()
    for source in sorted({r["source_id"] for r in catalog["rows"]}):
        count = 0
        while count < ctx.parameters["per_source"]:
            progress = False
            for partition in ("train", "dev", "eval"):
                pool = groups[source, partition]
                while pool and pool[0]["family_id"] in seen:
                    pool.pop(0)
                if pool and count < ctx.parameters["per_source"]:
                    row = pool.pop(0)
                    seen.add(row["family_id"])
                    row["protected_gates"]["duplicates"] = True
                    selected.append(row)
                    count += 1
                    progress = True
            if not progress:
                break
        if count != ctx.parameters["per_source"]:
            raise ValueError("source quota cannot be met without duplicate families")
    catalog["input_count"] = sum(s["rows_inspected"] for s in catalog["sources"])
    catalog["eligible_count"] = len(catalog["rows"])
    catalog["rows"] = selected
    catalog["selection_policy"] = dict(ctx.parameters)
    return {"catalog": ctx.write_json("selection.json", catalog)}


def clean(ctx):
    catalog = ctx.read("catalog")
    for row in catalog["rows"]:
        ctx.check()
        parent_ref = ArtifactRef.from_dict(row["image"])
        with Image.open(ctx.store.resolve(parent_ref)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((ctx.parameters["max_side"], ctx.parameters["max_side"]), Image.Resampling.LANCZOS)
            image.info.clear()
            out = io.BytesIO()
            image.save(out, format="PNG")
        ref = ctx.write_bytes(f"assets/{row['id']}.png", out.getvalue(), "image/png")
        row["image"] = ref.to_dict()
        row["output_size"] = list(image.size)
        row["lineage"] = {"parent": parent_ref.to_dict(), "operator": "images.clean", "version": "1",
                          "parameters": dict(ctx.parameters), "family_id": row["family_id"], "split": row["split"]}
    return {"catalog": ctx.write_json("cleaned.json", catalog)}


def score(ctx):
    from dataevolver.adapters.model_contracts import ModelRequest

    catalog = ctx.read("catalog")
    rows = catalog["rows"]
    schema = {"type": "object", "required": ["scores"], "additionalProperties": False,
              "properties": {"scores": {"type": "array", "minItems": len(rows), "maxItems": len(rows),
              "items": {"type": "object", "required": ["artifact_sha256", "similarities"], "additionalProperties": False,
              "properties": {"artifact_sha256": {"type": "string"}, "similarities": {"type": "array",
              "minItems": len(ctx.parameters["texts"]), "maxItems": len(ctx.parameters["texts"]),
              "items": {"type": "number", "minimum": -1, "maximum": 1}}}}}}}
    request = ModelRequest(request_id=ctx.node.node_id, role=ctx.node.role,
                           messages=({"role": "user", "content": canonical_json({"texts": ctx.parameters["texts"]})},),
                           response_schema=schema, artifacts=tuple(ArtifactRef.from_dict(r["image"]) for r in rows),
                           max_tokens=1, timeout_seconds=ctx.parameters["timeout_seconds"])
    receipt = ctx.invoke(request)
    for row, result in zip(rows, receipt.output["scores"]):
        if result["artifact_sha256"] != row["image"]["sha256"]:
            raise ValueError("model image score order or input binding changed")
        margin = result["similarities"][0] - max(result["similarities"][1:])
        row["semantic_evidence"] = {"similarities": result["similarities"], "texts": ctx.parameters["texts"],
                                    "margin": margin, "matches_task": margin >= ctx.parameters["min_margin"],
                                    "receipt_hash": receipt.without_reuse_hash(), "calibrated": False}
        row["governance"] = "technical_candidate_pending_independent_review"
    catalog["model_receipts"] = [receipt.to_dict()]
    return {"catalog": ctx.write_json("scored.json", catalog)}


def export(ctx):
    from .curation_loader import load_export

    catalog = ctx.read("catalog")
    exported, quarantined, members = [], list(catalog["rejected"]), {}

    def write(name, data, media="application/octet-stream"):
        ref = ctx.write_bytes("export/" + name, data, media)
        members[name] = ref.sha256
        return ref

    for row in catalog["rows"]:
        if ctx.parameters["require_semantic_match"] and not row.get("semantic_evidence", {}).get("matches_task", False):
            quarantined.append({"id": row["id"], "reason": "semantic_mismatch"})
            continue
        if any(row["protected_gates"].get(g) is not True for g in ("integrity", "provenance", "license", "duplicates", "leakage")):
            raise ValueError("protected deterministic gate missing")
        # Unknown privacy/safety stays unknown: export is an engineering candidate.
        ref = ArtifactRef.from_dict(row["image"])
        relative = f"images/{row['id']}.png"
        write(relative, ctx.store.resolve(ref).read_bytes(), "image/png")
        exported.append({**row, "image": relative, "image_sha256": ref.sha256})
    if not exported:
        raise ValueError("no candidate images survived the declared task filter")
    if {r["source_id"] for r in exported} != {s["source_id"] for s in catalog["sources"]}:
        raise ValueError("export would drop an entire declared source after task filtering")
    source_records = []
    for source in catalog["sources"]:
        evidence = []
        for i, value in enumerate(source["license_evidence"]):
            ref = ArtifactRef.from_dict(value)
            name = f"source_evidence/{source['source_id']}-{i}.txt"
            write(name, ctx.store.resolve(ref).read_bytes(), "text/plain")
            evidence.append(name)
        source_records.append({k: source[k] for k in ("source_id", "dataset", "revision", "source_split", "license", "rows_inspected")})
        source_records[-1]["license_evidence"] = evidence
        source_records[-1]["source_file_sha256"] = source["file"]["sha256"]
    write("manifest.jsonl", ("\n".join(canonical_json(r) for r in exported) + "\n").encode(), "application/x-ndjson")
    write("sources.json", canonical_json(source_records).encode(), "application/json")
    write("quarantine.json", canonical_json(quarantined).encode(), "application/json")
    write("model_receipts.json", canonical_json(catalog.get("model_receipts", [])).encode(), "application/json")
    if "label_mapping" in catalog:
        write("label_mapping.json", canonical_json(catalog["label_mapping"]).encode(), "application/json")
    if "quality_calibration" in catalog:
        controls = []
        for control in catalog["calibration_controls"]:
            ref = ArtifactRef.from_dict(control["image"])
            name = f"controls/{control['id']}.png"
            write(name, ctx.store.resolve(ref).read_bytes(), "image/png")
            controls.append({**control, "image": name, "image_sha256": ref.sha256})
        write("quality_calibration.json", canonical_json(catalog["quality_calibration"]).encode(), "application/json")
        write("calibration_controls.json", canonical_json(controls).encode(), "application/json")
    card = ("# Image curation engineering candidate\n\n"
            "Governance: technical_candidate_pending_independent_review. Training and publication are not validated.\n"
            "Privacy and safety assessments are incomplete. Model scores are auxiliary; semantic quality is not calibrated.\n"
            "Splits apply only to this engineering slice; they are not a sealed benchmark.\n"
            "Duplicate grouping uses exact pixels, dHash and available source families; semantic deduplication was not performed.\n\n"
            + "\n".join(f"- {s['dataset']} @ {s['revision']}: {s['license']}; attribution and terms in source_evidence/." for s in source_records))
    write("DATA_CARD.md", card.encode(), "text/markdown")
    metadata = {"schema_version": "dataevolver.curated_images.v1", "count": len(exported), "members": members,
                "governance": "technical_candidate_pending_independent_review", "policy_hash": ctx.policy.compute_hash(),
                "duplicate_method": catalog["duplicate_method"], "split_policy": catalog["split_policy"]}
    index = ctx.write_json("export/dataset.json", metadata)
    output_root = ctx.store.resolve(index).parent
    verified = load_export(output_root)
    report = {"input_count": catalog["input_count"], "eligible_count": catalog["eligible_count"], "selected_count": len(catalog["rows"]),
              "exported_count": len(verified), "quarantined_count": len(quarantined),
              "source_counts": {s["source_id"]: sum(r["source_id"] == s["source_id"] for r in verified) for s in source_records},
              "split_counts": {s: sum(r["split"] == s for r in verified) for s in ("train", "dev", "eval")},
              "source_label_counts": {s["source_id"]: {label: sum(r["source_id"] == s["source_id"] and str(r["source_label"]) == label for r in verified)
                                      for label in sorted({str(r["source_label"]) for r in verified if r["source_id"] == s["source_id"]})}
                                      for s in source_records},
              "source_family_metadata_counts": {s["source_id"]: sum(r["source_id"] == s["source_id"] and bool(r["source_family"]) for r in verified)
                                                for s in source_records},
              "independent_loader_verified": True, "source_integrity_verified": True,
              "privacy_safety_calibrated": False, "training_validated": False}
    report["family_split_scope"] = "observed_exact_perceptual_and_declared_source_families"
    report["global_leakage_verified"] = False
    return {"dataset": index, "report": ctx.write_json("export_report.json", report)}


# Each entry declares kind, input ports, parameter names, output ports and behavior.
STEPS = {
    "images.audit": ("data", {"sources"}, {"max_rows_per_source", "min_side", "max_pixels"}, {"catalog"}, audit),
    "images.families": ("data", {"catalog"}, {"hamming_distance"}, {"catalog"}, families),
    "images.split": ("data", {"catalog"}, {"ratios", "seed"}, {"catalog"}, split),
    "images.select": ("data", {"catalog"}, {"per_source", "seed"}, {"catalog"}, select),
    "images.clean": ("data", {"catalog"}, {"max_side"}, {"catalog"}, clean),
    "images.score": ("model", {"catalog"}, {"texts", "min_margin", "timeout_seconds"}, {"catalog"}, score),
    "images.export": ("data", {"catalog"}, {"require_semantic_match"}, {"dataset", "report"}, export),
}

from . import curation_quality
STEPS.update(curation_quality.STEPS)


def validate_step(node):
    if node.operator not in STEPS or node.operator_version != "1":
        raise ValueError("unregistered data Operator or revision")
    kind, inputs, parameters, _, _ = STEPS[node.operator]
    if node.kind != kind or set(node.inputs) != inputs or set(node.parameters) != parameters:
        raise ValueError("node does not match its registered Operator contract")
    if node.operator in curation_quality.STEPS:
        curation_quality.validate_step(node)
        return
    for name in ("max_rows_per_source", "min_side", "max_pixels", "per_source", "max_side", "timeout_seconds"):
        if name in node.parameters and (type(node.parameters[name]) is not int or node.parameters[name] <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if node.operator == "images.families" and (type(node.parameters["hamming_distance"]) is not int or not 0 <= node.parameters["hamming_distance"] <= 8):
        raise ValueError("unsupported dHash distance")
    if node.operator == "images.score":
        texts = node.parameters["texts"]
        if not isinstance(texts, list) or not 2 <= len(texts) <= 16 or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("model scoring requires bounded positive/negative text prompts")
        if node.max_model_calls != 1:
            raise ValueError("image scoring declares exactly one bounded model batch")
        margin = node.parameters["min_margin"]
        if type(margin) not in (int, float) or not -2 <= margin <= 2:
            raise ValueError("image scoring margin must be bounded")
    if "seed" in node.parameters and (type(node.parameters["seed"]) is not int or node.parameters["seed"] < 0):
        raise ValueError("seed must be a non-negative integer")
    if node.operator == "images.split":
        ratios = node.parameters["ratios"]
        if (not isinstance(ratios, dict) or set(ratios) != {"train", "dev", "eval"} or
                any(type(v) not in (int, float) or v < 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9):
            raise ValueError("split ratios must specify train/dev/eval and sum to one")
    if node.operator == "images.export" and type(node.parameters["require_semantic_match"]) is not bool:
        raise ValueError("export filter must be an explicit boolean")
