"""Registered pure-data operations using the existing node context."""
from __future__ import annotations

import io

from PIL import Image, ImageFilter

from .contracts import ArtifactRef, payload_sha256
from .quality import (ROLES, RANKINGS, clarity_features, counts, fit_blur_calibration,
                      map_rows, select_balanced, validate_mapping, validate_quotas)


def labels(ctx):
    catalog, mapping = ctx.read("catalog"), ctx.read("mapping")
    catalog["rows"] = map_rows(catalog["rows"], mapping)
    catalog["label_mapping"] = mapping
    return {"catalog": ctx.write_json("mapped.json", catalog)}


def partition(ctx):
    catalog = ctx.read("catalog")
    roles = ctx.parameters["per_label_roles"]
    selected = select_balanced(catalog["rows"], ctx.parameters["quotas"], seed=ctx.parameters["seed"])
    by_label = {}
    for row in selected:
        by_label.setdefault((row["source_id"], row["label_id"]), []).append(row)
    for group in by_label.values():
        group.sort(key=lambda r: payload_sha256([ctx.parameters["seed"], r["id"]]))
        offset = 0
        for role, count in roles.items():
            for row in group[offset:offset+count]:
                row["evaluation_role"] = role
                row["split"] = "train" if role == "development" else "dev" if role in {"calibration", "canary"} else "eval"
                row["protected_gates"]["duplicates"] = True
                row["protected_gates"]["leakage"] = True
            offset += count
        if offset != len(group):
            raise ValueError("role counts do not match the source/class pool quota")
    catalog.update(input_count=sum(s["rows_inspected"] for s in catalog["sources"]),
                   eligible_count=len(catalog["rows"]), rows=selected,
                   selection_policy=dict(ctx.parameters),
                   split_policy={"scope": "frozen_engineering_pool_with_unread_final_role",
                                 "roles": roles, "seed": ctx.parameters["seed"]})
    return {"catalog": ctx.write_json("partitioned.json", catalog)}


def calibrate(ctx):
    catalog = ctx.read("catalog")
    calibration, controls, index = [], [], {}
    for row in catalog["rows"]:
        if row["evaluation_role"] == "sealed":
            continue
        ctx.check()
        ref = ArtifactRef.from_dict(row["image"])
        with Image.open(ctx.store.resolve(ref)) as opened:
            image = opened.convert("RGB")
            row["quality"] = clarity_features(image)
            if row["evaluation_role"] != "calibration":
                continue
            key = row["source_id"], row["label_id"]
            n = index.get(key, 0)
            index[key] = n+1
            partition_name = "fit" if n % 2 == 0 else "validation"
            blurred = image.filter(ImageFilter.GaussianBlur(ctx.parameters["blur_radius"]))
            features = clarity_features(blurred)
            output = io.BytesIO()
            blurred.save(output, "PNG")
        control_id = row["id"] + "-blur"
        control = ctx.write_bytes(f"controls/{control_id}.png", output.getvalue(), "image/png")
        controls.append({"id": control_id, "reference_id": row["id"], "family_id": row["family_id"],
                         "partition": partition_name, "kind": "controlled_gaussian_blur", "image": control.to_dict()})
        calibration.append({"id": row["id"], "image_sha256": ref.sha256, "family_id": row["family_id"],
                            "partition": partition_name, "reference": row["quality"]["clarity_proxy"],
                            "blurred": features["clarity_proxy"], "control_sha256": control.sha256})
    report = fit_blur_calibration(calibration)
    report["blur_radius"] = ctx.parameters["blur_radius"]
    report["minimum_validation_accuracy"] = ctx.parameters["minimum_validation_accuracy"]
    report["passed"] = report["validation_accuracy"] >= ctx.parameters["minimum_validation_accuracy"]
    if not report["passed"]:
        raise ValueError("objective blur calibration failed its frozen validation criterion")
    catalog["quality_calibration"], catalog["calibration_controls"] = report, controls
    return {"catalog": ctx.write_json("quality.json", catalog), "calibration": ctx.write_json("calibration.json", report)}


def balanced(ctx):
    catalog = ctx.read("catalog")
    selected = select_balanced(catalog["rows"], ctx.parameters["quotas"], ranking=ctx.parameters["ranking"],
                               seed=ctx.parameters["seed"], min_clarity=ctx.parameters["min_clarity"])
    catalog.update(input_count=sum(s["rows_inspected"] for s in catalog["sources"]),
                   eligible_count=len(catalog["rows"]), rows=selected, selection_policy=dict(ctx.parameters))
    for row in selected:
        row["protected_gates"]["duplicates"] = True
    return {"catalog": ctx.write_json("balanced.json", catalog)}


STEPS = {
    "images.labels": ("data", {"catalog", "mapping"}, set(), {"catalog"}, labels),
    "images.partition": ("data", {"catalog"}, {"quotas", "per_label_roles", "seed"}, {"catalog"}, partition),
    "images.calibrate_quality": ("data", {"catalog"}, {"blur_radius", "minimum_validation_accuracy"}, {"catalog", "calibration"}, calibrate),
    "images.select_balanced": ("data", {"catalog"}, {"quotas", "ranking", "seed", "min_clarity"}, {"catalog"}, balanced),
}


def validate_step(node):
    parameters = node.parameters
    if node.operator == "images.labels":
        return
    if "seed" in parameters and (type(parameters["seed"]) is not int or parameters["seed"] < 0):
        raise ValueError("selection seed must be non-negative")
    if "quotas" in parameters:
        validate_quotas(parameters["quotas"])
    if node.operator == "images.partition":
        roles = parameters["per_label_roles"]
        if (not isinstance(roles, dict) or set(roles) != ROLES or any(type(v) is not int or v < 1 for v in roles.values())
                or roles["calibration"] < 2 or sum(roles.values()) > 256):
            raise ValueError("pool roles require positive bounded counts and separate calibration controls")
        if any(n != sum(roles.values()) for labels in parameters["quotas"].values() for n in labels.values()):
            raise ValueError("pool quotas and per-label role counts differ")
    if node.operator == "images.calibrate_quality":
        if (type(parameters["blur_radius"]) not in (int, float) or not 1 <= parameters["blur_radius"] <= 16 or
                type(parameters["minimum_validation_accuracy"]) not in (int, float) or
                not .5 <= parameters["minimum_validation_accuracy"] <= 1):
            raise ValueError("invalid objective calibration conditions")
    if node.operator == "images.select_balanced":
        if (parameters["ranking"] not in RANKINGS or type(parameters["min_clarity"]) not in (int, float)
                or not 0 <= parameters["min_clarity"] <= 1):
            raise ValueError("invalid class selection policy")
