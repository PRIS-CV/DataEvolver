"""Label identity, family-aware quotas and a limited objective clarity calibration.

The calibration distinguishes reference images from controlled Gaussian blur.
It supplies no human semantic, privacy, safety or downstream-training labels.
"""
from __future__ import annotations

import math
from collections import defaultdict

from PIL import Image, ImageFilter, ImageOps, ImageStat

from .contracts import identifier, payload_sha256


RANKINGS = frozenset({"hash", "clarity", "review", "hybrid"})
ROLES = frozenset({"development", "calibration", "canary", "reuse", "sealed"})


def validate_mapping(mapping):
    if (set(mapping) != {"schema_version", "sources"} or
            mapping["schema_version"] != "dataevolver.label_mapping.v1" or not mapping["sources"]):
        raise ValueError("invalid versioned label mapping")
    identities = {}
    for source, labels in mapping["sources"].items():
        identifier(source, "label source")
        if not isinstance(labels, dict) or not labels:
            raise ValueError("source label mapping is empty")
        for raw, target in labels.items():
            if not isinstance(raw, str) or set(target) != {"label_id", "name", "evidence"}:
                raise ValueError("label mapping requires identity, name and evidence")
            identifier(target["label_id"], "canonical label")
            if not target["name"].strip() or not target["evidence"].strip():
                raise ValueError("label mapping lacks its declared meaning/evidence")
            meaning = (target["name"], target["evidence"])
            if target["label_id"] in identities and identities[target["label_id"]] != meaning:
                raise ValueError("conflicting cross-source label identity")
            identities[target["label_id"]] = meaning
    return mapping


def map_rows(rows, mapping):
    validate_mapping(mapping)
    result = []
    for row in rows:
        try:
            target = mapping["sources"][row["source_id"]][str(row["source_label"])]
        except KeyError as exc:
            raise ValueError("unmapped source label; an explicit mapping is required") from exc
        result.append({**row, "label_id": target["label_id"], "label_name": target["name"],
                       "label_mapping_hash": payload_sha256(mapping)})
    return result


def validate_quotas(quotas):
    if not isinstance(quotas, dict) or not quotas:
        raise ValueError("source/class quotas must be explicit")
    for source, labels in quotas.items():
        identifier(source, "quota source")
        if not isinstance(labels, dict) or not labels:
            raise ValueError("empty class quota")
        for label, count in labels.items():
            identifier(label, "quota label")
            if type(count) is not int or count <= 0:
                raise ValueError("class quota must be a positive integer")


def clarity_features(image):
    # A fixed interior crop avoids ranking image borders as useful detail.
    gray = ImageOps.exif_transpose(image).convert("L").resize((128, 128), Image.Resampling.LANCZOS)
    crop = gray.crop((8, 8, 120, 120))
    edge = ImageStat.Stat(crop.filter(ImageFilter.FIND_EDGES).crop((1, 1, 111, 111))).var[0]
    contrast = ImageStat.Stat(crop).stddev[0]
    return {"edge_variance": float(edge), "contrast": float(contrast),
            "clarity_proxy": float(math.log1p(edge) / math.log1p(65025))}


def rank_key(row, ranking, seed):
    if ranking not in RANKINGS:
        raise ValueError("unregistered recipe ranking")
    tie = payload_sha256([seed, row["id"]])
    if ranking == "hash":
        return (0., tie)
    clarity = row["quality"]["clarity_proxy"]
    review = row.get("review_score")
    if ranking in {"review", "hybrid"} and (type(review) not in (int, float) or not 0 <= review <= 4):
        raise ValueError("recipe requires a bound reviewer score")
    score = clarity if ranking == "clarity" else review / 4 if ranking == "review" else (clarity + review / 4) / 2
    return (-score, tie)


def select_balanced(rows, quotas, *, ranking="hash", seed=0, min_clarity=0.):
    """Satisfy all class/source quotas with distinct families, or report failure.

    An augmenting assignment avoids false infeasibility when source/class
    strata share a family. Ranking is a preference, never quota authority.
    """
    validate_quotas(quotas)
    for row in rows:
        if not isinstance(row, dict) or any(k not in row for k in ('id','source_id','label_id','family_id')):
            raise ValueError('balanced selection requires row, source, label and family identities')
        quality = row.get('quality', {})
        if not isinstance(quality, dict) or (ranking != 'hash' and 'clarity_proxy' not in quality):
            raise ValueError('balanced ranking requires measured clarity')
        clarity = quality.get('clarity_proxy', 1.)
        if type(clarity) not in (int, float) or not math.isfinite(clarity) or not 0 <= clarity <= 1:
            raise ValueError('balanced ranking requires finite clarity in [0, 1]')
    candidates = defaultdict(dict)
    for row in sorted(rows, key=lambda r: rank_key(r, ranking, seed)):
        if row.get("quality", {}).get("clarity_proxy", 1.) < min_clarity:
            continue
        candidates[row["source_id"], row["label_id"]].setdefault(row.get('canonical_family_id',row['family_id']), row)
    slots = [(source, label, i) for source, labels in sorted(quotas.items())
             for label, count in sorted(labels.items()) for i in range(count)]
    owner, assignment = {}, {}

    def place(slot, visited):
        for family, row in candidates[slot[:2]].items():
            if family in visited:
                continue
            visited.add(family)
            if family not in owner or place(owner[family], visited):
                owner[family], assignment[slot] = slot, row
                return True
        return False

    for slot in sorted(slots, key=lambda s: (len(candidates[s[:2]]), s)):
        if not place(slot, set()):
            raise ValueError(f"class quota infeasible without duplicate families: {slot[0]}/{slot[1]}")
    return [assignment[s] for s in sorted(slots)]


def counts(rows):
    output = {}
    for row in rows:
        labels = output.setdefault(row["source_id"], {})
        labels[row["label_id"]] = labels.get(row["label_id"], 0) + 1
    return output


def fit_blur_calibration(pairs):
    """Fit on explicit fit controls; validate once on separate control families."""
    if not pairs or {p["partition"] for p in pairs} != {"fit", "validation"}:
        raise ValueError("calibration needs fit and validation controls")
    if len({p["family_id"] for p in pairs}) != len(pairs):
        raise ValueError("calibration control families overlap")
    fit = [p for p in pairs if p["partition"] == "fit"]
    thresholds = sorted({0., *(p[k] for p in fit for k in ("reference", "blurred"))})
    thresholds = sorted(set(thresholds + [(a+b)/2 for a, b in zip(thresholds, thresholds[1:])]))

    def accuracy(items, threshold):
        return sum((p["reference"] >= threshold) + (p["blurred"] < threshold) for p in items) / (2 * len(items))

    threshold = max(thresholds, key=lambda t: (accuracy(fit, t), -t))
    validation = [p for p in pairs if p["partition"] == "validation"]
    return {"schema_version": "dataevolver.clarity_calibration.v1", "threshold": threshold,
            "fit_accuracy": accuracy(fit, threshold), "validation_accuracy": accuracy(validation, threshold),
            "scope": "reference_vs_controlled_gaussian_blur_only", "human_gold": False,
            "semantic_quality_calibrated": False, "pairs": pairs}
