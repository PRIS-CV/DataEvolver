#!/usr/bin/env python3
"""Aggregate high-quality render control states into warm-start priors."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EXCLUDED_NUMERIC_FIELDS = {"object.yaw_deg"}


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_category_map(objects_file: Optional[Path]) -> Dict[str, str]:
    if not objects_file:
        return {}
    payload = read_json(objects_file, default=[]) or []
    return {
        str(item.get("id")): str(item.get("category"))
        for item in payload
        if isinstance(item, dict) and item.get("id") and item.get("category")
    }


def flatten_numeric(payload: dict, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_numeric(value, path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if path not in EXCLUDED_NUMERIC_FIELDS:
                out[path] = float(value)
    return out


def set_nested(payload: dict, dotted_path: str, value: float) -> None:
    current = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def build_nested(values: Dict[str, float]) -> dict:
    payload: dict = {}
    for path, value in sorted(values.items()):
        set_nested(payload, path, value)
    return payload


def median_values(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if value is not None:
                buckets[key].append(float(value))
    return {
        key: float(statistics.median(values))
        for key, values in buckets.items()
        if values
    }


def split_material_fields(values: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    material = {}
    other = {}
    for key, value in values.items():
        if key.startswith("material."):
            material[key] = value
        else:
            other[key] = value
    return other, material


def discover_records(source_roots: List[Path], min_hybrid_score: float, category_map: Dict[str, str]) -> List[dict]:
    records = []
    for root in source_roots:
        for agent_path in sorted(root.rglob("agent_round*.json")):
            agent = read_json(agent_path, default={}) or {}
            try:
                score = float(agent.get("hybrid_score"))
            except (TypeError, ValueError):
                continue
            if score < min_hybrid_score:
                continue
            state_path = Path(str(agent.get("state_out") or ""))
            if not state_path.is_absolute():
                state_path = (agent_path.parent / state_path).resolve()
            state = read_json(state_path, default=None)
            if not isinstance(state, dict):
                continue
            obj_id = str(agent.get("obj_id") or agent_path.parent.name.split("_yaw")[0])
            records.append(
                {
                    "obj_id": obj_id,
                    "category": category_map.get(obj_id, "unknown"),
                    "hybrid_score": score,
                    "agent_path": str(agent_path),
                    "state_path": str(state_path),
                    "numeric": flatten_numeric(state),
                }
            )
    return records


def build_prior_library(
    source_roots: List[Path],
    objects_file: Optional[Path],
    output_path: Path,
    min_hybrid_score: float,
) -> dict:
    category_map = load_category_map(objects_file)
    records = discover_records([root.resolve() for root in source_roots], min_hybrid_score, category_map)
    if not records:
        raise FileNotFoundError("No high-quality control states found for render prior library")

    all_non_material_rows = []
    all_material_rows = []
    category_material_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for record in records:
        non_material, material = split_material_fields(record["numeric"])
        all_non_material_rows.append(non_material)
        all_material_rows.append(material)
        category_material_rows[record["category"]].append(material)

    global_non_material = median_values(all_non_material_rows)
    global_material = median_values(all_material_rows)
    global_control = build_nested({**global_non_material, **global_material})

    categories = {}
    for category, material_rows in sorted(category_material_rows.items()):
        category_material = median_values(material_rows)
        if not category_material:
            category_material = global_material
        categories[category] = {
            "record_count": len(material_rows),
            "control_state": build_nested({**global_non_material, **category_material}),
            "material_field_count": len(category_material),
        }

    payload = {
        "schema_version": "render_prior_library_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_roots": [str(root.resolve()) for root in source_roots],
        "objects_file": str(objects_file.resolve()) if objects_file else None,
        "min_hybrid_score": min_hybrid_score,
        "record_count": len(records),
        "excluded_numeric_fields": sorted(EXCLUDED_NUMERIC_FIELDS),
        "global": {
            "control_state": global_control,
            "non_material_field_count": len(global_non_material),
            "material_field_count": len(global_material),
        },
        "categories": categories,
        "source_records": [
            {
                "obj_id": record["obj_id"],
                "category": record["category"],
                "hybrid_score": record["hybrid_score"],
                "state_path": record["state_path"],
                "agent_path": record["agent_path"],
            }
            for record in records
        ],
    }
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build render prior library from high-quality yaw000 states")
    parser.add_argument("--source-root", action="append", required=True, help="Evolution/bootstrap root; may be repeated")
    parser.add_argument("--objects-file", default=None)
    parser.add_argument("--min-hybrid-score", type=float, default=0.78)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_prior_library(
        source_roots=[Path(item) for item in args.source_root],
        objects_file=Path(args.objects_file) if args.objects_file else None,
        output_path=Path(args.output),
        min_hybrid_score=args.min_hybrid_score,
    )
    print(json.dumps({k: payload[k] for k in ("schema_version", "record_count", "min_hybrid_score")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
