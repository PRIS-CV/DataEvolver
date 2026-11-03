"""Standalone candidate loader: no experiment, cache or source-root dependency."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .contracts import file_sha256, payload_sha256


def load_export(root: str | Path, *, decode_roles: set[str] | None = None) -> list[dict]:
    from PIL import Image
    root = Path(root).resolve()
    metadata = json.loads((root / "dataset.json").read_text())
    if metadata.get("schema_version") == "dataevolver.curated_images.v2":
        from .layout_loader import load_layout_export
        return load_layout_export(root)
    if metadata.get("schema_version") != "dataevolver.curated_images.v1":
        raise ValueError("unsupported exported image dataset")
    if metadata.get("governance") != "technical_candidate_pending_independent_review":
        raise ValueError("unsupported candidate governance state")
    members = metadata["members"]
    if not {"manifest.jsonl", "sources.json", "quarantine.json", "model_receipts.json", "DATA_CARD.md"}.issubset(members):
        raise ValueError("export is missing required evidence")
    for relative, digest in members.items():
        path = PurePosixPath(relative)
        resolved = (root / relative).resolve()
        if path.is_absolute() or ".." in path.parts or str(path) != relative or not resolved.is_relative_to(root):
            raise ValueError("export member escapes dataset root")
        if not resolved.is_file() or file_sha256(resolved) != digest:
            raise ValueError("export member is missing or modified")
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line]
    if not rows or len(rows) != metadata["count"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("export membership or row count changed")
    families, images = {}, set()
    sources = {s["source_id"]: s for s in json.loads((root / "sources.json").read_text())}
    mapping = None
    if "label_mapping.json" in members:
        from .quality import validate_mapping
        mapping = validate_mapping(json.loads((root/"label_mapping.json").read_text()))
    for row in rows:
        if row["image"] not in members or row["image_sha256"] != members[row["image"]] or row["image"] in images:
            raise ValueError("manifest image binding or uniqueness changed")
        images.add(row["image"])
        if decode_roles is None or row.get("evaluation_role") in decode_roles:
            with Image.open(root / row["image"]) as opened:
                opened.load()
                if opened.format != "PNG" or opened.mode != "RGB" or list(opened.size) != row["output_size"]:
                    raise ValueError("exported image format or dimensions changed")
        family = row["family_id"]
        if family in families:
            raise ValueError("duplicate family in curated membership")
        families[family] = row["split"]
        if row["split"] not in {"train", "dev", "eval"} or row["lineage"]["split"] != row["split"] or row["lineage"]["family_id"] != family:
            raise ValueError("lineage/split binding changed")
        source = sources[row["source_id"]]
        if row["license"] != source["license"] or row["source_revision"] != source["revision"]:
            raise ValueError("source provenance or license binding changed")
        if row.get("governance", metadata["governance"]) != metadata["governance"]:
            raise ValueError("row governance changed")
        if mapping is not None:
            target = mapping["sources"].get(row["source_id"], {}).get(str(row["source_label"]))
            if (target is None or row.get("label_id") != target["label_id"] or row.get("label_name") != target["name"]
                    or row.get("label_mapping_hash") != payload_sha256(mapping)):
                raise ValueError("exported label mapping or raw-label binding changed")
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    rows = load_export(args.root)
    print(json.dumps({"verified": True, "count": len(rows), "splits": sorted({r["split"] for r in rows})}))


if __name__ == "__main__":
    main()
