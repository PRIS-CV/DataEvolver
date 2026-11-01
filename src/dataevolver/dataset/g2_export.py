"""Portable pair bundle preserving the original TaskSpec and manifest bytes."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .contracts import ArtifactRef, ArtifactStore, file_sha256
from .g2_pipeline import load_pair_manifest


def export_pairs(manifest, store, destination, *, metadata_files=None):
    manifest, destination = Path(manifest), Path(destination)
    if destination.exists():
        raise ValueError("refusing to overwrite a portable G2 export")
    load_pair_manifest(manifest, store, include_rejected=True)
    value = json.loads(manifest.read_text())
    destination.mkdir(parents=True)
    stores = {}
    for row in value["pairs"]:
        refs = [row["task"]["source"]] + ([row["output"]] if row["output"] else [])
        for raw in refs:
            ref = ArtifactRef.from_dict(raw)
            source = store.resolve(ref)
            stores[ref.store] = f"stores/{ref.store}"
            target = destination/stores[ref.store]/ref.path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            if file_sha256(target) != ref.sha256:
                raise ValueError("portable artifact bytes changed")
    shutil.copy2(manifest, destination/"pairs.json")
    for name, path in (metadata_files or {}).items():
        if Path(name).name != name or name in {"pairs.json", "stores.json", "checksums.json"}:
            raise ValueError("invalid portable metadata filename")
        shutil.copy2(path, destination/name)
    (destination/"stores.json").write_text(json.dumps(stores, indent=2))
    roots = {k: destination/v for k, v in stores.items()}
    pairs = load_pair_manifest(destination/"pairs.json", ArtifactStore(roots), include_rejected=True)
    hashes = {p.relative_to(destination).as_posix(): file_sha256(p) for p in destination.rglob("*") if p.is_file()}
    (destination/"checksums.json").write_text(json.dumps(hashes, indent=2))
    return {"bundle": str(destination), "manifest_sha256": file_sha256(destination/"pairs.json"),
            "tasks": value["summary"]["tasks"], "loadable_pairs": len(pairs), "artifact_files": len(hashes)}


def load_bundle(destination, *, include_rejected=False):
    root = Path(destination).resolve()
    hashes = json.loads((root/"checksums.json").read_text())
    for relative, digest in hashes.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError("portable bundle hash mismatch")
    stores = json.loads((root/"stores.json").read_text())
    roots = {}
    for key, relative in stores.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("portable store escapes bundle")
        roots[key] = path
    return load_pair_manifest(root/"pairs.json", ArtifactStore(roots), include_rejected=include_rejected)
