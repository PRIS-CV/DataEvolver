"""Bounded read-only package inspection, also sent to the configured SSH Python."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys

JSON_LIMIT = 4 * 1024 * 1024
LINE_LIMIT = 512 * 1024
IMAGE_LIMIT = 8 * 1024 * 1024
TABLES = {"manifest.jsonl", "quarantine.jsonl", "reviews.jsonl"}


def contained(root, name):
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("A nonempty relative path is required")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Use a path relative to the configured dataset root")
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root):
        raise ValueError("Path escapes the configured dataset root")
    return result


def bounded_bytes(path, limit):
    if not path.is_file():
        raise ValueError("Expected a regular file")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"File exceeds the {limit}-byte viewer limit")
    return value


def inspect_package(request):
    root = Path(request["root"])
    if not root.is_absolute():
        raise ValueError("Configure an absolute dataset directory on the selected host")
    root = root.resolve(strict=True)
    package = contained(root, request["package"])
    if not package.is_dir():
        raise ValueError("Package must be a directory")
    metadata_name = "dataset.json" if (package / "dataset.json").exists() else "bundle.json"
    metadata_bytes = bounded_bytes(contained(package, metadata_name), JSON_LIMIT)
    metadata = json.loads(metadata_bytes)
    expected_schema = "dataevolver.curated_images.v2" if metadata_name == "dataset.json" else "dataevolver.source_bundle.v1"
    if metadata.get("schema_version") != expected_schema:
        raise ValueError("Unsupported dataset package schema")
    members = metadata.get("members", {})
    result = {"package": request["package"], "metadata": metadata,
              "metadataSha256": hashlib.sha256(metadata_bytes).hexdigest(),
              "validation": "viewer_only_not_independent_loader_validation"}
    if request.get("image"):
        name = request["image"]
        types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        media_type = types.get(Path(name).suffix.lower())
        if not name.startswith("images/") or not media_type or name not in members:
            raise ValueError("Select an indexed exported image")
        image = bounded_bytes(contained(package, name), IMAGE_LIMIT)
        if hashlib.sha256(image).hexdigest() != members[name]:
            raise ValueError("Image member hash mismatch")
        return {"dataUrl": f"data:{media_type};base64," + base64.b64encode(image).decode()}
    table = request.get("table")
    if not table:
        return result
    if metadata_name != "dataset.json" or table not in TABLES or table not in members:
        raise ValueError("Select manifest.jsonl, quarantine.jsonl or reviews.jsonl from a curated export")
    cursor = request.get("cursor", 0)
    if type(cursor) is not int or cursor < 0:
        raise ValueError("Invalid byte cursor")
    path = contained(package, table)
    if not path.is_file():
        raise ValueError("Expected a regular table file")
    rows = []
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        if cursor > size:
            raise ValueError("Cursor exceeds file size; reload the package")
        if cursor:
            stream.seek(cursor - 1)
            if stream.read(1) != b"\n":
                raise ValueError("Cursor is not at a record boundary; reload the package")
        stream.seek(cursor)
        consumed = 0
        while len(rows) < 20 and consumed < JSON_LIMIT:
            line = stream.readline(LINE_LIMIT + 1)
            if not line:
                break
            if len(line) > LINE_LIMIT:
                raise ValueError("Record exceeds the viewer line limit")
            consumed += len(line)
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object record")
                rows.append(row)
        next_cursor = stream.tell()
    result.update(rows=rows, table=table, cursor=cursor,
                  nextCursor=next_cursor if next_cursor < size else None, fileSize=size)
    return result


if __name__ == "__main__":
    try:
        request = json.loads(sys.argv[1])
        print(json.dumps(inspect_package(request), ensure_ascii=True, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
