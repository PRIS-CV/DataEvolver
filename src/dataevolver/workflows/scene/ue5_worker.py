"""Use the extracted native UE5 batch backend as one registered Harness effect."""
import argparse
import json
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json
from .ue5 import render_batch


def verify_render_index(path):
    path = Path(path).resolve()
    index = json.loads(path.read_text())
    if index.get("schema_version") != "dataevolver.ue5_render.v1":
        raise ValueError("expected a native UE5 render index")
    manifest = Path(index["manifest"]["path"]).resolve()
    batch = Path(index["batch"]).resolve()
    if (not manifest.is_relative_to(path.parent) or not batch.is_relative_to(path.parent) or
            file_sha256(manifest) != index["manifest"]["sha256"]):
        raise ValueError("native UE5 manifest or batch location changed")
    plan = render_batch.validate_manifest(json.loads(manifest.read_text()))
    contract = json.loads((batch / "contract.json").read_text())
    progress = json.loads((batch / "batch-exit.json").read_text())
    if (contract != render_batch.package_contract(plan) or progress["status"] != "completed"
            or progress.get("contract_sha256") != contract["sha256"]):
        raise ValueError("native UE5 batch is incomplete or belongs to another plan")
    for name, digest in contract["file_hashes"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError("native UE5 dependency changed")
    rows = {row["id"]: row for row in progress["tasks"]}
    if len(rows) != len(progress["tasks"]) or set(rows) != {task["id"] for task in plan["tasks"]}:
        raise ValueError("native UE5 task inventory changed")
    for task in plan["tasks"]:
        row = rows[task["id"]]
        if row["status"] != "completed" or not Path(row["run"]).resolve().is_relative_to(batch):
            raise ValueError("native UE5 task did not complete inside its batch")
        render_batch.verified_resume(plan, task, row["run"])
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    source = request["inputs"]["render_manifest"]
    if request["stage"] != "ue5" or file_sha256(Path(source["path"])) != source["sha256"]:
        raise ValueError("render request or manifest changed")
    index = args.output / request["outputs"]["render_index"]
    index.parent.mkdir(parents=True, exist_ok=True)
    manifest = index.parent / "render-manifest.json"
    manifest.write_bytes(Path(source["path"]).read_bytes())
    batch = index.parent / "render"
    code = render_batch.run_batch(json.loads(manifest.read_text()), batch)
    if code != 0:
        raise RuntimeError("native UE5 batch failed; retain its attempts and process receipts")
    atomic_write_json(index, {"schema_version": "dataevolver.ue5_render.v1",
        "manifest": {"path": str(manifest.resolve()), "sha256": file_sha256(manifest)}, "batch": str(batch.resolve())})
    verify_render_index(index)
    atomic_write_json(args.output / "result.json", {"status": "completed", "request_sha256": payload_sha256(request)})


if __name__ == "__main__":
    main()
