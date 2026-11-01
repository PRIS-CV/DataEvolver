"""Independent planning readback and material-variation acceptance.

These checks inspect real artifacts or explicit test fixtures. They never run a
model or establish production quality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import validate_plan
from .planner import read_lines, render_files


DELIVERABLES = ("goal.json", "research.md", "dataset_plan.yaml", "todo.md")


def load_verified_plan(root):
    import tempfile
    root = Path(root)
    status = json.loads((root / "status.json").read_text())
    if status["state"] != "COMPLETE":
        raise ValueError("Planner run is not complete")
    plan = json.loads((root / "plan.json").read_text())
    request = json.loads((root / "request.json").read_text())
    result = validate_plan(plan, request, json.loads((root / "catalog.json").read_text()),
                           json.loads((root / "experience-snapshot.json").read_text()),
                           read_lines(root / "source-evidence.jsonl"))
    for name, digest in status["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("planning file SHA changed: " + name)
    with tempfile.TemporaryDirectory() as temporary:
        render_files(plan, request, Path(temporary))
        if any((Path(temporary) / name).read_bytes() != (root / name).read_bytes() for name in DELIVERABLES):
            raise ValueError("four files do not agree with the canonical plan")
    trace = read_lines(root / "tool-trace.jsonl")
    if any(not any(t["tool"] == name and t["ok"] for t in trace)
           for name in ("read_source", "query_operator_catalog", "query_experiences")):
        raise ValueError("required actual tool record is absent")
    successful_search = any(e.get("type") == "item.completed" and e.get("item", {}).get("type") == "web_search"
                            for e in read_lines(root / "codex-events.jsonl"))
    if not successful_search:
        raise ValueError("actual native search record is absent")
    result.update(files={name: status["files"][name] for name in DELIVERABLES},
                  tool_calls=len(trace), failed_tool_calls=sum(not t["ok"] for t in trace))
    return plan, request, result


def check_material_variation(baseline, changed, changed_request):
    old = {n["operator"] for n in baseline["dataset_plan"]["nodes"]}
    new = {n["operator"] for n in changed["dataset_plan"]["nodes"]}
    nodes = {n["id"]: n for n in changed["dataset_plan"]["nodes"]}
    if "subjects.extract" not in old or "image.compose" not in new:
        raise ValueError("material variation did not change the primary supervision route")
    compose = next(n for n in nodes.values() if n["operator"] == "image.compose")
    ancestors, queue = set(), list(compose["depends_on"])
    while queue:
        identity = queue.pop()
        if identity not in ancestors:
            ancestors.add(identity)
            queue.extend(nodes[identity]["depends_on"])
    if any(nodes[n]["operator"] == "subjects.extract" for n in ancestors):
        raise ValueError("provided references replaced by target-derived extraction")
    supplied = {m["id"] for m in changed_request["materials"]}
    consumed = {b["source"][9:] for n in [compose, *(nodes[a] for a in ancestors)]
                for b in n["input_bindings"] if b["source"].startswith("material:")}
    if consumed != supplied:
        raise ValueError("composition does not consume all supplied material references")
    if baseline["goal"]["construction_mode"] == changed["goal"]["construction_mode"]:
        raise ValueError("supervision construction mode did not change")
    if baseline["todo"] == changed["todo"]:
        raise ValueError("material variation did not change TODOs")
    return {"valid": True, "baseline_operators": sorted(old), "changed_operators": sorted(new),
            "removed_operators": sorted(old-new), "added_operators": sorted(new-old),
            "composition_materials": sorted(consumed), "target_derived_references_required": False}


def research_implementation_coverage(plan):
    paths = {c["path"] for c in plan["research"]["operator_candidates"]}
    required = {"dataflow/operators/core_text/generate/prompted_generator.py",
                "dataflow/statics/pipelines/api_pipelines/text_sft_synthesis_pipeline.py",
                "dataflow/operators/core_vision/eval/image_vqascore_evaluator.py"}
    if not required.issubset(paths):
        raise ValueError("concrete DataFlow/MM implementation analysis remains incomplete")
    return sorted(paths)
