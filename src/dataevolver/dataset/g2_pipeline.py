"""Four bounded visual operations and an export that loads without a campaign."""
from __future__ import annotations

import io
import json
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.dataset.contracts import ArtifactRef, canonical_json
from dataevolver.dataset.g2_contracts import CONSTRAINTS, G2Task, compile_prompt
from dataevolver.harness.state_store import atomic_write_json


EDIT_SCHEMA = {"type": "object", "required": ["artifact", "actual_parameters", "image_size", "runtime"],
               "additionalProperties": False, "properties": {"artifact": {"type": "object"},
               "actual_parameters": {"type": "object"}, "image_size": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
               "runtime": {"type": "object"}}}
CHECK_SCHEMA = {"type": "object", "required": ["checks"], "additionalProperties": False,
               "properties": {"checks": {"type": "object", "additionalProperties": False, "required": list(CONSTRAINTS),
                   "properties": {k: {"type": "object", "required": ["status", "evidence"], "additionalProperties": False,
                       "properties": {"status": {"enum": ["pass", "fail", "uncertain"]},
                                      "evidence": {"type": "string", "maxLength": 500}}} for k in CONSTRAINTS}}}}
REGION_SCHEMA = {"type": "object", "required": ["observation", "answer"], "additionalProperties": False,
                 "properties": {"observation": {"type": "string", "maxLength": 800},
                                "answer": {"enum": ["supports_change", "supports_keep", "uncertain"]}}}


def edit_request(task, recipe, config, request_id):
    parameters = {k: v for k, v in config["inference"].items() if k != "timeout_seconds"}
    parameters.update(prompt=compile_prompt(task, recipe, config["allowed_recipe_values"][task.task_id]), seed=task.seed)
    return ModelRequest(request_id=request_id, role="visual.edit",
                        messages=({"role": "user", "content": canonical_json(parameters)},),
                        response_schema=EDIT_SCHEMA, artifacts=(task.source,), seed=task.seed,
                        max_tokens=1, timeout_seconds=config["inference"]["timeout_seconds"])


class VisualOperations:
    """An operation set inside the existing Harness Operator, not a controller."""
    def __init__(self, config, store, gateway, work_dir, budget=None):
        self.config, self.store, self.gateway = config, store, gateway
        self.work = Path(work_dir)
        self.budget = budget
        self.requests = []
        self.attempts = []

    def invoke(self, request, *, purpose, task):
        # Write the exact request/purpose before any real effect can occur.
        path = self.work / "requests" / f"{request.request_id}.json"
        if self.budget is not None:
            self.budget.check()
            if request.timeout_seconds > self.budget.remaining_seconds():
                raise RuntimeError("remaining batch time cannot cover the frozen request timeout")
            if not path.exists():
                self.budget.reserve_operator(request.request_id)
        role_limits = {"visual.edit": self.config["budget"]["max_edit_calls"],
                       "visual.inspect_region": self.config["budget"]["max_probe_calls"]}
        usage = self.gateway.audit_usage()["attempts"]
        if request.role in role_limits and not any(x["request_id"] == request.request_id for x in usage):
            if sum(x["attempts"] for x in usage if x["role"] == request.role) >= role_limits[request.role]:
                raise RuntimeError("frozen role invocation budget exhausted")
        value = {"request": request.to_dict(), "purpose": purpose, "task_hash": task.compute_hash(),
                 "candidate_eligible": purpose == "candidate"}
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("request identity or artifact purpose changed")
        atomic_write_json(path, value)
        self.attempts.append(value)
        try:
            receipt = self.gateway.invoke(request)
        except Exception as exc:
            value["error_code"] = getattr(exc, "code", type(exc).__name__)
            raise
        value["receipt_reused"] = receipt.reused
        self.requests.append({"request": request.to_dict(), "purpose": purpose,
                              "receipt_hash": receipt.without_reuse_hash()})
        return receipt

    def edit(self, task, recipe, request_id):
        request = edit_request(task, recipe, self.config, request_id)
        receipt = self.invoke(request, purpose="candidate", task=task)
        ref = ArtifactRef.from_dict(receipt.output["artifact"])
        if ref.store != self.config["dataset_store"]:
            raise ValueError("editing output escapes dataset store")
        self.store.resolve(ref)
        if receipt.output["actual_parameters"] != json.loads(request.messages[0]["content"]):
            raise ValueError("editing backend parameters differ from request")
        return ref, receipt

    def compare_preservation(self, task, output):
        """Fixed ROI pixel check; semantic preservation still needs VLM review."""
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(self.store.resolve(task.source)) as source, Image.open(self.store.resolve(output)) as result:
            source = ImageOps.exif_transpose(source).convert("RGB")
            result = ImageOps.exif_transpose(result).convert("RGB")
            if source.size != result.size:
                raise ValueError("source/output dimensions differ; implicit alignment is forbidden")
            before, after = np.asarray(source, dtype=np.float32)/255, np.asarray(result, dtype=np.float32)/255
        h, w = before.shape[:2]
        x0, y0, x1, y1 = task.target_box
        mask = np.zeros((h, w), dtype=bool)
        mask[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = True
        difference = np.abs(before-after).mean(axis=2)
        target_mae, outside_mae = float(difference[mask].mean()), float(difference[~mask].mean())
        evaluation = self.config["evaluation"]
        return {"operator": "visual.compare_preservation", "task_hash": task.compute_hash(),
                "source_hash": task.source.sha256, "output_hash": output.sha256,
                "target_box": list(task.target_box), "target_mae": target_mae, "outside_mae": outside_mae,
                "target_changed": target_mae >= evaluation["min_target_mae"],
                "outside_preserved": outside_mae <= evaluation["max_outside_mae"],
                "thresholds": {k: evaluation[k] for k in ("min_target_mae", "max_outside_mae")},
                "calibration_status": evaluation["calibration_status"]}

    def review_constraints(self, task, output, request_id):
        prompt = {"operation": "visual.review_constraints", "image_order": ["original", "edited"],
                  "original_task": task.original_instruction, "target": task.target, "required_color": task.color,
                  "constraints": {"target_color": "Only the specified target has the requested color.",
                      "target_shape": "Target identity, shape, geometry and position are preserved.",
                      "other_objects": "Every non-target object is preserved; pass if none exist.",
                      "background": "Background, framing and lighting remain consistent.",
                      "no_artifacts": "No added text, logos or visible editing artifacts."},
                  "rule": "Judge against the ORIGINAL task. Do not infer success from a prompt. Evidence must be short and visible. Use uncertain if ambiguous."}
        evaluation = self.config["evaluation"]
        request = ModelRequest(request_id=request_id, role="visual.review_constraints",
                 messages=({"role": "user", "content": canonical_json(prompt)},), response_schema=CHECK_SCHEMA,
                 artifacts=(task.source, output), seed=task.seed,
                 max_tokens=evaluation["review_max_tokens"], timeout_seconds=evaluation["review_timeout_seconds"])
        receipt = self.invoke(request, purpose="final_check", task=task)
        return receipt.output, receipt

    def inspect_region(self, task, output, request_id, question, box):
        if not isinstance(question, str) or not question.strip() or len(question) > 300:
            raise ValueError("probe requires a bounded question")
        if list(box) not in [list(task.target_box), [0, 0, 1, 1]]:
            raise ValueError("probe region is outside the frozen region allowlist")
        from PIL import Image
        crops = []
        root = self.store.roots[self.config["dataset_store"]]
        for ref in (task.source, output):
            with Image.open(self.store.resolve(ref)) as image:
                w, h = image.size
                crop = image.convert("RGB").crop(tuple(int(x*y) for x, y in zip(box, (w, h, w, h))))
                data = io.BytesIO(); crop.save(data, format="PNG")
            import hashlib
            digest = hashlib.sha256(data.getvalue()).hexdigest()
            relative = f"observations/{digest}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data.getvalue())
            crops.append(ArtifactRef(self.config["dataset_store"], relative, digest, "image/png"))
        evaluation = self.config["evaluation"]
        request = ModelRequest(request_id=request_id, role="visual.inspect_region",
                 messages=({"role": "user", "content": canonical_json({"question": question,
                     "original_task": task.original_instruction, "image_order": ["original_crop", "edited_crop"]})},),
                 response_schema=REGION_SCHEMA, artifacts=tuple(crops), seed=task.seed,
                 max_tokens=evaluation["review_max_tokens"], timeout_seconds=evaluation["review_timeout_seconds"])
        receipt = self.invoke(request, purpose="observation_only", task=task)
        return {"question": question, "region": list(box), "observation": receipt.output,
                "crop_artifacts": [r.to_dict() for r in crops], "candidate_eligible": False,
                "receipt_hash": receipt.without_reuse_hash()}


def classify_pair(pixel, review):
    checks = review["checks"]
    if not pixel["target_changed"] or not pixel["outside_preserved"] or any(x["status"] == "fail" for x in checks.values()):
        return "fail"
    if any(x["status"] != "pass" for x in checks.values()):
        return "uncertain"
    return "pass"


def load_pair_manifest(path, store, *, include_rejected=False):
    """Independent loader: no controller, model, process or runtime imports."""
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != "dataevolver.g2_pairs.v1":
        raise ValueError("unsupported pair manifest")
    seen, pairs = set(), []
    for row in value["pairs"]:
        if row["task_id"] in seen:
            raise ValueError("duplicate pair identity")
        seen.add(row["task_id"])
        task = G2Task.from_dict(row["task"])
        if task.task_id != row["task_id"] or task.compute_hash() != row["task_hash"]:
            raise ValueError("pair original task identity changed")
        if row["status"] not in {"pass", "fail", "uncertain", "failed", "not_run"}:
            raise ValueError("invalid pair status")
        source_path = store.resolve(task.source)
        if row["output"] is None:
            if row["status"] not in {"failed", "not_run"}:
                raise ValueError("successful pair has no edited image")
            continue
        output = ArtifactRef.from_dict(row["output"])
        output_path = store.resolve(output)
        if row["status"] == "pass" and classify_pair(row["preservation"], row["review"]) != "pass":
            raise ValueError("export falsely marks a failed constraint as passed")
        if include_rejected or row["status"] == "pass":
            pairs.append({**row, "source_path": str(source_path), "output_path": str(output_path)})
    if len(seen) != value["summary"]["tasks"]:
        raise ValueError("manifest task denominator changed")
    return pairs
