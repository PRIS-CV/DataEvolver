"""Read-only capability discovery and conservative existing-experience reuse."""
from __future__ import annotations

import hashlib
import json
import ast
from pathlib import Path


# These are task contracts and reuse candidates, not assertions of runnable routes.
OPERATIONS = [
    ("scene.plan", "Training goal and coverage to natural scenes and subject lists", "training_goal", "scene_plan", None),
    ("image.generate", "Generate a complete natural scene or a background", "scene_plan", "image", "workflows/multimodal/generators/qwen_image.py"),
    ("subjects.resolve", "Bind requested subject IDs to observed instances", "image,subject_list", "instance_bindings", None),
    ("subjects.segment", "Return all selected instance masks with stable IDs", "image,instance_bindings", "mask_list", "workflows/stages/sam_segment.py"),
    ("subjects.extract", "Extract variable-length reference assets with transforms", "image,mask_list", "subject_references", None),
    ("background.complete", "Remove selected subjects and complete occluded background", "image,mask_list", "background_visible,background_completed", "workflows/multimodal/generators/qwen_image_edit.py"),
    ("image.compose", "Construct targets using provided background and independent references", "background,subject_references,instruction", "target_image", None),
    ("sample.assemble", "Bind background, references, instruction and target as supervision", "background,subject_references,target_image", "training_record", None),
    ("sample.review", "Check identity binding, removal, preservation and completion", "training_record", "review_evidence", "annotation/vlm_review_stage.py"),
    ("dataset.export", "Export portable media and family-disjoint manifests", "training_records", "dataset_bundle", "dataset/g2_export.py"),
    ("events.plan", "Separate initial state, intervention and expected events", "training_goal", "event_plan", None),
    ("video.generate", "H3 FL2VA initial-frame and action conditioned video", "initial_frame,action_condition", "video", "workflows/multimodal/generators/wan_t2v.py"),
    ("subjects.track", "Track bound subject identities across actual video frames", "video,instance_bindings", "tracks", None),
    ("video.inspect", "Locate events and record evidence, uncertainty and timestamps", "video,tracks,event_plan", "event_observations", None),
    ("video.assemble", "Bind initial frame and action to future video and auxiliary evidence", "initial_frame,action_condition,video,event_observations", "video_training_record", None),
]


def operator_catalog(package_root: Path, model_config: dict | None = None, *, registry=None) -> dict:
    rows = []
    for name, description, inputs, outputs, relative in OPERATIONS:
        source = package_root / relative if relative else None
        exists = bool(source and source.is_file())
        rows.append({
            "operator": name, "version": "1", "description": description,
            "inputs": inputs.split(","), "outputs": outputs.split(","),
            "status": "needs_implementation", "verification_status": "designed",
            "reuse_candidate": relative if exists else None,
            "reuse_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if exists else None,
            "adaptation": (
                "Existing SAM3 helper selects one best mask; implement instance-list binding; no silent color fallback."
                if name == "subjects.segment" else
                "Existing adapter targets older models/formats; verify the requested model and task I/O before use."
                if exists else "No task-specific implementation verified."
            ),
            "allowed_experience_fields": ["parameters", "checks"],
        })
    if registry is not None:
        for row in registry.catalog()['operators']:
            spec, declaration = row['spec'], row['contract']
            rows.append({'operator': spec['operator_id'], 'version': spec['version'],
                'description': 'Bound registered operator declaration',
                'inputs': sorted(declaration['input_schemas']) if declaration else [],
                'outputs': sorted(declaration['output_schemas']) if declaration else [],
                'parameters': sorted(declaration['parameter_schema'].get('properties', {})) if declaration else [],
                'status': 'implemented', 'verification_status': 'registry_binding_verified',
                'spec_sha256': registry.step(spec['operator_id']).spec.sha256() if row.get('binding_kind') == 'node_factory' else registry.get(spec['operator_id']).spec.sha256(),
                'contract': declaration, 'production_qualification': row['production_qualification'],
                'allowed_experience_fields': ['parameters', 'checks']})
        return {'schema_version': 'dataevolver.planning_catalog.v1', 'operators': rows,
                'models': model_config or {}, 'production_execution_performed': False}
    # Compatibility view for historical/source-only callers; it grants no binding.
    for relative in ("dataset/curation.py", "dataset/curation_quality.py"):
        source = package_root / relative
        if not source.is_file():
            continue
        tree = ast.parse(source.read_text())
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or not any(isinstance(t, ast.Name) and t.id == "STEPS" for t in statement.targets):
                continue
            if not isinstance(statement.value, ast.Dict):
                continue
            for key, value in zip(statement.value.keys, statement.value.values):
                if not isinstance(value, ast.Tuple) or len(value.elts) != 5:
                    continue
                name = ast.literal_eval(key)
                kind, inputs, parameters, outputs = [ast.literal_eval(x) for x in value.elts[:4]]
                rows.append({"operator": name, "version": "1", "description": f"Existing registered {kind} data step",
                             "inputs": sorted(inputs), "outputs": sorted(outputs), "parameters": sorted(parameters),
                             "status": "implemented", "verification_status": "source_registration_verified",
                             "reuse_candidate": relative, "reuse_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "adaptation": "Existing image catalog I/O; does not by itself implement subject-list or video supervision.",
                             "allowed_experience_fields": ["parameters", "checks"]})
    return {"schema_version": "dataevolver.planning_catalog.v1", "operators": rows,
            "models": model_config or {}, "production_execution_performed": False}


def experience_snapshot(path: Path | None, task: str, models: dict, catalog: dict,
                        *, constraints=None, model_versions=None) -> dict:
    """Adapt CapabilityStore data without changing it or treating candidates as promoted."""
    from dataevolver.harness.capabilities import CapabilityStore
    allowed = {row["operator"] for row in catalog["operators"]}
    accepted, excluded = [], []
    if path is not None and path.is_file():
        # Parse through the existing store's public API; never append/promote here.
        typed = {x.capability_id: x for x in CapabilityStore(path).list_capabilities()}
        for raw in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            identity = str(raw["capability_id"])
            scope = raw.get("planning_scope", raw.get("intervention", {}).get("parameters", {}).get("planning_scope", {}))
            reason = None
            if typed[identity].status.value != "PROMOTED":
                reason = "not_promoted"
            elif not raw.get("validation_evidence"):
                reason = "missing_validation_evidence"
            elif scope.get("task") != task:
                reason = "incompatible_or_unspecified_task"
            elif scope.get("models") != models:
                reason = "incompatible_or_unspecified_models"
            elif not model_versions or scope.get("model_versions") != model_versions:
                reason = "model_versions_not_frozen_or_incompatible"
            elif constraints and scope.get("compatible_constraints") != constraints:
                reason = "constraint_compatibility_unknown_or_conflict"
            elif scope.get("operator") not in allowed or scope.get("operator_version") != next(
                    (r.get('version', '1') for r in catalog['operators'] if r['operator'] == scope.get('operator')), None):
                reason = "incompatible_operator"
            elif not scope.get("changes") or any(
                c.get("field", "").split(".", 1)[0] not in {"parameters", "checks"}
                for c in scope["changes"]
            ):
                reason = "unsafe_or_missing_changes"
            if reason:
                excluded.append({"id": identity, "reason": reason})
            else:
                accepted.append({"id": identity, "version": typed[identity].version,
                                 "scope": scope, "validation_evidence": raw["validation_evidence"]})
    return {"schema_version": "dataevolver.planning_experiences.v1",
            "task": task, "records": accepted, "excluded": excluded,
            "empty_reason": (None if accepted else "no_compatible_promoted_experience") if path and path.is_file() else "experience_store_not_configured",
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None,
            "read_only": True}
