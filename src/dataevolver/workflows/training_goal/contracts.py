"""Independent planning contracts: media relationships are not Harness GoalSpec."""
from __future__ import annotations

import json
import re


def obj(**fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def arr(item):
    return {"type": "array", "items": item}


S = {"type": "string"}
STRINGS = arr(S)
PORT = obj(name=S, kind=S, description=S)
CHANGE = obj(field=S, before_json=S, after_json=S)
PLAN_SCHEMA = obj(
    goal=obj(task_type=S, capability=S, training_inputs=arr(PORT), training_targets=arr(PORT),
             supervision_kind=S, construction_mode=S, constraints=arr(obj(id=S, text=S)),
             assumptions=arr(obj(id=S, text=S, basis=S)),
             material_usage=arr(obj(material_id=S, role=S, preservation=S))),
    research=obj(
        sources=arr(obj(evidence_id=S, title=S, url=S, version=S, finding=S,
                        design_impact=S, node_ids=STRINGS)),
        operator_candidates=arr(obj(operation=S, repository=S, path=S, revision=S,
                                    input_adaptation=S, output_adaptation=S,
                                    verification_status={"type": "string", "enum": ["candidate"]}, evidence_id=S, node_ids=STRINGS)),
        revisions=arr(obj(field=S, before=S, after=S, reason=S, evidence_ids=STRINGS))),
    dataset_plan=obj(
        nodes=arr(obj(id=S, operator=S, status=S, depends_on=STRINGS,
                      input_bindings=arr(obj(port=S, source=S)), outputs=arr(PORT),
                      parameters_json=S, evidence_ids=STRINGS, experience_ids=STRINGS, rationale=S)),
        model_roles=arr(obj(role=S, model=S, deployment_status=S)),
        checks=arr(obj(id=S, criterion=S, evidence_method=S, failure_action=S)),
        coverage_json=S, split_strategy=S, sample_example_json=S),
    todo=arr(obj(id=S, title=S, depends_on=STRINGS, node_ids=STRINGS,
                 deliverable=S, acceptance=S, status=S)),
    experiences_used=arr(obj(id=S, version=S, node_id=S, why=S, changes=arr(CHANGE))),
)


def validate_shape(value, schema, path="plan"):
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["properties"]):
            raise ValueError(f"{path}: missing or unknown object fields")
        for key, child in schema["properties"].items():
            validate_shape(value[key], child, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        for i, item in enumerate(value):
            validate_shape(item, schema["items"], f"{path}[{i}]")
    elif kind == "string" and not isinstance(value, str):
        raise ValueError(f"{path}: expected string")


def _unique(rows, key, description):
    keys = [row[key] for row in rows]
    if any(not k for k in keys) or len(keys) != len(set(keys)):
        raise ValueError(f"{description}: empty or repeated identity")
    return set(keys)


def _lookup(value, path):
    for key in path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            raise ValueError("input references missing goal/config field: " + path)
    return value


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def validate_plan(plan, request, catalog, snapshot, tool_evidence):
    validate_shape(plan, PLAN_SCHEMA)
    goal, data, research = plan["goal"], plan["dataset_plan"], plan["research"]
    if goal["constraints"] != request.get("constraints", []):
        raise ValueError("user constraints changed or were replaced by assumptions")
    materials = request.get("materials", [])
    material_ids = _unique(materials, "id", "materials") if materials else set()
    used = _unique(goal["material_usage"], "material_id", "material usage") if goal["material_usage"] else set()
    if material_ids != used:
        raise ValueError("provided/declared materials were dropped or invented")
    if not goal["training_inputs"] or not goal["training_targets"]:
        raise ValueError("training input/target relationship is absent")
    evidence = {e["evidence_id"]: e for e in tool_evidence if e.get("ok")}
    source_ids = _unique(research["sources"], "evidence_id", "research sources")
    if len(source_ids) < 3:
        raise ValueError("research requires at least three actually read primary sources")
    for source in research["sources"]:
        actual = evidence.get(source["evidence_id"])
        if not actual or source["url"] not in {actual["url"], actual["resolved_url"]}:
            raise ValueError("research source was not actually fetched by the read-only tool")
        if source["version"] != actual["version"]:
            raise ValueError("source version differs from fetched evidence")
        if not source["finding"].strip() or not source["design_impact"].strip():
            raise ValueError("source has no finding or concrete plan impact")
    nodes = {n["id"]: n for n in data["nodes"]}
    _unique(data["nodes"], "id", "nodes")
    known = {r["operator"]: r for r in catalog["operators"]}
    todo_ids = _unique(plan["todo"], "id", "TODO")
    for item in plan["todo"]:
        if set(item["depends_on"]) - todo_ids or set(item["node_ids"]) - set(nodes):
            raise ValueError("TODO references missing dependency or node")
        if not item["acceptance"].strip() or not item["deliverable"].strip():
            raise ValueError("TODO lacks an acceptance criterion or deliverable")
    visited, visiting = set(), set()

    def visit(identity):
        if identity in visiting:
            raise ValueError("plan dependency cycle")
        if identity not in nodes:
            raise ValueError("missing node dependency")
        if identity in visited:
            return
        visiting.add(identity)
        for dep in nodes[identity]["depends_on"]:
            visit(dep)
        visiting.remove(identity)
        visited.add(identity)

    for node in nodes.values():
        visit(node["id"])
        if node["status"] not in {"implemented", "needs_implementation"}:
            raise ValueError("invalid implementation status")
        registered = known.get(node["operator"])
        if node["status"] == "implemented" and (not registered or registered["status"] != "implemented"):
            raise ValueError("unknown or unverified operator marked executable")
        if node["status"] == "needs_implementation" and not any(
            node["id"] in item["node_ids"] for item in plan["todo"]
        ):
            raise ValueError("capability gap missing from TODO")
        if not set(node["evidence_ids"]).issubset(source_ids) or not node["evidence_ids"]:
            raise ValueError("node lacks actually read research evidence")
        parameters = json.loads(node["parameters_json"])
        if not isinstance(parameters, dict):
            raise ValueError("node parameters must be an object")
        output_names = _unique(node["outputs"], "name", "output ports")
        if not output_names:
            raise ValueError("node output ports absent")
        for binding in node["input_bindings"]:
            origin = binding["source"]
            if origin.startswith("node:"):
                parent, separator, port = origin[5:].partition(".")
                if (not separator or parent not in node["depends_on"] or parent not in nodes or
                        port not in {p["name"] for p in nodes[parent]["outputs"]}):
                    raise ValueError("node input does not bind a declared upstream output")
            elif origin.startswith("material:"):
                if origin[9:] not in material_ids:
                    raise ValueError("node input references missing provided material")
            elif origin.startswith("goal:"):
                _lookup({**goal, **request, "original_request": request["request"]}, origin[5:])
            elif origin.startswith("config:"):
                config = request.get("task_config", {})
                _lookup({"task_config": config, **config}, origin[7:])
            else:
                raise ValueError("input must bind node, material, goal or task config")
    bound_materials = {b["source"][9:] for n in nodes.values() for b in n["input_bindings"]
                       if b["source"].startswith("material:")}
    if bound_materials != material_ids:
        raise ValueError("declared materials do not actually enter the production graph")
    for source in research["sources"]:
        if set(source["node_ids"]) - set(nodes):
            raise ValueError("research impact references missing node")
    for candidate in research["operator_candidates"]:
        actual = evidence.get(candidate["evidence_id"], {})
        if candidate["evidence_id"] not in source_ids or candidate["verification_status"] != "candidate":
            raise ValueError("external implementation is only a research candidate")
        if not candidate["path"] or not candidate["revision"] or candidate["revision"] != actual.get("version"):
            raise ValueError("external code candidate lacks the actually fetched version")
        resolved = actual.get("resolved_url", "")
        if "github.com/" in resolved or "githubusercontent.com/" in resolved:
            match = re.fullmatch(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([0-9a-f]{40})/(.+)", resolved)
            if (not match or candidate["path"] != match[4] or candidate["revision"] != match[3]
                    or candidate["repository"].rstrip("/") != f"https://github.com/{match[1]}/{match[2]}"):
                raise ValueError("GitHub code candidate must bind an actually read commit, repository and file")
        if set(candidate["node_ids"]) - set(nodes):
            raise ValueError("external candidate references missing node")
    if not research["operator_candidates"]:
        raise ValueError("concrete external operator adaptation analysis is missing")
    for revision in research["revisions"]:
        if not revision["reason"] or not set(revision["evidence_ids"]).issubset(source_ids):
            raise ValueError("design revision lacks source evidence")
        if any(part in {"constraints", "materials", "original_request"} for part in revision["field"].split(".")):
            raise ValueError("research cannot revise user-owned requirements")
    expected_models = request.get("task_config", {}).get("models", {})
    for role, model in expected_models.items():
        if not any(r["role"] == role and r["model"] == model for r in data["model_roles"]):
            raise ValueError(f"user-selected model missing or substituted: {role}")
    example = json.loads(data["sample_example_json"])
    if not isinstance(example, dict) or example.get("is_placeholder") is not True:
        raise ValueError("sample example must be explicitly marked as a placeholder")
    if not example.get("family_id") or not isinstance(example.get("input"), dict):
        raise ValueError("sample example lacks family or input binding")
    forbidden = {"review", "review_evidence", "target_masks", "observed_events", "target_instances"}
    if forbidden.intersection(_keys(example["input"])):
        raise ValueError("target/review-side information leaked into model input")
    targets = example.get("target", {})
    annotations = example.get("annotations", {})
    if "subjects" in example["input"]:
        subjects = example["input"]["subjects"]
        identities = _unique(subjects, "id", "example subjects")
        if not identities or any(not row.get("reference") for row in subjects):
            raise ValueError("example subject references are not bound")
        if {row["subject_id"] for row in annotations.get("target_instances", [])} != identities:
            raise ValueError("subject references do not bind target instances")
        if len(_unique(annotations["target_instances"], "target_instance_id", "target instances")) != len(identities):
            raise ValueError("subject references must bind unique target instances")
        if not example["input"].get("background") or not example["input"].get("instruction") or not targets.get("image"):
            raise ValueError("multi-subject supervision incomplete")
    if targets.get("video"):
        if not example["input"].get("initial_frame") or not example["input"].get("action_condition"):
            raise ValueError("video action/initial frame absent from training input")
        evidence_bound = "evidence" in annotations or any(
            isinstance(event, dict) and "evidence" in event
            for event in annotations.get("observed_events", []))
        if not all(k in annotations for k in ("expected_events", "observed_events")) or not evidence_bound:
            raise ValueError("video expected/observed events and evidence must be separate")
    selected = {x["id"]: x for x in snapshot["records"]}
    adopted = {x["id"]: x for x in plan["experiences_used"]}
    for use in plan["experiences_used"]:
        exp = selected.get(use["id"])
        if not exp or exp["version"] != use["version"] or use["node_id"] not in nodes:
            raise ValueError("unavailable or incompatible experience adopted")
        node = nodes[use["node_id"]]
        if exp["scope"]["operator"] != node["operator"] or use["id"] not in node["experience_ids"]:
            raise ValueError("experience does not bind its actual operator node")
        for change in use["changes"]:
            allowed = [x for x in exp["scope"]["changes"] if x["field"] == change["field"]]
            if not allowed or allowed[0]["after_json"] != change["after_json"]:
                raise ValueError("adopted experience changed beyond its validated intervention")
            root, _, key = change["field"].partition(".")
            current = json.loads(node["parameters_json"]) if root == "parameters" else data["checks"]
            actual = current.get(key) if isinstance(current, dict) else current
            if actual != json.loads(change["after_json"]):
                raise ValueError("experience did not change actual plan parameters/checks")
    for node in nodes.values():
        if set(node["experience_ids"]) - set(adopted):
            raise ValueError("node cites an experience without adoption rationale")
    json.loads(data["coverage_json"])
    return {"valid": True, "task_type": goal["task_type"], "nodes": len(nodes),
            "sources_read": len(source_ids), "experiences_adopted": len(adopted),
            "production_execution_performed": False}
