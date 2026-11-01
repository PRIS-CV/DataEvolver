"""Codex CLI adapter and deterministic four-file planning export."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .catalog import experience_snapshot, operator_catalog
from .contracts import PLAN_SCHEMA, validate_plan


def write_json(path, value):
    p = Path(path)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(p)


def read_lines(path):
    if not Path(path).exists():
        return []
    result = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def render_files(plan, request, root):
    goal = {"schema_version": "dataevolver.training_goal.v1",
            "original_request": request["request"], "materials": request.get("materials", []),
            "task_config": request.get("task_config", {}), **plan["goal"]}
    data = {"schema_version": "dataevolver.dataset_plan.v1", **copy.deepcopy(plan["dataset_plan"])}
    data["coverage"] = json.loads(data.pop("coverage_json"))
    data["sample_example"] = json.loads(data.pop("sample_example_json"))
    for node in data["nodes"]:
        node["parameters"] = json.loads(node.pop("parameters_json"))
    data["experience_adoptions"] = plan["experiences_used"]
    write_json(root / "goal.json", goal)
    # JSON is a standard YAML 1.2 subset. Avoid an additional runtime dependency.
    write_json(root / "dataset_plan.yaml", data)
    lines = ["# Research supporting the dataset plan", "", "## Sources and design impact", ""]
    for row in plan["research"]["sources"]:
        lines.extend([f"### {row['evidence_id']} — {row['title']}", f"- Source: {row['url']}",
                      f"- Version: {row['version']}", f"- Finding: {row['finding']}",
                      f"- Design impact: {row['design_impact']}",
                      f"- Nodes: {', '.join(row['node_ids'])}", ""])
    lines.extend(["## External operator candidates", ""])
    for row in plan["research"]["operator_candidates"]:
        lines.extend([f"### {row['operation']}", f"- Repository: {row['repository']}",
                      f"- File: {row['path']}", f"- Revision: {row['revision']}",
                      f"- Input adaptation: {row['input_adaptation']}",
                      f"- Output adaptation: {row['output_adaptation']}",
                      f"- Verification: {row['verification_status']}",
                      f"- Evidence: {row['evidence_id']}", ""])
    lines.extend(["## Research-driven design revisions", ""])
    for row in plan["research"]["revisions"]:
        lines.extend([f"- {row['field']}: {row['before']} → {row['after']}",
                      f"  Reason: {row['reason']}; evidence: {', '.join(row['evidence_ids'])}"])
    lines.extend(["", "## Experience use", ""])
    if not plan["experiences_used"]:
        lines.append("No compatible promoted experience was adopted.")
    for row in plan["experiences_used"]:
        lines.extend([f"- {row['id']} version {row['version']} → {row['node_id']}: {row['why']}",
                      f"  Changes: {json.dumps(row['changes'], ensure_ascii=False)}"])
    (root / "research.md").write_text("\n".join(lines) + "\n")
    lines = ["# Implementation and execution TODO", ""]
    for row in plan["todo"]:
        lines.extend([f"- [ ] {row['id']}: {row['title']}",
                      f"  - Dependencies: {', '.join(row['depends_on']) or 'none'}",
                      f"  - Nodes: {', '.join(row['node_ids'])}",
                      f"  - Deliverable: {row['deliverable']}",
                      f"  - Acceptance: {row['acceptance']}",
                      f"  - Status: {row['status']}", ""])
    (root / "todo.md").write_text("\n".join(lines))


INSTRUCTIONS = """You are DataEvolver's training-goal Planner. Produce a data and supervision
design, not a generic image/video-generation prompt. Respond only with the supplied JSON schema.
Use the user's language for prose. User constraints and supplied/declared material IDs are immutable.
System-inferred supervision may change after research, with reasons in research.revisions.

REQUIRED TOOLS: perform native live web search; actually read at least three task-relevant primary
sources using read_source; query_operator_catalog; query_experiences. Search must affect the design.
Read relevant official code, including DataFlow operator/Pipeline mechanisms and, where useful,
DataFlow-MM visual verification. External implementations remain candidates. Use read_source's
operator_candidates.verification_status must be the literal value "candidate"; put qualifications
in input/output adaptation prose. Read the official model documentation for selected model roles.
evidence_id, URL and EXACT version. For code candidates read a raw.githubusercontent.com file;
read_source pins mutable branches to a real commit. Candidate revision must equal that source version.
Map research findings to actual node IDs and meaningful input/output adaptations.

Plans are not executed. Catalog entries marked needs_implementation must stay that way and enter TODO.
Use node:<node_id>.<output_port>, material:<material_id>, goal:<field> or config:<field> for input
bindings. Every node dependency and port must exist. All nodes need fetched evidence IDs.
Four files are rendered from this one response. Include an explicitly marked is_placeholder=true
sample_example_json with input/target/annotations/family_id relationships, not fictional real assets.
Use placeholder: references in this example. Auxiliary masks, target_instances, observed_events
and review results are NOT model inputs unless the training goal explicitly demands them.

Adapt the route to actual material conditions. With no source materials, scene decomposition can
construct paired supervision: complete natural scene -> resolve requested instances -> SAM3
multi-instance masks -> references -> background_visible/background_completed -> training record.
With provided backgrounds and independent subject references, preserve those material IDs and
construct new targets using multi-reference composition and identity checks. Do not recreate user
references by cropping generated targets; segmentation may remain an auxiliary check. Background
extensions must respect constraints and be stated as assumptions/TODO if not resolved.
For video prediction/generation, initial_frame and action_condition belong in input; target.video,
annotations.expected_events, observed_events and evidence must be separate. Required events are
expectations until observed; uncertain events must not silently become labels.

Variable subject counts and production quantities come from task_config, never global templates.
For multi-subject examples input.subjects is an array of {id,reference}, input.background and
input.instruction are required; target.image plus annotations.target_instances (subject_id,
target_instance_id) bind every subject. Every derived artifact shares a family_id.
Experience adoption is optional. Cite only compatible records returned by query_experiences;
record exact before/after JSON changes and actually apply them to the indicated node. Never invent
a successful experience or weaken a user constraint. Absence is valid.
Use selected model roles exactly; assets or model-card claims do not establish runnable adapters.
Production uses usage accounting, not a new arbitrary model-attempt or memory threshold. Preserve
timeouts, cancellation, receipt reconciliation and no-progress stopping. Runtime reports are processed
by the parent on the execution host; do not add reporting infrastructure assumptions to the training design.
"""


def finalize_response(root, request, catalog, snapshot, state):
    plan = json.loads((root / "final-response.json").read_text())
    evidence = read_lines(root / "source-evidence.jsonl")
    validation = validate_plan(plan, request, catalog, snapshot, evidence)
    events = read_lines(root / "codex-events.jsonl")
    searches = [e for e in events if e.get("type") == "item.completed"
                and e.get("item", {}).get("type") == "web_search"
                and e["item"].get("status") not in {"failed", "cancelled"}]
    if not searches:
        raise ValueError("native web search has no completed execution event")
    trace = read_lines(root / "tool-trace.jsonl")
    for name in ("read_source", "query_operator_catalog", "query_experiences"):
        if not any(t["tool"] == name and t["ok"] for t in trace):
            raise ValueError(f"required read-only tool was not actually called: {name}")
    write_json(root / "plan.json", plan)
    render_files(plan, request, root)
    validation.update(native_search_events=len(searches), actual_tool_calls=len(trace))
    write_json(root / "validation.json", validation)
    state.update(state="COMPLETE", finished_at=time.time(), return_code=0,
                 files={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ("goal.json", "research.md", "dataset_plan.yaml", "todo.md", "plan.json")},
                 usage=[e.get("usage") for e in events if e.get("type") == "turn.completed"],
                 monetary_cost=None, production_execution_performed=False)
    write_json(root / "status.json", state)
    return state


def plan_goal(request, output_dir, instance, *, executable="codex", timeout=1800,
              experience_store=None, resume=False, revalidate=False, revise=False, revision_note=""):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    instance = Path(instance).expanduser().resolve()
    if not (instance / "config.toml").is_file():
        raise ValueError("Planner instance config is missing")
    if not isinstance(request, dict) or not str(request.get("request", "")).strip():
        raise ValueError("training request is required")
    for material in request.get("materials", []):
        if material.get("origin") not in {"user_provided", "user_declared"}:
            raise ValueError("material origin must distinguish provided from declared")
        if material["origin"] == "user_declared" and material.get("content_inspected") is not False:
            raise ValueError("declared material cannot claim pixel inspection")
    package = Path(__file__).resolve().parents[2]
    from dataevolver.harness.operator_sdk import builtin_step_registry
    catalog = operator_catalog(package, request.get("task_config", {}).get("models", {}), registry=builtin_step_registry())
    snapshot = experience_snapshot(Path(experience_store) if experience_store else None,
                                   request.get("task_hint", ""), catalog["models"], catalog,
                                   constraints=request.get("constraints", []),
                                   model_versions=request.get("task_config", {}).get("model_versions"))
    binding = hashlib.sha256(json.dumps(
        {"request": request, "catalog": catalog, "experience": snapshot,
         "model": "gpt-6.1-sol", "instructions": INSTRUCTIONS, "schema": PLAN_SCHEMA},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    status_path = root / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    if previous and previous.get("binding") != binding:
        raise ValueError("existing run belongs to different request/code/catalog/experience")
    if revise and (not previous or not revision_note.strip()):
        raise ValueError("explicit revision needs an existing run and a reason")
    if previous and previous["state"] == "COMPLETE" and not revise:
        validate_plan(json.loads((root / "plan.json").read_text()), request, catalog, snapshot,
                      read_lines(root / "source-evidence.jsonl"))
        for name, digest in previous["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("completed planning artifact changed")
        return {**previous, "reused": True, "new_model_invocations": 0}
    if previous:
        pid = previous.get("planner_pid")
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("Planner process still exists; refusing duplicate submission")
        if not (resume or revalidate or revise):
            raise ValueError("partial planning run exists; inspect evidence and explicitly resume")
        if not revalidate and not previous.get("thread_id"):
            raise ValueError("no resumable Codex thread; a new output directory is required")
        archive = root / "attempts" / str(time.time_ns())
        archive.mkdir(parents=True)
        for name in ("status.json", "final-response.json", "invocation.json", "code-provenance.json", "prompt.txt",
                     "plan.json", "goal.json", "research.md", "dataset_plan.yaml", "todo.md", "validation.json"):
            if (root / name).exists():
                shutil.copy2(root / name, archive / name)
        if revalidate:
            state = finalize_response(root, request, catalog, snapshot, dict(previous))
            state.update(revalidated=True, new_model_invocations=0)
            write_json(status_path, state)
            return state
    for name, value in [("request.json", request), ("catalog.json", catalog),
                        ("experience-snapshot.json", snapshot), ("output-schema.json", PLAN_SCHEMA)]:
        write_json(root / name, value)
    write_json(root / "code-provenance.json", {
        "module_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob("*.py")},
        "python_version": sys.version, "driver_entry": str(Path(__file__).resolve()),
        "recorded_at": time.time()})
    binary = shutil.which(executable) or executable
    env = os.environ.copy()
    env["CODEX_HOME"] = str(instance)
    env["PYTHONPATH"] = str(package.parent) + os.pathsep + env.get("PYTHONPATH", "")
    prompt = INSTRUCTIONS + "\n\nUSER REQUEST AND TASK CONFIG:\n" + json.dumps(request, ensure_ascii=False)
    if revision_note:
        prompt += "\n\nREVIEW FINDINGS TO ADDRESS (user constraints remain immutable):\n" + revision_note
    # Codex deliberately filters inherited environment variables for MCP children.
    # An absolute, stdlib-only entry avoids depending on inherited PYTHONPATH.
    bridge = {"command": sys.executable,
              "args": [str(Path(__file__).with_name("tools.py")), "--run-dir", str(root)]}
    messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "preflight", "version": "1"}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}]
    check = subprocess.run([bridge["command"], *bridge["args"]],
                           input="\n".join(json.dumps(x) for x in messages) + "\n",
                           capture_output=True, text=True, env=env, timeout=15)
    replies = [json.loads(line) for line in check.stdout.splitlines()]
    if check.returncode or len(replies) != 2 or any("error" in r for r in replies):
        raise RuntimeError("read-only MCP bridge preflight failed: " + check.stderr[-1000:])
    write_json(root / "bridge-preflight.json", {"ok": True, "tools": [t["name"] for t in replies[1]["result"]["tools"]],
                                             "protocol_only": True, "real_model_call": False})
    command = [binary, "--search", "--disable", "shell_tool", "--disable", "unified_exec",
               "--disable", "multi_agent", "-c",
               "mcp_servers.dataevolver_planning={ command = " + json.dumps(bridge["command"]) +
               ", args = " + json.dumps(bridge["args"]) + " }",
               "exec", "--ignore-user-config", "--model", "gpt-6.1-sol", "--sandbox", "read-only",
               "--skip-git-repo-check", "--cd", str(root),
               "--json", "--output-schema", str(root / "output-schema.json"),
               "--output-last-message", str(root / "final-response.json")]
    if previous:
        # Resume preserves the existing research session instead of repeating submission.
        command.extend(["resume", previous["thread_id"]])
        prompt = ("Continue the existing research session. The prior local attempt ended with: " +
                  str(previous.get("error", "interrupted")) +
                  ". The read-only MCP bridge is now available. Preserve original requirements and existing evidence. "
                  "Repair the actual validation error. Concrete external operator_candidates cannot be empty and "
                  "verification_status must be exactly candidate. If GitHub branch resolution is rate-limited, "
                  "discover the full commit through native search, then fetch the raw file at that commit. "
                  "research.sources contains only successful reads with nonempty evidence IDs; failed reads "
                  "remain in tool-trace.jsonl and must not be added as empty-ID sources. "
                  "Do not claim a runtime-verified implementation.\n\n" + prompt)
    command.append("-")
    (root / "prompt.txt").write_text(prompt)
    write_json(root / "invocation.json", {"argv": command, "instance": str(instance),
               "model": "gpt-6.1-sol", "credential_values_recorded": False, "binding": binding,
               "explicit_revision": revise, "revision_note": revision_note})
    state = {"state": "STARTING", "binding": binding, "started_at": time.time(),
             "driver_pid": os.getpid(), "model": "gpt-6.1-sol"}
    write_json(status_path, state)
    process = None
    try:
        with (root / "codex-events.jsonl").open("a") as stdout, (root / "codex.stderr.log").open("a") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, env=env, text=True)
            process.stdin.write(prompt)
            process.stdin.close()
            state.update(state="RUNNING", planner_pid=process.pid)
            write_json(status_path, state)
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                for event in read_lines(root / "codex-events.jsonl"):
                    if event.get("type") == "thread.started":
                        state["thread_id"] = event["thread_id"]
                state.update(updated_at=time.time(), sources_read=len(read_lines(root / "source-evidence.jsonl")),
                             tool_calls=len(read_lines(root / "tool-trace.jsonl")))
                write_json(status_path, state)
                if time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TimeoutError("Planner deadline exceeded; retained thread/evidence, no automatic resubmission")
                time.sleep(2)
        if process.returncode != 0:
            raise RuntimeError(f"Codex exited {process.returncode}; inspect codex.stderr.log and events")
        return finalize_response(root, request, catalog, snapshot, state)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        state.update(state="CANCELLED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                     error_type=type(exc).__name__, error=str(exc)[:2000],
                     return_code=process.returncode if process else None, finished_at=time.time())
        state["usage"] = [e.get("usage") for e in read_lines(root / "codex-events.jsonl") if e.get("type") == "turn.completed"]
        state["monetary_cost"] = None
        write_json(status_path, state)
        with (root / "attempt-history.jsonl").open("a") as history:
            history.write(json.dumps(state, ensure_ascii=False) + "\n")
        raise


def register_parser(subparsers):
    parser = subparsers.add_parser("plan-goal", help="Research a training goal and export a supervision/data plan")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--goal")
    source.add_argument("--goal-file")
    parser.add_argument("--materials", help="JSON list of provided or declared material records")
    parser.add_argument("--task-config", help="JSON task configuration, including selected model roles")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--planner-instance", default=os.environ.get("DATAEVOLVER_PLANNER_INSTANCE"))
    parser.add_argument("--experience-store")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true")
    recovery.add_argument("--revalidate", action="store_true", help="Validate a retained final response without another model call")
    recovery.add_argument("--revise", action="store_true", help="Explicitly revise an existing plan in its original thread")
    parser.add_argument("--revision-note-file", help="Review findings for an explicit plan revision")
    parser.set_defaults(func=command_plan)


def command_plan(args):
    if not args.planner_instance:
        raise ValueError("--planner-instance or DATAEVOLVER_PLANNER_INSTANCE is required")
    if args.goal_file:
        path = Path(args.goal_file)
        if path.suffix == ".json":
            request = json.loads(path.read_text())
        else:
            request = {"request": path.read_text()}
    else:
        request = {"request": args.goal}
    if args.materials:
        request["materials"] = json.loads(Path(args.materials).read_text())
    if args.task_config:
        request["task_config"] = json.loads(Path(args.task_config).read_text())
    result = plan_goal(request, args.output_dir, args.planner_instance, executable=args.codex,
                       timeout=args.timeout_seconds, experience_store=args.experience_store,
                       resume=args.resume, revalidate=args.revalidate, revise=args.revise,
                       revision_note=Path(args.revision_note_file).read_text() if args.revision_note_file else "")
    print(json.dumps({"state": result["state"], "output_dir": args.output_dir,
                      "reused": result.get("reused", False)}, ensure_ascii=False))
    return 0
