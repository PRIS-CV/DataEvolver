"""Production preflight, worker lifecycle, and resumable task execution."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


PROFILE_SCHEMA = "dataevolver.production.v1"
STATE_SCHEMA = "dataevolver.run_state.v1"
CORE_REQUIRED_PATH_KEYS = (
    "qwen_image_model",
    "sam3_checkpoint",
    "sam3_source",
    "hunyuan3d_source",
    "hunyuan3d_weights",
    "dino_model",
    "realesrgan_checkpoint",
)
HYWORLD_BASE_PATH_KEYS = (
    "hyworld_source",
    "hyworld_weights",
)
HYWORLD_REAL3D_PATH_KEYS = (
    "hyworld_moge_model",
    "hyworld_zim_model",
    "hyworld_grounding_dino_model",
    "hyworld_sam3_model",
    "hyworld_worldstereo_weights",
    "hyworld_wan_base_model",
)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output)


def resolve_profile_path(profile_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_absolute():
        return expanded
    return (profile_path.parent / expanded).resolve()


def load_profile(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    profile_path = Path(path).resolve()
    profile = load_json(profile_path)
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"Unsupported production profile schema: {profile.get('schema')}")
    return profile_path, profile


def profile_uses_hyworld(profile: dict[str, Any]) -> bool:
    profile_name = str(profile.get("profile") or "")
    target_workflow = str(profile.get("target_workflow") or "")
    paths = profile.get("paths") or {}
    return (
        profile_name == "world_model"
        or target_workflow == "world_model_scene"
        or any(paths.get(key) for key in HYWORLD_BASE_PATH_KEYS + HYWORLD_REAL3D_PATH_KEYS)
    )


def _run_probe(
    command: list[str],
    timeout: int = 20,
    *,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
            cwd=cwd,
        )
        return {
            "success": completed.returncode == 0,
            "return_code": completed.returncode,
            "output": completed.stdout[-4000:],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:
        return {
            "success": False,
            "return_code": None,
            "output": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }


def build_runtime_environment(profile_path: Path, profile: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        str(key): os.path.expandvars(str(value))
        for key, value in (profile.get("environment") or {}).items()
        if value is not None
    })
    offline = profile.get("offline") or {}
    environment.update({
        "HF_HUB_OFFLINE": "1" if offline.get("hf_hub_offline", True) else "0",
        "TRANSFORMERS_OFFLINE": "1" if offline.get("hf_hub_offline", True) else "0",
        "DIFFUSERS_OFFLINE": "1" if offline.get("hf_hub_offline", True) else "0",
    })
    # HYWorld uses HYWORLD_NO_MODEL_DOWNLOADS=1 to skip MoGe/ZIM/GD loading and
    # require precomputed depth. Production real-3D must load local cached
    # models while HF remains offline; only opt into skipping via the explicit
    # legacy flag.
    skip_hyworld_depth_models = bool(offline.get("skip_hyworld_depth_models", False))
    environment["HYWORLD_NO_MODEL_DOWNLOADS"] = "1" if skip_hyworld_depth_models else "0"

    workspace = resolve_profile_path(profile_path, profile.get("workspace_root"))
    paths = profile.get("paths") or {}
    env_path_map = {
        "hyworld_moge_model": ("HYWORLD_MOGE_MODEL_PATH", "MOGE_MODEL_PATH"),
        "hyworld_zim_model": "HYWORLD_ZIM_MODEL_PATH",
        "hyworld_grounding_dino_model": "HYWORLD_GROUNDING_DINO_MODEL_PATH",
        "hyworld_sam3_model": "HYWORLD_SAM3_MODEL_PATH",
        "hyworld_worldstereo_weights": "HYWORLD_WORLDSTEREO_PATH",
        "hyworld_wan_base_model": ("HYWORLD_WAN_BASE_MODEL", "WORLDSTEREO_BASE_MODEL_PATH"),
    }
    for key, env_names in env_path_map.items():
        candidate = resolve_profile_path(profile_path, paths.get(key))
        if candidate:
            if isinstance(env_names, str):
                env_names = (env_names,)
            for env_name in env_names:
                environment[env_name] = str(candidate)
    pythonpath_entries: list[str] = []
    if workspace:
        pythonpath_entries.append(str(workspace / "pipeline"))
    for key in ("hyworld_source", "sam3_source", "hunyuan3d_source"):
        candidate = resolve_profile_path(profile_path, paths.get(key))
        if candidate and candidate.exists():
            pythonpath_entries.append(str(candidate))
            if key == "hyworld_source":
                pythonpath_entries.extend([
                    str(candidate / "hyworld2" / "panogen"),
                    str(candidate / "hyworld2" / "worldgen"),
                ])
    current_pythonpath = environment.get("PYTHONPATH")
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))
    return environment


def doctor_profile(path: str | os.PathLike[str], run_imports: bool = True) -> dict[str, Any]:
    profile_path, profile = load_profile(path)
    checks: list[dict[str, Any]] = []

    def add(name: str, success: bool, detail: str, *, blocking: bool = True):
        checks.append({"name": name, "success": bool(success), "blocking": blocking, "detail": detail})

    runtime = profile.get("runtime") or {}
    dataevolver_python = resolve_profile_path(profile_path, runtime.get("dataevolver_python"))
    hyworld_python = resolve_profile_path(profile_path, runtime.get("hyworld_python"))
    blender = resolve_profile_path(profile_path, runtime.get("blender_bin"))
    workspace = resolve_profile_path(profile_path, profile.get("workspace_root"))
    environment = build_runtime_environment(profile_path, profile)
    uses_hyworld = profile_uses_hyworld(profile)

    add("workspace", bool(workspace and workspace.is_dir()), str(workspace))
    runtime_executables = [
        ("dataevolver_python", dataevolver_python),
        ("blender", blender),
    ]
    if uses_hyworld:
        runtime_executables.append(("hyworld_python", hyworld_python))
    for name, executable in runtime_executables:
        add(name, bool(executable and executable.is_file() and os.access(executable, os.X_OK)), str(executable))

    paths = profile.get("paths") or {}
    for key in CORE_REQUIRED_PATH_KEYS:
        candidate = resolve_profile_path(profile_path, paths.get(key))
        add(f"path.{key}", bool(candidate and candidate.exists()), str(candidate))

    if uses_hyworld:
        for key in HYWORLD_BASE_PATH_KEYS + HYWORLD_REAL3D_PATH_KEYS:
            candidate = resolve_profile_path(profile_path, paths.get(key))
            add(f"path.{key}", bool(candidate and candidate.exists()), str(candidate))

        hyworld_weights = resolve_profile_path(profile_path, paths.get("hyworld_weights"))
        worldmirror_dir = hyworld_weights / "HY-WorldMirror-2.0" if hyworld_weights else None
        add(
            "path.hyworld_worldmirror",
            bool(
                worldmirror_dir
                and (worldmirror_dir / "model.safetensors").is_file()
                and ((worldmirror_dir / "config.yaml").is_file() or (worldmirror_dir / "config.json").is_file())
            ),
            str(worldmirror_dir),
        )

    offline = profile.get("offline") or {}
    add("offline.no_model_downloads", offline.get("no_model_downloads") is True, str(offline.get("no_model_downloads")))
    add("offline.hf_hub_offline", offline.get("hf_hub_offline") is True, str(offline.get("hf_hub_offline")))
    if uses_hyworld:
        add(
            "offline.hyworld_depth_models_enabled",
            offline.get("skip_hyworld_depth_models", False) is False,
            f"skip_hyworld_depth_models={offline.get('skip_hyworld_depth_models', False)}",
        )

    probes: dict[str, Any] = {}
    if dataevolver_python and dataevolver_python.exists():
        probes["dataevolver_python"] = _run_probe(
            [
                str(dataevolver_python),
                "-c",
                "import json,sys,torch; print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}))",
            ],
            env=environment,
            cwd=workspace,
        )
    if hyworld_python and hyworld_python.exists():
        probes["hyworld_python"] = _run_probe(
            [
                str(hyworld_python),
                "-c",
                "import json,sys,torch; print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}))",
            ],
            env=environment,
            cwd=workspace,
        )
    if blender and blender.exists():
        probes["blender"] = _run_probe([str(blender), "--version"], env=environment, cwd=workspace)

    if run_imports:
        imports = profile.get("import_checks") or {}
        for runtime_name, modules in imports.items():
            executable = dataevolver_python if runtime_name == "dataevolver" else hyworld_python
            if not executable or not executable.exists():
                continue
            code = "; ".join(f"import {module}" for module in modules)
            probes[f"imports.{runtime_name}"] = _run_probe(
                [str(executable), "-c", code],
                timeout=60,
                env=environment,
                cwd=workspace,
            )

    for name, result in probes.items():
        add(f"probe.{name}", result["success"], result["output"][-1000:])

    endpoints = profile.get("health_endpoints") or []
    for endpoint in endpoints:
        url = endpoint.get("url")
        required = bool(endpoint.get("required", True))
        try:
            with urllib.request.urlopen(url, timeout=float(endpoint.get("timeout", 3))) as response:
                success = 200 <= response.status < 300
                detail = f"HTTP {response.status}"
        except Exception as exc:
            success = False
            detail = str(exc)
        add(f"endpoint.{endpoint.get('name', url)}", success, detail, blocking=required)

    success = all(check["success"] or not check["blocking"] for check in checks)
    return {
        "schema": "dataevolver.doctor.v1",
        "success": success,
        "profile": str(profile_path),
        "checks": checks,
        "probes": probes,
    }


def _socket_is_live(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1.0)
            client.connect(str(path))
            client.sendall(b'{"action":"status"}\n')
            return bool(client.recv(4096))
    except OSError:
        return False


def start_workers(path: str | os.PathLike[str], skip_doctor: bool = False) -> dict[str, Any]:
    profile_path, profile = load_profile(path)
    if not skip_doctor:
        doctor = doctor_profile(path)
        if not doctor["success"]:
            raise RuntimeError("Production doctor failed; workers were not started")

    workspace = resolve_profile_path(profile_path, profile["workspace_root"])
    runtime_dir = resolve_profile_path(profile_path, profile.get("runtime_dir", ".dataevolver/runtime"))
    assert workspace is not None and runtime_dir is not None
    runtime_dir.mkdir(parents=True, exist_ok=True)
    environment = build_runtime_environment(profile_path, profile)

    started = []
    for service in profile.get("workers") or []:
        name = service["name"]
        socket_path = runtime_dir / f"{name}.sock"
        if _socket_is_live(socket_path):
            started.append({"name": name, "status": "already_running", "socket": str(socket_path)})
            continue
        python_key = service.get("python", "dataevolver_python")
        python_path = resolve_profile_path(profile_path, profile["runtime"][python_key])
        if python_path is None:
            raise RuntimeError(f"Worker {name} has no Python executable")
        log_path = runtime_dir / f"{name}.log"
        command = [
            str(python_path),
            str(workspace / "pipeline" / "model_worker.py"),
            "serve",
            "--kind",
            service["kind"],
            "--device",
            service.get("device", "cuda:0"),
            "--socket",
            str(socket_path),
        ]
        model_path = service.get("model_path")
        if model_path:
            command.extend(["--model-path", os.path.expandvars(model_path)])
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=workspace,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.time() + float(service.get("startup_timeout", 30))
        while time.time() < deadline and process.poll() is None and not _socket_is_live(socket_path):
            time.sleep(0.25)
        if not _socket_is_live(socket_path):
            raise RuntimeError(f"Worker {name} failed to start; inspect {log_path}")
        atomic_write_json(runtime_dir / f"{name}.pid.json", {
            "name": name,
            "pid": process.pid,
            "socket": str(socket_path),
            "log": str(log_path),
            "command": command,
        })
        started.append({"name": name, "status": "started", "pid": process.pid, "socket": str(socket_path)})
    return {"success": True, "runtime_dir": str(runtime_dir), "workers": started}


def run_resumable_manifest(path: str | os.PathLike[str], state_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = load_json(manifest_path)
    state_file = Path(state_path).resolve() if state_path else manifest_path.with_suffix(".state.json")
    if state_file.exists():
        state = load_json(state_file)
    else:
        state = {"schema": STATE_SCHEMA, "manifest": str(manifest_path), "tasks": {}}

    for task in manifest.get("tasks") or []:
        task_id = task["id"]
        expected = [Path(os.path.expandvars(value)).resolve() for value in task.get("expected_outputs") or []]
        previous = state["tasks"].get(task_id) or {}
        if previous.get("status") == "completed" and all(output.exists() for output in expected):
            continue
        command = [os.path.expandvars(str(value)) for value in task["command"]]
        started = time.time()
        completed = subprocess.run(
            command,
            cwd=os.path.expandvars(task.get("cwd", str(manifest_path.parent))),
            env={**os.environ, **{str(k): os.path.expandvars(str(v)) for k, v in task.get("environment", {}).items()}},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        outputs_ready = all(output.exists() for output in expected)
        status = "completed" if completed.returncode == 0 and outputs_ready else "failed"
        state["tasks"][task_id] = {
            "status": status,
            "return_code": completed.returncode,
            "started_at": started,
            "elapsed_seconds": round(time.time() - started, 3),
            "expected_outputs": [str(output) for output in expected],
            "stdout_tail": completed.stdout[-4000:],
        }
        atomic_write_json(state_file, state)
        if status != "completed":
            raise RuntimeError(f"Task {task_id} failed; state saved to {state_file}")
    return {"success": True, "state": str(state_file), "tasks": state["tasks"]}
