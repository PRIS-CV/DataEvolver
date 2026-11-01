"""Run one configured native worker under the existing Controller and receipt ABI."""
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path, PurePosixPath

from dataevolver.dataset.contracts import file_sha256, identifier, payload_sha256
from ..artifact_operator import ArtifactWorkloadOperator
from ..state_store import atomic_write_json


class NativeSceneOperator(ArtifactWorkloadOperator):
    """Host configuration selects code; a planner can only request its bound task."""

    def __init__(self, *, output_root, configuration):
        required = {"schema_version", "stage", "argv", "files", "inputs", "outputs",
                    "timeout_seconds", "allowed_env_keys"}
        if (set(configuration) != required or
                configuration["schema_version"] != "dataevolver.native_scene.v1"):
            raise ValueError("expected a frozen native worker configuration")
        identifier(configuration["stage"], "worker stage")
        argv = configuration["argv"]
        if (not isinstance(argv, list) or not argv or
                any(not isinstance(x, str) or not x for x in argv) or
                "{request}" not in argv or "{output}" not in argv):
            raise ValueError("scene worker requires argv with request and output arguments")
        binary = Path(shutil.which(argv[0]) or argv[0]).resolve()
        if str(binary) not in configuration["files"]:
            raise ValueError("scene executable must belong to the frozen file inventory")
        if type(configuration["timeout_seconds"]) is not int or not 1 <= configuration["timeout_seconds"] <= 86400:
            raise ValueError("scene worker timeout must be 1..86400 seconds")
        for key in configuration["allowed_env_keys"]:
            identifier(key, "environment key")
        if not configuration["outputs"]:
            raise ValueError("scene worker must declare its output artifacts")
        if len(set(configuration["outputs"].values())) != len(configuration["outputs"]):
            raise ValueError("scene output roles must reference distinct files")
        if configuration["stage"] == "ue5" and "render_index" not in configuration["outputs"]:
            raise ValueError("UE5 requires a verifiable native render index")
        for role, name in configuration["outputs"].items():
            identifier(role, "output role")
            relative = PurePosixPath(name)
            if (name in {"request.json", "intent.json", "result.json", "exit.json", "native.json", "worker.log"}
                    or not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or str(relative) != name):
                raise ValueError("scene outputs must be normalized relative paths")
        super().__init__(operator_id="scene." + configuration["stage"], output_root=output_root,
                         configuration=configuration)
        self._check_sources()

    def _check_sources(self):
        for name, digest in self.configuration["files"].items():
            path = Path(name).resolve()
            if not Path(name).is_absolute() or path.is_relative_to(self.output_root) or file_sha256(path) != digest:
                raise ValueError("scene executable or dependency changed")
        for role, record in self.configuration["inputs"].items():
            identifier(role, "input role")
            path = Path(record["path"]).resolve()
            if not Path(record["path"]).is_absolute() or path.is_relative_to(self.output_root) or file_sha256(path) != record["sha256"]:
                raise ValueError("scene input changed or overlaps the writable campaign")

    def _outputs(self, directory):
        outputs = {}
        for role, name in self.configuration["outputs"].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or not path.stat().st_size:
                raise ValueError("scene output is missing, empty or outside its worker directory")
            with path.open("rb") as stream:
                prefix = stream.read(12)
            if path.suffix == ".blend" and not prefix.startswith(b"BLENDER"):
                raise ValueError("invalid Blender output header")
            if path.suffix == ".glb" and not prefix.startswith(b"glTF"):
                raise ValueError("invalid binary glTF output header")
            if path.suffix == ".json":
                json.loads(path.read_text())
            if self.configuration["stage"] == "ue5" and role == "render_index":
                from dataevolver.workflows.scene.ue5_worker import verify_render_index
                verify_render_index(path)
            outputs[role] = {"path": str(path), "sha256": file_sha256(path)}
        return outputs

    def workload(self, call):
        self._check_sources()
        directory = self.path(call).parent / "native"
        directory.mkdir(parents=True, exist_ok=True)
        request = {"schema_version": "dataevolver.scene_request.v1", "stage": self.configuration["stage"],
                   "inputs": self.configuration["inputs"], "outputs": self.configuration["outputs"],
                   "configuration_sha256": payload_sha256(self.configuration), "call": call.to_dict()}
        request_hash = payload_sha256(request)
        native = directory / "native.json"
        intent = directory / "intent.json"
        if native.exists():
            report = json.loads(native.read_text())
            if report["request_sha256"] != request_hash:
                raise ValueError("scene native receipt belongs to another request")
            self.validate_report(call, report)
            return report
        if intent.exists() or any(directory.iterdir()):
            raise RuntimeError("scene attempt has no verified completion; reconcile retained native files before retrying")
        request_path = directory / "request.json"
        atomic_write_json(request_path, request)
        argv = [str(request_path) if x == "{request}" else str(directory) if x == "{output}" else x
                for x in self.configuration["argv"]]
        environment = {key: os.environ[key] for key in self.configuration["allowed_env_keys"] if key in os.environ}
        atomic_write_json(intent, {"request_sha256": request_hash, "argv": argv, "status": "STARTED"})
        with (directory / "worker.log").open("w") as log:
            process = subprocess.Popen(argv, cwd=directory, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=self.configuration["timeout_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                atomic_write_json(directory / "exit.json", {"returncode": process.returncode, "timed_out": True})
                raise
        atomic_write_json(directory / "exit.json", {"returncode": code, "timed_out": False})
        if code != 0:
            raise RuntimeError("scene worker failed; inspect the retained worker.log and exit.json")
        result = json.loads((directory / "result.json").read_text())
        if result.get("status") != "completed" or result.get("request_sha256") != request_hash:
            raise ValueError("scene worker result does not bind its completed request")
        self._check_sources()
        report = {"stage": self.configuration["stage"], "request_sha256": request_hash,
                  "outputs": self._outputs(directory), "quality_verified": False}
        atomic_write_json(native, report)
        return report

    def validate_report(self, call, report):
        self._check_sources()
        directory = self.path(call).parent / "native"
        request = json.loads((directory / "request.json").read_text())
        if (report.get("request_sha256") != payload_sha256(request) or
                request.get("configuration_sha256") != payload_sha256(self.configuration) or
                request.get("call") != call.to_dict() or report.get("stage") != self.configuration["stage"] or
                report.get("quality_verified") is not False or report.get("outputs") != self._outputs(directory)):
            raise ValueError("scene worker evidence changed")
        exit_status = json.loads((directory / "exit.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        if (exit_status != {"returncode": 0, "timed_out": False} or
                result.get("status") != "completed" or result.get("request_sha256") != report["request_sha256"]):
            raise ValueError("scene process or result receipt changed")
