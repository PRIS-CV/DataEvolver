"""Real G2 batch Operator executed/recovered by EvolutionController."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway, ModelError
from dataevolver.adapters.visual_worker import VisualLocalProvider, validate_visual_worker
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, canonical_json, file_sha256, identifier, payload_sha256
from dataevolver.dataset.g2_contracts import validate_g2_config
from dataevolver.dataset.g2_pipeline import VisualOperations, classify_pair, load_pair_manifest
from ..data_budget import DataBudgetJournal
from ..data_contracts import DataRunBudget
from ..models import EffectVerification, EvidenceStatus, ExpectedOutputSpec, OperatorReceipt, OperatorSpec, RecoveryResult, RecoveryStatus, StageStatus
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import atomic_write_json


OPERATOR_ID = "data.visual_g2"


class VisualG2Operator:
    def __init__(self, *, allowed_write_root, contract, roots=None, providers=None):
        self.contract = json.loads(canonical_json(contract))
        self.tasks = validate_g2_config(self.contract)
        if (len(self.tasks) > 3 or any(t.split != "pilot" for t in self.tasks) or
                self.contract["budget"]["max_edit_calls"] != len(self.tasks) or
                self.contract["budget"]["max_model_calls"] != 2*len(self.tasks) or
                self.contract["budget"]["max_probe_calls"] != 0):
            raise ValueError("Step 1 binds at most three pilot tasks, one edit/review each and no probe calls")
        self.root = Path(allowed_write_root).resolve()
        if roots is None:
            roots = {}
            for name, env in self.contract["stores"].items():
                identifier(env)
                if not os.environ.get(env):
                    raise ValueError(f"missing G2 store environment: {env}")
                roots[name] = os.environ[env]
        if set(roots) != set(self.contract["stores"]):
            raise ValueError("G2 store membership changed")
        self.store = ArtifactStore(roots)
        self.dataset_store = self.contract["dataset_store"]
        self.dataset_root = self.store.roots[self.dataset_store]
        if self.dataset_root.is_relative_to(self.root) or self.root.is_relative_to(self.dataset_root):
            raise ValueError("dataset and runtime roots must be disjoint")
        for key, path in self.store.roots.items():
            if key != self.dataset_store and (path.is_relative_to(self.dataset_root) or self.dataset_root.is_relative_to(path)):
                raise ValueError("source and writable dataset stores must be disjoint")
        for task in self.tasks:
            if task.source.store == self.dataset_store:
                raise ValueError("original tasks require a read-only source store")
            self.store.resolve(task.source)
        models = self.contract["models"]
        if set(models) != {"profiles", "policy", "workers"}:
            raise ValueError("unexpected G2 model fields")
        self.profiles = tuple(ModelProfile.from_dict(x) for x in models["profiles"])
        self.policy = ModelPolicy.from_dict(models["policy"])
        roles = {"visual.edit", "visual.inspect_region", "visual.review_constraints"}
        if set(self.policy.bindings) != roles or self.policy.max_calls != self.contract["budget"]["max_model_calls"]:
            raise ValueError("G2 model roles or total budget changed")
        if any(p.backend != "local_python" for p in self.profiles) or self.policy.allowed_local_origins:
            raise ValueError("real G2 accepts only explicitly registered offline workers")
        if set(models["workers"]) != {p.profile_id for p in self.profiles}:
            raise ValueError("G2 worker membership changed")
        for profile in self.profiles:
            worker = models["workers"][profile.profile_id]
            validate_visual_worker(worker)
            if profile.revision != "sha256:" + payload_sha256(worker["file_hashes"]):
                raise ValueError("model revision is not the frozen asset inventory digest")
        if models["workers"][self.policy.bindings["visual.edit"]]["kind"] != "qwen_edit_gpu":
            raise ValueError("editing role must bind the real Qwen editing worker")
        if any(models["workers"][self.policy.bindings[role]]["kind"] != "qwen_vl_cpu" for role in roles - {"visual.edit"}):
            raise ValueError("review/observation roles must bind the frozen vision worker")
        self.providers = providers
        source = Path(__file__).resolve()
        self.spec = OperatorSpec(operator_id=OPERATOR_ID, version="1", retry_safe=True,
                  source_path=str(source), code_revision=file_sha256(source),
                  model_config_hash=payload_sha256(self.contract), allowed_write_roots=(str(self.root), str(self.dataset_root)),
                  expected_output_specs=(ExpectedOutputSpec("effect.json"), ExpectedOutputSpec("pairs.json")))

    def gateway(self, call):
        factories = self.providers
        if factories is None:
            factories = {p.profile_id: (lambda p=p: VisualLocalProvider(p, self.contract["models"]["workers"][p.profile_id],
                         self.store, self.dataset_store, Path(call.work_dir)/"native_effects"/p.profile_id)) for p in self.profiles}
        return ModelGateway(profiles=self.profiles, policy=self.policy, receipt_root=Path(call.work_dir)/"model_receipts",
                            artifact_store=self.store, providers=factories)

    def _validate(self, call):
        if (call.operator_id, call.operator_version) != (OPERATOR_ID, "1") or payload_sha256(call.parameters) != payload_sha256(self.contract):
            raise ValueError("G2 call differs from frozen task/configuration")
        if not Path(call.work_dir).resolve().is_relative_to(self.root):
            raise ValueError("G2 work directory escapes campaign")

    def verify_effect(self, call, receipt):
        self._validate(call)
        path = Path(call.work_dir)/"effect.json"
        if not path.exists():
            return EffectVerification(EvidenceStatus.UNOBSERVED, "G2 batch is incomplete")
        try:
            value = json.loads(path.read_text())
            if value["call_hash"] != payload_sha256(call.to_dict()) or value["config_hash"] != payload_sha256(self.contract):
                raise ValueError("G2 effect identity changed")
            pairs = Path(call.work_dir)/"pairs.json"
            if file_sha256(pairs) != value["pairs_sha256"]:
                raise ValueError("G2 pair manifest changed")
            load_pair_manifest(pairs, self.store, include_rejected=True)
            manifest = json.loads(pairs.read_text())
            if manifest["usage"] != self.gateway(call).audit_usage():
                raise ValueError("export model-call accounting differs from the reservation ledger")
            if {p["task_id"] for p in manifest["pairs"]} != {t.task_id for t in self.tasks}:
                raise ValueError("G2 batch dropped a task")
            for row in manifest["pairs"]:
                task = next(t for t in self.tasks if t.task_id == row["task_id"])
                if row["task"] != task.to_dict():
                    raise ValueError("export original task changed")
                if row["recipe"] != self.contract["initial_recipe"][task.task_id]:
                    raise ValueError("export execution recipe differs from frozen configuration")
                for model in row["model_requests"]:
                    request = ModelRequest.from_dict(model["request"])
                    recovered = self.gateway(call).recover(request)
                    if recovered is None or recovered.without_reuse_hash() != model["receipt_hash"]:
                        raise ValueError("G2 model receipt changed")
            if receipt is not None and receipt.effect_hash != file_sha256(path):
                raise ValueError("G2 Operator receipt changed")
            return EffectVerification(EvidenceStatus.VERIFIED, "all G2 tasks, images, requests and receipts verified",
                                      artifact_hashes={"effect.json": file_sha256(path), "pairs.json": file_sha256(pairs)})
        except (OSError, ValueError, KeyError, TypeError, ModelError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def recover(self, call):
        verification = self.verify_effect(call, _load_receipt(_receipt_path(call)))
        if verification.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED, self._receipt(call, True))
        if verification.status is EvidenceStatus.UNOBSERVED:
            # Each child request recovers through ModelGateway before invoking.
            return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason="resume bound G2 tasks through their existing model intents")
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason=verification.reason)

    def _receipt(self, call, reused):
        work = Path(call.work_dir)
        effect, pairs = json.loads((work/"effect.json").read_text()), json.loads((work/"pairs.json").read_text())
        summary = pairs["summary"]
        all_complete = summary["reviewed_pairs"] == summary["tasks"]
        receipt = OperatorReceipt(call_id=call.call_id, operator_id=OPERATOR_ID, operator_version="1",
                    status=StageStatus.SUCCESS if all_complete else StageStatus.FAILED,
                    started_at=effect["started_at"], finished_at=effect["finished_at"], return_code=0 if all_complete else 1,
                    effect_hash=file_sha256(work/"effect.json"), reused=reused,
                    output_artifacts=({"path": str(work/"pairs.json"), "sha256": file_sha256(work/"pairs.json")},),
                    stage_result={"operator_id": OPERATOR_ID, "operator_version": "1",
                        "status": "SUCCESS" if all_complete else "FAILED",
                        "metrics": {"completion_ratio": summary["reviewed_pairs"]/summary["tasks"],
                                    "qualified_ratio": summary["qualified_pairs"]/summary["tasks"],
                                    "local_model_calls": pairs["usage"]["model_calls"]},
                        "hard_constraints": {"artifact_valid": True, "all_tasks_recorded": True},
                        "failure_code": "" if all_complete else "G2_INCOMPLETE",
                        "evidence_refs": [str(work/"pairs.json")],
                        "cost": {"model_calls": pairs["usage"]["model_calls"],
                                 "observed_wall_seconds": pairs["usage"]["observed_wall_seconds"]}})
        return _write_receipt(_receipt_path(call), receipt)

    def execute(self, call):
        self._validate(call)
        recovery = self.recover(call)
        if recovery.status is RecoveryStatus.RECOVERED:
            return recovery.receipt
        if recovery.status is not RecoveryStatus.RETRY_SAFE:
            raise ValueError(recovery.reason)
        work = Path(call.work_dir); work.mkdir(parents=True, exist_ok=True)
        started_path = work/"started.json"
        if not started_path.exists():
            atomic_write_json(started_path, {"started_at": datetime.now(timezone.utc).isoformat()})
        limits = self.contract["budget"]
        budget = DataBudgetJournal(work/"data_budget.json", DataRunBudget(max_operator_calls=4*len(self.tasks),
                    max_model_calls=limits["max_model_calls"], max_model_images=2*limits["max_model_calls"], max_input_rows=len(self.tasks),
                    max_output_bytes=limits["max_output_bytes"], max_wall_seconds=limits["max_wall_seconds"]),
                    payload_sha256(call.to_dict()))
        gateway = self.gateway(call)
        ops = VisualOperations(self.contract, self.store, gateway, work, budget)
        prefix = payload_sha256(call.to_dict())[:16]
        rows = []
        for task in self.tasks:
            row_path = work/"tasks"/f"{task.task_id}.json"
            if row_path.exists():
                rows.append(json.loads(row_path.read_text())); continue
            row = {"task_id": task.task_id, "task": task.to_dict(), "task_hash": task.compute_hash(),
                   "recipe": self.contract["initial_recipe"][task.task_id], "purpose": "candidate", "status": "not_run",
                   "output": None, "preservation": None, "review": None, "error": None, "model_requests": [], "attempts": []}
            request_start = len(ops.requests)
            attempt_start = len(ops.attempts)
            try:
                budget.check(); budget.reserve_rows(task.task_id, 1)
                output, receipt = ops.edit(task, row["recipe"], f"{prefix}-{task.task_id}-edit")
                row["output"] = output.to_dict()
                budget.reserve_file(output.path+":"+output.sha256, self.store.resolve(output).stat().st_size)
                row["preservation"] = ops.compare_preservation(task, output)
                budget.check()
                row["review"], _ = ops.review_constraints(task, output, f"{prefix}-{task.task_id}-review")
                row["status"] = classify_pair(row["preservation"], row["review"])
            except (ModelError, RuntimeError, ValueError, OSError) as exc:
                row["status"] = "failed"
                row["error"] = {"code": getattr(exc, "code", type(exc).__name__), "detail": str(exc)[:800]}
            row["model_requests"] = ops.requests[request_start:]
            row["attempts"] = ops.attempts[attempt_start:]
            atomic_write_json(row_path, row)
            rows.append(row)
        usage = gateway.audit_usage()
        summary = {"tasks": len(self.tasks), "produced_pairs": sum(x["output"] is not None for x in rows),
                   "reviewed_pairs": sum(x["review"] is not None for x in rows),
                   "qualified_pairs": sum(x["status"] == "pass" for x in rows),
                   "failed_tasks": sum(x["status"] in {"failed", "not_run"} for x in rows),
                   "uncertain_pairs": sum(x["status"] == "uncertain" for x in rows),
                   "rejected_pairs": sum(x["status"] == "fail" for x in rows),
                   "calls_per_qualified_pair": usage["model_calls"]/sum(x["status"] == "pass" for x in rows) if any(x["status"] == "pass" for x in rows) else None}
        atomic_write_json(work/"pairs.json", {"schema_version": "dataevolver.g2_pairs.v1", "config_hash": payload_sha256(self.contract),
                          "pairs": rows, "summary": summary, "usage": usage, "calibration_status": self.contract["evaluation"]["calibration_status"]})
        load_pair_manifest(work/"pairs.json", self.store, include_rejected=True)
        atomic_write_json(work/"effect.json", {"call_hash": payload_sha256(call.to_dict()), "config_hash": payload_sha256(self.contract),
                    "started_at": json.loads(started_path.read_text())["started_at"], "finished_at": datetime.now(timezone.utc).isoformat(),
                    "pairs_sha256": file_sha256(work/"pairs.json"), "summary": summary})
        return self._receipt(call, False)
