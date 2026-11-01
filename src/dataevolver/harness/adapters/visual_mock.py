"""Composite mock graph Operator using the existing Controller and receipts.

This route verifies contracts and recovery, not visual quality or real model
readiness. Each graph node is bounded to one registered mock model request.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelReceipt, ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway, validate_response_schema
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, canonical_json, file_sha256, payload_sha256
from ..models import EffectVerification, EvidenceStatus, ExpectedOutputSpec, OperatorCall, OperatorReceipt, OperatorSpec, RecoveryResult, RecoveryStatus, StageStatus
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import atomic_write_json
from ..task_contracts import DatasetIntent, NodeRun, NodeSpecGraph, QualityPolicy


OPERATOR_ID = "visual.contract_mock"


class VisualContractMockOperator:
    def __init__(self, *, allowed_write_root: Path, contract: dict):
        if set(contract) != {"intent", "quality_policy", "graph", "model_profiles", "model_policy"}:
            raise ValueError("visual contract requires only the declared task, policy, graph and model fields")
        self.contract = json.loads(canonical_json(contract))
        self.intent = DatasetIntent.from_dict(contract["intent"])
        self.policy = QualityPolicy.from_dict(contract["quality_policy"])
        self.graph = NodeSpecGraph.from_dict(contract["graph"])
        self.profiles = tuple(ModelProfile.from_dict(x) for x in contract["model_profiles"])
        self.model_policy = ModelPolicy.from_dict(contract["model_policy"])
        if any(x.backend != "mock" for x in self.profiles) or self.model_policy.allowed_local_origins:
            raise ValueError("visual contract mock route accepts only inference-free mock profiles")
        if self.model_policy.max_calls > self.policy.max_model_calls:
            raise ValueError("model gateway budget exceeds the task policy")
        self.graph.validate_policy(self.policy, set(self.model_policy.bindings))
        if len(self.graph.nodes) > self.model_policy.max_calls:
            raise ValueError("model gateway budget cannot cover the node graph")
        for node in self.graph.nodes:
            if node.operator != "model.invoke" or node.operator_version != "1":
                raise ValueError("unregistered mock node operator or revision")
            if set(node.parameters) - {"messages", "response_schema", "max_tokens", "seed"}:
                raise ValueError("mock node contains unsupported parameters")
            self._request(node, {})
            validate_response_schema(node.parameters["response_schema"])
        self.root = Path(allowed_write_root).resolve()
        ModelGateway(profiles=self.profiles, policy=self.model_policy,
                     receipt_root=self.root / "model_validation").contract_hash()
        source = Path(__file__).resolve()
        self.spec = OperatorSpec(
            operator_id=OPERATOR_ID, version="1", retry_safe=True,
            expected_output_specs=(ExpectedOutputSpec("effect.json"),),
            model_config_hash=payload_sha256(self.contract), code_revision=file_sha256(source),
            allowed_write_roots=(str(self.root),), source_path=str(source),
        )

    def _validate(self, call):
        if (call.operator_id, call.operator_version) != (OPERATOR_ID, "1"):
            raise ValueError("visual mock call identity does not match the registered Operator")
        if not Path(call.work_dir).resolve().is_relative_to(self.root):
            raise ValueError("visual mock work directory escapes its allowed root")
        if payload_sha256(call.parameters) != payload_sha256(self.contract):
            raise ValueError("visual mock call differs from its registered contract")

    def _request(self, node, outputs):
        messages = list(node.parameters.get("messages", []))
        if node.dependencies:
            messages.append({"role": "user", "content": canonical_json({
                "dependencies": {key: outputs.get(key) for key in node.dependencies}
            })})
        return ModelRequest(
            request_id=node.node_id, role=node.role, messages=tuple(messages),
            response_schema=node.parameters["response_schema"], artifacts=node.inputs,
            max_tokens=node.parameters.get("max_tokens", 256), seed=node.parameters.get("seed", 0),
        )

    def _gateway(self, call):
        return ModelGateway(profiles=self.profiles, policy=self.model_policy,
                            receipt_root=Path(call.work_dir) / "model_receipts")

    def _node_path(self, call, node):
        path = (Path(call.work_dir) / "node_runs" / node.node_id).resolve()
        if not path.is_relative_to(Path(call.work_dir).resolve()):
            raise ValueError("node output path escapes the Operator work directory")
        return path

    def _check_node(self, call, node):
        path = self._node_path(call, node) / "node_run.json"
        if not path.exists():
            return None
        run = NodeRun.from_dict(json.loads(path.read_text()))
        if (run.node_id != node.node_id or run.node_spec_hash != node.compute_hash() or
                run.campaign_id != call.campaign_id or run.call_id != call.call_id):
            raise ValueError("NodeRun identity or NodeSpec binding changed")
        expected = f"node_runs/{node.node_id}/output.json"
        if run.output.store != "node_outputs" or run.output.path != expected:
            raise ValueError("NodeRun output reference changed")
        output = ArtifactStore({"node_outputs": call.work_dir}).resolve(run.output)
        content = json.loads(output.read_text())
        wrapper = json.loads((Path(call.work_dir) / "model_receipts" / f"{node.node_id}.receipt.json").read_text())
        receipt = ModelReceipt.from_dict(wrapper["receipt"])
        if (wrapper["sha256"] != payload_sha256(wrapper["receipt"]) or
                receipt.without_reuse_hash() != run.receipt_hash or receipt.request_hash != run.request_hash or
                receipt.request_id != run.request_id or receipt.output != content):
            raise ValueError("NodeRun model receipt or output binding changed")
        return run, content

    def _inspect(self, call):
        self._validate(call)
        try:
            for node in self.graph.nodes:
                self._check_node(call, node)
            effect_path = Path(call.work_dir) / "effect.json"
            if not effect_path.exists():
                return EffectVerification(EvidenceStatus.UNOBSERVED, "mock graph is incomplete"), ()
            effect = json.loads(effect_path.read_text())
            if effect.get("contract_hash") != payload_sha256(self.contract):
                raise ValueError("mock graph contract binding changed")
            expected_paths = {"effect.json"}
            for node in self.graph.nodes:
                if self._check_node(call, node) is None:
                    raise ValueError("committed graph is missing a NodeRun")
                expected_paths.update({f"node_runs/{node.node_id}/node_run.json", f"node_runs/{node.node_id}/output.json"})
            recorded = effect["artifacts"]
            if set(recorded) != expected_paths - {"effect.json"}:
                raise ValueError("mock graph artifact membership changed")
            hashes = {"effect.json": file_sha256(effect_path)}
            for relative, digest in recorded.items():
                path = (Path(call.work_dir) / relative).resolve()
                if not path.is_relative_to(Path(call.work_dir).resolve()) or file_sha256(path) != digest:
                    raise ValueError("mock graph artifact is missing or changed")
                hashes[relative] = digest
            artifacts = tuple({"path": key, "sha256": value} for key, value in sorted(hashes.items()))
            return EffectVerification(EvidenceStatus.VERIFIED, "mock graph artifacts verified", artifact_hashes=hashes), artifacts
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc)), ()

    def _receipt(self, call, reused):
        verification, artifacts = self._inspect(call)
        if verification.status is not EvidenceStatus.VERIFIED:
            raise ValueError(verification.reason)
        now = datetime.now(timezone.utc).isoformat()
        return OperatorReceipt(
            call_id=call.call_id, operator_id=OPERATOR_ID, operator_version="1", status=StageStatus.SUCCESS,
            started_at=now, finished_at=now, return_code=0, output_artifacts=artifacts,
            effect_hash=payload_sha256(verification.artifact_hashes), reused=reused,
            stage_result={"operator_id": OPERATOR_ID, "operator_version": "1", "status": "SUCCESS",
                "metrics": {"completion_ratio": 1.0, "resource_policy_respected": 1.0},
                "hard_constraints": {"artifact_valid": True},
                "matched_trial_key": call.matched_trial_key.to_dict() if call.matched_trial_key else None},
        )

    def execute(self, call):
        self._validate(call)
        prior = _load_receipt(_receipt_path(call))
        if prior is not None:
            if self.verify_effect(call, prior).status is not EvidenceStatus.VERIFIED:
                raise ValueError("mock Operator receipt or artifacts changed")
            return prior.with_updates(reused=True)
        verification, _ = self._inspect(call)
        if verification.status is EvidenceStatus.CONTRADICTED:
            raise ValueError(verification.reason)
        if verification.status is EvidenceStatus.VERIFIED:
            return _write_receipt(_receipt_path(call), self._receipt(call, True))
        gateway, outputs, artifacts = self._gateway(call), {}, {}
        for node in self.graph.ordered():
            request = self._request(node, outputs)
            existing = self._check_node(call, node)
            model_receipt = gateway.invoke(request)
            if existing is not None and existing[0].receipt_hash != model_receipt.without_reuse_hash():
                raise ValueError("NodeRun model receipt changed")
            path = self._node_path(call, node)
            output_path = path / "output.json"
            atomic_write_json(output_path, model_receipt.output)
            output_ref = ArtifactRef("node_outputs", f"node_runs/{node.node_id}/output.json", file_sha256(output_path))
            run = NodeRun(call.campaign_id, call.call_id, node.node_id, node.compute_hash(), request.request_id,
                          model_receipt.request_hash, model_receipt.without_reuse_hash(), output_ref)
            atomic_write_json(path / "node_run.json", run.to_dict())
            outputs[node.node_id] = model_receipt.output
            for name in ("output.json", "node_run.json"):
                artifacts[f"node_runs/{node.node_id}/{name}"] = file_sha256(path / name)
        atomic_write_json(Path(call.work_dir) / "effect.json", {"contract_hash": payload_sha256(self.contract), "artifacts": artifacts, "evidence_level": "mock_verified"})
        return _write_receipt(_receipt_path(call), self._receipt(call, False))

    def recover(self, call):
        self._validate(call)
        try:
            receipt = _load_receipt(_receipt_path(call))
            verification, _ = self._inspect(call)
            if receipt is not None:
                verification = self.verify_effect(call, receipt)
            if verification.status is EvidenceStatus.VERIFIED:
                result = receipt or _write_receipt(_receipt_path(call), self._receipt(call, True))
                return RecoveryResult(RecoveryStatus.RECOVERED, result, "mock graph recovered")
            if verification.status is EvidenceStatus.UNOBSERVED:
                return RecoveryResult(RecoveryStatus.RETRY_SAFE, None, "resume remaining bounded mock nodes")
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, verification.reason)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, None, str(exc))

    def verify_effect(self, call, receipt):
        verification, _ = self._inspect(call)
        if receipt is not None and verification.status is EvidenceStatus.VERIFIED:
            if (receipt.call_id != call.call_id or receipt.operator_id != OPERATOR_ID or
                    receipt.effect_hash != payload_sha256(verification.artifact_hashes)):
                return EffectVerification(EvidenceStatus.CONTRADICTED, "mock receipt binding changed")
        return verification
