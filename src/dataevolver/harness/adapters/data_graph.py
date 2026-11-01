"""Registered image/data steps inside the existing EvolutionController.

This is a bounded composite Operator, not a second campaign controller. Child
Operators own effects/receipts; ModelGateway owns model-attempt reservations.
"""
from __future__ import annotations

import json
import hashlib
import tempfile
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway
from dataevolver.dataset import curation
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, canonical_json, file_sha256, identifier, payload_sha256
from ..data_budget import DataBudgetJournal
from ..data_contracts import DataNodeGraph as DataNodeGraph, DataNodeRun, DataRunBudget, read_data_graph
from ..models import EffectVerification, EvidenceStatus, ExpectedOutputSpec, OperatorCall, OperatorReceipt, OperatorSpec, RecoveryResult, RecoveryStatus, StageStatus
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import CampaignLease, atomic_write_json
from ..task_contracts import DatasetIntent, QualityPolicy


OPERATOR_ID = "data.image_curation"


def _refs(value):
    if isinstance(value, dict):
        if value.get("schema_version") == ArtifactRef.schema:
            yield ArtifactRef.from_dict(value)
        else:
            for item in value.values():
                yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


class NodeContext:
    def __init__(self, graph, parent, call, node, inputs, budget):
        self.graph, self.parent, self.call, self.node = graph, parent, call, node
        self.inputs, self.budget = inputs, budget
        self.parameters, self.policy, self.store = node.parameters, graph.policy, graph.store
        binding = payload_sha256([parent.campaign_id, parent.call_id, graph.contract_hash])[:24]
        self.prefix = f"catalog/{binding}/{node.node_id}"
        self.writes, self.model_requests = {}, []

    def check(self):
        self.budget.check()

    def read(self, name):
        path = self.store.resolve(self.inputs[name])
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("node JSON input exceeds the bounded catalog size")
        value = json.loads(path.read_text())
        for ref in _refs(value):
            self.store.resolve(ref)
        return value

    def write_bytes(self, name, data, media_type):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name or "\\" in name:
            raise ValueError("node output must be a normalized relative path")
        path = (self.graph.dataset_root / self.prefix / name).resolve()
        if not path.is_relative_to(self.graph.dataset_root):
            raise ValueError("node output escapes dataset store")
        relative_name = path.relative_to(self.graph.dataset_root).as_posix()
        ref = ArtifactRef(self.graph.dataset_store, relative_name, hashlib.sha256(data).hexdigest(), media_type)
        self.budget.reserve_file(relative_name + ":" + ref.sha256, len(data))
        if path.exists():
            self.store.resolve(ref)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        self.writes[relative_name] = ref
        return ref

    def write_json(self, name, value):
        return self.write_bytes(name, canonical_json(value).encode(), "application/json")

    def invoke(self, request):
        self.check()
        if self.node.kind != "model" or len(self.model_requests) >= self.node.max_model_calls:
            raise ValueError("node exceeded its declared model authority")
        if len(request.artifacts) > self.node.max_model_images:
            raise ValueError("node exceeded its declared image budget")
        if request.role != self.node.role or request.max_attempts != 1:
            raise ValueError("node model role/retry policy changed")
        if request.timeout_seconds > self.budget.remaining_seconds():
            raise ValueError("model timeout exceeds remaining run budget")
        receipt = self.graph.gateway(self.parent).invoke(request)
        self.model_requests.append({"request": request.to_dict(), "receipt_hash": receipt.without_reuse_hash()})
        self.check()
        return receipt


class DataStepOperator:
    """A reusable registered step implementing execute/recover/verify_effect."""
    def __init__(self, context):
        self.ctx = context
        self.definition = context.graph.registry.step(context.node.operator, context.node.operator_version)
        self.spec = self.definition.spec

    def verify_effect(self, call, receipt):
        path = Path(call.work_dir) / "effect.json"
        if not path.exists():
            return EffectVerification(EvidenceStatus.UNOBSERVED, "step effect is absent")
        try:
            value = json.loads(path.read_text())
            if value["binding"] != payload_sha256(call.to_dict()) or value["operator_revision"] != self.spec.code_revision:
                raise ValueError("step identity or implementation changed")
            if set(value["outputs"]) != set(self.definition.contract.output_schemas):
                raise ValueError("step output ports changed")
            hashes = {"effect.json": file_sha256(path)}
            for ref in _refs(value):
                asset = self.ctx.store.resolve(ref)
                hashes[f"{ref.store}/{ref.path}"] = ref.sha256
                if ref.media_type == "application/json":
                    for nested in _refs(json.loads(asset.read_text())):
                        self.ctx.store.resolve(nested)
            for model in value["model_requests"]:
                request = ModelRequest.from_dict(model["request"])
                cached = self.ctx.graph.gateway(self.ctx.parent).recover(request)
                if cached is None or cached.without_reuse_hash() != model["receipt_hash"]:
                    raise ValueError("step model receipt is missing or changed")
            if receipt is not None and receipt.effect_hash != hashes["effect.json"]:
                raise ValueError("step receipt effect hash changed")
            return EffectVerification(EvidenceStatus.VERIFIED, "step artifacts and model receipts verified", artifact_hashes=hashes)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def _receipt(self, call, reused):
        path = Path(call.work_dir) / "effect.json"
        value = json.loads(path.read_text())
        return OperatorReceipt(call_id=call.call_id, operator_id=self.spec.operator_id, operator_version=self.spec.version,
                               status=StageStatus.SUCCESS, started_at=value["started_at"], finished_at=value["finished_at"],
                               return_code=0, effect_hash=file_sha256(path), reused=reused,
                               output_artifacts=tuple(value["outputs"].values()),
                               stage_result={"outputs": value["outputs"], "model_requests": value["model_requests"]})

    def recover(self, call):
        verification = self.verify_effect(call, _load_receipt(_receipt_path(call)))
        if verification.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(_receipt_path(call), self._receipt(call, True)))
        if verification.status is EvidenceStatus.UNOBSERVED:
            return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason="registered step can resume its bound deterministic writes")
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason=verification.reason)

    def execute(self, call):
        recovery = self.recover(call)
        if recovery.status is RecoveryStatus.RECOVERED:
            return recovery.receipt
        if recovery.status is not RecoveryStatus.RETRY_SAFE:
            raise ValueError(recovery.reason)
        started = datetime.now(timezone.utc).isoformat()
        self.definition.validate_ports(self.ctx.inputs, self.ctx.store)
        outputs = self.definition.implementation(self.ctx)
        self.ctx.check()
        if set(outputs) != set(self.definition.contract.output_schemas):
            raise ValueError("step did not produce its declared output ports")
        self.definition.validate_ports(outputs, self.ctx.store, output=True)
        effect = {"binding": payload_sha256(call.to_dict()), "operator_revision": self.spec.code_revision,
                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "outputs": {k: v.to_dict() for k, v in outputs.items()},
                  "artifacts": [v.to_dict() for v in self.ctx.writes.values()], "model_requests": self.ctx.model_requests}
        atomic_write_json(Path(call.work_dir) / "effect.json", effect)
        return _write_receipt(_receipt_path(call), self._receipt(call, False))


class DataGraphOperator:
    def __init__(self, *, allowed_write_root, contract, roots=None, registry=None, providers=None,source_gateways=None):
        expected = {"intent", "quality_policy", "graph", "budget", "stores", "dataset_store", "models"}
        if set(contract) != expected:
            raise ValueError("unexpected mixed data contract fields")
        self.contract = json.loads(canonical_json(contract))
        self.intent, self.policy = DatasetIntent.from_dict(contract["intent"]), QualityPolicy.from_dict(contract["quality_policy"])
        self.graph, self.budget = read_data_graph(contract["graph"]), DataRunBudget.from_dict(contract["budget"])
        from ..operator_sdk import builtin_step_registry
        self.registry = registry if registry is not None else builtin_step_registry()
        self.providers = dict(providers or {})
        generic = self.graph.schema == "dataevolver.node_graph.v3"
        self.root = Path(allowed_write_root).resolve()
        if roots is None:
            roots = {}
            for key, env in contract["stores"].items():
                identifier(env, "store environment variable")
                if not os.environ.get(env):
                    raise ValueError(f"missing artifact store environment: {env}")
                roots[key] = os.environ[env]
        if set(roots) != set(contract["stores"]):
            raise ValueError("injected stores differ from declared stores")
        self.store, self.dataset_store = ArtifactStore(roots), contract["dataset_store"]
        self.dataset_root = self.store.roots[self.dataset_store]
        for key, path in self.store.roots.items():
            if key != self.dataset_store and (path.is_relative_to(self.dataset_root) or self.dataset_root.is_relative_to(path)):
                raise ValueError("input and writable dataset stores must be disjoint")
        if self.root.is_relative_to(self.dataset_root) or self.dataset_root.is_relative_to(self.root):
            raise ValueError("dataset and campaign runtime roots must be disjoint")
        models = contract["models"]
        if models and (set(models) - {"attempt_limits", "ledger_prefix"}) != {"profiles", "policy", "workers"}:
            raise ValueError("unexpected model configuration fields")
        self.profiles = tuple(ModelProfile.from_dict(p) for p in models.get("profiles", []))
        self.model_policy = ModelPolicy.from_dict(models["policy"]) if models else None
        self.workers = models.get("workers", {})
        self.attempt_limits = None
        if 'attempt_limits' in models:
            from ..attempt_limits import ModelAttemptLimits
            self.attempt_limits = ModelAttemptLimits(**models['attempt_limits'])
        self.ledger_prefix = models.get('ledger_prefix')
        if self.ledger_prefix is not None:
            prefix = PurePosixPath(self.ledger_prefix)
            if (not self.ledger_prefix or prefix.is_absolute() or '..' in prefix.parts
                    or str(prefix) != self.ledger_prefix or '\\' in self.ledger_prefix
                    or str(prefix) == '.'):
                raise ValueError('shared model ledger requires a normalized dataset-relative prefix')
        roles = set(self.model_policy.bindings) if self.model_policy else set()
        profiles = {p.profile_id: p for p in self.profiles}
        if len(profiles) != len(self.profiles) or (self.model_policy and set(self.model_policy.bindings.values()) - set(profiles)):
            raise ValueError("model profile identities or role bindings are invalid")
        self.graph.validate_policy(self.policy, self.budget, roles)
        if self.model_policy and self.model_policy.max_calls > self.budget.max_model_calls:
            raise ValueError("model gateway exceeds run budget")
        if sum(n.max_model_calls for n in self.graph.nodes) and not models:
            raise ValueError("model nodes require an explicit gateway")
        ports = {n.node_id: set(self.registry.step(n.operator, n.operator_version).contract.output_schemas) for n in self.graph.nodes}
        approved_sources = {canonical_json(ref.to_dict()) for ref in self.intent.sources}
        for node in self.graph.nodes:
            self.registry.step(node.operator, node.operator_version).validate_node(node)
            if node.operator in curation.STEPS and node.operator_version == '1':
                curation.validate_step(node)
            if node.kind == "model" and node.max_model_images and "vision" not in profiles[self.model_policy.bindings[node.role]].capabilities:
                raise ValueError("image model role lacks declared vision capability")
            for value in node.inputs.values():
                if value.node and value.output not in ports[value.node]:
                    raise ValueError("upstream output port is not registered")
                if value.artifact and value.artifact.store == self.dataset_store:
                    raise ValueError("external graph inputs must use a read-only source store")
                if value.artifact and canonical_json(value.artifact.to_dict()) not in approved_sources:
                    raise ValueError("graph input is outside the DatasetIntent source scope")
        for profile in self.profiles:
            if not generic and profile.backend not in {"mock", "local_python"}:
                raise ValueError("Image curation route requires offline mock or local CPU workers")
            if profile.backend == "local_python" and profile.profile_id not in self.providers:
                worker = self.workers.get(profile.profile_id, {})
                from dataevolver.adapters.visual_models import SCHEMA, validate_visual_worker
                if worker.get('schema_version') == SCHEMA:
                    if not generic:
                        raise ValueError('visual workers require the explicit v3 graph contract')
                    validate_visual_worker(worker)
                elif worker.get('schema_version') == 'dataevolver.generation_benchmark_worker.v1':
                    if not generic:
                        raise ValueError('benchmark workers require an explicit v3 graph contract')
                    from dataevolver.adapters.benchmark_worker import validate_worker
                    validate_worker(worker)
                else:
                    from dataevolver.adapters.clip_worker import validate_worker_config
                    validate_worker_config(worker)
        # Bind injected locations without placing private absolute paths in contracts.
        self.contract_hash = payload_sha256({"contract": self.contract, "stores": {k: str(p) for k, p in self.store.roots.items()}})
        source = Path(__file__).resolve()
        self.spec = OperatorSpec(operator_id=OPERATOR_ID, version="2", retry_safe=True,
                                 expected_output_specs=(ExpectedOutputSpec("effect.json"),),
                                 model_config_hash=self.contract_hash, code_revision=file_sha256(source), source_path=str(source),
                                 allowed_write_roots=(str(self.root), str(self.dataset_root)))
        if generic:
            self.spec = self.spec.with_updates(operator_id="data.graph", version="1",
                model_config_hash=payload_sha256([self.contract_hash, [s.to_dict() for s in self.registry.step_specs()]]))
        self.source_gateways=dict(source_gateways or {})
        self._source_binding=self._source_gateway_binding()
        if self.source_gateways:
            self.spec=self.spec.with_updates(model_config_hash=payload_sha256(
                [self.spec.model_config_hash,self._source_binding]))

    def _source_gateway_binding(self):
        roles=set(self.model_policy.bindings) if self.model_policy else set()
        binding={}
        for role,gateway in self.source_gateways.items():
            if type(gateway) is not ModelGateway or role not in gateway.policy.bindings or role in roles:
                raise ValueError('source gateway needs a disjoint original model role')
            binding[role]={'contract_hash':gateway.contract_hash(),'receipt_root':str(gateway.root.resolve())}
        return binding

    def _check_source_gateways(self):
        if self._source_gateway_binding()!=self._source_binding:
            raise ValueError('frozen source gateway binding changed')

    def _validate(self, call):
        self._check_source_gateways()
        if (call.operator_id, call.operator_version) != (self.spec.operator_id, self.spec.version) or payload_sha256(call.parameters) != payload_sha256(self.contract):
            raise ValueError("data graph call differs from registered contract")
        path = Path(call.work_dir).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("data graph work directory escapes campaign root")

    def gateway(self, call):
        self._check_source_gateways()
        root = self.ledger_root(call)
        if any(g.root.resolve()==root for g in self.source_gateways.values()):
            raise ValueError('read-only source and writable model ledgers must differ')
        if self.source_gateways and self.model_policy is None:
            from dataevolver.adapters.model_gateway import RoutedModelGateway
            return RoutedModelGateway(read_only=self.source_gateways)
        native_root = root / 'native_models' if self.ledger_prefix is not None else Path(call.work_dir) / 'native_models'
        providers = dict(self.providers)
        for profile in self.profiles:
            if profile.backend == "local_python" and profile.profile_id not in providers:
                from dataevolver.adapters.visual_models import SCHEMA, LocalVisualProvider
                if self.workers[profile.profile_id].get('schema_version') == SCHEMA:
                    providers[profile.profile_id] = lambda p=profile: LocalVisualProvider(
                        p, self.workers[p.profile_id], self.store, self.dataset_store,
                        native_root / p.profile_id)
                elif self.workers[profile.profile_id].get('schema_version') == 'dataevolver.generation_benchmark_worker.v1':
                    from dataevolver.adapters.benchmark_worker import LocalBenchmarkProvider
                    providers[profile.profile_id] = lambda p=profile: LocalBenchmarkProvider(
                        p, self.workers[p.profile_id], self.store, self.dataset_store, native_root / p.profile_id)
                else:
                    from dataevolver.adapters.clip_worker import ClipCPUProvider
                    providers[profile.profile_id] = lambda p=profile: ClipCPUProvider(
                        p, self.workers[p.profile_id], self.store, native_root)
        primary=ModelGateway(profiles=self.profiles, policy=self.model_policy, artifact_store=self.store,
                            receipt_root=root, providers=providers, attempt_limits=self.attempt_limits)
        if self.source_gateways:
            from dataevolver.adapters.model_gateway import RoutedModelGateway
            return RoutedModelGateway(primary,read_only=self.source_gateways)
        return primary

    def ledger_root(self, call):
        root = (self.dataset_root / self.ledger_prefix if self.ledger_prefix is not None
                else Path(call.work_dir) / 'model_receipts').resolve()
        if self.ledger_prefix is not None and not root.is_relative_to(self.dataset_root):
            raise ValueError('shared model ledger escaped its authorized store')
        return root

    def journal(self, call):
        path = Path(call.work_dir) / "data_budget.json"
        if not path.exists() and (list(Path(call.work_dir).glob("node_runs/*/node_intent.json")) or (Path(call.work_dir) / "effect.json").exists()):
            raise ValueError("data budget journal is missing for existing effects")
        return DataBudgetJournal(path, self.budget, payload_sha256([self.contract_hash, call.to_dict()]))

    def model_usage(self, call):
        from dataevolver.adapters.model_contracts import ModelReceipt
        root = self.ledger_root(call)
        attempts = [json.loads(p.read_text()) for p in root.glob("*.intent.json")]
        selected_ids = None
        if self.ledger_prefix is not None:
            # A later batch may add to the shared journal. Its costs must not
            # change the already committed usage/effect of this graph call.
            selected_ids = set()
            for path in (Path(call.work_dir) / 'node_runs').glob('*/effect.json'):
                for entry in json.loads(path.read_text()).get('model_requests', []):
                    selected_ids.add(entry['request']['request_id'])
            attempts = [json.loads(p.read_text()) for p in root.glob('*.intent.json')
                        if p.name[:-len('.intent.json')] in selected_ids]
        calls = sum(r["attempts"] for r in attempts)
        receipts = [ModelReceipt.from_dict(json.loads(p.read_text())["receipt"]) for p in root.glob("*.receipt.json")
                    if selected_ids is None or p.name[:-len('.receipt.json')] in selected_ids]
        if any(r.usage.get("images") is None for r in receipts):
            raise ValueError("completed image model receipt has unknown image usage")
        images = sum(r.usage["images"] for r in receipts)
        if calls > self.budget.max_model_calls or images > self.budget.max_model_images:
            raise ValueError("model usage exceeds approved data-run budget")
        result = {"model_calls": calls, "images": images}
        if self.attempt_limits is not None:
            result['native_units_reserved'] = sum(r['attempts'] * r.get('native_units_per_attempt', 1)
                                                  for r in attempts)
        for name in ("input_tokens", "output_tokens"):
            values = [r.usage.get(name) for r in receipts]
            result[name] = sum(values) if all(v is not None for v in values) else None
        return result

    def _step(self, parent, node, outputs, budget):
        inputs = {k: v.artifact if v.artifact else outputs[v.node][v.output] for k, v in node.inputs.items()}
        for ref in inputs.values():
            self.store.resolve(ref)
        call = OperatorCall(campaign_id=parent.campaign_id, call_id="node-" + payload_sha256([parent.call_id, node.node_id])[:24],
                            idempotency_key=payload_sha256([parent.idempotency_key, node.compute_hash()]),
                            operator_id=node.operator, operator_version=node.operator_version, round_index=parent.round_index,
                            work_dir=str(Path(parent.work_dir) / "node_runs" / node.node_id),
                            parameters={"node_spec": node.to_dict(), "inputs": {k: v.to_dict() for k, v in inputs.items()}})
        return call, DataStepOperator(NodeContext(self, parent, call, node, inputs, budget))

    def _run_node(self, call, node, step, receipt, budget):
        outputs = {k: ArtifactRef.from_dict(v) for k, v in receipt.stage_result["outputs"].items()}
        run = DataNodeRun(call.campaign_id, call.call_id, node.node_id, node.compute_hash(), step.spec.code_revision,
                          step.ctx.inputs, outputs, replace(receipt, reused=False).sha256(),
                          tuple(r["receipt_hash"] for r in receipt.stage_result["model_requests"]))
        path = Path(call.work_dir) / "node_run.json"
        if path.exists() and DataNodeRun.from_dict(json.loads(path.read_text())).to_dict() != run.to_dict():
            raise ValueError("NodeRun identity or receipt binding changed")
        if budget.data["attempts"].get(node.node_id, 0) < 1:
            raise ValueError("NodeRun has no corresponding Operator budget reservation")
        return run, path

    def _inspect(self, call):
        self._validate(call)
        try:
            if not (Path(call.work_dir) / "effect.json").exists():
                return EffectVerification(EvidenceStatus.UNOBSERVED, "data graph is incomplete"), ()
            budget, outputs, artifacts = self.journal(call), {}, []
            for ref in self.intent.sources:
                source = self.store.resolve(ref)
                if ref.media_type == "application/json":
                    for nested in _refs(json.loads(source.read_text())):
                        self.store.resolve(nested)
            for node in self.graph.ordered():
                child, step = self._step(call, node, outputs, budget)
                receipt = _load_receipt(_receipt_path(child))
                if receipt is None or step.verify_effect(child, receipt).status is not EvidenceStatus.VERIFIED:
                    raise ValueError("data step receipt or artifacts changed")
                run, path = self._run_node(child, node, step, receipt, budget)
                if not path.is_file():
                    raise ValueError("data graph is missing a committed NodeRun")
                outputs[node.node_id] = run.outputs
                artifacts.extend([{"path": str(path), "sha256": file_sha256(path)},
                                  {"path": str(Path(child.work_dir) / "effect.json"), "sha256": receipt.effect_hash}])
            effect_path = Path(call.work_dir) / "effect.json"
            effect = json.loads(effect_path.read_text())
            if effect["contract_hash"] != self.contract_hash or effect["artifacts"] != artifacts:
                raise ValueError("data graph effect membership changed")
            for name in ("operator_calls", "input_rows", "output_bytes", "limits"):
                if effect["budget"][name] != budget.summary()[name]:
                    raise ValueError("data budget accounting changed after completion")
            if effect["model_usage"] != self.model_usage(call):
                raise ValueError("model budget accounting changed after completion")
            artifacts.append({"path": str(effect_path), "sha256": file_sha256(effect_path)})
            return EffectVerification(EvidenceStatus.VERIFIED, "data graph verified",
                                      artifact_hashes={a["path"]: a["sha256"] for a in artifacts}), tuple(artifacts)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc)), ()

    def _receipt(self, call, reused):
        verification, artifacts = self._inspect(call)
        if verification.status is not EvidenceStatus.VERIFIED:
            raise ValueError(verification.reason)
        now = datetime.now(timezone.utc).isoformat()
        usage = self.model_usage(call)
        local_calls = usage["model_calls"] if self.profiles and all(p.backend == "local_python" for p in self.profiles) else 0
        return OperatorReceipt(call_id=call.call_id, operator_id=self.spec.operator_id, operator_version=self.spec.version, status=StageStatus.SUCCESS,
                               started_at=now, finished_at=now, return_code=0, output_artifacts=artifacts,
                               effect_hash=payload_sha256(verification.artifact_hashes), reused=reused,
                               stage_result={"operator_id": self.spec.operator_id, "operator_version": self.spec.version, "status": "SUCCESS",
                               "metrics": {"completion_ratio": 1.0, "resource_policy_respected": 1.0,
                                           "local_model_calls": local_calls, "model_images": usage["images"]},
                               "hard_constraints": {"artifact_valid": True},
                               "matched_trial_key": call.matched_trial_key.to_dict() if call.matched_trial_key else None})

    def execute(self, call):
        from dataevolver.adapters.model_gateway import ModelError
        from ..data_budget import DataBudgetExceeded

        self._validate(call)
        with CampaignLease(Path(call.work_dir) / ".data-lease", ttl_seconds=self.budget.max_wall_seconds + 60):
            try:
                return self._execute_locked(call)
            except (DataBudgetExceeded, ModelError, ValueError) as exc:
                # Completed truth is never replaced by a fresh failure receipt.
                if (Path(call.work_dir) / "effect.json").exists() or not (Path(call.work_dir) / "data_budget.json").exists():
                    raise
                if (Path(call.work_dir) / "failure.json").exists() and _load_receipt(_receipt_path(call)) is not None:
                    raise
                code = "DATA_BUDGET_EXHAUSTED" if isinstance(exc, DataBudgetExceeded) else (
                    "MODEL_" + exc.code if isinstance(exc, ModelError) else "INVALID_DATA_EFFECT")
                failure = {"call_hash": payload_sha256(call.to_dict()), "code": code, "detail": str(exc)}
                path = Path(call.work_dir) / "failure.json"
                atomic_write_json(path, failure)
                now = datetime.now(timezone.utc).isoformat()
                receipt = OperatorReceipt(call_id=call.call_id, operator_id=self.spec.operator_id, operator_version=self.spec.version,
                    status=StageStatus.FAILED, started_at=now, finished_at=now, return_code=1, effect_hash=file_sha256(path),
                    stage_result={"operator_id": self.spec.operator_id, "operator_version": self.spec.version, "status": "FAILED",
                    "failure_code": code, "failure_detail": str(exc), "metrics": {"completion_ratio": 0., "resource_policy_respected": 1.},
                    "hard_constraints": {"artifact_valid": False}})
                return _write_receipt(_receipt_path(call), receipt)

    def _execute_locked(self, call):
        if (Path(call.work_dir) / "effect.json").exists():
            return _write_receipt(_receipt_path(call), self._receipt(call, True))
        budget, outputs, artifacts, executed = self.journal(call), {}, [], False
        prior = _load_receipt(_receipt_path(call))
        if prior is not None and prior.status is StageStatus.FAILED:
            failure = Path(call.work_dir) / "failure.json"
            if not failure.is_file() or file_sha256(failure) != prior.effect_hash or json.loads(failure.read_text())["call_hash"] != payload_sha256(call.to_dict()):
                raise ValueError("data failure receipt binding changed")
            return prior.with_updates(reused=True)
        for node in self.graph.ordered():
            child, step = self._step(call, node, outputs, budget)
            intent_path = Path(child.work_dir) / "node_intent.json"
            if intent_path.exists() and json.loads(intent_path.read_text()) != child.to_dict():
                raise ValueError("child Operator intent binding changed")
            if not intent_path.exists():
                atomic_write_json(intent_path, child.to_dict())
            recovery = step.recover(child)
            if recovery.status is RecoveryStatus.AMBIGUOUS:
                raise ValueError(recovery.reason)
            if recovery.status is RecoveryStatus.RECOVERED:
                receipt = recovery.receipt
            else:
                budget.reserve_operator(node.node_id)
                receipt = step.execute(child)
                executed = True
            run, path = self._run_node(child, node, step, receipt, budget)
            if not path.exists():
                atomic_write_json(path, run.to_dict())
            outputs[node.node_id] = run.outputs
            artifacts.extend([{"path": str(path), "sha256": file_sha256(path)},
                              {"path": str(Path(child.work_dir) / "effect.json"), "sha256": receipt.effect_hash}])
        if executed:
            budget.check()
        atomic_write_json(Path(call.work_dir) / "effect.json", {"contract_hash": self.contract_hash,
                          "artifacts": artifacts, "budget": budget.summary(),
                          "model_usage": self.model_usage(call),
                          "evidence_level": "mock_verified" if any(p.backend == "mock" for p in self.profiles) else "real_execution_verified"})
        return _write_receipt(_receipt_path(call), self._receipt(call, False))

    def recover(self, call):
        prior = _load_receipt(_receipt_path(call))
        if prior is not None and prior.status is StageStatus.FAILED:
            verification = self.verify_effect(call, prior)
            if verification.status is EvidenceStatus.VERIFIED:
                return RecoveryResult(RecoveryStatus.RECOVERED, prior.with_updates(reused=True))
            return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason=verification.reason)
        verification, _ = self._inspect(call)
        if verification.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(_receipt_path(call), self._receipt(call, True)))
        if verification.status is EvidenceStatus.UNOBSERVED:
            return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason="resume registered child Operators")
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason=verification.reason)

    def verify_effect(self, call, receipt):
        if receipt is not None and receipt.status is StageStatus.FAILED:
            self._validate(call)
            try:
                self.journal(call)
                path = Path(call.work_dir) / "failure.json"
                failure = json.loads(path.read_text())
                if (receipt.call_id != call.call_id or receipt.operator_id != self.spec.operator_id or
                        receipt.operator_version != self.spec.version or file_sha256(path) != receipt.effect_hash or
                        failure["call_hash"] != payload_sha256(call.to_dict()) or
                        failure["code"] != receipt.stage_result["failure_code"] or
                        receipt.stage_result["hard_constraints"].get("artifact_valid") is not False):
                    raise ValueError("data failure receipt binding changed")
                return EffectVerification(EvidenceStatus.VERIFIED, "bounded failure evidence verified; candidate validation failed",
                                          artifact_hashes={str(path): receipt.effect_hash})
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))
        verification, _ = self._inspect(call)
        if receipt and verification.status is EvidenceStatus.VERIFIED and receipt.effect_hash != payload_sha256(verification.artifact_hashes):
            return EffectVerification(EvidenceStatus.CONTRADICTED, "data graph receipt changed")
        return verification
