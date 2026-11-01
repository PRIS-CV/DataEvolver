"""Frozen image-pool recipes and local model roles under EvolutionController.

This Operator never adopts a recipe. It returns reproducible selections and
limited clarity evidence; the existing CandidateGate/Controller decides.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import math
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelError, ModelGateway
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, canonical_json, file_sha256, payload_sha256
from dataevolver.dataset.curation_loader import load_export
from dataevolver.dataset.quality import RANKINGS, counts, select_balanced, validate_quotas
from ..data_budget import DataBudgetJournal, DataBudgetExceeded
from ..data_contracts import DataRunBudget
from ..models import (EffectVerification, EvidenceStatus, ExpectedOutputSpec, FailureSignature,
                      Intervention, InterventionLevel, OperatorReceipt, OperatorSpec,
                      RecoveryResult, RecoveryStatus, StageStatus)
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import atomic_write_json


OPERATOR_ID = "data.image_recipe"


class ImageRecipeOperator:
    def __init__(self, *, allowed_write_root, contract, roots=None):
        expected = {"pool", "stores", "dataset_store", "quotas", "seed", "baseline", "models", "budget", "review"}
        if not expected.issubset(contract) or set(contract)-expected-{"deadline_epoch", "development_limit_per_label"} or contract["baseline"] != {"ranking": "hash"}:
            raise ValueError("unexpected frozen image-recipe contract")
        if "deadline_epoch" in contract and (type(contract["deadline_epoch"]) not in (int,float) or
                                            not math.isfinite(contract["deadline_epoch"])):
            raise ValueError("invalid inherited absolute deadline")
        self.contract = json.loads(canonical_json(contract))
        self.root = Path(allowed_write_root).resolve()
        self.store = ArtifactStore(roots if roots is not None else {k: os.environ[v] for k, v in contract["stores"].items()})
        if set(self.store.roots) != set(contract["stores"]):
            raise ValueError("recipe artifact roots differ from contract")
        self.pool_ref = ArtifactRef.from_dict(contract["pool"])
        self.pool_root = self.store.resolve(self.pool_ref).parent
        self.dataset_root = self.store.roots[contract["dataset_store"]]
        paths = [self.pool_root, self.dataset_root, self.root]
        if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[i+1:]):
            raise ValueError("pool, candidate dataset and campaign roots must be disjoint")
        self.rows = load_export(self.pool_root, decode_roles={"development", "calibration", "canary", "reuse"})
        if any(r.get("evaluation_role") not in {"development", "calibration", "canary", "reuse", "sealed"} for r in self.rows):
            raise ValueError("frozen pool roles are missing")
        self.calibration = json.loads((self.pool_root / "quality_calibration.json").read_text())
        if not self.calibration.get("passed") or self.calibration.get("human_gold") is not False:
            raise ValueError("objective calibration is missing or misrepresented")
        self.minimum_clarity = self.calibration["threshold"]
        validate_quotas(contract["quotas"])
        if type(contract["seed"]) is not int or contract["seed"] < 0:
            raise ValueError("recipe selection seed must be fixed")
        self.budget = DataRunBudget.from_dict(contract["budget"])
        limit = contract.get("development_limit_per_label")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 256):
            raise ValueError("development review limit must be a positive bounded count")
        models = contract["models"]
        if set(models) != {"profiles", "policy", "workers"}:
            raise ValueError("unexpected recipe model configuration")
        self.profiles = tuple(ModelProfile.from_dict(p) for p in models["profiles"])
        self.policy = ModelPolicy.from_dict(models["policy"])
        self.workers = models["workers"]
        if set(self.policy.bindings) != {"planner", "reviewer"} or self.policy.max_calls > self.budget.max_model_calls:
            raise ValueError("recipe model roles or shared call budget changed")
        if len({p.profile_id for p in self.profiles}) != len(self.profiles):
            raise ValueError("duplicate model profile")
        for profile in self.profiles:
            if profile.backend not in {"mock", "local_python"} or "vision" not in profile.capabilities:
                raise ValueError("recipe route requires a local vision/text model")
            if profile.backend == "local_python":
                from dataevolver.adapters.chat_worker import validate_chat_config
                validate_chat_config(self.workers[profile.profile_id])
        review = contract["review"]
        if (set(review) != {"batch_size", "max_tokens", "timeout_seconds", "minimum_pair_accuracy"} or
                type(review["batch_size"]) is not int or not 1 <= review["batch_size"] <= 8 or
                type(review["max_tokens"]) is not int or not 32 <= review["max_tokens"] <= 512 or
                type(review["timeout_seconds"]) is not int or not 1 <= review["timeout_seconds"] <= 1800 or
                type(review["minimum_pair_accuracy"]) not in (int, float) or not .5 <= review["minimum_pair_accuracy"] <= 1):
            raise ValueError("review budget and control criterion must be explicit")
        for role in ("development", "canary", "reuse"):
            select_balanced(self.role_rows(role), contract["quotas"],
                            seed=contract["seed"], min_clarity=self.minimum_clarity)
        self.contract_hash = payload_sha256({"contract": self.contract,
                            "roots": {k: str(v) for k, v in self.store.roots.items()}})
        source = Path(__file__).resolve()
        self.spec = OperatorSpec(operator_id=OPERATOR_ID, version="1", retry_safe=True,
            input_artifact_refs=(self.pool_ref.to_dict(),), expected_output_specs=(ExpectedOutputSpec("effect.json"),),
            source_path=str(source), code_revision=file_sha256(source), model_config_hash=self.contract_hash,
            allowed_write_roots=(str(self.root), str(self.dataset_root)))

    def journal(self):
        path = self.root / "recipe_budget.json"
        if not path.exists() and (list(self.root.glob("model_bounds/*.json")) or list(self.root.glob("model_receipts/*.intent.json"))
                                 or list(self.root.glob("**/effect.json")) or list(self.root.glob("**/failure.json"))):
            raise ValueError("recipe budget missing for existing effects")
        return DataBudgetJournal(path, self.budget, self.contract_hash)

    def gateway(self):
        providers = {}
        for profile in self.profiles:
            if profile.backend == "local_python":
                from dataevolver.adapters.chat_worker import ChatCPUProvider
                providers[profile.profile_id] = lambda p=profile: ChatCPUProvider(
                    p, self.workers[p.profile_id], self.store, self.root / "native_models")
        return ModelGateway(profiles=self.profiles, policy=self.policy, artifact_store=self.store,
                            receipt_root=self.root / "model_receipts", providers=providers)

    def invoke(self, request):
        if request.role == "planner" and request.artifacts:
            raise ValueError("planner cannot request pool images")
        if request.role == "reviewer":
            controls = json.loads((self.pool_root/"calibration_controls.json").read_text())
            allowed = {canonical_json(self.image_ref(r).to_dict()) for r in self.rows if r["evaluation_role"] != "sealed"}
            allowed.update(canonical_json(self.image_ref(r).to_dict()) for r in controls)
            if not request.artifacts or any(canonical_json(r.to_dict()) not in allowed for r in request.artifacts):
                raise ValueError("reviewer input exceeds frozen non-sealed pool/control scope")
        budget = self.journal()
        budget.check()
        remaining = min(budget.remaining_seconds(), self.contract.get("deadline_epoch", float("inf"))-time.time())
        if request.max_attempts != 1 or request.timeout_seconds > remaining:
            raise ValueError("recipe model request exceeds remaining deadline/retry budget")
        bounds = self.root / "model_bounds"
        bounds.mkdir(parents=True, exist_ok=True)
        path = bounds / f"{request.request_id}.json"
        value = {"hash": payload_sha256(request.to_dict()), "images": len(request.artifacts)}
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("recipe model reservation binding changed")
        else:
            if (self.root / "model_receipts" / f"{request.request_id}.intent.json").exists():
                raise ValueError("model image reservation is missing")
            used = sum(json.loads(p.read_text())["images"] for p in bounds.glob("*.json"))
            if used + len(request.artifacts) > self.budget.max_model_images:
                raise ValueError("recipe shared model image budget exhausted")
            atomic_write_json(path, value)
        result = self.gateway().invoke(request)
        budget.check()
        return result

    def image_ref(self, row):
        relative = self.pool_root.relative_to(self.store.roots[self.pool_ref.store]) / row["image"]
        return ArtifactRef(self.pool_ref.store, relative.as_posix(), row["image_sha256"], "image/png")

    def _model_task(self, request_id, prompt, schema, images=(), role="reviewer"):
        review = self.contract["review"]
        return ModelRequest(request_id, role, ({"role": "user", "content": prompt},), schema,
                            artifacts=tuple(images), max_tokens=review["max_tokens"],
                            timeout_seconds=review["timeout_seconds"], seed=self.contract["seed"])

    def _cached_review(self, path):
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value["binding"] != self.contract_hash or value["sha256"] != payload_sha256(value["result"]):
            raise ValueError("review effect changed")
        for entry in value["requests"]:
            receipt = self.gateway().recover(ModelRequest.from_dict(entry["request"]))
            if receipt is None or receipt.without_reuse_hash() != entry["receipt"]:
                raise ValueError("review receipt changed")
        return value["result"]

    def _save_review(self, path, result, requests):
        atomic_write_json(path, {"binding": self.contract_hash, "result": result,
                          "sha256": payload_sha256(result), "requests": requests})
        return result

    def reviewer_calibration(self):
        path = self.root / "reviewer_calibration.json"
        cached = self._cached_review(path)
        if cached is not None:
            return cached
        controls = json.loads((self.pool_root / "calibration_controls.json").read_text())
        by_id, results, requests = {r["id"]: r for r in self.rows}, [], []
        for i, control in enumerate(c for c in controls if c["partition"] == "validation"):
            reference = by_id[control["reference_id"]]
            if reference["evaluation_role"] != "calibration":
                raise ValueError("reviewer calibration attempted non-calibration image")
            pair = [self.image_ref(reference), self.image_ref(control)]
            answer = 0
            if int(payload_sha256(control["id"])[0], 16) % 2:
                pair.reverse(); answer = 1
            schema = {"type": "object", "required": ["clearer"], "additionalProperties": False,
                      "properties": {"clearer": {"type": "integer", "enum": [0, 1]}}}
            request = self._model_task(f"calibration-{i:03d}",
                "Which image has sharper leaf detail? Return clearer=0 for the first image or 1 for the second. Judge sharpness only.",
                schema, pair)
            receipt = self.invoke(request)
            results.append({"control_id": control["id"], "predicted": receipt.output["clearer"],
                            "expected": answer, "family_id": reference["family_id"]})
            requests.append({"request": request.to_dict(), "receipt": receipt.without_reuse_hash()})
        if not results:
            raise ValueError("reviewer calibration controls absent")
        accuracy = sum(r["predicted"] == r["expected"] for r in results) / len(results)
        return self._save_review(path, {"accuracy": accuracy, "pairs": results,
                "passed": accuracy >= self.contract["review"]["minimum_pair_accuracy"],
                "scope": "held_reference_vs_controlled_blur_only", "semantic_quality_calibrated": False}, requests)

    def reviews(self, reuse=False):
        self.reviewer_calibration()
        scope = "reuse" if reuse else "search"
        path = self.root / f"review-{scope}.json"
        cached = self._cached_review(path)
        if cached is not None:
            return cached
        rows = self.role_rows("reuse") if reuse else self.role_rows("development") + self.role_rows("canary")
        scores, requests = {}, []
        size = self.contract["review"]["batch_size"]
        for offset in range(0, len(rows), size):
            batch = rows[offset:offset+size]
            schema = {"type": "object", "required": ["scores"], "additionalProperties": False,
                      "properties": {"scores": {"type": "array", "minItems": len(batch), "maxItems": len(batch),
                                      "items": {"type": "integer", "minimum": 0, "maximum": 4}}}}
            request = self._model_task(f"review-{scope}-{offset:03d}",
                "Rate visible leaf detail sharpness in each image: 0=unusable blur, 1=strong blur, 2=some blur, 3=clear, 4=very clear. "
                "Ignore disease and background. Return scores in image order. Scores confer no label, privacy or safety approval.",
                schema, [self.image_ref(r) for r in batch])
            receipt = self.invoke(request)
            scores.update({r["image_sha256"]: s for r, s in zip(batch, receipt.output["scores"])})
            requests.append({"request": request.to_dict(), "receipt": receipt.without_reuse_hash()})
        return self._save_review(path, scores, requests)

    def compile_recipe(self, patch, attempted=()):
        if set(patch) != {"ranking"} or patch["ranking"] not in RANKINGS - {"hash"}:
            raise ValueError("recipe patch may only select a registered ranking")
        if patch["ranking"] in attempted:
            raise ValueError("duplicate recipe candidate")
        if patch["ranking"] in {"review", "hybrid"} and not self.reviewer_calibration()["passed"]:
            raise ValueError("reviewer ranking failed its independent control calibration")
        return dict(patch)

    def _validate(self, call):
        if (call.operator_id, call.operator_version) != (OPERATOR_ID, "1"):
            raise ValueError("recipe Operator identity changed")
        if set(call.parameters) != {"recipe", "phase"} or call.parameters["phase"] not in {"search", "reuse"}:
            raise ValueError("recipe call cannot change frozen policy or pool")
        recipe = call.parameters["recipe"]
        if recipe != self.contract["baseline"]:
            self.compile_recipe(recipe)
        if not Path(call.work_dir).resolve().is_relative_to(self.root):
            raise ValueError("recipe work directory escapes campaign")
        self.store.resolve(self.pool_ref)

    def _selection(self, recipe, role):
        scores = self.reviews(reuse=role == "reuse")
        rows = [{**r, "review_score": scores[r["image_sha256"]]} for r in self.role_rows(role)]
        return select_balanced(rows, self.contract["quotas"], ranking=recipe["ranking"],
                               seed=self.contract["seed"], min_clarity=self.minimum_clarity)

    def role_rows(self, role):
        rows = [r for r in self.rows if r["evaluation_role"] == role]
        limit = self.contract.get("development_limit_per_label") if role == "development" else None
        if limit is None:
            return rows
        groups = {}
        for row in sorted(rows, key=lambda r:payload_sha256([self.contract["seed"], r["id"]])):
            group = groups.setdefault((row["source_id"],row["label_id"]), [])
            if len(group) < limit and row["quality"]["clarity_proxy"] >= self.minimum_clarity:
                group.append(row)
        return [row for group in groups.values() for row in group]

    def _write(self, relative, data):
        path = (self.dataset_root / relative).resolve()
        if not path.is_relative_to(self.dataset_root):
            raise ValueError("recipe output escapes dataset store")
        digest = hashlib.sha256(data).hexdigest()
        self.journal().reserve_file(relative + ":" + digest, len(data))
        if path.exists():
            if file_sha256(path) != digest:
                raise ValueError("recipe output changed")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name); handle.write(data); handle.flush(); os.fsync(handle.fileno())
            temporary.replace(path)
        return digest

    def _export(self, call, role, rows):
        prefix = "recipes/" + payload_sha256(call.to_dict())[:24] + "/" + role
        members = {}
        def write(name, data):
            members[name] = self._write(prefix + "/" + name, data)
        source_meta = json.loads((self.pool_root / "dataset.json").read_text())
        for name in source_meta["members"]:
            if name in {"sources.json", "label_mapping.json"} or name.startswith("source_evidence/"):
                write(name, (self.pool_root / name).read_bytes())
        for row in rows:
            write(row["image"], (self.pool_root / row["image"]).read_bytes())
        write("manifest.jsonl", ("\n".join(canonical_json(r) for r in rows)+"\n").encode())
        write("quarantine.json", b"[]")
        receipts = [json.loads(p.read_text())["receipt"] for p in sorted((self.root/"model_receipts").glob("*.receipt.json"))]
        write("model_receipts.json", canonical_json(receipts).encode())
        write("recipe.json", canonical_json({"recipe": call.parameters["recipe"], "role": role,
              "pool": self.pool_ref.to_dict(), "contract_hash": self.contract_hash}).encode())
        write("DATA_CARD.md", b"# Frozen-pool engineering candidate\n\nClarity proxies and controlled blur calibration only.\n"
              b"Human semantic, privacy/safety and downstream training validation remain incomplete.\n")
        metadata = {**source_meta, "count": len(rows), "members": members, "pool_hash": self.pool_ref.sha256}
        self._write(prefix+"/dataset.json", canonical_json(metadata).encode())
        root = self.dataset_root / prefix
        load_export(root)
        return {"root": prefix, "sha256": file_sha256(root/"dataset.json")}

    def _expected(self, call):
        recipe, phase = call.parameters["recipe"], call.parameters["phase"]
        roles = ["reuse"] if phase == "reuse" else ["development", "canary"]
        selections = {role: self._selection(recipe, role) for role in roles}
        measured = selections[roles[-1]]
        mean = lambda rows, name: sum(r["quality"][name] for r in rows)/len(rows)
        metrics = {"clarity_proxy": mean(measured, "clarity_proxy"), "contrast": mean(measured, "contrast"),
                   "class_coverage": 1., "source_coverage": 1., "count_ratio": 1.,
                   "development_clarity": mean(selections[roles[0]], "clarity_proxy")}
        if phase == "reuse":
            base = self._selection(self.contract["baseline"], "reuse")
            metrics["reuse_delta"] = metrics["clarity_proxy"] - mean(base, "clarity_proxy")
            selections["reuse_baseline"] = base
        return selections, metrics

    def _receipt(self, call, reused):
        path = Path(call.work_dir)/"effect.json"
        effect = json.loads(path.read_text())
        return OperatorReceipt(call.call_id, OPERATOR_ID, "1", StageStatus.SUCCESS, effect["at"], effect["at"],
                return_code=0, effect_hash=file_sha256(path), reused=reused,
                output_artifacts=tuple(effect["exports"].values()), stage_result=effect["stage_result"])

    def execute(self, call):
        self._validate(call)
        recovered = self.recover(call)
        if recovered.status is RecoveryStatus.RECOVERED:
            return recovered.receipt
        if recovered.status is not RecoveryStatus.RETRY_SAFE:
            raise ValueError(recovered.reason)
        budget = self.journal()
        try:
            if time.time() >= self.contract.get("deadline_epoch", float("inf")):
                raise DataBudgetExceeded("inherited recipe deadline exhausted")
            budget.reserve_operator(call.call_id)
            for row in self.rows:
                budget.reserve_rows(row["id"], 1)
            selections, metrics = self._expected(call)
            exports = {role: self._export(call, role, rows) for role, rows in selections.items()}
        except (ValueError, ModelError, DataBudgetExceeded) as exc:
            failure = {"binding": payload_sha256([self.contract_hash, call.to_dict()]),
                       "code": getattr(exc, "code", type(exc).__name__), "detail": str(exc)}
            path = Path(call.work_dir)/"failure.json"
            atomic_write_json(path, failure)
            now = datetime.now(timezone.utc).isoformat()
            receipt = OperatorReceipt(call.call_id, OPERATOR_ID, "1", StageStatus.FAILED, now, now,
                return_code=1, effect_hash=file_sha256(path), stage_result={"operator_id":OPERATOR_ID,
                "operator_version":"1", "status":"FAILED", "metrics":{}, "hard_constraints":{"artifact_valid":False},
                "failure_code":failure["code"], "failure_detail":str(exc)})
            return _write_receipt(_receipt_path(call), receipt)
        stage = {"operator_id": OPERATOR_ID, "operator_version": "1", "status": "SUCCESS", "metrics": metrics,
                 "hard_constraints": {"artifact_valid": True, "quotas_met": True, "pool_fixed": True, "sealed_unread": True}}
        effect = {"binding": payload_sha256([self.contract_hash, call.to_dict()]), "at": datetime.now(timezone.utc).isoformat(),
                  "exports": exports, "selections": {role:[r["id"] for r in rows] for role,rows in selections.items()}, "stage_result": stage}
        atomic_write_json(Path(call.work_dir)/"effect.json", effect)
        return _write_receipt(_receipt_path(call), self._receipt(call, False))

    def verify_effect(self, call, receipt):
        path = Path(call.work_dir)/"effect.json"
        failure = Path(call.work_dir)/"failure.json"
        if failure.exists():
            try:
                self._validate(call); self.journal()
                value = json.loads(failure.read_text())
                if (receipt is None or receipt.status is not StageStatus.FAILED or
                        value["binding"] != payload_sha256([self.contract_hash, call.to_dict()]) or
                        receipt.effect_hash != file_sha256(failure) or receipt.stage_result["hard_constraints"] != {"artifact_valid":False}):
                    raise ValueError("recipe failure receipt binding changed")
                return EffectVerification(EvidenceStatus.VERIFIED, "failure verified; candidate evidence is incomplete",
                                          artifact_hashes={str(failure):file_sha256(failure)})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))
        if not path.exists():
            return EffectVerification(EvidenceStatus.UNOBSERVED, "recipe effect absent")
        try:
            self._validate(call); self.journal()
            effect = json.loads(path.read_text())
            if effect["binding"] != payload_sha256([self.contract_hash, call.to_dict()]):
                raise ValueError("recipe call binding changed")
            selections, metrics = self._expected(call)
            if (effect["stage_result"]["metrics"] != metrics or effect["selections"] != {r:[x["id"] for x in rows] for r,rows in selections.items()}
                    or set(effect["exports"]) != set(selections) or effect["stage_result"]["hard_constraints"] !=
                    {"artifact_valid":True, "quotas_met":True, "pool_fixed":True, "sealed_unread":True}):
                raise ValueError("recipe metrics or membership changed")
            for role, output in effect["exports"].items():
                root = (self.dataset_root/output["root"]).resolve()
                if not root.is_relative_to(self.dataset_root) or file_sha256(root/"dataset.json") != output["sha256"]:
                    raise ValueError("recipe export binding changed")
                rows = load_export(root)
                if [r["id"] for r in rows] != effect["selections"][role] or counts(rows) != self.contract["quotas"]:
                    raise ValueError("recipe export quotas or identities changed")
            if receipt and receipt.effect_hash != file_sha256(path):
                raise ValueError("recipe receipt effect hash changed")
            return EffectVerification(EvidenceStatus.VERIFIED, "frozen pool, selections, quality metrics and model receipts verified",
                                      artifact_hashes={str(path): file_sha256(path)})
        except (OSError, ValueError, KeyError, TypeError, ModelError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def recover(self, call):
        result = self.verify_effect(call, _load_receipt(_receipt_path(call)))
        if result.status is EvidenceStatus.VERIFIED:
            prior = _load_receipt(_receipt_path(call))
            if prior is not None and prior.status is StageStatus.FAILED:
                return RecoveryResult(RecoveryStatus.RECOVERED, prior.with_updates(reused=True))
            return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(_receipt_path(call), self._receipt(call, True)))
        if result.status is EvidenceStatus.UNOBSERVED:
            return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason="resume deterministic writes and native model receipts")
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason=result.reason)


class RecipePlanner:
    planner_id = "local-image-recipe-v1"

    def __init__(self, operator):
        self.operator = operator

    def config(self):
        return {"kind": "recipe_rsi", "contract_hash": self.operator.contract_hash}

    def diagnose(self, result, report):
        return FailureSignature("CLARITY_CURATION_GAP", detail="Compare a bounded selection recipe under fixed coverage and clarity policy.")

    def propose_recipe(self, state):
        attempted = state.data.get("attempted_rankings", [])
        options = sorted(RANKINGS - {"hash"} - set(attempted))
        if not self.operator.reviewer_calibration()["passed"]:
            options = [v for v in options if v == "clarity"]
        if not options:
            return None
        schema = {"type": "object", "required": ["ranking"], "additionalProperties": False,
                  "properties": {"ranking": {"type": "string", "enum": options}}}
        prompt = {"task": "Choose one untried ranking to improve leaf image detail clarity. Keep all frozen source/class quotas and policy.",
                  "rankings": {"clarity": "objective edge detail", "review": "calibrated local reviewer sharpness",
                               "hybrid": "fixed equal combination of clarity and reviewer"},
                  "allowed": options, "development_clarity": state.data["last_stage_result"]["metrics"]["development_clarity"],
                  "history": state.data.get("recipe_history", [])}
        request = self.operator._model_task(f"planner-{state.planner_calls:03d}", canonical_json(prompt), schema, role="planner")
        receipt = self.operator.invoke(request)
        patch = self.operator.compile_recipe(receipt.output, attempted)
        trace = {"request": request.to_dict(), "receipt_hash": receipt.without_reuse_hash(), "patch": patch}
        atomic_write_json(self.operator.root/"planner"/f"p{state.planner_calls:03d}.json", trace)
        return Intervention("recipe-"+patch["ranking"], "CLARITY_CURATION_GAP", OPERATOR_ID, InterventionLevel.L1,
                            parameters={"recipe": patch, "phase": "search"}, candidate_delta=patch,
                            rollback_plan={"action": "retain_incumbent_recipe_and_exports"})
