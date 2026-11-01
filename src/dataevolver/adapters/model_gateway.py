"""Local-first model routing with cache-before-load and explicit effect recovery.

The gateway is a bounded service used under the Harness, not a planner or a
second campaign controller. The reserved external_api backend has no transport.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Protocol

from dataevolver.dataset.contracts import ArtifactRef as ArtifactRef, ArtifactStore, canonical_json, payload_sha256
from dataevolver.harness.state_store import CampaignLease, atomic_write_json
from .model_contracts import ModelPolicy, ModelProfile, ModelReceipt, ModelRequest, origin


class ModelError(RuntimeError):
    def __init__(self, code: str, message: str, *, retry_safe: bool = False):
        super().__init__(message)
        self.code, self.retry_safe = code, retry_safe


class ModelProvider(Protocol):
    def health(self) -> dict[str, Any]: ...
    def invoke(self, request: ModelRequest, request_hash: str) -> dict[str, Any]: ...
    def lookup(self, request: ModelRequest, request_hash: str) -> dict[str, Any] | None: ...


def validate_response_schema(schema, output=None):
    def check_refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef"} and (not isinstance(item, str) or not item.startswith("#")):
                    raise ModelError("INVALID_SCHEMA", "remote JSON schema references are not allowed")
                if key in {"$ref", "$dynamicRef"} and item != "#":
                    if not item.startswith("#/"):
                        raise ModelError("INVALID_SCHEMA", "schema references must use local JSON pointers")
                    target = schema
                    try:
                        for part in item[2:].split("/"):
                            part = part.replace("~1", "/").replace("~0", "~")
                            target = target[int(part)] if isinstance(target, list) else target[part]
                    except (KeyError, IndexError, TypeError, ValueError):
                        raise ModelError("INVALID_SCHEMA", "local schema reference does not resolve") from None
                check_refs(item)
        elif isinstance(value, list):
            for item in value:
                check_refs(item)
    check_refs(schema)
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
        if output is not None:
            Draft202012Validator(schema).validate(output)
    except ImportError as exc:
        raise ModelError("DEPENDENCY_MISSING", "install the harness extra for JSON schema validation") from exc
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError("INVALID_OUTPUT" if output is not None else "INVALID_SCHEMA", "JSON schema validation failed") from exc


class MockModelProvider:
    """Durable fixture effects support crash-after-effect recovery tests."""

    def __init__(self, profile: ModelProfile, effect_root: Path):
        self.profile, self.root = profile, effect_root

    def health(self):
        return {"status": "MOCK", "model": self.profile.model, "revision_verified": False}

    def invoke(self, request, request_hash):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{request.request_id}.json"
        if path.exists():
            response = self.lookup(request, request_hash)
            if response is not None:
                return response
        response = {
            "model": self.profile.model, "output": self.profile.mock_output,
            "usage": {"input_tokens": 0, "output_tokens": 0, "images": 0, "cost": 0.0},
            "finish_reason": "stop", "provider_request_id": request.request_id,
        }
        atomic_write_json(path, {"request_hash": request_hash, "response": response})
        return response

    def lookup(self, request, request_hash):
        path = self.root / f"{request.request_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data["request_hash"] != request_hash or data["response"]["output"] != self.profile.mock_output:
                raise ValueError("mock effect binding changed")
            return data["response"]
        except (OSError, ValueError, KeyError) as exc:
            raise ModelError("INVALID_RECEIPT", "mock provider effect is invalid") from exc


class ExternalAPIProvider:
    """Reserved backend with no request transport."""

    def health(self):
        return {"status": "DISABLED"}

    def invoke(self, request, request_hash):
        raise ModelError("EXTERNAL_API_DISABLED", "external_api backend is disabled")

    def lookup(self, request, request_hash):
        raise ModelError("EXTERNAL_API_DISABLED", "external_api backend is disabled")


class LocalPythonProvider:
    """Trusted registered worker; loading happens only on a confirmed cache miss.

    Profiles cannot name arbitrary imports or Python snippets. The embedding
    application supplies the worker factory and, when available, native lookup.
    """

    def __init__(self, factory: Callable[[], Any], lookup: Callable | None = None):
        self.factory, self.worker, self.native_lookup = factory, None, lookup

    def health(self):
        return {"status": "REGISTERED", "loaded": self.worker is not None}

    def invoke(self, request, request_hash):
        if self.worker is None:
            self.worker = self.factory()
        return self.worker.invoke(request, request_hash)

    def lookup(self, request, request_hash):
        # Loading a GPU model merely to find out whether an effect exists is
        # forbidden. Workers need a separate receipt lookup for safe recovery.
        if self.native_lookup is not None:
            return self.native_lookup(request, request_hash)
        raise ModelError("AMBIGUOUS_EFFECT", "local worker has no independent effect lookup")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalEndpointProvider:
    """OpenAI-compatible JSON/vision transport for an explicitly allowed origin."""

    def __init__(self, profile: ModelProfile, endpoint: str, model: str, store: ArtifactStore | None):
        self.profile, self.endpoint, self.model, self.store = profile, endpoint.rstrip("/"), model, store
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _send(self, suffix, *, payload=None, timeout=10):
        headers = {"Content-Type": "application/json"}
        if self.profile.api_key_env:
            key = os.environ.get(self.profile.api_key_env, "")
            if not key:
                raise ModelError("CONFIG_MISSING", "model credential environment is missing")
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            self.endpoint + suffix, headers=headers,
            data=canonical_json(payload).encode() if payload is not None else None,
            method="POST" if payload is not None else "GET",
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise ModelError("INVALID_OUTPUT", "model response exceeded its bound")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            code = "RATE_LIMIT" if exc.code == 429 else "PROVIDER_HTTP_ERROR"
            raise ModelError(code, f"model endpoint returned HTTP {exc.code}", retry_safe=exc.code in {400, 401, 403, 404, 429}) from None
        except ModelError:
            raise
        except (OSError, ValueError):
            raise ModelError("AMBIGUOUS_EFFECT" if payload is not None else "UNAVAILABLE", "model endpoint did not return a valid complete response") from None

    def health(self):
        payload = self._send("/models")
        models = {row.get("id") for row in payload.get("data", []) if isinstance(row, dict)}
        return {"status": "AVAILABLE" if self.model in models else "MODEL_NOT_SERVED", "model": self.model, "revision_verified": False}

    def _media_content(self, request):
        images = []
        for ref in request.artifacts:
            if self.store is None or ref.media_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise ModelError("UNSUPPORTED_MEDIA", "model requires a registered image artifact store")
            encoded = base64.b64encode(self.store.resolve(ref).read_bytes()).decode("ascii")
            images.append({"type": "image_url", "image_url": {"url": f"data:{ref.media_type};base64,{encoded}"}})
        return images

    def invoke(self, request, request_hash):
        messages = [dict(x) for x in request.messages]
        if request.artifacts:
            messages.append({"role": "user", "content": self._media_content(request)})
        schema_instruction = "Return one JSON object satisfying this schema: " + canonical_json(request.response_schema)
        if messages[0]['role'] == 'system':
            messages[0]['content'] = schema_instruction + '\n\n' + messages[0]['content']
        else:
            messages.insert(0, {"role": "system", "content": schema_instruction})
        payload = self._send("/chat/completions", payload={
            "model": self.model, "messages": messages, "temperature": 0,
            "seed": request.seed, "max_tokens": request.max_tokens,
            "response_format": {"type": "json_object"},
        }, timeout=request.timeout_seconds)
        try:
            choice = payload["choices"][0]
            usage = payload.get("usage") or {}
            return {
                "model": payload["model"], "output": json.loads(choice["message"]["content"]),
                "finish_reason": choice["finish_reason"], "provider_request_id": str(payload.get("id") or ""),
                "usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"), "images": len(request.artifacts), "cost": None},
            }
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelError("INVALID_OUTPUT", "model endpoint response is malformed") from None

    def lookup(self, request, request_hash):
        raise ModelError("AMBIGUOUS_EFFECT", "endpoint has no registered native request lookup; automatic resend is forbidden")


class ModelGateway:
    def __init__(self, *, profiles: tuple[ModelProfile, ...], policy: ModelPolicy,
                 receipt_root: str | Path, artifact_store: ArtifactStore | None = None,
                 providers: dict[str, Callable[[], ModelProvider]] | None = None, attempt_limits=None):
        self.profiles = {x.profile_id: ModelProfile.from_dict(x.to_dict()) for x in profiles}
        if len(self.profiles) != len(profiles) or set(policy.bindings.values()) - set(self.profiles):
            raise ValueError("model profile IDs must be unique and all role bindings must resolve")
        self.policy = ModelPolicy.from_dict(policy.to_dict())
        self.root, self.store, self.factories = Path(receipt_root).resolve(), artifact_store, providers or {}
        self.attempt_limits = attempt_limits

    def _binding(self, role):
        if role not in self.policy.bindings:
            raise ModelError("ROLE_NOT_ALLOWED", "model role is not registered")
        profile = self.profiles[self.policy.bindings[role]]
        if profile.backend == "external_api":
            raise ModelError("EXTERNAL_API_DISABLED", "external_api backend is disabled")
        model = os.environ.get(profile.model_env, "") if profile.model_env else profile.model
        endpoint = os.environ.get(profile.endpoint_env, "") if profile.endpoint_env else ""
        if not model:
            raise ModelError("CONFIG_MISSING", "model identity environment is missing")
        if profile.api_key_env and not os.environ.get(profile.api_key_env):
            raise ModelError("CONFIG_MISSING", "model credential environment is missing")
        if profile.backend == "local_endpoint":
            if not endpoint:
                raise ModelError("CONFIG_MISSING", "local model endpoint environment is missing")
            if origin(endpoint) not in self.policy.allowed_local_origins:
                raise ModelError("ENDPOINT_NOT_ALLOWED", "local model origin is not allowlisted")
        binding_hash = payload_sha256({"profile": profile.to_dict(), "model": model, "endpoint": endpoint})
        return profile, model, endpoint, binding_hash

    def contract_hash(self):
        value = {"policy": self.policy.to_dict(), "bindings": {
            role: {"profile": self.profiles[key].to_dict(),
                   "model_env_value": os.environ.get(self.profiles[key].model_env, ""),
                   "endpoint_env_value": os.environ.get(self.profiles[key].endpoint_env, "")}
            for role, key in self.policy.bindings.items()
        }}
        if self.attempt_limits is not None:
            value['attempt_limits'] = self.attempt_limits.to_dict()
        return payload_sha256(value)

    def identity(self, role):
        profile, model, _, profile_hash = self._binding(role)
        return {"model": model, "revision": profile.revision, "backend": profile.backend, "profile_hash": profile_hash}

    def _provider(self, profile, model, endpoint):
        if profile.profile_id in self.factories:
            return self.factories[profile.profile_id]()
        if profile.backend == "mock":
            return MockModelProvider(profile, self.root / "effects" / profile.profile_id)
        if profile.backend == "local_endpoint":
            return LocalEndpointProvider(profile, endpoint, model, self.store)
        raise ModelError("PROVIDER_NOT_REGISTERED", "local Python worker needs an explicit trusted factory")

    def health(self, role):
        profile, model, endpoint, _ = self._binding(role)
        return self._provider(profile, model, endpoint).health()

    def _prepare(self, request):
        profile, model, endpoint, profile_hash = self._binding(request.role)
        validate_response_schema(request.response_schema)
        if request.artifacts and "vision" not in profile.capabilities:
            raise ModelError("CAPABILITY_MISMATCH", "model profile does not support vision inputs")
        if request.artifacts:
            if self.store is None:
                raise ModelError("UNREGISTERED_STORE", "media artifact store is required")
            for ref in request.artifacts:
                self.store.resolve(ref)
        request_hash = payload_sha256({"request": request.to_dict(), "profile_hash": profile_hash})
        return profile, model, endpoint, profile_hash, request_hash

    def _receipt(self, request, prepared, response, *, reused):
        profile, model, _, profile_hash, request_hash = prepared
        if response.get("model") != model:
            raise ModelError("MODEL_MISMATCH", "response model identity differs from the bound model")
        validate_response_schema(request.response_schema, response.get("output"))
        try:
            result = ModelReceipt(
                request.request_id, request_hash, profile_hash, model, profile.revision, profile.backend,
                response["output"], response["usage"], response["finish_reason"],
                str(response.get("provider_request_id") or ""), reused,
            )
        except (ValueError, KeyError, TypeError):
            raise ModelError("INVALID_OUTPUT", "model output, completion status or usage is invalid") from None
        if result.usage.get("output_tokens") is not None and result.usage["output_tokens"] > request.max_tokens:
            raise ModelError("BUDGET_EXCEEDED", "model reported output beyond the approved token bound")
        path = self.root / f"{request.request_id}.receipt.json"
        stored = replace(result, reused=False).to_dict()
        atomic_write_json(path, {"receipt": stored, "sha256": payload_sha256(stored)})
        return result

    def _recover(self, request, prepared):
        _, _, _, profile_hash, request_hash = prepared
        path = self.root / f"{request.request_id}.receipt.json"
        intent = self.root / f"{request.request_id}.intent.json"
        try:
            record = json.loads(intent.read_text()) if intent.exists() else None
            if record is not None:
                if (record["request_hash"] != request_hash or record["contract_hash"] != self.contract_hash() or
                        type(record["attempts"]) is not int or record["attempts"] < 1):
                    raise ModelError("REQUEST_CONFLICT", "request intent binding or accounting changed")
            if path.exists():
                if record is None:
                    raise ModelError("INVALID_RECEIPT", "model receipt has no matching budget intent")
                wrapper = json.loads(path.read_text())
                data = wrapper["receipt"]
                if wrapper["sha256"] != payload_sha256(data):
                    raise ValueError("receipt content hash changed")
                result = ModelReceipt.from_dict(data)
                if result.request_hash != request_hash or result.profile_hash != profile_hash:
                    raise ModelError("REQUEST_CONFLICT", "request ID is already bound to different inputs or model configuration")
                if (result.request_id != request.request_id or result.model != prepared[1] or
                        result.backend != prepared[0].backend or result.revision != prepared[0].revision):
                    raise ModelError("INVALID_RECEIPT", "model receipt identity changed")
                validate_response_schema(request.response_schema, result.output)
                return replace(result, reused=True)
            if not intent.exists():
                return None
            if record.get("state") == "RETRY_SAFE":
                return None
            if record.get("state") == "INVALID_OUTPUT":
                raise ModelError("INVALID_OUTPUT", "prior model output was rejected; it cannot be silently regenerated")
            response = self._provider(*prepared[:3]).lookup(request, request_hash)
            if response is not None and record.get("wall_seconds") is not None:
                # A reconciled native effect must retain the complete observed
                # attempt duration, including startup before native inference.
                response={**response,"usage":{**response.get("usage",{}),"wall_seconds":max(
                    record["wall_seconds"],response.get("usage",{}).get("wall_seconds") or 0)}}
            return self._receipt(request, prepared, response, reused=True) if response is not None else None
        except ModelError:
            raise
        except (OSError, ValueError, KeyError, TypeError):
            raise ModelError("INVALID_RECEIPT", "model receipt or intent is malformed") from None

    def recover(self, request):
        prepared = self._prepare(request)
        with CampaignLease(self.root / ".lease", ttl_seconds=request.timeout_seconds + 60):
            return self._recover(request, prepared)

    def _reserve(self, request, prepared):
        from contextlib import nullcontext
        lease = self.attempt_limits.reservation_lease() if self.attempt_limits is not None else nullcontext()
        with lease:
            contract_hash = self.contract_hash()
            records = []
            for path in self.root.glob("*.intent.json"):
                try:
                    record = json.loads(path.read_text())
                    if record["contract_hash"] != contract_hash or type(record["attempts"]) is not int or record["attempts"] < 1:
                        raise ValueError("model ledger binding or attempt count changed")
                    records.append(record)
                except (ValueError, KeyError):
                    raise ModelError("INVALID_RECEIPT", "model budget journal is malformed") from None
            prior = self.root / f"{request.request_id}.intent.json"
            attempts = json.loads(prior.read_text())["attempts"] if prior.exists() else 0
            if self.attempt_limits is not None:
                try:
                    self.attempt_limits.check(request, records, now=time.time(), current_root=self.root,
                                              artifact_store=self.store)
                except ValueError as exc:
                    raise ModelError('BUDGET_EXHAUSTED', str(exc)) from exc
            if (attempts >= request.max_attempts or (self.policy.max_calls is not None and
                    sum(x["attempts"] for x in records) >= self.policy.max_calls)):
                raise ModelError("BUDGET_EXHAUSTED", "approved model-call or retry budget is exhausted")
            record = {"request_hash": prepared[4], "contract_hash": contract_hash, "attempts": attempts + 1,
                      "state": "STARTED", "started_at": time.time(), "role": request.role}
            if self.attempt_limits is not None:
                old = json.loads(prior.read_text()) if prior.exists() else {}
                record['first_started_at'] = old.get('first_started_at', old.get('started_at', record['started_at']))
                if self.attempt_limits.unit_rules:
                    record['native_units_per_attempt'] = self.attempt_limits.units(request, artifact_store=self.store)
            atomic_write_json(prior, record)
            return record, prior

    def invoke(self, request: ModelRequest) -> ModelReceipt:
        prepared = self._prepare(request)
        with CampaignLease(self.root / ".lease", ttl_seconds=request.timeout_seconds + 60):
            cached = self._recover(request, prepared)
            if cached is not None:
                return cached
            if prepared[0].backend == "local_python" and prepared[0].profile_id not in self.factories:
                raise ModelError("PROVIDER_NOT_REGISTERED", "local Python worker needs an explicit trusted factory")
            record, prior = self._reserve(request, prepared)
            started = time.monotonic()
            try:
                response = self._provider(*prepared[:3]).invoke(request, prepared[4])
                response = {**response, "usage": {**response.get("usage", {}), "wall_seconds": time.monotonic() - started}}
                return self._receipt(request, prepared, response, reused=False)
            except ModelError as exc:
                record.update(error_code=exc.code, wall_seconds=time.monotonic()-started,
                              finished_at=time.time())
                if exc.retry_safe or exc.code in {"INVALID_OUTPUT", "MODEL_MISMATCH", "BUDGET_EXCEEDED"}:
                    record["state"] = "RETRY_SAFE" if exc.retry_safe else "INVALID_OUTPUT"
                else:
                    record["state"] = "AMBIGUOUS" if exc.code == "AMBIGUOUS_EFFECT" else "FAILED"
                atomic_write_json(prior, record)
                raise
            except BaseException as exc:
                # A crash/cancellation may follow a real effect. Retain its
                # elapsed cost and require native lookup before any recovery.
                record.update(state="AMBIGUOUS", error_code=type(exc).__name__,
                              wall_seconds=time.monotonic()-started, finished_at=time.time())
                atomic_write_json(prior, record)
                raise

    def audit_usage(self):
        """Read the reservation ledger, retaining failed and unknown attempts.

        A reservation is not evidence of a successful effect. Unknown monetary
        cost/duration stays unknown; cache recovery does not reserve another call.
        """
        rows, native_totals = [], {}
        for path in sorted(self.root.glob("*.intent.json")):
            record = json.loads(path.read_text())
            if record["contract_hash"] != self.contract_hash():
                raise ModelError("REQUEST_CONFLICT", "usage ledger contract changed")
            request_id = path.name[:-len(".intent.json")]
            receipt_path = self.root / f"{request_id}.receipt.json"
            usage, state = {}, record.get("state", "UNKNOWN")
            if receipt_path.exists():
                wrapper = json.loads(receipt_path.read_text())
                if wrapper["sha256"] != payload_sha256(wrapper["receipt"]):
                    raise ModelError("INVALID_RECEIPT", "usage receipt hash changed")
                usage, state = wrapper["receipt"]["usage"], "COMPLETED"
            row = {"request_id": request_id, "request_hash": record["request_hash"],
                   "role": record.get("role"), "attempts": record["attempts"], "state": state,
                   "native_units_reserved": record['attempts'] * record.get('native_units_per_attempt', 1),
                   "error_code": record.get("error_code"),
                   "wall_seconds": usage.get("wall_seconds", record.get("wall_seconds")),
                   "monetary_cost": usage.get("cost"), "usage": usage}
            rule = self.attempt_limits.unit_rules.get(record.get('role')) if self.attempt_limits is not None else None
            if rule in {'image_subject', 'video_frame_subject'}:
                provider = self._provider(*self._binding(record['role'])[:3])
                auditor = getattr(provider, 'audit_native_usage', None)
                if auditor is not None:
                    details, totals = auditor(request_id, record, rule, self.store)
                    if row.keys() & details.keys():
                        raise ModelError('INVALID_RECEIPT', 'native accounting cannot replace ledger fields')
                    row.update(details)
                    for key, count in totals.items():
                        if type(count) is not int or count < 0:
                            raise ModelError('INVALID_RECEIPT', 'native accounting totals must be nonnegative integers')
                        native_totals[key] = native_totals.get(key, 0) + count
            rows.append(row)
        summary = {"model_calls": sum(r["attempts"] for r in rows),
                "native_units_reserved": sum(r['native_units_reserved'] for r in rows), "attempts": rows,
                "observed_wall_seconds": sum(r["wall_seconds"] or 0 for r in rows),
                "unknown_duration_attempts": sum(r["wall_seconds"] is None for r in rows),
                "monetary_cost": None if any(r["monetary_cost"] is None for r in rows) else sum(r["monetary_cost"] for r in rows)}
        if summary.keys() & native_totals.keys():
            raise ModelError('INVALID_RECEIPT', 'native accounting cannot replace ledger totals')
        return {**summary, **native_totals}


class RoutedModelGateway:
    """Route disjoint model roles while retaining each original ledger contract.

    Adding an auxiliary model must not rewrite old request intents or invalidate
    their independently recoverable receipts. No planning or retry is added.
    """
    def __init__(self,*gateways,read_only=None):
        self.gateways=gateways;self.roles={}
        for gateway in gateways:
            for role in gateway.policy.bindings:
                if role in self.roles:raise ValueError('ambiguous model gateway role')
                self.roles[role]=gateway
        self.read_only=dict(read_only or {})
        for role,gateway in self.read_only.items():
            if type(gateway) is not ModelGateway or role not in gateway.policy.bindings or role in self.roles:
                raise ValueError('source role must name a disjoint original ModelGateway binding')
            self.roles[role]=gateway
            matches=[g for g in self.gateways if g.root==gateway.root]
            if any(g.contract_hash()!=gateway.contract_hash() for g in matches):
                raise ValueError('source ledger has conflicting model contracts')
            if not matches:self.gateways=(*self.gateways,gateway)

    def _gateway(self,role):
        if role not in self.roles:raise ModelError('ROLE_NOT_ALLOWED','model role is not registered')
        return self.roles[role]

    def invoke(self,request):
        if request.role in self.read_only:
            raise ModelError('ROLE_READ_ONLY','source model roles cannot dispatch')
        return self._gateway(request.role).invoke(request)
    def recover(self,request):
        gateway=self._gateway(request.role)
        if request.role in self.read_only and any(not (gateway.root/(request.request_id+suffix)).is_file()
                for suffix in ('.intent.json','.receipt.json')):
            raise ModelError('SOURCE_NOT_CONFIRMED','source model completion is absent; no repair is allowed')
        return gateway.recover(request)
    def identity(self,role):return self._gateway(role).identity(role)
    def health(self,role):
        if role in self.read_only:
            return {'status':'READ_ONLY_CACHE',**self.identity(role),'inference_performed':False}
        return self._gateway(role).health(role)

    def audit_usage(self):
        rows=sorted([row for gateway in self.gateways for row in gateway.audit_usage()['attempts']],key=lambda row:row['request_id'])
        if len({row['request_id'] for row in rows})!=len(rows):raise ModelError('REQUEST_CONFLICT','request ID occurs in two ledgers')
        return {'model_calls':sum(row['attempts'] for row in rows),'attempts':rows,
            'native_units_reserved':sum(row.get('native_units_reserved',row['attempts']) for row in rows),
            'observed_wall_seconds':sum(row['wall_seconds'] or 0 for row in rows),
            'unknown_duration_attempts':sum(row['wall_seconds'] is None for row in rows),
            'monetary_cost':None if any(row['monetary_cost'] is None for row in rows) else sum(row['monetary_cost'] for row in rows)}
