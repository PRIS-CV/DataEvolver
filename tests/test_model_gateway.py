import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ExternalAPIProvider, LocalPythonProvider, ModelError, ModelGateway, MockModelProvider
from dataevolver.dataset.contracts import payload_sha256
from dataevolver.harness.model_backend import GatewayPlannerBackend


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


def request(request_id="request-1"):
    return ModelRequest(request_id, "planner", ({"role": "user", "content": "Return the fixture."},), SCHEMA)


def profile():
    return ModelProfile("fixture", "mock", "fixture-model", "1", mock_output={"ok": True})


def gateway(tmp_path, *, profiles=None, policy=None, providers=None):
    return ModelGateway(profiles=profiles or (profile(),), policy=policy or ModelPolicy({"planner": "fixture"}, 4),
                        receipt_root=tmp_path / "receipts", providers=providers)


def test_cache_hit_precedes_provider_creation_and_model_loading(tmp_path):
    original = gateway(tmp_path).invoke(request())
    def forbidden():
        raise AssertionError("cache hit instantiated a provider")
    cached = gateway(tmp_path, providers={"fixture": forbidden}).invoke(request())
    assert cached.reused and cached.without_reuse_hash() == original.without_reuse_hash()
    assert len(list((tmp_path / "receipts/effects/fixture").glob("*.json"))) == 1


def test_usage_accounting_keeps_receipts_without_a_global_call_ceiling(tmp_path):
    policy = ModelPolicy({"planner": "fixture"}, None, usage_mode="accounting")
    assert ModelPolicy.from_dict(policy.to_dict()).usage_mode == "accounting"
    model = gateway(tmp_path, policy=policy)
    for i in range(6):
        model.invoke(request(f"accounted-{i}"))
    assert model.audit_usage()["model_calls"] == 6
    assert model.invoke(request("accounted-0")).reused
    assert model.audit_usage()["model_calls"] == 6
    # Bounded legacy configuration is byte-for-byte compatible in its fields.
    assert "usage_mode" not in ModelPolicy({"planner": "fixture"}, 4).to_dict()
    with pytest.raises(ValueError, match="accounting mode"):
        ModelPolicy({"planner": "fixture"}, 1, usage_mode="accounting")


def test_accounting_does_not_resend_an_unknown_effect(tmp_path):
    class Unknown:
        def invoke(self, req, digest):
            raise ModelError("AMBIGUOUS_EFFECT", "fixture unknown receipt")

        def lookup(self, req, digest):
            raise ModelError("AMBIGUOUS_EFFECT", "fixture not reconciled")

    model = gateway(tmp_path, policy=ModelPolicy({"planner": "fixture"}, None, usage_mode="accounting"),
                    providers={"fixture": Unknown})
    for _ in range(2):
        with pytest.raises(ModelError, match="fixture"):
            model.invoke(request())
    assert model.audit_usage()["model_calls"] == 1


def test_native_reconciliation_retains_full_observed_attempt_cost(tmp_path):
    class InterruptedAfterEffect(MockModelProvider):
        def invoke(self, req, digest):
            super().invoke(req,digest)
            raise RuntimeError("fixture checkpoint after committed effect")
    model=gateway(tmp_path,providers={"fixture":lambda:InterruptedAfterEffect(profile(),tmp_path/"native")})
    with pytest.raises(RuntimeError):model.invoke(request())
    intent=tmp_path/"receipts/request-1.intent.json"
    record=json.loads(intent.read_text());record["wall_seconds"]=123.5
    intent.write_text(json.dumps(record))
    receipt=model.recover(request())
    assert receipt.reused and receipt.usage["wall_seconds"]==123.5
    assert model.audit_usage()["model_calls"]==1
    assert model.audit_usage()["observed_wall_seconds"]==123.5


def test_request_id_cannot_be_rebound_and_receipts_cannot_be_tampered(tmp_path):
    model = gateway(tmp_path)
    model.invoke(request())
    with pytest.raises(ModelError) as error:
        model.invoke(replace(request(), messages=({"role": "user", "content": "Changed"},)))
    assert error.value.code == "REQUEST_CONFLICT"
    path = tmp_path / "receipts/request-1.receipt.json"
    wrapper = json.loads(path.read_text())
    wrapper["receipt"]["request_id"] = "another-request"
    wrapper["sha256"] = payload_sha256(wrapper["receipt"])
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ModelError) as error:
        model.invoke(request())
    assert error.value.code == "INVALID_RECEIPT"


def test_cache_requires_the_budget_journal_and_same_policy(tmp_path):
    gateway(tmp_path).invoke(request())
    with pytest.raises(ModelError) as error:
        gateway(tmp_path, policy=ModelPolicy({"planner": "fixture"}, 10)).invoke(request())
    assert error.value.code == "REQUEST_CONFLICT"
    (tmp_path / "receipts/request-1.intent.json").unlink()
    with pytest.raises(ModelError) as error:
        gateway(tmp_path).invoke(request())
    assert error.value.code == "INVALID_RECEIPT"


def test_crash_after_effect_recovers_without_a_duplicate_call(tmp_path):
    counts = {"invocations": 0}
    class CrashAfterEffect(MockModelProvider):
        def invoke(self, req, digest):
            counts["invocations"] += 1
            super().invoke(req, digest)
            raise RuntimeError("simulated process interruption")
    def factory():
        return CrashAfterEffect(profile(), tmp_path / "receipts/effects/fixture")
    with pytest.raises(RuntimeError):
        gateway(tmp_path, providers={"fixture": factory}).invoke(request())
    failed = gateway(tmp_path).audit_usage()["attempts"][0]
    assert failed["state"] == "AMBIGUOUS" and failed["wall_seconds"] >= 0
    assert failed["error_code"] == "RuntimeError"
    recovered = gateway(tmp_path).invoke(request())
    assert recovered.reused and recovered.output == {"ok": True}
    assert counts["invocations"] == 1
    assert json.loads((tmp_path / "receipts/request-1.intent.json").read_text())["attempts"] == 1


def test_ambiguous_model_effect_never_resends(tmp_path):
    calls = []
    class Ambiguous:
        def invoke(self, req, digest):
            calls.append(req.request_id)
            raise ModelError("AMBIGUOUS_EFFECT", "timeout")
        def lookup(self, req, digest):
            raise ModelError("AMBIGUOUS_EFFECT", "no native lookup")
    worker = Ambiguous()
    model = gateway(tmp_path, providers={"fixture": lambda: worker})
    for _ in range(2):
        with pytest.raises(ModelError) as error:
            model.invoke(request())
        assert error.value.code == "AMBIGUOUS_EFFECT"
    assert calls == ["request-1"]


def test_retry_safe_failure_consumes_budget_and_can_only_retry_within_it(tmp_path):
    calls = []
    class Limited(MockModelProvider):
        def invoke(self, req, digest):
            calls.append(req.request_id)
            if len(calls) == 1:
                raise ModelError("RATE_LIMIT", "fixture limit", retry_safe=True)
            return super().invoke(req, digest)
    model = gateway(tmp_path, policy=ModelPolicy({"planner": "fixture"}, 2), providers={
        "fixture": lambda: Limited(profile(), tmp_path / "receipts/effects/fixture")})
    req = replace(request(), max_attempts=2)
    with pytest.raises(ModelError) as error:
        model.invoke(req)
    assert error.value.code == "RATE_LIMIT"
    assert model.invoke(req).output == {"ok": True}
    with pytest.raises(ModelError) as error:
        model.invoke(request("over-budget"))
    assert error.value.code == "BUDGET_EXHAUSTED" and len(calls) == 2


def test_external_api_is_disabled_before_provider_creation_or_any_effect(tmp_path):
    cloud = replace(profile(), profile_id="cloud", backend="external_api")
    touched = []
    model = gateway(tmp_path, profiles=(profile(), cloud),
                    policy=ModelPolicy({"planner": "fixture", "future": "cloud"}, 4),
                    providers={"cloud": lambda: touched.append(True)})
    assert model.invoke(request()).output == {"ok": True}
    with pytest.raises(ModelError) as error:
        model.invoke(replace(request("cloud-request"), role="future"))
    assert error.value.code == "EXTERNAL_API_DISABLED" and touched == []
    assert not (tmp_path / "receipts/cloud-request.intent.json").exists()
    with pytest.raises(ValueError, match="disabled"):
        ModelPolicy({"planner": "cloud"}, 1, external_api_enabled=True)
    with pytest.raises(ModelError):
        ExternalAPIProvider().invoke(request(), "hash")


def test_schema_failure_happens_before_dispatch_and_remote_refs_are_blocked(tmp_path):
    touched = []
    model = gateway(tmp_path, providers={"fixture": lambda: touched.append(True)})
    bad_schema = {"type": "object", "properties": {"x": {"$ref": "https://invalid.example/schema"}}}
    with pytest.raises(ModelError) as error:
        model.invoke(replace(request(), response_schema=bad_schema))
    assert error.value.code == "INVALID_SCHEMA" and touched == []
    assert not (tmp_path / "receipts").exists()
    with pytest.raises(ModelError) as error:
        model.invoke(replace(request(), response_schema={"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}}))
    assert error.value.code == "INVALID_SCHEMA" and touched == []


def test_invalid_model_output_cannot_be_accepted_or_silently_regenerated(tmp_path):
    invalid = replace(profile(), mock_output={"ok": "not-a-boolean"})
    model = gateway(tmp_path, profiles=(invalid,))
    for _ in range(2):
        with pytest.raises(ModelError) as error:
            model.invoke(request())
        assert error.value.code == "INVALID_OUTPUT"
    assert json.loads((tmp_path / "receipts/request-1.intent.json").read_text())["attempts"] == 1


def test_local_python_loading_is_lazy_and_gateway_planner_is_compatible(tmp_path):
    loads = []
    class Worker:
        def invoke(self, req, digest):
            return {"model": "fixture-model", "output": {"ok": True}, "usage": {"cost": None}, "finish_reason": "stop"}
    worker_profile = replace(profile(), backend="local_python")
    def build_worker():
        loads.append(True)
        return Worker()
    model = gateway(tmp_path, profiles=(worker_profile,), providers={"fixture": lambda: LocalPythonProvider(build_worker)})
    backend = GatewayPlannerBackend(model)
    assert loads == []
    assert json.loads(backend.complete_json(list(request().messages), schema=SCHEMA)) == {"ok": True}
    assert backend.complete_json(list(request().messages), schema=SCHEMA) == '{"ok":true}'
    assert backend.model_id == "fixture-model" and loads == [True]


@pytest.fixture
def endpoint():
    state = {"posts": [], "finish_reason": "stop", "model": "served-model", "usage": None, "redirect": False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "served-model"}]}).encode())
        def do_POST(self):
            state["posts"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if state["redirect"]:
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/redirected")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id": "fixture-call", "model": state["model"], "usage": state["usage"],
                "choices": [{"finish_reason": state["finish_reason"], "message": {"content": '{"ok":true}'}}]}).encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def endpoint_gateway(tmp_path, monkeypatch, endpoint):
    url, state = endpoint
    monkeypatch.setenv("TEST_MODEL_URL", url + "/v1")
    monkeypatch.setenv("TEST_MODEL_KEY", "fixture-secret-value")
    p = ModelProfile("server", "local_endpoint", "served-model", "declared-deployment", endpoint_env="TEST_MODEL_URL", api_key_env="TEST_MODEL_KEY")
    model = gateway(tmp_path, profiles=(p,), policy=ModelPolicy({"planner": "server"}, 2, (url,)))
    return model, state


def test_local_endpoint_health_json_transport_unknown_usage_and_secret_redaction(tmp_path, monkeypatch, endpoint):
    model, state = endpoint_gateway(tmp_path, monkeypatch, endpoint)
    assert model.health("planner")["status"] == "AVAILABLE" and state["posts"] == []
    receipt = model.invoke(request())
    assert receipt.output == {"ok": True} and receipt.usage["input_tokens"] is None and receipt.usage["cost"] is None
    assert len(state["posts"]) == 1 and state["posts"][0]["response_format"] == {"type": "json_object"}
    assert all("fixture-secret-value" not in p.read_text() for p in (tmp_path / "receipts").glob("*.json"))
    assert model.invoke(request()).reused and len(state["posts"]) == 1


@pytest.mark.parametrize("field,value,code", [("model", "wrong-model", "MODEL_MISMATCH"), ("finish_reason", "length", "INVALID_OUTPUT")])
def test_endpoint_model_mismatch_and_truncation_fail_closed(tmp_path, monkeypatch, endpoint, field, value, code):
    model, state = endpoint_gateway(tmp_path, monkeypatch, endpoint)
    state[field] = value
    with pytest.raises(ModelError) as error:
        model.invoke(request())
    assert error.value.code == code
    assert not (tmp_path / "receipts/request-1.receipt.json").exists()


def test_endpoint_origin_and_redirect_are_enforced(tmp_path, monkeypatch, endpoint):
    model, state = endpoint_gateway(tmp_path, monkeypatch, endpoint)
    monkeypatch.setenv("TEST_MODEL_URL", "https://unapproved.example/v1")
    with pytest.raises(ModelError) as error:
        model.invoke(request())
    assert error.value.code == "ENDPOINT_NOT_ALLOWED" and state["posts"] == []
    monkeypatch.setenv("TEST_MODEL_URL", endpoint[0] + "/v1")
    state["redirect"] = True
    with pytest.raises(ModelError) as error:
        model.invoke(request())
    assert error.value.code == "PROVIDER_HTTP_ERROR" and len(state["posts"]) == 1


def test_endpoint_token_overrun_is_rejected_without_regeneration(tmp_path, monkeypatch, endpoint):
    model, state = endpoint_gateway(tmp_path, monkeypatch, endpoint)
    state["usage"] = {"prompt_tokens": 3, "completion_tokens": 300}
    with pytest.raises(ModelError) as error:
        model.invoke(request())
    assert error.value.code == "BUDGET_EXCEEDED"
    with pytest.raises(ModelError) as error:
        model.invoke(request())
    assert error.value.code == "INVALID_OUTPUT" and len(state["posts"]) == 1


def test_missing_endpoint_credential_and_python_worker_fail_before_effects(tmp_path, monkeypatch, endpoint):
    model, state = endpoint_gateway(tmp_path, monkeypatch, endpoint)
    monkeypatch.delenv("TEST_MODEL_KEY")
    with pytest.raises(ModelError) as error:
        model.invoke(request())
    assert error.value.code == "CONFIG_MISSING" and state["posts"] == []
    assert not (tmp_path / "receipts").exists()
    local = gateway(tmp_path, profiles=(replace(profile(), backend="local_python"),))
    with pytest.raises(ModelError) as error:
        local.invoke(request())
    assert error.value.code == "PROVIDER_NOT_REGISTERED"
    assert not (tmp_path / "receipts/request-1.intent.json").exists()
