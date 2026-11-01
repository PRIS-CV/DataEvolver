"""PlannerBackend bridge through the typed model gateway.

Legacy planner transport remains intact. A goal must explicitly select this
backend and bind its profiles and policy before any model call is possible.
"""
from __future__ import annotations

from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway
from dataevolver.dataset.contracts import canonical_json, payload_sha256


class GatewayPlannerBackend:
    backend_id = "model-gateway-v1"

    def __init__(self, gateway: ModelGateway, *, role="planner", max_tokens=1024, timeout_seconds=30):
        self.gateway, self.role = gateway, role
        self.max_tokens, self.timeout_seconds = max_tokens, timeout_seconds
        self.artifacts = ()
        self._last_trace = {}

    def set_artifacts(self, artifacts):
        self.artifacts = tuple(artifacts)

    def last_trace(self):
        return dict(self._last_trace)

    @property
    def model_id(self):
        return self.gateway.identity(self.role)["model"]

    def complete_json(self, messages, *, schema):
        request_id = "planner-" + payload_sha256({"messages": messages, "schema": schema,
            **({'artifacts': [r.to_dict() for r in self.artifacts]} if self.artifacts else {})})[:32]
        request = ModelRequest(request_id, self.role, tuple(messages), schema,
                               artifacts=self.artifacts, max_tokens=self.max_tokens, timeout_seconds=self.timeout_seconds)
        receipt = self.gateway.invoke(request)
        self._last_trace = {'request': request.to_dict(), 'receipt': receipt.to_dict(), 'receipt_hash': receipt.without_reuse_hash()}
        return canonical_json(receipt.output)
