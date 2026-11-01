"""Actual subsequent requests resolved by adopted ID/version/snapshot.

Uses the existing confined runner and host oracle. Candidate code is data until
that runner executes it, and can never grant itself image/model permissions.
"""
import json
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import payload_sha256, file_sha256
from ..candidate_trust import CapabilityRequestBinding
from ..authority_qualification import validate_fixture, crypto_identity, check_authority_output
from ..operator_contracts import OperatorContract
from ..operator_lifecycle import CommittedOperatorAdoptionAuthority, SYNTHETIC_SCOPE
from .operator_candidate_checks import CandidateChecksOperator
from .operator_contract_qualification import CandidateContractQualificationOperator


class AdoptedOperatorInvocation(CandidateContractQualificationOperator):
    def __init__(self, *, adoption_authority, **kwargs):
        if not isinstance(adoption_authority, CommittedOperatorAdoptionAuthority):
            raise ValueError("invocation requires an accepted committed adoption")
        self.adoption_authority = adoption_authority
        self.adoption_identity = adoption_authority.identity()
        self.adoption_event = adoption_authority.read()
        super().__init__(**kwargs)
        self.spec = replace(self.spec, operator_id="evolution.invoke_adopted_operator",
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                "qualifier": self.spec.code_revision, "adapter": file_sha256(Path(__file__)),
                "adoption_authority": self.adoption_identity, "adoption_event": self.adoption_event}))
        self.operator_contract = OperatorContract(self.spec.operator_id, "1", self.spec.sha256(), "basic",
            {"adopted_operator_request": {"type": "object"}}, {"bound_result": {"type": "object"}},
            {"type": "object", "additionalProperties": False, "properties": {}}, ("json", "source_code"),
            "pure_data", "verify_then_reuse", (
                "Exact adopted ID/version/snapshot and runtime are resolved before dispatch.",
                "Only synthetic pure-data requests fit this qualification scope.",
                "Revocation or catalog drift refuses new dispatch; completed evidence remains verifiable.",
                "A completed snapshot read alone is not execution, K adoption or image quality."))

    def _check_protocol(self):
        p = self.protocol
        base = {"schema_version", "plan_sha256", "package_sha256", "runtime_sha256", "cases", "budget"}
        extra = {"qualification_request", "trust_contexts", "expectations", "scope", "adoption", "catalog_head"}
        if (set(p) != base | extra or p["schema_version"] != "dataevolver.adopted_operator_invocation.v1"
                or p["scope"] != SYNTHETIC_SCOPE or len(p["cases"]) != 1
                or p["adoption"] != {"authority": self.adoption_identity,
                    "event_sha256": self.adoption_event["sha256"],
                    "snapshot_sha256": self.adoption_event["snapshot_sha256"],
                    "operator_id": self.adoption_event["snapshot"]["operator_id"],
                    "version": self.adoption_event["snapshot"]["version"]}):
            raise ValueError("subsequent request must prebind one exact scope and adopted version")
        self.protocol = {**{k: p[k] for k in base}, "schema_version": "dataevolver.candidate_behavior_checks.v1"}
        try:
            CandidateChecksOperator._check_protocol(self)
        finally:
            self.protocol = p
        event = self.adoption_authority.read()
        snapshot = event["snapshot"]
        if (self.adoption_authority.identity() != self.adoption_identity or event != self.adoption_event
                or snapshot["package_sha256"] != self.package.compute_hash() or snapshot["runtime"] != self.runtime
                or self.budget.resource_deadline_epoch != snapshot["resource_deadline_epoch"]
                or any(self.budget.prior_usage[k] < v for k, v in self.adoption_authority.usage().items())):
            raise ValueError("adopted version, runtime or original costs/deadline changed")
        self.request_binding = CapabilityRequestBinding.from_dict(p["qualification_request"])
        if self.qualification_request_authority.identity() != self.qualification_request_identity:
            raise ValueError("qualification request authority changed")
        self.qualification_request_authority.verify(self.request_binding.verification_view(self.plan), self.store)
        ids = {case["case_id"] for case in p["cases"]}
        if set(p["trust_contexts"]) != ids or set(p["expectations"]) != ids:
            raise ValueError("subsequent input needs an exact protected context and oracle")
        for case in p["cases"]:
            context, expected, payload = self._case_data(case)
            if (context.package_sha256 != self.package.compute_hash()
                    or context.implementation_sha256 != snapshot["implementation_sha256"]
                    or context.qualification_request_sha256 != self.request_binding.request["sha256"]
                    or context.input_artifact != case["input"]
                    or case["input"]["sha256"] in snapshot["qualification_inputs"]
                    or context.fixture_family_id in snapshot["qualification_families"]
                    or (case["kind"] == "positive") != (expected.status == "executable")):
                raise ValueError("subsequent request is not bound to the qualified interface")
            validate_fixture(payload, context, expected)
        if crypto_identity() != self.crypto:
            raise ValueError("host signature verifier changed")

    def _report(self, call):
        report = CandidateChecksOperator._report(self, call)
        for index, case in enumerate(self.protocol["cases"]):
            raw = json.loads((self._path(call).parent / "executions" / (str(index) + ".json")).read_text())
            context, expected, payload = self._case_data(case)
            checks = (check_authority_output(payload, raw["output"], context, expected)
                      if raw["isolation_verified"] and raw["status"] == "COMPLETED" else {"execution": False})
            report["cases"][index]["passed"] &= all(v is True for v in checks.values())
            report["cases"][index]["checks"] = checks
        report["all_checks_passed"] = all(x["passed"] for x in report["cases"])
        report["positive_behavior_verified"] = any(c["passed"] and c["kind"] == "positive" for c in report["cases"])
        report.update(scope=self.protocol["scope"], adopted_operator=self.protocol["adoption"],
            catalog_head_at_dispatch=self.protocol["catalog_head"],
            actual_subsequent_execution=True, adopted_operator_reused=report["all_checks_passed"],
            production_qualification="NOT_ASSESSED", production_adoption_allowed=False)
        return report

    def _receipt(self, call, path, reused):
        receipt = CandidateChecksOperator._receipt(self, call, path, reused)
        report = json.loads(path.read_text())
        return replace(receipt, stage_result={**receipt.stage_result, "metrics": {
            **receipt.stage_result["metrics"], "adopted_operator_reused": float(report["adopted_operator_reused"])}})

    def execute(self, call):
        # A catalog lease spans resolution and child dispatch. A subsequent
        # revocation cannot race between them. Recovery never dispatches again.
        catalog = self.adoption_authority.operator.catalog
        with catalog.lease():
            path = self._path(call)
            if not (path.parent / "check_intent.json").exists():
                catalog.active(self.adoption_authority.read(), scope=self.protocol["scope"],
                               head=self.protocol["catalog_head"])
            return super().execute(call)
