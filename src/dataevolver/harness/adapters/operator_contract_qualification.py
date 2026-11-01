"""Registered host-controlled synthetic contract qualification, not image RSI.

The original package and production qualification design stay unchanged. A new
explicit host trust-context interface and expected behavior are frozen here.
"""
import json
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from ..authority_qualification import AuthorityFixtureExpectation, check_authority_output, crypto_identity, validate_fixture
from ..candidate_sandbox import TrustedContextRunner
from ..candidate_trust import CandidateTrustContext, CapabilityRequestBinding, canonical_digest
from ..operator_contracts import OperatorContract
from .operator_candidate_checks import CandidateChecksOperator


class CandidateContractQualificationOperator(CandidateChecksOperator):
    def __init__(self, *, plan, package, store, request_authority, runner, check_protocol,
                 qualification_request_authority, output_root):
        if not isinstance(runner, TrustedContextRunner):
            raise ValueError("contract qualification requires the explicit trusted-context profile")
        self.qualification_request_authority = qualification_request_authority
        self.qualification_request_identity = qualification_request_authority.identity()
        self.crypto = crypto_identity()
        super().__init__(plan=plan, package=package, store=store, request_authority=request_authority,
                         runner=runner, check_protocol=check_protocol, output_root=output_root)
        self.spec = replace(self.spec, operator_id="evolution.qualify_operator_contract",
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                "checks": self.spec.code_revision, "adapter": file_sha256(Path(__file__)),
                "oracle": file_sha256(Path(__file__).parents[1]/"authority_qualification.py"),
                "trust": file_sha256(Path(__file__).parents[1]/"candidate_trust.py"),
                "crypto": self.crypto, "qualification_request_authority": self.qualification_request_identity}))
        self.operator_contract = OperatorContract(self.spec.operator_id,"1",self.spec.sha256(),"evaluation",
            {"candidate_package":{"type":"object"},"trusted_context_protocol":{"type":"object"}},
            {"contract_qualification":{"type":"object"}},
            {"type":"object","additionalProperties":False,"properties":{}},
            ("json","source_code"),"pure_data","verify_then_reuse",(
                "Public host trust material is provided by a separate frozen synthetic-only context.",
                "Expected behavior and signature verification are independent of candidate code.",
                "Passing this scope does not pass the original image/independent qualification protocol.",
                "No production trust, image submission, promotion, reuse or method benefit is granted."))

    def _check_protocol(self):
        p = self.protocol
        extras = {"qualification_request","trust_contexts","expectations","qualification_scope"}
        base = {"schema_version","plan_sha256","package_sha256","runtime_sha256","cases","budget"}
        if (set(p) != base | extras or p["schema_version"] != "dataevolver.candidate_contract_qualification.v1"
                or p["qualification_scope"] != "synthetic_authority_contract_engineering"):
            raise ValueError("unsupported explicit qualification protocol")
        # V1 execution/budget semantics are reused without changing its contract.
        self.protocol = {**{k:p[k] for k in base}, "schema_version":"dataevolver.candidate_behavior_checks.v1"}
        try: super()._check_protocol()
        finally: self.protocol = p
        ids = {case["case_id"] for case in p["cases"]}
        if (not isinstance(p["trust_contexts"],dict) or not isinstance(p["expectations"],dict)
                or set(p["trust_contexts"]) != ids or set(p["expectations"]) != ids
                or {case["kind"] for case in p["cases"]} != {"positive","negative"}):
            raise ValueError("qualification requires positive and adversarial cases with host contexts")
        self.request_binding = CapabilityRequestBinding.from_dict(p["qualification_request"])
        if self.request_binding.request["store"] == self.plan.candidate_store:
            raise ValueError("a candidate cannot own its qualification request")
        if self.qualification_request_authority.identity() != self.qualification_request_identity:
            raise ValueError("qualification request authority changed")
        self.qualification_request_authority.verify(self.request_binding.verification_view(self.plan), self.store)
        for case in p["cases"]:
            context,expected,payload = self._case_data(case)
            if (context.package_sha256 != self.package.compute_hash()
                    or context.implementation_sha256 != self.package.implementation["sha256"]
                    or context.qualification_request_sha256 != self.request_binding.request["sha256"]
                    or context.input_artifact != case["input"]
                    or (case["kind"] == "positive") != (expected.status == "executable")):
                raise ValueError("fixture trust, request, source or kind changed")
            validate_fixture(payload, context, expected)
        if crypto_identity() != self.crypto: raise ValueError("host signature verifier changed")

    def _case_data(self, case):
        def read(raw):
            ref=ArtifactRef.from_dict(raw)
            if ref.store not in self.plan.protected_stores:
                raise ValueError("trust context and oracle must belong to protected host stores")
            return json.loads(self.store.resolve(ref).read_text())
        context=CandidateTrustContext.from_dict(read(self.protocol["trust_contexts"][case["case_id"]]))
        expected=AuthorityFixtureExpectation.from_dict(read(self.protocol["expectations"][case["case_id"]]))
        payload=json.loads(self.store.resolve(ArtifactRef.from_dict(case["input"])).read_text())
        if context.payload_sha256 != canonical_digest(payload): raise ValueError("host context input changed")
        return context,expected,payload

    def _execution_binding(self, case, payload):
        binding=super()._execution_binding(case,payload)
        binding["trusted_context_sha256"]=self._case_data(case)[0].compute_hash()
        return binding

    def _run_case(self, case, source, payload):
        context,_,_ = self._case_data(case)
        return self.runner.run_with_context(source,payload,context)

    def _report(self, call):
        result=super()._report(call)
        results=[]
        for index,case in enumerate(self.protocol["cases"]):
            raw=json.loads((self._path(call).parent/"executions"/(str(index)+".json")).read_text())
            context,expected,payload=self._case_data(case)
            checks=(check_authority_output(payload,raw["output"],context,expected)
                    if raw["isolation_verified"] and raw["status"]=="COMPLETED" else {"execution":False})
            passed=result["cases"][index]["passed"] and all(v is True for v in checks.values())
            result["cases"][index]["passed"]=passed
            results.append({"case_id":case["case_id"],"fixture_family_id":context.fixture_family_id,
                "trusted_context_sha256":context.compute_hash(),"expectation_sha256":expected.compute_hash(),
                "checks":checks,"passed":passed})
        result["all_checks_passed"]=all(x["passed"] for x in results)
        result["positive_behavior_verified"]=any(c["passed"] and c["kind"]=="positive" for c in result["cases"])
        result.update(contract_checks=results,qualification_request_sha256=self.request_binding.request["sha256"],
            qualification_scope=self.protocol["qualification_scope"],
            pure_data_engineering_qualification="PASSED" if result["all_checks_passed"] else "FAILED",
            image_quality_verified=False,production_adoption_allowed=False,
            source_bytes_unchanged=True,host_context_interface="dataevolver.candidate_trust_context.v1")
        return result

    def _receipt(self, call, path, reused):
        receipt=super()._receipt(call,path,reused)
        report=json.loads(path.read_text())
        return replace(receipt,stage_result={**receipt.stage_result,"metrics":{**receipt.stage_result["metrics"],
            "contract_qualification_passed":float(report["pure_data_engineering_qualification"]=="PASSED")}})
