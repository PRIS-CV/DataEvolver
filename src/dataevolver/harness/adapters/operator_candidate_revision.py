"""A single model-owned source revision using committed engineering feedback."""
from dataclasses import replace
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.dataset.contracts import ArtifactRef, canonical_json, file_sha256, payload_sha256
from ..operator_contracts import OperatorContract
from ..operator_development import CandidatePackage, inspect_candidate
from .operator_candidate_build import CandidateBuildOperator


class CandidateRevisionBuildOperator(CandidateBuildOperator):
    def __init__(self, *, predecessor_plan, predecessor_store, feedback_authority, **kwargs):
        self.predecessor_plan, self.predecessor_store = predecessor_plan, predecessor_store
        self.feedback_authority = feedback_authority
        self.feedback_identity = feedback_authority.identity()
        super().__init__(**kwargs)
        feedback = self._feedback()
        self.feedback_sha256 = payload_sha256(feedback)
        self.spec = replace(self.spec, operator_id="evolution.revise_operator_candidate",
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                "builder": self.spec.code_revision, "revision_adapter": file_sha256(Path(__file__)),
                "feedback": self.feedback_identity, "feedback_sha256": self.feedback_sha256,
                "predecessor_plan_sha256": predecessor_plan.compute_hash()}))
        self.operator_contract = OperatorContract(self.spec.operator_id,"1",self.spec.sha256(),"evolution",
            {"development_plan":{"type":"object"},"committed_engineering_feedback":{"type":"object"}},
            {"candidate_package":{"type":"object"}},
            {"type":"object","additionalProperties":False,"properties":{}},
            ("json","source_code"),"model","verify_then_retry",(
                "One model-owned revision retains predecessor bytes and all prior cost.",
                "Only committed synthetic development feedback is disclosed; heldout qualification stays sealed.",
                "A revised package remains inert and unqualified; no production permission follows."))

    def _feedback(self):
        if self.feedback_authority.identity() != self.feedback_identity:
            raise ValueError("revision feedback authority changed")
        value = self.feedback_authority.read()
        predecessor = CandidatePackage.from_dict(value["package"])
        inspect_candidate(self.predecessor_plan, predecessor, self.predecessor_store, self.authority)
        if (value["schema_version"] != "dataevolver.candidate_revision_feedback.v1"
                or value["scope"] != "synthetic_authority_contract_engineering"
                or value["pure_data_engineering_qualification"] != "FAILED"
                or value["production_adoption_allowed"] is not False
                or predecessor.compute_hash() != value["origin"]["package_sha256"]
                or self.plan.capability_request != self.predecessor_plan.capability_request
                or self.plan.target_interface != self.predecessor_plan.target_interface
                or self.plan.qualification_protocol != self.predecessor_plan.qualification_protocol
                or self.budget.resource_deadline_epoch != value["resource_deadline_epoch"]
                or any(self.budget.prior_usage[k] < v for k,v in value["lifecycle_total"].items())
                or any(root.is_relative_to(self.store.roots[self.plan.candidate_store])
                       or self.store.roots[self.plan.candidate_store].is_relative_to(root)
                       for root in self.predecessor_store.roots.values())):
            raise ValueError("revision must retain original interface, isolated predecessor, deadline and complete costs")
        if hasattr(self,"feedback_sha256") and payload_sha256(value) != self.feedback_sha256:
            raise ValueError("committed revision feedback changed")
        return value

    def _request(self, call):
        request = super()._request(call)
        feedback = self._feedback()
        package = CandidatePackage.from_dict(feedback["package"])
        source = self.predecessor_store.resolve(ArtifactRef.from_dict(package.implementation)).read_text()
        documentation = [{"sha256":raw["sha256"],"content":self.predecessor_store.resolve(ArtifactRef.from_dict(raw)).read_text()}
            for raw in package.files if raw["path"].endswith(".md")]
        # No suggested source edits or per-task answers are supplied. The model
        # receives the actual failed engineering behavior and owns its revision.
        message = {"predecessor_source":{"sha256":package.implementation["sha256"],"content":source,
                    "documentation":documentation},
            "committed_engineering_feedback":{"scope":feedback["scope"],"origin":feedback["origin"],
                "cases":feedback["cases"],"report_sha256":feedback["report_sha256"]},
            "host_runtime_interface":{"profile":"dataevolver.linux_pure_data_context.v2",
                "public_namespace":"TRUSTED_REGISTRARS",
                "value_schema":{"registrar_id":{"n":"public RSA integer","e":65537,"manifest_sha256":"current manifest digest"}},
                "payload_can_change_trust":False,"source_unchanged_until_new_package":True},
            "requirements":["Revise reusable behavior from this evidence; do not special-case fixture identifiers.",
                "Keep the frozen external interface and all refusal/protection/uncertainty gates.",
                "No image requests, production activation or claimed independent qualification."]}
        encoded = canonical_json(message)
        if len(encoded.encode()) > 131072:
            raise ValueError("bounded source revision feedback is too large")
        return ModelRequest("revise-"+payload_sha256([self.plan.compute_hash(),self.feedback_sha256,call.to_dict()])[:28],
            request.role,(*request.messages,{"role":"user","content":encoded}),request.response_schema,
            max_tokens=request.max_tokens,timeout_seconds=request.timeout_seconds,max_attempts=1)
