"""Registered CPU feedback from a completed typed witness capture."""
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .. import boundary_witness_readback as feedback
from ..models import OperatorSpec, ExpectedOutputSpec
from ..operator_contracts import OperatorContract
from .training_candidate_feedback import TrainingCandidateFeedbackOperator
from .operator_development import CandidateInspectionOperator


class BoundaryWitnessFeedbackOperator(TrainingCandidateFeedbackOperator):
    def __init__(self, *, authority, output_root):
        if type(authority) is not feedback.FrozenBoundaryWitnessFeedbackAuthority:
            raise ValueError('witness feedback needs its original frozen Controller authority')
        self.authority = authority; self.authority_binding = authority.identity()
        self.output_root = Path(output_root).resolve(); self.summary, proof = authority.read()
        self.summary_hash = payload_sha256(self.summary); self.proof_hash = payload_sha256(proof)
        protected = [authority.config, *(Path(p) for p in proof['artifact_hashes']),
                     *(Path(p) for p in proof['input_store_roots'].values())]
        if any(p.resolve().is_relative_to(self.output_root) or self.output_root.is_relative_to(p.resolve())
               for p in protected):
            raise ValueError('witness feedback outputs overlap original evidence')
        self.spec = OperatorSpec('evolution.inspect_boundary_witness_feedback', '1', retry_safe=True,
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                'adapter': file_sha256(Path(__file__)), 'reader': file_sha256(Path(feedback.__file__)),
                'authority': self.authority_binding, 'summary': self.summary_hash, 'proof': self.proof_hash}),
            allowed_write_roots=(str(self.output_root),), expected_output_specs=(ExpectedOutputSpec('witness_feedback.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'evaluation',
            {'registered_witness_capture': {'type': 'object'}}, {'feedback': {'type': 'object'}},
            {'type': 'object', 'additionalProperties': False, 'properties': {}}, ('json',),
            'pure_data', 'verify_then_reuse', (
                'The completed artifact and its independent frozen verification are read without executing it again.',
                'Contract conformance retains insufficient pixel evidence, preservation conflicts and all prior costs.',
                'This capture grants no model, probe, image, qualification or adoption permission.'))

    def _validate(self, call):
        path = CandidateInspectionOperator._validate(self, call).with_name('witness_feedback.json')
        if path.is_symlink():
            raise ValueError('witness feedback output cannot be a symlink')
        return path

    def _receipt(self, call, path, reused):
        receipt = super()._receipt(call, path, reused)
        return replace(receipt, stage_result={**receipt.stage_result, 'metrics': {
            'production_feedback_captured': 1.0, 'positive_production_behavior_verified': 0.0}})
