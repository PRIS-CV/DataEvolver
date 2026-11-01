"""Capture verified operational qualification gaps for the existing Planner."""
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .. import boundary_qualification_readback as feedback
from ..models import OperatorSpec, ExpectedOutputSpec
from ..operator_contracts import OperatorContract
from .training_candidate_feedback import TrainingCandidateFeedbackOperator
from .operator_development import CandidateInspectionOperator


class BoundaryQualificationFeedbackOperator(TrainingCandidateFeedbackOperator):
    def __init__(self, *, authority, output_root):
        if type(authority) is not feedback.FrozenBoundaryQualificationFeedbackAuthority:
            raise ValueError('qualification feedback needs its own frozen Controller authority')
        self.authority = authority; self.authority_binding = authority.identity()
        self.output_root = Path(output_root).resolve(); self.summary, proof = authority.read()
        self.summary_hash = payload_sha256(self.summary); self.proof_hash = payload_sha256(proof)
        protected = [authority.config, *(Path(p) for p in proof['artifact_hashes']),
                     *(Path(p) for p in proof['input_store_roots'].values())]
        if any(p.resolve().is_relative_to(self.output_root) or self.output_root.is_relative_to(p.resolve())
               for p in protected):
            raise ValueError('qualification feedback overlaps original evidence')
        self.spec = OperatorSpec('evolution.inspect_boundary_qualification_feedback', '1', retry_safe=True,
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                'adapter': file_sha256(Path(__file__)), 'reader': file_sha256(Path(feedback.__file__)),
                'authority': self.authority_binding, 'summary': self.summary_hash, 'proof': self.proof_hash}),
            allowed_write_roots=(str(self.output_root),),
            expected_output_specs=(ExpectedOutputSpec('qualification_feedback.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'evaluation',
            {'completed_operational_qualification': {'type': 'object'}}, {'feedback': {'type': 'object'}},
            {'type': 'object', 'additionalProperties': False, 'properties': {}}, ('json',),
            'pure_data', 'verify_then_reuse', (
                'Completed operational findings are read with their own frozen source and original evidence closure.',
                'Unmet production evidence prerequisites remain unmet; contract completion is not task qualification.',
                'This feedback performs no model, probe, image, qualification, promotion or adoption execution.'))

    def _validate(self, call):
        path = CandidateInspectionOperator._validate(self, call).with_name('qualification_feedback.json')
        if path.is_symlink():
            raise ValueError('qualification feedback cannot be a symlink')
        return path
