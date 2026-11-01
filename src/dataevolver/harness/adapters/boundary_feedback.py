"""Capture already committed native-probe feedback without rerunning it."""
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256,payload_sha256
from .. import boundary_feedback as feedback
from ..models import OperatorSpec,ExpectedOutputSpec
from ..operator_contracts import OperatorContract
from .training_candidate_feedback import TrainingCandidateFeedbackOperator


class BoundaryProbeFeedbackOperator(TrainingCandidateFeedbackOperator):
    def __init__(self, *, authority, output_root):
        if type(authority) is not feedback.FrozenBoundaryFeedbackAuthority:
            raise ValueError('boundary feedback requires its frozen original Controller authority')
        self.authority=authority;self.authority_binding=authority.identity()
        self.output_root=Path(output_root).resolve();self.summary,proof=authority.read()
        self.summary_hash=payload_sha256(self.summary);self.proof_hash=payload_sha256(proof)
        protected=[Path(authority.config),*[Path(p) for p in proof['artifact_hashes']],
                   *[Path(p) for p in proof['input_store_roots'].values()]]
        protected.extend(Path(p) for p in authority.descriptor.values())
        if any(p.resolve().is_relative_to(self.output_root) or self.output_root.is_relative_to(p.resolve()) for p in protected):
            raise ValueError('boundary feedback outputs overlap original evidence')
        self.spec=OperatorSpec('evolution.inspect_boundary_probe_feedback','1',retry_safe=True,
            source_path=str(Path(__file__).resolve()),code_revision=payload_sha256({
                'adapter':file_sha256(Path(__file__)),'reader':file_sha256(Path(feedback.__file__)),
                'authority':self.authority_binding,'summary':self.summary_hash,'proof':self.proof_hash}),
            allowed_write_roots=(str(self.output_root),),expected_output_specs=(ExpectedOutputSpec('boundary_feedback.json'),))
        self.operator_contract=OperatorContract(self.spec.operator_id,'1',self.spec.sha256(),'evaluation',
            {'completed_registered_probe':{'type':'object'}},{'feedback':{'type':'object'}},
            {'type':'object','additionalProperties':False,'properties':{}},('json',),'pure_data','verify_then_retry',(
                'Only actual frozen, replayable native segmentation and runtime visual-review evidence is read.',
                'Local visual support, original masks and proposal differences establish no pixel truth or editing authority.',
                'No probe, image, candidate, Planner, qualification, promotion or adoption is executed by this capture.'))

    def _validate(self,call):
        # Reuse the shared inspection path/receipt/symlink checks, then select
        # the explicit output name recorded in this operator's own contract.
        from .operator_development import CandidateInspectionOperator
        path=CandidateInspectionOperator._validate(self,call).with_name('boundary_feedback.json')
        if path.is_symlink():raise ValueError('boundary feedback output cannot be a symlink')
        return path

    def _result(self):
        value=super()._result()
        value['schema_version']='dataevolver.registered_boundary_probe_feedback.v1'
        return value
