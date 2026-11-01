"""CPU-only sufficiency receipt requested by an actual runtime decision."""
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .. import evidence_sufficiency as sufficiency
from ..models import ExpectedOutputSpec
from ..operator_contracts import OperatorContract
from ..training_candidate_feedback import FrozenTrainingCandidateFeedbackAuthority
from ..production_feedback_request import FrozenProductionFeedbackRequestAuthority
from .training_candidate_feedback import TrainingCandidateFeedbackOperator


class EvidenceSufficiencyOperator(TrainingCandidateFeedbackOperator):
    def __init__(self, *, authority, request_authority, declarations, output_root):
        if (type(authority) is not FrozenTrainingCandidateFeedbackAuthority
                or type(request_authority) is not FrozenProductionFeedbackRequestAuthority):
            raise ValueError('sufficiency requires the registered frozen production/runtime authorities')
        self.request_authority = request_authority
        self.request_binding = request_authority.identity()
        self.declarations = sufficiency.validate_evidence_declarations(declarations)
        super().__init__(authority=authority, output_root=output_root)
        self.request = request_authority.read()
        if any(Path(p).is_relative_to(self.output_root) for p in self.request['artifact_hashes']):
            raise ValueError('sufficiency cannot overwrite the original runtime request')
        if self.output_root.is_relative_to(Path(self.request_binding['config'])):
            raise ValueError('sufficiency outputs cannot be inside the original runtime review')
        self.spec = replace(self.spec, operator_id='evolution.inspect_evidence_sufficiency',
            source_path=str(Path(__file__).resolve()),
            code_revision=payload_sha256({'feedback': self.spec.code_revision,
                'adapter': file_sha256(Path(__file__)), 'contract': file_sha256(Path(sufficiency.__file__)),
                'request_authority': self.request_binding, 'declarations': self.declarations}),
            expected_output_specs=(ExpectedOutputSpec('evidence_sufficiency.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'evaluation',
            {'registered_diagnostic': {'type': 'object'}, 'runtime_contract_request': {'type': 'object'}},
            {'sufficiency_receipt': {'type': 'object'}},
            {'type': 'object', 'additionalProperties': False, 'properties': {}}, ('json',),
            'pure_data', 'verify_then_reuse', (
                'Evidence declarations are requirements, not qualified witnesses or permissions.',
                'CPU intersections concern supplied declared sets; actual task feasibility stays indeterminate.',
                'Correct refusal remains blocked; no models, images, candidate reruns or geometry are generated.'))

    def _validate(self, call):
        path = super()._validate(call).with_name('evidence_sufficiency.json')
        if path.is_symlink(): raise ValueError('sufficiency output cannot be a symlink')
        return path

    def _result(self):
        super()._result()
        if self.request_authority.identity() != self.request_binding:
            raise ValueError('runtime sufficiency request identity changed')
        request = self.request_authority.read()
        if request != self.request: raise ValueError('runtime sufficiency request changed')
        records, feedback, _ = self.authority.records()
        return sufficiency.compile_sufficiency_receipt(feedback, records, self.declarations,
                                                       request_binding=request)

    def _receipt(self, call, path, reused):
        receipt = super()._receipt(call, path, reused)
        return replace(receipt, stage_result={**receipt.stage_result, 'metrics': {
            'evidence_sufficiency_receipt_captured': 1.0, 'production_qualification_passed': 0.0}})
