"""Capture the runtime-requested witness through the existing Controller."""
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .. import boundary_witness as witness
from ..models import ExpectedOutputSpec, EvidenceStatus
from ..operator_contracts import OperatorContract
from ..production_feedback_request import FrozenProductionFeedbackRequestAuthority
from .boundary_feedback import BoundaryProbeFeedbackOperator


class BoundaryWitnessOperator(BoundaryProbeFeedbackOperator):
    def __init__(self, *, authority, request_authority, output_root):
        if type(request_authority) is not FrozenProductionFeedbackRequestAuthority:
            raise ValueError('boundary witness needs its frozen registered runtime request')
        self.request_authority = request_authority
        self.request_binding = request_authority.identity()
        self.request = request_authority.read()
        super().__init__(authority=authority, output_root=output_root)
        self.context = authority.input_context()
        protected = [Path(self.request_binding['config']),
                     *(Path(p) for p in self.request['artifact_hashes'])]
        if any(p.resolve().is_relative_to(self.output_root) or self.output_root.is_relative_to(p.resolve())
               for p in protected):
            raise ValueError('witness output overlaps original runtime request')
        self.spec = replace(self.spec, operator_id='evolution.capture_boundary_witness',
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                'feedback': self.spec.code_revision, 'adapter': file_sha256(Path(__file__)),
                'contract': file_sha256(Path(witness.__file__)), 'request': self.request_binding,
                'context': payload_sha256(self.context)}),
            expected_output_specs=(ExpectedOutputSpec('boundary_witness.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'evaluation',
            {'registered_boundary_feedback': {'type': 'object'}, 'runtime_request': {'type': 'object'}},
            {'witness': witness.WITNESS_SCHEMA, 'validation': {'type': 'object'}},
            {'type': 'object', 'additionalProperties': False, 'properties': {}}, ('json',),
            'pure_data', 'verify_then_reuse', (
                'Every local finding retains exact registered report, receipt, scope and immutable correspondence.',
                'Mask differences and local visual support provide no pixel labels, effect ownership or edit authority.',
                'Construction and validation execute no probes, images, qualification, promotion or adoption.'))

    def _validate(self, call):
        path = super()._validate(call).with_name('boundary_witness.json')
        if path.is_symlink():
            raise ValueError('boundary witness cannot be a symlink')
        return path

    def _result(self):
        super()._result()
        if (self.request_authority.identity() != self.request_binding
                or self.request_authority.read() != self.request
                or self.authority.input_context() != self.context):
            raise ValueError('boundary witness original request or constraints changed')
        value = witness.compile_boundary_witness(self.summary, self.context, request_binding=self.request)
        receipts = {self.summary[k+'_request']['request_id']: self.summary[k+'_receipt_sha256']
                    for k in ('segmentation', 'visual')}
        validation = witness.validate_boundary_witness(value, self.summary, self.context,
            request_binding=self.request, available_receipts=receipts)
        if not validation['contract_conformant']:
            raise ValueError('compiled boundary witness does not validate')
        return {'schema_version': 'dataevolver.registered_boundary_witness.v1',
            'witness': value, 'validation': validation, 'new_model_calls': 0,
            'new_probe_executions': 0, 'production_qualification_passed': False,
            'image_submission_allowed': False, 'candidate_adoption_allowed': False,
            'new_accepted_records': 0}

    def _receipt(self, call, path, reused):
        receipt = super()._receipt(call, path, reused)
        return replace(receipt, stage_result={**receipt.stage_result, 'metrics': {
            'boundary_witness_contract_captured': 1.0, 'production_qualification_passed': 0.0}})

    def verify_effect(self, call, receipt):
        verified = super().verify_effect(call, receipt)
        if verified.status is EvidenceStatus.VERIFIED:
            _, proof = self.authority.read()
            verified = replace(verified, artifact_hashes={**proof['artifact_hashes'],
                **self.request['artifact_hashes'], **verified.artifact_hashes})
        return verified
