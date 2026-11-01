"""Registered read-only feedback capture for the existing RSI Controller."""
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import payload_sha256, file_sha256
from .. import training_candidate_feedback as feedback
from ..operator_contracts import OperatorContract
from ..models import ExpectedOutputSpec
from .operator_development import CandidateInspectionOperator


class TrainingCandidateFeedbackOperator(CandidateInspectionOperator):
    def __init__(self, *, authority, output_root):
        self.authority = authority
        self.authority_binding = authority.identity()
        self.output_root = Path(output_root).resolve()
        self.summary, proof = authority.read()
        self.summary_hash = payload_sha256(self.summary)
        self.proof_hash = payload_sha256(proof)
        protected = [Path(p).resolve() for p in proof['artifact_hashes']]
        # A new sibling file inside an old config is still a historical write,
        # even when it does not overlap any existing file in the closure.
        if 'config' in proof:
            protected.append(Path(proof['config']).resolve())
            descriptor = proof['descriptor']
            protected.extend(Path(descriptor[k]).resolve() for k in
                             ('request_config', 'trial_config', 'observation_config'))
            protected.extend(Path(p).resolve() for p in descriptor['stores'].values())
        if any(p.is_relative_to(self.output_root) or self.output_root.is_relative_to(p) for p in protected):
            raise ValueError('feedback outputs overlap historical evidence')
        from ..models import OperatorSpec
        self.spec = OperatorSpec('evolution.inspect_training_candidate_feedback', '1', retry_safe=True,
            source_path=str(Path(__file__).resolve()),
            code_revision=payload_sha256({'adapter': file_sha256(Path(__file__)),
                'reader': file_sha256(Path(feedback.__file__)), 'authority': self.authority_binding,
                'feedback_sha256': self.summary_hash, 'proof_sha256': self.proof_hash}),
            allowed_write_roots=(str(self.output_root),),
            expected_output_specs=(ExpectedOutputSpec('candidate_feedback.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'evaluation',
            {'committed_production_diagnostic': {'type': 'object'}}, {'feedback': {'type': 'object'}},
            {'type': 'object', 'additionalProperties': False, 'properties': {}}, ('json',),
            'pure_data', 'verify_then_reuse', (
                'Only a committed, frozen, replayable production diagnostic is accepted.',
                'Safe refusal remains unqualified; no geometry, permissions or image requests are generated.',
                'The candidate is not executed again; drift blocks cached feedback.'))

    def _validate(self, call):
        path = super()._validate(call).with_name('candidate_feedback.json')
        if path.is_symlink():
            raise ValueError('feedback output cannot be a symlink')
        return path

    def _result(self):
        if self.authority.identity() != self.authority_binding:
            raise ValueError('feedback authority changed')
        summary, proof = self.authority.read()
        if payload_sha256(summary) != self.summary_hash or payload_sha256(proof) != self.proof_hash:
            raise ValueError('captured production feedback changed')
        return {'schema_version': 'dataevolver.registered_production_feedback.v1',
            'feedback': summary, 'feedback_sha256': self.summary_hash,
            'authority_sha256': payload_sha256(self.authority_binding), 'proof_sha256': self.proof_hash,
            'production_qualification_passed': False, 'new_candidate_executions': 0, 'new_model_calls': 0}

    def _receipt(self, call, path, reused):
        receipt = super()._receipt(call, path, reused)
        return replace(receipt, stage_result={**receipt.stage_result, 'metrics': {
            'production_feedback_captured': 1.0, 'positive_production_behavior_verified': 0.0}})
