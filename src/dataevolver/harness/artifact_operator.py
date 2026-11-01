"""Receipt adapter for host-defined, resumable workloads under the Controller."""
import json
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .models import (OperatorSpec, OperatorReceipt, StageStatus, EffectVerification,
                     EvidenceStatus, RecoveryResult, RecoveryStatus, ExpectedOutputSpec)
from .operators import _receipt_path, _load_receipt, _write_receipt
from .state_store import atomic_write_json


class ArtifactWorkloadOperator:
    """Subclasses must reconcile every model/external effect before re-execution."""
    def __init__(self, *, operator_id, output_root, configuration, parameter_schema=None):
        self.output_root = Path(output_root).resolve()
        self.configuration = json.loads(json.dumps(configuration, allow_nan=False))
        source = Path(__import__('inspect').getsourcefile(type(self))).resolve()
        self.spec = OperatorSpec(operator_id, '1', retry_safe=True, source_path=str(source),
            code_revision=payload_sha256({'source': file_sha256(source), 'configuration': self.configuration}),
            allowed_write_roots=(str(self.output_root),), expected_output_specs=(ExpectedOutputSpec('workload.json'),))
        self._parameter_schema = parameter_schema or {'type': 'object', 'additionalProperties': False, 'properties': {}}
        self._declare_contract()

    def _declare_contract(self):
        from .operator_contracts import OperatorContract
        self.operator_contract = OperatorContract(self.spec.operator_id, self.spec.version, self.spec.sha256(),
            'evolution' if self.spec.operator_id.startswith('evolution.') else 'basic', {}, {'report': {'type': 'object'}},
            self._parameter_schema, ('json',), 'composite', 'verify_then_reuse',
            ('Host-defined workload only; execution success does not establish data quality or training benefit.',))

    def path(self, call):
        self.operator_contract.validate_parameters(call.parameters)
        p = Path(call.work_dir).resolve()
        if ((call.operator_id, call.operator_version) != (self.spec.operator_id, self.spec.version)
                or not p.is_relative_to(self.output_root) or p == self.output_root):
            raise ValueError('workload call is outside its bound identity/root')
        return p/'workload.json'

    def workload(self, call):
        raise NotImplementedError

    def validate_report(self, call, report):
        pass

    def result(self, call, report):
        return {'operator_id': self.spec.operator_id, 'operator_version': self.spec.version,
                'status': 'SUCCESS', 'metrics': {'completion_ratio': 1.0},
                'hard_constraints': {'artifact_valid': True}, 'evidence_refs': [str(self.path(call))],
                'matched_trial_key': call.matched_trial_key.to_dict() if call.matched_trial_key else None}

    def _receipt(self, call, document, reused):
        result = self.result(call, document['report'])
        status = StageStatus(result['status'])
        return OperatorReceipt(call.call_id, self.spec.operator_id, self.spec.version, status,
            document['started_at'], document['finished_at'], return_code=0 if status is StageStatus.SUCCESS else 1, reused=reused,
            effect_hash=file_sha256(self.path(call)), stage_result=result,
            output_artifacts=({'path': str(self.path(call)), 'sha256': file_sha256(self.path(call))},))

    def verify_effect(self, call, receipt):
        path = self.path(call)
        if not path.exists(): return EffectVerification(EvidenceStatus.UNOBSERVED, 'workload effect absent')
        try:
            doc = json.loads(path.read_text())
            if doc['binding'] != payload_sha256([call.to_dict(), self.spec.to_dict()]):
                raise ValueError('workload effect identity changed')
            self.validate_report(call, doc['report'])
            expected = self._receipt(call, doc, False)
            if receipt and (receipt.effect_hash != expected.effect_hash or receipt.stage_result != expected.stage_result
                            or receipt.status is not expected.status or receipt.return_code != expected.return_code):
                raise ValueError('workload receipt changed')
            return EffectVerification(EvidenceStatus.VERIFIED, 'host workload artifacts verified', {str(path): file_sha256(path)})
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def recover(self, call):
        check = self.verify_effect(call, _load_receipt(_receipt_path(call)))
        if check.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED,
                _write_receipt(_receipt_path(call), self._receipt(call, json.loads(self.path(call).read_text()), True)))
        if check.status is EvidenceStatus.UNOBSERVED:
            return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason='resume only through bound workload journals')
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason=check.reason)

    def execute(self, call):
        recovered = self.recover(call)
        if recovered.status is RecoveryStatus.RECOVERED: return recovered.receipt
        if recovered.status is not RecoveryStatus.RETRY_SAFE: raise ValueError(recovered.reason)
        started = datetime.now(timezone.utc).isoformat()
        report = self.workload(call)
        self.validate_report(call, report)
        document = {'binding': payload_sha256([call.to_dict(), self.spec.to_dict()]), 'report': report,
                    'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat()}
        atomic_write_json(self.path(call), document)
        return _write_receipt(_receipt_path(call), self._receipt(call, document, False))


class CommittedEffectAuthority:
    """Read a completed Controller's exact committed host-workload effect."""
    def __init__(self, controller, operator):
        self.controller, self.operator = controller, operator
        self.frozen_sha256 = file_sha256(controller.layout.frozen_contract_path)

    def identity(self):
        c = self.controller
        if file_sha256(c.layout.frozen_contract_path) != self.frozen_sha256:
            raise ValueError('workload frozen contract changed')
        return {'campaign_id': c.goal.campaign_id, 'frozen_sha256': self.frozen_sha256,
                'operator_spec_sha256': self.operator.spec.sha256()}

    def read_effect(self):
        from .models import CampaignState, OperatorCall, OperatorReceipt, StageResult
        self.identity()
        c = self.controller
        state = CampaignState.from_dict(json.loads(c.layout.state_path.read_text()))
        if (state.state.value != 'COMPLETE' or state.data.get('pending_call') or not c._frozen()[0]
                or c.replay().to_dict() != state.to_dict()):
            raise ValueError('workload Controller must be complete, frozen and replayable')
        committed = [e for e in c.store.events.read() if e.event_type == 'ACTION_COMMITTED'
                     and e.payload['receipt']['operator_id'] == self.operator.spec.operator_id]
        if len(committed) != 1: raise ValueError('one committed workload effect is required')
        receipt = OperatorReceipt.from_dict(committed[0].payload['receipt'])
        planned = [e for e in c.store.events.read() if e.event_type == 'ACTION_PLANNED'
                   and e.payload.get('call_id') == receipt.call_id]
        if len(planned) != 1 or planned[0].sequence >= committed[0].sequence:
            raise ValueError('workload effect has no prior exact intent')
        call = OperatorCall.from_dict(planned[0].payload['operator_call'])
        check = self.operator.verify_effect(call, receipt)
        if (check.status is not EvidenceStatus.VERIFIED or receipt.status is not StageStatus.SUCCESS
                or check.artifact_hashes != committed[0].payload['effect_verification']['artifact_hashes']
                or not c._goal_satisfied(c.evaluator.evaluate(StageResult.from_dict(receipt.stage_result), effect_status=EvidenceStatus.VERIFIED))):
            raise ValueError('committed workload was not accepted by its frozen evaluator')
        return call, receipt


class CommittedWorkloadAuthority(CommittedEffectAuthority):
    def read(self):
        call, receipt = self.read_effect()
        return {'report': json.loads(self.operator.path(call).read_text())['report'],
                'origin': self.identity(), 'call': call.to_dict(), 'receipt_sha256': receipt.sha256()}
