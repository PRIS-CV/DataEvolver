"""CPU protocol handoff for an already completed, Controller-bound build.

No ModelGateway invoke, new model attempt, candidate execution or old-run edit.
"""
import json
import math
import hashlib
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelRequest, ModelReceipt
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from .. import operator_development as development
from ..models import CampaignState, EffectVerification, EvidenceStatus, ExpectedOutputSpec, OperatorReceipt, OperatorSpec, RecoveryResult, RecoveryStatus, StageStatus
from ..operator_contracts import OperatorContract
from ..operators import _load_receipt, _receipt_path, _write_receipt
from .operator_candidate_build import CandidateSourceCompiler


class ControllerBuildOutputAuthority:
    def __init__(self, controller, operator, call_id):
        self.controller, self.operator, self.call_id = controller, operator, call_id

    def identity(self):
        c = self.controller
        return {"campaign_id": c.goal.campaign_id, "frozen_sha256": file_sha256(c.layout.frozen_contract_path),
                "operator_spec_sha256": self.operator.spec.sha256(), "call_id": self.call_id}

    def read(self):
        from ..models import OperatorCall
        c, op = self.controller, self.operator
        state = CampaignState.from_dict(json.loads(c.layout.state_path.read_text()))
        if not c._frozen()[0] or c.replay().to_dict() != state.to_dict() or state.state.value not in {"ABSTAIN", "COMPLETE"}:
            raise ValueError("original build is not frozen, replayable and terminal")
        events = c.store.events.read()
        planned = [e for e in events if e.event_type == 'ACTION_PLANNED' and e.payload.get('call_id') == self.call_id]
        committed = [e for e in events if e.event_type == 'ACTION_COMMITTED' and e.payload.get('call_id') == self.call_id]
        if len(planned) != 1 or len(committed) != 1 or planned[0].sequence >= committed[0].sequence:
            raise ValueError("original build was not committed exactly once")
        call = OperatorCall.from_dict(planned[0].payload['operator_call'])
        receipt = OperatorReceipt.from_dict(committed[0].payload['receipt'])
        if call.operator_id != 'evolution.build_operator_candidate' or op.spec.sha256() != c.registry.get(call.operator_id).spec.sha256():
            raise ValueError("original builder identity changed")
        verification = op.verify_effect(call, receipt)
        if verification.status is not EvidenceStatus.VERIFIED or verification.artifact_hashes != committed[0].payload['effect_verification']['artifact_hashes']:
            raise ValueError("original build effect changed")
        effect = json.loads(Path(receipt.output_artifacts[0]['path']).read_text())
        request = ModelRequest.from_dict(effect['request'])
        response = op.gateway.recover(request)
        usage = op.gateway.audit_usage()
        if (response is None or response.without_reuse_hash() != effect['model_receipt_sha256']
                or usage != effect['model_usage'] or usage['model_calls'] != 1 or usage['unknown_duration_attempts']):
            raise ValueError("original native completion or costs changed")
        return ModelReceipt.from_dict(response.to_dict()), {**self.identity(), 'receipt_sha256': receipt.sha256(),
            'model_receipt_sha256': response.without_reuse_hash(), 'usage': usage}


class CandidateBuildRecoveryOperator(CandidateSourceCompiler):
    def __init__(self, *, plan, store, request_authority, build_authority, output_root):
        self.plan = development.read_development_plan(plan.to_dict())
        self.store, self.authority, self.build_authority = ArtifactStore(store.roots), request_authority, build_authority
        self.output_root = Path(output_root).resolve()
        self.budget = development.DevelopmentBudget.from_dict(plan.budget)
        self.interface = development.OperatorInterface.from_dict(json.loads(store.resolve(ArtifactRef.from_dict(plan.target_interface)).read_text()))
        original = build_authority.operator.plan
        for name in ('capability_request', 'request_campaign_id', 'request_frozen_sha256', 'qualification_protocol', 'target_interface'):
            if getattr(plan, name) != getattr(original, name):
                raise ValueError("CPU protocol recovery cannot change original request/interface/protocol")
        response, proof = build_authority.read()
        prior = self.budget.prior_usage
        if (prior['model_calls'] < 1 or prior['operator_calls'] < 1
                or self.budget.limits['model_calls'] != prior['model_calls']
                or self.budget.limits['planner_calls'] != prior['planner_calls']
                or self.budget.limits['image_experiments'] != prior['image_experiments']
                or prior['wall_seconds'] < math.ceil(proof['usage']['observed_wall_seconds'])
                or self.budget.resource_deadline_epoch != development.DevelopmentBudget.from_dict(original.budget).resource_deadline_epoch):
            raise ValueError("CPU recovery must retain completed usage and authorize no new model/image effects")
        self.bound_source = build_authority.identity()
        for root in self.store.roots.values():
            if self.output_root.is_relative_to(root) or root.is_relative_to(self.output_root):
                raise ValueError("CPU recovery output overlaps an input store")
        self.spec = OperatorSpec('evolution.recover_operator_candidate', '1', retry_safe=True,
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({'source': self.bound_source,
                'adapter': file_sha256(Path(__file__)), 'compiler': file_sha256(Path(__file__).with_name('operator_candidate_build.py')),
                'plan': plan.to_dict(), 'stores': {k: str(v) for k,v in self.store.roots.items()}}),
            allowed_write_roots=(str(self.output_root), str(self.store.roots[plan.candidate_store])),
            expected_output_specs=(ExpectedOutputSpec('candidate_recovery.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id,'1',self.spec.sha256(),'evolution',
            {'completed_build': {'type':'object'}}, {'candidate_package': {'type':'object'}},
            {'type':'object','additionalProperties':False,'properties':{}},('json','source_code'),'pure_data','verify_then_retry',
            ('Only exact previously completed source bytes are materialized; no candidate is executed.',
             'Engineering/independent qualification, promotion, reuse and method benefit remain unverified.'))

    def _validate(self, call):
        self.operator_contract.validate_parameters(call.parameters)
        work=Path(call.work_dir)
        if ((call.operator_id,call.operator_version)!=(self.spec.operator_id,'1') or any(p.is_symlink() for p in (work,*work.parents))
                or not work.resolve().is_relative_to(self.output_root) or work.resolve()==self.output_root
                or self.build_authority.identity()!=self.bound_source):
            raise ValueError('CPU recovery call or source binding changed')
        if any(p.is_symlink() for p in (work/'candidate_recovery.json',work/'candidate-package.json',work/'.receipts',_receipt_path(call))):
            raise ValueError('CPU recovery metadata symlink is forbidden')
        return work.resolve()

    def _effect(self, call, *, materialize=True):
        work=self._validate(call);response,proof=self.build_authority.read()
        if materialize:
            package,inspection=self._materialize(call,response)
        else:
            package=development.CandidatePackage.from_dict(json.loads((work/'candidate-package.json').read_text()))
            inspection=development.inspect_candidate(self.plan,package,self.store,self.authority)
        originals={item['path']:hashlib.sha256(item['content'].encode()).hexdigest() for item in response.output['files']}
        entries={ref['path']:ref['sha256'] for ref in package.files}
        if any(entries.get(name)!=digest for name,digest in originals.items()) or package.implementation['path']!=response.output['entrypoint'].split(':')[0]:
            raise ValueError('recovered source bytes differ from original native output')
        return {'call_sha256':payload_sha256(call.to_dict()),'operator_spec_sha256':self.spec.sha256(),
            'source':proof,'package_sha256':package.compute_hash(),'inspection':inspection,
            'original_entrypoint':response.output['entrypoint'],'materialized_entrypoint':package.implementation['path'],
            'new_model_calls':0,'candidate_code_executed':False,'status':'BUILT_UNQUALIFIED',
            'engineering_qualification':'NOT_ASSESSED','independent_qualification':'NOT_ASSESSED',
            'promoted':False,'reused':False,'method_improvement_verified':False}

    def verify_effect(self,call,receipt):
        try:
            path=self._validate(call)/'candidate_recovery.json'
            if not path.exists():return EffectVerification(EvidenceStatus.UNOBSERVED,'CPU candidate recovery absent')
            value=json.loads(path.read_text())
            if value!=self._effect(call,materialize=False) or (receipt is not None and (
                    receipt.effect_hash!=file_sha256(path) or receipt.call_id!=call.call_id
                    or receipt.operator_id!=self.spec.operator_id or receipt.operator_version!='1'
                    or receipt.stage_result!=self._receipt(call,False).stage_result)):
                raise ValueError('CPU candidate recovery evidence changed')
            return EffectVerification(EvidenceStatus.VERIFIED,'exact completed source output recovered without model submission',artifact_hashes={str(path):file_sha256(path)})
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED,str(exc))

    def _receipt(self,call,reused):
        from datetime import datetime,timezone
        path=self._validate(call)/'candidate_recovery.json';now=datetime.now(timezone.utc).isoformat()
        return OperatorReceipt(call.call_id,self.spec.operator_id,'1',StageStatus.SUCCESS,now,now,return_code=0,reused=reused,
            effect_hash=file_sha256(path),output_artifacts=({'path':str(path),'sha256':file_sha256(path)},),
            stage_result={'operator_id':self.spec.operator_id,'operator_version':'1','status':'SUCCESS',
                'metrics':{'candidate_recovered':1.0},'hard_constraints':{'artifact_valid':True},'evidence_refs':[str(path)]})

    def recover(self,call):
        prior=_load_receipt(_receipt_path(call));verification=self.verify_effect(call,prior)
        if verification.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED,_write_receipt(_receipt_path(call),self._receipt(call,True)))
        if verification.status is EvidenceStatus.UNOBSERVED and prior is None:
            return RecoveryResult(RecoveryStatus.RETRY_SAFE,reason='CPU-only immutable materialization may resume')
        return RecoveryResult(RecoveryStatus.AMBIGUOUS,reason=verification.reason)

    def execute(self,call):
        recovered=self.recover(call)
        if recovered.status is RecoveryStatus.RECOVERED:return recovered.receipt
        if recovered.status is not RecoveryStatus.RETRY_SAFE:raise ValueError(recovered.reason)
        self._write(self._validate(call)/'candidate_recovery.json',canonical_bytes(self._effect(call)))
        return _write_receipt(_receipt_path(call),self._receipt(call,False))


def canonical_bytes(value):
    """Bind recovery to strict UTF-8 dataset JSON, including unescaped Unicode."""
    from dataevolver.dataset.contracts import canonical_json
    return canonical_json(value).encode()
