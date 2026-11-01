"""Drive host-registered production workloads through a composite Operator.

The existing Controller owns this action; workloads keep their original Gateway
and effect verifiers. Candidate code never becomes an in-process operator. A
preflight-only invocation can expose missing bindings without launching work.
"""
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.dataset.supervision import artifact_closure
from ..models import (canonical_json, EffectVerification, EvidenceStatus, ExpectedOutputSpec, OperatorCall,
                      OperatorReceipt, OperatorSpec, RecoveryResult, RecoveryStatus, StageStatus)
from ..operator_contracts import OperatorContract
from ..operator_development import DevelopmentBudget, _usage
from ..operators import OperatorRegistry, _load_receipt, _receipt_path, _write_receipt
from ..production_qualification import EVIDENCE_ROLES, inspect_validator_bindings
from ..state_store import CampaignLease, atomic_write_json


class ProductionQualificationOperator:
    def __init__(self, *, stage, package, store, predecessor_authority, workload_registry,
                 output_root, evidence_store, execution=None, workloads=(), evaluators=None,
                 admission_probes=None):
        if stage not in {'engineering', 'independent'} or not isinstance(workload_registry, OperatorRegistry):
            raise ValueError('a declared stage and host workload registry are required')
        self.stage, self.package = stage, package
        self.store = ArtifactStore(store.roots)
        self.predecessor, self.registry = predecessor_authority, workload_registry
        self.parent_identity = predecessor_authority.identity()
        self.output_root = Path(output_root).resolve()
        self.evidence_store = evidence_store
        if self.store.roots.get(evidence_store) != self.output_root:
            raise ValueError('producer evidence store must be its exact host-owned output root')
        if evidence_store in {r['store'] for r in package.files}:
            raise ValueError('candidate files cannot own production qualification outputs')
        for name, root in self.store.roots.items():
            if name != evidence_store and (root.is_relative_to(self.output_root) or self.output_root.is_relative_to(root)):
                raise ValueError('qualification output overlaps an input store')
        self.execution, self.workloads = execution, copy.deepcopy(list(workloads))
        self.evaluators = dict(evaluators or {})
        self.admission_probes = dict(admission_probes or {})
        self.admission_identities = {key: probe.identity() for key, probe in self.admission_probes.items()}
        if any(key not in {s.operator_id for s in workload_registry.specs()} for key in self.admission_probes):
            raise ValueError('resource admission must belong to a registered workload')
        for identity in self.admission_identities.values():
            root = Path(identity['output_root']).resolve()
            if root == self.output_root or not root.is_relative_to(self.output_root):
                raise ValueError('resource admission output must stay inside producer write roots')
        self.registry_identity = workload_registry.catalog()
        self.workload_sources = {s.source_path:file_sha256(Path(s.source_path)) for s in workload_registry.specs()
                                 if s.source_path and Path(s.source_path).is_file()}
        self.spec = OperatorSpec('evolution.qualify_production_operator', '1', retry_safe=False,
            source_path=str(Path(__file__).resolve()), timeout_seconds=3600,
            code_revision=payload_sha256({'adapter':file_sha256(Path(__file__)),
                'contracts':file_sha256(Path(__file__).parents[1]/'production_qualification.py'),
                'stage':stage,'parent':self.parent_identity,'package':package.compute_hash(),
                'stores':{k:str(v) for k,v in self.store.roots.items()},
                'registry':self.registry_identity,'workload_sources':self.workload_sources,
                'resource_admission':self.admission_identities,
                'preflight_only':execution is None}),
            allowed_write_roots=(str(self.output_root),),
            expected_output_specs=(ExpectedOutputSpec('production_qualification.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'evaluation',
            {'execution_contract':{'type':'object'},'host_workloads':{'type':'array'}},
            {'production_qualification':{'type':'object'}},
            {'type':'object','additionalProperties':False,
             'required':['execution_contract_sha256','workloads_sha256'],
             'properties':{k:{'type':'string','pattern':'^[0-9a-f]{64}$'}
                           for k in ('execution_contract_sha256','workloads_sha256')}},
            ('json','image_mask'), 'composite', 'reconcile_only', (
                'Workloads must already be host-registered and independently verify actual candidate use.',
                'A preflight-only call performs no candidate or model execution and cannot qualify a version.',
                'Completed workload receipts are reused; an intent without a durable receipt is never resent.',
                'No production adoption, K/M or training benefit follows from this action alone.'))

    def _path(self, call):
        if (call.operator_id,call.operator_version)!=(self.spec.operator_id,'1'):
            raise ValueError('producer call identity changed')
        self.operator_contract.validate_parameters(call.parameters)
        if call.parameters!=self.parameters():raise ValueError('frozen producer execution or workload manifest changed')
        work=Path(call.work_dir)
        if any(p.is_symlink() for p in (work,*work.parents)):
            raise ValueError('qualification output cannot contain symlinks')
        work=work.resolve()
        if work==self.output_root or not work.is_relative_to(self.output_root):
            raise ValueError('producer call must belong to its frozen output root')
        for p in (work/'production_qualification.json',work/'intent.json',work/'.receipts',_receipt_path(call)):
            if p.is_symlink():raise ValueError('producer files cannot be symlinks')
        return work/'production_qualification.json'

    def parameters(self):
        # These hashes are frozen in ACTION_PLANNED/Goal parameters. Keeping
        # execution data out of OperatorSpec avoids self-referential spec hashes.
        return {'execution_contract_sha256':self.execution.compute_hash() if self.execution else '0'*64,
                'workloads_sha256':payload_sha256(self.workloads)}

    def _stable(self):
        if (self.predecessor.identity()!=self.parent_identity or self.registry.catalog()!=self.registry_identity):
            raise ValueError('producer predecessor or registered workload identity changed')
        if any(file_sha256(Path(path))!=sha for path,sha in self.workload_sources.items()):
            raise ValueError('registered workload source changed')
        if {key: probe.identity() for key, probe in self.admission_probes.items()} != self.admission_identities:
            raise ValueError('registered resource admission implementation changed')
        for raw in self.package.files:self.store.resolve(ArtifactRef.from_dict(raw))

    def _preflight(self):
        self._stable()
        parent=self.predecessor.read()
        if self.execution is None:
            required=parent['production_requirements']
            return {'admitted':False,'reason':'EXECUTION_CONTRACT_NOT_BOUND',
                'bindings':inspect_validator_bindings(required['stages'],self.registry,self.evaluators),
                'predecessor_summary_sha256':payload_sha256(parent),
                'current_version_status':parent['current_status'],'prior_lifecycle_usage':parent['lifecycle_total'],
                'new_model_calls':0,'new_candidate_executions':0}
        self.execution.verify_predecessor(self.predecessor)
        if self.execution.package_sha256!=self.package.compute_hash():
            raise ValueError('production package differs from the frozen execution contract')
        candidate_spec=json.loads(self.store.resolve(ArtifactRef.from_dict(self.package.operator_spec)).read_text())
        if (candidate_spec['operator_id'],candidate_spec['version'])!=(self.execution.operator_id,self.execution.operator_version):
            raise ValueError('candidate package does not bind the requested production ID/version')
        rule=self.execution.stages[self.stage]
        if (rule['validator_operator_id']!=self.spec.operator_id or rule['validator_spec_sha256']!=self.spec.sha256()
                or rule['validator_contract_sha256']!=self.operator_contract.compute_hash()):
            raise ValueError('production execution does not bind this registered producer')
        design=self.execution.execution_protocol()
        expected={s['family_id']:s['artifact'] for s in design.sources if s['family_id'] in design.families(self.stage)}
        if not 1<=len(self.workloads)<=16 or len(self.workloads)!=len(expected) or {w['family_id'] for w in self.workloads}!=set(expected):
            raise ValueError('all original stage families need exactly one predeclared workload')
        budget=DevelopmentBudget.from_dict(self.execution.budget)
        reserve={k:0 for k in budget.prior_usage};reserve.update(operator_calls=1,
            wall_seconds=self.spec.timeout_seconds,output_bytes=65536)
        calls=[OperatorCall.from_dict(w['call']) for w in self.workloads]
        for field in ('call_id','idempotency_key','work_dir'):
            if len({getattr(c,field) for c in calls})!=len(calls):
                raise ValueError('workload owners and work directories must be distinct')
        for work in self.workloads:
            if set(work)!={'family_id','input','call','upper_bound'} or work['input']!=expected[work['family_id']]:
                raise ValueError('workload source or frozen fields changed')
            self.store.resolve(ArtifactRef.from_dict(work['input']))
            child=OperatorCall.from_dict(work['call']);self.registry.validate_call(child)
            if any(p.is_symlink() for p in (Path(child.work_dir),*Path(child.work_dir).parents)):
                raise ValueError('workload work directory cannot contain symlinks')
            if (not Path(child.work_dir).resolve().is_relative_to(self.output_root/'workloads')
                    or not child.input_artifact_refs or work['input'] not in child.input_artifact_refs):
                raise ValueError('workload must bind its source and isolated host work directory')
            contract=next((c for c in self.registry.contracts() if c['operator_id']==child.operator_id),None)
            if contract is None or 'production_case' not in contract['output_schemas']:
                raise ValueError('workload lacks the host production_case verifier interface')
            for k,v in _usage(work['upper_bound']).items():reserve[k]+=v*(12 if k=='output_bytes' else 1)
            probe = self.admission_probes.get(child.operator_id)
            if probe is not None:
                bound = self.admission_identities[child.operator_id].get('max_output_bytes')
                if type(bound) is not int or not 1 <= bound <= 1048576:
                    raise ValueError('resource admission needs a finite reserved evidence bound')
                reserve['output_bytes'] += bound
        budget.check_usage((reserve,))
        common = {'new_model_calls':0, 'new_candidate_executions':0,
                  'prior_lifecycle_usage':parent['lifecycle_total'], 'resources':[]}
        if any(call.operator_id not in self.admission_probes for call in calls):
            return {**common, 'admitted':False, 'reason':'RESOURCE_ADMISSION_NOT_BOUND'}
        for child in calls:
            probe = self.admission_probes[child.operator_id]
            observed = probe.preflight(child, budget.resource_deadline_epoch)
            root = Path(self.admission_identities[child.operator_id]['output_root']).resolve()
            if (observed.get('call_sha256') != payload_sha256(child.to_dict())
                    or observed.get('resource_deadline_epoch') != budget.resource_deadline_epoch
                    or observed.get('per_request_gateway_admission_required') is not True
                    or type(observed.get('admitted')) is not bool
                    or not observed.get('artifact_hashes')
                    or any(not Path(name).resolve().is_relative_to(root) for name in observed['artifact_hashes'])
                    or probe.verify(child, observed).status is not EvidenceStatus.VERIFIED):
                raise ValueError('registered workload resource admission does not verify')
            common['resources'].append({'operator_id':child.operator_id, 'call':child.to_dict(),
                                        'proof':observed})
            if not observed['admitted']:
                return {**common, 'admitted':False, 'reason':'WORKLOAD_RESOURCES_UNAVAILABLE'}
        return {**common,'admitted':True,'reason':'BOUND_WORKLOADS'}

    def _read_case(self, work):
        child=OperatorCall.from_dict(work['call']);op=self.registry.get(child.operator_id)
        receipt=_load_receipt(_receipt_path(child))
        if receipt is None:raise ValueError('workload has no durable receipt; reconcile without resend')
        if (receipt.call_id,receipt.operator_id,receipt.operator_version)!=(child.call_id,child.operator_id,child.operator_version):
            raise ValueError('workload receipt belongs to another call or registered implementation')
        verified=op.verify_effect(child,receipt)
        if verified.status is not EvidenceStatus.VERIFIED:
            raise ValueError('workload effect does not verify')
        case=receipt.stage_result.get('production_case')
        if not isinstance(case,dict) or set(case)!={'binding','checks','artifacts','usage','candidate_invocation'}:
            raise ValueError('host workload did not produce the closed production_case interface')
        expected={'execution_contract_sha256':self.execution.compute_hash(),'package_sha256':self.package.compute_hash(),
            'stage':self.stage,'family_id':work['family_id'],'source':work['input'],
            'matched_context_sha256':self.execution.execution_protocol().matched_context_sha256,
            'runtime':self.execution.runtime}
        invocation=case['candidate_invocation']
        if (case['binding']!=expected or not isinstance(invocation,dict)
                or invocation.get('package_sha256')!=self.package.compute_hash()
                or invocation.get('implementation_sha256')!=self.package.implementation['sha256']
                or invocation.get('actual_execution') is not True):
            raise ValueError('actual candidate invocation was not verified for this exact workload')
        checks=case['checks']
        if set(checks)!=set(self.execution.stages[self.stage]['check_ids']) or any(type(v)is not bool for v in checks.values()):
            raise ValueError('every original production check needs a definite host observation')
        if set(case['artifacts'])!=EVIDENCE_ROLES:raise ValueError('production evidence roles are incomplete')
        execution_ref=ArtifactRef.from_dict(invocation['execution_artifact'])
        raw=json.loads(self.store.resolve(execution_ref).read_text())
        if (raw.get('schema_version')!='dataevolver.isolated_candidate_execution.v2'
                or raw.get('status')!='COMPLETED' or raw.get('isolation_verified') is not True
                or raw.get('return_code')!=0 or json.loads(raw['stdout'])!=raw['output']
                or raw['binding'].get('source_sha256')!=self.package.implementation['sha256']
                or payload_sha256(raw['runtime'])!=self.execution.runtime['runner_sha256']
                or raw['runtime'].get('context_scope')!='registered_production_runtime'
                or not raw['binding'].get('trusted_context_sha256')):
            raise ValueError('captured candidate execution does not match the production runtime')
        ArtifactRef('check','context',raw['binding']['trusted_context_sha256'])
        refs=list(case['artifacts'].values())+[work['input'],self.execution.matched_context,execution_ref.to_dict()]
        hashes={}
        for ref in artifact_closure(refs,self.store):
            if ref.store in {x['store'] for x in self.package.files}:
                raise ValueError('candidate-owned reports cannot qualify production')
            path=self.store.resolve(ref)
            if verified.artifact_hashes.get(str(path))!=ref.sha256:
                raise ValueError('host workload verifier omitted an evidence artifact')
            hashes[str(path)]=ref.sha256
        usage=_usage(case['usage'])
        if any(usage[k]>work['upper_bound'][k] for k in usage):raise ValueError('workload exceeded reserved usage')
        return {'family_id':work['family_id'],'call_id':child.call_id,'receipt_sha256':receipt.sha256(),
            'finished_at':receipt.finished_at,
            'receipt_status':receipt.status.value,'checks':checks,'artifacts':case['artifacts'],
            'candidate_invocation':invocation,'usage':usage,'artifact_hashes':hashes,
            'passed':receipt.status is StageStatus.SUCCESS and receipt.return_code==0 and all(checks.values())}

    def _build_report(self, call):
        path=self._path(call);intent=json.loads((path.parent/'intent.json').read_text())
        if intent['call_sha256']!=payload_sha256(call.to_dict()) or intent['spec_sha256']!=self.spec.sha256():
            raise ValueError('production intent changed')
        self._stable()
        admission=intent['admission']
        for entry in admission.get('resources', []):
            child = OperatorCall.from_dict(entry['call'])
            if self.admission_probes[entry['operator_id']].verify(child, entry['proof']).status is not EvidenceStatus.VERIFIED:
                raise ValueError('frozen resource admission evidence changed')
        cases=[self._read_case(w) for w in self.workloads] if admission['admitted'] else []
        usage={k:0 for k in admission['prior_lifecycle_usage']};usage['operator_calls']=1
        for case in cases:
            for k,n in case['usage'].items():usage[k]+=n
        ended=max([datetime.fromisoformat(intent['at']),
                   *[datetime.fromisoformat(c['finished_at']) for c in cases]])
        usage['wall_seconds']=math.ceil((ended-datetime.fromisoformat(intent['started_at'])).total_seconds())
        report={'call_sha256':intent['call_sha256'],'operator_spec_sha256':self.spec.sha256(),'at':intent['at'],
            'admission':admission,'cases':cases,'qualified':bool(cases) and all(c['passed'] for c in cases),
            'lifecycle_usage':usage,'production_adoption_allowed':False,'new_execution_or_budget_authorization':False}
        if admission['admitted']:
            manifests={}
            for role in EVIDENCE_ROLES:
                file=path.parent/(role+'.json')
                payload={'role':role,'cases':[{'family_id':c['family_id'],'artifact':c['artifacts'][role],
                    'receipt_sha256':c['receipt_sha256']} for c in cases]}
                if file.exists() and json.loads(file.read_text())!=payload:raise ValueError('role manifest changed')
                manifests[role]=(file,payload)
            report['role_manifests']={k:{'path':str(f),'payload':v} for k,(f,v) in manifests.items()}
        child_bytes=sum(c['usage']['output_bytes'] for c in cases)
        fixed_bytes=(path.parent/'intent.json').stat().st_size+(self.output_root/'single-call.json').stat().st_size
        resources = {name for entry in admission.get('resources', [])
                     for name in entry['proof']['artifact_hashes']}
        fixed_bytes += sum(Path(name).stat().st_size for name in resources)
        fixed_bytes+=sum(len((canonical_json(v['payload'])+'\n').encode()) for v in report.get('role_manifests',{}).values())
        for _ in range(12):
            if admission['admitted']:
                report['lifecycle_total']=DevelopmentBudget.from_dict(self.execution.budget).check_usage((usage,))
            size=child_bytes+fixed_bytes+len((canonical_json(report)+'\n').encode())
            if usage['output_bytes']==size:break
            usage['output_bytes']=size
        else:raise ValueError('producer byte accounting did not stabilize')
        return report

    def _receipt(self, call, path, reused=False):
        report=json.loads(path.read_text());admitted=report['admission']['admitted']
        result={'operator_id':self.spec.operator_id,'operator_version':'1','status':'SUCCESS',
            'metrics':{'production_qualification_passed':float(report['qualified']),
                       'production_workloads_admitted':float(admitted)},
            'hard_constraints':{'artifact_valid':True},'evidence_refs':[str(path)],
            'lifecycle_usage':report['lifecycle_usage']}
        if admitted:
            protocol=self.execution.execution_protocol()
            result.update(qualification_binding={'plan_sha256':self.execution.compute_hash(),
                'package_sha256':self.package.compute_hash(),'protocol_sha256':protocol.compute_hash(),
                'stage':self.stage,'matched_context_sha256':protocol.matched_context_sha256},
                source_family_ids=protocol.families(self.stage),
                check_results={k:all(c['checks'][k] for c in report['cases']) for k in self.execution.stages[self.stage]['check_ids']},
                production_evidence={'schema_version':'dataevolver.registered_production_evidence.v1',
                    'execution_contract_sha256':self.execution.compute_hash(),'stage':self.stage,
                    'runtime':self.execution.runtime,'source_artifacts':[s['artifact'] for s in protocol.sources
                        if s['family_id'] in protocol.families(self.stage)],
                    'artifacts':{k:ArtifactRef(self.evidence_store,Path(v['path']).relative_to(self.output_root).as_posix(),
                        file_sha256(Path(v['path']))).to_dict() for k,v in report['role_manifests'].items()}})
        return OperatorReceipt(call.call_id,self.spec.operator_id,'1',StageStatus.SUCCESS,report['at'],report['at'],
            return_code=0,reused=reused,effect_hash=file_sha256(path),
            output_artifacts=({'path':str(path),'sha256':file_sha256(path)},),stage_result=result)

    def verify_effect(self,call,receipt):
        try:
            path=self._path(call)
            if not path.exists():return EffectVerification(EvidenceStatus.UNOBSERVED,'producer report absent')
            value=json.loads(path.read_text())
            if value!=self._build_report(call):raise ValueError('producer evidence changed')
            if receipt is not None and (receipt.effect_hash!=file_sha256(path)
                    or receipt.stage_result!=self._receipt(call,path).stage_result):
                raise ValueError('producer receipt changed')
            hashes={str(path):file_sha256(path)}
            for entry in value['admission'].get('resources', []):
                hashes.update(entry['proof']['artifact_hashes'])
            for case in value['cases']:hashes.update(case['artifact_hashes'])
            for entry in value.get('role_manifests',{}).values():
                f=Path(entry['path'])
                if json.loads(f.read_text())!=entry['payload']:raise ValueError('producer role evidence changed')
                hashes[str(f)]=file_sha256(f)
            return EffectVerification(EvidenceStatus.VERIFIED,'actual registered workload receipts and evidence verified',
                artifact_hashes=hashes)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as e:
            return EffectVerification(EvidenceStatus.CONTRADICTED,str(e))

    def recover(self,call):
        path=self._path(call);receipt=_load_receipt(_receipt_path(call))
        if path.exists():
            check=self.verify_effect(call,receipt)
            if check.status is EvidenceStatus.VERIFIED:
                return RecoveryResult(RecoveryStatus.RECOVERED,_write_receipt(_receipt_path(call),self._receipt(call,path,True)))
            return RecoveryResult(RecoveryStatus.AMBIGUOUS,reason=check.reason)
        if (path.parent/'intent.json').exists():
            try:
                report=self._build_report(call)
                self._save_report(path,report)
                return RecoveryResult(RecoveryStatus.RECOVERED,_write_receipt(_receipt_path(call),self._receipt(call,path,True)))
            except (OSError,ValueError,KeyError,TypeError,RuntimeError) as e:
                return RecoveryResult(RecoveryStatus.AMBIGUOUS,reason=str(e))
        return RecoveryResult(RecoveryStatus.RETRY_SAFE,reason='no producer intent or effect')

    def _save_report(self,path,report):
        for value in report.get('role_manifests',{}).values():atomic_write_json(Path(value['path']),value['payload'])
        atomic_write_json(path,report)

    def execute(self,call):
        path=self._path(call);self.output_root.mkdir(parents=True,exist_ok=True)
        with CampaignLease(self.output_root/'.production-qualification.lease',ttl_seconds=3600):
            recovered=self.recover(call)
            if recovered.status is RecoveryStatus.RECOVERED:return recovered.receipt
            if recovered.status is not RecoveryStatus.RETRY_SAFE:raise ValueError(recovered.reason)
            journal=self.output_root/'single-call.json';binding=payload_sha256(call.to_dict())
            if journal.exists() and json.loads(journal.read_text())!=binding:raise ValueError('one frozen producer cannot reset its allowance')
            started_at=datetime.now(timezone.utc).isoformat()
            admission=self._preflight()
            atomic_write_json(journal,binding)
            atomic_write_json(path.parent/'intent.json',{'call_sha256':binding,'spec_sha256':self.spec.sha256(),
                'admission':admission,'at':datetime.now(timezone.utc).isoformat(),'started_at':started_at})
            if admission['admitted']:
                for work in self.workloads:
                    child=OperatorCall.from_dict(work['call'])
                    if _load_receipt(_receipt_path(child)) is not None:
                        raise ValueError('a new production stage cannot reuse an older workload owner')
                    self.registry.get(child.operator_id).execute(child)
                    self._read_case(work)
            report=self._build_report(call);self._save_report(path,report)
            return _write_receipt(_receipt_path(call),self._receipt(call,path))
