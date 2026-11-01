"""Registered confined execution on receipt-backed training evidence.

This is the candidate side diagnostic, not production qualification or adoption.
The same Controller owns its intent, actual child output and recovery. There is
no image/Planner budget, model access or candidate import in the host process.
"""
import json
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef,file_sha256,payload_sha256
from ..authority_qualification import authenticated,crypto_identity
from ..candidate_sandbox import ProductionContextRunner
from ..candidate_trust import ProductionCandidateTrustContext,canonical_digest
from ..operator_contracts import OperatorContract
from ..training_candidate_context import compile_training_evidence,verify_requested_observation
from .operator_candidate_checks import CandidateChecksOperator


class TrainingCandidateExecutionOperator(CandidateChecksOperator):
    def __init__(self,*,plan,package,store,request_authority,trial_authority,observation_authority,
                 runner,check_protocol,context_store,output_root):
        if type(runner) is not ProductionContextRunner:
            raise ValueError('an explicit registered production confinement profile is required')
        self.context_store=context_store
        self.trial_authority,self.observation_authority=trial_authority,observation_authority
        self.origins={'trial':trial_authority.read(),'observation':observation_authority.read()}
        self.origin_identities={'trial':trial_authority.identity(),'observation':observation_authority.identity(),
                                'request':request_authority.identity()}
        self.crypto=crypto_identity()
        self.compiled=compile_training_evidence(self.origins['trial'],self.origins['observation'],store)
        super().__init__(plan=plan,package=package,store=store,request_authority=request_authority,
                         runner=runner,check_protocol=check_protocol,output_root=output_root)
        self.spec=replace(self.spec,operator_id='evolution.execute_training_candidate',
            source_path=str(Path(__file__).resolve()),code_revision=payload_sha256({
                'checks':self.spec.code_revision,'adapter':file_sha256(Path(__file__)),
                'compiler':file_sha256(Path(__file__).parents[1]/'training_candidate_context.py'),
                'crypto':self.crypto,'context_store':self.context_store,'origins':self.origin_identities}))
        self.operator_contract=OperatorContract(self.spec.operator_id,'1',self.spec.sha256(),'evaluation',
            {'candidate_package':{'type':'object'},'registered_training_evidence':{'type':'object'}},
            {'confined_candidate_result':{'type':'object'}},
            {'type':'object','additionalProperties':False,'properties':{}},
            ('json','source_code','image'),'pure_data','verify_then_reuse',(
                'Actual registered evidence and exact candidate bytes are bound before isolated execution.',
                'Boundary envelopes and uncertainty grant no pixels; unsupported observations stay blocked.',
                'This diagnostic cannot qualify, promote or reactivate a revoked candidate.',
                'No image, Planner or model request is authorized; a missing durable child result is ambiguous.'))

    def _protected_input_stores(self):
        return (*self.plan.protected_stores,self.context_store)

    def _read_host(self,raw):
        ref=ArtifactRef.from_dict(raw)
        if ref.store!=self.context_store:raise ValueError('production evidence must belong to its dedicated host store')
        return json.loads(self.store.resolve(ref).read_text())

    def _case_data(self,case):
        context=ProductionCandidateTrustContext.from_dict(self._read_host(self.protocol['contexts'][case['case_id']]))
        payload=self._read_host(case['input'])
        return context,payload

    def _check_protocol(self):
        p=self.protocol;base={'schema_version','plan_sha256','package_sha256','runtime_sha256','cases','budget'}
        if set(p)!=base|{'contexts','origin_snapshots','projection'} or p['schema_version']!='dataevolver.training_candidate_execution.v1':
            raise ValueError('unsupported registered training execution protocol')
        context_root=self.store.roots[self.context_store]
        if self.context_store in self.plan.protected_stores or self.context_store==self.plan.candidate_store:
            raise ValueError('production context must be a distinct host store')
        for key,root in self.store.roots.items():
            if key!=self.context_store and (root.is_relative_to(context_root) or context_root.is_relative_to(root)):
                raise ValueError('production context overlaps another store')
        if len(p['cases'])!=1 or set(p['contexts'])!={p['cases'][0]['case_id']}:
            raise ValueError('one predeclared registered regression execution is required')
        self.protocol={**{k:p[k] for k in base},'schema_version':'dataevolver.candidate_behavior_checks.v1'}
        try:super()._check_protocol()
        finally:self.protocol=p
        if (self.trial_authority.identity()!=self.origin_identities['trial']
                or self.observation_authority.identity()!=self.origin_identities['observation']
                or self.request_authority.identity()!=self.origin_identities['request']):
            raise ValueError('registered training origin changed')
        if set(p['origin_snapshots'])!={'trial','observation'}:
            raise ValueError('registered origin snapshots are incomplete')
        for key,raw in p['origin_snapshots'].items():
            if self._read_host(raw)!=self.origins[key]:raise ValueError('registered origin snapshot changed')
            for name,digest in self.origins[key]['artifact_hashes'].items():
                if file_sha256(Path(name))!=digest:raise ValueError('registered original evidence changed')
        evidence,manifest,projection=self.compiled
        request=json.loads(self.store.resolve(ArtifactRef.from_dict(self.plan.capability_request)).read_text())
        registered=self.origins['observation'];trial=self.origins['trial']['trial']
        verify_requested_observation(request,self.origins['trial'],registered)
        if self._read_host(p['projection'])!=projection:raise ValueError('authority projection changed')
        case=p['cases'][0];context,payload=self._case_data(case)
        from ..operator_development import QualificationProtocol
        design=QualificationProtocol.from_dict(json.loads(self.store.resolve(
            ArtifactRef.from_dict(self.plan.qualification_protocol)).read_text()))
        if (context.fixture_family_id not in design.families('engineering')
                or {payload_sha256(x) for x in context.evidence_artifacts}
                   !={payload_sha256(x) for x in [*p['origin_snapshots'].values(),p['projection']]}):
            raise ValueError('production context changed its predeclared family or origin closure')
        source=next(x['artifact'] for x in design.sources if x['family_id']==context.fixture_family_id)
        source=json.loads(self.store.resolve(ArtifactRef.from_dict(source)).read_text())
        if (source.get('kind')!='existing-failed-source-regression'
                or source.get('request_sha256')!=self.plan.capability_request['sha256']):
            raise ValueError('registered regression cannot borrow another qualification source family')
        if (context.package_sha256!=self.package.compute_hash()
                or context.implementation_sha256!=self.package.implementation['sha256']
                or context.qualification_request_sha256!=self.plan.capability_request['sha256']
                or context.input_artifact!=case['input'] or context.payload_sha256!=canonical_digest(payload)
                or context.origin_bindings!={
                    'trial_authority_sha256':payload_sha256(self.origin_identities['trial']),
                    'observation_authority_sha256':payload_sha256(self.origin_identities['observation']),
                    'request_authority_sha256':payload_sha256(self.origin_identities['request']),
                    'trial_sha256':self.origins['trial']['trial_sha256'],
                    'observation_sha256':self.origins['observation']['observation_sha256'],
                    'compiler_sha256':file_sha256(Path(__file__).parents[1]/'training_candidate_context.py'),
                    'manifest_sha256':canonical_digest(manifest)}):
            raise ValueError('production context source/request/origin binding changed')
        for ref in context.evidence_artifacts:self.store.resolve(ArtifactRef.from_dict(ref))
        unsigned={k:v for k,v in payload['evidence'].items() if k!='source'}
        if (unsigned!=evidence or payload['parameters']!={} or payload['evidence']['source']['manifest']!=manifest
                or not authenticated(payload,context)):
            raise ValueError('signed production payload is not the exact registered evidence projection')
        if len(json.dumps(payload,separators=(',',':')).encode())>self.runner.limits.input_bytes:
            raise ValueError('registered production input exceeds the explicit finite runner bound')
        # Unsupported provenance cannot be turned into an executable positive
        # merely by changing a test expectation. It remains an actual negative.
        if projection['registered_status']=='unsupported' and (case['kind']!='negative'
                or case['expected_subset'].get('status')!='unsupported'):
            raise ValueError('unsupported registered evidence cannot authorize a positive result')

    def _execution_binding(self,case,payload):
        return {**super()._execution_binding(case,payload),
                'trusted_context_sha256':self._case_data(case)[0].compute_hash()}

    def _run_case(self,case,source,payload):
        return self.runner.run_with_context(source,payload,self._case_data(case)[0])

    def _report(self,call):
        result=super()._report(call)
        raw=json.loads((self._path(call).parent/'executions/0.json').read_text())
        output=raw.get('output') or {}
        blocked=(raw['status']=='COMPLETED' and raw['isolation_verified']
            and output.get('status')=='unsupported' and output.get('authorized_support')=={}
            and output.get('image_submission_allowed') is False
            and output.get('candidate_adoption_allowed') is False)
        result.update(registered_evidence_used=True,production_context_scope='registered_production_runtime',
            original_registered_status=self.compiled[2]['registered_status'],
            unsupported_submission_block_verified=blocked,production_qualification_passed=False,
            engineering_qualification='NOT_ASSESSED',independent_qualification='NOT_ASSESSED',
            image_request_submitted=False,new_accepted_records=0)
        return result

    def _receipt(self,call,path,reused):
        receipt=super()._receipt(call,path,reused);report=json.loads(path.read_text())
        return replace(receipt,stage_result={**receipt.stage_result,'metrics':{
            'registered_candidate_execution_captured':1.0,
            'unsupported_submission_block_verified':float(report['unsupported_submission_block_verified']),
            'production_qualification_passed':0.0}})
