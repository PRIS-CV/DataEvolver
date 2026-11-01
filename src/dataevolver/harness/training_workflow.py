"""A frozen training queue inside the existing EvolutionController.

The runtime Planner owns observations and R proposals. Registered effects distil
K, validate it on new sources, promote it, and deliver independently loaded data.
No source-specific repair is supplied by the workflow compiler.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .adapters.training_rsi import (TRIAL_ID, OBSERVE_ID, TrainingRSIDomain,
    TrainingRSITrialOperator, TrainingRSIObservationOperator, _RuntimeEffect, _json, _write_immutable)
from .capabilities import CapabilityStore
from .models import (
    Capability,
    CapabilityStatus,
    ControllerState as ControllerState,
    ExpectedOutputSpec,
    FailureSignature,
    Intervention,
    InterventionLevel,
    OperatorSpec,
    RecoveryStatus,
)
from .state_store import atomic_write_json
from .training_workflow_delivery import (assemble_collection, verify_trial_bundle,
                                         require_visual_acceptance, capture_execution_audit)

SCHEMA = 'dataevolver.training_workflow.v1'
POLICY_SCHEMA = {'type':'object', 'additionalProperties':False,
    'required':['failure_mode','applicability','probe_instruction','repair_instruction','precautions'],
    'properties':{k:({'type':'array','items':{'type':'string'},'minItems':1} if k=='precautions' else
                    {'type':'string','minLength':1}) for k in
                  ['failure_mode','applicability','probe_instruction','repair_instruction','precautions']}}


def task_scope(task, models):
    """Conservative empirical scope; no selected/all or count extrapolation."""
    return {'selection_mode':task['selection_mode'], 'subjects':len(task['subjects']),
            'selected':len(task['selected_subject_ids']), 'models_sha256':payload_sha256(models),
            'subject_descriptions':sorted(s.get('description','') for s in task['subjects']),
            'operator':'data.training_rsi_trial:1'}


def validate_policy(value):
    from jsonschema import validate
    validate(value,POLICY_SCHEMA)
    import re
    text=json.dumps(value,ensure_ascii=False)
    if re.search(r'(?i)(box_xyxy|subject_id|main-train-|/data\d|\.png|\b(?:exec|subprocess)\s*\(|\b0?\.\d+\b)',text):
        raise ValueError('a reusable policy cannot contain source coordinates, IDs, artifact paths or code')
    return value


def record_by_phase(trial):
    family=_json(Path(trial['inner_work_dir'])/'families'/(trial['source_family']+'.json'))
    phases={}
    for row in family['model_requests']:
        phase=next((p for p in ('refine-generate','generate','resolve','segment','shadow-probe',
                               'refine-background','background','review')
                    if row['request']['request_id'].endswith('-'+p)),None)
        if phase:phases[phase]=row
    return phases


def verify_runtime_pair(baseline,candidate,history,*,experience_id=None):
    require_visual_acceptance(candidate)
    if baseline.get('qualified') is not False or baseline['source']!=candidate['source']:
        raise ValueError('knowledge needs an actual failed baseline and same-source accepted repair')
    if not any(x['decision']['decision']=='ADOPT_CANDIDATE' and
               x['intervention']['candidate_delta']==candidate['patch'] for x in history):
        raise ValueError('repair was not adopted by the frozen Evaluator')
    delta=candidate['patch']
    if (delta.get('proposal_actor')!='runtime_agent' or delta.get('layer')!='R' or
            delta['source']!=candidate['source'] or not delta.get('evidence')):
        raise ValueError('knowledge requires runtime observations and a runtime R proposal')
    for item in delta['evidence']:
        evidence=_json(item['path'])
        if (payload_sha256(evidence)!=item['sha256'] or evidence['source']!=candidate['source'] or
                evidence['proposal_actor']!='runtime_agent'):
            raise ValueError('runtime proposal evidence is stale or belongs to another source')
    old,new=record_by_phase(baseline),record_by_phase(candidate)
    for name in ('refine-generate','generate','resolve','segment','shadow-probe'):
        if old[name]['request']!=new[name]['request'] or old[name]['receipt_hash']!=new[name]['receipt_hash']:
            raise ValueError('matched repair changed source, references, masks or original probe')
    a,b=old['background']['request'],new['background']['request']
    # Request IDs identify attempts; they do not establish an operational change.
    def operational(value):
        if isinstance(value,list):return [operational(v) for v in value]
        if not isinstance(value,dict):return value
        if value.get('schema_version')=='dataevolver.artifact_ref.v1':
            return {'sha256':value['sha256'],'media_type':value['media_type']}
        return {k:operational(v) for k,v in value.items() if k!='request_id'}
    if (operational(a)==operational(b) or
            any(a[k]!=b[k] for k in ('role','seed','response_schema'))):
        raise ValueError('knowledge must change the actual request payload under matched model settings')
    if experience_id is not None and (candidate.get('experience_id')!=experience_id or
            delta.get('experience',{}).get('capability_id')!=experience_id):
        raise ValueError('experience did not affect the actual compiled request')
    return {'verified':True,'source_sha256':candidate['source']['sha256'],
        'before_sha256':baseline['candidate']['sha256'],'after_sha256':candidate['candidate']['sha256'],
        'before_request_sha256':payload_sha256(a),'after_request_sha256':payload_sha256(b),
        'conditioning_changed':baseline['media'][5]['artifact']!=candidate['media'][5]['artifact'],
        'experience_id':experience_id,'proposal_actor':'runtime_agent',
        'training_benefit_verified':False,'method_improvement_verified':False}


class TrainingWorkflowDomain:
    def __init__(self,config,campaign_dir,*,backend=None):
        self.value=copy.deepcopy(config);self.campaign=Path(campaign_dir).resolve();self.backend=backend
        if self.value['schema_version']!=SCHEMA:raise ValueError('unknown workflow schema')
        self.stages=self.value['stages']
        if len({s['id'] for s in self.stages})!=len(self.stages):raise ValueError('duplicate workflow stage')
        for name,digest in self.value['input_hashes'].items():
            if file_sha256(Path(name))!=digest:raise ValueError('frozen workflow input changed')

    def config(self):return self.value

    def workflow(self,state):
        return state.data.get('training_workflow',{'index':0,'completed':[],'proofs':[],
            'validation_pairs':[],'reuse_pairs':[],'planner_start':0,'experiment_start':0})

    def stage(self,state):return self.stages[self.workflow(state)['index']]

    def asset_id(self,state):return self.stage(state).get('task',{}).get('slot_id','')

    def allow_knowledge_learning(self,state):
        stage=self.stage(state)
        return stage['kind']=='main' and stage['task']['split']=='train'

    def anchor(self,stage):return self.campaign/'workflow'/stage['id']/'baseline.json'

    def _child(self,stage):
        contract=copy.deepcopy(self.value['production_contract']);task=stage['task']
        contract['production'].update(count=1,tasks=[task],split_counts={k:int(task['split']==k)
            for k in ('train','validation','test')},checker_precheck=False,max_consecutive_failures=1)
        contract['production'].pop('model_reuse',None)
        root=Path(self.value['dataset_root'])/stage['id']; portable=Path(self.value['bundle_root'])/stage['id']
        scopes={n:{'dataset':str(root/n),'bundle':str(portable/n)} for n in
            ['baseline',*[f'candidate-{i:03d}' for i in range(self.value['limits']['max_experiments'])]]}
        roots=dict(self.value['roots'])
        roots[contract['dataset_store']]=scopes['baseline']['dataset']
        roots[contract['bundle_store']]=scopes['baseline']['bundle']
        anchor=self.anchor(stage)
        seed={'fresh_generation':True,'family_id':task['slot_id']+'-a0','source_sha256':None,
            'config_sha256':payload_sha256(contract),'model_records':[],'file_hashes':{},
            'work_dir':str(self.campaign/'workflow'/stage['id']/'empty-seed'),
            'dataset_root':str(root/'empty-seed-cache')}
        if anchor.exists():
            binding=_json(anchor);trial=_json(binding['trial_path'])
            if payload_sha256(trial)!=binding['trial_sha256']:raise ValueError('baseline anchor changed')
            family=_json(Path(trial['inner_work_dir'])/'families'/(trial['source_family']+'.json'))
            seed.update(fresh_generation=False,source_sha256=trial['source']['sha256'],
                model_records=family['model_requests'],work_dir=trial['inner_work_dir'],
                dataset_root=scopes['baseline']['dataset'])
        domain={'schema_version':'dataevolver.training_rsi_domain.v1','production_contract':contract,
            'roots':roots,'scopes':scopes,'seed':seed,
            'evidence_root':str(Path(self.value['evidence_root'])/stage['id'])}
        if self.value.get('surface_capability'):domain['surface_capability']=self.value['surface_capability']
        return TrainingRSIDomain(domain,self.campaign,backend=self.backend)

    def _state(self):
        from .models import CampaignState
        return CampaignState.from_dict(_json(self.campaign/'state.json'))

    def _experience(self,stage,*,candidate=False):
        path=self.campaign/'knowledge'/('candidates.jsonl' if candidate else 'promoted.jsonl')
        state_path=self.campaign/'state.json'
        expected=_json(state_path).get('data',{}).get('training_workflow',{}).get('promoted_sha256') if state_path.exists() else None
        if not candidate and expected and file_sha256(path)!=expected:
            raise ValueError('frozen promoted knowledge changed during production')
        values=CapabilityStore(path).list_capabilities()
        scope=task_scope(stage['task'],self.value['production_contract']['models'])
        status=CapabilityStatus.CANDIDATE if candidate else CapabilityStatus.PROMOTED
        return [x for x in values if x.status is status and x.intervention.parameters['scope']==scope]

    def context(self,state):
        stage=self.stage(state);wf=self.workflow(state)
        if stage['kind'] in {'validation','main'}:
            result=self._child(stage).context(state)
            candidate=stage['kind']=='validation'
            values=self._experience(stage,candidate=candidate)
            result.update(compatible_experience_ids=[] if candidate else [x.capability_id for x in values],
                knowledge_validation_only=candidate,testable_experience_ids=[x.capability_id for x in values] if candidate else [],
                experience_input={'records':[x.to_dict() for x in values],
                    'interpretation':'candidate test only' if candidate else 'frozen promoted policy; source-specific observations still required'})
        else:
            result={'available_evidence_ids':['current_stage'],'allowed_observations':[],
                'allowed_patches':[],'compatible_experience_ids':[],
                'current_stage':_json(state.data['last_stage_result']['evidence_refs'][0]),
                'completion_rule':'COMPLETE only if the registered stage succeeded; otherwise ABSTAIN. No data repair tools in this stage.'}
        result['workflow']={'stage':stage['id'],'kind':stage['kind'],'completed':wf['completed'],
            'remaining':len(self.stages)-wf['index']-1,'no_skip':True}
        return result

    def planner_artifacts(self,state):
        if self.stage(state)['kind'] in {'validation','main'}:return self._child(self.stage(state)).planner_artifacts(state)
        return ()

    def initial_parameters(self,state):
        return {'workflow_stage':self.stage(state)['id'],'scope':'baseline','patch':{},'experience_id':''}

    def auto_complete_stage(self,state):
        # These registered effects contain their own deterministic acceptance.
        # Do not add a model request after the final exact execution audit.
        return self.stage(state)['kind'] in {'distill','promote','export'}

    def trial_key(self,name,state):
        stage=self.stage(state);value=payload_sha256(stage)
        return {'input_artifact_hash':value,'parent_artifact_hash':value,'seed':stage.get('task',{}).get('seed',0),
            'operator_id':TRIAL_ID,'operator_version':'1','model_config_hash':payload_sha256(self.value['production_contract']['models']),
            'preprocessing_hash':payload_sha256(self.value['production_contract']['planning']),
            'evaluator_hash':self.value['evaluator_sha256'],'environment_contract_hash':payload_sha256(self.value['roots'])}

    def decision_budget_available(self,state):
        wf=self.workflow(state)
        # A recorded decision must still be consumed when it used the last slot.
        return bool(state.data.get('rsi_pending_decision')) or state.planner_calls-wf['planner_start']<self.value['limits']['planner_per_task']

    def compile_observation(self,decision,state):
        stage=self.stage(state)
        if stage['kind'] not in {'validation','main'}:raise ValueError('stage has no observation authority')
        operator,params=self._child(stage).compile_observation(decision,state)
        return operator,{**params,'workflow_stage':stage['id']}

    def compile_capability_request(self,decision,state):
        stage=self.stage(state)
        if stage['kind'] not in {'validation','main'}:
            raise ValueError('stage has no data capability request authority')
        result=self._child(stage).compile_capability_request(decision,state)
        return {**result,'workflow_stage':stage['id']}

    def compile_intervention(self,decision,state):
        stage=self.stage(state);wf=self.workflow(state)
        if stage['kind'] not in {'validation','main'}:raise ValueError('stage has no R authority')
        if state.experiments_used-wf['experiment_start']>=self.value['limits']['experiments_per_task']:
            raise ValueError('per-task frozen experiment budget exhausted')
        result=self._child(stage).compile_intervention(decision,state)
        if decision['experience_id']:
            values=self._experience(stage,candidate=stage['kind']=='validation')
            k=next((x for x in values if x.capability_id==decision['experience_id']),None)
            if k is None:raise ValueError('experience scope, status or model version is incompatible')
            experience={'capability_id':k.capability_id,'version':k.version,'status':k.status.value,
                        'snapshot_sha256':payload_sha256([x.to_dict() for x in values])}
            delta={**result.candidate_delta,'experience':experience}
            result=result.with_updates(candidate_delta=delta,parameters={**result.parameters,'patch':delta})
        return result.with_updates(parameters={**result.parameters,'workflow_stage':stage['id']})

    def propose_transition(self,state):
        from .domains import TransitionProposal
        stage=self.stage(state);wf=copy.deepcopy(self.workflow(state))
        trial=_json(state.data['last_stage_result']['evidence_refs'][0])
        if stage['kind']=='promote':
            wf['promoted_sha256']=trial['promoted_store_sha256']
            if file_sha256(self.campaign/'knowledge/promoted.jsonl')!=wf['promoted_sha256']:
                raise ValueError('promoted snapshot changed before stage commit')
        if stage['kind'] in {'validation','main'}:
            proof=verify_trial_bundle(trial,self.campaign/'verified'/stage['id'],
                receipt_roots=[self.campaign],allow_fixture=self.value.get('allow_fixture',False))
            baseline=_json(_json(self.anchor(stage))['trial_path'])
            if stage['kind']=='validation':
                if trial['experience_id']:
                    pair=verify_runtime_pair(baseline,trial,state.data.get('rsi_history',[]),experience_id=trial['experience_id'])
                    pilot=_json(self.value['pilot_trial'])
                    if pair['source_sha256']==pilot['source']['sha256']:raise ValueError('validation source leaked from development')
                    wf['validation_pairs'].append({**pair,'proof':proof,'history':state.data.get('rsi_history',[]),
                        'candidate_trial':state.data['last_stage_result']['evidence_refs'][0],
                        'baseline_trial':_json(self.anchor(stage))['trial_path']})
                # A successful R0 without a K experiment is explicitly inconclusive.
            else:
                wf['proofs'].append(proof)
                if trial['experience_id']:
                    if not self._experience(stage):raise ValueError('main reuse lacks promoted compatible K')
                    pair=verify_runtime_pair(baseline,trial,state.data.get('rsi_history',[]),experience_id=trial['experience_id'])
                    forbidden={x['source_sha256'] for x in wf['validation_pairs']}|{_json(self.value['pilot_trial'])['source']['sha256']}
                    if pair['source_sha256'] in forbidden:raise ValueError('reuse source is not independent')
                    wf['reuse_pairs'].append(pair)
        wf['completed'].append({'stage':stage['id'],'kind':stage['kind'],
            'result_sha256':payload_sha256(trial),'history':state.data.get('rsi_history',[]),
            'observations':state.data.get('rsi_observations',[])})
        wf.update(index=wf['index']+1,planner_start=state.planner_calls,experiment_start=state.experiments_used)
        archive=self.campaign/'workflow'/stage['id']/'completed.json';_write_immutable(archive,wf['completed'][-1])
        if wf['index']==len(self.stages):
            proofs=[_json(x) for x in self.value['accepted_proofs']]+wf['proofs']
            assets=[case['family_id'] for p in proofs for case in p['cases']]
            if len(assets)!=30 or len(set(assets))!=30:raise ValueError('final workflow source count mismatch')
            return TransitionProposal.from_state(state,{**state.data,'training_workflow':wf,
                'accepted_assets':assets},complete=True,effect='WORKFLOW_DELIVERED',
                completion_evidence=wf['completed'][-1])
        data={k:v for k,v in state.data.items() if k not in {'rsi_history','rsi_observations','rsi_incumbent_patch',
            'attempted_interventions','proposed_intervention','rsi_last_decision','canary_phase','candidate_decision',
            'experiment_candidate_evaluation','experiment_candidate_stage_result','experiment_baseline_evaluation',
            'last_stage_result','last_evaluation','last_effect_verification','failure_signature','pending_call','adopted_intervention'}}
        data['training_workflow']=wf
        return TransitionProposal.from_state(state,data,counters={'round_index':state.round_index+1},
            completion_evidence=wf['completed'][-1])

    def complete_stage(self,c,state):
        return c.commit_domain_transition(state,self.propose_transition(state))


class WorkflowOperator(_RuntimeEffect):
    """Dispatch existing registered effects; it is not a second Controller."""
    def __init__(self,domain,*,observation=False):
        self.domain=domain;self.observation=observation
        self.spec=OperatorSpec(OBSERVE_ID if observation else TRIAL_ID,'1',retry_safe=True,
            source_path=str(Path(__file__)),code_revision=file_sha256(Path(__file__)),
            model_config_hash=payload_sha256(domain.config()),allowed_write_roots=tuple(map(str,
                [domain.campaign,domain.value['dataset_root'],domain.value['bundle_root'],domain.value['evidence_root']])),
            expected_output_specs=(ExpectedOutputSpec('runtime-effect.json'),ExpectedOutputSpec('observation.json' if observation else 'trial.json')))

    def verify_effect(self,call,receipt):
        result=super().verify_effect(call,receipt)
        if result.status.value!='VERIFIED' or self.observation:return result
        try:
            value=_json(Path(call.work_dir)/'trial.json')
            if 'media' in value:
                from PIL import Image
                from dataevolver.dataset.contracts import ArtifactRef
                stage=next(x for x in self.domain.stages if x['id']==call.parameters['workflow_stage'])
                store=self.domain._child(stage).scope(value['scope'])
                for item in value['media']:
                    with Image.open(store.resolve(ArtifactRef.from_dict(item['artifact']))) as image:image.verify()
                if file_sha256(Path(value['inner_work_dir'])/'production.json')!=value['production_sha256']:
                    raise ValueError('production evidence changed')
            return result
        except (OSError,ValueError,KeyError) as exc:
            from .models import EffectVerification,EvidenceStatus
            return EffectVerification(EvidenceStatus.CONTRADICTED,str(exc))

    def execute(self,call):
        self._check(call);recovered=self.recover(call)
        if recovered.status is RecoveryStatus.RECOVERED:return recovered.receipt
        if recovered.status is not RecoveryStatus.RETRY_SAFE:raise ValueError(recovered.reason)
        started=datetime.now(timezone.utc).isoformat();state=self.domain._state();stage=self.domain.stage(state)
        if call.parameters['workflow_stage']!=stage['id']:raise ValueError('call does not belong to the current frozen task')
        params={k:v for k,v in call.parameters.items() if k!='workflow_stage'}
        if stage['kind'] in {'validation','main'}:
            compiled_path=Path(call.work_dir)/'compiled-domain.json'
            if compiled_path.exists():
                child=TrainingRSIDomain(_json(compiled_path),self.domain.campaign,backend=self.domain.backend)
            else:
                child=self.domain._child(stage)
                _write_immutable(compiled_path,child.config())
            inner=TrainingRSIObservationOperator(child) if self.observation else TrainingRSITrialOperator(child,visual_backend=self.domain.backend)
            inner_call=call.with_updates(call_id=call.call_id+':task',idempotency_key=call.idempotency_key+':task',
                                        parameters=params,work_dir=str(Path(call.work_dir)/'task'))
            receipt=inner.execute(inner_call);verify=inner.verify_effect(inner_call,receipt)
            if verify.status.value!='VERIFIED':raise ValueError('child effect could not be verified')
            value=_json(receipt.stage_result['evidence_refs'][0])
            if not self.observation and params['scope']=='baseline':
                _write_immutable(self.domain.anchor(stage),{'trial_path':receipt.stage_result['evidence_refs'][0],
                    'trial_sha256':payload_sha256(value)})
            result={k:v for k,v in receipt.stage_result.items() if k not in {'operator_id','operator_version','evidence_refs'}}
        else:
            if self.observation:raise ValueError('non-media stage cannot probe')
            value=self._control(stage,state)
            result={'status':'SUCCESS','metrics':{'qualified_ratio':1.0},
                    'hard_constraints':{'artifact_valid':True,'source_identity':True,'quality_pass':True}}
        path=Path(call.work_dir)/('observation.json' if self.observation else 'trial.json');atomic_write_json(path,value)
        return self._finish(call,result,path,started)

    def _control(self,stage,state):
        d=self.domain;wf=d.workflow(state)
        if stage['kind']=='distill':
            pilot=_json(d.value['pilot_trial']);baseline=_json(d.value['pilot_baseline'])
            comparison=verify_runtime_pair(baseline,pilot,d.value['pilot_history'])
            d.backend.set_artifacts(())
            prompt={'task':'Distil a reusable conditional failure-investigation policy from this actual runtime comparison.',
                'evidence':comparison,'history':d.value['pilot_history'],
                'rules':'Do not copy coordinates, source IDs, paths or task answers. Describe how to obtain new source evidence and decide a bounded repair. Candidate only; no claims of generalized or training benefit.'}
            policy=validate_policy(json.loads(d.backend.complete_json([{'role':'user','content':json.dumps(prompt)}],schema=POLICY_SCHEMA)))
            capability=Capability(capability_id=state.campaign_id+':runtime-policy',version='1',status=CapabilityStatus.CANDIDATE,
                failure_signature=FailureSignature('BACKGROUND_QUALITY_GAP'),
                intervention=Intervention('runtime-policy','BACKGROUND_QUALITY_GAP',TRIAL_ID,InterventionLevel.L1,
                    parameters={'policy':policy,'scope':d.value['experience_scope']},candidate_delta={'layer':'K','policy':policy}),
                validation_evidence=(d.value['pilot_trial'],d.value['pilot_proof']),
                compatibility_hash=payload_sha256(d.value['experience_scope']),source_asset_id=pilot['source_family'],
                novelty_key=payload_sha256(policy))
            store=CapabilityStore(d.campaign/'knowledge/candidates.jsonl');store.append_candidate(capability)
            actual=next(x for x in store.list_capabilities() if x.capability_id==capability.capability_id)
            if actual.to_dict()!=capability.to_dict():raise ValueError('candidate policy identity collision')
            return {'kind':'distill','candidate':actual.to_dict(),'comparison':comparison,'model_call':d.backend.last_trace()}
        if stage['kind']=='promote':
            pairs=wf['validation_pairs']
            if not pairs:raise ValueError('no independent source has validated candidate K; promotion blocked')
            candidates=CapabilityStore(d.campaign/'knowledge/candidates.jsonl').list_capabilities()
            if len(candidates)!=1:raise ValueError('promotion must identify one system-distilled candidate')
            candidate=candidates[0]
            for pair in pairs:
                actual=verify_runtime_pair(_json(pair['baseline_trial']),_json(pair['candidate_trial']),pair['history'],experience_id=candidate.capability_id)
                if actual['source_sha256']!=pair['source_sha256']:raise ValueError('independent comparison changed')
                if not pair['proof']['accepted']:raise ValueError('independent bundle is unaccepted')
            promoted=candidate.with_updates(status=CapabilityStatus.PROMOTED,version='2',
                validation_evidence=(*candidate.validation_evidence,*[p['candidate_trial'] for p in pairs]))
            store=CapabilityStore(d.campaign/'knowledge/promoted.jsonl');store.append_candidate(promoted)
            if store.list_capabilities()!=(promoted,):raise ValueError('promoted store readback changed')
            return {'kind':'promote','promoted':promoted.to_dict(),'independent_comparisons':pairs,
                    'promoted_store_sha256':file_sha256(store.path),
                    'training_benefit_verified':False,'method_improvement_verified':False}
        if stage['kind']=='export':
            if not wf['reuse_pairs']:raise ValueError('no actual compatible promoted K reuse; final delivery blocked')
            proofs=[_json(x) for x in d.value['accepted_proofs']]+wf['proofs']
            delivery=assemble_collection(proofs,d.value['delivery_root'],{'train':18,'validation':6,'test':6},
                receipt_roots=[d.campaign],allow_fixture=d.value.get('allow_fixture',False))
            audit=capture_execution_audit([d.campaign,*d.value['historical_audit_roots']],
                Path(d.value['delivery_root'])/'execution-audit',config_roots=d.value['historical_config_roots'])
            if not {r['request_id'] for r in delivery['usage']['attempts']}<={r['request_id'] for r in audit['usage']['attempts']}:
                raise ValueError('execution audit omitted historical failed attempts')
            root=Path(d.value['delivery_root']);audit_sha=file_sha256(root/'execution-audit/audit.json')
            index=_json(root/'collection.json')
            index['execution_audit']={'path':'execution-audit/audit.json','sha256':audit_sha}
            atomic_write_json(root/'collection.json',index)
            import subprocess,sys
            result=subprocess.run([sys.executable,'-I',str(root/'load_samples.py')],check=True,capture_output=True,text=True)
            (root/'independent-load.jsonl').write_text(result.stdout)
            delivery.update(collection_sha256=file_sha256(root/'collection.json'),execution_audit_sha256=audit_sha,
                complete_model_usage=audit['usage'])
            atomic_write_json(root/'delivery-validation.json',delivery)
            return {'kind':'export','delivery':delivery,'runtime_experience_reuse':wf['reuse_pairs'],
                'execution_audit_sha256':audit_sha,
                'complete_model_usage':audit['usage'],
                'planner_usage':d.backend.gateway.audit_usage(),'method_improvement_verified':False,'training_benefit_verified':False}
        raise ValueError('unregistered workflow stage')
