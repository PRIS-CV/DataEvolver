"""Read committed native probes through their own frozen verifier, without new effects."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess

from dataevolver.dataset.contracts import file_sha256


def summarize_boundary_probe(value):
    report=value['report'];rows=report['mask_diagnostics'];visual=report['visual_review']['receipt']['output']
    if (report['proposal_status']!='OBSERVED_UNQUALIFIED' or report['boundary_witness_qualification']!='NOT_ASSESSED'
            or report['effect_attribution']!='NOT_ASSESSED' or report['image_submission_allowed']
            or report['candidate_adoption_allowed'] or report['new_accepted_records'] or report['new_permissions']
            or report['immutable_masks_modified'] or report['generation_model_calls']):
        raise ValueError('boundary feedback must preserve its original unqualified scope')
    ids={row['subject_id']:row['target_instance_id'] for row in rows}
    if (not ids or len(ids)!=len(rows)
            or {row['subject_id']:row['target_instance_id'] for row in visual['assessments']}!=ids
            or len(visual['assessments'])!=len(ids)):
        raise ValueError('boundary feedback changed the original selected-instance mapping')
    summary={'schema_version':'dataevolver.boundary_probe_feedback.v1','scope':'registered_native_boundary_probe',
        'origin':value['origin'],'runtime_query':report['runtime_query'],'mask_diagnostics':rows,
        'runtime_visual_assessment':visual,'segmentation_request':report['segmentation']['request'],
        'segmentation_receipt_sha256':report['segmentation']['receipt_hash'],
        'visual_request':report['visual_review']['request'],'visual_receipt_sha256':report['visual_review']['receipt_hash'],
        'lifecycle_total':value['lifecycle_total'],'lifecycle_usage':value['lifecycle_usage'],
        'proposal_status':'OBSERVED_UNQUALIFIED','boundary_witness_qualification':'NOT_ASSESSED',
        'effect_attribution':'NOT_ASSESSED','image_submission_allowed':False,'candidate_adoption_allowed':False,
        'production_qualification_passed':False,'algorithm_defect_established':False,'new_accepted_records':0,
        'mask_differences_establish_pixel_truth':False,'local_visual_support_is_full_task_completion':False,
        'new_model_calls_for_feedback':0,'new_probe_executions':0}
    if len(json.dumps(summary,ensure_ascii=False).encode())>65536:
        raise ValueError('boundary feedback exceeds its 64KiB bound')
    return copy.deepcopy(summary)


class FrozenBoundaryFeedbackAuthority:
    def __init__(self, *, config, descriptor, python):
        required={'run','failed_config','failed_run','profile_path','worker_path','bootstrap_path'}
        if set(descriptor)!=required:raise ValueError('boundary feedback needs exact frozen host locations')
        self.config=Path(config).resolve();self.descriptor={k:str(Path(v).resolve()) for k,v in descriptor.items()}
        self.python=Path(python).absolute();self._identity=self._current_identity();self._cached=None

    def _current_identity(self):
        paths=[*self.config.rglob('*'),self.python,Path(self.descriptor['run'])/'exit_code',
            *(Path(self.descriptor[k]) for k in ('profile_path','worker_path','bootstrap_path'))]
        return {'schema_version':'dataevolver.boundary_feedback_authority.v1','config':str(self.config),
            'descriptor':copy.deepcopy(self.descriptor),'python':str(self.python),'reader_sha256':file_sha256(Path(__file__)),
            'files':{str(p):file_sha256(p) for p in paths if p.is_file() and '__pycache__' not in p.parts}}

    def identity(self):
        if self._current_identity()!=self._identity:raise ValueError('original committed boundary probe changed')
        return copy.deepcopy(self._identity)

    def read(self):
        self.identity()
        if self._cached is None:
            script='import json,sys\n'+inspect.getsource(_read_completed_probe)
            script+='\nprint(json.dumps(_read_completed_probe(*json.load(sys.stdin))))'
            result=subprocess.run([str(self.python),'-B','-c',script],input=json.dumps([str(self.config),self.descriptor,str(self.python)]),
                text=True,capture_output=True,timeout=300,cwd=self.config/'source',
                env={**os.environ,'PYTHONPATH':str(self.config/'source/src'),'CUDA_VISIBLE_DEVICES':'',
                     'PYTHONDONTWRITEBYTECODE':'1','GIT_OPTIONAL_LOCKS':'0'})
            self.identity()
            if result.returncode:raise ValueError('frozen boundary feedback validation failed: '+result.stderr[-2500:])
            self._cached=json.loads(result.stdout)
        for name,digest in self._cached['artifact_hashes'].items():
            if file_sha256(Path(name))!=digest:raise ValueError('original boundary feedback artifact changed')
        self.identity()
        return summarize_boundary_probe(self._cached),{**self.identity(),'artifact_hashes':self._cached['artifact_hashes'],
            'input_store_roots':self._cached['store_roots'],'verification':self._cached['verification']}

    def input_context(self):
        """Original constraints and correspondence, with no new labels or geometry."""
        self.read()
        return copy.deepcopy(self._cached['input_context'])


def _read_completed_probe(config, descriptor, python):
    import json
    from pathlib import Path
    from dataevolver.dataset.contracts import ArtifactRef,ArtifactStore,file_sha256,payload_sha256
    from dataevolver.harness.boundary_recovery import FrozenBoundaryFailureAuthority,BoundaryProbeRecoveryDomain
    from dataevolver.harness.training_candidate_feedback import FrozenTrainingCandidateFeedbackAuthority
    from dataevolver.harness.adapters.training_candidate_feedback import TrainingCandidateFeedbackOperator
    from dataevolver.harness.adapters.boundary_evidence import BoundaryEvidenceProbeOperator
    from dataevolver.harness.adapters.training_supervision import TrainingSupervisionOperator
    from dataevolver.adapters.model_contracts import ModelProfile,ModelPolicy
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.adapters.codex_planner import RecoveringPlannerBackend
    from dataevolver.adapters.cli_planner_service import CLIPlannerEndpointProvider
    from dataevolver.adapters.training_media_worker import TrainingMediaProvider
    from dataevolver.harness.models import GoalSpec,OperatorCall,OperatorReceipt,EvidenceStatus
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.operators import OperatorRegistry
    descriptor=dict(descriptor)
    cfg=Path(config);protected=cfg/'protected';output=cfg/'output'
    assert (Path(descriptor['run'])/'exit_code').read_text().strip()=='0'
    def tree(root):return {str(f):file_sha256(f) for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    before=tree(cfg)
    origin=json.loads((protected/'origin.json').read_text());diagnostic_descriptor=json.loads((protected/'descriptor.json').read_text())
    authority=FrozenTrainingCandidateFeedbackAuthority(config=origin['config'],descriptor=diagnostic_descriptor,python=python)
    inputs=authority.probe_inputs()
    failure=FrozenBoundaryFailureAuthority(config=descriptor['failed_config'],run=descriptor['failed_run'],profile_path=descriptor['profile_path'],worker_path=descriptor['worker_path'],bootstrap_path=descriptor['bootstrap_path'],python=python)
    contract=json.loads((protected/'resource-contract.json').read_text());roots=json.loads((protected/'resource-roots.json').read_text())
    resource=TrainingSupervisionOperator(allowed_write_root=cfg/'resource-admission',contract=contract,roots=roots)
    identity=json.loads((cfg/'resident-identity.json').read_text());store=ArtifactStore({**inputs['store_roots'],'boundary_probe':identity['authorized_dataset_roots'][0]})
    profile=ModelProfile.from_dict(json.loads(Path(descriptor['profile_path']).read_text()));worker=json.loads(Path(descriptor['worker_path']).read_text())
    pg=ModelGateway(profiles=(profile,),policy=ModelPolicy({'agent.runtime_rsi':profile.profile_id},2),receipt_root=output/'planner_model_receipts',artifact_store=store)
    pg.factories[profile.profile_id]=lambda:CLIPlannerEndpointProvider(profile,worker,store,output/'native')
    backend=RecoveringPlannerBackend(pg,model_id=profile.model,timeout_seconds=900)
    sam=next(ModelProfile.from_dict(x) for x in contract['models']['profiles'] if x['profile_id']=='segmentation')
    sg=ModelGateway(profiles=(sam,),policy=ModelPolicy({'training.segment':'segmentation'},1),receipt_root=output/'segment_model_receipts',artifact_store=store)
    sg.factories['segmentation']=lambda:TrainingMediaProvider(sam,contract['models']['workers']['segmentation'],store,'boundary_probe',output/'native-sam')
    bootstrap=TrainingCandidateFeedbackOperator(authority=authority,output_root=output)
    probe=BoundaryEvidenceProbeOperator(authority=authority,segment_gateway=sg,review_backend=backend,resource_operator=resource,output_root=output)
    domain=BoundaryProbeRecoveryDomain(feedback_authority=authority,failure_authority=failure,observation_operator_id=probe.spec.operator_id,admission=json.loads((protected/'admission.json').read_text()))
    from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
    goal=GoalSpec.from_json((protected/'goal.json').read_text());planner=RuntimeRSIPlanner(backend=backend,domain=domain,trace_root=output/'campaigns'/goal.campaign_id/'planner')
    registry=OperatorRegistry();registry.register(bootstrap);registry.register(probe)
    c=EvolutionController.open(repo_root=cfg/'source',campaigns_root=output/'campaigns',goal=goal,registry=registry,planner=planner,evaluator_contract=EvaluatorContract.from_path(cfg/'evaluator.json'),schema_paths=tuple(protected.rglob('*.json'))+(cfg/'resident-identity.json',),execution_paths=tuple((cfg/'source/src/dataevolver').rglob('*.py'))+(cfg/'host-launcher.py',))
    from dataevolver.harness.training_trial_authority import _read_only_event
    import sys
    def forbidden(*a, **kw):raise RuntimeError('boundary feedback reader cannot execute or infer')
    EvolutionController.run=EvolutionController.tick=RuntimeRSIPlanner.plan=forbidden
    BoundaryEvidenceProbeOperator.execute=BoundaryEvidenceProbeOperator.recover=forbidden
    ModelGateway.invoke=ModelGateway._provider=forbidden
    sys.addaudithook(_read_only_event)
    s=c.status();assert c._frozen()[0] and c.replay().to_dict()==s.to_dict() and s.state.value=='ABSTAIN'
    events=c.store.events.read();rows=[e for e in events if e.event_type=='ACTION_COMMITTED' and e.payload['receipt']['operator_id']==probe.spec.operator_id]
    assert len(rows)==1
    receipt=OperatorReceipt.from_dict(rows[0].payload['receipt']);planned=next(e for e in events if e.event_type=='ACTION_PLANNED' and e.payload['call_id']==receipt.call_id);call=OperatorCall.from_dict(planned.payload['operator_call'])
    verified=probe.verify_effect(call,receipt);assert verified.status is EvidenceStatus.VERIFIED,verified.reason
    assert verified.artifact_hashes==rows[0].payload['effect_verification']['artifact_hashes']
    assert sg.audit_usage()['model_calls']==pg.audit_usage()['model_calls']==1
    assert tree(cfg)==before
    proof={'frozen_contract_valid':True,'replay_matches_snapshot':True,'committed_boundary_effect_verified':True,'own_files_unchanged':len(before),'new_model_calls':0,'new_operator_executions':0,'new_planner_decisions':0,'old_runtime_query_unchanged':call.parameters==failure.read()['pending_call']['parameters'],'artifact_closure_files':len(verified.artifact_hashes),'new_quality_or_editing_authority':False}
    report_path=Path(receipt.output_artifacts[0]['path']);report=json.loads(report_path.read_text())
    old=authority.read()[0]
    paths={Path(k) for k in verified.artifact_hashes}
    paths.update(p for p in cfg.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    snapshot=authority.config/'protected/registered-evidence/original-trial.json'
    assert file_sha256(snapshot)==inputs['original_trial_snapshot_sha256']
    paths.add(snapshot)
    original_store=ArtifactStore(inputs['store_roots'])
    paths.update(original_store.resolve(ArtifactRef.from_dict(row['artifact'])) for row in inputs['trial']['media'])
    usage={'model_calls':2,'planner_calls':0,'operator_calls':s.operator_calls,
     'wall_seconds':__import__('math').ceil(sg.audit_usage()['observed_wall_seconds']+pg.audit_usage()['observed_wall_seconds']),
     'output_bytes':sum(f.stat().st_size for f in output.rglob('*') if f.is_file())+sum(f.stat().st_size for f in Path(identity['authorized_dataset_roots'][0]).rglob('*') if f.is_file()),'image_experiments':0}
    prior=domain.admission['budget']['prior_usage']
    return {'report':report,'origin':{'campaign_id':goal.campaign_id,'call_id':receipt.call_id,'source_commit':(cfg/'source-commit').read_text().strip(),
     'frozen_contract_sha256':file_sha256(c.layout.frozen_contract_path),'receipt_sha256':receipt.sha256(),
     'report_sha256':file_sha256(report_path),'package_sha256':old['origin']['package_sha256'],
     'resource_deadline_epoch':domain.admission['resource_deadline_epoch'],'original_request_sha256':old['origin']['original_request_sha256']},
     'lifecycle_usage':usage,'lifecycle_total':{k:prior[k]+usage[k] for k in prior},
     'store_roots':{**inputs['store_roots'],'boundary_probe':identity['authorized_dataset_roots'][0]},
     'input_context':{'source':inputs['trial']['source'],'original_segment_parameters':inputs['original_segment_parameters'],
        'selected_subject_ids':inputs['selected_subject_ids'],'media':inputs['trial']['media'],
        'original_task_constraints':inputs['contract']['production']['tasks'],
        'original_contract_sha256':payload_sha256(inputs['contract']),
        'original_snapshot':{'path':str(snapshot),'sha256':file_sha256(snapshot)},
        'resource_deadline_epoch':inputs['contract']['production']['resource_deadline_epoch']},
     'verification':proof,'artifact_hashes':{str(p):file_sha256(p) for p in paths}}
