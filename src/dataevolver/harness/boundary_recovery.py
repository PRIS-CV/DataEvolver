"""Read-only admission of an exact runtime probe after a proven pre-forward failure.

A failed reservation stays spent. A new frozen Controller may execute the same
runtime decision against a repaired resident with one separately admitted SAM
attempt and its unused visual-review allowance. This reader never invokes it.
"""
import copy
import inspect
import json
import math
import os
from pathlib import Path
import subprocess

from dataevolver.dataset.contracts import file_sha256, payload_sha256


def check_failed_boundary_stage(value):
    """Reject ambiguity, completed/partial effects and changed runtime arguments."""
    d, call, request = value['decision'], value['pending_call'], value['request']
    failure, job, intent = value['native_failure'], value['native_job'], value['gateway_intent']
    if (d['action'] != 'OBSERVE' or d['observation']['tool'] != 'probe_instance_boundaries'
            or call['operator_id'] != 'data.probe_instance_boundaries'
            or json.loads(d['observation']['parameters_json']) != call['parameters']
            or job['request'] != request or job['request_hash'] != intent['request_hash']
            or intent['state'] != 'FAILED' or intent['error_code'] != 'RECORDED_FAILURE'
            or type(intent.get('wall_seconds')) not in (int, float)
            or not math.isfinite(intent['wall_seconds']) or intent['wall_seconds'] < 0
            or failure['code'] != 'KeyError' or failure['message'] != "'loading_info'"
            or failure['binding'] != job['binding'] or failure['resident_pid'] != value['old_resident_pid']
            or value['successful_output_present'] or value['new_dataset_files']
            or value['sam_forward_count'] != 0 or value['visual_review_calls'] != 0
            or value['unknown_duration_attempts'] != 0
            or not value['missing_component_before_forward_verified']):
        raise ValueError('boundary recovery requires an exact known pre-forward failure without any output')
    return copy.deepcopy(d)


class FrozenBoundaryFailureAuthority:
    def __init__(self, *, config, run, profile_path, worker_path, bootstrap_path, python):
        self.config = Path(config).resolve()
        self.locations = {k: str(Path(v).resolve()) for k, v in dict(run=run,
            profile_path=profile_path, worker_path=worker_path, bootstrap_path=bootstrap_path).items()}
        self.python = Path(python).absolute()
        self._identity = self._current_identity(); self._cached = None

    def _current_identity(self):
        paths = [*self.config.rglob('*'), *(Path(v) for k,v in self.locations.items() if k != 'run'),
                 Path(self.locations['run'])/'exit_code']
        return {'schema_version': 'dataevolver.frozen_boundary_failure_authority.v1',
            'config': str(self.config), 'locations': self.locations, 'python': str(self.python),
            'reader_sha256': file_sha256(Path(__file__)),
            'files': {str(p): file_sha256(p) for p in paths if p.is_file() and '__pycache__' not in p.parts}}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError('original boundary failure evidence changed')
        return copy.deepcopy(self._identity)

    def read(self):
        self.identity()
        if self._cached is None:
            script = 'import json,sys\n'+inspect.getsource(_read_frozen_boundary_failure)
            script += '\nprint(json.dumps(_read_frozen_boundary_failure(*json.load(sys.stdin))))'
            result = subprocess.run([str(self.python), '-B', '-c', script],
                input=json.dumps([str(self.config), self.locations, str(self.python)]),
                text=True, capture_output=True, timeout=300, cwd=self.config/'source',
                env={**os.environ, 'PYTHONPATH': str(self.config/'source/src'),
                     'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'GIT_OPTIONAL_LOCKS': '0'})
            self.identity()
            if result.returncode:
                raise ValueError('original frozen boundary failure validation failed: '+result.stderr[-2500:])
            self._cached = json.loads(result.stdout)
        check_failed_boundary_stage(self._cached)
        return copy.deepcopy(self._cached)


def _read_frozen_boundary_failure(config, locations, python):
    import ast, json, math
    from pathlib import Path
    from dataevolver.dataset.contracts import ArtifactStore, file_sha256, payload_sha256
    from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy, ModelReceipt
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.adapters.codex_planner import RecoveringPlannerBackend
    from dataevolver.adapters.cli_planner_service import CLIPlannerEndpointProvider
    from dataevolver.adapters.training_media_worker import TrainingMediaProvider
    from dataevolver.harness.training_candidate_feedback import FrozenTrainingCandidateFeedbackAuthority
    from dataevolver.harness.adapters.training_candidate_feedback import TrainingCandidateFeedbackOperator
    from dataevolver.harness.boundary_evidence import BoundaryProbeDomain
    from dataevolver.harness.adapters.boundary_evidence import BoundaryEvidenceProbeOperator
    from dataevolver.harness.adapters.training_supervision import TrainingSupervisionOperator
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.models import GoalSpec, OperatorCall
    from dataevolver.harness.evaluator import EvaluatorContract
    cfg=Path(config); protected=cfg/'protected'; output=cfg/'output'; run=Path(locations['run'])
    if (run/'exit_code').read_text().strip() != '1': raise ValueError('original failed process is not stopped')
    profile=ModelProfile.from_dict(json.loads(Path(locations['profile_path']).read_text()))
    worker=json.loads(Path(locations['worker_path']).read_text())
    descriptor=json.loads((protected/'descriptor.json').read_text())
    origin=json.loads((protected/'origin.json').read_text())
    authority=FrozenTrainingCandidateFeedbackAuthority(config=origin['config'],descriptor=descriptor,python=python)
    feedback,_=authority.read();inputs=authority.probe_inputs()
    goal=GoalSpec.from_json((protected/'goal.json').read_text()); campaign=output/'campaigns'/goal.campaign_id
    contract=json.loads((protected/'resource-contract.json').read_text())
    roots=json.loads((protected/'resource-roots.json').read_text())
    resource=TrainingSupervisionOperator(allowed_write_root=cfg/'resource-admission',contract=contract,roots=roots)
    old_identity=json.loads((cfg/'resident-identity.json').read_text())
    dataset=old_identity['authorized_dataset_roots']
    if len(dataset)!=1:raise ValueError('original boundary store mapping changed')
    store=ArtifactStore({**inputs['store_roots'],'boundary_probe':dataset[0]})
    pg=ModelGateway(profiles=(profile,),policy=ModelPolicy({'agent.runtime_rsi':profile.profile_id},2),receipt_root=output/'planner_model_receipts',artifact_store=store)
    pg.factories[profile.profile_id]=lambda:CLIPlannerEndpointProvider(profile,worker,store,output/'native')
    backend=RecoveringPlannerBackend(pg,model_id=profile.model,timeout_seconds=900)
    sam=next(ModelProfile.from_dict(x) for x in contract['models']['profiles'] if x['profile_id']=='segmentation')
    sg=ModelGateway(profiles=(sam,),policy=ModelPolicy({'training.segment':'segmentation'},1),receipt_root=output/'segment_model_receipts',artifact_store=store)
    sg.factories['segmentation']=lambda:TrainingMediaProvider(sam,contract['models']['workers']['segmentation'],store,'boundary_probe',output/'native-sam')
    bootstrap=TrainingCandidateFeedbackOperator(authority=authority,output_root=output)
    probe=BoundaryEvidenceProbeOperator(authority=authority,segment_gateway=sg,review_backend=backend,resource_operator=resource,output_root=output)
    domain=BoundaryProbeDomain(authority=authority,observation_operator_id=probe.spec.operator_id,admission=json.loads((protected/'admission.json').read_text()))
    planner=RuntimeRSIPlanner(backend=backend,domain=domain,trace_root=campaign/'planner')
    registry=OperatorRegistry();registry.register(bootstrap);registry.register(probe)
    controller=EvolutionController.open(repo_root=cfg/'source',campaigns_root=output/'campaigns',goal=goal,registry=registry,planner=planner,
        evaluator_contract=EvaluatorContract.from_path(cfg/'evaluator.json'),schema_paths=tuple(protected.rglob('*.json'))+(cfg/'resident-identity.json',),
        execution_paths=tuple((cfg/'source/src/dataevolver').rglob('*.py'))+(cfg/'host-launcher.py',))
    state=controller.status()
    if not controller._frozen()[0] or controller.replay().to_dict()!=state.to_dict():raise ValueError('original frozen/replay mismatch')
    if state.state.value!='PROPOSE' or state.planner_calls!=1 or state.operator_calls!=1 or state.experiments_used:
        raise ValueError('original pending runtime observation phase changed')
    call=OperatorCall.from_dict(state.data['pending_call']);tracepath=campaign/'planner/p000/trace.json';trace=json.loads(tracepath.read_text())
    decision=state.data['rsi_pending_decision'];receipt=ModelReceipt.from_dict(trace['model_call']['receipt'])
    if (trace['outcome']!='VALID_DECISION' or trace['proposal_actor']!='runtime_agent' or trace['decision']!=decision
            or receipt.output!=decision or receipt.without_reuse_hash()!=trace['model_call']['receipt_hash']):
        raise ValueError('original runtime decision lost its actual receipt')
    saved=ModelReceipt.from_dict(json.loads((pg.root/(receipt.request_id+'.receipt.json')).read_text())['receipt'])
    if saved.without_reuse_hash()!=receipt.without_reuse_hash():raise ValueError('runtime receipt journal changed')
    if (not controller.store.events.read() or len([e for e in controller.store.events.read() if e.event_type=='ACTION_PLANNED' and e.payload.get('call_id')==call.call_id])!=1
            or any(e.event_type=='ACTION_COMMITTED' and e.payload.get('call_id')==call.call_id for e in controller.store.events.read())):
        raise ValueError('failed observation call is not one original uncommitted action')
    request=probe._segment_request(call);prepared=sg._prepare(request)
    request_hash=prepared[4]
    native=output/'native-sam';rid=request.request_id
    job=json.loads((native/(rid+'.job.json')).read_text());failure=json.loads((native/(rid+'.native.failure.json')).read_text())
    intent=json.loads((sg.root/(rid+'.intent.json')).read_text())
    if job['request']!=request.to_dict() or job['request_hash']!=intent['request_hash']:
        raise ValueError('failed native job differs from the exact frozen request')
    if request_hash!=intent['request_hash']:
        raise ValueError('frozen Gateway request hash changed')
    su,pu=sg.audit_usage(),pg.audit_usage()
    if su['model_calls']!=1 or pu['model_calls']!=1 or any(x['unknown_duration_attempts'] for x in (su,pu)):
        raise ValueError('additional or unknown failed probe reservations')
    workerpath=cfg/'source/src/dataevolver/adapters/training_media_worker.py'
    fn=next(n for n in ast.parse(workerpath.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_sam_instances')
    reads=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='components' and isinstance(n.slice,ast.Constant) and n.slice.value=='loading_info']
    forwards=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model']
    bootstrap_ast=ast.parse(Path(locations['bootstrap_path']).read_text());keys=[]
    for n in ast.walk(bootstrap_ast):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='models' and isinstance(t.slice,ast.Constant) and t.slice.value=='sam3_instances' for t in n.targets):
            if not isinstance(n.value,ast.Dict):raise ValueError('original bootstrap differs from the failed component mapping')
            keys.append([k.value for k in n.value.keys])
    preforward=len(reads)==1 and bool(forwards) and max(reads)<min(forwards) and len(keys)==1 and 'loading_info' not in keys[0]
    native_names={f.name for f in native.iterdir() if f.is_file()}
    expected_names={rid+suffix for suffix in ('.job.json','.native.failure.json','.native.resident.intent.json')}
    if native_names!=expected_names or list((output/'native').rglob('*/final-response.json'))!=[output/'native'/receipt.request_id/'final-response.json']:
        raise ValueError('failed stage includes unexpected partial or completed native effects')
    usage={'model_calls':2,'planner_calls':1,'operator_calls':2,'wall_seconds':math.ceil(su['observed_wall_seconds']+pu['observed_wall_seconds']),
        'output_bytes':sum(f.stat().st_size for f in output.rglob('*') if f.is_file()),'image_experiments':0}
    prior=json.loads((protected/'admission.json').read_text())['budget']['prior_usage']
    return {'decision':decision,'pending_call':call.to_dict(),'request':request.to_dict(),'native_job':job,'native_failure':failure,'gateway_intent':intent,
        'old_resident_pid':old_identity['pid'],'successful_output_present':(sg.root/(rid+'.receipt.json')).exists(),
        'new_dataset_files':[str(f) for f in Path(dataset[0]).rglob('*') if f.is_file()],
        'sam_forward_count':0 if preforward else None,'visual_review_calls':pu['model_calls']-1,'unknown_duration_attempts':su['unknown_duration_attempts']+pu['unknown_duration_attempts'],
        'missing_component_before_forward_verified':preforward,'frozen_contract_valid':True,'replay_matches_snapshot':True,
        'trace_path':str(tracepath),'trace_sha256':file_sha256(tracepath),'feedback_origin':origin,'feedback_sha256':payload_sha256(feedback),
        'resource_contract':contract,'original_deadline_epoch':json.loads((protected/'admission.json').read_text())['resource_deadline_epoch'],
        'lifecycle_usage':usage,'lifecycle_total':{k:prior[k]+usage[k] for k in prior},'sam_usage':su,'planner_usage':pu}


class BoundaryProbeRecoveryDomain:
    def __init__(self, *, feedback_authority, failure_authority, observation_operator_id, admission):
        from .operator_development import DevelopmentBudget
        self.authority=feedback_authority;self.failure=failure_authority;self.binding=feedback_authority.identity()
        self.inputs=feedback_authority.probe_inputs();self.operator_id=observation_operator_id
        self.admission=copy.deepcopy(admission);old=failure_authority.read()
        if payload_sha256(feedback_authority.read()[0])!=old['feedback_sha256']:raise ValueError('recovery changed original feedback')
        b=DevelopmentBudget.from_dict(admission['budget']);inc={'model_calls':2,'planner_calls':0,'operator_calls':2,'wall_seconds':3600,'output_bytes':67108864,'image_experiments':0}
        if (admission.get('schema_version')!='dataevolver.boundary_probe_recovery_admission.v1'
                or admission.get('failed_authority_sha256')!=payload_sha256(failure_authority.identity())
                or admission.get('new_planner_decisions')!=0 or admission.get('max_segmentation_calls')!=1 or admission.get('max_visual_review_calls')!=1
                or b.resource_deadline_epoch!=old['original_deadline_epoch']
                or any(b.prior_usage[k]<old['lifecycle_total'][k] or b.limits[k]!=b.prior_usage[k]+inc[k] for k in inc)):
            raise ValueError('recovery must retain failed costs and separately admit one SAM retry without a new Planner decision')

    def config(self):
        return {'domain':'registered-boundary-known-failure-recovery-v1','authority':self.binding,'failure':self.failure.identity(),
                'operator_id':self.operator_id,'admission':self.admission,'implementation_sha256':file_sha256(Path(__file__))}

    def recover_preflight_decision(self,state):
        if state.data.get('rsi_observations') or state.data.get('preflight_proposal_recovered'):return None
        old=self.failure.read()
        return old['decision'],old['trace_path'],{'planner_calls':1}

    def context(self,state):
        from .boundary_evidence import BoundaryProbeDomain
        return BoundaryProbeDomain.context(self,state)

    def _media(self):
        from .boundary_evidence import BoundaryProbeDomain
        return BoundaryProbeDomain._media(self)

    def compile_observation(self,decision,state):
        old=self.failure.read()
        if decision!=old['decision']:raise ValueError('recovery cannot edit the original runtime decision')
        from .boundary_evidence import compile_boundary_query
        compile_boundary_query(old['pending_call']['parameters'],self.inputs)
        return self.operator_id,copy.deepcopy(old['pending_call']['parameters'])

    def compile_intervention(self,*args):raise ValueError('boundary recovery grants no image or editing authority')
