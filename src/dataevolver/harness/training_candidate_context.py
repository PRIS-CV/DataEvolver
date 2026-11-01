"""Compile registered training evidence for a confined authority candidate.

No task answer is inferred here. Immutable masks, registered polygons and the
original transform supply every pixel. Outer boundary envelopes remain coverage
requirements, never removal authority. Unsupported observations remain blocked.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from .candidate_trust import ProductionCandidateTrustContext, canonical_bytes, canonical_digest
from .training_trial_authority import FrozenTrainingTrialAuthority


class FrozenTrainingObservationAuthority(FrozenTrainingTrialAuthority):
    """Reopen a committed observation using its own original frozen verifier."""
    def _current_identity(self):
        return {**super()._current_identity(), 'observation_reader_sha256':file_sha256(Path(__file__))}

    def read(self):
        import inspect
        identity=self.identity()
        before={str(p):file_sha256(p) for p in self.config.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
        shell=('set -a; source "$1"; export PYTHONPATH="$2" CUDA_VISIBLE_DEVICES= '
               'PYTHONDONTWRITEBYTECODE=1; cd "$6/source"; exec "$3" -B -c "$4" "$5" "$6" "$7"')
        script=('import json,sys\nfrom pathlib import Path\n'+inspect.getsource(_read_observation)
                +'\n__file__=sys.argv[1]\nprint(json.dumps(_read_observation(sys.argv[2],sys.argv[3])))')
        result=subprocess.run(['bash','-ec',shell,'read-frozen-observation',str(self.config/'env.sh'),
            str(self.config/'source/src'),str(self.python),script,str(Path(__file__).resolve()),str(self.config),self.call_id],
            capture_output=True,text=True,timeout=180,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','GIT_OPTIONAL_LOCKS':'0'})
        after={str(p):file_sha256(p) for p in self.config.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts}
        if before!=after:raise ValueError('observation verification modified historical evidence')
        if result.returncode:raise ValueError('registered observation verification failed: '+result.stderr[-2000:])
        value=json.loads(result.stdout)
        if (value['call']['call_id']!=self.call_id or value['config']!=str(self.config)
                or value['source_commit']!=identity['source_commit'] or not value['frozen'] or not value['replay']):
            raise ValueError('registered observation origin changed')
        self.identity()
        return value


def _read_observation(config,call_id):
    # All imports here resolve from the original frozen framework. The audit
    # helper is host code loaded by file, not from any candidate-owned root.
    import importlib.util
    from dataevolver.dataset.contracts import file_sha256,payload_sha256
    from dataevolver.harness.composition import open_controller
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.adapters.training_rsi import TrainingRSIObservationOperator
    from dataevolver.harness.models import OperatorCall,OperatorReceipt,EvidenceStatus
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.adapters.model_contracts import ModelRequest
    spec=importlib.util.spec_from_file_location('read_only_training_guard',Path(__file__).with_name('training_trial_authority.py'))
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    def forbidden(*a,**kw):raise RuntimeError('observation reader cannot submit or recover effects')
    EvolutionController.run=EvolutionController.tick=forbidden
    TrainingRSIObservationOperator.execute=TrainingRSIObservationOperator.recover=forbidden
    ModelGateway.invoke=ModelGateway._provider=ModelGateway.adopt_completed=forbidden
    def existing(gateway,request):
        if not (gateway.root/(request.request_id+'.receipt.json')).is_file():
            raise ValueError('registered observation has no committed Gateway receipt')
        return gateway._recover(request,gateway._prepare(request))
    ModelGateway.recover=existing
    sys.addaudithook(guard._read_only_event)
    cfg=Path(config).resolve();frozen=json.loads((cfg/'frozen.json').read_text())
    if file_sha256(cfg/'goal.json')!=frozen['goal_sha256'] or file_sha256(cfg/'source.tar')!=frozen['source_archive_sha256']:
        raise ValueError('registered observation frozen source changed')
    for name,digest in frozen['source_hashes'].items():
        if file_sha256(cfg/'source'/name)!=digest:raise ValueError('original observation source changed')
    c=open_controller(cfg/'campaigns',json.loads((cfg/'goal.json').read_text())['campaign_id'])
    if (c.status().state.value not in {'ABSTAIN','COMPLETE'} or not c._frozen()[0]
            or c.replay().to_dict()!=c.status().to_dict()):
        raise ValueError('registered observation Controller is not terminal and replayable')
    events=c.store.events.read()
    planned=[e for e in events if e.event_type=='ACTION_PLANNED' and e.payload['call_id']==call_id]
    committed=[e for e in events if e.event_type=='ACTION_COMMITTED' and e.payload['call_id']==call_id]
    if len(planned)!=1 or len(committed)!=1:raise ValueError('one committed registered observation is required')
    call=OperatorCall.from_dict(planned[0].payload['operator_call'])
    receipt=OperatorReceipt.from_dict(committed[0].payload['receipt']);op=c.registry.get(call.operator_id)
    if type(op) is not TrainingRSIObservationOperator:raise ValueError('unregistered observation authority')
    verified=op.verify_effect(call,receipt)
    if verified.status is not EvidenceStatus.VERIFIED:raise ValueError(verified.reason)
    path=Path(call.work_dir)/'observation.json';observation=json.loads(path.read_text())
    if observation['tool']!='inspect_surfaces' or observation['execution_actor']!='registered_operator':
        raise ValueError('fresh registered surface observation required')
    trial_path=Path(call.parameters['trial_path']);trial=json.loads(trial_path.read_text())
    if payload_sha256(trial)!=call.parameters['trial_sha256']:raise ValueError('observation target trial changed')
    gateway=op.domain.backend.gateway;gateway.store=op.domain.scope(trial['scope'])
    trace=observation['model_call'];request=ModelRequest.from_dict(trace['request'])
    cached=gateway.recover(request)
    if cached is None or cached.without_reuse_hash()!=trace['receipt_hash'] or cached.output!=observation['output']:
        raise ValueError('registered observation lost its exact model output/receipt')
    paths={path,trial_path,Path(call.work_dir)/'runtime-effect.json'}
    paths.update(Path(p) for p in verified.artifact_hashes)
    paths.update(p for p in gateway.root.rglob('*') if p.is_file())
    return {'config':str(cfg),'source_commit':frozen['source_commit'],'call':call.to_dict(),
        'frozen':True,'replay':True,'observation':observation,'observation_path':str(path),
        'observation_sha256':file_sha256(path),'observation_payload_sha256':payload_sha256(observation),
        'receipt_sha256':receipt.sha256(),
        'artifact_hashes':{str(p):file_sha256(p) for p in paths},'new_model_calls':0}


def verify_requested_observation(request,origin,registered):
    """Original capability citations use canonical payload, not file hashes.

    The original committed effect separately authenticates exact file bytes,
    including their trailing newline. Neither digest is substituted for the
    other. Both remain in the registered origin snapshot.
    """
    trial=origin['trial']
    payload_hash=payload_sha256(registered['observation'])
    if (registered.get('observation_payload_sha256')!=payload_hash
            or request['source']!=trial['source'] or request['rejected_candidate']!=trial['candidate']
            or {'path':registered['observation_path'],'sha256':payload_hash} not in request['evidence']):
        raise ValueError('production evidence was not cited by the original registered request')


def compile_training_evidence(origin, registered, store):
    """Lossless authority projection on the finite support decision universe.

    Protected pixels outside all required/prior support cannot affect any set
    decision. Surface pixels outside their declared effects likewise cannot.
    Counts and original hashes retain these projections explicitly. No mask is
    downsampled, no polygon envelope is granted authority, no status is relabeled.
    """
    import numpy as np
    from PIL import Image,ImageDraw
    from dataevolver.dataset.surface_composition import validate_coverage_observation
    from dataevolver.dataset.training_pipeline import QUALITY_CHECKS
    trial=origin['trial'];obs=registered['observation'];value=obs['output']
    for item in trial['media']:store.resolve(ArtifactRef.from_dict(item['artifact']))
    if trial['source']!=trial['media'][0]['artifact'] or trial['candidate']!=trial['media'][1]['artifact']:
        raise ValueError('training media and source/candidate bindings differ')
    if obs['source']!=trial['source'] or obs['candidate']!=trial['candidate'] or obs['existing_authorized_support']!=trial['media'][3]['artifact']:
        raise ValueError('registered observation is stale for these actual training artifacts')
    refs={item['label']:item['artifact'] for item in trial['media']}
    ids=[label.removeprefix('instance mask ') for label in refs if label.startswith('instance mask ')]
    validate_coverage_observation(value,ids,structural_only=True)
    def load(ref,mode):
        with Image.open(store.resolve(ArtifactRef.from_dict(ref))) as im:return im.convert(mode)
    size=load(trial['source'],'RGB').size
    if size[0]*size[1]>16777216:raise ValueError('source raster exceeds production context bound')
    def mask(ref,*,binary=False):
        im=load(ref,'L')
        if im.size!=size:raise ValueError('immutable mask or support dimensions changed')
        values=np.asarray(im)
        if binary and not np.isin(values,[0,255]).all():
            raise ValueError('existing authority must be binary; display alpha is not permission')
        return values>127
    def polygon(points):
        im=Image.new('L',size);ImageDraw.Draw(im).polygon(
            [(round(x*(size[0]-1)),round(y*(size[1]-1))) for x,y in points],fill=255)
        return np.asarray(im)>0
    def points(a):return [[int(x),int(y)] for y,x in np.argwhere(a)]
    old=mask(trial['media'][3]['artifact'],binary=True)
    if trial['media'][3]['artifact']!=trial['media'][4]['artifact']:
        if not np.array_equal(old,mask(trial['media'][4]['artifact'],binary=True)):
            raise ValueError('original conditioning and composition support differ')
    zero=np.zeros((size[1],size[0]),dtype=bool)
    masks={sid:mask(refs['instance mask '+sid]) for sid in ids}
    boundaries={sid:m.copy() for sid,m in masks.items()}
    for row in value['removal_boundaries']:boundaries[row['subject_id']]|=polygon(row['polygon'])
    effect_rows=[(row,polygon(row['polygon'])) for row in value['effects']]
    universe=old.copy()
    for m in list(boundaries.values())+[m for _,m in effect_rows]:universe|=m
    protected=zero.copy()
    for row in value['preservation_regions']:protected|=polygon(row['polygon'])
    full_protected=int(protected.sum());protected &= universe
    surface_masks={row['surface_id']:polygon(row['polygon']) for row in value['surfaces']}
    relevant_surfaces={sid:zero.copy() for sid in surface_masks}
    for row,m in effect_rows:relevant_surfaces[row['surface_id']] |= m & surface_masks[row['surface_id']]
    subjects={row['subject_id']:row for row in value['subjects']}
    instances=[];observations=[];uncertainty=[]
    for sid in ids:
        immutable=points(masks[sid]);reference=refs['reference '+sid]
        inst={'id':sid,'mask':immutable,'mask_sha256':canonical_digest(immutable),
            'reference_sha256':reference['sha256'],'required_boundary':points(boundaries[sid]),'effects':[]}
        # The witness may confirm the existing silhouette only. In particular,
        # holes/apertures enclosed by an outer polygon receive no new authority.
        witness=masks[sid] if subjects[sid]['boundary_assessment']=='observed' else zero
        observations.append({'id':'boundary-'+sid,'kind':'boundary_witness','subject':sid,
            'mask_sha256':inst['mask_sha256'],'reference_sha256':reference['sha256'],
            'pixels':points(witness),'effect':None,'surface':None,'basis':'registered_removal_boundary'})
        if subjects[sid]['boundary_assessment']=='uncertain':
            uncertainty.append({'kind':'boundary','pixels':points(boundaries[sid]),'subject':sid,'required':True})
        if subjects[sid]['effect_assessment']=='uncertain':
            uncertainty.append({'kind':'attribution','pixels':[],'subject':sid,'required':True})
        for index,(row,m) in enumerate(effect_rows):
            if row['subject_id']!=sid:continue
            eid='effect-'+str(index);surface=row['surface_id']
            inst['effects'].append({'id':eid,'surface':surface,'pixels':points(m)})
            observations.append({'id':eid,'kind':'attributed_effect','subject':sid,
                'mask_sha256':inst['mask_sha256'],'reference_sha256':reference['sha256'],
                'pixels':points(m & surface_masks[surface]),'effect':eid,'surface':surface,
                'basis':'registered_separate_effect_attribution'})
        instances.append(inst)
    for row in value['uncertain_regions']:
        m=polygon(row['polygon']) & universe
        for sid in row['subject_ids']:
            if row['kind']=='effect_attribution':
                uncertainty.append({'kind':'attribution','pixels':points(m),'subject':sid,'required':True})
            else:
                # Hidden texture uncertainty inside prior support is not visible
                # preservation. Outside support it can never grant authority.
                uncertainty.append({'kind':'reconstruction','pixels':points(m & ~old),'subject':sid,'required':False})
    # The exact previously selected crop is retained; no new padding is chosen.
    crop=trial['patch']['parameters']['preflight_receipt']['crop']['source_crop_xyxy']
    if (not isinstance(crop,list) or len(crop)!=4 or any(type(x) is not int for x in crop)
            or not 0<=crop[0]<crop[2]<=size[0] or not 0<=crop[1]<crop[3]<=size[1]):
        raise ValueError('original source crop is not an exact bounded pixel transform')
    transform={'kind':'integer_translation_crop','crop':[crop[0],crop[1],crop[2]-crop[0],crop[3]-crop[1]],
               'inverse_projection':'add_crop_origin'}
    models=[{'id':name,'sha256':payload_sha256(worker)} for name,worker in sorted(origin['contract']['models']['workers'].items())]
    tasks=[x for x in origin['contract']['production']['tasks'] if trial['source_family'].startswith(x['slot_id']+'-')]
    if len(tasks)!=1 or set(tasks[0]['selected_subject_ids'])!=set(ids):
        raise ValueError('immutable selected masks do not match the original selected task')
    task=tasks[0]
    frozen={'models':models,'seed':task['seed'],'generation_settings':{
            'recipe':origin['contract']['production']['recipe'],'task':task,
            'actual_gateway_requests_sha256':payload_sha256(origin['family_records'])},
        'quality_thresholds':{name:'pass' for name in QUALITY_CHECKS},'required_model_checks':list(QUALITY_CHECKS),
        'independent_visual_audit_required':True}
    evidence={'instances':instances,'observations':observations,
        'prior_support':{'conditioning':points(old),'composition':points(old),'transform_sha256':canonical_digest(transform)},
        'protected':{'pixels':points(protected)},'uncertainty':uncertainty,
        'receiving_surfaces':[{'id':sid,'pixels':points(m)} for sid,m in relevant_surfaces.items()]}
    assets={name:trial['media'][index]['artifact']['sha256'] for name,index in {
        'original':0,'rejected_candidate':1,'projected_edit':2,'conditioning_image':5,'composition_support':4}.items()}
    manifest={'assets':assets,'width':size[0],'height':size[1],'frozen':frozen,'transform':transform,
        **{name+'_sha256':canonical_digest(evidence[name]) for name in ('instances','prior_support','protected','receiving_surfaces')}}
    for row in observations:row['manifest_sha256']=canonical_digest(manifest)
    projection={'registered_status':value['status'],'source_size':list(size),
        'decision_universe_pixels':int(universe.sum()),'full_protected_pixels':full_protected,
        'relevant_protected_pixels':int(protected.sum()),'boundary_envelope_added_authority_pixels':0,
        'original_observation_output_sha256':payload_sha256(value),
        'transform_model_resampling_retained_separately':trial['patch']['parameters']['preflight_receipt']['crop'],
        'image_submission_allowed':False,'candidate_adoption_allowed':False}
    return evidence,manifest,projection


def sign_production_context(evidence,manifest,*,context_id,package,request_sha256,input_artifact,
                            family_id,origin_bindings,evidence_artifacts):
    """Sign only already compiled host evidence; private key never leaves host."""
    from cryptography.hazmat.primitives.asymmetric import rsa,padding
    from cryptography.hazmat.primitives import hashes
    private=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    registrar='registered-training-evidence'
    evidence={**evidence,'source':{'registrar':registrar,'manifest':manifest}}
    signature=private.sign(canonical_bytes(evidence),padding.PKCS1v15(),hashes.SHA256())
    evidence['source']['signature']=base64.b64encode(signature).decode('ascii')
    payload={'evidence':evidence,'parameters':{}}
    from .models import canonical_json
    input_artifact={**input_artifact,'sha256':hashlib.sha256((canonical_json(payload)+'\n').encode()).hexdigest()}
    public=private.public_key().public_numbers()
    context=ProductionCandidateTrustContext(context_id,package.compute_hash(),package.implementation['sha256'],
        request_sha256,input_artifact,canonical_digest(payload),family_id,
        {registrar:{'n_hex':format(public.n,'x'),'e':public.e,'manifest_sha256':canonical_digest(manifest)}},
        {**origin_bindings,'manifest_sha256':canonical_digest(manifest)},tuple(evidence_artifacts))
    return payload,context


if __name__=='__main__':
    if len(sys.argv)!=4 or sys.argv[1]!='--read-observation':raise SystemExit(2)
    print(json.dumps(_read_observation(sys.argv[2],sys.argv[3])))
