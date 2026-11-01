"""Portable fixture media exercise; these are not real training samples."""
import copy
import json
from pathlib import Path

import pytest

from test_training_pipeline import pipeline
from dataevolver.dataset.contracts import file_sha256
from dataevolver.harness.training_workflow_delivery import verify_trial_bundle,assemble_collection


def accepted_fixture(pipeline):
    operations,task,_,tmp=pipeline
    row=operations.run_family(task,0)
    operations.export([row],tmp/'bundle',[copy.deepcopy(operations.last_outputs)])
    review=row['quality'] if 'quality' in row else json.loads((tmp/'data/families/fixture-slot-a0/review-evidence.json').read_text())['review']
    return {'qualified':True,'runtime_visual_actor':'runtime_agent','model_review':review,
        'runtime_visual_review':review,'source':row['target']['image'],'candidate':row['input']['background'],
        'bundle':str(tmp/'bundle'),'model_usage':operations.gateway.audit_usage()},tmp,operations


def test_relocated_full_media_and_isolated_loader_without_model_calls(pipeline):
    trial,tmp,operations=accepted_fixture(pipeline)
    before=operations.gateway.audit_usage()
    proof=verify_trial_bundle(trial,tmp/'proof',receipt_roots=[tmp/'gateway'],allow_fixture=True)
    assert proof['accepted'] and proof['decoded_images']
    delivery=assemble_collection([proof],tmp/'collection',{'pilot':1},receipt_roots=[tmp/'gateway'],allow_fixture=True)
    assert delivery['loaded_independently'] and delivery['new_model_requests']==0
    assert operations.gateway.audit_usage()==before
    assert verify_trial_bundle(trial,tmp/'proof',allow_fixture=True)['manifest_sha256']==proof['manifest_sha256']
    assert assemble_collection([proof],tmp/'collection',{'pilot':1},allow_fixture=True)==delivery


def test_duplicate_sources_and_wrong_split_are_rejected(pipeline):
    trial,tmp,_=accepted_fixture(pipeline)
    proof=verify_trial_bundle(trial,tmp/'proof',allow_fixture=True)
    with pytest.raises(ValueError,match='split counts'):
        assemble_collection([proof],tmp/'wrong',{'train':1},allow_fixture=True)
    with pytest.raises(ValueError,match='duplicate source'):
        assemble_collection([proof,proof],tmp/'duplicate',{'pilot':2},allow_fixture=True)


def test_accepted_media_tampering_cannot_be_hidden_by_previous_proof(pipeline):
    trial,tmp,_=accepted_fixture(pipeline)
    proof=verify_trial_bundle(trial,tmp/'proof',allow_fixture=True)
    copied=tmp/'proof/relocated-bundle'
    image=next(copied.rglob('*.png'));image.write_bytes(b'fixture corruption')
    with pytest.raises(ValueError):verify_trial_bundle(trial,tmp/'proof',allow_fixture=True)
    with pytest.raises(ValueError):assemble_collection([proof],tmp/'collection',{'pilot':1},allow_fixture=True)


def test_registered_workflow_dispatches_fresh_source_and_recovers_same_effect(pipeline,monkeypatch):
    import shutil
    from types import SimpleNamespace
    from dataevolver.harness.training_workflow import TrainingWorkflowDomain,WorkflowOperator,SCHEMA
    from dataevolver.harness.models import OperatorCall
    from dataevolver.harness.adapters.training_rsi import TRIAL_ID
    from dataevolver.harness.adapters.training_supervision import TrainingSupervisionOperator
    from dataevolver.harness.state_store import atomic_write_json
    operations,task,_,tmp=pipeline
    monkeypatch.setattr(TrainingSupervisionOperator,'_wait',lambda self,call,role:None)
    task['split']='train'
    contract=copy.deepcopy(operations.config)
    contract.update(stores={'data':'FIXTURE_DATA','bundle':'FIXTURE_BUNDLE'},dataset_store='data')
    contract['models']={'profiles':[p.to_dict() for p in operations.gateway.profiles.values()],
        'policy':operations.gateway.policy.to_dict(),'workers':{}}
    root=tmp/'campaign';stage={'id':'fixture-stage','kind':'main','task':task}
    value={'schema_version':SCHEMA,'stages':[stage],'production_contract':contract,
        'roots':{'data':str(tmp/'data'),'bundle':str(tmp/'bundle')},
        'dataset_root':str(tmp/'workflow-data'),'bundle_root':str(tmp/'workflow-bundle'),
        'evidence_root':str(tmp/'evidence'),'input_hashes':{},'limits':{'max_experiments':2}}
    class Backend:
        def set_artifacts(self,artifacts):self.artifacts=artifacts
        def complete_json(self,messages,*,schema):
            if 'coordinate_system' in schema['properties']:
                return json.dumps({'status':'pass','reason':'fixture observation','coordinate_system':'normalized_0_1',
                    'shadows':[],'no_shadow_subject_ids':['s1','s2']})
            return json.dumps(operations.gateway.profiles['review'].mock_output)
        def last_trace(self):return {'fixture':True}
    backend=Backend();backend.gateway=SimpleNamespace(store=None)
    domain=TrainingWorkflowDomain(value,root,backend=backend)
    domain._state=lambda:SimpleNamespace(data={})
    target=tmp/'workflow-data/fixture-stage/baseline'
    shutil.copytree(tmp/'data',target)
    call=OperatorCall('fixture-workflow','fixture-initial','fixture-initial',TRIAL_ID,'1',0,
        str(root/'rounds/r000/execute'),{'workflow_stage':'fixture-stage','scope':'baseline','patch':{},'experience_id':''})
    operator=WorkflowOperator(domain);receipt=operator.execute(call)
    assert receipt.stage_result['status']=='SUCCESS'
    assert operator.verify_effect(call,receipt).status.value=='VERIFIED'
    # Domain has now switched to its committed source anchor. Recovery still uses
    # the exact initial compiled-domain, never a different regenerate request.
    assert domain.anchor(stage).exists()
    before={str(p):file_sha256(p) for p in root.rglob('*.intent.json')}
    assert operator.execute(call).reused
    assert {str(p):file_sha256(p) for p in root.rglob('*.intent.json')}==before
    from dataevolver.dataset.contracts import payload_sha256
    from dataevolver.harness.adapters.training_rsi import OBSERVE_ID
    trial_path=root/'rounds/r000/execute/trial.json'
    probe=call.with_updates(call_id='fixture-observation',idempotency_key='fixture-observation',operator_id=OBSERVE_ID,
        work_dir=str(root/'observations/fixture'),parameters={'workflow_stage':'fixture-stage',
        'tool':'inspect_with_planner','parameters':{'question':'Inspect these actual fixture media.'},
        'trial_path':str(trial_path),'trial_sha256':payload_sha256(json.loads(trial_path.read_text()))})
    receipt=WorkflowOperator(domain,observation=True).execute(probe)
    assert receipt.stage_result['status']=='SUCCESS' and len(backend.artifacts)>=3
    assert json.loads((Path(probe.work_dir)/'observation.json').read_text())['tool']=='inspect_with_planner'


def test_terminal_snapshot_is_attached_after_controller_complete_and_replay(tmp_path):
    from types import SimpleNamespace
    from dataevolver.harness.training_workflow_delivery import attach_terminal_proof
    from dataevolver.harness.state_store import atomic_write_json
    root=tmp_path/'delivery';root.mkdir();config=tmp_path/'config';config.mkdir()
    atomic_write_json(root/'collection.json',{'records':30,'split_counts':{'train':18,'validation':6,'test':6},
        'fixture_evidence_only':True})
    (root/'load_samples.py').write_text('from pathlib import Path\nassert (Path(__file__).parent/"terminal-harness-proof.json").exists()\n')
    for name in ['goal.json','evaluator.json','source.tar']:(config/name).write_bytes(b'fixture only')
    paths={}
    for name in ['state','events','frozen_contract']:
        p=tmp_path/(name+'.json');p.write_text('{"fixture_only":true}')
        paths[name+'_path']=p
    state=SimpleNamespace(state=SimpleNamespace(value='COMPLETE'),data={'accepted_assets':[f'fixture-{i}' for i in range(30)]},
                          to_dict=lambda:{'fixture':True})
    controller=SimpleNamespace(status=lambda:state,replay=lambda:state,_frozen=lambda:(True,()),
        layout=SimpleNamespace(**paths),planner=SimpleNamespace(domain=SimpleNamespace(value={'delivery_root':str(root)})))
    proof=attach_terminal_proof(config,controller)
    assert proof['replay_matches_snapshot'] and proof['new_model_requests']==0
    assert attach_terminal_proof(config,controller)==proof
    paths['state_path'].write_text('{"tampered_fixture":true}')
    with pytest.raises(ValueError,match='terminal proof input changed'):attach_terminal_proof(config,controller)
    state.state.value='ABSTAIN'
    with pytest.raises(ValueError,match='actual frozen Controller completion'):attach_terminal_proof(config,controller)
