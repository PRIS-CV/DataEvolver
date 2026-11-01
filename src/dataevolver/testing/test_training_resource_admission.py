"""Exercise the original resource wait, including available-after-expiry."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dataevolver.adapters.model_contracts import ModelPolicy
from dataevolver.adapters.model_gateway import ModelError
from dataevolver.dataset.contracts import ArtifactStore
from dataevolver.harness.adapters import training_supervision as training
from dataevolver.harness.models import EvidenceStatus, OperatorCall
from dataevolver.harness.training_resource_admission import TrainingResourceAdmission


def fixture(tmp_path):
    # Skip full model-asset construction only in this engineering fixture.
    op=training.TrainingSupervisionOperator.__new__(training.TrainingSupervisionOperator)
    op.fixture=True;op.custom_wait=None
    op.store=ArtifactStore({'source':tmp_path/'inputs'})
    op.root=tmp_path/'workload'
    op.policy=ModelPolicy({'training.background':'image'},None,usage_mode='accounting')
    op.prompt_policy=ModelPolicy({'training.refine_i2i':'pe'},None,usage_mode='accounting')
    op.contract={'production':{'resource_deadline_epoch':1},
                 'models':{'workers':{'pe':{'kind':'pe_i2i'},'image':{'kind':'image_edit'}}}}
    probe=TrainingResourceAdmission(operator=op,workload_operator_id='data.production_workload',
        workload_operator_version='1',roles=('training.refine_i2i',),output_root=tmp_path/'observations',
        allowed_gpu_uuids=('GPU-fixture-only',))
    call=OperatorCall('campaign','call','call','data.production_workload','1',0,str(tmp_path/'work'))
    return op,probe,call


def test_available_expired_service_retains_original_two_observations(monkeypatch,tmp_path):
    op,probe,call=fixture(tmp_path)
    polls=[];sleeps=[]
    def ready(config):
        polls.append(config)
        return {'device':'resident_cpu','resident_pid':17,'weights_already_loaded':True}
    monkeypatch.setattr(training,'pe_observation',ready)
    monkeypatch.setattr(training.time,'sleep',sleeps.append)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','original-mapping')
    proof=probe.preflight(call,1)
    assert proof['admitted'] and len(polls)==2 and sleeps==[30]
    assert probe.verify(call,proof).status is EvidenceStatus.VERIFIED
    assert probe.preflight(call,1)==proof and len(polls)==2
    assert __import__('os').environ['CUDA_VISIBLE_DEVICES']=='original-mapping'
    value=json.loads(Path(proof['report_path']).read_text())
    assert value['new_model_calls']==value['new_image_calls']==0
    assert value['observations'][0]['status']['stable_observations']==2


def test_unavailable_after_original_deadline_refuses_without_sleep(monkeypatch,tmp_path):
    _,probe,call=fixture(tmp_path)
    polls=[]
    def busy(config):
        polls.append(config)
        raise ModelError('RESOURCE_BUSY','fixture busy')
    monkeypatch.setattr(training,'pe_observation',busy)
    monkeypatch.setattr(training.time,'sleep',lambda seconds:pytest.fail('expired unavailable service must not wait'))
    proof=probe.preflight(call,1)
    assert not proof['admitted'] and len(polls)==1
    assert 'expired' in proof['reason']
    assert probe.verify(call,proof).status is EvidenceStatus.VERIFIED


def test_deadline_role_and_owner_changes_refuse_without_observing(monkeypatch,tmp_path):
    op,probe,call=fixture(tmp_path)
    monkeypatch.setattr(training,'pe_observation',lambda c:pytest.fail('no observation allowed'))
    with pytest.raises(ValueError,match='deadline'):probe.preflight(call,2)
    with pytest.raises(ValueError,match='workload'):probe.preflight(replace(call,operator_id='other'),1)
    op.contract['models']['workers']['pe']['kind']='changed'
    with pytest.raises(ValueError,match='changed'):probe.preflight(call,1)


def test_interrupted_observation_is_not_restarted(monkeypatch,tmp_path):
    _,probe,call=fixture(tmp_path)
    polls=[]
    def failed(c):
        polls.append(c)
        raise OSError('fixture observation interrupted')
    monkeypatch.setattr(training,'pe_observation',failed)
    with pytest.raises(OSError):probe.preflight(call,1)
    with pytest.raises(ValueError,match='incomplete'):probe.preflight(call,1)
    assert len(polls)==1


def test_completed_evidence_drift_refuses_without_health(monkeypatch,tmp_path):
    _,probe,call=fixture(tmp_path)
    monkeypatch.setattr(training,'pe_observation',lambda c:{'device':'resident_cpu'})
    monkeypatch.setattr(training.time,'sleep',lambda n:None)
    proof=probe.preflight(call,1)
    file=next(Path(p) for p in proof['artifact_hashes'] if p.endswith('role-0.json'))
    file.write_text('{}')
    monkeypatch.setattr(training,'pe_observation',lambda c:pytest.fail('completed effect must not observe again'))
    assert probe.verify(call,proof).status is EvidenceStatus.CONTRADICTED
    with pytest.raises(ValueError):probe.preflight(call,1)


def test_production_cannot_override_wait_or_change_gpu(monkeypatch,tmp_path):
    op,_,_=fixture(tmp_path);op.fixture=False
    op.custom_wait=lambda role:None
    with pytest.raises(ValueError,match='replace'):
        TrainingResourceAdmission(operator=op,workload_operator_id='data.work',workload_operator_version='1',
            roles=('training.refine_i2i',),output_root=tmp_path/'other',allowed_gpu_uuids=('GPU-0',))
    op.custom_wait=None
    from dataevolver.adapters import training_prompt_enhancement as pe
    monkeypatch.setattr(pe,'pe_identity',lambda c:{'gpu_uuid':'GPU-1',
        'models':{'pe_i2i':{'device_map':{'model':'cuda:0'}}}})
    with pytest.raises(ValueError,match='GPU mapping'):
        TrainingResourceAdmission(operator=op,workload_operator_id='data.work',workload_operator_version='1',
            roles=('training.refine_i2i',),output_root=tmp_path/'other',allowed_gpu_uuids=('GPU-0',))


@pytest.mark.parametrize('path', ['inputs','inputs/nested','workload','workload/nested','.'])
def test_resource_observations_cannot_write_inputs_or_workloads(tmp_path,path):
    op,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='protected'):
        TrainingResourceAdmission(operator=op,workload_operator_id='data.work',workload_operator_version='1',
            roles=('training.refine_i2i',),output_root=tmp_path/path,allowed_gpu_uuids=('GPU-0',))
