import copy
import json
from pathlib import Path
import pytest

from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy, ModelRequest
from dataevolver.dataset.contracts import ArtifactRef, file_sha256
from dataevolver.harness.adapters.data_graph import DataGraphOperator
from dataevolver.harness.data_contracts import DataInput, DataNodeSpecV3, DataNodeGraphV3, DataRunBudget
from dataevolver.harness.models import OperatorCall, EvidenceStatus
from dataevolver.harness.operator_sdk import builtin_step_registry, declare_step
from dataevolver.harness.task_contracts import DatasetIntent, QualityPolicy, PROTECTED_GATES


def request_step(ctx):
    request = ModelRequest('batch-'+ctx.parent.call_id, 'text', ({'role':'user','content':'fixture'},), {'type':'object'})
    return {'result':ctx.write_json('response.json',ctx.invoke(request).output)}


def graph(tmp_path, *, quota=2):
    registry=builtin_step_registry()
    registry.register_step(declare_step('test.text',request_step,inputs={},outputs={'result':{'type':'object'}},
        parameters={'type':'object','properties':{},'additionalProperties':False},kind='model'))
    profile=ModelProfile('fixture','mock','fixture','1',mock_output={'ok':True})
    node=DataNodeSpecV3('text','test.text','1','model',{}, {}, 'text',1,0)
    contract={'intent':DatasetIntent('engineering','generate','catalog','json','engineering fixture').to_dict(),
        'quality_policy':QualityPolicy('engineering',tuple(sorted(PROTECTED_GATES)),('test.text',),{},4).to_dict(),
        'graph':DataNodeGraphV3((node,)).to_dict(),'budget':DataRunBudget(1,4,0,1,100000,60).to_dict(),
        'stores':{'data':'TEST_DATA'},'dataset_store':'data','models':{'profiles':[profile.to_dict()],
            'policy':ModelPolicy({'text':'fixture'},4).to_dict(),'workers':{},'ledger_prefix':'gateway',
            'attempt_limits':{'quotas':{'text':quota},'role_buckets':{'text':'text'}}}}
    op=DataGraphOperator(allowed_write_root=tmp_path/'runtime',contract=contract,roots={'data':tmp_path/'data'},registry=registry)
    def call(name):return OperatorCall('engineering',name,name,'data.graph','1',0,str(tmp_path/'runtime'/name),parameters=contract)
    return op,call


def test_shared_native_ledger_reuses_and_preserves_earlier_graph_effect(tmp_path):
    op,call=graph(tmp_path)
    first=op.execute(call('first'))
    assert first.status.value=='SUCCESS'
    second=op.execute(call('second'))
    assert second.status.value=='SUCCESS'
    assert op.gateway(call('first')).audit_usage()['model_calls']==2
    assert op.model_usage(call('first'))['model_calls']==1
    assert op.verify_effect(call('first'),first).status is EvidenceStatus.VERIFIED
    assert op.execute(call('first')).reused
    assert op.gateway(call('first')).audit_usage()['model_calls']==2


def test_shared_quota_refuses_next_batch_before_reservation(tmp_path):
    op,call=graph(tmp_path,quota=1)
    assert op.execute(call('first')).status.value=='SUCCESS'
    failed=op.execute(call('second'))
    assert failed.stage_result['failure_code']=='MODEL_BUDGET_EXHAUSTED'
    assert not (op.gateway(call('second')).root/'batch-second.intent.json').exists()


def test_persistent_family_is_available_through_standard_registry():
    definition=builtin_step_registry().step('images.persistent_families','1')
    assert definition.contract.input_schemas.keys()=={'catalog'}
    assert {'supervision.append','supervision.export_t2i'} <= {s.operator_id for s in builtin_step_registry().step_specs()}


def caption_graph(tmp_path, *, objective='Faithful captions for text-to-image training.',
                  correction_objective=None, malformed_envelope=False):
    from PIL import Image
    data = tmp_path / 'data'
    data.mkdir()
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    Image.new('RGB', (16, 16), 'red').save(inputs / 'image.png')
    image = ArtifactRef('inputs', 'image.png', file_sha256(inputs / 'image.png'), 'image/png')
    source = {'id': 'imported-example', 'image': image.to_dict(), 'prompt': 'A blue image.'}
    if malformed_envelope:
        source['request'] = {}
    (inputs / 'source.json').write_text(json.dumps(source))
    source_ref = ArtifactRef('inputs', 'source.json', file_sha256(inputs / 'source.json'))
    dims = ['caption_fidelity']
    def review(status):
        return {'status': status, 'summary': status,
                'criteria': {'caption_fidelity': {'status': status, 'reason': status}}}
    profiles = [ModelProfile(name, 'mock', name, '1', ('json', 'vision'), mock_output=output)
                for name, output in [('reject', review('fail')),
                    ('correct', {'status': 'pass', 'prompt': 'A red image.', 'reason': 'Corrected color.'})]]
    parameters = {'dimensions': dims, 'seed': 0, 'max_tokens': 128, 'timeout_seconds': 1}
    if objective is not None:
        parameters['training_objective'] = objective
    correction = {**parameters, 'training_objective': correction_objective or objective or 'Explicit objective.'}
    nodes = (
        DataNodeSpecV3('before', 'visual.review.image', '1', 'model',
            {'media': DataInput(artifact=source_ref)}, parameters, 'visual.review.image', 1, 1),
        DataNodeSpecV3('correct', 'visual.caption.refine', '1', 'model',
            {'media': DataInput(artifact=source_ref), 'review': DataInput(node='before', output='review')},
            correction, 'visual.caption.refine', 1, 1),
        DataNodeSpecV3('after', 'visual.review.image', '1', 'model',
            {'media': DataInput(node='correct', output='media')}, parameters, 'visual.review.image', 1, 1))
    contract = {'intent': DatasetIntent('caption-review', 'curate', 'catalog', 'json',
        'Review and correct imported captions.', (source_ref,)).to_dict(),
        'quality_policy': QualityPolicy('caption-review', tuple(sorted(PROTECTED_GATES)),
            ('visual.review.image', 'visual.caption.refine'), {}, 3).to_dict(),
        'graph': DataNodeGraphV3(nodes).to_dict(), 'budget': DataRunBudget(3, 3, 3, 1, 100000, 60).to_dict(),
        'stores': {'inputs': 'TEST_INPUTS', 'data': 'TEST_DATA'}, 'dataset_store': 'data',
        'models': {'profiles': [p.to_dict() for p in profiles],
                   'policy': ModelPolicy({'visual.review.image': 'reject', 'visual.caption.refine': 'correct'}, 3).to_dict(),
                   'workers': {}}}
    op = DataGraphOperator(allowed_write_root=tmp_path / 'runtime', contract=contract,
        roots={'inputs': inputs, 'data': data}, registry=builtin_step_registry())
    call = OperatorCall('caption-review', 'review', 'review', 'data.graph', '1', 0,
                       str(tmp_path / 'runtime/review'), parameters=contract)
    return op, call, source


def test_imported_caption_review_binds_goal_and_recovers_without_new_calls(tmp_path):
    from dataevolver.dataset.contracts import payload_sha256
    op, call, source = caption_graph(tmp_path)
    receipt = op.execute(call)
    assert receipt.status.value == 'SUCCESS'
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    effects = {}
    for node in ('before', 'correct', 'after'):
        value = json.loads((Path(call.work_dir) / 'node_runs' / node / 'effect.json').read_text())
        effects[node] = {key: json.loads(op.store.resolve(ArtifactRef.from_dict(ref)).read_text())
                         for key, ref in value['outputs'].items()}
    scope = effects['before']['feedback']['scope']
    assert scope['source_record_sha256'] == payload_sha256(source)
    assert scope['training_objective_sha256'] == payload_sha256('Faithful captions for text-to-image training.')
    corrected = effects['correct']['media']
    assert corrected['prompt'] == 'A red image.'
    assert corrected['image'] == source['image']
    assert effects['before']['review']['output']['status'] == 'fail'
    assert effects['after']['review']['output']['status'] == 'fail'
    assert effects['after']['feedback']['scope']['source_record_sha256'] == payload_sha256(corrected)
    assert op.gateway(call).audit_usage()['model_calls'] == 3
    assert op.execute(call).reused
    assert op.gateway(call).audit_usage()['model_calls'] == 3


def test_caption_correction_cannot_change_the_training_objective(tmp_path):
    op, call, _ = caption_graph(tmp_path, correction_objective='Prefer a different training target.')
    receipt = op.execute(call)
    assert receipt.status.value != 'SUCCESS'
    assert op.gateway(call).audit_usage()['model_calls'] == 1


def repair_graph(tmp_path, *, mismatch=None, review_status='fail'):
    op, old_call, source = caption_graph(tmp_path)
    contract = copy.deepcopy(op.contract)
    before = DataNodeGraphV3.from_dict(contract['graph']).nodes[0]
    parameters = {k: v for k, v in before.parameters.items() if k in ('training_objective', 'dimensions')}
    parameters['repair_instruction'] = 'Correct this rejected image.'
    media = before.inputs['media']
    if mismatch in {'caption', 'record', 'image'}:
        row = copy.deepcopy(source)
        if mismatch == 'caption': row['prompt'] = 'Another caption.'
        if mismatch == 'record': row['id'] = 'another-record'
        if mismatch == 'image':
            from PIL import Image
            image = tmp_path/'inputs/another.png'
            Image.new('RGB', (16, 16), 'blue').save(image)
            row['image'] = ArtifactRef('inputs', image.name, file_sha256(image), 'image/png').to_dict()
        path = tmp_path/'inputs/another.json'; path.write_text(json.dumps(row))
        media = DataInput(artifact=ArtifactRef('inputs', path.name, file_sha256(path)))
        contract['intent']['sources'].append(media.artifact.to_dict())
    if mismatch == 'objective': parameters['training_objective'] = 'Different goal.'
    if mismatch == 'dimensions': parameters['dimensions'] = ['different_dimension']
    if mismatch == 'blank_instruction': parameters['repair_instruction'] = ' '
    repair = DataNodeSpecV3('repair', 'visual.prompt.repair', '1', 'data',
        {'media': media, 'review': DataInput(node='before', output='review')}, parameters)
    contract['graph'] = DataNodeGraphV3((before, repair)).to_dict()
    contract['quality_policy'] = QualityPolicy('caption-review', tuple(sorted(PROTECTED_GATES)),
        ('visual.review.image', 'visual.prompt.repair'), {}, 3).to_dict()
    for profile in contract['models']['profiles']:
        if profile['profile_id'] == 'reject':
            profile['mock_output']['status'] = review_status
            profile['mock_output']['criteria']['caption_fidelity']['status'] = review_status
            profile['mock_output']['criteria']['caption_fidelity']['reason'] = review_status
    op = DataGraphOperator(allowed_write_root=tmp_path/'repair-runtime', contract=contract,
        roots={'inputs': tmp_path/'inputs', 'data': tmp_path/'data'}, registry=builtin_step_registry())
    call = old_call.with_updates(work_dir=str(tmp_path/'repair-runtime/review'), parameters=contract)
    return op, call, source


@pytest.mark.parametrize('status', ['fail', 'uncertain'])
def test_repair_prompt_uses_confirmed_rejection_without_another_model_call(tmp_path, status):
    from dataevolver.dataset.contracts import payload_sha256
    op, call, source = repair_graph(tmp_path, review_status=status)
    receipt = op.execute(call)
    assert receipt.status.value == 'SUCCESS'
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    outputs = json.loads((Path(call.work_dir)/'node_runs/repair/effect.json').read_text())['outputs']
    prompt = json.loads(op.store.resolve(ArtifactRef.from_dict(outputs['prompt'])).read_text())
    trace = json.loads(op.store.resolve(ArtifactRef.from_dict(outputs['repair'])).read_text())
    assert prompt == {'prompt': 'Correct this rejected image.\nOriginal unchanged goal: '
        'Faithful captions for text-to-image training.\nRecorded failed criteria: '
        + json.dumps({'caption_fidelity': status}, sort_keys=True)}
    assert trace['source_image'] == source['image']
    assert trace['source_media_sha256'] == payload_sha256(source)
    assert trace['prompt_sha256'] == payload_sha256(prompt['prompt'])
    assert trace['production_qualification'] is False
    definition = op.registry.step('visual.prompt.repair', '1')
    assert definition.kind == 'data' and definition.contract.effect_semantics == 'pure_data'
    assert op.model_usage(call)['model_calls'] == 1  # Only the fixture Review.
    before = {p: p.read_bytes() for p in op.ledger_root(call).glob('*.json')}
    assert op.execute(call).reused
    assert before == {p: p.read_bytes() for p in op.ledger_root(call).glob('*.json')}


@pytest.mark.parametrize('mismatch', ['caption', 'record', 'image', 'objective', 'dimensions', 'blank_instruction'])
def test_repair_prompt_rejects_changed_source_or_goal_before_output(tmp_path, mismatch):
    op, call, _ = repair_graph(tmp_path, mismatch=mismatch)
    assert op.execute(call).status.value != 'SUCCESS'
    assert op.model_usage(call)['model_calls'] == 1
    assert not (Path(call.work_dir)/'node_runs/repair/effect.json').exists()


def test_repair_prompt_does_not_rework_an_accepted_image(tmp_path):
    op, call, _ = repair_graph(tmp_path, review_status='pass')
    assert op.execute(call).status.value != 'SUCCESS'
    assert op.model_usage(call)['model_calls'] == 1
    assert not (Path(call.work_dir)/'node_runs/repair/effect.json').exists()


@pytest.mark.parametrize('objective,malformed', [(None, False), ('Explicit objective.', True)])
def test_imported_review_rejects_unbound_goal_or_malformed_native_envelope(tmp_path, objective, malformed):
    op, call, _ = caption_graph(tmp_path, objective=objective, malformed_envelope=malformed)
    assert op.execute(call).status.value != 'SUCCESS'
    assert op.gateway(call).audit_usage()['model_calls'] == 0


@pytest.mark.parametrize('prefix',['../outside','/outside','.','a/../../outside','a\\b'])
def test_shared_ledger_cannot_escape_registered_store(tmp_path,prefix):
    op,_=graph(tmp_path)
    contract=copy.deepcopy(op.contract);contract['models']['ledger_prefix']=prefix
    with pytest.raises(ValueError,match='ledger'):
        DataGraphOperator(allowed_write_root=tmp_path/'runtime',contract=contract,roots={'data':tmp_path/'data'},registry=op.registry)


def test_resident_execution_observes_supplied_host_without_self_rpc(monkeypatch):
    from dataevolver.adapters import training_media_worker, training_media_resident
    host={'pid':123,'state':'EXECUTING'}
    observed=[]
    def read(config,*,status=None):
        assert status is host  # A missing status would recurse through the serial socket.
        observed.append(status)
        return {'device':'resident_cuda','devices':[{'uuid':'GPU-example'}]}
    monkeypatch.setattr(training_media_resident,'resident_observation',read)
    actual=training_media_worker.deployment_observation({'resident':{'identity':'bound'}},resident_status=host)
    assert actual['devices'][0]['uuid']=='GPU-example'
    assert len(observed)==1


def test_two_linked_gateways_cannot_both_spend_last_native_unit(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading,time
    from dataevolver.harness.attempt_limits import ModelAttemptLimits
    from dataevolver.harness.state_store import LeaseBusyError
    from dataevolver.adapters.model_gateway import ModelGateway,ModelError
    roots=(tmp_path/'a',tmp_path/'b')
    limits=ModelAttemptLimits({'text':1},{'text':'text'},ledger_roots=tuple(map(str,roots)),reservation_lock=str(tmp_path/'quota.lease'))
    profile=ModelProfile('fixture','mock','fixture','1',mock_output={'ok':True})
    gateways=[ModelGateway(profiles=(profile,),policy=ModelPolicy({'text':'fixture'},2),receipt_root=root,attempt_limits=limits) for root in roots]
    original=ModelAttemptLimits.check;first=threading.Event()
    def checked(self,*a,**k):
        original(self,*a,**k)
        if not first.is_set():first.set();time.sleep(.1)  # Hold after reading zero, before reserving.
    monkeypatch.setattr(ModelAttemptLimits,'check',checked)
    def call(index):
        try:return gateways[index].invoke(ModelRequest('r'+str(index),'text',({'role':'user','content':'fixture'},),{'type':'object'}))
        except (ModelError,LeaseBusyError) as e:return e
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(call,0);first.wait(2);b=pool.submit(call,1);results=[a.result(),b.result()]
    assert sum(isinstance(r,(ModelError,LeaseBusyError)) for r in results)==1
    assert sum(g.audit_usage()['model_calls'] for g in gateways)==1
    assert sum(len(list(root.glob('*.receipt.json'))) for root in roots)==1
