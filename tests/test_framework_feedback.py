import json
from pathlib import Path

import pytest

from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy, ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway, ModelError
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256
from dataevolver.harness.attempt_limits import ModelAttemptLimits
from dataevolver.harness.feedback import FeedbackSignal, PolicySpec
from dataevolver.harness.training_feedback import TrainingJobSpec, ExternalTrainingRunAdapter


def test_subquotas_charge_native_attempts_but_not_cache_recovery(tmp_path):
    profile = ModelProfile('fixture', 'mock', 'fixture', 'r1', mock_output={'ok': True})
    limits = ModelAttemptLimits({'audit': 1, 'main': 2}, {'audit': 'audit', 'main': 'main'})
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({'audit': 'fixture', 'main': 'fixture'}, 3), receipt_root=tmp_path, attempt_limits=limits)
    def request(name, role):
        return ModelRequest(name, role, ({'role': 'user', 'content': 'CPU fixture'},), {'type': 'object'})
    first = request('audit-1', 'audit')
    assert not gateway.invoke(first).reused
    assert gateway.invoke(first).reused
    with pytest.raises(ModelError, match='subquota'): gateway.invoke(request('audit-2', 'audit'))
    gateway.invoke(request('main-1', 'main'))
    assert gateway.audit_usage()['model_calls'] == 2
    assert not (tmp_path/'audit-2.intent.json').exists()


def test_deadline_is_derived_from_first_native_attempt_and_remains_frozen():
    limits = ModelAttemptLimits({'main': 32}, {'review': 'main'}, 7200)
    request = ModelRequest('review', 'review', ({'role': 'user', 'content': 'fixture'},), {'type': 'object'}, timeout_seconds=60)
    records = [{'attempts': 1, 'role': 'review', 'started_at': 150, 'first_started_at': 100}]
    with pytest.raises(ValueError, match='window'): limits.check(request, records, now=7301)
    with pytest.raises(ValueError, match='timeout'): limits.check(request, records, now=7250)


def test_policy_cannot_change_frozen_evaluation_or_budget():
    assert PolicySpec('p', '1', 'evolution', {'candidate_limit': 1}).compute_hash()
    with pytest.raises(ValueError): PolicySpec('p', '2', 'evolution', {'budget': 100})
    with pytest.raises(ValueError): PolicySpec('p', '2', 'evolution', {'evaluator': 'weaker'})


def policy_resolution_fixture(tmp_path, *, scope, bind_policy):
    """CPU contract fixture only; no policy qualification or native adoption."""
    from dataevolver.harness.operator_lifecycle import OperatorVersionStore, EMPTY_HEAD
    policy = PolicySpec('fixture.policy', '1', 'evolution', {'candidate_limit': 1})
    snapshot = {'operator_id': 'fixture.operator', 'version': '1', 'scope': scope}
    if bind_policy:
        snapshot['policy_sha256'] = policy.compute_hash()
    store = OperatorVersionStore(tmp_path/'versions')
    with store.lease():
        event = store.append(action='ADOPTED', snapshot=snapshot,
                             owner={'fixture': 'policy-resolution'}, expected_head=EMPTY_HEAD)
    return policy, store, event


@pytest.mark.parametrize('kind', ['catalog', 'selection'])
def test_policy_resolver_forwards_scope_authority_but_rejects_operator_only_qualification(tmp_path, kind):
    from unittest.mock import Mock
    from dataevolver.harness.catalog_candidate import CatalogQualificationAuthority, SCOPE
    from dataevolver.harness.selection_candidate import SelectionQualificationAuthority, SCOPE as SELECTION_SCOPE
    from dataevolver.harness.feedback import PolicyResolver
    scope, authority_type = ((SCOPE, CatalogQualificationAuthority) if kind == 'catalog'
                             else (SELECTION_SCOPE, SelectionQualificationAuthority))
    policy, store, event = policy_resolution_fixture(tmp_path, scope=scope, bind_policy=False)
    authority = Mock(spec=authority_type)
    with pytest.raises(ValueError, match='policy version differs from qualified snapshot'):
        PolicyResolver(store, authority).resolve(event, scope=scope, expected_head=store.head(),
                                                policy=policy.to_dict())
    authority.verify.assert_called_with(event['snapshot'])


def test_policy_resolver_rejects_authority_for_another_scope(tmp_path):
    from unittest.mock import Mock
    from dataevolver.harness.catalog_candidate import SCOPE
    from dataevolver.harness.selection_candidate import SelectionQualificationAuthority
    from dataevolver.harness.feedback import PolicyResolver
    policy, store, event = policy_resolution_fixture(tmp_path, scope=SCOPE, bind_policy=False)
    authority = Mock(spec=SelectionQualificationAuthority)
    with pytest.raises(ValueError, match='matching qualification authority'):
        PolicyResolver(store, authority).resolve(event, scope=SCOPE, expected_head=store.head(),
                                                policy=policy.to_dict())
    authority.verify.assert_not_called()


def test_policy_resolver_requires_exact_policy_and_current_unrevoked_catalog(tmp_path):
    from unittest.mock import Mock
    from dataevolver.harness.feedback import PolicyResolver
    from dataevolver.harness.operator_lifecycle import SYNTHETIC_SCOPE, EMPTY_HEAD
    policy, store, event = policy_resolution_fixture(tmp_path, scope=SYNTHETIC_SCOPE, bind_policy=True)
    resolver = PolicyResolver(store, Mock())
    head = store.head()
    assert resolver.resolve(event, scope=SYNTHETIC_SCOPE, expected_head=head, policy=policy.to_dict()) == policy
    changed = PolicySpec(policy.policy_id, policy.version, policy.role, {'candidate_limit': 2})
    with pytest.raises(ValueError, match='policy version differs'):
        resolver.resolve(event, scope=SYNTHETIC_SCOPE, expected_head=head, policy=changed.to_dict())
    with pytest.raises(ValueError, match='catalog snapshot is incompatible'):
        resolver.resolve(event, scope=SYNTHETIC_SCOPE, expected_head=EMPTY_HEAD, policy=policy.to_dict())
    with store.lease():
        store.append(action='REVOKED', snapshot=event['snapshot'], owner={'fixture': 'revoke'},
                     expected_head=head, reason='CPU fixture revocation')
    with pytest.raises(ValueError, match='revoked'):
        resolver.resolve(event, scope=SYNTHETIC_SCOPE, expected_head=store.head(), policy=policy.to_dict())


def training_fixture(tmp_path, *, split='development'):
    store = ArtifactStore({'evidence': tmp_path/'evidence'})
    root = store.roots['evidence']; root.mkdir()
    def ref(name, value):
        p = root/name; p.write_text(json.dumps(value)); return ArtifactRef('evidence', name, file_sha256(p))
    dataset, init, config, evaluation = [ref(n+'.json', {n: 'fixture'}) for n in ('dataset', 'init', 'config', 'evaluation')]
    checkpoint = ref('checkpoint.json', {'fixture_checkpoint': True})
    job = TrainingJobSpec('external-job', dataset, init, config, evaluation, 'fixture-model', 9, split)
    adapter = ExternalTrainingRunAdapter(store=store, work_root=tmp_path/'adapter')
    adapter.submit(job, external_job_id='external-123')
    value = {'job_sha256': job.compute_hash(), 'dataset_sha256': dataset.sha256,
        'initialization_sha256': init.sha256, 'training_config_sha256': config.sha256, 'evaluation_sha256': evaluation.sha256,
        'model': job.model, 'seed': job.seed, 'evaluation_split': split, 'exit_code': 0, 'external_job_id': 'external-123',
        'checkpoint': checkpoint.to_dict(), 'metrics': {'accuracy': .6}, 'slices': {'a': .4}, 'usage': {'gpu_seconds': None}}
    return job, adapter, value, ref


def test_one_external_adapter_collects_bound_result_and_creates_next_plan_context(tmp_path):
    job, adapter, value, ref = training_fixture(tmp_path)
    receipt = ref('receipt.json', value)
    result = adapter.collect(job, receipt)
    assert result.job_sha256 == job.compute_hash()
    context = adapter.next_data_plan_context(job, receipt)
    assert context['previous_dataset'] == job.dataset.to_dict()
    assert context['feedback'][0]['kind'] == 'training_result'
    signal = FeedbackSignal.from_dict(context['feedback'][0])
    assert signal.to_dict() == context['feedback'][0]
    assert signal.validate_evidence(adapter.store)
    assert context['training_benefit_verified'] is False


def catalog_feedback_fixture(tmp_path, *, split='development',collect=True):
    from test_framework_integration import compile_fixture
    from dataevolver.harness.catalog_rsi import CatalogRSIDomain
    training=tmp_path/'training';training.mkdir()
    job,adapter,value,ref=training_fixture(training,split=split)
    receipt=ref('receipt.json',value)
    if collect:adapter.collect(job,receipt)
    catalog=tmp_path/'catalog';catalog.mkdir()
    compiled,context,_=compile_fixture(catalog)
    config={'compiled_run':compiled.to_dict(),'source':compiled.contract['intent']['sources'][0],
        'baseline_ranking':'hash','allowed_rankings':['hash','clarity'],'quotas':{'a':{'fixture':3}},
        'target_quality':.8,'maximum_candidates':0}
    return job,adapter,receipt,config,context


def test_collected_training_feedback_enters_next_runtime_decision_without_new_effects(tmp_path):
    from dataevolver.harness.catalog_rsi import CatalogRSIDomain
    from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
    from dataevolver.harness.models import CampaignState
    from dataevolver.testing.test_runtime_rsi import action
    job,adapter,receipt,config,context=catalog_feedback_fixture(tmp_path)
    before={p:p.read_bytes() for p in adapter.root.rglob('*') if p.is_file()}
    domain=CatalogRSIDomain(configuration=config,registry=context.registry,roots=context.roots,
        training_feedback=(adapter,job,receipt))
    class Backend:
        backend_id=model_id='fixture-feedback'
        def complete_json(self,messages,*,schema):
            value=json.loads(messages[-1]['content'])
            assert value['feedback_signals'][0]['scope']['job_sha256']==job.compute_hash()
            assert value['feedback_signals'][0]['metrics']['task']=={'accuracy':.6}
            assert value['feedback_signals'][0]['metrics']['slices']=={'a':.4}
            assert value['training_feedback_scope']['requires_new_frozen_run'] is True
            assert value['training_feedback_scope']['evidence_id']=='training-feedback:'+receipt.sha256
            assert value['training_feedback_scope']['evidence_id'] in value['available_evidence_ids']
            assert value['allowed_actions']==['COMPLETE','ABSTAIN']
            return json.dumps(action('ABSTAIN'))
    planner=RuntimeRSIPlanner(backend=Backend(),domain=domain,trace_root=tmp_path/'trace')
    state=CampaignState(campaign_id='feedback-next',planner_calls=0,
                        data={'last_stage_result':{'evidence_refs':['baseline']}})
    assert planner.plan(state)['action']=='ABSTAIN'
    assert planner.config()['domain']['training_feedback']['context']['feedback'][0]['evidence']==[receipt.to_dict()]
    assert before=={p:p.read_bytes() for p in adapter.root.rglob('*') if p.is_file()}
    config['training_feedback']={'context':{'fake':'summary'}}
    with pytest.raises(ValueError,match='registered adapter'):
        CatalogRSIDomain(configuration=config,registry=context.registry,roots=context.roots)


@pytest.mark.parametrize('problem',['sealed','not_collected','drift'])
def test_next_catalog_run_rejects_unavailable_or_changed_training_feedback(tmp_path,problem):
    from dataevolver.harness.catalog_rsi import CatalogRSIDomain
    from dataevolver.harness.models import CampaignState
    job,adapter,receipt,config,context=catalog_feedback_fixture(tmp_path,
        split='sealed' if problem=='sealed' else 'development',collect=problem!='not_collected')
    kwargs=dict(configuration=config,registry=context.registry,roots=context.roots,training_feedback=(adapter,job,receipt))
    before={p:p.read_bytes() for p in adapter.root.rglob('*') if p.is_file()}
    if problem=='drift':
        domain=CatalogRSIDomain(**kwargs)
        path=adapter.store.resolve(receipt);value=json.loads(path.read_text());value['metrics']['accuracy']=.99
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):domain.context(CampaignState(campaign_id='feedback-next'))
    else:
        with pytest.raises(ValueError,match='sealed' if problem=='sealed' else 'already collected'):
            CatalogRSIDomain(**kwargs)
    assert before=={p:p.read_bytes() for p in adapter.root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('field', ['dataset_sha256', 'initialization_sha256', 'training_config_sha256', 'evaluation_sha256', 'model', 'seed', 'external_job_id', 'exit_code'])
def test_wrong_or_failed_training_receipt_is_rejected(tmp_path, field):
    job, adapter, value, ref = training_fixture(tmp_path)
    value[field] = 1 if field in {'seed', 'exit_code'} else 'wrong'
    with pytest.raises(ValueError): adapter.collect(job, ref('receipt.json', value))


def test_sealed_training_result_cannot_reenter_optimization(tmp_path):
    job, adapter, value, ref = training_fixture(tmp_path, split='sealed')
    receipt = ref('receipt.json', value)
    assert adapter.collect(job, receipt)
    with pytest.raises(ValueError, match='sealed'): adapter.next_data_plan_context(job, receipt)


def test_feedback_inspection_keeps_existing_job_and_rejects_sealed_inputs(tmp_path):
    from dataevolver.harness.adapters.training_feedback_inspection import TrainingFeedbackInspectionOperator
    from dataevolver.harness.models import OperatorCall,EvidenceStatus
    job,adapter,value,ref=training_fixture(tmp_path)
    receipt=ref('receipt.json',value)
    adapter.collect(job,receipt)
    before=adapter._path(job).read_bytes()
    operator=TrainingFeedbackInspectionOperator(adapter=adapter,job=job,receipt=receipt,output_root=tmp_path/'inspect')
    call=OperatorCall('fixture','inspect','inspect',operator.spec.operator_id,'1',0,str(tmp_path/'inspect/call'))
    result=operator.execute(call)
    assert operator.verify_effect(call,result).status is EvidenceStatus.VERIFIED
    report=json.loads(operator.path(call).read_text())['report']
    assert report['next_run_required'] is True and report['new_training_jobs']==0
    assert report['training_benefit_verified'] is False
    assert adapter._path(job).read_bytes()==before
    with pytest.raises(ValueError,match='sealed'):
        TrainingFeedbackInspectionOperator(adapter=adapter,job=job.__class__(job.job_id,job.dataset,job.initialization,
            job.training_config,job.evaluation,job.model,job.seed,'sealed'),receipt=receipt,output_root=tmp_path/'sealed')


def test_promoted_label_and_existing_files_do_not_authorize_knowledge_reuse(tmp_path):
    from dataevolver.harness.catalog_knowledge import compatible_catalog_experiences
    from dataevolver.harness.capabilities import CapabilityStore
    from dataevolver.harness.models import Capability, CapabilityStatus, FailureSignature, Intervention, InterventionLevel
    from dataevolver.dataset.contracts import payload_sha256
    scope = {'task': 'catalog_sampling'}
    fake = tmp_path/'unqualified.json'; fake.write_text('{}')
    capability = Capability('unqualified', '1', CapabilityStatus.PROMOTED, FailureSignature('CATALOG_QUALITY_GAP'),
        Intervention('unqualified', 'CATALOG_QUALITY_GAP', 'data.catalog_trial', InterventionLevel.L1,
            parameters={'ranking': 'clarity', 'scope': scope}), validation_evidence=(str(fake),),
        compatibility_hash=payload_sha256(scope))
    store = CapabilityStore(tmp_path/'capabilities.jsonl'); store.append_candidate(capability)
    assert compatible_catalog_experiences(store, scope=scope)['empty_reason'] == 'NO_ENTRY'


def test_registered_external_process_consumes_frozen_request_and_never_resubmits(tmp_path):
    import sys
    from dataevolver.adapters.external_training import RegisteredTrainingRunner
    job, old, value, ref = training_fixture(tmp_path)
    script=tmp_path/'external_protocol_fixture.py'
    script.write_text('''import json,sys\nfrom pathlib import Path\nr=json.loads(Path(sys.argv[2]).read_text());j=r['job']\nroot=Path(r['stores'][r['result_store']]);p=root/r['result_path'];p.parent.mkdir(parents=True,exist_ok=True)\nvalue=r['inputs']['receipt'];p.write_text(json.dumps(value))\nc=root/'submissions';c.write_text(str(int(c.read_text())+1) if c.exists() else '1')\n''')
    runner=RegisteredTrainingRunner(store=old.store,work_root=tmp_path/'registered',command=(sys.executable,str(script)),
        source_hashes={str(script):file_sha256(script)},environment={'CUDA_VISIBLE_DEVICES':''},max_wall_seconds=10)
    value['external_job_id']='registered-id'
    kwargs={'external_job_id':'registered-id','result_store':'evidence','result_path':'native-result.json','runner_inputs':{'receipt':value}}
    first=runner.run(job,**kwargs)
    assert first.job_sha256==job.compute_hash()
    assert runner.run(job,**kwargs)==first
    assert (old.store.roots['evidence']/'submissions').read_text()=='1'
    changed={**kwargs,'runner_inputs':{'different':'request'}}
    with pytest.raises(ValueError,match='binding'):runner.run(job,**changed)
    script.write_text(script.read_text()+'\n# changed\n')
    with pytest.raises(ValueError,match='implementation'):runner.run(job,**kwargs)


def test_uncertain_training_launch_is_not_silently_retried(tmp_path,monkeypatch):
    import sys
    from dataevolver.adapters.external_training import RegisteredTrainingRunner
    job,old,_,_=training_fixture(tmp_path)
    script=tmp_path/'fixture.py';script.write_text('raise RuntimeError("fixture")')
    runner=RegisteredTrainingRunner(store=old.store,work_root=tmp_path/'registered',command=(sys.executable,str(script)),
        source_hashes={str(script):file_sha256(script)},environment={'CUDA_VISIBLE_DEVICES':''},max_wall_seconds=10)
    from dataevolver.adapters import external_training
    def crash(*a,**k):raise RuntimeError('injected after launch intent')
    monkeypatch.setattr(external_training.subprocess,'Popen',crash)
    kwargs={'external_job_id':'registered-id','result_store':'evidence','result_path':'native.json'}
    with pytest.raises(RuntimeError,match='injected'):runner.run(job,**kwargs)
    with pytest.raises(RuntimeError,match='cannot be resubmitted'):runner.run(job,**kwargs)
    launch=json.loads((tmp_path/'registered'/('external-job.runner')/'launch.json').read_text())
    assert launch['state']=='UNCERTAIN'
