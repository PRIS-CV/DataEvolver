"""Evidence/cost-preserving model-owned revisions of isolated candidates."""
import json
import os
import platform
from dataclasses import replace

import pytest

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
from dataevolver.adapters.model_gateway import ModelGateway
from dataevolver.dataset.contracts import ArtifactStore, file_sha256, payload_sha256
from dataevolver.harness.candidate_revision import ControllerContractFeedbackAuthority
from dataevolver.harness.adapters.operator_candidate_revision import CandidateRevisionBuildOperator
from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.evaluator import EvaluatorContract
from dataevolver.harness.models import ControllerState, EvidenceStatus, OperatorCall, RecoveryStatus
from dataevolver.harness.operator_development import DevelopmentBudget, CandidatePackage
from dataevolver.harness.operators import OperatorRegistry
from dataevolver.harness.planner import DeterministicPlanner
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.testing.test_candidate_trust_qualification import qualification_fixture
from dataevolver.testing.test_harness_controller import build_controller

pytestmark=pytest.mark.skipif(platform.system()!='Linux' or os.geteuid()!=0,
                             reason='actual Linux isolation required')


def fixture(tmp_path):
    op,_=qualification_fixture(tmp_path)
    template,_=build_controller(tmp_path/'template')
    evaluator=tmp_path/'qualifier-evaluator.json'
    atomic_write_json(evaluator,{'schema_version':'dataevolver.harness.evaluator.v1',
        'evaluator_id':'revision-fixture-contract-qualification','revision':'1',
        'hard_constraints':{'artifact_valid':{'required':True}},
        'target_metrics':{'contract_qualification_passed':{'required':True,'minimum':1}},'protected_metrics':{}})
    goal=template.goal.with_updates(campaign_id='revision-feedback-fixture',initial_operator_id=op.spec.operator_id,
        objective={**template.goal.objective,'candidate_package_sha256':op.package.compute_hash(),
            'contract_qualification_protocol_sha256':payload_sha256(op.protocol)},
        resource_policy={'allow_gpu':False,'allowed_operators':[op.spec.operator_id]},
        frozen_evaluator={'path':str(evaluator),'sha256':file_sha256(evaluator)},interventions=())
    registry=OperatorRegistry();registry.register(op)
    c=EvolutionController.create(repo_root=tmp_path,campaigns_root=op.output_root/'campaigns',goal=goal,
        registry=registry,planner=DeterministicPlanner(()),evaluator_contract=EvaluatorContract.from_path(evaluator),
        idempotency_key=goal.campaign_id)
    end=c.run();assert end.state is ControllerState.ABSTAIN
    call_id=next(e.payload['call_id'] for e in c.store.events.read() if e.event_type=='ACTION_COMMITTED')
    authority=ControllerContractFeedbackAuthority(c,op,call_id)
    feedback=authority.read();prior=feedback['lifecycle_total']
    budget=DevelopmentBudget({**op.plan.budget['limits'],'operator_calls':prior['operator_calls']+1,
        'model_calls':prior['model_calls']+1},prior,op.budget.resource_deadline_epoch)
    plan=replace(op.plan,plan_id='model-owned-fixture-revision',candidate_store='revision',budget=budget.to_dict())
    revision=tmp_path/'revision';revision.mkdir()
    store=ArtifactStore({**{k:v for k,v in op.store.roots.items() if k!='candidate'},'revision':revision})
    output={'files':[{'path':'revision.py','content':'def resolve(p): return {"status":"unsupported"}\n'}],
            'entrypoint':'revision.py','dependencies':[],'change_description':'Fixture model-owned unqualified revision.'}
    profile=ModelProfile('revision-model-fixture','mock','fixture-only','1',mock_output=output)
    root=tmp_path/'revision-output'
    gateway=ModelGateway(profiles=(profile,),policy=ModelPolicy({'agent.runtime_rsi':profile.profile_id},1),
                         receipt_root=root/'receipts')
    builder=CandidateRevisionBuildOperator(plan=plan,store=store,request_authority=op.request_authority,
        gateway=gateway,role='agent.runtime_rsi',output_root=root,
        native_roots=(gateway.root/'effects'/profile.profile_id,),max_tokens=2048,timeout_seconds=60,
        predecessor_plan=op.plan,predecessor_store=op.store,feedback_authority=authority)
    call=OperatorCall('revision-fixture','revision-1','revision-1',builder.spec.operator_id,'1',0,str(root/'call'))
    return builder,call,c,op,authority


def test_committed_failed_checks_feed_one_model_owned_inert_revision(tmp_path):
    builder,call,c,op,authority=fixture(tmp_path)
    before={str(p):file_sha256(p) for p in op.store.roots['candidate'].iterdir()}
    req=builder._request(call);text=json.dumps(req.to_dict())
    assert req.request_id.startswith('revise-') and 'predecessor_source' in text
    assert 'committed_engineering_feedback' in text and 'independent-a' not in text
    receipt=builder.execute(call)
    assert builder.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    package=CandidatePackage.from_dict(json.loads((builder.output_root/'call/candidate-package.json').read_text()))
    assert package.compute_hash()!=op.package.compute_hash() and package.implementation['store']=='revision'
    assert {str(p):file_sha256(p) for p in op.store.roots['candidate'].iterdir()}==before
    assert builder.execute(call).reused and builder.gateway.audit_usage()['model_calls']==1


def test_revision_retains_prior_cost_and_original_deadline(tmp_path):
    builder,call,_,op,_=fixture(tmp_path)
    assert builder.budget.prior_usage['model_calls']==1 and builder.budget.prior_usage['operator_calls']==2
    old=builder.budget
    builder.budget=replace(old,prior_usage={**old.prior_usage,'model_calls':0})
    with pytest.raises(ValueError,match='complete costs'):builder._request(call)
    builder.budget=replace(old,resource_deadline_epoch=old.resource_deadline_epoch+1)
    with pytest.raises(ValueError,match='deadline'):builder._request(call)


def test_forged_qualification_commit_or_captured_output_is_rejected(tmp_path):
    builder,call,c,op,authority=fixture(tmp_path)
    # The committed output itself is re-verified, not a hand-written feedback file.
    record=next(op.output_root.rglob('executions/0.json'))
    value=json.loads(record.read_text());value['output']={'forged':True};atomic_write_json(record,value)
    with pytest.raises(ValueError,match='no longer verify'):authority.read()
    with pytest.raises(ValueError):builder._request(call)
    assert builder.gateway.audit_usage()['model_calls']==0


def test_changed_goal_or_uncommitted_call_cannot_authorize_revision(tmp_path):
    _,_,c,op,authority=fixture(tmp_path)
    with pytest.raises(ValueError,match='exact committed'):
        ControllerContractFeedbackAuthority(c,op,'not-committed').read()
    c.goal=c.goal.with_updates(objective={**c.goal.objective,'candidate_package_sha256':'a'*64})
    with pytest.raises(ValueError):authority.read()


def test_interrupted_revision_reuses_same_completed_response(tmp_path,monkeypatch):
    builder,call,_,_,_=fixture(tmp_path)
    finish=builder._finish
    monkeypatch.setattr(builder,'_finish',lambda *args:(_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(RuntimeError):builder.execute(call)
    monkeypatch.setattr(builder,'_finish',finish)
    assert builder.recover(call).status is RecoveryStatus.RECOVERED
    assert builder.gateway.audit_usage()['model_calls']==1
