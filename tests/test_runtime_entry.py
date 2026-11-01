"""Entry integration with actual Controllers/graphs and explicitly synthetic models.

No research candidates, native inference, new images or training are dispatched.
"""
import copy
import json
from pathlib import Path

import pytest

from dataevolver.dataset.contracts import file_sha256
from dataevolver.harness import cli, runtime_entry
from dataevolver.harness.models import BudgetSpec, GoalSpec
from dataevolver.harness.feedback import PolicySpec
from dataevolver.harness.evaluator import EvaluatorContract
from dataevolver.harness.state_store import atomic_write_json
from test_policy_qualification import PolicyFixture


def reference(controller):
    return {'campaign_root': str(controller.layout.campaigns_root), 'campaign_id': controller.goal.campaign_id}


@pytest.fixture
def synthetic_gateway(monkeypatch):
    """Only the provider output is synthetic; gateway journals/recovery are real."""
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.testing.test_runtime_rsi import action
    def gateway(**kwargs):
        profile = kwargs['profiles'][0]
        class Provider:
            def lookup(self, *args):
                raise AssertionError('confirmed fixture effects must be recovered from their gateway receipts')
            def invoke(self, request, digest):
                context = json.loads(request.messages[-1]['content'])
                if ('RUN_EXPERIMENT' in context['allowed_actions'] and not context['history']
                        and context['current_metrics']['quality'] < context['target_quality']):
                    decision = action('RUN_EXPERIMENT', layer='R', patch='sampling_recipe',
                        parameters={'ranking': 'clarity'}, evidence=context['available_evidence_ids'])
                elif context['current_metrics']['quality'] >= context['target_quality']:
                    decision = action('COMPLETE', evidence=context['available_evidence_ids'])
                else:
                    decision = action('ABSTAIN')
                return {'model': profile.model, 'output': decision,
                        'usage': {'input_tokens': 1, 'output_tokens': 1}, 'finish_reason': 'stop'}
        return ModelGateway(**kwargs, providers={profile.profile_id: Provider})
    monkeypatch.setattr(runtime_entry, 'ModelGateway', gateway)


class EntryFixture(PolicyFixture):
    """Reuse the existing qualification fixture through the public CLI factory."""
    def new_run(self, source, policy, protocol, suffix, *, model_id='fixture-policy-agent',
                target_quality=None, admission=None, feedback=None):
        from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy
        name = source['family_id']; domain = self.domains[name]
        campaign = name+'-'+suffix
        configuration = copy.deepcopy(domain.value)
        if target_quality is not None:
            configuration['target_quality'] = target_quality
        runtime = {'domain_kind': 'catalog', 'domain': configuration,
            'roots': {k: str(v) for k, v in domain.roots.items()}, 'policy': policy.to_dict(),
            'gateway': {'profiles': [ModelProfile('fixture', 'mock', model_id, 'fixture-1').to_dict()],
                'policy': ModelPolicy({'agent.runtime_rsi': 'fixture'}, 8).to_dict(),
                'receipt_root': str(self.root/'gateways'/campaign)}}
        if admission is not None:
            runtime['policy_adoption'] = admission
        if feedback is not None:
            runtime['training_feedback'] = feedback
        binding = {'protocol_sha256': protocol.compute_hash() if protocol else '0'*64,
            'stage': 'engineering' if source['split'] in {'development', 'regression'} else 'independent',
            'source': source, 'policy_sha256': policy.compute_hash()}
        goal = GoalSpec('dataevolver.harness.goal.v1', campaign, {'kind': 'policy-fixture',
            'initial_trial_key': configuration['matched_trial_key'], 'policy_comparison': binding},
            'data.catalog_trial', (), BudgetSpec(4, 5, 1, 0, 600, 4),
            {'allowed_operators': ['data.catalog_trial']},
            {'path': str(self.eval_path), 'sha256': self.evaluator.sha256},
            initial_operator_parameters={'ranking': 'hash', 'experience_id': ''},
            planner={'kind': 'runtime_rsi', 'runtime': runtime})
        c = cli._controller(goal, self.root/'controllers', create=True, idempotency_key=campaign)
        self.operators[name] = c.registry.get('data.catalog_trial')
        return c

    def qualification_config(self):
        return {'kind': 'policy', 'engineering': reference(self.eng), 'independent': reference(self.ind)}

    def execute(self, op, name, evaluator_path, parameters):
        from dataevolver.harness.policy_qualification import PolicyComparisonChecks
        from dataevolver.dataset.contracts import payload_sha256
        if isinstance(op, PolicyComparisonChecks):
            host = {'kind': 'policy_comparison', 'configuration': {
                'baseline_policy': op.baseline.to_dict(), 'policy': op.policy.to_dict(),
                'protocol': op.protocol.to_dict(), 'roots': {k: str(v) for k, v in op.store.roots.items()},
                'pairs': [[reference(c) for c, _ in pair] for pair in op.pairs],
                'phase': op.phase, 'budget': op.budget.to_dict(), 'output_root': str(op.output_root)}}
        else:
            values = {'catalog_root': str(op.catalog.root), 'expected_head': op.expected_head,
                'budget': op.budget.to_dict(), 'output_root': str(op.output_root)}
            if op.adoption is not None:
                values.update(adoption=reference(op.adoption.controller), reason=op.reason)
            else:
                values.update(qualification=self.qualification_config(),
                              roots={k: str(v) for k, v in self.store.roots.items()})
            host = {'kind': 'version_mutation', 'configuration': values}
        key = {'input_artifact_hash': payload_sha256(parameters), 'parent_artifact_hash': payload_sha256(parameters),
            'seed': 0, 'operator_id': op.spec.operator_id, 'operator_version': op.spec.version,
            'model_config_hash': 'fixture', 'preprocessing_hash': 'fixture',
            'evaluator_hash': file_sha256(evaluator_path), 'environment_contract_hash': 'fixture'}
        metric = 'qualification_passed' if isinstance(op, PolicyComparisonChecks) else 'operator_'+('adopted' if name=='adopt' else 'revoked')
        goal = GoalSpec('dataevolver.harness.goal.v1', 'qual-'+name,
            {'initial_trial_key': key, 'target_metrics': {metric: 1}, 'host_workload': host},
            op.spec.operator_id, (), BudgetSpec(1, 1, 0, 0, 600), {'allowed_operators': [op.spec.operator_id]},
            {'path': str(evaluator_path), 'sha256': file_sha256(evaluator_path)}, initial_operator_parameters=parameters)
        c = cli._controller(goal, op.output_root, create=True, idempotency_key='qual-'+name)
        c.run()
        return c

    def adopt(self):
        from dataevolver.harness.operator_lifecycle import OperatorVersionStore, CommittedOperatorAdoptionAuthority
        from dataevolver.harness.adapters.operator_version_lifecycle import OperatorVersionMutation
        from dataevolver.harness.operator_development import DevelopmentBudget
        self.catalog = OperatorVersionStore(self.root/'versions')
        snapshot = self.authority.read(); prior = snapshot['lifecycle_total']
        limits = {k: v+(1000000 if k=='output_bytes' else 1000) for k,v in prior.items()}
        for k in ('model_calls', 'planner_calls', 'image_experiments'):
            limits[k] = prior[k]
        self.mutation_budget = DevelopmentBudget(limits, prior, self.deadline)
        self.adopt_eval = self.root/'adopt-evaluator.json'
        atomic_write_json(self.adopt_eval, {'schema_version': 'dataevolver.harness.evaluator.v1',
            'evaluator_id': 'entry-adopt', 'revision': '1', 'hard_constraints': {'artifact_valid': {'required': True}},
            'target_metrics': {'operator_adopted': {'minimum': 1}}, 'protected_metrics': {}})
        op = OperatorVersionMutation(catalog=self.catalog, expected_head=self.catalog.head(),
            budget=self.mutation_budget, output_root=self.root/'adopt', store=self.store,
            qualification_authority=self.authority)
        self.adopted = self.execute(op, 'adopt', self.adopt_eval, {})
        call = next(e.payload['call_id'] for e in self.adopted.store.events.read() if e.event_type=='ACTION_PLANNED')
        self.adoption = CommittedOperatorAdoptionAuthority(self.adopted, self.adopted.registry.get(op.spec.operator_id), call)
        return {'catalog_root': str(self.catalog.root), 'expected_head': self.catalog.head(),
            'adoption': reference(self.adopted), 'qualification': self.qualification_config()}


def test_catalog_cli_create_execute_recover_and_inspect(tmp_path, synthetic_gateway, capsys):
    fixture = EntryFixture(tmp_path)
    source = {'family_id': 'successor', 'split': 'development', 'artifact': fixture.domains['successor'].catalog_ref.to_dict()}
    c = fixture.new_run(source, fixture.policy, fixture.protocol, 'entry')
    assert c.status().state.value == 'INIT' and c.status().operator_calls == 0
    assert not list((tmp_path/'gateways/successor-entry').glob('*.intent.json'))
    assert cli.main(['create','--goal',str(c.layout.goal_path),'--campaign-root',str(c.layout.campaigns_root),
                     '--idempotency-key',c.goal.campaign_id]) == 0
    assert json.loads(capsys.readouterr().out)['operator_calls'] == 0
    args = ['--campaign', c.goal.campaign_id, '--campaign-root', str(c.layout.campaigns_root)]
    assert cli.main(['resume', *args]) == 0
    assert json.loads(capsys.readouterr().out)['state'] == 'COMPLETE'
    before = {p: p.read_bytes() for p in c.layout.campaign_dir.rglob('*') if p.is_file()}
    assert cli.main(['resume', *args]) == 0
    capsys.readouterr()
    assert before == {p:p.read_bytes() for p in c.layout.campaign_dir.rglob('*') if p.is_file()}
    assert cli.main(['replay', *args]) == 0
    assert json.loads(capsys.readouterr().out)['matches_snapshot'] is True


def test_policy_two_batches_exact_admission_recovery_and_revocation(tmp_path, synthetic_gateway, monkeypatch, capsys):
    from dataevolver.harness.feedback import PolicyResolver
    from dataevolver.harness.adapters.operator_version_lifecycle import OperatorVersionMutation
    fixture = EntryFixture(tmp_path); fixture.qualify()
    source = {'family_id': 'successor', 'split': 'development', 'artifact': fixture.domains['successor'].catalog_ref.to_dict()}
    first = fixture.new_run(source, fixture.baseline, fixture.protocol, 'batch1'); first.run()
    first_bytes = first.layout.goal_path.read_bytes()
    binding = fixture.adopt()
    wrong=PolicySpec(fixture.policy.policy_id,'wrong','evolution',{'candidate_limit':1})
    with pytest.raises(ValueError,match='qualified version'):
        fixture.new_run(source,wrong,fixture.protocol,'wrong-policy',admission=binding)
    with pytest.raises(ValueError,match='qualified context'):
        fixture.new_run(source,fixture.policy,fixture.protocol,'wrong-model',admission=binding,model_id='wrong-fixture-model')
    second = fixture.new_run(source, fixture.policy, fixture.protocol, 'batch2', admission=binding)
    assert second.status().state.value == 'INIT'
    assert second.status().operator_calls == second.status().planner_calls == 0
    admission = json.loads((second.layout.campaign_dir/'runtime-admission.json').read_text())
    assert admission['provenance']['policy']['policy_sha256'] == fixture.policy.compute_hash()
    assert second.replay().to_dict() == second.status().to_dict()
    path=second.layout.campaign_dir/'runtime-admission.json'
    atomic_write_json(path,{**admission,'bindings_sha256':'0'*64})
    with pytest.raises(ValueError,match='pinned runtime admission'):
        second.tick()
    assert second.status().operator_calls == second.status().planner_calls == 0
    atomic_write_json(path,admission)
    paused=fixture.new_run(source,fixture.policy,fixture.protocol,'paused-consumer',admission=binding)
    paused.tick(1)
    assert paused.status().operator_calls==paused.status().planner_calls==0
    # INIT resolution is complete. Recovery must validate the pinned evidence,
    # not try to readopt or call the INIT-only PolicyResolver again.
    monkeypatch.setattr(PolicyResolver, 'resolve', lambda *a, **k: (_ for _ in ()).throw(AssertionError('resolved again')))
    second.tick(3)
    args = ['--campaign', second.goal.campaign_id, '--campaign-root', str(second.layout.campaigns_root)]
    assert cli.main(['resume', *args]) == 0
    capsys.readouterr()
    assert second.status().state.value == 'COMPLETE'
    final=json.loads(second.layout.final_report_path.read_text())
    assert final['runtime_version_provenance']==admission['provenance']
    assert first.layout.goal_path.read_bytes() == first_bytes
    assert second.status().experiments_used == 1 and first.status().experiments_used == 0
    before = {p:p.read_bytes() for p in (tmp_path/'gateways').rglob('*') if p.is_file()}
    assert cli.main(['resume', *args]) == 0
    capsys.readouterr()
    assert before == {p:p.read_bytes() for p in (tmp_path/'gateways').rglob('*') if p.is_file()}
    evaluation = json.loads(fixture.adopt_eval.read_text()); evaluation['target_metrics']={'operator_revoked': {'minimum':1}}
    path = tmp_path/'revoke-evaluator.json'; atomic_write_json(path,evaluation)
    from dataevolver.harness.operator_development import DevelopmentBudget
    prior = fixture.adoption.usage()
    limits = {k: v+(1000000 if k=='output_bytes' else 1000) for k,v in prior.items()}
    for k in ('model_calls', 'planner_calls', 'image_experiments'): limits[k] = prior[k]
    op = OperatorVersionMutation(catalog=fixture.catalog, expected_head=fixture.catalog.head(),
        budget=DevelopmentBudget(limits,prior,fixture.deadline), output_root=tmp_path/'revoke',
        adoption=fixture.adoption, reason='Explicit fixture withdrawal')
    fixture.execute(op, 'revoke', path, {})
    with pytest.raises(ValueError):paused.resume()
    assert paused.status().operator_calls==paused.status().planner_calls==0
    frozen = second.layout.frozen_contract_path.read_bytes()
    with pytest.raises(ValueError):
        fixture.new_run(source, fixture.policy, fixture.protocol, 'batch3', admission=binding)
    refused = tmp_path/'controllers/successor-batch3'
    state = json.loads((refused/'state.json').read_text())
    assert state['operator_calls'] == state['planner_calls'] == 0
    assert (refused/'runtime-admission-refusal.json').is_file()
    assert second.layout.frozen_contract_path.read_bytes() == frozen
    assert cli.main(['status', *args]) == 0
    assert json.loads(capsys.readouterr().out)['state'] == 'COMPLETE'
    atomic_write_json(tmp_path.parent/'policy-entry-acceptance.json', {
        'evidence_scope': 'CPU_fixture_only_no_native_qualification',
        'first_batch': {**reference(first), 'policy_sha256': fixture.baseline.compute_hash(), 'experiments': first.status().experiments_used},
        'second_batch': {**reference(second), 'policy_sha256': fixture.policy.compute_hash(), 'experiments': second.status().experiments_used},
        'admission': admission, 'replay_matches': second.replay().to_dict()==second.status().to_dict(),
        'old_batch_goal_unchanged': first.layout.goal_path.read_bytes()==first_bytes,
        'restore_new_model_requests': 0, 'post_revocation_new_batch_refused': True,
        'new_experiment_native_calls': 0, 'new_research_candidates': 0, 'new_training_steps': 0})


@pytest.mark.parametrize('field', ['policy', 'model', 'goal'])
def test_active_catalog_cannot_replace_frozen_configuration(tmp_path, synthetic_gateway, field):
    fixture = EntryFixture(tmp_path)
    c = fixture.new_run(fixture.cases[0], fixture.policy, fixture.protocol, 'frozen')
    value = c.goal.to_dict()
    if field == 'policy': value['planner']['runtime']['policy']['config']['candidate_limit'] = 2
    elif field == 'model': value['planner']['runtime']['gateway']['profiles'][0]['revision'] = 'changed'
    else: value['objective']['changed'] = True
    with pytest.raises(ValueError, match='different GoalSpec'):
        cli._controller(GoalSpec.from_dict(value), c.layout.campaigns_root, create=True, idempotency_key=c.goal.campaign_id)
    assert c.status().operator_calls == c.status().planner_calls == 0


@pytest.mark.parametrize('rule', ['limit', 'no_improvement'])
def test_policy_changes_actual_allowed_actions_or_stop_point(tmp_path, synthetic_gateway, rule):
    fixture = EntryFixture(tmp_path)
    from dataevolver.harness.models import CampaignState
    if rule == 'limit':
        policy = PolicySpec('fixture-policy','3','evolution',{'candidate_limit':0})
        c = fixture.new_run(fixture.cases[0],policy,fixture.protocol,'limit')
        c.run()
        assert c.status().experiments_used == c.status().planner_calls == 0
        assert c.status().terminal_reason == 'POLICY_CANDIDATE_LIMIT'
    else:
        policy = PolicySpec('fixture-policy','3','evolution',{'stop_rule':'no_improvement'})
        c = fixture.new_run(fixture.cases[0],policy,fixture.protocol,'stop')
        state = CampaignState(campaign_id='stop', data={'rsi_history':[{'decision':{'decision':'ROLLBACK'}}]})
        context = c.planner.context(state)
        assert context['allowed_actions'] == ['COMPLETE','ABSTAIN']
        assert context['policy_stop_reason'] == 'POLICY_NO_IMPROVEMENT'


def test_collected_feedback_is_consumed_by_configured_cli_without_training(tmp_path, synthetic_gateway):
    from test_framework_feedback import training_fixture
    root = tmp_path/'training';root.mkdir()
    job,adapter,value,ref = training_fixture(root)
    receipt = ref('receipt.json',value); adapter.collect(job,receipt)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    fixture=EntryFixture(tmp_path)
    feedback={'roots':{k:str(v) for k,v in adapter.store.roots.items()},'work_root':str(adapter.root),
              'job':job.to_dict(),'receipt':receipt.to_dict()}
    c=fixture.new_run(fixture.cases[0],fixture.policy,fixture.protocol,'feedback',feedback=feedback)
    c.run()
    traces=list(c.planner.trace_root.glob('*/trace.json'))
    assert traces and json.loads(traces[0].read_text())['context']['feedback_signals'][0]['scope']['job_sha256']==job.compute_hash()
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_entry_model_subquota_inherits_other_batch_before_new_effect(tmp_path):
    from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy, ModelRequest
    from dataevolver.adapters.model_gateway import ModelError
    config={'profiles':[ModelProfile('p','mock','fixture','1',mock_output={'ok':True}).to_dict()],
        'policy':ModelPolicy({'agent.runtime_rsi':'p'},3).to_dict(),
        'receipt_root':str(tmp_path/'batch1'),
        'attempt_limits':{'quotas':{'text':1},'role_buckets':{'agent.runtime_rsi':'text'},
                         'ledger_roots':[str(tmp_path/'batch1'),str(tmp_path/'batch2')]}}
    def request(name): return ModelRequest(name,'agent.runtime_rsi',({'role':'user','content':'fixture'},),{'type':'object'})
    first=runtime_entry.model_gateway(config,roots={})
    first.invoke(request('first'))
    second=runtime_entry.model_gateway({**config,'receipt_root':str(tmp_path/'batch2')},roots={})
    with pytest.raises(ModelError,match='quota'):second.invoke(request('second'))
    assert not (tmp_path/'batch2/second.intent.json').exists()
    assert first.audit_usage()['model_calls']==1 and second.audit_usage()['model_calls']==0


def test_evidence_reader_refuses_incomplete_source_without_dispatch(tmp_path,synthetic_gateway):
    from dataevolver.harness.runtime_references import workload
    fixture=EntryFixture(tmp_path)
    c=fixture.new_run(fixture.cases[0],fixture.policy,fixture.protocol,'unfinished')
    with pytest.raises(ValueError,match='terminal, reconciled'):
        workload(reference(c))
    assert c.status().operator_calls==c.status().planner_calls==0
    assert not list((tmp_path/'gateways/dev-unfinished').glob('*.intent.json'))


def test_uncertain_entry_effect_stops_without_resubmitting(tmp_path, synthetic_gateway, monkeypatch):
    from dataevolver.adapters.model_gateway import ModelError
    original=runtime_entry.ModelGateway; calls=[]
    class Broken:
        def invoke(self,*a):calls.append(True);raise ModelError('UNKNOWN_EFFECT','fixture interrupted after dispatch')
        def lookup(self,*a):raise AssertionError('unconfirmed effects cannot be silently repaired')
    def gateway(**kwargs):
        g=original(**kwargs);g.factories['fixture']=Broken;return g
    monkeypatch.setattr(runtime_entry,'ModelGateway',gateway)
    fixture=EntryFixture(tmp_path)
    c=fixture.new_run(fixture.cases[0],fixture.policy,fixture.protocol,'uncertain');c.run()
    assert c.status().state.value=='ABSTAIN'
    assert c.status().terminal_reason=='RSI_PLANNER_FAILED'
    reopened=cli._existing_controller(c.layout.campaigns_root,c.goal.campaign_id);reopened.resume()
    assert len(calls)==1
    assert len(list((tmp_path/'gateways/dev-uncertain').glob('*.intent.json')))==1


@pytest.mark.parametrize('problem',['sealed','missing','drift'])
def test_entry_rejects_unusable_feedback_before_any_effect(tmp_path,synthetic_gateway,problem):
    from test_framework_feedback import training_fixture
    training=tmp_path/'training';training.mkdir()
    job,adapter,value,ref=training_fixture(training,split='sealed' if problem=='sealed' else 'development')
    receipt=ref('receipt.json',value)
    if problem!='missing':adapter.collect(job,receipt)
    if problem=='drift':(training/'evidence/receipt.json').write_text('{}')
    fixture=EntryFixture(tmp_path)
    config={'roots':{k:str(v) for k,v in adapter.store.roots.items()},'work_root':str(adapter.root),
            'job':job.to_dict(),'receipt':receipt.to_dict()}
    with pytest.raises((ValueError,FileNotFoundError)):
        fixture.new_run(fixture.cases[0],fixture.policy,fixture.protocol,'bad-feedback',feedback=config)
    assert not list((tmp_path/'gateways/dev-bad-feedback').glob('*.intent.json'))


def test_qualified_repair_uses_cli_graph_binding_and_revocation(tmp_path,monkeypatch):
    """Actual CLI/graph/sandbox, explicitly synthetic build and qualification."""
    from dataevolver.harness import runtime_references as refs, repair_qualification as repair
    from dataevolver.harness.operator_lifecycle import OperatorVersionStore
    from dataevolver.harness.operator_development import DevelopmentBudget
    from dataevolver.harness.adapters.operator_version_lifecycle import OperatorVersionMutation
    from dataevolver.harness.candidate_sandbox import LinuxPureDataRunner
    from dataevolver.harness.plan_compiler import compile_plan,CompileContext,builtin_domains
    from dataevolver.dataset.contracts import payload_sha256
    from test_repair_candidate import package_fixture
    from test_engineering_graph import repair_graph
    package_root=tmp_path/'package';package_root.mkdir()
    store,package,build=package_fixture(package_root,monkeypatch);sandbox=LinuxPureDataRunner()
    prior=dict(operator_calls=9,planner_calls=2,model_calls=4,image_experiments=0,output_bytes=100,wall_seconds=1)
    snapshot={'operator_id':repair.OPERATOR_ID,'version':'2','scope':repair.SCOPE,
        'package_sha256':package.compute_hash(),'runtime':sandbox.identity(),
        'lifecycle_total':prior,'resource_deadline_epoch':1}
    authority=object.__new__(repair.RepairQualificationAuthority);authority.read=lambda:copy.deepcopy(snapshot)
    # These test-only replacements cannot be selected by real configuration.
    monkeypatch.setattr(refs,'qualification',lambda config:authority)
    monkeypatch.setattr(refs,'candidate',lambda config:dict(plan=None,package=package,store=store,
        request_authority=None,build_authority=build,runner=sandbox))
    catalog=OperatorVersionStore(tmp_path/'versions')
    budget=DevelopmentBudget({**prior,'operator_calls':20,'output_bytes':500000,'wall_seconds':300},prior,1)
    values={'catalog_root':str(catalog.root),'expected_head':catalog.head(),'budget':budget.to_dict(),
        'output_root':str(tmp_path/'adopt-output'),'qualification':{'kind':'TEST_ONLY_SYNTHETIC'},
        'roots':{k:str(v) for k,v in store.roots.items()}}
    def execute_host(name,values,metric):
        host={'kind':'version_mutation','configuration':values};op=refs.host_workload(host)
        evaluator=tmp_path/(name+'-eval.json')
        atomic_write_json(evaluator,{'schema_version':'dataevolver.harness.evaluator.v1',
            'evaluator_id':name,'revision':'1','hard_constraints':{'artifact_valid':{'required':True}},
            'target_metrics':{metric:{'minimum':1}},'protected_metrics':{}})
        key={'input_artifact_hash':payload_sha256({}),'parent_artifact_hash':payload_sha256({}),
            'seed':0,'operator_id':op.spec.operator_id,'operator_version':op.spec.version,
            'model_config_hash':'fixture','preprocessing_hash':'fixture','evaluator_hash':file_sha256(evaluator),
            'environment_contract_hash':'fixture'}
        goal=GoalSpec('dataevolver.harness.goal.v1',name,{'initial_trial_key':key,'host_workload':host},
            op.spec.operator_id,(),BudgetSpec(1,1,0,0,600),{'allowed_operators':[op.spec.operator_id]},
            {'path':str(evaluator),'sha256':file_sha256(evaluator)})
        c=cli._controller(goal,op.output_root,create=True,idempotency_key=name);c.run()
        assert c.status().state.value=='COMPLETE';return c
    adopted=execute_host('adopt-repair',values,'operator_adopted')
    binding={'catalog_root':str(catalog.root),'expected_head':catalog.head(),
        'adoption':reference(adopted),'qualification':{'kind':'TEST_ONLY_SYNTHETIC'},'candidate':{}}
    folder=tmp_path/'graph';folder.mkdir();original,_,_=repair_graph(folder)
    contract=copy.deepcopy(original.contract);node=contract['graph']['nodes'][-1]
    node['operator_version']='2';del node['parameters']['repair_instruction']
    registry=refs.step_registry({'repair':binding})
    compiled=compile_plan({'domain':'visual_generation','contract':contract},CompileContext(registry,
        builtin_domains(),tuple(n['operator'] for n in contract['graph']['nodes']),roots=original.store.roots))
    evaluator=tmp_path/'graph-eval.json'
    atomic_write_json(evaluator,{'schema_version':'dataevolver.harness.evaluator.v1','evaluator_id':'graph',
        'revision':'1','hard_constraints':{'artifact_valid':{'required':True}},
        'target_metrics':{'completion_ratio':{'minimum':1}},'protected_metrics':{}})
    key={'input_artifact_hash':payload_sha256(compiled.contract['intent']['sources']),
        'parent_artifact_hash':payload_sha256(compiled.contract['intent']['sources']),
        'seed':0,'operator_id':'data.graph','operator_version':'1',
        'model_config_hash':payload_sha256(compiled.contract['models']),
        'preprocessing_hash':compiled.compute_hash(),'evaluator_hash':file_sha256(evaluator),
        'environment_contract_hash':'fixture'}
    goal=GoalSpec('dataevolver.harness.goal.v1','adopted-graph',{'compiled_run':compiled.to_dict(),
        'initial_trial_key':key,'artifact_roots':{k:str(v) for k,v in original.store.roots.items()},
        'step_bindings':{'repair':binding}},
        'data.graph',(),BudgetSpec(1,1,0,0,600),{'allowed_operators':['data.graph']},
        {'path':str(evaluator),'sha256':file_sha256(evaluator)},initial_operator_parameters=compiled.contract)
    consumer=cli._controller(goal,tmp_path/'campaigns',create=True,idempotency_key=goal.campaign_id)
    consumer.run();assert consumer.status().state.value=='COMPLETE'
    effects=list(consumer.layout.campaign_dir.rglob('candidate-repair-execution.json'))
    # The sandbox evidence belongs to the registered dataset store, not source code.
    effects=list(original.store.roots['data'].rglob('candidate-repair-execution.json'))
    assert effects and json.loads(effects[0].read_text())['execution']['isolation_verified'] is True
    consumer.registry.get('data.graph')
    before={p:p.read_bytes() for p in original.store.roots['data'].rglob('*.intent.json')}
    cli._existing_controller(tmp_path/'campaigns',goal.campaign_id).resume()
    assert before=={p:p.read_bytes() for p in original.store.roots['data'].rglob('*.intent.json')}
    owner=refs.adoption(reference(adopted));usage=owner.usage()
    new_budget=DevelopmentBudget({**usage,'operator_calls':20,'output_bytes':500000,'wall_seconds':300},usage,1)
    revoke={**values,'expected_head':catalog.head(),'budget':new_budget.to_dict(),
        'output_root':str(tmp_path/'revoke-output'),'adoption':reference(adopted),'reason':'Fixture withdrawal'}
    revoke.pop('qualification');execute_host('revoke-repair',revoke,'operator_revoked')
    with pytest.raises(ValueError):refs.step_registry({'repair':binding})
    atomic_write_json(tmp_path.parent/'repair-entry-acceptance.json',{
        'evidence_scope':'CPU_fixture_build_and_qualification_actual_graph_and_Linux_sandbox',
        'adoption':reference(adopted),'consumer':reference(consumer),
        'admission':json.loads((consumer.layout.campaign_dir/'runtime-admission.json').read_text()),
        'replay_matches':consumer.replay().to_dict()==consumer.status().to_dict(),
        'restore_new_model_requests':0,'revoked_version_refused':True,
        'new_research_candidates':0,'new_experiment_native_calls':0})


def native_graph_configuration(tmp_path):
    from test_engineering_graph import caption_graph
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy, ModelRequest
    from dataevolver.adapters.local_process_endpoint import LocalProcessEndpointProvider
    from dataevolver.dataset.contracts import payload_sha256
    from dataevolver.harness.plan_compiler import compile_plan, CompileContext, builtin_domains
    from dataevolver.harness.operator_sdk import builtin_step_registry
    original, _, _ = caption_graph(tmp_path)
    roots = {k: str(v) for k, v in original.store.roots.items()}
    source_config = {'profiles': [ModelProfile('old', 'mock', 'source-fixture', '1', mock_output={'ok': True}).to_dict()],
        'policy': ModelPolicy({'source': 'old'}, 1).to_dict(), 'receipt_root': str(tmp_path/'original-journal')}
    source = ModelGateway(profiles=tuple(ModelProfile.from_dict(p) for p in source_config['profiles']),
        policy=ModelPolicy.from_dict(source_config['policy']), receipt_root=Path(source_config['receipt_root']))
    request = ModelRequest('original-confirmed', 'source', ({'role': 'user', 'content': 'fixture'},), {'type': 'object'})
    source.invoke(request)
    deployment = {'origin': 'http://127.0.0.1:1', 'model': 'not-started-fixture', 'served_model': 'not-started-fixture'}
    profile = ModelProfile('transport', 'local_python', deployment['model'],
        LocalProcessEndpointProvider.profile_revision(deployment), ('json', 'vision'))
    contract = copy.deepcopy(original.contract)
    contract['models']['profiles'].append(profile.to_dict())
    contract['models']['policy']['bindings']['transport'] = 'transport'
    compiled = compile_plan({'domain': 'visual_generation', 'contract': contract}, CompileContext(
        builtin_step_registry(), builtin_domains(), ('visual.review.image', 'visual.caption.refine'), roots=roots))
    evaluator = tmp_path/'entry-evaluator.json'
    atomic_write_json(evaluator, {'schema_version': 'dataevolver.harness.evaluator.v1',
        'evaluator_id': 'configured-graph', 'revision': '1',
        'hard_constraints': {'artifact_valid': {'required': True}},
        'target_metrics': {'completion_ratio': {'minimum': 1}}, 'protected_metrics': {}})
    key = {'input_artifact_hash': payload_sha256({}), 'parent_artifact_hash': payload_sha256({}),
        'seed': 0, 'operator_id': 'data.graph', 'operator_version': '1',
        'model_config_hash': payload_sha256(contract['models']), 'preprocessing_hash': compiled.compute_hash(),
        'evaluator_hash': file_sha256(evaluator), 'environment_contract_hash': 'fixture'}
    runtime = {'process_endpoints': {'transport': {'deployment': deployment,
        'native_root': str(Path(roots['data'])/'native-transport')}}, 'source_gateways': {
        'source': {'gateway': source_config, 'contract_hash': source.contract_hash()}}}
    goal = GoalSpec('dataevolver.harness.goal.v1', 'configured-native-graph',
        {'compiled_run': compiled.to_dict(), 'artifact_roots': roots, 'graph_runtime': runtime,
         'initial_trial_key': key}, 'data.graph', (), BudgetSpec(1, 1, 0, 0, 120),
        {'allowed_operators': ['data.graph']}, {'path': str(evaluator), 'sha256': file_sha256(evaluator)},
        initial_operator_parameters=contract)
    return goal, source, request


def test_cli_graph_reconstructs_process_transport_and_readonly_source(tmp_path):
    from dataevolver.adapters.local_process_endpoint import LocalProcessEndpointProvider
    from dataevolver.adapters.model_gateway import ModelError
    from dataevolver.harness.models import OperatorCall
    goal, source, request = native_graph_configuration(tmp_path)
    before = {p: p.read_bytes() for p in source.root.rglob('*') if p.is_file()}
    c = cli._controller(goal, tmp_path/'campaigns', create=True, idempotency_key=goal.campaign_id)
    op = c.registry.get('data.graph')
    call = OperatorCall(goal.campaign_id, 'bound', 'bound', 'data.graph', '1', 0,
        str(c.layout.campaign_dir/'bound'), parameters=goal.initial_operator_parameters)
    routed = op.gateway(call)
    assert routed.recover(request).output == {'ok': True}
    with pytest.raises(ModelError, match='read.only|source'):
        routed.invoke(request)
    primary = routed._gateway('transport')
    profile, model, endpoint, _ = primary._binding('transport')
    provider = primary._provider(profile, model, endpoint)
    assert isinstance(provider, LocalProcessEndpointProvider)
    assert not provider.root.exists()  # No endpoint, health check or model submission.
    assert c.status().operator_calls == c.status().planner_calls == 0
    c.run()
    assert c.status().state.value == 'COMPLETE'
    cli._existing_controller(tmp_path/'campaigns', goal.campaign_id).resume()
    assert before == {p: p.read_bytes() for p in source.root.rglob('*') if p.is_file()}
    assert not provider.root.exists()


@pytest.mark.parametrize('problem', ['journal_escape', 'source_hash', 'model_identity', 'unknown_binding'])
def test_configured_graph_binding_refuses_before_effect(tmp_path, problem):
    goal, source, _ = native_graph_configuration(tmp_path)
    before = {p: p.read_bytes() for p in source.root.rglob('*') if p.is_file()}
    value = goal.to_dict(); runtime = value['objective']['graph_runtime']
    if problem == 'journal_escape': runtime['process_endpoints']['transport']['native_root'] = str(tmp_path/'outside')
    elif problem == 'source_hash': runtime['source_gateways']['source']['contract_hash'] = '0'*64
    elif problem == 'model_identity': runtime['process_endpoints']['transport']['deployment']['model'] = 'changed'
    else: runtime['unknown'] = {}
    with pytest.raises(ValueError):
        cli._controller(GoalSpec.from_dict(value), tmp_path/'campaigns', create=True, idempotency_key=goal.campaign_id)
    assert not (tmp_path/'campaigns'/goal.campaign_id/'state.json').exists()
    assert before == {p: p.read_bytes() for p in source.root.rglob('*') if p.is_file()}
