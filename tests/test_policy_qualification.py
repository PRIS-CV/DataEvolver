"""CPU qualification contracts; fixture scores are not native policy benefit."""
from pathlib import Path
import pytest


@pytest.mark.parametrize('change', ['revision', 'binding', 'call_limit'])
def test_gateway_contract_drift_invalidates_frozen_runtime(change, tmp_path):
    from dataevolver.adapters.codex_planner import GatewayPlannerBackend
    from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
    from dataevolver.testing.test_runtime_rsi import FixtureDomain
    from dataevolver.testing.test_harness_controller import build_controller
    old, _ = build_controller(tmp_path)
    profiles = (ModelProfile('p1','mock','fixture-model','r1'),
                ModelProfile('p2','mock','fixture-model','r2'))
    gateway = ModelGateway(profiles=profiles, policy=ModelPolicy({'agent.runtime_rsi':'p1'},3),
                           receipt_root=tmp_path/'gateway')
    backend = GatewayPlannerBackend(gateway, model_id='fixture-model')
    planner = RuntimeRSIPlanner(backend=backend,domain=FixtureDomain(),trace_root=tmp_path/'traces')
    goal = old.goal.with_updates(campaign_id='gateway-bound',planner={'kind':'runtime_rsi'})
    c = EvolutionController.create(repo_root=tmp_path,campaigns_root=tmp_path/'campaigns',
        goal=goal,registry=old.registry,planner=planner,evaluator_contract=old.evaluator.contract,
        idempotency_key='gateway-bound')
    assert c._frozen()[0]
    assert planner.config()['backend_contract_hash']==gateway.contract_hash()
    if change=='revision':
        gateway.profiles['p1']=ModelProfile('p1','mock','fixture-model','changed')
    elif change=='binding':
        gateway.policy=ModelPolicy({'agent.runtime_rsi':'p2'},3)
    else:
        gateway.policy=ModelPolicy({'agent.runtime_rsi':'p1'},2)
    assert backend.model_id=='fixture-model'
    assert not c._frozen()[0]
    assert c.status().operator_calls==c.status().planner_calls==0
    assert not list((tmp_path/'gateway').glob('*.intent.json'))


def test_policy_qualification_rejects_json_summaries():
    from dataevolver.harness.policy_qualification import PolicyQualificationAuthority
    with pytest.raises(TypeError,match='committed host policy comparison'):
        PolicyQualificationAuthority(engineering={'passed':True},independent={'passed':True})


class PolicyFixture:
    """Actual Controllers and catalog effects, deliberately synthetic scores."""
    def __init__(self, root, *, independent_gap=.8, overlap=False, gateway_backend=False):
        import json
        from dataevolver.dataset.contracts import ArtifactStore, file_sha256, payload_sha256
        from dataevolver.dataset.families import assign_families
        from dataevolver.dataset.quality import select_balanced
        from dataevolver.harness.feedback import PolicySpec
        from dataevolver.harness.catalog_rsi import CatalogRSIDomain, CatalogTrialOperator
        from dataevolver.harness.operator_sdk import builtin_step_registry
        from dataevolver.harness.plan_compiler import compile_plan, CompileContext, builtin_domains
        from dataevolver.harness.evaluator import EvaluatorContract
        from dataevolver.harness.operator_development import QualificationProtocol
        from dataevolver.harness.policy_qualification import policy_context, policy_check_spec, CHECK_IDS
        from test_framework_integration import catalog_contract
        self.root, self.cases, self.operators, self.domains, self.runs = root, [], {}, {}, []
        self.gateway_backend=gateway_backend
        self.baseline=PolicySpec('fixture-policy','1','evolution',{'candidate_limit':0})
        self.policy=PolicySpec('fixture-policy','2','evolution',{'candidate_limit':1})
        self.eval_path=root/'trial-evaluator.json'
        self.eval_path.write_text(json.dumps({'schema_version':'dataevolver.harness.evaluator.v1',
            'evaluator_id':'policy-fixture','revision':'1','hard_constraints':{'artifact_valid':{'required':True}},
            'target_metrics':{'quality':{'minimum':.8,'minimum_improvement':0}},
            'protected_metrics':{'coverage':{'max_regression':0}}}))
        self.evaluator=EvaluatorContract.from_path(self.eval_path)
        self.qual_eval=root/'qualification-evaluator.json'
        self.qual_eval.write_text(json.dumps({'schema_version':'dataevolver.harness.evaluator.v1',
            'evaluator_id':'policy-qualification-fixture','revision':'1',
            'hard_constraints':{'artifact_valid':{'required':True}},
            'target_metrics':{'qualification_passed':{'minimum':1}},'protected_metrics':{}}))
        roots={}
        for number,(name,split) in enumerate([('dev','development'),('reg','regression'),
                ('ind-a','independent'),('ind-b','independent'),('sealed','heldout'),('successor','development')]):
            folder=root/name;folder.mkdir()
            contract, local, rows=catalog_contract(folder)
            for index,row in enumerate(rows):
                row.update(label_id='fixture',source_family=f'{"dev" if overlap and name=="ind-a" else name}-{index}',
                    pixel_sha256=f'{number*16+index:064x}',dhash=f'{number*16+index:016x}')
            assigned=assign_families(rows,0)
            selected=select_balanced(assigned,{'a':{'fixture':3}},ranking='hash',seed=1,min_clarity=0)
            ids={r['id'] for r in selected}
            low=.79 if split=='independent' and independent_gap<.05 else .1
            high=low+independent_gap if split=='independent' else .9
            for row in rows:
                row['quality']['clarity_proxy']=.9 if name=='reg' else low if row['id'] in ids else high
            source=local['raw']/'source.json'
            source.write_text(json.dumps({'rows':rows,'sources':[{'rows_inspected':len(rows)}],
                'fixture_nonce':name,'heldout_secret':'DO_NOT_FEED_HELDOUT_ANSWER' if name=='sealed' else ''}))
            ref=contract['graph']['nodes'][0]['inputs']['source']['artifact']
            ref.update(store=name,sha256=file_sha256(source));contract['intent']['sources'][0]=dict(ref)
            contract['stores']={name:'FIXTURE_RAW','dataset':'FIXTURE_DATASET'}
            local={name:local['raw'],'dataset':local['dataset']};roots[name]=local[name]
            node=contract['graph']['nodes'][2]
            node.update(operator='images.select_balanced',parameters={
                'quotas':{'a':{'fixture':3}},'ranking':'hash','seed':1,'min_clarity':0.0})
            contract['quality_policy']['allowed_operations']=[n['operator'] for n in contract['graph']['nodes']]
            registry=builtin_step_registry()
            compiled=compile_plan({'domain':'catalog','contract':contract},CompileContext(
                registry,builtin_domains(),tuple(n['operator'] for n in contract['graph']['nodes']),roots=local))
            key={'input_artifact_hash':ref['sha256'],'parent_artifact_hash':ref['sha256'],'seed':1,
                'operator_id':'data.catalog_trial','operator_version':'1','model_config_hash':'fixture',
                'preprocessing_hash':compiled.compute_hash(),'evaluator_hash':file_sha256(self.eval_path),
                'environment_contract_hash':'fixture'}
            domain=CatalogRSIDomain(configuration={'compiled_run':compiled.to_dict(),'source':dict(ref),
                'baseline_ranking':'hash','allowed_rankings':['hash','clarity'],'quotas':{'a':{'fixture':3}},
                'target_quality':.8,'matched_trial_key':key,'maximum_candidates':1},registry=registry,roots=local)
            self.domains[name]=domain;self.operators[name]=CatalogTrialOperator(domain=domain,output_root=root/'controllers')
            if name!='successor':self.cases.append({'family_id':name,'split':split,'artifact':dict(ref)})
        self.store=ArtifactStore(roots)
        prototype=self.new_run(self.cases[0],self.baseline,None,'prototype')
        stages={phase:{'validator_operator_id':'evolution.policy_'+phase,
            'validator_spec_sha256':policy_check_spec(phase,root/'checks'/phase).sha256(),
            'evaluator_sha256':file_sha256(self.qual_eval),'check_ids':list(CHECK_IDS)}
            for phase in ('engineering','independent')}
        self.protocol=QualificationProtocol('policy-fixture',tuple(self.cases),stages,
                                           payload_sha256(policy_context(prototype)))

    def new_run(self, source, policy, protocol, suffix, *, model_id='fixture-policy-agent',target_quality=None):
        import json
        from dataevolver.harness.controller import EvolutionController
        from dataevolver.harness.models import GoalSpec,BudgetSpec
        from dataevolver.harness.operators import OperatorRegistry
        from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
        from dataevolver.testing.test_runtime_rsi import action
        fixture=self
        class Backend:
            backend_id='fixture-policy-backend'
            def __init__(self):self.model_id=model_id;self.contexts=[]
            def complete_json(self,messages,*,schema):
                context=json.loads(messages[-1]['content']);self.contexts.append(context)
                if ('RUN_EXPERIMENT' in context['allowed_actions'] and not context['history']
                        and context['current_metrics']['quality']<context['target_quality']):
                    return json.dumps(action('RUN_EXPERIMENT',layer='R',patch='sampling_recipe',
                        parameters={'ranking':'clarity'},evidence=context['available_evidence_ids']))
                return json.dumps(action('COMPLETE',evidence=context['available_evidence_ids']))
        name=source['family_id'];op=self.operators[name];domain=self.domains[name]
        if target_quality is not None:
            from dataevolver.harness.catalog_rsi import CatalogRSIDomain, CatalogTrialOperator
            domain=CatalogRSIDomain(configuration={**domain.config(),'target_quality':target_quality},
                                    registry=domain.registry,roots=domain.roots)
            op=CatalogTrialOperator(domain=domain,output_root=self.root/'controllers')
        campaign=name+'-'+suffix
        binding={'protocol_sha256':protocol.compute_hash() if protocol else '0'*64,
                 'stage':'engineering' if source['split'] in {'development','regression'} else 'independent',
                 'source':source,'policy_sha256':policy.compute_hash()}
        goal=GoalSpec('dataevolver.harness.goal.v1',campaign,{'kind':'policy-fixture',
            'initial_trial_key':domain.value['matched_trial_key'],'policy_comparison':binding},
            op.spec.operator_id,(),BudgetSpec(4,5,1,0,600,4),{'allowed_operators':[op.spec.operator_id]},
            {'path':str(self.eval_path),'sha256':self.evaluator.sha256},
            initial_operator_parameters={'ranking':'hash','experience_id':''},planner={'kind':'runtime_rsi'})
        registry=OperatorRegistry();registry.register(op)
        backend=Backend();planner=RuntimeRSIPlanner(backend=backend,domain=domain,
            trace_root=self.root/'traces'/campaign,policy=policy)
        if self.gateway_backend:
            from dataevolver.adapters.codex_planner import GatewayPlannerBackend
            from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy
            from dataevolver.adapters.model_gateway import ModelGateway
            class Provider:
                def lookup(self,request,request_hash):
                    raise AssertionError('qualification must not repair a missing cached model receipt')
                def invoke(self,request,request_hash):
                    return {'model':model_id,'output':json.loads(backend.complete_json(
                        request.messages,schema=request.response_schema)),
                        'usage':{'input_tokens':1,'output_tokens':1},'finish_reason':'stop'}
            gateway=ModelGateway(profiles=(ModelProfile('fixture','mock',model_id,'fixture-1'),),
                policy=ModelPolicy({'agent.runtime_rsi':'fixture'},8),
                receipt_root=self.root/'gateways'/campaign,providers={'fixture':Provider})
            planner=RuntimeRSIPlanner(backend=GatewayPlannerBackend(gateway,model_id=model_id),
                domain=domain,trace_root=self.root/'traces'/campaign,policy=policy)
        package=Path(__file__).resolve().parents[1]/'src/dataevolver'
        c=EvolutionController.create(repo_root=self.root,campaigns_root=self.root/'controllers',goal=goal,
            registry=registry,planner=planner,evaluator_contract=self.evaluator,
            execution_paths=tuple(sorted(package.rglob('*.py')))+(Path(__file__).resolve(),),idempotency_key=campaign)
        return c

    def phase(self, phase, prior):
        from dataevolver.harness.policy_qualification import PolicyComparisonChecks
        from dataevolver.harness.operator_development import DevelopmentBudget,USAGE_KEYS
        import time
        pairs=[]
        for source in self.cases:
            if source['split'] not in ({'development','regression'} if phase=='engineering' else {'independent'}):continue
            a=self.new_run(source,self.baseline,self.protocol,'baseline')
            b=self.new_run(source,self.policy,self.protocol,'candidate')
            a.run();b.run();self.runs.extend([a,b]);pairs.append(((a,self.operators[source['family_id']]),(b,self.operators[source['family_id']])))
        used=dict(prior)
        used['operator_calls']+=sum(c.status().operator_calls for pair in pairs for c,_ in pair)
        used['planner_calls']+=sum(c.status().planner_calls for pair in pairs for c,_ in pair)
        model_calls=sum(c.planner.backend.gateway.audit_usage()['model_calls'] for pair in pairs for c,_ in pair
                        if hasattr(c.planner.backend,'gateway'))
        used['model_calls']+=model_calls
        used['output_bytes']+=sum(p.stat().st_size for pair in pairs for c,_ in pair
            for p in c.layout.campaign_dir.rglob('*') if p.is_file())
        used['wall_seconds']+=600*len(pairs)*2  # Deliberately conservative source-run reservation.
        limits={k:max(used[k],10000000 if k=='output_bytes' else 10000) for k in USAGE_KEYS}
        limits['model_calls']=used['model_calls'];limits['image_experiments']=0
        budget=DevelopmentBudget(limits,used,self.deadline)
        op=PolicyComparisonChecks(baseline_policy=self.baseline,policy=self.policy,protocol=self.protocol,
            store=self.store,pairs=pairs,phase=phase,budget=budget,output_root=self.root/'checks'/phase)
        c=self.execute(op,phase,self.qual_eval,op.parameters())
        return c,op

    def execute(self,op,name,evaluator_path,parameters):
        from dataevolver.harness.controller import EvolutionController
        from dataevolver.harness.models import GoalSpec,BudgetSpec
        from dataevolver.harness.operators import OperatorRegistry
        from dataevolver.harness.planner import DeterministicPlanner
        from dataevolver.harness.evaluator import EvaluatorContract
        from dataevolver.dataset.contracts import file_sha256,payload_sha256
        registry=OperatorRegistry();registry.register(op)
        key={'input_artifact_hash':payload_sha256(parameters),'parent_artifact_hash':payload_sha256(parameters),
            'seed':0,'operator_id':op.spec.operator_id,'operator_version':op.spec.version,
            'model_config_hash':'fixture','preprocessing_hash':'fixture',
            'evaluator_hash':file_sha256(evaluator_path),'environment_contract_hash':'fixture'}
        metric='qualification_passed' if name.startswith(('engineering','independent')) else 'operator_'+('adopted' if name=='adopt' else 'revoked')
        goal=GoalSpec('dataevolver.harness.goal.v1','qual-'+name,
            {'initial_trial_key':key,'target_metrics':{metric:1}},op.spec.operator_id,(),
            BudgetSpec(1,1,0,0,600),{'allowed_operators':[op.spec.operator_id]},
            {'path':str(evaluator_path),'sha256':file_sha256(evaluator_path)},initial_operator_parameters=parameters)
        package=Path(__file__).resolve().parents[1]/'src/dataevolver'
        c=EvolutionController.create(repo_root=self.root,campaigns_root=op.output_root,goal=goal,
            registry=registry,planner=DeterministicPlanner(()),evaluator_contract=EvaluatorContract.from_path(evaluator_path),
            execution_paths=tuple(sorted(package.rglob('*.py')))+(Path(__file__).resolve(),),idempotency_key='qual-'+name)
        c.run()
        return c

    def qualify(self):
        import time
        from dataevolver.harness.operator_development import USAGE_KEYS
        from dataevolver.harness.artifact_operator import CommittedWorkloadAuthority
        from dataevolver.harness.policy_qualification import PolicyQualificationAuthority
        self.deadline=time.time()+10000
        self.eng,self.eng_op=self.phase('engineering',{k:0 for k in USAGE_KEYS})
        assert self.eng.status().state.value=='COMPLETE',self.eng.status().to_dict()
        self.eng_reader=CommittedWorkloadAuthority(self.eng,self.eng_op)
        self.ind,self.ind_op=self.phase('independent',self.eng_reader.read()['report']['lifecycle_total'])
        self.ind_reader=CommittedWorkloadAuthority(self.ind,self.ind_op)
        self.authority=PolicyQualificationAuthority(engineering=self.eng_reader,independent=self.ind_reader)
        return self.authority


def test_real_controller_policy_comparisons_produce_exact_snapshot(tmp_path):
    from dataevolver.harness.policy_qualification import SCOPE
    fixture=PolicyFixture(tmp_path);authority=fixture.qualify();snapshot=authority.read()
    assert snapshot['policy_sha256']==fixture.policy.compute_hash()
    assert snapshot['scope']==SCOPE and snapshot['production_adoption_allowed'] is False
    assert snapshot['model_authority'] is False and snapshot['training_benefit_verified'] is False
    assert len(snapshot['qualification_receipts'])==2
    assert all('DO_NOT_FEED_HELDOUT_ANSWER' not in str(context) for c in fixture.runs
               for context in c.planner.backend.contexts)
    assert fixture.eng_reader.read()['report']['check_results']['regression_preserved']
    assert fixture.ind_reader.read()['report']['mean_delta']>=.05


def test_policy_independent_5pp_gate_is_not_relaxed(tmp_path):
    fixture=PolicyFixture(tmp_path,independent_gap=.049);fixture.qualify()
    assert fixture.ind.status().state.value=='ABSTAIN'
    from dataevolver.harness.models import OperatorCall
    import json
    planned=next(e for e in fixture.ind.store.events.read() if e.event_type=='ACTION_PLANNED')
    call=OperatorCall.from_dict(planned.payload['operator_call'])
    report=json.loads(fixture.ind_op.path(call).read_text())['report']
    assert report['mean_delta']==pytest.approx(.049)
    assert report['check_results']['independent_delta'] is False
    with pytest.raises(ValueError):fixture.authority.read()


def test_policy_qualification_rejects_actual_family_overlap(tmp_path):
    fixture=PolicyFixture(tmp_path,overlap=True);fixture.qualify()
    with pytest.raises(ValueError,match='actual source families or media overlap'):
        fixture.authority.read()


def test_policy_adoption_new_frozen_consumer_and_revocation_use_existing_lifecycle(tmp_path):
    import json
    from dataevolver.harness.feedback import PolicyResolver,PolicySpec
    from dataevolver.harness.operator_lifecycle import OperatorVersionStore,CommittedOperatorAdoptionAuthority
    from dataevolver.harness.adapters.operator_version_lifecycle import OperatorVersionMutation
    from dataevolver.harness.operator_development import DevelopmentBudget
    from dataevolver.dataset.contracts import payload_sha256
    from dataevolver.harness.policy_qualification import SCOPE
    fixture=PolicyFixture(tmp_path);authority=fixture.qualify();snapshot=authority.read()
    catalog=OperatorVersionStore(tmp_path/'versions')
    def budget(prior):
        limits={k:v+(1000000 if k=='output_bytes' else 1000) for k,v in prior.items()}
        for key in ('model_calls','planner_calls','image_experiments'):limits[key]=prior[key]
        return DevelopmentBudget(limits,prior,fixture.deadline)
    def evaluation(name,metric):
        path=tmp_path/(name+'-evaluator.json')
        path.write_text(json.dumps({'schema_version':'dataevolver.harness.evaluator.v1',
            'evaluator_id':name,'revision':'1','hard_constraints':{'artifact_valid':{'required':True}},
            'target_metrics':{metric:{'minimum':1}},'protected_metrics':{}}))
        return path
    adopt=OperatorVersionMutation(catalog=catalog,expected_head=catalog.head(),
        budget=budget(snapshot['lifecycle_total']),output_root=tmp_path/'adopt',
        store=fixture.store,qualification_authority=authority)
    parent=fixture.execute(adopt,'adopt',evaluation('adopt','operator_adopted'),{})
    assert parent.status().state.value=='COMPLETE'
    planned=next(e for e in parent.store.events.read() if e.event_type=='ACTION_PLANNED')
    adoption=CommittedOperatorAdoptionAuthority(parent,adopt,planned.payload['call_id'])
    event=adoption.read();assert event['snapshot']['policy_sha256']==fixture.policy.compute_hash()
    resolver=PolicyResolver(catalog,authority)
    with pytest.raises(ValueError,match='actual frozen consumer'):
        resolver.resolve(event,scope=SCOPE,expected_head=catalog.head(),policy=fixture.policy.to_dict())
    source={'family_id':'successor','split':'development',
            'artifact':fixture.domains['successor'].catalog_ref.to_dict()}
    consumer=fixture.new_run(source,fixture.policy,fixture.protocol,'consumer')
    resolved=resolver.resolve(event,scope=SCOPE,expected_head=catalog.head(),
                              policy=fixture.policy.to_dict(),controller=consumer)
    assert resolved==fixture.policy and consumer.status().operator_calls==0
    wrong=PolicySpec(fixture.policy.policy_id,fixture.policy.version,fixture.policy.role,{'candidate_limit':2})
    with pytest.raises(ValueError,match='policy version differs'):
        resolver.resolve(event,scope=SCOPE,expected_head=catalog.head(),policy=wrong.to_dict(),controller=consumer)
    other=fixture.new_run(source,fixture.policy,fixture.protocol,'other-model',model_id='different-fixture-model')
    with pytest.raises(ValueError,match='qualified context'):
        resolver.resolve(event,scope=SCOPE,expected_head=catalog.head(),policy=fixture.policy.to_dict(),controller=other)
    assert other.status().operator_calls==0
    threshold=fixture.new_run(source,fixture.policy,fixture.protocol,'different-threshold',target_quality=.81)
    with pytest.raises(ValueError,match='qualified context'):
        resolver.resolve(event,scope=SCOPE,expected_head=catalog.head(),policy=fixture.policy.to_dict(),controller=threshold)
    assert threshold.status().operator_calls==0
    consumer.run();assert consumer.status().state.value=='COMPLETE'
    assert consumer.status().data['rsi_history'][0]['decision']['decision']=='ADOPT_CANDIDATE'
    before=consumer.layout.events_path.read_bytes();consumer.run()
    assert consumer.layout.events_path.read_bytes()==before
    with pytest.raises(ValueError,match='active batch'):
        resolver.resolve(event,scope=SCOPE,expected_head=catalog.head(),policy=fixture.policy.to_dict(),controller=consumer)
    revoked=OperatorVersionMutation(catalog=catalog,expected_head=catalog.head(),budget=budget(adoption.usage()),
        output_root=tmp_path/'revoke',adoption=adoption,reason='Fixture withdrawal, preserve history')
    stop=fixture.execute(revoked,'revoke',evaluation('revoke','operator_revoked'),{})
    assert stop.status().state.value=='COMPLETE'
    assert len(catalog.events())==2
    next_consumer=fixture.new_run(source,fixture.policy,fixture.protocol,'after-revoke')
    with pytest.raises(ValueError,match='revoked'):
        resolver.resolve(event,scope=SCOPE,expected_head=catalog.head(),policy=fixture.policy.to_dict(),controller=next_consumer)
    assert next_consumer.status().operator_calls==next_consumer.status().planner_calls==0


def test_gateway_policy_proofs_are_readonly_and_failed_attempt_costs_cannot_reset(tmp_path):
    import json
    from dataevolver.harness.policy_qualification import _planner_proofs,PolicyComparisonChecks,PolicyQualificationAuthority
    from dataevolver.harness.artifact_operator import CommittedWorkloadAuthority
    from dataevolver.harness.operator_development import DevelopmentBudget
    fixture=PolicyFixture(tmp_path,gateway_backend=True)
    authority=fixture.qualify();snapshot=authority.read()
    source=next(c for c in fixture.runs if c.status().planner_calls)
    gateway=source.planner.backend.gateway
    before={str(p):p.read_bytes() for p in (tmp_path/'gateways').rglob('*.json')}
    for c in fixture.runs:
        c.planner.backend.gateway.invoke=lambda *_:pytest.fail('reader dispatched a new model request')
    assert authority.read()==snapshot
    assert {str(p):p.read_bytes() for p in (tmp_path/'gateways').rglob('*.json')}==before
    assert fixture.eng_reader.read()['report']['source_cost_lower_bound']['model_calls']>0
    assert fixture.eng_reader.read()['report']['source_cost_lower_bound']['wall_seconds']>0
    # A carried-forward engineering total cannot also pay for the independent source runs.
    old=fixture.ind_op.budget
    eng_total=fixture.eng_reader.read()['report']['lifecycle_total']
    lower=fixture.ind_reader.read()['report']['source_cost_lower_bound']
    reset=DevelopmentBudget(old.limits,{k:max(v,lower.get(k,0)) for k,v in eng_total.items()},old.resource_deadline_epoch)
    op=PolicyComparisonChecks(baseline_policy=fixture.baseline,policy=fixture.policy,protocol=fixture.protocol,
        store=fixture.store,pairs=fixture.ind_op.pairs,phase='independent',budget=reset,
        output_root=fixture.ind_op.output_root)
    parent=fixture.execute(op,'independent-reset',fixture.qual_eval,op.parameters())
    assert parent.status().state.value=='COMPLETE'
    reset_reader=PolicyQualificationAuthority(engineering=fixture.eng_reader,independent=CommittedWorkloadAuthority(parent,op))
    with pytest.raises(ValueError,match='reset costs'):
        reset_reader.read()
    receipt=next(gateway.root.glob('*.receipt.json'))
    original=receipt.read_bytes();receipt.unlink()
    with pytest.raises(ValueError,match='already confirmed'):
        _planner_proofs(source)
    assert not receipt.exists()
    receipt.write_bytes(original)
    assert authority.read()==snapshot
