"""Read-only catalog policy comparisons in the existing qualification lifecycle.

The host supplies frozen Controllers, never candidate-authored result summaries.
The scope measures catalog quality proxies; it grants no production, model,
training-benefit or general policy-improvement authority.
"""
import copy
import json
import math
import time
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, canonical_json, file_sha256, payload_sha256
from .artifact_operator import ArtifactWorkloadOperator, CommittedWorkloadAuthority
from .catalog_knowledge import read_trials
from .feedback import PolicySpec
from .models import CampaignState, EvidenceStatus, ExpectedOutputSpec, OperatorSpec, StageResult
from .experiments import CandidateDecision
from .operator_development import DevelopmentBudget, QualificationProtocol
from .runtime_rsi import RuntimeRSIPlanner

SCOPE = 'catalog_policy.v1'
CHECK_IDS = ('policy_binding','source_isolation','frozen_gates','regression_preserved',
             'behavior_changed','independent_delta')


def policy_check_spec(phase, output_root):
    if phase not in {'engineering', 'independent'}:
        raise ValueError('unknown policy qualification stage')
    return OperatorSpec('evolution.policy_'+phase, '1', retry_safe=True,
        source_path=str(Path(__file__)), code_revision=file_sha256(Path(__file__)),
        allowed_write_roots=(str(Path(output_root).resolve()),),
        expected_output_specs=(ExpectedOutputSpec('workload.json'),))


def policy_context(controller):
    """Common comparator settings; exact per-case domain binding is checked too."""
    c = controller
    from .controller import EvolutionController
    from .catalog_rsi import CatalogRSIDomain
    import inspect
    if not isinstance(c.planner, RuntimeRSIPlanner):
        raise TypeError('policy qualification requires the actual runtime planner')
    if not isinstance(c, EvolutionController):
        raise TypeError('policy qualification requires the actual Controller')
    from . import runtime_rsi
    required={Path(runtime_rsi.__file__).resolve(),Path(inspect.getsourcefile(EvolutionController)).resolve(),
              Path(inspect.getsourcefile(type(c.evaluator))).resolve()}
    if not required.issubset(set(c.contract_guard.execution_paths)):
        raise ValueError('policy execution did not freeze its runtime implementation')
    if type(c.planner.domain.domain) is not CatalogRSIDomain:
        raise TypeError('policy context is limited to the original catalog domain')
    config = c.planner.config()
    domain=copy.deepcopy(config['domain'])
    source=ArtifactRef.from_dict(domain['source']).to_dict()
    def normalize(value):
        if isinstance(value,dict):
            if value==source:return {'case_source':True}
            return {k:normalize(v) for k,v in value.items()}
        if isinstance(value,list):return [normalize(v) for v in value]
        return value
    domain=normalize(domain)
    key=domain.get('matched_trial_key',{})
    for name in ('input_artifact_hash','parent_artifact_hash','preprocessing_hash'):
        key.pop(name,None)
    compiled=domain['compiled_run']['contract']
    compiled['intent'].pop('intent_id',None)
    stores=compiled['stores']
    stores['CASE_SOURCE']=stores.pop(source['store'])
    objective = copy.deepcopy(c.goal.objective)
    for key in ('initial_trial_key', 'source_family', 'policy_comparison'):
        objective.pop(key, None)
    frozen = json.loads(c.layout.frozen_contract_path.read_text())
    goal_planner = c.goal.planner
    if goal_planner.get('runtime', {}).get('domain_kind') == 'catalog':
        # The entry repeats policy/source configuration to reconstruct a run.
        # Compare its effective request settings here; exact policies and each
        # actual source/compiled graph are checked separately below. Receipt
        # directories and adoption references are consumer provenance, not a
        # new model, source, budget or evaluator authority.
        backend = c.planner.backend
        goal_planner = {'kind': 'runtime_rsi', 'domain_kind': 'catalog',
            'planner_role': backend.role, 'max_tokens': backend.max_tokens,
            'timeout_seconds': backend.timeout_seconds,
            'review_roles': c.goal.planner['runtime'].get('review_roles', [])}
    return {'objective': objective, 'budgets': c.goal.budgets.to_dict(),
        'initial_operator_id': c.goal.initial_operator_id,
        'initial_operator_parameters': c.goal.initial_operator_parameters,
        'route_candidates': list(c.goal.route_candidates), 'interventions': list(c.goal.interventions),
        'goal_planner': goal_planner,
        'catalog_scope':domain,
        'resource_policy': c.goal.resource_policy, 'evaluator_sha256': c.evaluator.contract.sha256,
        'backend_id': config['backend_id'], 'model_id': config['model_id'],
        'backend_contract_hash': config.get('backend_contract_hash'),
        'system_prompt_sha256': payload_sha256(config['system_prompt']),
        'schema_sha256': payload_sha256(config['schema']),
        'harness_code_inventory_hash': frozen['harness_code_inventory_hash'],
        'promotion_policy_hash': frozen['promotion_policy_hash']}


def _origin(controller, operator):
    return {'campaign_id': controller.goal.campaign_id,
        'frozen_sha256': file_sha256(controller.layout.frozen_contract_path),
        'operator_spec_sha256': operator.spec.sha256()}


def _model_attempt(gateway, request):
    if hasattr(gateway,'_gateway'):gateway=gateway._gateway(request.role)
    if gateway is None or any(not (gateway.root/(request.request_id+suffix)).is_file()
                             for suffix in ('.intent.json','.receipt.json')):
        raise ValueError('policy model effect is not already confirmed; no repair is allowed')
    value=json.loads((gateway.root/(request.request_id+'.intent.json')).read_text())
    if type(value.get('attempts')) is not int or value['attempts']<1:
        raise ValueError('policy source model attempts are unknown')
    return {'ledger_root':str(gateway.root),'request_id':request.request_id,'attempts':value['attempts']}


def _planner_proofs(controller):
    """Bind each actual planning context/decision and any existing model receipt."""
    from dataevolver.adapters.model_contracts import ModelRequest, ModelReceipt
    from .runtime_rsi import ACTION_SCHEMA, SYSTEM_PROMPT, validate_decision
    from .models import canonical_json as planner_json
    c = controller
    proofs, previous = [], None
    for event in c.store.events.read():
        if event.event_type == 'RSI_DECISION_RECORDED':
            if previous is None:
                raise ValueError('policy decision has no prior Controller state')
            state = CampaignState.from_dict(previous)
            path = c.planner.trace_root/f'p{state.planner_calls:03d}'/'trace.json'
            trace = json.loads(path.read_text())
            context = c.planner.context(state)
            if (trace['outcome']!='VALID_DECISION' or trace['context']!=context
                    or trace['context_hash']!=payload_sha256({'config':c.planner.config(),'context':context})
                    or trace['decision']!=event.payload['state_after']['data']['rsi_pending_decision']):
                raise ValueError('policy planning trace differs from the committed Controller decision')
            proof = {'trace_sha256':file_sha256(path),'event_sequence':event.sequence}
            gateway = getattr(c.planner.backend,'gateway',None)
            if gateway is not None:
                model = trace.get('model_call',{})
                request = ModelRequest.from_dict(model['request'])
                if (request.role!='agent.runtime_rsi' or list(request.messages)!=[{'role':'system','content':SYSTEM_PROMPT},
                        {'role':'user','content':planner_json(context)}] or request.response_schema!=ACTION_SCHEMA
                        ):
                    raise ValueError('policy model receipt/context is absent; readers cannot repair or dispatch')
                attempt=_model_attempt(gateway,request)
                receipt = gateway.recover(request)
                if (receipt is None or receipt.model!=c.planner.backend.model_id
                        or receipt.without_reuse_hash()!=model['receipt_hash']
                        or ModelReceipt.from_dict(model['receipt']).without_reuse_hash()!=receipt.without_reuse_hash()
                        or validate_decision(receipt.output,context)!=trace['decision']):
                    raise ValueError('policy planner lost its exact model completion')
                proof.update(request_sha256=request.compute_hash(),receipt_sha256=receipt.without_reuse_hash(),
                             usage=receipt.usage,backend=receipt.backend,model=receipt.model,revision=receipt.revision,
                             model_attempt=attempt)
            proofs.append(proof)
        previous = event.payload.get('state_after',previous)
    if len(proofs)!=c.status().planner_calls:
        raise ValueError('policy planner failures or missing decisions cannot qualify')
    return proofs


class PolicyComparisonChecks(ArtifactWorkloadOperator):
    """Compare already executed policy runs, without executing either run again."""
    def __init__(self, *, baseline_policy, policy, protocol, store, pairs, phase, budget, output_root):
        self.baseline = PolicySpec.from_dict(baseline_policy.to_dict())
        self.policy = PolicySpec.from_dict(policy.to_dict())
        self.protocol = QualificationProtocol.from_dict(protocol.to_dict())
        self.budget = DevelopmentBudget.from_dict(budget.to_dict())
        reservation={key:0 for key in self.budget.prior_usage}
        reservation.update(operator_calls=1,wall_seconds=30)
        self.budget.check_usage((reservation,))
        if (self.baseline.policy_id != self.policy.policy_id or self.baseline.role != self.policy.role
                or self.baseline.version == self.policy.version or self.baseline.config == self.policy.config):
            raise ValueError('comparison needs distinct versions/configurations of the same policy role')
        self.store, self.pairs, self.phase = store, tuple(pairs), phase
        splits = {'development', 'regression'} if phase == 'engineering' else {'independent'}
        self.cases = [s for s in self.protocol.sources if s['split'] in splits]
        if phase not in {'engineering', 'independent'} or len(self.pairs) != len(self.cases):
            raise ValueError('policy comparisons must cover the exact frozen qualification cohort')
        for source in self.cases:
            store.resolve(ArtifactRef.from_dict(source['artifact']))
        destination=Path(output_root).resolve()
        for pair in self.pairs:
            for controller, operator in pair:
                if destination.is_relative_to(controller.layout.campaign_dir.resolve()):
                    raise ValueError('policy comparison cannot write inside source evidence')
        self.origins = [[_origin(c, op) for c, op in pair] for pair in self.pairs]
        if any(len(pair) != 2 for pair in self.pairs):
            raise ValueError('each policy source needs one baseline and one changed run')
        self.binding = {'baseline_policy_sha256': self.baseline.compute_hash(),
            'policy_sha256': self.policy.compute_hash(), 'protocol_sha256': self.protocol.compute_hash(),
            'origins_sha256': payload_sha256(self.origins), 'budget_sha256': self.budget.compute_hash()}
        super().__init__(operator_id='evolution.policy_'+phase, output_root=output_root,
            configuration={**self.binding, 'phase': phase, 'scope': SCOPE},
            parameter_schema={'type':'object','additionalProperties':False,
                'properties':{k:{'type':'string','pattern':'^[a-f0-9]{64}$'} for k in self.binding},
                'required':list(self.binding)})
        self.spec = policy_check_spec(phase, output_root)
        self._declare_contract()
        if self.protocol.stages[phase]['validator_spec_sha256'] != self.spec.sha256():
            raise ValueError('policy checker differs from the frozen protocol')
        if self.protocol.stages[phase]['validator_operator_id'] != self.spec.operator_id:
            raise ValueError('policy checker ID differs from the frozen protocol')
        if set(self.protocol.stages[phase]['check_ids'])!=set(CHECK_IDS):
            raise ValueError('policy protocol cannot omit or replace required checks')

    def parameters(self):
        return copy.deepcopy(self.binding)

    def path(self, call):
        current={'baseline_policy_sha256':self.baseline.compute_hash(),'policy_sha256':self.policy.compute_hash(),
            'protocol_sha256':self.protocol.compute_hash(),'origins_sha256':payload_sha256(self.origins),
            'budget_sha256':self.budget.compute_hash()}
        if current!=self.binding or call.parameters != self.binding:
            raise ValueError('policy comparison call binding changed')
        return super().path(call)

    def _read_run(self, controller, operator, source, policy):
        c = controller
        from datetime import datetime
        from .catalog_rsi import CatalogTrialOperator
        if type(operator) is not CatalogTrialOperator:
            raise TypeError('this policy scope requires the original host catalog trial')
        if c.planner.domain.config()!=operator.domain.config():
            raise ValueError('policy planner domain differs from its actual catalog trial')
        if any(n['kind']!='data' for n in operator.domain.compiled.contract['graph']['nodes']):
            raise ValueError('catalog policy comparison must retain its pre-existing pure-data selection graph')
        if not isinstance(c.planner, RuntimeRSIPlanner) or c.planner.policy is None:
            raise ValueError('the policy must actually participate in the runtime planner')
        if c.planner.policy.to_dict() != policy.to_dict():
            raise ValueError('runtime policy differs from the frozen comparison version')
        expected = {'protocol_sha256': self.protocol.compute_hash(), 'stage': self.phase,
                    'source': source, 'policy_sha256': policy.compute_hash()}
        if c.goal.objective.get('policy_comparison') != expected:
            raise ValueError('policy comparison was not prebound in the frozen Goal')
        context = policy_context(c)
        if payload_sha256(context) != self.protocol.matched_context_sha256:
            raise ValueError('policy comparison model, budget, evaluator or runtime context changed')
        # Cache repair is not qualification. Require every separate catalog review
        # to already have both its immutable intent and local completion receipt.
        from dataevolver.adapters.model_contracts import ModelRequest
        source_state=c.status()
        if (not c._frozen()[0] or c.replay().to_dict()!=source_state.to_dict()
                or source_state.state.value not in {'COMPLETE','ABSTAIN'} or source_state.data.get('pending_call')):
            raise ValueError('policy source Controller must be terminal, frozen and replayable')
        review_attempts=[]
        for event in c.store.events.read():
            if event.event_type!='ACTION_COMMITTED' or event.payload['receipt']['operator_id']!=operator.spec.operator_id:
                continue
            from .models import OperatorCall
            planned=[e for e in c.store.events.read() if e.event_type=='ACTION_PLANNED'
                     and e.payload['call_id']==event.payload['receipt']['call_id']]
            if len(planned)!=1:raise ValueError('policy source trial has no unique intent')
            call=OperatorCall.from_dict(planned[0].payload['operator_call'])
            raw=json.loads(operator.path(call).read_text())['report']
            for review in raw.get('reviews',[]):
                request=ModelRequest.from_dict(review['request'])
                review_attempts.append(_model_attempt(operator.gateway,request))
        rows = read_trials(c, operator, include_receipts=True)
        if c.registry.get(operator.spec.operator_id).spec.sha256()!=operator.spec.sha256():
            raise ValueError('policy trial operator differs from its frozen registration')
        state = c.status()
        def committed_result(row):
            receipt=row['receipt']
            # The Controller binds outer artifacts/key/stdout when committing a receipt.
            return StageResult.from_dict({**receipt['stage_result'],
                'output_artifacts':receipt['output_artifacts'],
                'matched_trial_key':row['call']['matched_trial_key'],'stdout_tail':receipt['stdout_tail']})
        kept = [row for row in rows if committed_result(row).to_dict()==state.data.get('last_stage_result')]
        if len(kept) != 1:
            raise ValueError('the exact kept trial is absent or ambiguous')
        item = kept[0]; report = item['report']; receipt = item['receipt']
        if (receipt['status'] != 'SUCCESS' or receipt['return_code'] != 0 or report.get('graph_failed')
                or report.get('source_sha256') != source['artifact']['sha256']
                or not report.get('source_families') or not report.get('rows')
                or report.get('rows') != report.get('unique_families')
                or any(r['output']['status'] != 'pass' for r in report.get('reviews', []))):
            raise ValueError('kept policy trial lost its source, protections or successful effect')
        result = committed_result(item)
        evaluation = c.evaluator.evaluate(result, effect_status=EvidenceStatus.VERIFIED)
        if (not evaluation.all_required_verified() or any(v.value is not True for v in evaluation.hard_constraints.values())
                or result.metrics.get('quality')!=report['quality']
                or not math.isfinite(report['quality']) or not 0 <= report['quality'] <= 1):
            raise ValueError('kept policy trial failed its frozen protected evaluation')
        signatures = [{'operator_id': row['call']['operator_id'], 'operator_version': row['call']['operator_version'],
                       'parameters': row['call']['parameters']} for row in rows]
        planner_proofs=_planner_proofs(c)
        return {'origin': _origin(c, operator), 'kept_report_sha256': item['sha256'],
            'kept_receipt_sha256': payload_sha256(receipt), 'quality': report['quality'],
            'counts': report['counts'], 'rows': report['rows'], 'source_sha256': report['source_sha256'],
            'source_families': report['source_families'], 'source_media': report.get('source_media', []),
            'actions': signatures, 'planner_calls': state.planner_calls, 'operator_calls': state.operator_calls,
            'experiments_used': state.experiments_used, 'context': context, 'evaluation':evaluation.to_dict(),
            'planner_proofs':planner_proofs,'source_model_attempts':review_attempts+
                [p['model_attempt'] for p in planner_proofs if 'model_attempt' in p],
            'source_output_bytes_minimum':sum(Path(r['path']).stat().st_size for r in rows),
            'source_wall_seconds_minimum':math.ceil((datetime.fromisoformat(state.updated_at)-
                datetime.fromisoformat(rows[0]['receipt']['started_at'])).total_seconds()),
            'first_trial_started_at':rows[0]['receipt']['started_at'],
            'planner_config': c.planner.config(), 'all_trial_report_sha256s': [r['sha256'] for r in rows]}

    def _report(self, wall_seconds):
        runs = []
        for source, pair, origins in zip(self.cases, self.pairs, self.origins):
            if [_origin(c, op) for c, op in pair] != origins:
                raise ValueError('policy comparison source origin changed')
            a, b = [self._read_run(c, op, source, policy) for (c, op), policy in
                    zip(pair, (self.baseline, self.policy))]
            ca, cb = copy.deepcopy(a['planner_config']), copy.deepcopy(b['planner_config'])
            ca.pop('policy'); cb.pop('policy')
            if (a['origin']['campaign_id'] == b['origin']['campaign_id'] or ca != cb
                    or a['source_families'] != b['source_families'] or a['source_media'] != b['source_media']
                    or a['rows'] != b['rows'] or a['counts'] != b['counts']):
                raise ValueError('policy comparison changed more than its exact policy version')
            from .models import EvaluationReport
            gate = pair[1][0].gate.decide(EvaluationReport.from_dict(a['evaluation']),
                                         EvaluationReport.from_dict(b['evaluation']))
            for value in (a,b):
                value['planner_config_sha256'] = payload_sha256(value.pop('planner_config'))
            runs.append({'case': source, 'baseline': a, 'candidate': b,
                'delta': b['quality']-a['quality'], 'behavior_changed': a['actions'] != b['actions'],
                'frozen_gate':gate.to_dict()})
        families_seen,media_seen=set(),set()
        for run in runs:
            families=set(run['baseline']['source_families'])
            media={m[k] for m in run['baseline']['source_media'] for k in ('image_sha256','pixel_sha256','dhash')}
            if families_seen & families or media_seen & media:
                raise ValueError('policy cohort actual source families or media overlap')
            families_seen.update(families);media_seen.update(media)
        changed = any(r['behavior_changed'] for r in runs)
        regression_ok = all(r['delta'] >= -1e-12 for r in runs if r['case']['split']=='regression')
        delta = sum(r['delta'] for r in runs)/len(runs)
        source_runs=[r[side] for r in runs for side in ('baseline','candidate')]
        paid={ (p['ledger_root'],p['request_id']):p['attempts'] for r in source_runs for p in r['source_model_attempts'] }
        lower={'operator_calls':sum(r['operator_calls'] for r in source_runs),
               'planner_calls':sum(r['planner_calls'] for r in source_runs),'model_calls':sum(paid.values()),
               'output_bytes':sum(r['source_output_bytes_minimum'] for r in source_runs),
               'wall_seconds':sum(r['source_wall_seconds_minimum'] for r in source_runs)}
        if any(self.budget.prior_usage[k]<v for k,v in lower.items()):
            raise ValueError('policy comparison did not retain already paid source costs')
        usage = {key:0 for key in self.budget.prior_usage}
        usage.update(operator_calls=1, wall_seconds=wall_seconds)
        checks={'policy_binding':True,'source_isolation':True,
            'frozen_gates':all(r['frozen_gate']['decision']==CandidateDecision.ADOPT_CANDIDATE.value for r in runs),
            'regression_preserved':regression_ok,'behavior_changed':changed,
            'independent_delta':self.phase!='independent' or delta+1e-12>=.05}
        report = {'scope':SCOPE, 'phase':self.phase, **self.binding, 'runs':runs, 'mean_delta':delta,
            'minimum_delta':.05, 'regression_preserved':regression_ok, 'behavior_changed':changed,
            'passed':all(checks.values()),'check_results':checks,
            'policy_context_sha256':self.protocol.matched_context_sha256,
            'lifecycle_usage':usage, 'lifecycle_total':self.budget.check_usage((usage,)),
            'resource_deadline_epoch':self.budget.resource_deadline_epoch,
            'quality_scope':'catalog_clarity_proxy_only', 'new_model_calls':0,
            'source_cost_lower_bound':lower,
            'production_adoption_allowed':False, 'training_benefit_verified':False,
            'usage_basis':'Conservative read-only operator/wall/output reservation; original run costs remain precharged.'}
        for _ in range(8):
            doc={'binding':'0'*64,'report':report,'started_at':'0'*32,'finished_at':'0'*32}
            size=len((canonical_json(doc)+'\n').encode())
            if usage['output_bytes']==size:
                break
            usage['output_bytes']=size
            report['lifecycle_total']=self.budget.check_usage((usage,))
        return report

    def workload(self, call):
        self.path(call)
        if time.time() >= self.budget.resource_deadline_epoch:
            raise ValueError('policy qualification deadline exhausted')
        started=time.monotonic()
        report=self._report(30)
        if time.monotonic()-started>30:
            raise ValueError('policy reader exceeded its wall reservation')
        return report

    def validate_report(self, call, report):
        self.path(call)
        if report['lifecycle_usage']['wall_seconds']!=30:
            raise ValueError('policy reader reservation changed')
        if report != self._report(report['lifecycle_usage']['wall_seconds']):
            raise ValueError('policy comparison report or underlying evidence changed')
        if self.path(call).exists() and self.path(call).stat().st_size>report['lifecycle_usage']['output_bytes']:
            raise ValueError('policy comparison output exceeded its recorded reservation')
        if self.path(call).exists():
            from datetime import datetime
            doc=json.loads(self.path(call).read_text())
            elapsed=(datetime.fromisoformat(doc['finished_at'])-datetime.fromisoformat(doc['started_at'])).total_seconds()
            if not 0<=elapsed<=report['lifecycle_usage']['wall_seconds']:
                raise ValueError('policy reader exceeded its wall reservation')

    def result(self, call, report):
        value = super().result(call, report)
        value['metrics'].update(qualification_passed=float(report['passed']),quality_delta=report['mean_delta'])
        return value


class PolicyQualificationAuthority:
    def __init__(self, *, engineering, independent):
        if any(not isinstance(a, CommittedWorkloadAuthority) or not isinstance(a.operator, PolicyComparisonChecks)
               for a in (engineering, independent)):
            raise TypeError('policy qualification needs committed host policy comparison readers')
        self.engineering, self.independent = engineering, independent

    def read(self):
        from datetime import datetime
        values = [self.engineering.read(), self.independent.read()]
        for value, authority, phase in zip(values,(self.engineering,self.independent),('engineering','independent')):
            report, op = value['report'], authority.operator
            if (report['scope']!=SCOPE or report['phase']!=phase or report['passed'] is not True
                    or authority.controller.evaluator.contract.sha256!=op.protocol.stages[phase]['evaluator_sha256']):
                raise ValueError('policy comparison did not pass the frozen stage')
        eng, ind = (v['report'] for v in values)
        if (eng['policy_sha256']!=ind['policy_sha256'] or eng['baseline_policy_sha256']!=ind['baseline_policy_sha256']
                or eng['protocol_sha256']!=ind['protocol_sha256']
                or eng['policy_context_sha256']!=ind['policy_context_sha256']
                or values[0]['origin']['campaign_id']==values[1]['origin']['campaign_id']):
            raise ValueError('policy qualification versions, protocol or campaigns differ')
        runs = eng['runs']+ind['runs']
        finished=datetime.fromisoformat(self.engineering.controller.status().updated_at)
        if any(datetime.fromisoformat(r[side]['first_trial_started_at'])<=finished
               for r in ind['runs'] for side in ('baseline','candidate')):
            raise ValueError('independent policy execution preceded engineering acceptance')
        if (len({r['case']['family_id'] for r in runs})!=len(runs)
                or len({r['case']['artifact']['sha256'] for r in runs})!=len(runs)):
            raise ValueError('policy qualification cohorts share declared sources')
        seen_families,seen_media=set(),set()
        for run in runs:
            families=set(run['baseline']['source_families'])
            media={m[k] for m in run['baseline']['source_media'] for k in ('image_sha256','pixel_sha256','dhash')}
            if seen_families & families or seen_media & media:
                raise ValueError('policy qualification actual source families or media overlap')
            seen_families.update(families);seen_media.update(media)
        if (eng['resource_deadline_epoch']!=ind['resource_deadline_epoch'] or any(
                self.independent.operator.budget.prior_usage[k]<v+ind['source_cost_lower_bound'].get(k,0)
                for k,v in eng['lifecycle_total'].items())):
            raise ValueError('policy qualification reset costs or deadline between stages')
        policy=self.independent.operator.policy
        return {'operator_id':'policy.'+policy.policy_id,'version':policy.version,'scope':SCOPE,
            'policy_sha256':policy.compute_hash(),'policy_context_sha256':ind['policy_context_sha256'],
            'baseline_policy_sha256':eng['baseline_policy_sha256'],'protocol_sha256':eng['protocol_sha256'],
            'engineering':values[0]['origin'],'independent':values[1]['origin'],
            'qualification_receipts':[v['receipt_sha256'] for v in values],
            'lifecycle_total':ind['lifecycle_total'],'resource_deadline_epoch':ind['resource_deadline_epoch'],
            'quality_scope':'catalog_clarity_proxy_only','production_adoption_allowed':False,
            'model_authority':False,'training_benefit_verified':False}

    def verify(self, snapshot):
        if snapshot != self.read():
            raise ValueError('qualified policy snapshot changed')
