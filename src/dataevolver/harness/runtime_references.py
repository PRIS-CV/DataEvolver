"""Reconstruct registered evidence readers from explicit frozen campaign paths.

No arbitrary imports, summary qualifications or extra authority types. Reader
construction does not tick a Controller, execute a candidate or submit a model.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactStore, identifier, payload_sha256

_reading = ContextVar('runtime_evidence_campaigns', default=())


def workload(reference):
    from .cli import _controller
    from .models import GoalSpec
    from .state_store import CampaignLayout, StateStore
    if set(reference) != {'campaign_root', 'campaign_id'}:
        raise ValueError('evidence requires an exact host campaign reference')
    identifier(reference['campaign_id'])
    root = Path(reference['campaign_root']).resolve()
    path = root/reference['campaign_id']/'goal.json'
    store = StateStore(CampaignLayout(root, reference['campaign_id']))
    state = store.load_state()
    if (state.campaign_id != reference['campaign_id'] or state.state.value not in {'COMPLETE', 'ABSTAIN'}
            or state.data.get('pending_call') or store.replay().to_dict() != state.to_dict()):
        raise ValueError('evidence readers require a terminal, reconciled and replayable campaign')
    active = _reading.get()
    if str(path) in active or len(active) >= 12:
        raise ValueError('runtime evidence references contain a cycle or exceed their bound')
    token = _reading.set((*active, str(path)))
    try:
        goal = GoalSpec.from_dict(json.loads(path.read_text()))
        if goal.campaign_id != reference['campaign_id']:
            raise ValueError('evidence campaign identity changed')
        controller = _controller(goal, root, create=False)
        return controller, controller.registry.get(goal.initial_operator_id)
    finally:
        _reading.reset(token)


def committed(reference):
    from .artifact_operator import CommittedWorkloadAuthority
    return CommittedWorkloadAuthority(*workload(reference))


def adoption(reference):
    from .operator_lifecycle import CommittedOperatorAdoptionAuthority
    controller, operator = workload(reference)
    planned = [e for e in controller.store.events.read() if e.event_type == 'ACTION_PLANNED'
               and e.payload['operator_call']['operator_id'] == 'evolution.adopt_operator_version']
    if len(planned) != 1:
        raise ValueError('version binding requires exactly one original adoption call')
    return CommittedOperatorAdoptionAuthority(controller, operator, planned[0].payload['call_id'])


def runner(config):
    from .candidate_sandbox import LinuxPureDataRunner, SandboxLimits
    return LinuxPureDataRunner(**{**config,
        **({'limits': SandboxLimits(**config['limits'])} if 'limits' in config else {})})


def candidate(config):
    from .operator_development import CandidatePackage, read_development_plan, ControllerRequestAuthority, FrozenSourceRequestAuthority
    from .candidate_build_authority import CommittedCandidateBuildAuthority
    request = config['request']
    authority = (FrozenSourceRequestAuthority(**request['frozen_source']) if 'frozen_source' in request
                 else ControllerRequestAuthority(workload(request)[0]))
    return {'plan': read_development_plan(config['plan']), 'package': CandidatePackage.from_dict(config['package']),
        'store': ArtifactStore(config['roots']), 'request_authority': authority,
        'build_authority': CommittedCandidateBuildAuthority(*workload(config['build'])),
        'runner': runner(config.get('runner', {}))}


def qualification(config):
    kind = config['kind']
    values = {'engineering': committed(config['engineering']), 'independent': committed(config['independent'])}
    if kind == 'policy':
        from .policy_qualification import PolicyQualificationAuthority
        return PolicyQualificationAuthority(**values)
    if kind == 'repair':
        from .repair_qualification import RepairQualificationAuthority
        from .operator_development import CandidatePackage
        return RepairQualificationAuthority(**values, package=CandidatePackage.from_dict(config['package']),
            runner=runner(config.get('runner', {})), lifecycle_total=config['lifecycle_total'],
            resource_deadline_epoch=config['resource_deadline_epoch'])
    raise ValueError('unknown registered qualification reader')


def step_registry(bindings):
    from .operator_sdk import builtin_step_registry
    from .repair_qualification import qualified_repair_step
    from .operator_lifecycle import OperatorVersionStore
    if set(bindings) - {'repair'}:
        raise ValueError('unregistered entry step binding')
    injection = {}
    if bindings.get('repair'):
        config = bindings['repair']
        injection['repair_step'] = qualified_repair_step(catalog=OperatorVersionStore(config['catalog_root']),
            adoption=adoption(config['adoption']), qualification=qualification(config['qualification']),
            expected_head=config['expected_head'], **candidate(config['candidate']))
    return builtin_step_registry(**injection)


def check_steps(bindings):
    from .operator_lifecycle import OperatorVersionStore
    provenance = {}
    for name, config in bindings.items():
        if name != 'repair':
            raise ValueError('unregistered entry step binding')
        event = adoption(config['adoption']).read()
        snapshot = OperatorVersionStore(config['catalog_root']).active(event, scope='image_repair_instruction.v1',
            head=config['expected_head'], qualification_authority=qualification(config['qualification']))
        provenance[name] = {'package_sha256': snapshot['package_sha256'],
            'adoption_sha256': event['sha256'], 'catalog_head': config['expected_head'],
            'scope': snapshot['scope'], 'qualification_sha256': payload_sha256(snapshot)}
    return provenance


def graph_dependencies(config, contract, roots):
    """Bind existing native transports and immutable read-only source ledgers."""
    if set(config) - {'process_endpoints', 'source_gateways'}:
        raise ValueError('unknown registered graph runtime binding')
    if not config:
        return {}
    from dataevolver.adapters.model_contracts import ModelProfile
    from dataevolver.adapters.local_process_endpoint import LocalProcessEndpointProvider
    from .runtime_entry import model_gateway
    store = ArtifactStore(roots)
    output = store.roots[contract['dataset_store']]
    profiles = {p.profile_id: p for p in map(ModelProfile.from_dict, contract['models']['profiles'])}
    providers, sources = {}, {}
    for name, value in config.get('process_endpoints', {}).items():
        if set(value) != {'deployment', 'native_root'}:
            raise ValueError('process endpoint requires exact deployment and native root')
        if name not in profiles or profiles[name].backend != 'local_python':
            raise ValueError('process endpoint differs from its graph model profile')
        path = Path(value['native_root']).resolve()
        if not path.is_relative_to(output):
            raise ValueError('native endpoint journal escapes graph dataset store')
        profile, deployment = profiles[name], json.loads(json.dumps(value['deployment']))
        factory = lambda p=profile, d=deployment, n=path: LocalProcessEndpointProvider(p, d, store, n)
        factory()  # Validate the frozen identity; constructor submits no request.
        providers[name] = factory
    for role, value in config.get('source_gateways', {}).items():
        if set(value) != {'gateway', 'contract_hash'}:
            raise ValueError('source gateway needs its exact original contract hash')
        gateway = model_gateway(value['gateway'], roots=roots)
        if gateway.contract_hash() != value['contract_hash']:
            raise ValueError('original source gateway contract changed')
        if gateway.root.resolve().is_relative_to(output) or output.is_relative_to(gateway.root.resolve()):
            raise ValueError('source model journal overlaps writable graph dataset')
        sources[role] = gateway
    return {'providers': providers, 'source_gateways': sources}


def host_workload(config):
    """Existing stage operators, reconstructed identically for committed readers."""
    from .operator_development import DevelopmentBudget, QualificationProtocol
    from .runtime_entry import model_gateway
    kind, values = config['kind'], config['configuration']
    if kind == 'native_scene':
        from .adapters.native_scene import NativeSceneOperator
        return NativeSceneOperator(**values)
    if kind == 'policy_comparison':
        from .policy_qualification import PolicyComparisonChecks
        from .feedback import PolicySpec
        return PolicyComparisonChecks(baseline_policy=PolicySpec.from_dict(values['baseline_policy']),
            policy=PolicySpec.from_dict(values['policy']), protocol=QualificationProtocol.from_dict(values['protocol']),
            store=ArtifactStore(values['roots']), pairs=[tuple(workload(ref) for ref in pair) for pair in values['pairs']],
            phase=values['phase'], budget=DevelopmentBudget.from_dict(values['budget']), output_root=values['output_root'])
    if kind == 'version_mutation':
        from .adapters.operator_version_lifecycle import OperatorVersionMutation
        from .operator_lifecycle import OperatorVersionStore
        from .operator_lifecycle_authority import HistoricalCatalogPrefix
        catalog = OperatorVersionStore(values['catalog_root'])
        # Constructors bind the actual historical prefix; effect verification
        # always sees the current complete catalog, including revocations.
        prefix = HistoricalCatalogPrefix(catalog, values['expected_head'])
        kwargs = ({'adoption': adoption(values['adoption']), 'reason': values['reason']}
                  if values.get('adoption') else {'qualification_authority': qualification(values['qualification']),
                                                  'store': ArtifactStore(values['roots'])})
        result = OperatorVersionMutation(catalog=prefix, expected_head=values['expected_head'],
            budget=DevelopmentBudget.from_dict(values['budget']), output_root=values['output_root'], **kwargs)
        result.catalog = catalog
        return result
    if kind == 'candidate_build':
        from .adapters.operator_candidate_build import CandidateBuildOperator
        from .operator_development import read_development_plan, ControllerRequestAuthority, FrozenSourceRequestAuthority
        from .candidate_limits import CandidateSlotLimits
        if not values.get('candidate_limits'):
            raise ValueError('entry builder requires explicit original candidate-slot accounting')
        request = values['request']
        authority = (FrozenSourceRequestAuthority(**request['frozen_source']) if 'frozen_source' in request
                     else ControllerRequestAuthority(workload(request)[0]))
        return CandidateBuildOperator(plan=read_development_plan(values['plan']), store=ArtifactStore(values['roots']),
            request_authority=authority, gateway=model_gateway(values['gateway'], roots=values['roots']),
            candidate_limits=CandidateSlotLimits(**values['candidate_limits']),
            **{k: values[k] for k in ('role', 'output_root', 'native_roots', 'max_tokens', 'timeout_seconds')})
    if kind == 'benchmark':
        from .adapters.generation_benchmark import GenerationBenchmarkOperator
        return GenerationBenchmarkOperator(gateway=model_gateway(values['gateway'], roots=values['roots']),
            store=ArtifactStore(values['roots']), **{k: values[k] for k in ('manifest', 'journal_root', 'output_root')})
    if kind == 'repair_comparison':
        from .repair_qualification import RepairCandidateChecks, CommittedRepairRunAuthority, CommittedRepairBenchmarkAuthority
        from .artifact_operator import CommittedEffectAuthority
        context = candidate(values['candidate'])
        pairs = {}
        for family, pair in values['run_pairs'].items():
            pairs[family] = [CommittedRepairRunAuthority(
                repair=CommittedEffectAuthority(*workload(item['repair'])),
                edited=CommittedEffectAuthority(*workload(item['edited'])), candidate=bool(i),
                **{k: item[k] for k in ('repair_node', 'edit_node', 'review_node')}, **context)
                for i, item in enumerate(pair)]
        return RepairCandidateChecks(**context, cases=values['cases'], phase=values['phase'], run_pairs=pairs,
            outcome_authorities=tuple(CommittedRepairBenchmarkAuthority(*workload(ref)) for ref in values['outcomes']),
            output_root=values['output_root'])
    raise ValueError('unknown registered host workload')
