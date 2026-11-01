"""Thin configuration assembly; existing Controllers and authorities own effects.

Constructing a runtime never submits a model request. Adopted policies are
admitted against the actual frozen INIT consumer, then pinned for recovery.
"""
from __future__ import annotations

import json
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
from dataevolver.adapters.model_gateway import ModelGateway
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from .feedback import PolicyResolver, PolicySpec
from .models import ControllerState
from .state_store import atomic_write_json


def model_gateway(config, *, roots):
    from .attempt_limits import ModelAttemptLimits
    gateway = ModelGateway(profiles=tuple(ModelProfile.from_dict(p) for p in config['profiles']),
        policy=ModelPolicy.from_dict(config['policy']), receipt_root=Path(config['receipt_root']),
        artifact_store=ArtifactStore(roots),
        attempt_limits=ModelAttemptLimits(**config['attempt_limits']) if config.get('attempt_limits') else None)
    # Reuse registered worker providers; configuration cannot import Python.
    for profile_id, worker in config.get('workers', {}).items():
        if worker['kind'] == 'visual':
            from dataevolver.adapters.visual_models import LocalVisualProvider
            provider = LocalVisualProvider
        elif worker['kind'] == 'benchmark':
            from dataevolver.adapters.benchmark_worker import LocalBenchmarkProvider
            provider = LocalBenchmarkProvider
        else:
            raise ValueError('unknown registered runtime worker')
        profile = gateway.profiles[profile_id]
        gateway.factories[profile_id] = lambda p=profile, w=worker, factory=provider: factory(
            p, w['configuration'], gateway.store, w['dataset_store'], Path(w['native_root']))
    return gateway


def collected_feedback(config):
    if config is None:
        return None
    from .training_feedback import ExternalTrainingRunAdapter, TrainingJobSpec
    adapter = ExternalTrainingRunAdapter(store=ArtifactStore(config['roots']), work_root=config['work_root'])
    return adapter, TrainingJobSpec.from_dict(config['job']), ArtifactRef.from_dict(config['receipt'])


def build_catalog_runtime(goal, campaign_root):
    from .catalog_rsi import CatalogRSIDomain, CatalogTrialOperator
    from .model_backend import GatewayPlannerBackend
    from .operators import OperatorRegistry
    from .runtime_rsi import RuntimeRSIPlanner
    from .runtime_references import step_registry
    if goal.initial_operator_id != 'data.catalog_trial' or goal.planner.get('kind') != 'runtime_rsi':
        raise ValueError('catalog runtime needs its explicit catalog Goal and runtime planner')
    config = goal.planner['runtime']
    if config.get('domain_kind') != 'catalog' or goal.interventions:
        raise ValueError('catalog runtime domain or static interventions changed')
    domain = CatalogRSIDomain(configuration=config['domain'], roots=config['roots'],
        registry=step_registry(config.get('step_bindings', {})),
        training_feedback=collected_feedback(config.get('training_feedback')))
    policy = PolicySpec.from_dict(config['policy']) if config.get('policy') else None
    if config.get('policy_adoption') and policy is None:
        raise ValueError('policy adoption requires the exact PolicySpec')
    gateway = model_gateway(config['gateway'], roots=config['roots'])
    if set(goal.resource_policy.get('allowed_operators', [])) != {'data.catalog_trial'}:
        raise ValueError('catalog runtime grants only its registered catalog trial')
    backend = GatewayPlannerBackend(gateway, role=config.get('planner_role', 'agent.runtime_rsi'),
        max_tokens=config.get('max_tokens', 4096), timeout_seconds=config.get('timeout_seconds', 300))
    profiles = tuple(gateway.profiles.values())
    if any(p.backend != 'mock' for p in profiles):
        if (goal.resource_policy.get('allow_network') is not True
                or goal.resource_policy.get('network_scope') != 'registered_planner_inference_only'
                or gateway.attempt_limits is None):
            raise ValueError('native catalog planning needs explicit registered inference authority')
    registry = OperatorRegistry()
    registry.register(CatalogTrialOperator(domain=domain, output_root=Path(campaign_root),
        gateway=gateway, review_roles=config.get('review_roles', ())))
    planner = RuntimeRSIPlanner(backend=backend, domain=domain, policy=policy,
        trace_root=Path(campaign_root)/goal.campaign_id/'planner/decisions')
    return registry, planner


def version_bindings(controller):
    goal = controller.goal
    config = goal.planner.get('runtime', {})
    return {'policy': config.get('policy_adoption'),
        'steps': config.get('step_bindings', {}) if goal.initial_operator_id == 'data.catalog_trial'
                 else goal.objective.get('step_bindings', {})}


def admit_execution(controller):
    try:
        return _admit_execution(controller)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        state = controller.status()
        atomic_write_json(controller.layout.campaign_dir/'runtime-admission-refusal.json', {
            'reason': str(exc), 'failure_type': type(exc).__name__,
            'frozen_sha256': file_sha256(controller.layout.frozen_contract_path),
            'bindings_sha256': payload_sha256(version_bindings(controller)),
            'operator_calls': state.operator_calls, 'planner_calls': state.planner_calls,
            'experiments_used': state.experiments_used, 'state': state.state.value})
        raise


def _admit_execution(controller):
    """First admission is committed in INIT; recovery never resolves a new policy.

    The event log anchors the admission artifact. Current qualification and
    revocation are rechecked before each subsequent Controller tick.
    """
    from .runtime_references import adoption, qualification, check_steps
    from .operator_lifecycle import OperatorVersionStore
    bindings = version_bindings(controller)
    if not bindings['policy'] and not bindings['steps']:
        return
    valid, changed = controller._frozen()
    if not valid:
        raise ValueError('runtime admission contract drift: '+','.join(changed))
    state = controller.status()
    if controller.replay().to_dict() != state.to_dict():
        raise ValueError('runtime admission needs replayable consumer state')
    provenance = check_steps(bindings['steps'])
    policy_binding = bindings['policy']
    if policy_binding:
        catalog = OperatorVersionStore(policy_binding['catalog_root'])
        authority = qualification(policy_binding['qualification'])
        event = adoption(policy_binding['adoption']).read()
        snapshot = catalog.active(event, scope='catalog_policy.v1', head=policy_binding['expected_head'],
                                  qualification_authority=authority)
        if snapshot.get('policy_sha256') != controller.planner.policy.compute_hash():
            raise ValueError('admitted policy differs from its qualified version')
        provenance['policy'] = {'policy_sha256': snapshot['policy_sha256'],
            'adoption_sha256': event['sha256'], 'catalog_head': policy_binding['expected_head'],
            'scope': snapshot['scope'], 'qualification_sha256': payload_sha256(snapshot)}
    body = {'schema_version': 'dataevolver.runtime_admission.v1',
        'campaign_id': controller.goal.campaign_id,
        'frozen_sha256': file_sha256(controller.layout.frozen_contract_path),
        'bindings_sha256': payload_sha256(bindings), 'provenance': provenance}
    path = controller.layout.campaign_dir/'runtime-admission.json'
    anchor = state.data.get('runtime_admission')
    if anchor is not None:
        if (anchor != {'sha256': payload_sha256(body)} or not path.is_file()
                or json.loads(path.read_text()) != body):
            raise ValueError('pinned runtime admission evidence changed')
        return
    if (state.state is not ControllerState.INIT or state.operator_calls or state.planner_calls
            or state.experiments_used):
        raise ValueError('runtime version admission must precede all consumer effects')
    if policy_binding:
        PolicyResolver(catalog, authority).resolve(event, scope='catalog_policy.v1',
            expected_head=policy_binding['expected_head'], policy=controller.planner.policy.to_dict(),
            controller=controller)
    if path.exists() and json.loads(path.read_text()) != body:
        raise ValueError('incomplete runtime admission artifact changed')
    atomic_write_json(path, body)
    controller._commit(state, ControllerState.INIT, 'RUNTIME_VERSIONS_ADMITTED', 'VERSIONS_PINNED',
        data={**state.data, 'runtime_admission': {'sha256': payload_sha256(body)}},
        event_payload={'admission': body})
