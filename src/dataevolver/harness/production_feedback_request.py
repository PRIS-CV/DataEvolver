"""Read an actual capability request from its frozen feedback-review Controller."""
import copy
import inspect
import json
import os
import subprocess
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256


class FrozenProductionFeedbackRequestAuthority:
    def __init__(self, *, config, profile_path, python):
        self.config = Path(config).resolve()
        self.profile_path = Path(profile_path).resolve()
        self.python = Path(python).absolute()
        self._identity = self._current_identity()
        self._cached = None

    def _current_identity(self):
        cfg = self.config
        goal = json.loads((cfg/'protected/goal.json').read_text())
        campaign = cfg/'output/campaigns'/goal['campaign_id']
        return {'config': str(cfg), 'profile_path': str(self.profile_path), 'python': str(self.python),
            'campaign_id': goal['campaign_id'], 'source_commit': (cfg/'source-commit').read_text().strip(),
            'reader_sha256': file_sha256(Path(__file__)),
            'files': {str(p): file_sha256(p) for p in (self.profile_path, self.python,
                cfg/'source.tar', cfg/'host-launcher.py', cfg/'env.sh', campaign/'frozen_contract.json',
                campaign/'state.json', campaign/'events.jsonl')}}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError('original production feedback review identity changed')
        return copy.deepcopy(self._identity)

    def read(self):
        self.identity()
        if self._cached is None:
            script = ('import json,sys\nfrom pathlib import Path\n'+inspect.getsource(_validate_review_terminal)
                      +'\n'+inspect.getsource(_read_request)
                      +'\nprint(json.dumps(_read_request(*json.load(sys.stdin))))')
            before = {str(p): file_sha256(p) for p in self.config.rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts}
            shell = ('set -a; source "$1/env.sh"; export PYTHONPATH="$1/source/src" '
                     'CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0; '
                     'cd "$1/source"; shift; exec "$@"')
            descriptor = json.loads((self.config/'protected/descriptor.json').read_text())
            # Operational feedback independently reopens both its original
            # witness and runtime request. This is CPU verification, not an
            # extension of any model timeout, resource deadline or call budget.
            readback_timeout = (1200 if descriptor.get('feedback_kind') ==
                                'registered_boundary_operational_qualification' else 300)
            result = subprocess.run(['bash', '-ec', shell, 'frozen-feedback-request', str(self.config),
                                     str(self.python), '-B', '-c', script],
                input=json.dumps(self._read_arguments()),
                capture_output=True, text=True, timeout=readback_timeout, cwd=self.config/'source',
                env={**os.environ, 'PYTHONPATH': str(self.config/'source/src'), 'CUDA_VISIBLE_DEVICES': '',
                     'PYTHONDONTWRITEBYTECODE': '1', 'GIT_OPTIONAL_LOCKS': '0'})
            after = {str(p): file_sha256(p) for p in self.config.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts}
            if before != after:
                raise ValueError('request verification modified the original review')
            if result.returncode:
                raise ValueError('frozen feedback request verification failed: '+result.stderr[-2500:])
            self._cached = json.loads(result.stdout)
        for name, digest in self._cached['artifact_hashes'].items():
            if file_sha256(Path(name)) != digest:
                raise ValueError('original feedback request artifact changed')
        self.identity()
        return copy.deepcopy(self._cached)


    def _read_arguments(self):
        return [str(self.config), str(self.profile_path), str(self.python)]


class FrozenProductionFeedbackReviewAuthority(FrozenProductionFeedbackRequestAuthority):
    """Read a completed decision, including ABSTAIN, without creating a request.

    This is separate from the strict request authority used by implementation
    operators. A launcher postcheck failure never authorizes replaying a model.
    """
    def _current_identity(self):
        return {**super()._current_identity(), 'kind': 'terminal_feedback_review'}

    def _read_arguments(self):
        return [*super()._read_arguments(), False]


def _validate_review_terminal(state, decision, require_capability):
    expected = {'CAPABILITY_REQUESTED': 'REQUEST_CAPABILITY', 'RSI_AGENT_ABSTAINED': 'ABSTAIN'}
    reason = state['terminal_reason']
    if (type(require_capability) is not bool or state['state'] != 'ABSTAIN'
            or reason not in expected or decision['action'] != expected[reason]
            or state['planner_calls'] != 1 or state['operator_calls'] != 1
            or state['experiments_used'] != 0 or state['data'].get('pending_call')
            or (require_capability and reason != 'CAPABILITY_REQUESTED')):
        raise ValueError('feedback review must retain its exact terminal decision and single-call budget')
    return reason == 'CAPABILITY_REQUESTED'


def _read_request(config, profile_path, python, require_capability=True):
    import sys
    from dataevolver.dataset.contracts import ArtifactStore, file_sha256
    from dataevolver.harness.training_candidate_feedback import FrozenTrainingCandidateFeedbackAuthority, ProductionFeedbackReviewDomain
    from dataevolver.harness.adapters.training_candidate_feedback import TrainingCandidateFeedbackOperator
    from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner, validate_decision
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.models import GoalSpec, OperatorCall, OperatorReceipt, EvidenceStatus
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.training_trial_authority import _read_only_event
    from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy, ModelRequest
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.adapters.codex_planner import RecoveringPlannerBackend
    def forbidden(*a, **kw): raise RuntimeError('feedback request reader cannot execute or infer')
    EvolutionController.run = EvolutionController.tick = RuntimeRSIPlanner.plan = forbidden
    TrainingCandidateFeedbackOperator.execute = TrainingCandidateFeedbackOperator.recover = forbidden
    ModelGateway.invoke = ModelGateway._provider = ModelGateway.adopt_completed = forbidden
    cfg = Path(config); protected = cfg/'protected'
    descriptor = json.loads((protected/'descriptor.json').read_text())
    old = json.loads((protected/'origin-proof.json').read_text())['config']
    if descriptor.get('feedback_kind') == 'registered_native_boundary_probe':
        from dataevolver.harness.boundary_feedback import FrozenBoundaryFeedbackAuthority
        from dataevolver.harness.adapters.boundary_feedback import BoundaryProbeFeedbackOperator
        if set(descriptor) != {'feedback_kind', 'boundary_descriptor'}:
            raise ValueError('boundary request reader has an unexpected origin descriptor')
        authority = FrozenBoundaryFeedbackAuthority(config=old, descriptor=descriptor['boundary_descriptor'], python=python)
        capture_type = BoundaryProbeFeedbackOperator
        _, boundary_proof = authority.read()
        stores = boundary_proof['input_store_roots']
    elif descriptor.get('feedback_kind') == 'registered_boundary_witness':
        from dataevolver.harness.boundary_witness_readback import FrozenBoundaryWitnessFeedbackAuthority
        from dataevolver.harness.adapters.boundary_witness_feedback import BoundaryWitnessFeedbackOperator
        if set(descriptor) != {'feedback_kind'}:
            raise ValueError('witness request reader has an unexpected descriptor')
        authority = FrozenBoundaryWitnessFeedbackAuthority(config=old, python=python)
        capture_type = BoundaryWitnessFeedbackOperator
        stores = authority.read()[1]['input_store_roots']
    elif descriptor.get('feedback_kind') == 'registered_boundary_operational_qualification':
        from dataevolver.harness.boundary_qualification_readback import FrozenBoundaryQualificationFeedbackAuthority
        from dataevolver.harness.adapters.boundary_qualification_feedback import BoundaryQualificationFeedbackOperator
        if set(descriptor) != {'feedback_kind'}:
            raise ValueError('qualification request reader has an unexpected descriptor')
        authority = FrozenBoundaryQualificationFeedbackAuthority(config=old, python=python)
        capture_type = BoundaryQualificationFeedbackOperator
        stores = authority.read()[1]['input_store_roots']
    else:
        authority = FrozenTrainingCandidateFeedbackAuthority(config=old, descriptor=descriptor, python=python)
        capture_type = TrainingCandidateFeedbackOperator
        stores = descriptor['stores']
    domain = ProductionFeedbackReviewDomain(authority=authority,
        admission=json.loads((protected/'admission.json').read_text()))
    store = ArtifactStore(stores)
    profile = ModelProfile.from_dict(json.loads(Path(profile_path).read_text()))
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({'agent.runtime_rsi': profile.profile_id}, 1),
        receipt_root=cfg/'output/model_receipts', artifact_store=store)
    goal = GoalSpec.from_dict(json.loads((protected/'goal.json').read_text()))
    planner = RuntimeRSIPlanner(backend=RecoveringPlannerBackend(gateway, model_id=profile.model, timeout_seconds=900),
        domain=domain, trace_root=cfg/'output/campaigns'/goal.campaign_id/'planner')
    op = capture_type(authority=authority, output_root=cfg/'output')
    registry = OperatorRegistry(); registry.register(op)
    sys.addaudithook(_read_only_event)
    c = EvolutionController.open(repo_root=cfg/'source', campaigns_root=cfg/'output/campaigns', goal=goal,
        registry=registry, planner=planner, evaluator_contract=EvaluatorContract.from_path(cfg/'evaluator.json'),
        schema_paths=tuple(protected.rglob('*.json')),
        execution_paths=tuple((cfg/'source/src/dataevolver').rglob('*.py'))+(cfg/'host-launcher.py',))
    state = c.status()
    if (not c._frozen()[0] or c.replay().to_dict() != state.to_dict()
            or state.data.get('pending_call')):
        raise ValueError('runtime feedback request must be terminal, frozen and replayable')
    events = c.store.events.read(); paths = set()
    planned = [e for e in events if e.event_type == 'ACTION_PLANNED']
    committed = [e for e in events if e.event_type == 'ACTION_COMMITTED']
    if len(planned) != 1 or len(committed) != 1:
        raise ValueError('runtime feedback was not captured exactly once')
    call = OperatorCall.from_dict(planned[0].payload['operator_call'])
    receipt = OperatorReceipt.from_dict(committed[0].payload['receipt'])
    verified = op.verify_effect(call, receipt)
    if (verified.status is not EvidenceStatus.VERIFIED or verified.artifact_hashes !=
            committed[0].payload['effect_verification']['artifact_hashes']):
        raise ValueError('registered runtime feedback effect no longer verifies')
    trace_path = c.layout.campaign_dir/'planner/p000/trace.json'; trace = json.loads(trace_path.read_text())
    has_request = _validate_review_terminal(state.to_dict(), trace['decision'], require_capability)
    recorded = [e for e in events if e.event_type == 'RSI_CAPABILITY_REQUESTED']
    if has_request:
        requested = state.data['rsi_capability_request']; path = Path(requested['path'])
        if file_sha256(path) != requested['sha256']:
            raise ValueError('actual runtime capability request bytes changed')
        value = json.loads(path.read_text())
        if (len(recorded) != 1 or recorded[0].payload['state_after']['data']['rsi_capability_request'] != requested
                or value != domain.compile_capability_request(value['decision'], state)):
            raise ValueError('capability request was not committed from the actual diagnostic')
    else:
        if recorded or state.data.get('rsi_capability_request'):
            raise ValueError('abstention cannot manufacture an implementation capability request')
        requested = None
        value = {'decision': trace['decision']}
    if trace['outcome'] != 'VALID_DECISION' or trace['decision'] != value['decision']:
        raise ValueError('runtime feedback decision does not match its original trace')
    validate_decision(value['decision'], trace['context'])
    request = ModelRequest.from_dict(trace['model_call']['request'])
    model_receipt = gateway._recover(request, gateway._prepare(request))
    if (model_receipt is None or model_receipt.without_reuse_hash() != trace['model_call']['receipt_hash']
            or model_receipt.output != value['decision']):
        raise ValueError('actual runtime request lost its exact Gateway receipt')
    native_root = cfg/'output/native'/request.request_id
    native = json.loads((native_root/'native.json').read_text())
    if (native['output_sha256'] != file_sha256(native_root/'final-response.json')
            or native['events_sha256'] != file_sha256(native_root/'events.jsonl')
            or native['profile_hash'] != profile.compute_hash()
            or native['request_hash'] != model_receipt.request_hash
            or native['response']['output'] != model_receipt.output):
        raise ValueError('runtime feedback native output/events changed')
    paths.update(Path(p) for p in verified.artifact_hashes)
    paths.update(p for p in (cfg/'output').rglob('*') if p.is_file() and p.name != '.controller.lease')
    paths.update(p for p in protected.rglob('*') if p.is_file())
    paths.update(p for p in (cfg/'source/src/dataevolver').rglob('*.py'))
    result = {'request': value if has_request else None, 'request_sha256': requested['sha256'] if requested else None,
        'feedback_sha256': domain.feedback_hash, 'original_model_request_id': request.request_id,
        'source_commit': (cfg/'source-commit').read_text().strip(), 'frozen': True, 'replay': True,
        'registered_effect_verified': True, 'native_and_receipt_verified': True,
        'new_model_calls': 0, 'new_candidate_executions': 0,
        'artifact_hashes': {str(p): file_sha256(p) for p in paths}}
    if not require_capability:
        result.update(decision=value['decision'], terminal_reason=state.terminal_reason,
            state=state.to_dict(), usage=gateway.audit_usage(), feedback=domain.feedback(),
            admission=json.loads((protected/'admission.json').read_text()))
    return result
