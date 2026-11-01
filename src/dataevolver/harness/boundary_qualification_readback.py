"""Independent readback of a frozen missing-evidence qualification stage.

Qualified production evidence uses the existing stage authorities. This reader
explicitly handles the absent-evidence admission; it cannot reconstruct a
positive stage by treating host JSON as a production authority.
"""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess

from dataevolver.dataset.contracts import file_sha256


def verify_frozen_boundary_qualification(config, *, python):
    cfg = Path(config).resolve()
    def tree():
        return {str(p): file_sha256(p) for p in cfg.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
    before = tree()
    host = json.loads((cfg/'protected/host.json').read_text())
    script = 'import json,sys\n'+inspect.getsource(_verify_qualification)
    script += '\nprint(json.dumps(_verify_qualification(*json.load(sys.stdin))))'
    completed = subprocess.run([str(python), '-B', '-c', script],
        input=json.dumps([str(cfg), str(python)]), text=True, capture_output=True,
        timeout=1800 if host.get('capture_kind') == 'independent_evidence_acquisition' else 900,
        cwd=cfg/'source', env={**os.environ, 'PYTHONPATH': str(cfg/'source/src'),
            'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '', 'GIT_OPTIONAL_LOCKS': '0'})
    if tree() != before:
        raise ValueError('qualification readback changed the original frozen capture')
    if completed.returncode:
        raise ValueError('frozen qualification readback failed: '+completed.stderr[-3500:])
    result = json.loads(completed.stdout)
    result['own_files_unchanged'] = len(before)
    return result


class FrozenBoundaryQualificationFeedbackAuthority:
    """Registered feedback from a completed operational evidence assessment."""
    def __init__(self, *, config, python):
        self.config = Path(config).resolve(); self.python = Path(python).absolute()
        self._identity = self._current_identity(); self._cached = None

    def _current_identity(self):
        return {'schema_version': 'dataevolver.boundary_qualification_feedback_authority.v1',
            'config': str(self.config), 'python': str(self.python), 'reader_sha256': file_sha256(Path(__file__)),
            'files': {str(p): file_sha256(p) for p in (*self.config.rglob('*'), self.python)
                      if p.is_file() and '__pycache__' not in p.parts}}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError('original operational qualification changed')
        return copy.deepcopy(self._identity)

    def read(self):
        self.identity()
        if self._cached is None:
            verified = verify_frozen_boundary_qualification(self.config, python=self.python)
            verified.pop('readback_seconds', None)
            actual = json.loads((self.config/'validation.json').read_text())
            admission = json.loads((self.config/'protected/admission.json').read_text())
            original = json.loads((self.config/'protected/feedback.json').read_text())
            origin_proof = json.loads((self.config/'protected/origin-proof.json').read_text())
            from .operator_development import DevelopmentBudget
            budget = DevelopmentBudget.from_dict(admission['budget'])
            if (actual['artifact_file_sha256'] != verified['artifact_sha256']
                    or actual['qualification_sha256'] != verified['qualification_sha256']
                    or actual['source_commit'] != verified['source_commit']
                    or any(actual['lifecycle_usage'][k] for k in ('model_calls', 'planner_calls', 'image_experiments'))
                    or actual['lifecycle_usage']['operator_calls'] != 1
                    or actual['lifecycle_usage']['output_bytes'] != sum(p.stat().st_size
                        for p in (self.config/'output').rglob('*') if p.is_file())
                    or budget.check_usage((actual['lifecycle_usage'],)) != actual['lifecycle_total']):
                raise ValueError('operational qualification usage or artifact identity changed')
            summary = {'schema_version': 'dataevolver.boundary_qualification_feedback.v1',
                'scope': 'registered_boundary_operational_qualification',
                'qualification': verified['qualification'],
                'completed_capabilities': ['provenance_bound_boundary_witness', 'independent_operational_evidence_qualifier'],
                'next_stage': 'Use the explicit unmet prerequisites to choose a distinct registered evidence '
                    'acquisition, refinement or production capability, or abstain. The witness and qualifier '
                    'are implemented and replayed; do not reimplement them or rerun unchanged probes. '
                    'Missing evidence is not an algorithm defect. Specific observations and any geometry remain runtime-owned.',
                'lifecycle_total': actual['lifecycle_total'],
                'origin': {'resource_deadline_epoch': budget.resource_deadline_epoch,
                    'source_commit': verified['source_commit'], 'artifact_sha256': verified['artifact_sha256'],
                    'qualification_sha256': verified['qualification_sha256'],
                    'package_sha256': original['origin']['package_sha256']},
                'image_submission_allowed': False, 'candidate_adoption_allowed': False,
                'production_qualification_passed': False, 'algorithm_defect_established': False,
                'new_accepted_records': 0, 'new_model_calls_for_feedback': 0}
            if 'acquisition' in verified:
                acquired = verified['acquisition']
                summary['completed_capabilities'].append('independent_evidence_acquisition_import')
                summary['acquisition'] = {k: acquired[k] for k in (
                    'provider_availability', 'providers', 'provider_gaps', 'accepted_evidence_records')}
                summary['next_stage'] = (
                    'The registered acquisition port is implemented and independently replayed. '
                    'It imports completed host-qualified evidence, and does not create a provider or authority. '
                    'Do not request another witness, qualifier, import wrapper or unchanged probe. '
                    'Choose an existing authorized provider capability and its concrete qualification prerequisites, '
                    'or explicitly retain the external evidence/authority blocker. Any provider implementation '
                    'must use the existing isolated candidate-build and independent production qualification '
                    'lifecycle with a bounded stopping rule; adding interfaces is not production progress. '
                    'Task observations and geometry remain runtime-owned.')
            self._cached = summary, {**self.identity(), 'verification': verified,
                'artifact_hashes': {**verified['artifact_hashes'], **self._identity['files']},
                'input_store_roots': origin_proof['input_store_roots']}
        for name, digest in self._cached[1]['artifact_hashes'].items():
            if file_sha256(Path(name)) != digest:
                raise ValueError('operational qualification original evidence changed')
        self.identity()
        return copy.deepcopy(self._cached)


def _verify_qualification(config, python):
    import json, sys, time
    from pathlib import Path
    from dataevolver.dataset.contracts import file_sha256, payload_sha256
    from dataevolver.harness.boundary_qualification import RegisteredBoundaryQualificationEvidence
    from dataevolver.harness.boundary_witness_readback import FrozenBoundaryWitnessFeedbackAuthority
    from dataevolver.harness.production_feedback_request import FrozenProductionFeedbackRequestAuthority
    from dataevolver.harness.adapters.boundary_qualification import BoundaryOperationalQualificationOperator
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.models import GoalSpec, OperatorCall, OperatorReceipt, EvidenceStatus, ControllerState
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.planner import DeterministicPlanner
    from dataevolver.harness.training_trial_authority import _read_only_event
    from dataevolver.adapters.model_gateway import ModelGateway
    cfg = Path(config); protected = cfg/'protected'; started = time.monotonic()
    host = json.loads((protected/'host.json').read_text())
    if host['qualification_evidence_kind'] != 'absent':
        raise ValueError('positive evidence requires its original registered production stage readers')
    witness = FrozenBoundaryWitnessFeedbackAuthority(config=host['witness_config'], python=python)
    request = FrozenProductionFeedbackRequestAuthority(config=host['request_config'],
        profile_path=host['profile_path'], python=python)
    evidence = RegisteredBoundaryQualificationEvidence()
    operator_type = BoundaryOperationalQualificationOperator
    if host.get('capture_kind') == 'independent_evidence_acquisition':
        from dataevolver.harness.boundary_qualification_readback import FrozenBoundaryQualificationFeedbackAuthority
        from dataevolver.harness.evidence_acquisition import RegisteredIndependentEvidenceProvider
        from dataevolver.harness.adapters.evidence_acquisition import IndependentEvidenceAcquisitionOperator
        original = FrozenBoundaryQualificationFeedbackAuthority(config=host['qualification_config'], python=python)
        op = IndependentEvidenceAcquisitionOperator(authority=original, witness_authority=witness,
            request_authority=request, provider=RegisteredIndependentEvidenceProvider(evidence_authority=evidence),
            output_root=cfg/'output')
        operator_type = IndependentEvidenceAcquisitionOperator
    else:
        op = BoundaryOperationalQualificationOperator(authority=witness, request_authority=request,
            evidence_authority=evidence, output_root=cfg/'output')
    registry = OperatorRegistry(); registry.register(op)
    goal = GoalSpec.from_dict(json.loads((protected/'goal.json').read_text()))
    def forbidden(*a, **kw):
        raise RuntimeError('qualification readback cannot infer, execute or grant authority')
    EvolutionController.run = EvolutionController.tick = forbidden
    operator_type.execute = operator_type.recover = forbidden
    ModelGateway.invoke = ModelGateway._provider = ModelGateway.adopt_completed = forbidden
    sys.addaudithook(_read_only_event)
    controller = EvolutionController.open(repo_root=cfg/'source', campaigns_root=cfg/'output/campaigns',
        goal=goal, registry=registry, planner=DeterministicPlanner(()),
        evaluator_contract=EvaluatorContract.from_path(cfg/'evaluator.json'),
        schema_paths=tuple(protected.rglob('*.json')),
        execution_paths=tuple((cfg/'source/src/dataevolver').rglob('*.py'))+(cfg/'host-launcher.py',))
    state = controller.status()
    if (state.state is not ControllerState.COMPLETE or not controller._frozen()[0]
            or controller.replay().to_dict() != state.to_dict() or state.data.get('pending_call')
            or state.operator_calls != 1 or state.planner_calls != 0):
        raise ValueError('qualification diagnostic must be complete, frozen and replayable')
    events = controller.store.events.read()
    planned = [e for e in events if e.event_type == 'ACTION_PLANNED']
    committed = [e for e in events if e.event_type == 'ACTION_COMMITTED']
    if len(planned) != 1 or len(committed) != 1:
        raise ValueError('qualification diagnostic needs exactly one registered effect')
    call = OperatorCall.from_dict(planned[0].payload['operator_call'])
    receipt = OperatorReceipt.from_dict(committed[0].payload['receipt'])
    verified = op.verify_effect(call, receipt)
    if (verified.status is not EvidenceStatus.VERIFIED
            or verified.artifact_hashes != committed[0].payload['effect_verification']['artifact_hashes']):
        raise ValueError('qualification effect or original evidence no longer verifies')
    path = op._validate(call); captured = json.loads(path.read_text())['inspection']; value = captured['qualification']
    if value['status'] != 'INSUFFICIENT_EVIDENCE' or value['qualified_production_chain'] is not None:
        raise ValueError('absent evidence cannot be upgraded during independent readback')
    result = {'schema_version': 'dataevolver.frozen_boundary_qualification_readback.v1',
        'config': str(cfg), 'source_commit': (cfg/'source-commit').read_text().strip(),
        'frozen_valid': True, 'replay_matches_snapshot': True, 'registered_effect_verified': True,
        'qualification': value, 'artifact_path': str(path), 'artifact_sha256': file_sha256(path),
        'qualification_sha256': payload_sha256(value), 'artifact_hashes': verified.artifact_hashes,
        'artifact_closure_files': len(verified.artifact_hashes), 'new_model_calls': 0,
        'new_operator_executions': 0, 'new_planner_decisions': 0,
        'image_submission_allowed': False, 'candidate_adoption_allowed': False,
        'production_qualification_passed': False, 'readback_seconds': time.monotonic()-started}
    if 'acquisition' in captured:
        result['acquisition'] = captured['acquisition']
    return result
