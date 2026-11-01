"""Independently re-open a witness capture with its own frozen framework."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess

from dataevolver.dataset.contracts import file_sha256, payload_sha256


def verify_frozen_boundary_witness(config, *, python):
    cfg = Path(config).resolve()
    def tree():
        return {str(p): file_sha256(p) for p in cfg.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
    before = tree()
    script = 'import json,sys\n'+inspect.getsource(_verify_witness)
    script += '\nprint(json.dumps(_verify_witness(*json.load(sys.stdin))))'
    completed = subprocess.run([str(python), '-B', '-c', script],
        input=json.dumps([str(cfg), str(python)]), capture_output=True, text=True,
        cwd=cfg/'source', timeout=600, env={**os.environ, 'PYTHONPATH': str(cfg/'source/src'),
            'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '', 'GIT_OPTIONAL_LOCKS': '0'})
    if tree() != before:
        raise ValueError('independent witness readback changed its original capture')
    if completed.returncode:
        raise ValueError('frozen witness readback failed: '+completed.stderr[-3500:])
    value = json.loads(completed.stdout)
    value['own_files_unchanged'] = len(before)
    return value


class FrozenBoundaryWitnessFeedbackAuthority:
    """Expose an independently verified witness to the existing runtime domain."""
    def __init__(self, *, config, python):
        self.config = Path(config).resolve()
        self.python = Path(python).absolute()
        self._identity = self._current_identity()
        self._cached = None

    def _current_identity(self):
        return {'schema_version': 'dataevolver.boundary_witness_feedback_authority.v1',
            'config': str(self.config), 'python': str(self.python),
            'reader_sha256': file_sha256(Path(__file__)),
            'files': {str(p): file_sha256(p) for p in (*self.config.rglob('*'), self.python)
                      if p.is_file() and '__pycache__' not in p.parts}}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError('original boundary witness capture changed')
        return copy.deepcopy(self._identity)

    def read(self):
        self.identity()
        if self._cached is None:
            verified = verify_frozen_boundary_witness(self.config, python=self.python)
            # Timing is audit data, not a stable authority or OperatorSpec input.
            verified.pop('readback_seconds', None)
            actual = json.loads((self.config/'validation.json').read_text())
            admission = json.loads((self.config/'protected/admission.json').read_text())
            from .operator_development import DevelopmentBudget
            budget = DevelopmentBudget.from_dict(admission['budget'])
            if (actual['artifact_file_sha256'] != verified['artifact_sha256']
                    or actual['witness_sha256'] != verified['witness_sha256']
                    or actual['source_commit'] != verified['source_commit']
                    or any(actual['lifecycle_usage'][k] for k in ('model_calls', 'planner_calls', 'image_experiments'))
                    or actual['lifecycle_usage']['operator_calls'] != 1
                    or actual['lifecycle_usage']['output_bytes'] != sum(p.stat().st_size
                        for p in (self.config/'output').rglob('*') if p.is_file())
                    or budget.check_usage((actual['lifecycle_usage'],)) != actual['lifecycle_total']):
                raise ValueError('boundary witness usage or artifact identity changed')
            summary = {'schema_version': 'dataevolver.boundary_witness_feedback.v1',
                'scope': 'registered_boundary_witness', 'witness': verified['witness'],
                'validation': verified['validation'], 'completed_capability': 'provenance_bound_boundary_witness',
                'artifact_conformance_is_operational_qualification': False,
                'next_stage': 'Distinct operational evidence, refinement or qualification capability; do not repeat '
                    'the already implemented witness contract or resubmit unchanged probes.',
                'lifecycle_total': actual['lifecycle_total'],
                'origin': {'resource_deadline_epoch': budget.resource_deadline_epoch,
                    'source_commit': verified['source_commit'], 'artifact_sha256': verified['artifact_sha256'],
                    'witness_sha256': verified['witness_sha256'],
                    'package_sha256': json.loads((self.config/'protected/feedback.json').read_text())['origin']['package_sha256']},
                'image_submission_allowed': False, 'candidate_adoption_allowed': False,
                'production_qualification_passed': False, 'algorithm_defect_established': False,
                'new_accepted_records': 0, 'new_model_calls_for_feedback': 0}
            self._cached = (summary, {**self.identity(), 'verification': verified,
                'artifact_hashes': {**verified['artifact_hashes'], **self._identity['files']},
                'input_store_roots': verified['input_store_roots']})
        for name, digest in self._cached[1]['artifact_hashes'].items():
            if file_sha256(Path(name)) != digest:
                raise ValueError('boundary witness original evidence changed')
        self.identity()
        return copy.deepcopy(self._cached)


def _verify_witness(config, python):
    import json, sys, time
    from pathlib import Path
    from dataevolver.dataset.contracts import file_sha256, payload_sha256
    from dataevolver.harness.boundary_feedback import FrozenBoundaryFeedbackAuthority
    from dataevolver.harness.production_feedback_request import FrozenProductionFeedbackRequestAuthority
    from dataevolver.harness.adapters.boundary_witness import BoundaryWitnessOperator
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.models import GoalSpec, OperatorCall, OperatorReceipt, EvidenceStatus, ControllerState
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.planner import DeterministicPlanner
    from dataevolver.harness.training_trial_authority import _read_only_event
    from dataevolver.adapters.model_gateway import ModelGateway
    started = time.monotonic(); cfg = Path(config); protected = cfg/'protected'
    host = json.loads((protected/'host.json').read_text())
    authority = FrozenBoundaryFeedbackAuthority(config=host['recovery_config'],
        descriptor=host['boundary_descriptor'], python=python)
    request = FrozenProductionFeedbackRequestAuthority(config=host['request_config'],
        profile_path=host['profile_path'], python=python)
    op = BoundaryWitnessOperator(authority=authority, request_authority=request, output_root=cfg/'output')
    registry = OperatorRegistry(); registry.register(op)
    goal = GoalSpec.from_dict(json.loads((protected/'goal.json').read_text()))
    def forbidden(*a, **kw):
        raise RuntimeError('independent boundary witness reader cannot execute effects')
    EvolutionController.run = EvolutionController.tick = forbidden
    BoundaryWitnessOperator.execute = BoundaryWitnessOperator.recover = forbidden
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
            or state.planner_calls != 0 or state.operator_calls != 1):
        raise ValueError('boundary witness capture must be complete, frozen and replayable')
    events = controller.store.events.read()
    planned = [e for e in events if e.event_type == 'ACTION_PLANNED']
    committed = [e for e in events if e.event_type == 'ACTION_COMMITTED']
    if len(planned) != 1 or len(committed) != 1:
        raise ValueError('boundary witness capture must have one registered effect')
    call = OperatorCall.from_dict(planned[0].payload['operator_call'])
    receipt = OperatorReceipt.from_dict(committed[0].payload['receipt'])
    verification = op.verify_effect(call, receipt)
    if (verification.status is not EvidenceStatus.VERIFIED
            or verification.artifact_hashes != committed[0].payload['effect_verification']['artifact_hashes']):
        raise ValueError('boundary witness original provenance closure no longer verifies')
    path = op._validate(call); artifact = json.loads(path.read_text())['inspection']
    return {'schema_version': 'dataevolver.frozen_boundary_witness_readback.v1',
        'config': str(cfg), 'source_commit': (cfg/'source-commit').read_text().strip(),
        'frozen_valid': True, 'replay_matches_snapshot': True, 'registered_effect_verified': True,
        'witness': artifact['witness'], 'validation': artifact['validation'],
        'artifact_path': str(path), 'artifact_sha256': file_sha256(path),
        'witness_sha256': payload_sha256(artifact['witness']),
        'artifact_closure_files': len(verification.artifact_hashes),
        'artifact_hashes': verification.artifact_hashes,
        'input_store_roots': authority.read()[1]['input_store_roots'],
        'new_model_calls': 0, 'new_operator_executions': 0, 'new_planner_decisions': 0,
        'production_qualification_passed': False, 'image_submission_allowed': False,
        'candidate_adoption_allowed': False, 'readback_seconds': time.monotonic()-started}
