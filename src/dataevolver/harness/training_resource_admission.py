"""Readiness evidence using the training Gateway's existing resource policy.

This does not load a model or reserve a model request. A completed observation
is only a preflight; each actual request must still run its own Gateway wait.
"""
import json
import os
import time
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .models import EffectVerification, EvidenceStatus
from .state_store import CampaignLease, atomic_write_json
from .adapters.training_supervision import TrainingSupervisionOperator


class TrainingResourceAdmission:
    max_output_bytes = 65536

    def __init__(self, *, operator, workload_operator_id, workload_operator_version,
                 roles, output_root, allowed_gpu_uuids):
        if type(operator) is not TrainingSupervisionOperator:
            raise ValueError('resource admission requires the existing training operator')
        if (not roles or len(roles) > 7 or len(set(roles)) != len(roles)
                or not allowed_gpu_uuids or len(set(allowed_gpu_uuids)) != len(allowed_gpu_uuids)):
            raise ValueError('explicit bounded roles and physical GPU identities are required')
        if not operator.fixture and operator.custom_wait is not None:
            raise ValueError('production resource admission cannot replace the Gateway wait')
        self.operator, self.roles = operator, tuple(roles)
        self.workload = (workload_operator_id, workload_operator_version)
        self.root = Path(output_root).absolute()
        self.gpus = tuple(allowed_gpu_uuids)
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError('resource evidence root cannot contain symlinks')
        self.root = self.root.resolve()
        protected = (*operator.store.roots.values(), operator.root)
        if any(self.root.is_relative_to(p) or p.is_relative_to(self.root) for p in protected):
            raise ValueError('resource evidence root overlaps a protected input or workload root')
        for role in self.roles:
            self._worker(role)
        self._identity = self._current_identity()

    def _worker(self, role):
        op = self.operator
        policy = op.prompt_policy if role.startswith('training.refine_') else op.policy
        if policy is None or role not in policy.bindings:
            raise ValueError('resource role is not in the original model policy')
        worker = op.contract['models']['workers'][policy.bindings[role]]
        if not op.fixture:
            # CPU workers use no GPU; CUDA workers must retain an explicit
            # resident identity. No automatic GPU selection or model loading.
            if worker.get('kind') not in {'pe_i2i', 'pe_t2i'} and worker.get('execution_device') != 'cpu':
                if not worker.get('resident'):
                    raise ValueError('qualification cannot load a nonresident CUDA worker')
                from dataevolver.adapters.training_media_resident import identity
                placement = identity(worker)
                if tuple(placement['gpu_uuids']) != self.gpus:
                    raise ValueError('resident GPU mapping differs from the explicit allowed GPUs')
            elif worker.get('kind') in {'pe_i2i', 'pe_t2i'}:
                from dataevolver.adapters.training_prompt_enhancement import pe_identity
                placement = pe_identity(worker)
                model = placement['models'][worker['kind']]
                if set(map(str, model['device_map'].values())) != {'cpu'} and placement.get('gpu_uuid') not in self.gpus:
                    raise ValueError('PE GPU mapping differs from the explicit allowed GPUs')
        return worker

    def _current_identity(self):
        if not self.operator.fixture and (self.operator.custom_wait is not None
                or getattr(self.operator._wait, '__func__', None) is not TrainingSupervisionOperator._wait):
            raise ValueError('production resource admission cannot replace the Gateway wait')
        return {'schema_version': 'dataevolver.training_resource_admission.v1',
                'adapter_sha256': file_sha256(Path(__file__)),
                'wait_implementation_sha256': file_sha256(Path(__file__).parent/'adapters/training_supervision.py'),
                'model_contract_sha256': payload_sha256(self.operator.contract),
                'workload': list(self.workload), 'roles': list(self.roles),
                'allowed_gpu_uuids': list(self.gpus), 'output_root': str(self.root),
                'protected_roots': [str(p) for p in (*self.operator.store.roots.values(), self.operator.root)],
                'evidence_kind': 'fixture' if self.operator.fixture else 'registered_runtime',
                'max_output_bytes': self.max_output_bytes,
                'per_request_gateway_admission_required': True}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError('resource admission implementation or model contract changed')
        return dict(self._identity)

    def _path(self, call):
        if (call.operator_id, call.operator_version) != self.workload:
            raise ValueError('resource evidence belongs to another registered workload')
        path = self.root/payload_sha256(call.to_dict())/'resource-admission.json'
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('resource evidence path cannot contain symlinks')
        return path

    def _proof(self, path):
        value = json.loads(path.read_text())
        hashes = {str(path): file_sha256(path), **value['files']}
        return {'admitted': value['admitted'], 'reason': value['reason'],
                'call_sha256': value['call_sha256'],
                'resource_deadline_epoch': value['resource_deadline_epoch'],
                'per_request_gateway_admission_required': True,
                'report_path': str(path), 'artifact_hashes': hashes}

    def verify(self, call, proof):
        """Verify immutable observations; never poll health or wait again."""
        try:
            path = self._path(call)
            value = json.loads(path.read_text())
            if (value['identity'] != self.identity() or proof != self._proof(path)
                    or value['call_sha256'] != payload_sha256(call.to_dict())
                    or value['resource_deadline_epoch'] != self.operator.contract['production']['resource_deadline_epoch']):
                raise ValueError('resource admission evidence changed')
            for name, digest in proof['artifact_hashes'].items():
                if not Path(name).resolve().is_relative_to(path.parent) or file_sha256(Path(name)) != digest:
                    raise ValueError('resource observation file changed or escaped its owner')
            if value['admitted'] and (len(value['observations']) != len(self.roles)
                    or any(x['role'] != role or x['status'].get('role') != role
                           or x['status'].get('stable_observations', 0) < 2
                           for role, x in zip(self.roles, value['observations']))):
                raise ValueError('readiness requires the original two stable observations')
            return EffectVerification(EvidenceStatus.VERIFIED, 'original Gateway resource observations verified',
                                      artifact_hashes=proof['artifact_hashes'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def preflight(self, call, resource_deadline_epoch):
        path = self._path(call)
        self.identity()
        if resource_deadline_epoch != self.operator.contract['production']['resource_deadline_epoch']:
            raise ValueError('resource admission cannot extend or replace the original deadline')
        self.root.mkdir(parents=True, exist_ok=True)
        with CampaignLease(self.root/'.resource-admission.lease', ttl_seconds=3600):
            if path.exists():
                proof = self._proof(path)
                if self.verify(call, proof).status is not EvidenceStatus.VERIFIED:
                    raise ValueError('existing resource evidence does not verify')
                return proof
            if (path.parent/'intent.json').exists():
                raise ValueError('resource observation is incomplete; do not reset its owner')
            started = time.monotonic()
            atomic_write_json(path.parent/'intent.json', {'identity': self.identity(),
                'call_sha256': payload_sha256(call.to_dict()), 'at': time.time(),
                'resource_deadline_epoch': resource_deadline_epoch})
            # _wait only observes readiness and preserves the existing deadline
            # behavior: unavailable after expiry refuses; available still needs
            # two observations 30 seconds apart. No Gateway/model is invoked.
            from dataclasses import replace
            owned_call = replace(call, work_dir=str(path.parent))
            observations, admitted, reason = [], True, 'GATEWAY_RESOURCES_OBSERVED'
            prior_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                for index, role in enumerate(self.roles):
                    self._worker(role)
                    try:
                        self.operator._wait(owned_call, role)
                    except TimeoutError as exc:
                        admitted, reason = False, str(exc)
                    status = json.loads((path.parent/'resource-status.json').read_text())
                    observations.append({'role': role, 'status': status})
                    atomic_write_json(path.parent/('role-'+str(index)+'.json'), observations[-1])
                    if not admitted:
                        break
            finally:
                if prior_cuda is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = prior_cuda
            if admitted and any(o['status'].get('stable_observations', 0) < 2 for o in observations):
                raise ValueError('resource role did not retain the required stable observations')
            files = {str(p): file_sha256(p) for p in path.parent.glob('*.json')}
            atomic_write_json(path, {'identity': self.identity(),
                'call_sha256': payload_sha256(call.to_dict()), 'resource_deadline_epoch': resource_deadline_epoch,
                'admitted': admitted, 'reason': reason, 'observations': observations, 'files': files,
                'observed_wall_seconds': time.monotonic()-started,
                'new_model_calls': 0, 'new_image_calls': 0, 'per_request_gateway_admission_required': True})
            proof = self._proof(path)
            if sum(Path(p).stat().st_size for p in proof['artifact_hashes']) > self.max_output_bytes:
                raise ValueError('resource evidence exceeds its frozen output bound')
            if self.verify(call, proof).status is not EvidenceStatus.VERIFIED:
                raise ValueError('resource admission did not verify')
            return proof
