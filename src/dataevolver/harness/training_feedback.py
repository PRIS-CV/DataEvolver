"""One external-training receipt adapter; no training framework or optimizer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

from dataevolver.dataset.contracts import ArtifactRef, Contract, canonical_json, identifier, payload_sha256
from .feedback import FeedbackSignal
from .state_store import atomic_write_json


@dataclass(frozen=True)
class TrainingJobSpec(Contract):
    schema: ClassVar[str] = "dataevolver.training_job.v1"
    job_id: str
    dataset: ArtifactRef
    initialization: ArtifactRef
    training_config: ArtifactRef
    evaluation: ArtifactRef
    model: str
    seed: int
    evaluation_split: str = "development"

    def __post_init__(self):
        identifier(self.job_id)
        if not self.model or type(self.seed) is not int or self.seed < 0:
            raise ValueError("training job must freeze model and seed")
        if self.evaluation_split not in {"development", "validation", "sealed"}:
            raise ValueError("unknown training evaluation split")
        for name in ('dataset', 'initialization', 'training_config', 'evaluation'):
            if not isinstance(getattr(self, name), ArtifactRef):
                raise ValueError("training inputs must be immutable ArtifactRef values")

    @classmethod
    def from_dict(cls, value):
        values = cls.read_fields(value)
        for name in ('dataset', 'initialization', 'training_config', 'evaluation'):
            values[name] = ArtifactRef.from_dict(values[name])
        return cls(**values)

    def to_dict(self):
        result = super().to_dict()
        for name in ('dataset', 'initialization', 'training_config', 'evaluation'):
            result[name] = getattr(self, name).to_dict()
        return result


@dataclass(frozen=True)
class TrainingFeedback(Contract):
    schema: ClassVar[str] = "dataevolver.training_feedback.v1"
    job_sha256: str
    receipt: ArtifactRef
    checkpoint: ArtifactRef
    metrics: dict
    slices: dict
    usage: dict

    def __post_init__(self):
        if len(self.job_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.job_sha256):
            raise ValueError("training feedback must bind a frozen job")
        if not all(isinstance(x, ArtifactRef) for x in (self.receipt, self.checkpoint)):
            raise ValueError('training outputs need exact references')
        canonical_json([self.metrics, self.slices, self.usage])
        for name, metric in self.metrics.items():
            if not name or type(metric) not in (int, float):
                raise ValueError('training metrics must be finite numbers')

    @classmethod
    def from_dict(cls, value):
        fields = cls.read_fields(value)
        for name in ('receipt', 'checkpoint'):
            fields[name] = ArtifactRef.from_dict(fields[name])
        return cls(**fields)

    def to_dict(self):
        value = super().to_dict()
        value.update(receipt=self.receipt.to_dict(), checkpoint=self.checkpoint.to_dict())
        return value


class TrainerAdapter(Protocol):
    def prepare(self, job: TrainingJobSpec): ...
    def submit(self, job: TrainingJobSpec, *, external_job_id: str): ...
    def status(self, job: TrainingJobSpec): ...
    def collect(self, job: TrainingJobSpec, receipt: ArtifactRef) -> TrainingFeedback: ...


class ExternalTrainingRunAdapter:
    """Register a real external job and collect its exact file-backed receipt.

    Submission is deliberately semi-automatic: the adapter records a scheduler
    identity, not shell code. A successful registration is not completed training.
    """
    def __init__(self, *, store, work_root):
        self.store, self.root = store, Path(work_root).resolve()

    def _path(self, job):
        return self.root / (job.job_id + '.job.json')

    def _validate(self, job):
        for name in ('dataset', 'initialization', 'training_config', 'evaluation'):
            self.store.resolve(getattr(job, name))

    def prepare(self, job):
        self._validate(job)
        path = self._path(job)
        value = {'job': job.to_dict(), 'job_sha256': job.compute_hash(), 'external_job_id': None}
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['job'] != value['job'] or existing['job_sha256'] != value['job_sha256']:
                raise ValueError('training job identity changed')
            return existing
        atomic_write_json(path, value)
        return value

    def submit(self, job, *, external_job_id):
        identifier(external_job_id, 'external job identity')
        value = self.prepare(job)
        if value['external_job_id'] not in (None, external_job_id):
            raise ValueError('external training submission identity changed')
        value['external_job_id'] = external_job_id
        atomic_write_json(self._path(job), value)
        return {'status': 'REGISTERED_EXTERNAL_JOB', **value}

    def status(self, job):
        value = self.prepare(job)
        if value.get('receipt'):
            result = self.collect(job, ArtifactRef.from_dict(value['receipt']), _record=False)
            return {'status': 'COMPLETE', 'job_sha256': result.job_sha256,
                    'external_job_id': value['external_job_id'], 'receipt': value['receipt']}
        return {'status': 'WAITING_EXTERNAL_RECEIPT' if value['external_job_id'] else 'AWAITING_EXTERNAL_SUBMISSION',
                'job_sha256': value['job_sha256'], 'external_job_id': value['external_job_id']}

    def collect(self, job, receipt, *, _record=True):
        self._validate(job)
        registered = self.prepare(job)
        raw = json.loads(self.store.resolve(receipt).read_text())
        expected = {'job_sha256': job.compute_hash(), 'dataset_sha256': job.dataset.sha256,
            'initialization_sha256': job.initialization.sha256, 'training_config_sha256': job.training_config.sha256,
            'evaluation_sha256': job.evaluation.sha256, 'model': job.model, 'seed': job.seed,
            'evaluation_split': job.evaluation_split}
        if (any(raw.get(k) != v for k, v in expected.items()) or type(raw.get('exit_code')) is not int
                or raw['exit_code'] != 0 or not raw.get('external_job_id')
                or raw['external_job_id'] != registered['external_job_id']):
            raise ValueError('training receipt is failed, incomplete or bound to another job')
        checkpoint = ArtifactRef.from_dict(raw['checkpoint'])
        self.store.resolve(checkpoint)
        try:
            config = json.loads(self.store.resolve(job.training_config).read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            config = None  # Existing external adapters may bind non-JSON trainer configs.
        if isinstance(config, dict) and config.get('schema_version') == 'dataevolver.lora_training.v1':
            from dataevolver.adapters.lora_training import validate_training_proof
            validate_training_proof(job, raw, self.store)
        result = TrainingFeedback(job.compute_hash(), receipt, checkpoint, raw['metrics'], raw.get('slices', {}), raw.get('usage', {}))
        if _record and registered.get('receipt') != receipt.to_dict():
            registered['receipt'] = receipt.to_dict()
            atomic_write_json(self._path(job), registered)
        return result

    def feedback_signal(self, job, receipt):
        if job.evaluation_split == 'sealed':
            raise ValueError('sealed training results cannot become optimization feedback')
        feedback = self.collect(job, receipt)
        return FeedbackSignal('training_result', (receipt,),
            {'job_sha256': job.compute_hash(), 'dataset': job.dataset.to_dict(),
             'evaluation': job.evaluation.to_dict(), 'evaluation_split': job.evaluation_split},
            {'task': feedback.metrics, 'slices': feedback.slices, 'training_benefit_verified': False}, 'supported')

    def next_data_plan_context(self, job, receipt):
        signal = self.feedback_signal(job, receipt)
        return {'feedback': [signal.to_dict()], 'previous_dataset': job.dataset.to_dict(),
                'allowed_changes': ['sampling', 'family_deduplication'],
                'requires_new_frozen_run': True, 'training_benefit_verified': False}
