"""Finite compiled-data -> external-training workflow in one Controller effect.

The graph, runner and job template are host-bound before execution. This adapter
does not choose a model, expand a budget, retry an uncertain process or train an
optimizer itself. The existing child journals own all resumable effects.
"""
from __future__ import annotations

import json
from pathlib import Path

from dataevolver.adapters.external_training import RegisteredTrainingRunner
from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.dataset.supervision import load_dataset
from ..artifact_operator import ArtifactWorkloadOperator
from ..models import EvidenceStatus, StageStatus
from ..state_store import atomic_write_json
from ..training_feedback import TrainingJobSpec
from ..data_contracts import DataNodeSpecV3, DataNodeGraphV3, DataRunBudget, DataInput
from ..task_contracts import DatasetIntent, QualityPolicy, PROTECTED_GATES
from ..plan_compiler import CompileContext, CompiledRun, compile_plan, builtin_domains


class DataTrainingWorkflowOperator(ArtifactWorkloadOperator):
    def verified_capability_level(self, state):
        if (state.state.value != 'COMPLETE'
                or state.data.get('last_effect_verification', {}).get('status') != 'VERIFIED'):
            return 'designed'
        # Record validation already rejects mock media and mock reload inference.
        return 'real_execution_verified'

    def __init__(self, *, compiled, registry, roots, runner, job_template,
                 delivery_node, runner_inputs, result_store, output_root,
                 family_ledger, data_call=None, compiled_slots=(), delivery_parameters=None):
        if not isinstance(runner, RegisteredTrainingRunner):
            raise TypeError('workflow requires an explicitly registered external trainer')
        self.graph = compiled.bind(registry, allowed_write_root=output_root, roots=roots)
        self.runner, self.template = runner, dict(job_template)
        self.delivery_node, self.runner_inputs = delivery_node, dict(runner_inputs)
        self.result_store = result_store
        self.family_ledger, self.data_call = family_ledger, data_call
        self.registry, self.roots = registry, dict(roots)
        self.slot_definitions = tuple(compiled_slots)
        if self.slot_definitions and data_call is not None:
            raise ValueError('choose frozen slot alternatives or one existing data call')
        self.delivery_parameters = dict(delivery_parameters or {})
        self.slots = tuple(tuple(c.bind(registry, allowed_write_root=output_root, roots=roots)
                                for c in slot['attempts']) for slot in self.slot_definitions)
        if any(not 1 <= len(slot) <= 3 for slot in self.slots) or (self.slots and not self.delivery_parameters):
            raise ValueError('slot alternatives and delivery parameters must be finitely frozen')
        if data_call is not None:
            self.graph._validate(data_call)
        for name in ('initialization', 'training_config', 'evaluation'):
            runner.store.resolve(ArtifactRef.from_dict(self.template[name]))
        super().__init__(operator_id='data.training_workflow', output_root=output_root,
            configuration={'compiled': compiled.to_dict(), 'job_template': job_template,
                'delivery_node': delivery_node, 'runner_inputs': runner_inputs,
                'runner_command': runner.command, 'runner_sources': runner.sources,
                'runner_environment_sha256': payload_sha256(runner.environment),
                'runner_timeout': runner.timeout, 'result_store': result_store,
                'data_call': data_call.to_dict() if data_call is not None else None,
                'family_ledger': {'path': str(family_ledger.path), 'namespace': family_ledger.namespace,
                                  'hamming_distance': family_ledger.distance},
                'slot_graphs': [{'record_node': slot['record_node'],
                    'attempts': [c.to_dict() for c in slot['attempts']]} for slot in self.slot_definitions],
                'delivery_parameters': self.delivery_parameters,
                'stores': {k: str(v) for k, v in runner.store.roots.items()}})
        self.spec = self.spec.with_updates(allowed_write_roots=tuple(sorted({str(self.output_root),
            str(self.graph.dataset_root), str(runner.root), str(runner.store.roots[result_store])})))
        self._declare_contract()

    def child(self, call):
        if self.data_call is not None:
            return self.data_call
        return call.with_updates(call_id=call.call_id + ':data', idempotency_key=call.idempotency_key + ':data',
            operator_id=self.graph.spec.operator_id, operator_version=self.graph.spec.version,
            work_dir=str(Path(call.work_dir) / 'data'), parameters=self.graph.contract,
            matched_trial_key=None)

    def _job(self, call):
        data_call = self.export_call(call) if self.slots else self.child(call)
        node = Path(data_call.work_dir) / 'node_runs' / self.delivery_node / 'node_run.json'
        output = ArtifactRef.from_dict(json.loads(node.read_text())['outputs']['delivery'])
        delivery = json.loads(self.graph.store.resolve(output).read_text())
        relative = delivery['bundle_relative'] + '/manifest.json'
        manifest = ArtifactRef(self.graph.dataset_store, relative, delivery['manifest_sha256'])
        source = self.graph.store.resolve(manifest)
        load_dataset(source.parent)
        self.family_ledger.require_release(delivery['release_id'],
            [r['id'] for r in json.loads(source.read_text())['records']], unique_families=True)
        job = TrainingJobSpec.from_dict({'schema_version': TrainingJobSpec.schema,
                                         **self.template, 'dataset': manifest.to_dict()})
        path = Path(call.work_dir) / 'training-job.json'
        if path.exists() and json.loads(path.read_text()) != job.to_dict():
            raise ValueError('next frozen training job differs from the delivered data')
        if not path.exists():
            atomic_write_json(path, job.to_dict())
        return job, delivery, output

    def slot_call(self, call, slot, attempt):
        op = self.slots[slot][attempt]
        return call.with_updates(call_id=f'{call.call_id}:s{slot}:a{attempt}',
            idempotency_key=f'{call.idempotency_key}:s{slot}:a{attempt}', operator_id=op.spec.operator_id,
            operator_version=op.spec.version, work_dir=str(Path(call.work_dir) / 'slots' / str(slot) / str(attempt)),
            parameters=op.contract, matched_trial_key=None)

    def export_call(self, call):
        compiled = CompiledRun.from_dict(json.loads((Path(call.work_dir) / 'export-compiled.json').read_text()))
        return call.with_updates(call_id=call.call_id + ':export', idempotency_key=call.idempotency_key + ':export',
            operator_id='data.graph', operator_version='1', work_dir=str(Path(call.work_dir) / 'export'),
            parameters=compiled.contract, matched_trial_key=None)

    def export_graph(self, call):
        compiled = CompiledRun.from_dict(json.loads((Path(call.work_dir) / 'export-compiled.json').read_text()))
        return compiled.bind(self.registry, allowed_write_root=self.output_root,
            roots={self.accepted_store: Path(call.work_dir) / 'accepted-inputs',
                   self.graph.dataset_store: self.graph.dataset_root})

    @property
    def accepted_store(self):
        return 'accepted' if self.graph.dataset_store != 'accepted' else 'accepted_sources'

    @staticmethod
    def replaceable(op, child, receipt):
        code = receipt.stage_result.get('failure_code')
        if code == 'MODEL_QUALITY_REJECTED':
            return True
        if code != 'INVALID_DATA_EFFECT':
            return False
        # Only a confirmed failed/uncertain semantic observation may authorize a
        # new frozen candidate. Hash errors and ambiguous effects do not.
        from dataevolver.dataset.visual_steps import confirmed
        for path in (Path(child.work_dir) / 'node_runs').glob('*/effect.json'):
            value = json.loads(path.read_text())
            for port, role in [('review', 'visual.review.image'), ('observation', 'visual.inspect')]:
                if port not in value['outputs']:
                    continue
                envelope = json.loads(op.store.resolve(ArtifactRef.from_dict(value['outputs'][port])).read_text())
                _, native = confirmed(op.gateway(child), envelope, {role})
                if native.output.get('status') in {'fail', 'uncertain'}:
                    return True
        return False

    def produce_slots(self, call):
        rows, trials = [], []
        for slot, operators in enumerate(self.slots):
            accepted = False
            for attempt, op in enumerate(operators):
                child = self.slot_call(call, slot, attempt)
                receipt = op.execute(child)
                if op.verify_effect(child, receipt).status is not EvidenceStatus.VERIFIED:
                    raise ValueError('slot effect cannot be reconciled')
                trials.append({'slot': slot, 'attempt': attempt, 'receipt': receipt.to_dict()})
                if receipt.status is StageStatus.SUCCESS:
                    node = Path(child.work_dir) / 'node_runs' / self.slot_definitions[slot]['record_node'] / 'node_run.json'
                    ref = ArtifactRef.from_dict(json.loads(node.read_text())['outputs']['record'])
                    rows.append(json.loads(op.store.resolve(ref).read_text()))
                    accepted = True
                    break
                if not self.replaceable(op, child, receipt):
                    break
            if not accepted:
                return None, trials
        root = Path(call.work_dir) / 'accepted-inputs'
        source = root / 'records.json'
        value = {'records': rows}
        if source.exists() and json.loads(source.read_text()) != value:
            raise ValueError('confirmed slot membership changed')
        if not source.exists():
            atomic_write_json(source, value)
        ref = ArtifactRef(self.accepted_store, 'records.json', file_sha256(source))
        node = DataNodeSpecV3(self.delivery_node, 'supervision.export_t2i', '1', 'data',
                             {'records': DataInput(artifact=ref)}, self.delivery_parameters)
        contract = {'intent': DatasetIntent('verified-delivery', 'curate', 'text_to_image_generation',
            'dataevolver.supervision.v1', 'Confirmed records from frozen slot alternatives', (ref,)).to_dict(),
            'quality_policy': QualityPolicy('verified-delivery', tuple(sorted(PROTECTED_GATES)),
                ('supervision.export_t2i',), {}, 1).to_dict(), 'graph': DataNodeGraphV3((node,)).to_dict(),
            'budget': DataRunBudget(1, 0, 0, len(rows), self.graph.budget.max_output_bytes, 120).to_dict(),
            'stores': {self.accepted_store: 'HOST_ACCEPTED', self.graph.dataset_store: 'HOST_DATASET'},
            'dataset_store': self.graph.dataset_store, 'models': {}}
        compiled = compile_plan({'domain': 'text_to_image_generation', 'contract': contract},
            CompileContext(self.registry, builtin_domains(), ('supervision.export_t2i',),
                roots={self.accepted_store: root, self.graph.dataset_store: self.graph.dataset_root}))
        if not isinstance(compiled, CompiledRun):
            raise ValueError('committed slot delivery did not compile')
        path = Path(call.work_dir) / 'export-compiled.json'
        if path.exists() and json.loads(path.read_text()) != compiled.to_dict():
            raise ValueError('next frozen delivery graph changed')
        if not path.exists():
            atomic_write_json(path, compiled.to_dict())
        op, child = self.export_graph(call), self.export_call(call)
        receipt = op.execute(child)
        if receipt.status is not StageStatus.SUCCESS or op.verify_effect(child, receipt).status is not EvidenceStatus.VERIFIED:
            raise ValueError('standalone delivery did not verify')
        return receipt, trials

    def workload(self, call):
        if self.slots:
            result, trials = self.produce_slots(call)
            child = self.slot_call(call, 0, 0)
            if result is None:
                return {'status': 'DATA_FAILED', 'slot_trials': trials,
                        'model_usage': self.graph.gateway(child).audit_usage()}
        else:
            child = self.child(call)
            result = self.graph.execute(child)
            if self.graph.verify_effect(child, result).status is not EvidenceStatus.VERIFIED:
                raise ValueError('compiled data effect is not independently recoverable')
            trials = []
        if result.status is not StageStatus.SUCCESS:
            return {'status': 'DATA_FAILED', 'graph_receipt': result.to_dict(),
                    'model_usage': self.graph.gateway(child).audit_usage()}
        job, delivery, output = self._job(call)
        inputs = {**self.runner_inputs, 'imagefolder': str(self.graph.dataset_root / delivery['training_relative'])}
        receipt_path = job.job_id + '/receipt.json'
        feedback = self.runner.run(job, external_job_id=job.job_id,
            result_store=self.result_store, result_path=receipt_path, runner_inputs=inputs)
        context = self.runner.next_data_plan_context(job, feedback.receipt)
        return {'status': 'COMPLETE', 'graph_receipt': result.to_dict(), 'slot_trials': trials, 'delivery': output.to_dict(),
                'job': job.to_dict(), 'training_feedback': feedback.to_dict(),
                'next_frozen_run_context': context, 'model_usage': self.graph.gateway(child).audit_usage(),
                'training_benefit_verified': False}

    def validate_report(self, call, report):
        from ..models import OperatorReceipt
        if self.slots:
            for row in report['slot_trials']:
                op = self.slots[row['slot']][row['attempt']]
                child = self.slot_call(call, row['slot'], row['attempt'])
                if op.verify_effect(child, OperatorReceipt.from_dict(row['receipt'])).status is not EvidenceStatus.VERIFIED:
                    raise ValueError('slot receipt or native evidence changed')
            if report['status'] == 'DATA_FAILED':
                return
        receipt = OperatorReceipt.from_dict(report['graph_receipt'])
        graph = self.export_graph(call) if self.slots else self.graph
        child = self.export_call(call) if self.slots else self.child(call)
        if graph.verify_effect(child, receipt).status is not EvidenceStatus.VERIFIED:
            raise ValueError('workflow data artifacts or native receipts changed')
        if report['status'] == 'DATA_FAILED':
            if receipt.status is not StageStatus.FAILED:
                raise ValueError('workflow data failure verdict changed')
            return
        job, _, output = self._job(call)
        if report['status'] != 'COMPLETE' or report['job'] != job.to_dict() or report['delivery'] != output.to_dict():
            raise ValueError('workflow job/delivery binding changed')
        native = ArtifactRef.from_dict(report['training_feedback']['receipt'])
        if self.runner.collect(job, native).to_dict() != report['training_feedback']:
            raise ValueError('workflow training proof changed')
        if self.runner.next_data_plan_context(job, native) != report['next_frozen_run_context']:
            raise ValueError('workflow feedback/next-run binding changed')

    def result(self, call, report):
        result = super().result(call, report)
        if report['status'] != 'COMPLETE':
            result.update(status='FAILED', failure_code='COMPILED_DATA_FAILED',
                          metrics={'completion_ratio': 0.0}, hard_constraints={'artifact_valid': False})
        return result
