"""Read-only reconstruction of completed operator-version lifecycle campaigns.

The host selects an original frozen source and environment. Catalog prefixes
represent actual historical states only during constructor binding. Every
effect is then verified against the complete current catalog. No Controller
tick/run, candidate dispatch or model invocation belongs to this reader.
"""
import copy
import json
import os
import subprocess
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, identifier
from dataevolver.harness.operator_lifecycle import EMPTY_HEAD, OperatorVersionStore
from dataevolver.harness import operator_lifecycle as lifecycle_contracts


class HistoricalCatalogPrefix:
    def __init__(self, catalog, head):
        self.root = catalog.root
        rows = catalog.events()
        if head == EMPTY_HEAD:
            self.rows = []
        else:
            matches = [i for i, row in enumerate(rows) if row['sha256'] == head]
            if len(matches) != 1:
                raise ValueError('historical catalog head is not an actual prefix')
            self.rows = rows[:matches[0]+1]
        self.catalog = catalog
        self.expected = head

    def head(self):
        self.events()
        return self.expected

    def events(self):
        if self.catalog.events()[:len(self.rows)] != self.rows:
            raise ValueError('historical catalog prefix changed')
        return copy.deepcopy(self.rows)

    def lease(self):
        raise ValueError('historical catalog views cannot mutate or dispatch')

    def append(self, **kwargs):
        raise ValueError('historical catalog views cannot mutate or dispatch')


class FrozenOperatorLifecycleAuthority:
    def __init__(self, *, config, module_root, env_path, python):
        self.config = Path(config).resolve()
        self.module_root = Path(module_root).resolve()
        self.env_path = Path(env_path).resolve()
        self.python = Path(python).absolute()
        if (self.module_root != self.config/'source'
                or not (self.config/'source-commit').is_file()
                or self.env_path.name != 'env.sh' or not (self.env_path.parent/'source-commit').is_file()
                or self.env_path.is_relative_to(self.config/'candidate')):
            raise ValueError('lifecycle readers require explicit host source and environment snapshots')
        protocol = self.config/'adoption/protocol.json'
        original = json.loads(protocol.read_text())['snapshot']['engineering']['authority']
        if (self.env_path != Path(original['descriptor']['env_path']).resolve()
                or self.python != Path(original['descriptor']['python']).absolute()
                or file_sha256(self.env_path) != original['env_sha256']
                or file_sha256(self.python) != original['python_sha256']):
            raise ValueError('lifecycle reader environment and interpreter must match the original host binding')
        self._identity = self._current_identity()
        self._cache = None

    def descriptor(self):
        return {'config': str(self.config), 'module_root': str(self.module_root),
                'env_path': str(self.env_path), 'python': str(self.python)}

    def _campaigns(self):
        result = {}
        for stage in ('adoption', 'invocation', 'revocation'):
            folder = self.config/stage
            goal = json.loads((folder/'goal.json').read_text())
            identifier(goal['campaign_id'])
            campaign = folder/'output/campaigns'/goal['campaign_id']
            result[stage] = campaign
        return result

    def _current_identity(self):
        return {'descriptor': self.descriptor(), 'reader_sha256': file_sha256(Path(__file__)),
            'catalog_contract_sha256': file_sha256(Path(lifecycle_contracts.__file__)),
            'python_sha256': file_sha256(self.python), 'env_sha256': file_sha256(self.env_path),
            'env_source_commit': (self.env_path.parent/'source-commit').read_text().strip(),
            'source_commit': (self.config/'source-commit').read_text().strip(),
            'frozen_contracts': {stage: file_sha256(root/'frozen_contract.json') for stage, root in self._campaigns().items()},
            'catalog_head': OperatorVersionStore(self.config/'operator-versions').head()}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError('lifecycle authority binding changed')
        return copy.deepcopy(self._identity)

    def _inventory(self):
        protocol = json.loads((self.config/'adoption/protocol.json').read_text())
        roots = {self.config, self.env_path.parent}
        for key in ('engineering', 'independent'):
            d = protocol['snapshot'][key]['authority']['descriptor']
            roots.update(Path(d[name]) for name in ('config', 'module_root'))
            roots.update(Path(d[name]).parent for name in ('plan_path', 'package_path', 'env_path'))
            roots.update(map(Path, d['stores'].values()))
            roots.update(map(Path, d['request_locations'].values()))
            roots.update(map(Path, d['execution_roots']))
        roots = {root for root in roots if not any(root != other and root.is_relative_to(other) for other in roots)}
        return {str(file): file_sha256(file) for root in sorted(roots) for file in sorted(root.rglob('*'))
                if file.is_file() and '__pycache__' not in file.parts}

    def read(self):
        identity = self.identity()
        inventory = self._inventory()
        if self._cache is not None:
            original, result = self._cache
            if inventory != original:
                raise ValueError('lifecycle evidence changed after verified read')
            return copy.deepcopy(result)
        shell = ('source "$1"; shift; export PYTHONPATH="$1" PYTHONDONTWRITEBYTECODE=1 '
                 'GIT_OPTIONAL_LOCKS=0 CUDA_VISIBLE_DEVICES=; shift; exec "$@"')
        done = subprocess.run(['bash', '-ec', shell, 'lifecycle-reader', str(self.env_path),
            str(self.module_root/'src'), str(self.python), str(Path(__file__).resolve()), '--read',
            json.dumps(self.descriptor())], capture_output=True, text=True, timeout=600,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
        if done.returncode:
            raise ValueError('historical lifecycle read failed: '+done.stderr[-3000:])
        result = json.loads(done.stdout)
        self.identity()
        if (result['frozen_contracts'] != identity['frozen_contracts']
                or result['catalog_head'] != identity['catalog_head'] or self._inventory() != inventory):
            raise ValueError('different or changing lifecycle evidence was returned')
        self._cache = (inventory, copy.deepcopy(result))
        return result


def _historical(d):
    # All these imports resolve to the explicitly selected historical source.
    from dataevolver.dataset.contracts import ArtifactStore, ArtifactRef
    from dataevolver.harness.operator_development import (CandidatePackage, read_development_plan,
        FrozenSourceRequestAuthority, DevelopmentBudget, QualificationProtocol)
    from dataevolver.harness.contract_execution_authority import FrozenContractExecutionAuthority
    from dataevolver.harness.frozen_candidate_request import FrozenCandidateReadinessRequestAuthority
    from dataevolver.harness.candidate_sandbox import TrustedContextRunner, SandboxLimits
    from dataevolver.harness.operator_lifecycle import CommittedOperatorAdoptionAuthority, OperatorVersionStore as HistoricalStore
    from dataevolver.harness.adapters.operator_version_lifecycle import OperatorVersionMutation
    from dataevolver.harness.adapters.adopted_operator_invocation import AdoptedOperatorInvocation
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.models import GoalSpec, OperatorCall, OperatorReceipt, StageResult, EvidenceStatus
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.planner import DeterministicPlanner
    from dataevolver.harness.evaluator import EvaluatorContract

    def forbidden(*args, **kwargs):
        raise RuntimeError('read-only lifecycle reconstruction cannot execute work')
    EvolutionController.run = EvolutionController.tick = forbidden
    TrustedContextRunner.run = TrustedContextRunner.run_with_context = forbidden
    OperatorVersionMutation.execute = OperatorVersionMutation.recover = forbidden
    AdoptedOperatorInvocation.execute = AdoptedOperatorInvocation.recover = forbidden
    cfg = Path(d['config'])
    protocols = {name: json.loads((cfg/name/'protocol.json').read_text()) for name in ('adoption', 'invocation', 'revocation')}
    snapshot = protocols['adoption']['snapshot']
    engineering = FrozenContractExecutionAuthority(**snapshot['engineering']['authority']['descriptor'])
    independent = FrozenContractExecutionAuthority(**snapshot['independent']['authority']['descriptor'])
    original = snapshot['engineering']['authority']['descriptor']
    plan = read_development_plan(json.loads(Path(original['plan_path']).read_text()))
    package = CandidatePackage.from_dict(json.loads(Path(original['package_path']).read_text()))
    if package.compute_hash() != snapshot['package_sha256']:
        raise ValueError('historical lifecycle package changed')
    stores = dict(original['stores']); stores['protocol'] = str(cfg/'protected')
    store = ArtifactStore(stores)
    locations = {k: Path(v) for k, v in original['request_locations'].items()}
    runner = TrustedContextRunner(python=d['python'], limits=SandboxLimits(**snapshot['runtime']['limits']))
    catalog = HistoricalStore(cfg/'operator-versions')

    def open_stage(name, operator):
        folder = cfg/name
        goal = GoalSpec.from_dict(json.loads((folder/'goal.json').read_text()))
        frozen_path = folder/'output/campaigns'/goal.campaign_id/'frozen_contract.json'
        frozen = json.loads(frozen_path.read_text())
        registry = OperatorRegistry(); registry.register(operator)
        c = EvolutionController.open(repo_root=cfg/'source', campaigns_root=folder/'output/campaigns', goal=goal,
            registry=registry, planner=DeterministicPlanner(()), evaluator_contract=EvaluatorContract.from_path(folder/'evaluator.json'),
            schema_paths=tuple(map(Path, frozen['schema_hashes'])),
            execution_paths=tuple((cfg/'source/src/dataevolver').rglob('*.py')))
        state = c.status()
        if state.state.value != 'COMPLETE' or not c._frozen()[0] or c.replay().to_dict() != state.to_dict():
            raise ValueError('lifecycle stage is not complete, frozen and replayable: '+name)
        planned = [e for e in c.store.events.read() if e.event_type == 'ACTION_PLANNED']
        committed = [e for e in c.store.events.read() if e.event_type == 'ACTION_COMMITTED']
        if len(planned) != 1 or len(committed) != 1 or planned[0].sequence >= committed[0].sequence:
            raise ValueError('one exact lifecycle commit is required')
        call = OperatorCall.from_dict(planned[0].payload['operator_call'])
        receipt = OperatorReceipt.from_dict(committed[0].payload['receipt'])
        check = operator.verify_effect(call, receipt)
        if (receipt.call_id != call.call_id or receipt.status.value != 'SUCCESS' or receipt.return_code != 0
                or check.status is not EvidenceStatus.VERIFIED
                or check.artifact_hashes != committed[0].payload['effect_verification']['artifact_hashes']
                or not c._goal_satisfied(c.evaluator.evaluate(StageResult.from_dict(receipt.stage_result), effect_status=check.status))):
            raise ValueError('lifecycle commit or frozen Evaluator no longer verifies: '+name)
        report = json.loads(Path(receipt.output_artifacts[0]['path']).read_text())
        return c, call, report, {'campaign_id': goal.campaign_id, 'call_id': call.call_id,
            'frozen_sha256': file_sha256(frozen_path), 'receipt_sha256': receipt.sha256(),
            'report_sha256': file_sha256(Path(receipt.output_artifacts[0]['path']))}

    adoption = OperatorVersionMutation(catalog=HistoricalCatalogPrefix(catalog, protocols['adoption']['catalog_head']),
        expected_head=protocols['adoption']['catalog_head'], budget=DevelopmentBudget.from_dict(protocols['adoption']['budget']),
        output_root=cfg/'adoption/output', package=package, store=store, engineering=engineering, independent=independent,
        runtime=runner.identity())
    adoption.catalog = catalog  # Effects always use full actual current history.
    ac, call, adopted_report, aorigin = open_stage('adoption', adoption)
    authority = CommittedOperatorAdoptionAuthority(ac, adoption, call.call_id)
    invoke = AdoptedOperatorInvocation(adoption_authority=authority, plan=plan, package=package, store=store,
        request_authority=FrozenSourceRequestAuthority(config_root=locations['request_origin'], python=d['python']),
        qualification_request_authority=FrozenCandidateReadinessRequestAuthority(**locations, python=d['python']),
        runner=runner, check_protocol=protocols['invocation'], output_root=cfg/'invocation/output')
    ic, _, invocation_report, iorigin = open_stage('invocation', invoke)
    revocation = OperatorVersionMutation(catalog=HistoricalCatalogPrefix(catalog, protocols['revocation']['expected_head']),
        expected_head=protocols['revocation']['expected_head'], budget=DevelopmentBudget.from_dict(protocols['revocation']['budget']),
        output_root=cfg/'revocation/output', adoption=authority, reason=protocols['revocation']['reason'])
    revocation.catalog = catalog
    rc, _, revoked_report, rorigin = open_stage('revocation', revocation)
    if (not invocation_report['adopted_operator_reused'] or not invocation_report['actual_subsequent_execution']
            or revoked_report['event']['action'] != 'REVOKED'
            or revoked_report['event']['snapshot_sha256'] != adopted_report['event']['snapshot_sha256']):
        raise ValueError('reuse or revocation does not bind this adopted version')
    design = QualificationProtocol.from_dict(json.loads(store.resolve(
        ArtifactRef.from_dict(plan.qualification_protocol)).read_text()))
    return {'schema_version': 'dataevolver.committed_operator_lifecycle_summary.v1',
        'operator_id': snapshot['operator_id'], 'version': snapshot['version'], 'scope': snapshot['scope'],
        'package_sha256': package.compute_hash(), 'adoption_snapshot_sha256': adopted_report['event']['snapshot_sha256'],
        'catalog_head': catalog.head(), 'origins': {'adoption': aorigin, 'invocation': iorigin, 'revocation': rorigin},
        'frozen_contracts': {key: origin['frozen_sha256'] for key, origin in
            (('adoption', aorigin), ('invocation', iorigin), ('revocation', rorigin))},
        'current_status': 'REVOKED', 'actual_subsequent_execution_verified': True,
        'lifecycle_total': revoked_report['lifecycle_total'], 'resource_deadline_epoch': snapshot['resource_deadline_epoch'],
        'production_qualification': 'NOT_ASSESSED', 'production_adoption_allowed': False,
        'production_gap_codes': ['VERSION_REVOKED', 'SYNTHETIC_SCOPE_ONLY',
            'PRODUCTION_ENGINEERING_NOT_VERIFIED', 'PRODUCTION_INDEPENDENT_NOT_VERIFIED'],
        'production_requirements': {'design_protocol_sha256': design.compute_hash(),
            'matched_context_sha256': design.matched_context_sha256, 'stages': design.stages,
            'families_by_stage': {stage: design.families(stage) for stage in ('engineering', 'independent')},
            'heldout_families': [s['family_id'] for s in design.sources if s['split'] == 'heldout'],
            'source_payloads_disclosed': False, 'qualification_execution_status': 'NOT_EXECUTED'},
        'remaining_model_calls_in_development_plan': max(0, plan.budget['limits']['model_calls']-revoked_report['lifecycle_total']['model_calls']),
        'remaining_planner_calls_in_development_plan': max(0, plan.budget['limits']['planner_calls']-revoked_report['lifecycle_total']['planner_calls']),
        'remaining_image_experiments_in_development_plan': max(0, plan.budget['limits']['image_experiments']-revoked_report['lifecycle_total']['image_experiments']),
        'new_candidate_executions': 0, 'new_model_calls': 0}


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != '--read':
        raise SystemExit(2)
    print(json.dumps(_historical(json.loads(sys.argv[2]))))
