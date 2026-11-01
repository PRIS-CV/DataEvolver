"""Host-bound readers for historical registered contract qualification.

All paths and the reconstruction layout are supplied by the trusted host. A
candidate cannot choose a reader, module root, artifact store or executable.
"""
import json
import copy
import os
import subprocess
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256


class FrozenContractExecutionAuthority:
    def __init__(self, *, config, module_root, plan_path, package_path, stores,
                 request_locations, execution_roots, sandbox_limits, env_path, python):
        self.config = Path(config).resolve()
        self.module_root = Path(module_root).resolve()
        self.plan_path, self.package_path = Path(plan_path).resolve(), Path(package_path).resolve()
        self.stores = {k: str(Path(v).resolve()) for k,v in stores.items()}
        self.locations = {k: str(Path(v).resolve()) for k,v in request_locations.items()}
        self.execution_roots = tuple(str(Path(p).resolve()) for p in execution_roots)
        self.python = Path(python).absolute()
        self.env_path = Path(env_path).resolve()
        from dataevolver.harness.candidate_sandbox import SandboxLimits
        self.limits = SandboxLimits(**sandbox_limits).__dict__.copy()
        if str(self.module_root) not in self.execution_roots or len(set(self.execution_roots)) != len(self.execution_roots):
            raise ValueError("the historical module root must be explicitly frozen")
        for root in map(Path,self.execution_roots):
            if root.name != 'source' or not (root.parent/'source-commit').is_file():
                raise ValueError("historical execution roots must be host source snapshots")
            if any(root.is_relative_to(Path(store)) or Path(store).is_relative_to(root) for store in self.stores.values()):
                raise ValueError("candidate or artifact stores cannot become a module root")
        plan=json.loads(self.plan_path.read_text())
        candidate_root=Path(self.stores[plan['candidate_store']])
        if (self.env_path.name!='env.sh' or not (self.env_path.parent/'source-commit').is_file()
                or self.env_path.is_relative_to(candidate_root)):
            raise ValueError('the historical environment must be a host source snapshot outside candidate files')
        self._identity = self._current_identity()
        self._verified_cache = None

    def descriptor(self):
        return {'config':str(self.config),'module_root':str(self.module_root),
                'plan_path':str(self.plan_path),'package_path':str(self.package_path),'stores':self.stores,
                'request_locations':self.locations,'execution_roots':list(self.execution_roots),
                'sandbox_limits':self.limits,'env_path':str(self.env_path),'python':str(self.python)}

    def _current_identity(self):
        goal = json.loads((self.config/'protected/qualification-goal.json').read_text())
        return {'descriptor':self.descriptor(),'reader_sha256':file_sha256(Path(__file__)),
                'python_sha256':file_sha256(self.python),'campaign_id':goal['campaign_id'],
                'frozen_sha256':file_sha256(self.config/'output/campaigns'/goal['campaign_id']/'frozen_contract.json'),
                'plan_sha256':file_sha256(self.plan_path),'package_manifest_sha256':file_sha256(self.package_path),
                'env_sha256':file_sha256(self.env_path),
                'env_source_commit':(self.env_path.parent/'source-commit').read_text().strip(),
                'source_commits':{root:(Path(root).parent/'source-commit').read_text().strip() for root in self.execution_roots}}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError("historical contract execution binding changed")
        return dict(self._identity)

    def _evidence_inventory(self):
        # A memoized result remains an authority only while every historical
        # dependency is byte-identical. Added/deleted files also invalidate it.
        roots = {self.config, self.plan_path.parent, self.package_path.parent, self.env_path.parent,
                 *map(Path, self.stores.values()), *map(Path, self.locations.values()),
                 *map(Path, self.execution_roots)}
        roots = {root for root in roots if not any(root != other and root.is_relative_to(other) for other in roots)}
        return {str(file): file_sha256(file) for root in sorted(roots) for file in sorted(root.rglob('*'))
                if file.is_file() and '__pycache__' not in file.parts}

    def read(self):
        identity = self.identity()
        inventory = self._evidence_inventory()
        if self._verified_cache is not None:
            frozen_inventory, value = self._verified_cache
            if inventory != frozen_inventory:
                raise ValueError('historical evidence changed after verified read')
            return copy.deepcopy(value)
        env = {**os.environ,'PYTHONPATH':str(self.module_root/'src'),'PYTHONDONTWRITEBYTECODE':'1',
               'GIT_OPTIONAL_LOCKS':'0','CUDA_VISIBLE_DEVICES':''}
        shell='source "$1"; shift; export PYTHONPATH="$1" PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 CUDA_VISIBLE_DEVICES=; shift; exec "$@"'
        done = subprocess.run(['bash','-ec',shell,'frozen-contract-reader',str(self.env_path),str(self.module_root/'src'),
            str(self.python),str(Path(__file__).resolve()),'--read',json.dumps(self.descriptor())],
            capture_output=True,text=True,timeout=180,env=env)
        if done.returncode:
            raise ValueError('historical contract read failed: '+done.stderr[-2200:])
        value = json.loads(done.stdout)
        if (value['origin']['campaign_id'] != identity['campaign_id']
                or value['origin']['frozen_sha256'] != identity['frozen_sha256']):
            raise ValueError('a different contract Controller was returned')
        self.identity()
        if self._evidence_inventory() != inventory:
            raise ValueError('historical evidence changed during verified read')
        self._verified_cache = (inventory, copy.deepcopy(value))
        return value


def _historical(d):
    # Import only the host-selected historical core. Candidate source remains
    # an ArtifactStore data input throughout this read-only reconstruction.
    from dataevolver.dataset.contracts import ArtifactStore
    from dataevolver.harness.operator_development import read_development_plan, CandidatePackage, FrozenSourceRequestAuthority
    from dataevolver.harness.frozen_candidate_request import FrozenCandidateReadinessRequestAuthority
    from dataevolver.harness.candidate_sandbox import TrustedContextRunner, SandboxLimits
    from dataevolver.harness.adapters.operator_contract_qualification import CandidateContractQualificationOperator
    from dataevolver.harness.candidate_revision import ControllerContractFeedbackAuthority
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.models import GoalSpec, OperatorReceipt, StageResult, EvidenceStatus, StageStatus
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.planner import DeterministicPlanner
    from dataevolver.harness.operators import OperatorRegistry
    cfg=Path(d['config']); loc={k:Path(v) for k,v in d['request_locations'].items()}
    plan=read_development_plan(json.loads(Path(d['plan_path']).read_text()))
    package=CandidatePackage.from_dict(json.loads(Path(d['package_path']).read_text()))
    store=ArtifactStore(d['stores'])
    protocol=json.loads((cfg/'protected/contract-qualification-protocol.json').read_text())
    # Limits are immutable protocol material, never inferred from candidate output.
    limits=d['sandbox_limits']
    operator_class=CandidateContractQualificationOperator;extra={}
    if protocol['schema_version']=='dataevolver.independent_contract_qualification.v1':
        from dataevolver.harness.adapters.operator_independent_contract import IndependentContractQualificationOperator
        from dataevolver.harness.contract_execution_authority import (
            FrozenContractExecutionAuthority as HistoricalEngineeringReader,
            read_independent_controller as historical_independent_summary)
        descriptor=json.loads((cfg/'protected/engineering-reader.json').read_text())
        parent_protocol=json.loads((Path(descriptor['config'])/'protected/contract-qualification-protocol.json').read_text())
        if parent_protocol.get('qualification_scope')!='synthetic_authority_contract_engineering':
            raise ValueError('independent qualification must reference an engineering stage directly')
        operator_class=IndependentContractQualificationOperator
        extra={'engineering_authority':HistoricalEngineeringReader(**descriptor)}
    op=operator_class(plan=plan,package=package,store=store,
        request_authority=FrozenSourceRequestAuthority(config_root=loc['request_origin'],python=d['python']),
        runner=TrustedContextRunner(python=d['python'],limits=SandboxLimits(**limits)),check_protocol=protocol,
        qualification_request_authority=FrozenCandidateReadinessRequestAuthority(**loc,python=d['python']),
        output_root=cfg/'output',**extra)
    registry=OperatorRegistry();registry.register(op)
    goal=GoalSpec.from_dict(json.loads((cfg/'protected/qualification-goal.json').read_text()))
    c=EvolutionController.open(repo_root=cfg/'source',campaigns_root=cfg/'output/campaigns',goal=goal,
        registry=registry,planner=DeterministicPlanner(()),evaluator_contract=EvaluatorContract.from_path(cfg/'evaluator.json'),
        schema_paths=tuple((cfg/'protected').rglob('*.json')),
        execution_paths=tuple(p for root in d['execution_roots'] for p in (Path(root)/'src/dataevolver').rglob('*.py')))
    call=next(e.payload['call_id'] for e in c.store.events.read() if e.event_type=='ACTION_COMMITTED')
    value=(historical_independent_summary(c,op,call) if extra
           else ControllerContractFeedbackAuthority(c,op,call).read())
    committed=next(e for e in c.store.events.read() if e.event_type=='ACTION_COMMITTED' and e.payload['call_id']==call)
    receipt=OperatorReceipt.from_dict(committed.payload['receipt'])
    evaluation=c.evaluator.evaluate(StageResult.from_dict(receipt.stage_result),effect_status=EvidenceStatus.VERIFIED)
    value.update(input_sha256s=[case['input']['sha256'] for case in protocol['cases']],
                 fixture_family_ids=sorted({case['checks']['fixture_family_id'] for case in value['cases']}),
                 runtime_sha256=payload_sha256(op.runtime),controller_state=c.status().state.value)
    accepted=(c.status().state.value=='COMPLETE' and receipt.status is StageStatus.SUCCESS
              and receipt.return_code==0 and c._goal_satisfied(evaluation))
    value['independent_evaluator_accepted' if extra else 'engineering_evaluator_accepted']=accepted
    return value


def read_independent_controller(c,op,call_id):
    """Promotion-oriented summaries; sealed independent payloads stay hidden."""
    from dataevolver.harness.models import CampaignState,OperatorCall,OperatorReceipt,EvidenceStatus
    from dataevolver.harness.adapters.operator_independent_contract import IndependentContractQualificationOperator
    if not isinstance(op,IndependentContractQualificationOperator):
        raise ValueError('an independent registered qualifier is required')
    state=CampaignState.from_dict(json.loads(c.layout.state_path.read_text()))
    if (not c._frozen()[0] or c.replay().to_dict()!=state.to_dict()
            or state.state.value not in {'ABSTAIN','COMPLETE'} or state.data.get('pending_call')):
        raise ValueError('independent qualification must be frozen, replayable and terminal')
    planned=[e for e in c.store.events.read() if e.event_type=='ACTION_PLANNED' and e.payload.get('call_id')==call_id]
    committed=[e for e in c.store.events.read() if e.event_type=='ACTION_COMMITTED' and e.payload.get('call_id')==call_id]
    if len(planned)!=1 or len(committed)!=1 or planned[0].sequence>=committed[0].sequence:
        raise ValueError('an exact committed independent invocation is required')
    call=OperatorCall.from_dict(planned[0].payload['operator_call']);receipt=OperatorReceipt.from_dict(committed[0].payload['receipt'])
    if (call.operator_id!=op.spec.operator_id or c.registry.get(call.operator_id).spec.sha256()!=op.spec.sha256()
            or c.goal.objective.get('candidate_package_sha256')!=op.package.compute_hash()
            or c.goal.objective.get('contract_qualification_protocol_sha256')!=payload_sha256(op.protocol)):
        raise ValueError('independent package/protocol was not prebound')
    verified=op.verify_effect(call,receipt)
    if (verified.status is not EvidenceStatus.VERIFIED
            or verified.artifact_hashes!=committed[0].payload['effect_verification']['artifact_hashes']):
        raise ValueError('committed independent effects no longer verify')
    report=json.loads(op._path(call).read_text())
    return {'schema_version':'dataevolver.independent_contract_summary.v1',
        'scope':report['qualification_scope'],'origin':{'campaign_id':c.goal.campaign_id,
            'frozen_sha256':file_sha256(c.layout.frozen_contract_path),'call_id':call_id,
            'operator_spec_sha256':op.spec.sha256(),'package_sha256':op.package.compute_hash()},
        'package':op.package.to_dict(),'cases':[{'case_id':x['case_id'],'checks':x} for x in report['contract_checks']],
        'report_sha256':file_sha256(op._path(call)),'pure_data_engineering_qualification':'PASSED',
        'pure_data_independent_qualification':report['pure_data_independent_qualification'],
        'engineering_origin':report['engineering_origin'],'lifecycle_total':report['lifecycle_total'],
        'resource_deadline_epoch':op.budget.resource_deadline_epoch,'runtime':op.runtime,
        'production_qualification':'NOT_ASSESSED','production_adoption_allowed':False}


if __name__=='__main__':
    import sys
    if len(sys.argv)!=3 or sys.argv[1]!='--read':raise SystemExit(2)
    print(json.dumps(_historical(json.loads(sys.argv[2]))))
