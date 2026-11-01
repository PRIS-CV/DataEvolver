"""Read generic candidate-readiness requests with their original frozen core.

The host supplies historical locations. No candidate path is imported, no
Controller tick/run and no ModelGateway invoke is available in this reader.
"""
import json
import os
import subprocess
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256


class FrozenCandidateReadinessRequestAuthority:
    def __init__(self, *, readiness_config, checks_config, build_config, recovery_config,
                 request_origin, python):
        self.locations={k:str(Path(v).resolve()) for k,v in dict(readiness_config=readiness_config,
            checks_config=checks_config,build_config=build_config,recovery_config=recovery_config,
            request_origin=request_origin).items()}
        self.python=Path(python).absolute()
        self._identity=self._current_identity()

    def _current_identity(self):
        cfg=Path(self.locations['readiness_config']);checks=Path(self.locations['checks_config'])
        goal=json.loads((cfg/'protected/goal.json').read_text())
        check_goal=json.loads((checks/'protected/behavior-goal.json').read_text())
        return {'campaign_id':goal['campaign_id'],
            'frozen_sha256':file_sha256(cfg/'output/campaigns'/goal['campaign_id']/'frozen_contract.json'),
            'checks_frozen_sha256':file_sha256(checks/'output/campaigns'/check_goal['campaign_id']/'frozen_contract.json'),
            'locations':dict(self.locations),'python':str(self.python),'python_sha256':file_sha256(self.python),
            'reader_sha256':file_sha256(Path(__file__)),
            'env_sha256':file_sha256(cfg/'env.sh'),
            'historical_core_commit':(checks/'source-commit').read_text().strip(),
            'readiness_source_commit':(cfg/'source-commit').read_text().strip()}

    def identity(self):
        if self._current_identity()!=self._identity:raise ValueError('historical candidate request identity changed')
        return dict(self._identity)

    def verify(self, plan, store):
        identity=self.identity()
        if plan.request_campaign_id!=identity['campaign_id'] or plan.request_frozen_sha256!=identity['frozen_sha256']:
            raise ValueError('qualification request binding changed')
        # Preserve readiness-path lookup and conversion before source checks.
        Path(self.locations['readiness_config'])
        checks = Path(self.locations['checks_config'])
        env={**os.environ,'PYTHONPATH':str(checks/'source/src'),'PYTHONDONTWRITEBYTECODE':'1',
             'GIT_OPTIONAL_LOCKS':'0','CUDA_VISIBLE_DEVICES':''}
        completed=subprocess.run([str(self.python),str(Path(__file__).resolve()),'--read-historical',
            json.dumps(self.locations),str(self.python)],input=json.dumps({'plan':plan.to_dict(),
                'stores':{k:str(v) for k,v in store.roots.items()}}),capture_output=True,text=True,timeout=120,env=env)
        if completed.returncode:
            raise ValueError('historical readiness verification failed: '+completed.stderr[-1800:])
        value=json.loads(completed.stdout)
        expected={'campaign_id':plan.request_campaign_id,'request_sha256':plan.capability_request['sha256'],
                  'frozen_sha256':plan.request_frozen_sha256}
        if value!=expected:raise ValueError('historical readiness reader returned a different request')
        self.identity()
        return value


def _historical(locations,python,value):
    import importlib.util
    import sys
    # This process starts with the approved historical checks source on
    # PYTHONPATH. Keep those module globals when loading its readiness domain.
    from dataevolver.dataset.contracts import ArtifactStore
    from dataevolver.harness.operator_development import (read_development_plan,CandidatePackage,
        FrozenSourceRequestAuthority,ControllerRequestAuthority)
    from dataevolver.harness.candidate_sandbox import LinuxPureDataRunner,SandboxLimits
    from dataevolver.harness.adapters.operator_candidate_checks import CandidateChecksOperator
    from dataevolver.harness.adapters.operator_development import CandidateInspectionOperator
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.models import GoalSpec
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.planner import DeterministicPlanner
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
    from dataevolver.adapters.model_contracts import ModelProfile,ModelPolicy,ModelRequest
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.adapters.codex_planner import RecoveringPlannerBackend
    cfg,checks,build,recovery,origin=(Path(locations[k]) for k in
        ('readiness_config','checks_config','build_config','recovery_config','request_origin'))
    parent_plan=read_development_plan(json.loads((checks/'protected/original-recovery-plan.json').read_text()))
    package=CandidatePackage.from_dict(json.loads(next((recovery/'output').glob('campaigns/*/rounds/*/execute/candidate-package.json')).read_text()))
    store=ArtifactStore({'request':origin,'protocol':checks/'protected','candidate':recovery/'candidate'})
    request_authority=FrozenSourceRequestAuthority(config_root=origin,python=python)
    protocol=json.loads((checks/'protected/behavior-check-protocol.json').read_text())
    operator=CandidateChecksOperator(plan=parent_plan,package=package,store=store,request_authority=request_authority,
        runner=LinuxPureDataRunner(python=python,limits=SandboxLimits(output_bytes=1024)),
        check_protocol=protocol,output_root=checks/'output')
    registry=OperatorRegistry();registry.register(operator)
    check_goal=GoalSpec.from_dict(json.loads((checks/'protected/behavior-goal.json').read_text()))
    c=EvolutionController.open(repo_root=checks/'source',campaigns_root=checks/'output/campaigns',goal=check_goal,
        registry=registry,planner=DeterministicPlanner(()),evaluator_contract=EvaluatorContract.from_path(checks/'evaluator.json'),
        schema_paths=tuple((checks/'protected').glob('*.json')),execution_paths=tuple((checks/'source/src/dataevolver').rglob('*.py')))
    call_id=next(e.payload['call_id'] for e in c.store.events.read() if e.event_type=='ACTION_COMMITTED')
    key='dataevolver.harness.operator_readiness'
    spec=importlib.util.spec_from_file_location(key,cfg/'source/src/dataevolver/harness/operator_readiness.py')
    module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
    authority=module.ControllerCandidateChecksAuthority(c,operator,call_id)
    domain=module.CandidateReadinessDomain(plan=parent_plan,package=package,store=store,
        request_authority=request_authority,checks_authority=authority)
    profile=ModelProfile.from_dict(json.loads((build/'protected/original-model-profile.json').read_text()))
    gateway=ModelGateway(profiles=(profile,),policy=ModelPolicy({'agent.runtime_rsi':profile.profile_id},1),
        receipt_root=cfg/'output/model_receipts',artifact_store=store)
    backend=RecoveringPlannerBackend(gateway,model_id=profile.model,timeout_seconds=900)
    goal=GoalSpec.from_dict(json.loads((cfg/'protected/goal.json').read_text()))
    planner=RuntimeRSIPlanner(backend=backend,domain=domain,trace_root=cfg/'output/campaigns'/goal.campaign_id/'planner')
    inspection=CandidateInspectionOperator(plan=parent_plan,package=package,store=store,
        request_authority=request_authority,output_root=cfg/'output')
    registry2=OperatorRegistry();registry2.register(inspection)
    controller=EvolutionController.open(repo_root=cfg/'source',campaigns_root=cfg/'output/campaigns',goal=goal,
        registry=registry2,planner=planner,evaluator_contract=EvaluatorContract.from_path(cfg/'evaluator.json'),
        schema_paths=tuple((cfg/'protected').glob('*.json')),
        execution_paths=tuple((cfg/'source/src/dataevolver').rglob('*.py'))+tuple((checks/'source/src/dataevolver').rglob('*.py')))
    trace=json.loads((cfg/'output/campaigns'/goal.campaign_id/'planner/p000/trace.json').read_text())
    if trace['outcome']!='VALID_DECISION':raise ValueError('original readiness did not produce a valid runtime decision')
    usage=gateway.audit_usage()
    receipt=gateway.recover(ModelRequest.from_dict(trace['model_call']['request']))
    if receipt is None or receipt.without_reuse_hash()!=trace['model_call']['receipt_hash'] or usage['model_calls']!=1 or usage['unknown_duration_attempts']:
        raise ValueError('original readiness model completion or costs do not verify')
    return ControllerRequestAuthority(controller).verify(read_development_plan(value['plan']),ArtifactStore(value['stores']))


if __name__=='__main__':
    import sys
    if len(sys.argv)!=4 or sys.argv[1]!='--read-historical':raise SystemExit(2)
    print(json.dumps(_historical(json.loads(sys.argv[2]),sys.argv[3],json.load(sys.stdin))))
