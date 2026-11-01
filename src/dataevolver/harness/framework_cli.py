"""Thin public compile/SDK commands; existing evolve/step own run state."""
import json
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .operator_sdk import builtin_step_registry, check_declaration, scaffold_operator
from .plan_compiler import CompileContext, CompilationGap, builtin_domains, compile_plan
from .state_store import atomic_write_json


def add_commands(commands):
    scene = commands.add_parser('worker-compile', aliases=['scene-compile'], help='freeze a configured native worker as one Harness goal; no execution')
    for name in ('profile', 'output', 'campaign', 'campaign-root'):
        scene.add_argument('--' + name, required=True)
    visual = commands.add_parser('visual-models', help='inspect exact cached visual model roles without inference')
    visual.add_argument('--kind', required=True, help='explicit optional visual adapter identifier')
    visual.add_argument('--root', required=True)
    visual.add_argument('--hash-weights', action='store_true')
    visual.add_argument('--output', required=True)
    cmd = commands.add_parser('compile-plan', help='bind a plan to registered operators')
    for name in ('plan', 'profile', 'output', 'campaign', 'evaluator'):
        cmd.add_argument('--'+name, required=True)
    actions = commands.add_parser('operator-sdk', help='shared operator SDK').add_subparsers(dest='operator_command', required=True)
    actions.add_parser('catalog')
    init = actions.add_parser('init')
    init.add_argument('--id', required=True); init.add_argument('--output', required=True)
    init.add_argument('--effect', choices=['pure_data', 'model', 'external', 'composite'], default='pure_data')
    for name in ('check', 'package'):
        cmd = actions.add_parser(name)
        cmd.add_argument('--spec', required=True); cmd.add_argument('--contract', required=True)
        if name == 'package': cmd.add_argument('--output', required=True)


def handle(args):
    if args.command in {'worker-compile', 'scene-compile'}:
        from dataevolver.workflows.scene.prepare import prepare
        return prepare(args.profile, args.output, args.campaign, args.campaign_root)
    if args.command == 'visual-models':
        from dataevolver.adapters.visual_models import inspect_assets
        result = inspect_assets(args.kind, args.root, hash_weights=args.hash_weights)
        atomic_write_json(Path(args.output), result)
        return {'model': result['model'], 'weight_files': len(result['weight_files']),
                'weights_hashed': result['weights_hashed'], 'inference_performed': False,
                'execution_ready': False, 'output': str(Path(args.output).resolve())}
    if args.command == 'operator-sdk':
        if args.operator_command == 'catalog': return builtin_step_registry().catalog()
        if args.operator_command == 'init':
            return {'path': str(scaffold_operator(args.output, operator_id=args.id, effect=args.effect)), 'execution_performed': False}
        spec, declaration = json.loads(Path(args.spec).read_text()), json.loads(Path(args.contract).read_text())
        result = check_declaration(spec, declaration)
        if args.operator_command == 'package':
            source = Path(spec['source_path']).resolve()
            if not source.is_file() or file_sha256(source) != spec['code_revision']:
                raise ValueError('package source differs from its declared revision')
            root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
            (root/'operator.py').write_bytes(source.read_bytes())
            atomic_write_json(root/'operator-spec.json', spec)
            atomic_write_json(root/'operator-contract.json', declaration)
            atomic_write_json(root/'package.json', {'files': {p.name: file_sha256(p) for p in root.iterdir()}, 'qualification_performed': False})
            result['path'] = str(root)
        return result
    plan, profile = json.loads(Path(args.plan).read_text()), json.loads(Path(args.profile).read_text())
    registry = builtin_step_registry()
    context = CompileContext(registry, builtin_domains(), tuple(profile['allowed_operators']),
        profile.get('contract', {}), profile.get('source_bindings', {}), profile.get('roots', {}), tuple(profile.get('policies', [])))
    compiled = compile_plan(plan, context)
    if isinstance(compiled, CompilationGap): return compiled.to_dict()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    evaluator = Path(args.evaluator).resolve()
    key = {'input_artifact_hash': payload_sha256(compiled.contract['intent']['sources']),
        'parent_artifact_hash': compiled.compute_hash(), 'seed': 0, 'operator_id': 'data.graph', 'operator_version': '1',
        'model_config_hash': payload_sha256(compiled.contract['models']), 'preprocessing_hash': payload_sha256(compiled.contract['graph']),
        'evaluator_hash': file_sha256(evaluator), 'environment_contract_hash': payload_sha256(compiled.contract['stores'])}
    atomic_write_json(root/'compiled_run.json', compiled.to_dict())
    goal = {'schema_version': 'dataevolver.harness.goal.v1', 'campaign_id': args.campaign,
        'objective': {'compiled_run': compiled.to_dict(), 'required_hard_constraints': ['artifact_valid'], 'target_metrics': {'completion_ratio': 1.0},
                      'initial_trial_key': key, 'canary_trial_key': key},
        'initial_operator_id': 'data.graph', 'route_candidates': ['compiled_graph'], 'initial_operator_parameters': compiled.contract,
        'planner': {'kind': 'deterministic'},
        'budgets': {'max_rounds': 1, 'max_operator_calls': 1, 'max_experiments': 0, 'max_retries': 0,
                    'max_wall_seconds': compiled.contract['budget']['max_wall_seconds'], 'max_planner_calls': 0},
        'resource_policy': {'allowed_operators': ['data.graph'], 'allow_gpu': profile.get('allow_gpu', False), 'allow_network': profile.get('allow_network', False)},
        'frozen_evaluator': {'path': str(evaluator), 'sha256': file_sha256(evaluator)}}
    atomic_write_json(root/'goal.json', goal)
    return {'compiled': True, 'compiled_run': str(root/'compiled_run.json'), 'goal': str(root/'goal.json'), 'execution_performed': False}
