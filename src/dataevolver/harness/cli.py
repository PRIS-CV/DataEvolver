from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dataevolver.paths import CONFIGS_ROOT, DATAEVOLVER_ROOT, REPO_ROOT, SRC_ROOT

from .adapters import MultimodalT2IDryRunOperator, ThreeDAssetEvolutionOperator
from .adapters.multimodal_canary import MultimodalT2ICanaryOperator
from .controller import EvolutionController
from .evaluator import EvaluatorContract, sha256_file
from .models import ExpectedOutputSpec, GoalSpec, OperatorSpec, StageStatus, canonical_json
from .operators import FakeOperator, OperatorRegistry
from .planner import AgenticRSIPlanner, DeterministicPlanner, OpenAICompatiblePlannerBackend
from .state_store import CampaignLayout, StateStore, atomic_write_json
from .strategy import ExperimentCompiler


def _campaign_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-root", default=str(DATAEVOLVER_ROOT / "campaigns"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DataEvolver Autonomous Harness V1")
    commands = parser.add_subparsers(dest="command", required=True)
    from .framework_cli import add_commands
    add_commands(commands)

    models = commands.add_parser("models", help="inspect explicit local model profiles without inference")
    models.add_argument("--config", required=True)
    models.add_argument("--check-endpoint", action="store_true", help="perform read-only /models health checks")

    for name in ("create", "evolve"):
        evolve = commands.add_parser(name)
        evolve.add_argument("--goal", required=True)
        evolve.add_argument("--campaign-root", default=str(DATAEVOLVER_ROOT / "campaigns"))
        evolve.add_argument("--idempotency-key", required=True)
        evolve.add_argument("--dry-run", action="store_true")

    step = commands.add_parser("step")
    _campaign_args(step)
    step.add_argument("--goal")
    step.add_argument("--idempotency-key", default="")
    step.add_argument("--max-actions", type=int, default=1)
    step.add_argument("--format", choices=["json"], default="json")
    step.add_argument("--dry-run", action="store_true")

    for name in ("pause", "resume", "status", "replay", "operators"):
        _campaign_args(commands.add_parser(name))
    explain = commands.add_parser("explain")
    _campaign_args(explain)
    explain.add_argument("--limit", type=int, default=20)
    return parser.parse_args(argv)


def _load_goal(path: Path) -> GoalSpec:
    return GoalSpec.from_json(path.read_text(encoding="utf-8"))


def _validate_creation_mode(goal: GoalSpec, *, dry_run: bool) -> None:
    if goal.initial_operator_id == 'data.catalog_trial' or goal.objective.get('host_workload'):
        if dry_run:
            raise ValueError('registered entry executes its explicitly bound real or mock profile')
        return
    if goal.initial_operator_id == 'data.graph':
        if dry_run:
            raise ValueError('compiled graph executes only its explicitly bound real or mock profile')
        if not goal.objective.get('compiled_run'):
            raise ValueError('compiled graph requires a compiled run binding')
        return
    if goal.initial_operator_id in {"data.image_curation", "data.image_recipe", "data.visual_g2", "data.training_supervision", "data.training_rsi_trial"}:
        if dry_run or goal.objective.get("real_data_execution") is not True:
            raise ValueError("Image curation requires explicit real_data_execution and must not use --dry-run")
        return
    if goal.initial_operator_id == "dataevolver.3d_asset_evolution":
        if dry_run:
            raise ValueError("Agentic 3D execution must not use --dry-run")
        required = (
            "DATAEVOLVER_AGENT_BASE_URL",
            "DATAEVOLVER_AGENT_API_KEY",
            "DATAEVOLVER_AGENT_MODEL",
            "QWEN_IMAGE_MODEL_PATH",
            "SAM3_CKPT",
            "SAM3_DIR",
            "HUNYUAN3D_REPO",
            "MODEL_HUB",
            "PAINT_MODEL_HUB",
            "DINO_MODEL_PATH",
            "VLM_MODEL_PATH",
            "BLENDER_BIN",
            "DATAEVOLVER_3D_SCENE_TEMPLATE",
            "DATAEVOLVER_3D_GPU_ID",
        )
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if missing:
            raise ValueError(f"Agentic 3D creation is missing environment: {', '.join(missing)}")
        missing_paths = [
            name
            for name in required[3:-1]
            if not Path(os.environ[name]).expanduser().exists()
        ]
        if missing_paths:
            raise ValueError(f"Agentic 3D resource path does not exist: {', '.join(missing_paths)}")
        int(os.environ["DATAEVOLVER_3D_GPU_ID"])
        return
    if goal.initial_operator_id == "multimodal.t2i_canary":
        if dry_run:
            raise ValueError("Real T2I canary must not use --dry-run")
        return
    if not dry_run:
        raise ValueError("Phase 0-3A mock Operators require --dry-run")


def _production_preflight(goal: GoalSpec, *, urlopen=urllib.request.urlopen, run=subprocess.run) -> dict:
    asset_ids = [str(item) for item in goal.objective.get("asset_ids", [])]
    allowed_kinds = {
        "PROMPT_REGENERATE",
        "RESEGMENT_RECONSTRUCT",
        "RECONSTRUCT_RETRY",
        "SCENE_REFINE",
    }
    if goal.objective.get("real_canary") is not True or goal.planner.get("kind") != "agentic_rsi":
        raise ValueError("Phase 3D requires an explicit real_canary Agentic Goal")
    if asset_ids != list(dict.fromkeys(asset_ids)) or len(asset_ids) != 1:
        raise ValueError("Phase 3D requires exactly one unique asset")
    if set(goal.initial_operator_parameters) != {"initial_prompt", "t2i_seed", "reconstruction_seed"}:
        raise ValueError("Phase 3D initial Operator parameters do not match the fixed canary contract")
    if not isinstance(goal.initial_operator_parameters.get("initial_prompt"), str) or not goal.initial_operator_parameters["initial_prompt"].strip():
        raise ValueError("Phase 3D requires one fixed non-empty initial prompt")
    if set(goal.objective.get("allowed_experiment_kinds", [])) != allowed_kinds:
        raise ValueError("Phase 3D experiment kinds do not match the frozen allowlist")
    if goal.objective.get("representative_views") != [[0, 0], [90, 0], [180, 0], [270, 0]]:
        raise ValueError("Phase 3D requires the frozen four-view contract")
    if goal.budgets.max_experiments > 3 or goal.budgets.max_operator_calls > 4:
        raise ValueError("Phase 3D candidate/operator budget exceeds the bounded canary contract")
    if goal.budgets.max_planner_calls <= 0 or goal.budgets.max_planner_calls > 4:
        raise ValueError("Phase 3D planner-call budget must be between 1 and 4")

    base_url = os.environ["DATAEVOLVER_AGENT_BASE_URL"].rstrip("/")
    api_key = os.environ["DATAEVOLVER_AGENT_API_KEY"]
    model = os.environ["DATAEVOLVER_AGENT_MODEL"]
    request = urllib.request.Request(
        f"{base_url}/models",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            models_payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Agent endpoint unavailable: {exc}") from exc
    model_ids = {str(item.get("id") or "") for item in models_payload.get("data", [])}
    if model not in model_ids:
        raise ValueError(f"Agent model is not served by the endpoint: {model}")

    scene_template_path = Path(os.environ["DATAEVOLVER_3D_SCENE_TEMPLATE"]).expanduser().resolve()
    scene_template = json.loads(scene_template_path.read_text(encoding="utf-8"))
    blend_path = Path(str(scene_template.get("blend_path") or "")).expanduser().resolve()
    if scene_template.get("require_scene_contract") is not True or not blend_path.is_file():
        raise ValueError("Agentic 3D scene template or referenced blend file is unavailable")
    action_space_path = (CONFIGS_ROOT / "scene_action_space.json").resolve()
    json.loads(action_space_path.read_text(encoding="utf-8"))

    environment = dict(os.environ)
    python_path_parts = [str(SRC_ROOT)]
    python_path_parts.extend(
        str(Path(os.environ[name]).expanduser().resolve())
        for name in ("HUNYUAN3D_REPO", "SAM3_DIR")
    )
    if environment.get("PYTHONPATH"):
        python_path_parts.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    dependency = run(
        [sys.executable, "-c", "import torch, PIL, numpy, trimesh, diffsynth"],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90,
        check=False,
        shell=False,
    )
    if dependency.returncode != 0:
        raise ValueError(f"Agentic 3D Python dependency preflight failed: {(dependency.stderr or dependency.stdout)[-1000:]}")
    blender = run(
        [os.environ["BLENDER_BIN"], "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )
    if blender.returncode != 0:
        raise ValueError(f"Blender preflight failed: {(blender.stderr or blender.stdout)[-1000:]}")

    gpu_id = int(os.environ["DATAEVOLVER_3D_GPU_ID"])
    gpu = run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )
    if gpu.returncode != 0:
        raise ValueError(f"GPU preflight failed: {(gpu.stderr or gpu.stdout)[-1000:]}")
    try:
        index, memory_used, memory_total, utilization = [int(part.strip()) for part in gpu.stdout.strip().split(",")]
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU preflight returned malformed metrics") from exc
    max_used = int(goal.resource_policy.get("max_preflight_gpu_memory_mb", 500))
    if index != gpu_id or memory_used >= max_used:
        raise ValueError(f"GPU {gpu_id} is not available for the bounded canary: {memory_used} MiB used")

    return {
        "schema_version": "dataevolver.harness.production_preflight.v1",
        "status": "VERIFIED",
        "agent_model": model,
        "agent_base_url_hash": hashlib.sha256(base_url.encode("utf-8")).hexdigest(),
        "scene_template": str(scene_template_path),
        "scene_template_hash": hashlib.sha256(scene_template_path.read_bytes()).hexdigest(),
        "blend_hash": hashlib.sha256(blend_path.read_bytes()).hexdigest(),
        "action_space_hash": hashlib.sha256(action_space_path.read_bytes()).hexdigest(),
        "gpu": {
            "index": index,
            "memory_used_mb": memory_used,
            "memory_total_mb": memory_total,
            "utilization_percent": utilization,
        },
        "python_dependencies": "VERIFIED",
        "blender_version": (blender.stdout or blender.stderr).splitlines()[0],
    }


def _evaluator_path(goal: GoalSpec) -> Path:
    path = Path(goal.frozen_evaluator["path"])
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _registry(goal: GoalSpec, campaign_root: Path) -> OperatorRegistry:
    registry = OperatorRegistry()
    if goal.objective.get('host_workload'):
        from .runtime_references import host_workload
        operator = host_workload(goal.objective['host_workload'])
        if operator.spec.operator_id != goal.initial_operator_id or goal.interventions:
            raise ValueError('host workload differs from its frozen registered operator')
        if goal.objective['host_workload']['kind'] == 'native_scene' and operator.output_root != (campaign_root / goal.campaign_id).resolve():
            raise ValueError('scene workload must write inside its own campaign')
        registry.register(operator)
        return registry
    if goal.initial_operator_id == 'data.graph':
        from .plan_compiler import CompiledRun
        compiled = CompiledRun.from_dict(goal.objective['compiled_run'])
        if compiled.contract != goal.initial_operator_parameters or goal.interventions:
            raise ValueError('compiled graph goal changed')
        from .runtime_references import step_registry, graph_dependencies
        steps = step_registry(goal.objective.get('step_bindings', {}))
        roots = goal.objective.get('artifact_roots')
        dependencies = graph_dependencies(goal.objective.get('graph_runtime', {}), compiled.contract, roots)
        registry.register(compiled.bind(steps, allowed_write_root=campaign_root/goal.campaign_id,
                                       roots=roots, **dependencies))
        return registry
    if goal.initial_operator_id == 'data.training_rsi_trial':
        from .training_rsi_run import build_runtime
        return build_runtime(goal, campaign_root)[0]
    if goal.initial_operator_id == "data.training_supervision":
        from .adapters.training_supervision import TrainingSupervisionOperator
        if (goal.interventions or goal.planner.get("kind", "deterministic") != "deterministic" or
                goal.resource_policy.get("allow_gpu") is not True or goal.resource_policy.get("allow_network") is not False or
                goal.budgets.max_operator_calls != 1 or goal.budgets.max_rounds != 1 or
                goal.budgets.max_experiments != 0 or goal.budgets.max_retries != 0):
            raise ValueError("training production requires one offline batch with explicit step recovery")
        registry.register(TrainingSupervisionOperator(allowed_write_root=campaign_root/goal.campaign_id,
                          contract=goal.initial_operator_parameters))
        return registry
    if goal.initial_operator_id == "data.visual_g2":
        from .adapters.visual_g2 import VisualG2Operator
        if (goal.interventions or goal.planner.get("kind", "deterministic") != "deterministic" or
                goal.resource_policy.get("allow_gpu") is not True or
                goal.resource_policy.get("allow_network") is not False or
                goal.budgets.max_operator_calls != 1 or goal.budgets.max_rounds != 1 or
                goal.budgets.max_experiments != 0 or goal.budgets.max_retries != 0):
            raise ValueError("G2 Step 1 requires one real offline GPU batch, fixed M and no retries/experiments")
        registry.register(VisualG2Operator(allowed_write_root=campaign_root/goal.campaign_id,
                          contract=goal.initial_operator_parameters))
        return registry
    if goal.initial_operator_id == "data.image_recipe":
        from .adapters.image_recipe import ImageRecipeOperator
        if (goal.planner.get("kind") != "recipe_rsi" or goal.interventions or
                goal.resource_policy.get("allow_gpu") is not False or goal.resource_policy.get("allow_network") is not False or
                goal.objective.get("real_data_execution") is not True):
            raise ValueError("recipe RSI requires explicit offline CPU authority and local roles")
        if (not 1 <= goal.budgets.max_experiments <= 3 or not 1 <= goal.budgets.max_planner_calls <= 3 or
                goal.budgets.max_operator_calls > 5 or goal.budgets.max_rounds > 4):
            raise ValueError("recipe search exceeds the three-candidate first-round budget")
        registry.register(ImageRecipeOperator(allowed_write_root=campaign_root/goal.campaign_id,
                          contract=goal.objective["recipe_contract"]))
        return registry
    if goal.initial_operator_id == "data.image_curation":
        from .adapters.data_graph import DataGraphOperator
        if goal.interventions or goal.resource_policy.get("allow_gpu") is not False or goal.resource_policy.get("allow_network") is not False:
            raise ValueError("P2 image curation requires offline CPU execution without interventions")
        if goal.budgets.max_operator_calls != 1 or goal.budgets.max_rounds != 1 or goal.budgets.max_experiments != 0:
            raise ValueError("P2 image curation binds one graph execution per campaign")
        registry.register(DataGraphOperator(allowed_write_root=campaign_root / goal.campaign_id,
                                            contract=goal.initial_operator_parameters))
        return registry
    if goal.initial_operator_id == "visual.contract_mock":
        from .adapters.visual_mock import VisualContractMockOperator
        if goal.interventions or goal.resource_policy.get("allow_gpu") is not False or goal.resource_policy.get("allow_network") is not False:
            raise ValueError("visual contract mock requires no interventions, GPU or network permissions")
        registry.register(VisualContractMockOperator(
            allowed_write_root=campaign_root / goal.campaign_id,
            contract=goal.initial_operator_parameters,
        ))
        return registry
    if goal.initial_operator_id == "dataevolver.3d_asset_evolution":
        if goal.interventions:
            raise ValueError("Agentic 3D GoalSpec must not declare static interventions")
        resource_names = (
            "QWEN_IMAGE_MODEL_PATH", "SAM3_CKPT", "SAM3_DIR", "HUNYUAN3D_REPO", "MODEL_HUB",
            "PAINT_MODEL_HUB", "DINO_MODEL_PATH", "VLM_MODEL_PATH", "BLENDER_BIN", "DATAEVOLVER_3D_SCENE_TEMPLATE",
        )
        resource_config = {
            "python_executable": sys.executable,
            "blender_executable": os.environ.get("BLENDER_BIN", "blender"),
            "scene_template": os.environ.get("DATAEVOLVER_3D_SCENE_TEMPLATE", ""),
            "action_space": str((CONFIGS_ROOT / "scene_action_space.json").resolve()),
            "device": "cuda:0",
            "physical_gpu_id": int(os.environ.get("DATAEVOLVER_3D_GPU_ID", "0")),
            "render_resolution": int(goal.objective.get("render_resolution", 512)),
            "t2i_height": int(goal.objective.get("t2i_height", 512)),
            "t2i_width": int(goal.objective.get("t2i_width", 512)),
            "t2i_steps": int(goal.objective.get("t2i_steps", 20)),
            "representative_views": [[0, 0], [90, 0], [180, 0], [270, 0]],
            "require_texture": True,
            "resource_paths": {name: os.environ.get(name, "") for name in resource_names},
        }
        registry.register(
            ThreeDAssetEvolutionOperator(
                allowed_write_root=campaign_root / goal.campaign_id,
                resource_config=resource_config,
            )
        )
        return registry
    if goal.initial_operator_id == "multimodal.t2i_canary":
        if goal.interventions:
            raise ValueError("Real T2I canary GoalSpec must not declare interventions")
        registry.register(
            MultimodalT2ICanaryOperator(allowed_write_root=campaign_root / goal.campaign_id)
        )
        return registry
    if goal.initial_operator_id == "multimodal.t2i_dryrun":
        if goal.interventions:
            raise ValueError("T2I dry-run GoalSpec must not declare interventions")
        registry.register(
            MultimodalT2IDryRunOperator(allowed_write_root=campaign_root / goal.campaign_id)
        )
        return registry
    if goal.initial_operator_id != "fake.baseline":
        raise ValueError(f"Operator is not registered by the V1 CLI: {goal.initial_operator_id}")
    fixture = goal.objective.get("dry_run_fixture")
    if not isinstance(fixture, dict):
        raise ValueError("V1 has no real Operator adapters; use a GoalSpec with dry_run_fixture")
    source = Path(__file__).with_name("operators.py").resolve()
    model_hash = str((goal.objective.get("initial_trial_key") or {}).get("model_config_hash") or "")

    def build(operator_id: str, row: dict) -> FakeOperator:
        spec = OperatorSpec(
            operator_id=operator_id,
            version="1",
            retry_safe=True,
            expected_output_specs=(ExpectedOutputSpec("effect.json"),),
            model_config_hash=model_hash,
            code_revision=sha256_file(source),
            source_path=str(source),
        )
        return FakeOperator(
            spec,
            status=StageStatus(row["status"]),
            metrics=dict(row.get("metrics") or {}),
            hard_constraints=dict(row.get("hard_constraints") or {}),
            failure_code=str(row.get("failure_code") or ""),
        )

    registry.register(build(goal.initial_operator_id, fixture["baseline"]))
    candidate_ids = {item["operator_id"] for item in goal.interventions}
    if len(candidate_ids) != 1:
        raise ValueError("V1 dry-run fixture requires exactly one candidate operator")
    registry.register(build(next(iter(candidate_ids)), fixture["candidate"]))
    return registry


def _planner(goal: GoalSpec, campaign_root: Path | None = None):
    kind = str(goal.planner.get("kind") or "deterministic")
    if kind == "deterministic":
        return DeterministicPlanner(goal.interventions)
    if kind != "agentic_rsi":
        raise ValueError(f"planner kind is not registered: {kind}")
    config_path = Path(str(goal.planner.get("config_path") or ""))
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise ValueError(f"agentic planner config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("backend") == "model_gateway":
        from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
        from dataevolver.adapters.model_gateway import ModelGateway
        from .model_backend import GatewayPlannerBackend
        if campaign_root is None:
            raise ValueError("model gateway planner requires an explicit campaign root")
        gateway = ModelGateway(
            profiles=tuple(ModelProfile.from_dict(x) for x in config["profiles"]),
            policy=ModelPolicy.from_dict(config["policy"]),
            receipt_root=campaign_root / goal.campaign_id / "planner" / "model_receipts",
        )
        backend = GatewayPlannerBackend(gateway, max_tokens=config.get("max_tokens", 1024))
        return AgenticRSIPlanner(backend=backend, config={**config, "gateway_contract_hash": gateway.contract_hash()})
    if config.get("backend") != "openai_compatible":
        raise ValueError(f"planner backend is not registered: {config.get('backend')}")
    return AgenticRSIPlanner(backend=OpenAICompatiblePlannerBackend(config), config=config)


def _controller(goal: GoalSpec, campaign_root: Path, *, create: bool, idempotency_key: str = ""):
    evaluator = EvaluatorContract.from_path(_evaluator_path(goal))
    if evaluator.sha256 != goal.frozen_evaluator["sha256"]:
        raise ValueError("evaluator hash does not match GoalSpec")
    if goal.planner.get('kind') == 'runtime_rsi':
        if goal.initial_operator_id == 'data.catalog_trial':
            from .runtime_entry import build_catalog_runtime
            registry, planner = build_catalog_runtime(goal, campaign_root)
        else:
            from .training_rsi_run import build_runtime
            registry, planner = build_runtime(goal, campaign_root)
    else:
        registry = _registry(goal, campaign_root)
    if goal.planner.get('kind') == 'runtime_rsi':
        pass
    elif goal.planner.get("kind") == "recipe_rsi":
        from .adapters.image_recipe import RecipePlanner
        planner = RecipePlanner(registry.get("data.image_recipe"))
    else:
        planner = _planner(goal, campaign_root)
    compiler = None
    if goal.planner.get("kind") == "agentic_rsi":
        spec = registry.get("dataevolver.3d_asset_evolution").spec
        compiler = ExperimentCompiler(
            operator_id=spec.operator_id,
            operator_version=spec.version,
            allowed_scene_actions=set(str(item) for item in goal.objective.get("allowed_scene_actions", [])),
        )
    schema_paths = tuple(sorted(CONFIGS_ROOT.glob("schemas/harness_*.schema.json")))
    execution_paths = list(sorted(Path(__file__).resolve().parent.rglob("*.py")))
    if goal.initial_operator_id in {"visual.contract_mock", "data.image_curation", "data.image_recipe", "data.visual_g2", "data.training_supervision"} or getattr(getattr(planner, "backend", None), "backend_id", "") == "model-gateway-v1":
        execution_paths.extend(REPO_ROOT / path for path in (
            "src/dataevolver/adapters/model_contracts.py",
            "src/dataevolver/adapters/model_gateway.py",
            "src/dataevolver/dataset/contracts.py",
        ))
    if goal.initial_operator_id in {"data.image_curation", "data.image_recipe", "data.graph"}:
        execution_paths.extend(REPO_ROOT / path for path in (
            "src/dataevolver/dataset/curation.py",
            "src/dataevolver/dataset/curation_loader.py",
            "src/dataevolver/adapters/clip_worker.py",
            "src/dataevolver/adapters/chat_worker.py",
            "src/dataevolver/dataset/quality.py",
            "src/dataevolver/dataset/curation_quality.py",
            "src/dataevolver/dataset/families.py",
            "src/dataevolver/dataset/catalog_steps.py",
        ))
    if goal.initial_operator_id == "data.training_supervision":
        execution_paths.extend(REPO_ROOT / path for path in (
            "src/dataevolver/adapters/training_media_worker.py",
            "src/dataevolver/dataset/supervision.py",
            "src/dataevolver/dataset/training_pipeline.py",
        ))
    if goal.initial_operator_id == "data.visual_g2":
        execution_paths.extend(REPO_ROOT / path for path in (
            "src/dataevolver/adapters/visual_worker.py",
            "src/dataevolver/adapters/chat_worker.py",
            "src/dataevolver/adapters/clip_worker.py",
            "src/dataevolver/dataset/g2_contracts.py",
            "src/dataevolver/dataset/g2_pipeline.py",
            "src/dataevolver/dataset/g2_export.py",
        ))
    if goal.planner.get("kind") == "agentic_rsi":
        planner_config_path = Path(str(goal.planner.get("config_path") or ""))
        if not planner_config_path.is_absolute():
            planner_config_path = (REPO_ROOT / planner_config_path).resolve()
        execution_paths.extend(
            (
                planner_config_path,
                REPO_ROOT / "src/dataevolver/workflows/stages/t2i_generate.py",
                REPO_ROOT / "src/dataevolver/workflows/stages/sam_segment.py",
                REPO_ROOT / "src/dataevolver/workflows/stages/image_to_3d.py",
                REPO_ROOT / "src/dataevolver/workflows/stages/mesh_sanitize.py",
                REPO_ROOT / "src/dataevolver/workflows/stages/scene_render.py",
                REPO_ROOT / "src/dataevolver/annotation/vlm_review_stage.py",
                REPO_ROOT / "src/dataevolver/agents/feedback_apply.py",
                CONFIGS_ROOT / "scene_action_space.json",
            )
        )
        scene_template = os.environ.get("DATAEVOLVER_3D_SCENE_TEMPLATE", "").strip()
        if scene_template:
            execution_paths.append(Path(scene_template).expanduser().resolve())
    execution_paths = tuple(execution_paths)
    if goal.planner.get('kind') == 'runtime_rsi':
        execution_paths = tuple(sorted((REPO_ROOT / 'src/dataevolver').rglob('*.py')))
    kwargs = dict(
        repo_root=REPO_ROOT,
        campaigns_root=campaign_root,
        goal=goal,
        registry=registry,
        planner=planner,
        evaluator_contract=evaluator,
        schema_paths=schema_paths,
        execution_paths=execution_paths,
        experiment_compiler=compiler,
    )
    if create:
        controller = EvolutionController.create(**kwargs, idempotency_key=idempotency_key)
        state = controller.status()
        if state.idempotency_key != idempotency_key:
            raise ValueError("campaign exists with a different top-level idempotency key")
    else:
        controller = EvolutionController.open(**kwargs)
    from .runtime_entry import admit_execution, version_bindings
    bindings = version_bindings(controller)
    if bindings['policy'] or bindings['steps']:
        if create:
            from .state_store import CampaignLease
            with CampaignLease(controller.layout.lease_path, ttl_seconds=goal.lease_seconds):
                admit_execution(controller)
    return controller


def _existing_controller(campaign_root: Path, campaign_id: str):
    layout = CampaignLayout(campaign_root, campaign_id)
    if not layout.goal_path.is_file():
        raise FileNotFoundError(f"campaign does not exist: {campaign_id}")
    return _controller(_load_goal(layout.goal_path), campaign_root, create=False)


def _print(payload) -> None:
    print(canonical_json(payload))


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.command in {'compile-plan', 'operator-sdk', 'visual-models', 'worker-compile', 'scene-compile'}:
            from .framework_cli import handle
            result = handle(args)
            _print(result)
            return 2 if result.get('compiled') is False else 0
        if args.command == "models":
            from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
            from dataevolver.adapters.model_gateway import ModelError, ModelGateway
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            gateway = ModelGateway(
                profiles=tuple(ModelProfile.from_dict(x) for x in config["profiles"]),
                policy=ModelPolicy.from_dict(config["policy"]),
                receipt_root=Path(args.config).resolve().parent / ".unused-model-inspection",
            )
            results = {}
            for role in gateway.policy.bindings:
                try:
                    results[role] = gateway.identity(role)
                    results[role]["health"] = gateway.health(role) if args.check_endpoint else {"status": "NOT_CHECKED"}
                except ModelError as exc:
                    results[role] = {"status": exc.code}
            _print({"external_api_enabled": False, "inference_performed": False, "roles": results})
            return 0
        campaign_root = Path(args.campaign_root)
        if args.command in {"create", "evolve"}:
            goal = _load_goal(Path(args.goal))
            _validate_creation_mode(goal, dry_run=args.dry_run)
            layout = CampaignLayout(campaign_root, goal.campaign_id)
            preflight = None
            if goal.initial_operator_id == "dataevolver.3d_asset_evolution" and not layout.goal_path.exists():
                preflight = _production_preflight(goal)
            controller = _controller(
                goal,
                campaign_root,
                create=True,
                idempotency_key=args.idempotency_key,
            )
            if preflight is not None:
                atomic_write_json(controller.layout.campaign_dir / "preflight.json", preflight)
            _print(controller.status() if args.command == 'create' else controller.run())
            return 0

        if args.command in {'status', 'replay', 'explain'}:
            # Inspection remains available after revocation or runtime drift.
            from dataevolver.dataset.contracts import identifier
            identifier(args.campaign)
            store = StateStore(CampaignLayout(campaign_root, args.campaign))
            snapshot = store.load_state()
            if args.command == 'status':
                _print(snapshot)
            elif args.command == 'replay':
                replayed = store.replay()
                _print({'state': replayed.to_dict(), 'matches_snapshot': replayed.to_dict() == snapshot.to_dict()})
            else:
                _print({'state': snapshot.to_dict(), 'events': [e.to_dict() for e in store.events.read()[-max(1,args.limit):]]})
            return 0

        if args.command == "step" and args.goal:
            goal = _load_goal(Path(args.goal))
            _validate_creation_mode(goal, dry_run=args.dry_run)
            if goal.campaign_id != args.campaign:
                raise ValueError("--campaign does not match GoalSpec campaign_id")
            layout = CampaignLayout(campaign_root, goal.campaign_id)
            preflight = None
            if goal.initial_operator_id == "dataevolver.3d_asset_evolution" and not layout.goal_path.exists():
                preflight = _production_preflight(goal)
            controller = _controller(
                goal,
                campaign_root,
                create=True,
                idempotency_key=args.idempotency_key,
            )
            if preflight is not None:
                atomic_write_json(controller.layout.campaign_dir / "preflight.json", preflight)
        else:
            controller = _existing_controller(campaign_root, args.campaign)

        if args.command == "step":
            _print(controller.tick(args.max_actions))
        elif args.command == "pause":
            controller.pause()
            controller.tick()
            _print(controller.status())
        elif args.command == "resume":
            _print(controller.resume())
        elif args.command == "status":
            _print(controller.status())
        elif args.command == "operators":
            catalog = controller.registry.catalog()
            frozen = json.loads(controller.layout.frozen_contract_path.read_text())
            drift = controller.contract_guard.verify(frozen)
            catalog.update(campaign_id=args.campaign, frozen_contract_matches=drift.valid,
                           contract_drift=list(drift.changed_fields))
            _print(catalog)
        elif args.command == "explain":
            _print(controller.explain(args.limit))
        elif args.command == "replay":
            replayed = controller.replay()
            snapshot = controller.status()
            _print({"state": replayed.to_dict(), "matches_snapshot": replayed.to_dict() == snapshot.to_dict()})
        return 0
    except Exception as exc:
        print(f"[harness] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
