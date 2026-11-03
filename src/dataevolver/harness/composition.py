"""Explicit campaign composition; CLI parsing and output live in cli.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dataevolver.paths import CONFIGS_ROOT, REPO_ROOT

from .adapters import MultimodalT2IDryRunOperator, ThreeDAssetEvolutionOperator
from .adapters.multimodal_canary import MultimodalT2ICanaryOperator
from .adapters.three_d_asset import validate_creation_resources
from .controller import EvolutionController
from .evaluator import EvaluatorContract, sha256_file
from .models import ExpectedOutputSpec, GoalSpec, OperatorSpec, StageStatus
from .operators import FakeOperator, OperatorRegistry
from .planner import (
    AgenticRSIPlanner,
    DeterministicPlanner,
    OpenAICompatiblePlannerBackend,
)
from .state_store import CampaignLayout
from .strategy import ExperimentCompiler


def load_goal(path: Path) -> GoalSpec:
    """Read a GoalSpec using the same versioned parser as campaign recovery."""
    return GoalSpec.from_json(path.read_text(encoding="utf-8"))


def validate_creation_mode(goal: GoalSpec, *, dry_run: bool) -> None:
    """Check route permissions before constructing or opening a campaign."""
    from .image_pipeline import is_image_pipeline
    if is_image_pipeline(goal):
        mode = goal.initial_operator_parameters["intent"]["parameters"].get("execution_mode")
        if mode not in {"fixture", "server"} or dry_run != (mode == "fixture"):
            raise ValueError("image fixture requires --dry-run; server execution must omit it")
        if goal.objective.get("real_data_execution") is not (mode == "server"):
            raise ValueError("image execution mode differs from real_data_execution authority")
        return
    if goal.initial_operator_id == "data.catalog_trial" or goal.objective.get(
        "host_workload"
    ):
        if dry_run:
            raise ValueError(
                "registered entry executes its explicitly bound real or mock profile"
            )
        return
    if goal.initial_operator_id == "data.graph":
        if dry_run:
            raise ValueError(
                "compiled graph executes only its explicitly bound real or mock profile"
            )
        if not goal.objective.get("compiled_run"):
            raise ValueError("compiled graph requires a compiled run binding")
        return
    if goal.initial_operator_id in {
        "data.image_curation",
        "data.image_recipe",
        "data.visual_g2",
        "data.training_supervision",
        "data.training_rsi_trial",
    }:
        if dry_run or goal.objective.get("real_data_execution") is not True:
            raise ValueError(
                "Image curation requires explicit real_data_execution and must not use --dry-run"
            )
        return
    if goal.initial_operator_id == "dataevolver.3d_asset_evolution":
        validate_creation_resources(dry_run=dry_run)
        return
    if goal.initial_operator_id == "multimodal.t2i_canary":
        if dry_run:
            raise ValueError("Real T2I canary must not use --dry-run")
        return
    if not dry_run:
        raise ValueError("Mock operators require --dry-run")


def _evaluator_path(goal: GoalSpec) -> Path:
    path = Path(goal.frozen_evaluator["path"])
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_registry(goal: GoalSpec, campaign_root: Path) -> OperatorRegistry:
    """Bind the GoalSpec's explicit route and its permitted operators."""
    registry = OperatorRegistry()
    from .image_pipeline import is_image_pipeline
    if is_image_pipeline(goal):
        from .adapters.image_pipeline import ImageCollectionOperator, ImageLayoutOperator
        mode = goal.initial_operator_parameters["intent"]["parameters"].get("execution_mode")
        if (mode not in {"fixture", "server"} or goal.interventions
                or goal.planner.get("kind", "deterministic") != "deterministic"
                or goal.resource_policy.get("allow_gpu") is not False
                or goal.resource_policy.get("allow_network") is not (mode == "server")
                or goal.objective.get("real_data_execution") is not (mode == "server")
                or goal.budgets.max_operator_calls != 1 or goal.budgets.max_rounds != 1
                or goal.budgets.max_experiments != 0 or goal.budgets.max_retries != 0):
            raise ValueError("image pipeline requires a bounded campaign with explicit fixture/server authority")
        operator = ImageCollectionOperator if goal.initial_operator_id == "data.image_collection" else ImageLayoutOperator
        registry.register(operator(allowed_write_root=campaign_root / goal.campaign_id,
                                   contract=goal.initial_operator_parameters))
        return registry
    if goal.objective.get("host_workload"):
        from .runtime_references import host_workload

        operator = host_workload(goal.objective["host_workload"])
        if operator.spec.operator_id != goal.initial_operator_id or goal.interventions:
            raise ValueError(
                "host workload differs from its frozen registered operator"
            )
        if (
            goal.objective["host_workload"]["kind"] == "native_scene"
            and operator.output_root != (campaign_root / goal.campaign_id).resolve()
        ):
            raise ValueError("scene workload must write inside its own campaign")
        registry.register(operator)
        return registry
    if goal.initial_operator_id == "data.graph":
        from .plan_compiler import CompiledRun

        compiled = CompiledRun.from_dict(goal.objective["compiled_run"])
        if compiled.contract != goal.initial_operator_parameters or goal.interventions:
            raise ValueError("compiled graph goal changed")
        from .runtime_references import step_registry, graph_dependencies

        steps = step_registry(goal.objective.get("step_bindings", {}))
        roots = goal.objective.get("artifact_roots")
        dependencies = graph_dependencies(
            goal.objective.get("graph_runtime", {}), compiled.contract, roots
        )
        registry.register(
            compiled.bind(
                steps,
                allowed_write_root=campaign_root / goal.campaign_id,
                roots=roots,
                **dependencies,
            )
        )
        return registry
    if goal.initial_operator_id == "data.training_rsi_trial":
        from .training_rsi_run import build_runtime

        return build_runtime(goal, campaign_root)[0]
    if goal.initial_operator_id == "data.training_supervision":
        from .adapters.training_supervision import TrainingSupervisionOperator

        if (
            goal.interventions
            or goal.planner.get("kind", "deterministic") != "deterministic"
            or goal.resource_policy.get("allow_gpu") is not True
            or goal.resource_policy.get("allow_network") is not False
            or goal.budgets.max_operator_calls != 1
            or goal.budgets.max_rounds != 1
            or goal.budgets.max_experiments != 0
            or goal.budgets.max_retries != 0
        ):
            raise ValueError(
                "training production requires one offline batch with explicit step recovery"
            )
        registry.register(
            TrainingSupervisionOperator(
                allowed_write_root=campaign_root / goal.campaign_id,
                contract=goal.initial_operator_parameters,
            )
        )
        return registry
    if goal.initial_operator_id == "data.visual_g2":
        from .adapters.visual_g2 import VisualG2Operator

        if (
            goal.interventions
            or goal.planner.get("kind", "deterministic") != "deterministic"
            or goal.resource_policy.get("allow_gpu") is not True
            or goal.resource_policy.get("allow_network") is not False
            or goal.budgets.max_operator_calls != 1
            or goal.budgets.max_rounds != 1
            or goal.budgets.max_experiments != 0
            or goal.budgets.max_retries != 0
        ):
            raise ValueError(
                "G2 Step 1 requires one real offline GPU batch, fixed M and no retries/experiments"
            )
        registry.register(
            VisualG2Operator(
                allowed_write_root=campaign_root / goal.campaign_id,
                contract=goal.initial_operator_parameters,
            )
        )
        return registry
    if goal.initial_operator_id == "data.image_recipe":
        from .adapters.image_recipe import ImageRecipeOperator

        if (
            goal.planner.get("kind") != "recipe_rsi"
            or goal.interventions
            or goal.resource_policy.get("allow_gpu") is not False
            or goal.resource_policy.get("allow_network") is not False
            or goal.objective.get("real_data_execution") is not True
        ):
            raise ValueError(
                "recipe RSI requires explicit offline CPU authority and local roles"
            )
        if (
            not 1 <= goal.budgets.max_experiments <= 3
            or not 1 <= goal.budgets.max_planner_calls <= 3
            or goal.budgets.max_operator_calls > 5
            or goal.budgets.max_rounds > 4
        ):
            raise ValueError(
                "recipe search exceeds the three-candidate first-round budget"
            )
        registry.register(
            ImageRecipeOperator(
                allowed_write_root=campaign_root / goal.campaign_id,
                contract=goal.objective["recipe_contract"],
            )
        )
        return registry
    if goal.initial_operator_id == "data.image_curation":
        from .adapters.data_graph import DataGraphOperator

        if (
            goal.interventions
            or goal.resource_policy.get("allow_gpu") is not False
            or goal.resource_policy.get("allow_network") is not False
        ):
            raise ValueError(
                "Image curation requires offline CPU execution without interventions"
            )
        if (
            goal.budgets.max_operator_calls != 1
            or goal.budgets.max_rounds != 1
            or goal.budgets.max_experiments != 0
        ):
            raise ValueError("Image curation binds one graph execution per campaign")
        registry.register(
            DataGraphOperator(
                allowed_write_root=campaign_root / goal.campaign_id,
                contract=goal.initial_operator_parameters,
            )
        )
        return registry
    if goal.initial_operator_id == "visual.contract_mock":
        from .adapters.visual_mock import VisualContractMockOperator

        if (
            goal.interventions
            or goal.resource_policy.get("allow_gpu") is not False
            or goal.resource_policy.get("allow_network") is not False
        ):
            raise ValueError(
                "visual contract mock requires no interventions, GPU or network permissions"
            )
        registry.register(
            VisualContractMockOperator(
                allowed_write_root=campaign_root / goal.campaign_id,
                contract=goal.initial_operator_parameters,
            )
        )
        return registry
    if goal.initial_operator_id == "dataevolver.3d_asset_evolution":
        if goal.interventions:
            raise ValueError(
                "Agentic 3D GoalSpec must not declare static interventions"
            )
        resource_names = (
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
            "resource_paths": {
                name: os.environ.get(name, "") for name in resource_names
            },
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
            MultimodalT2ICanaryOperator(
                allowed_write_root=campaign_root / goal.campaign_id
            )
        )
        return registry
    if goal.initial_operator_id == "multimodal.t2i_dryrun":
        if goal.interventions:
            raise ValueError("T2I dry-run GoalSpec must not declare interventions")
        registry.register(
            MultimodalT2IDryRunOperator(
                allowed_write_root=campaign_root / goal.campaign_id
            )
        )
        return registry
    if goal.initial_operator_id != "fake.baseline":
        raise ValueError(
            f"Operator is not registered by the CLI: {goal.initial_operator_id}"
        )
    fixture = goal.objective.get("dry_run_fixture")
    if not isinstance(fixture, dict):
        raise ValueError("fake.baseline requires a GoalSpec with dry_run_fixture")
    source = Path(__file__).with_name("operators.py").resolve()
    model_hash = str(
        (goal.objective.get("initial_trial_key") or {}).get("model_config_hash") or ""
    )

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
        raise ValueError("Dry-run fixture requires exactly one candidate operator")
    registry.register(build(next(iter(candidate_ids)), fixture["candidate"]))
    return registry


def build_planner(
    goal: GoalSpec, campaign_root: Path | None = None
) -> DeterministicPlanner | AgenticRSIPlanner:
    """Construct a deterministic or agentic planner from its explicit config."""
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
            receipt_root=campaign_root
            / goal.campaign_id
            / "planner"
            / "model_receipts",
        )
        backend = GatewayPlannerBackend(
            gateway, max_tokens=config.get("max_tokens", 1024)
        )
        return AgenticRSIPlanner(
            backend=backend,
            config={**config, "gateway_contract_hash": gateway.contract_hash()},
        )
    if config.get("backend") != "openai_compatible":
        raise ValueError(f"planner backend is not registered: {config.get('backend')}")
    return AgenticRSIPlanner(
        backend=OpenAICompatiblePlannerBackend(config), config=config
    )


def build_controller(
    goal: GoalSpec, campaign_root: Path, *, create: bool, idempotency_key: str = ""
) -> EvolutionController:
    """Bind a campaign, freeze its sources and preserve runtime admission order."""
    evaluator = EvaluatorContract.from_path(_evaluator_path(goal))
    if evaluator.sha256 != goal.frozen_evaluator["sha256"]:
        raise ValueError("evaluator hash does not match GoalSpec")
    if goal.planner.get("kind") == "runtime_rsi":
        if goal.initial_operator_id == "data.catalog_trial":
            from .runtime_entry import build_catalog_runtime

            registry, planner = build_catalog_runtime(goal, campaign_root)
        else:
            from .training_rsi_run import build_runtime

            registry, planner = build_runtime(goal, campaign_root)
    else:
        registry = build_registry(goal, campaign_root)
    if goal.planner.get("kind") != "runtime_rsi":
        if goal.planner.get("kind") == "recipe_rsi":
            from .adapters.image_recipe import RecipePlanner

            planner = RecipePlanner(registry.get("data.image_recipe"))
        else:
            planner = build_planner(goal, campaign_root)
    compiler = None
    if goal.planner.get("kind") == "agentic_rsi":
        spec = registry.get("dataevolver.3d_asset_evolution").spec
        compiler = ExperimentCompiler(
            operator_id=spec.operator_id,
            operator_version=spec.version,
            allowed_scene_actions=set(
                str(item) for item in goal.objective.get("allowed_scene_actions", [])
            ),
        )
    schema_paths = tuple(sorted(CONFIGS_ROOT.glob("schemas/harness_*.schema.json")))
    execution_paths = list(sorted(Path(__file__).resolve().parent.rglob("*.py")))
    execution_paths.append(REPO_ROOT / "src/dataevolver/dataset/contracts.py")
    from .image_pipeline import is_image_pipeline
    if is_image_pipeline(goal):
        execution_paths.extend(REPO_ROOT / path for path in (
            "src/dataevolver/dataset/collection.py",
            "src/dataevolver/dataset/collection_spec.py",
            "src/dataevolver/dataset/image_records.py",
            "src/dataevolver/dataset/layout_quality.py",
            "src/dataevolver/dataset/curation_layout.py",
            "src/dataevolver/dataset/layout_loader.py",
            "src/dataevolver/adapters/model_contracts.py",
            "src/dataevolver/adapters/model_gateway.py",
        ))
    if (
        goal.initial_operator_id
        in {
            "visual.contract_mock",
            "data.image_curation",
            "data.image_recipe",
            "data.visual_g2",
            "data.training_supervision",
        }
        or getattr(getattr(planner, "backend", None), "backend_id", "")
        == "model-gateway-v1"
    ):
        execution_paths.extend(
            REPO_ROOT / path
            for path in (
                "src/dataevolver/adapters/model_contracts.py",
                "src/dataevolver/adapters/model_gateway.py",
            )
        )
    if goal.initial_operator_id in {
        "data.image_curation",
        "data.image_recipe",
        "data.graph",
    }:
        execution_paths.extend(
            REPO_ROOT / path
            for path in (
                "src/dataevolver/dataset/curation.py",
                "src/dataevolver/dataset/curation_loader.py",
                "src/dataevolver/adapters/clip_worker.py",
                "src/dataevolver/adapters/chat_worker.py",
                "src/dataevolver/dataset/quality.py",
                "src/dataevolver/dataset/curation_quality.py",
                "src/dataevolver/dataset/families.py",
                "src/dataevolver/dataset/catalog_steps.py",
            )
        )
    if goal.initial_operator_id == "data.training_supervision":
        execution_paths.extend(
            REPO_ROOT / path
            for path in (
                "src/dataevolver/adapters/training_media_worker.py",
                "src/dataevolver/dataset/supervision.py",
                "src/dataevolver/dataset/training_pipeline.py",
            )
        )
    if goal.initial_operator_id == "data.visual_g2":
        execution_paths.extend(
            REPO_ROOT / path
            for path in (
                "src/dataevolver/adapters/visual_worker.py",
                "src/dataevolver/adapters/chat_worker.py",
                "src/dataevolver/adapters/clip_worker.py",
                "src/dataevolver/dataset/g2_contracts.py",
                "src/dataevolver/dataset/g2_pipeline.py",
                "src/dataevolver/dataset/g2_export.py",
            )
        )
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
    if goal.planner.get("kind") == "runtime_rsi":
        execution_paths = tuple(sorted((REPO_ROOT / "src/dataevolver").rglob("*.py")))
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
        controller = EvolutionController.create(
            **kwargs, idempotency_key=idempotency_key
        )
        state = controller.status()
        if state.idempotency_key != idempotency_key:
            raise ValueError(
                "campaign exists with a different top-level idempotency key"
            )
    else:
        controller = EvolutionController.open(**kwargs)
    from .runtime_entry import admit_execution, version_bindings

    bindings = version_bindings(controller)
    if bindings["policy"] or bindings["steps"]:
        if create:
            from .state_store import CampaignLease

            with CampaignLease(
                controller.layout.lease_path, ttl_seconds=goal.lease_seconds
            ):
                admit_execution(controller)
    return controller


def open_controller(campaign_root: Path, campaign_id: str) -> EvolutionController:
    """Reconstruct an existing campaign from its stored GoalSpec."""
    layout = CampaignLayout(campaign_root, campaign_id)
    if not layout.goal_path.is_file():
        raise FileNotFoundError(f"campaign does not exist: {campaign_id}")
    return build_controller(load_goal(layout.goal_path), campaign_root, create=False)
