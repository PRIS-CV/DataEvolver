from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataevolver.paths import DATAEVOLVER_ROOT

from .adapters.three_d_asset import production_preflight as _production_preflight

# Keep the existing helper imports available to downstream callers.
from .composition import (
    _evaluator_path as _evaluator_path,
    build_controller as _controller,
    build_planner as _planner,  # noqa: F401 -- compatibility import
    build_registry as _registry,  # noqa: F401 -- compatibility import
    load_goal as _load_goal,
    open_controller as _existing_controller,
    validate_creation_mode as _validate_creation_mode,
)
from .models import canonical_json
from .state_store import CampaignLayout, StateStore, atomic_write_json


def _campaign_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-root", default=str(DATAEVOLVER_ROOT / "campaigns"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DataEvolver Harness")
    commands = parser.add_subparsers(dest="command", required=True)
    from .framework_cli import add_commands

    add_commands(commands)

    plan = commands.add_parser("plan", help="inspect an image pipeline without network or inference")
    plan.add_argument("--goal", required=True)

    models = commands.add_parser(
        "models", help="inspect explicit local model profiles without inference"
    )
    models.add_argument("--config", required=True)
    models.add_argument(
        "--check-endpoint",
        action="store_true",
        help="perform read-only /models health checks",
    )

    for name in ("create", "evolve"):
        evolve = commands.add_parser(name)
        evolve.add_argument("--goal", required=True)
        evolve.add_argument(
            "--campaign-root", default=str(DATAEVOLVER_ROOT / "campaigns")
        )
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


def _print(payload) -> None:
    print(canonical_json(payload))


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            from .image_pipeline import offline_plan
            _print(offline_plan(_load_goal(Path(args.goal))))
            return 0
        if args.command in {
            "compile-plan",
            "operator-sdk",
            "visual-models",
            "worker-compile",
            "scene-compile",
        }:
            from .framework_cli import handle

            result = handle(args)
            _print(result)
            return 2 if result.get("compiled") is False else 0
        if args.command == "models":
            from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
            from dataevolver.adapters.model_gateway import ModelError, ModelGateway

            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            gateway = ModelGateway(
                profiles=tuple(ModelProfile.from_dict(x) for x in config["profiles"]),
                policy=ModelPolicy.from_dict(config["policy"]),
                receipt_root=Path(args.config).resolve().parent
                / ".unused-model-inspection",
            )
            results = {}
            for role in gateway.policy.bindings:
                try:
                    results[role] = gateway.identity(role)
                    results[role]["health"] = (
                        gateway.health(role)
                        if args.check_endpoint
                        else {"status": "NOT_CHECKED"}
                    )
                except ModelError as exc:
                    results[role] = {"status": exc.code}
            _print(
                {
                    "external_api_enabled": False,
                    "inference_performed": False,
                    "roles": results,
                }
            )
            return 0
        campaign_root = Path(args.campaign_root)
        if args.command in {"create", "evolve"}:
            goal = _load_goal(Path(args.goal))
            _validate_creation_mode(goal, dry_run=args.dry_run)
            layout = CampaignLayout(campaign_root, goal.campaign_id)
            preflight = None
            if (
                goal.initial_operator_id == "dataevolver.3d_asset_evolution"
                and not layout.goal_path.exists()
            ):
                preflight = _production_preflight(goal)
            controller = _controller(
                goal,
                campaign_root,
                create=True,
                idempotency_key=args.idempotency_key,
            )
            if preflight is not None:
                atomic_write_json(
                    controller.layout.campaign_dir / "preflight.json", preflight
                )
            _print(
                controller.status() if args.command == "create" else controller.run()
            )
            return 0

        if args.command in {"status", "replay", "explain"}:
            # Inspection remains available after revocation or runtime drift.
            from dataevolver.dataset.contracts import identifier

            identifier(args.campaign)
            store = StateStore(CampaignLayout(campaign_root, args.campaign))
            snapshot = store.load_state()
            if args.command == "status":
                _print(snapshot)
            elif args.command == "replay":
                replayed = store.replay()
                _print(
                    {
                        "state": replayed.to_dict(),
                        "matches_snapshot": replayed.to_dict() == snapshot.to_dict(),
                    }
                )
            else:
                _print(
                    {
                        "state": snapshot.to_dict(),
                        "events": [
                            e.to_dict()
                            for e in store.events.read()[-max(1, args.limit) :]
                        ],
                    }
                )
            return 0

        if args.command == "step" and args.goal:
            goal = _load_goal(Path(args.goal))
            _validate_creation_mode(goal, dry_run=args.dry_run)
            if goal.campaign_id != args.campaign:
                raise ValueError("--campaign does not match GoalSpec campaign_id")
            layout = CampaignLayout(campaign_root, goal.campaign_id)
            preflight = None
            if (
                goal.initial_operator_id == "dataevolver.3d_asset_evolution"
                and not layout.goal_path.exists()
            ):
                preflight = _production_preflight(goal)
            controller = _controller(
                goal,
                campaign_root,
                create=True,
                idempotency_key=args.idempotency_key,
            )
            if preflight is not None:
                atomic_write_json(
                    controller.layout.campaign_dir / "preflight.json", preflight
                )
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
        elif args.command == "operators":
            catalog = controller.registry.catalog()
            frozen = json.loads(controller.layout.frozen_contract_path.read_text())
            drift = controller.contract_guard.verify(frozen)
            catalog.update(
                campaign_id=args.campaign,
                frozen_contract_matches=drift.valid,
                contract_drift=list(drift.changed_fields),
            )
            _print(catalog)
        return 0
    except Exception as exc:
        print(f"[harness] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
