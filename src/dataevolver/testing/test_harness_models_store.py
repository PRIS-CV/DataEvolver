import hashlib
import json
from datetime import datetime, timezone

import pytest

from dataevolver.harness.models import (
    BudgetSpec,
    EvidenceStatus,
    EvidenceValue,
    GoalSpec,
    MatchedTrialKey,
)
from dataevolver.harness.events import EventLog
from dataevolver.harness.models import CampaignState, ControllerState
from dataevolver.harness.state_store import CampaignLayout, CampaignLease, LeaseBusyError, StateStore


def test_goal_and_evidence_use_stable_canonical_json():
    goal = GoalSpec.from_dict(
        {
            "schema_version": "dataevolver.harness.goal.v1",
            "campaign_id": "demo",
            "objective": {"kind": "quality", "target_count": 1},
            "initial_operator_id": "fake.baseline",
            "route_candidates": ["fake"],
            "budgets": {
                "max_rounds": 2,
                "max_operator_calls": 3,
                "max_experiments": 1,
                "max_retries": 0,
                "max_wall_seconds": 60,
            },
            "resource_policy": {"allow_gpu": False, "allowed_operators": ["fake.baseline"]},
            "frozen_evaluator": {"path": "configs/harness/fake_evaluator.json", "sha256": "abc"},
            "interventions": [],
        }
    )

    assert goal.budgets == BudgetSpec(2, 3, 1, 0, 60)
    assert GoalSpec.from_json(goal.to_json()).to_dict() == goal.to_dict()
    assert goal.to_json() == json.dumps(
        goal.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    assert goal.sha256() == hashlib.sha256(goal.to_json().encode("utf-8")).hexdigest()

    evidence = EvidenceValue(
        metric="identity_preservation",
        status=EvidenceStatus.UNOBSERVED,
        value=None,
        required=True,
        reason="identity evaluator was not executed",
    )
    assert evidence.to_dict()["status"] == "UNOBSERVED"


def test_goal_spec_preserves_optional_initial_operator_parameters():
    goal = GoalSpec.from_dict(
        {
            "schema_version": "dataevolver.harness.goal.v1",
            "campaign_id": "t2i-dryrun",
            "objective": {"kind": "t2i_dryrun"},
            "initial_operator_id": "multimodal.t2i_dryrun",
            "initial_operator_parameters": {"user_request": "desk objects", "num_samples": 2},
            "route_candidates": ["t2i"],
            "budgets": {
                "max_rounds": 1,
                "max_operator_calls": 1,
                "max_experiments": 0,
                "max_retries": 0,
                "max_wall_seconds": 60,
            },
            "resource_policy": {"allow_gpu": False, "allowed_operators": ["multimodal.t2i_dryrun"]},
            "frozen_evaluator": {"path": "evaluator.json", "sha256": "a" * 64},
            "interventions": [],
        }
    )

    assert goal.initial_operator_parameters == {"user_request": "desk objects", "num_samples": 2}
    assert GoalSpec.from_json(goal.to_json()).initial_operator_parameters == goal.initial_operator_parameters


def test_budget_and_campaign_state_preserve_optional_planner_call_limit():
    budget = BudgetSpec.from_dict(
        {
            "max_rounds": 3,
            "max_operator_calls": 4,
            "max_experiments": 3,
            "max_retries": 1,
            "max_wall_seconds": 3600,
            "max_planner_calls": 4,
        }
    )
    state = CampaignState(campaign_id="canary", planner_calls=2)

    assert budget.max_planner_calls == 4
    assert CampaignState.from_dict(state.to_dict()).planner_calls == 2


def test_matched_trial_key_changes_when_provenance_changes():
    key = MatchedTrialKey(
        input_artifact_hash="input-a",
        parent_artifact_hash="parent-a",
        seed=17,
        operator_id="fake.baseline",
        operator_version="1",
        model_config_hash="model-a",
        preprocessing_hash="pre-a",
        evaluator_hash="eval-a",
        environment_contract_hash="env-a",
    )
    changed = MatchedTrialKey.from_dict({**key.to_dict(), "seed": 18})

    assert key.digest() != changed.digest()


def test_event_log_deduplicates_logical_commit_and_ignores_partial_tail(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path, campaign_id="demo")

    first = log.append("ACTION_COMMITTED", {"result": "ok"}, idempotency_key="call-1")
    duplicate = log.append("ACTION_COMMITTED", {"result": "different"}, idempotency_key="call-1")
    with path.open("ab") as handle:
        handle.write(b'{"partial":')

    assert duplicate.event_id == first.event_id
    assert [event.event_type for event in log.read()] == ["ACTION_COMMITTED"]


def test_state_store_replays_last_event_when_snapshot_is_missing(tmp_path):
    layout = CampaignLayout(tmp_path, "demo")
    store = StateStore(layout)
    started = datetime.now(timezone.utc).isoformat()
    initial = CampaignState(campaign_id="demo", started_at=started, updated_at=started)
    store.initialize_state(initial)
    planned = initial.with_updates(state=ControllerState.PLAN, updated_at=started)
    store.commit(planned, "STATE_TRANSITION", {"effect": "PLAN_CREATED"}, "transition-plan")
    layout.state_path.unlink()

    recovered = store.load_state()

    assert recovered.state is ControllerState.PLAN
    assert recovered.last_event_seq == 1


def test_campaign_lease_blocks_live_owner_and_reclaims_expired_owner(tmp_path):
    path = tmp_path / "campaign.lease.json"
    with CampaignLease(path, ttl_seconds=60):
        with pytest.raises(LeaseBusyError):
            with CampaignLease(path, ttl_seconds=60):
                pass
    path.write_text(
        json.dumps({"hostname": "dead-host", "pid": 999999, "token": "old", "expires_at": 0}),
        encoding="utf-8",
    )
    with CampaignLease(path, ttl_seconds=60) as lease:
        assert lease.token != "old"
