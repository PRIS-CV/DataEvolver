"""Controller ownership and recovery tests; fixture quality is not visual evidence."""
import json

import pytest

from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.models import (BudgetSpec, CapabilityStatus, ControllerState,
                                        Intervention, InterventionLevel, OperatorSpec, StageStatus)
from dataevolver.harness.operators import FakeOperator, InjectedOperatorCrash
from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
from dataevolver.testing.test_harness_controller import build_controller


def action(kind, *, tool="", patch="", parameters=None, evidence=None, layer="none", experience=""):
    return {"action": kind, "layer": layer, "hypothesis": "Inspect and test the quality gap.",
            "evidence_ids": evidence or ["baseline"],
            "observation": {"tool": tool, "parameters_json": "{}"},
            "patch": {"kind": patch, "parameters_json": json.dumps(parameters or {})},
            "experience_id": experience, "reason": "Fixture decision for a bounded scenario."}


class ScriptedBackend:
    backend_id, model_id = "fixture", "fixture-scripted-planner"

    def __init__(self, decisions):
        self.decisions, self.calls = decisions, []

    def complete_json(self, messages, *, schema):
        self.calls.append(json.loads(messages[-1]["content"]))
        return json.dumps(self.decisions[len(self.calls) - 1])


class FixtureDomain:
    def config(self):
        return {"domain": "fixture-quality", "evidence_kind": "fixture"}

    def context(self, state):
        observed = bool(state.data.get("rsi_observations"))
        return {"allowed_observations": ["inspect"], "allowed_patches": ["quality"],
                "available_evidence_ids": ["baseline"] + (["probe"] if observed else []),
                "compatible_experience_ids": []}

    def compile_observation(self, decision, state):
        return "fake.probe", {"question": "inspect actual evidence"}

    def compile_intervention(self, decision, state):
        parameters = json.loads(decision["patch"]["parameters_json"])
        if (decision["layer"] != "R" or not state.data.get("rsi_observations")
                or set(parameters) != {"quality_patch"}):
            raise ValueError("fixture intervention needs observed evidence and its registered R field")
        return Intervention("observed-quality-fix", "LOW_QUALITY", "fake.candidate", InterventionLevel.L1,
                            parameters=parameters, candidate_delta={"layer": "R", **parameters})


def make_runtime(tmp_path, decisions, *, crash_probe=False, candidate_metrics=None,
                 policy=None, budgets=None):
    old, source = build_controller(tmp_path, candidate_metrics=candidate_metrics)
    goal = old.goal.with_updates(campaign_id="runtime-rsi", planner={"kind": "runtime_rsi"},
                                 interventions=(), budgets=budgets or BudgetSpec(4, 6, 2, 0, 60, 5),
                                 resource_policy={"allow_gpu": False,
                                                  "allowed_operators": ["fake.baseline", "fake.candidate", "fake.probe"]})
    spec = OperatorSpec(operator_id="fake.probe", version="1", retry_safe=True,
                        source_path=str(source), code_revision="fixture-probe", model_config_hash="fixture")
    old.registry.register(FakeOperator(spec, status=StageStatus.SUCCESS, metrics={},
                                       hard_constraints={"artifact_valid": True},
                                       crash_after_effect=crash_probe))
    backend = ScriptedBackend(decisions)
    planner = RuntimeRSIPlanner(backend=backend, domain=FixtureDomain(),
                                trace_root=tmp_path / "campaigns/runtime-rsi/planner", policy=policy)
    c = EvolutionController.create(repo_root=tmp_path, campaigns_root=tmp_path / "campaigns", goal=goal,
                                    registry=old.registry, planner=planner,
                                    evaluator_contract=old.evaluator.contract, execution_paths=(source,),
                                    idempotency_key="runtime-fixture")
    return c, backend


def decisions(terminal="COMPLETE"):
    return [action("OBSERVE", tool="inspect"),
            action("RUN_EXPERIMENT", patch="quality", layer="R", evidence=["probe"],
                   parameters={"quality_patch": "improve"}), action(terminal)]


def test_runtime_probe_trial_adoption_and_post_result_decision_use_one_controller(tmp_path):
    c, backend = make_runtime(tmp_path, decisions())
    final = c.run()
    assert final.state is ControllerState.COMPLETE
    assert final.planner_calls == 3 and final.operator_calls == 3
    assert len(backend.calls) == 3
    assert not backend.calls[0]["observations"] and backend.calls[1]["observations"]
    assert backend.calls[2]["history"][0]["decision"]["decision"] == "ADOPT_CANDIDATE"
    capabilities = c.capabilities.list_capabilities()
    assert len(capabilities) == 1 and capabilities[0].status is CapabilityStatus.CANDIDATE
    assert final.data["rsi_history"][0]["independent_source_validation"] is False
    events = c.store.events.read()
    names = [e.event_type for e in events]
    assert names.index("RSI_OBSERVATION_RECORDED") < names.index("INTERVENTION_PROPOSED")
    before = c.layout.events_path.read_bytes()
    c.run()
    assert before == c.layout.events_path.read_bytes() and len(backend.calls) == 3


def test_observation_can_name_recipe_target_without_applying_a_patch(tmp_path):
    sequence = decisions(); sequence[0]['layer'] = 'R'
    c, _ = make_runtime(tmp_path, sequence)
    assert c.run().state is ControllerState.COMPLETE
    assert c.status().experiments_used == 1


def test_runtime_observation_recovers_after_effect_without_another_proposal(tmp_path):
    c, backend = make_runtime(tmp_path, decisions(), crash_probe=True)
    with pytest.raises(InjectedOperatorCrash):
        c.run()
    assert len(backend.calls) == 1
    result = c.run()
    assert result.state is ControllerState.COMPLETE
    assert len(backend.calls) == 3
    events = c.store.events.read()
    observed_calls = [e for e in events if e.event_type == "ACTION_COMMITTED"
                      and e.payload.get("call_id", "").endswith(":observe")]
    assert len(observed_calls) == 1


def test_agent_complete_cannot_override_failed_frozen_goal(tmp_path):
    c, _ = make_runtime(tmp_path, [action("COMPLETE")])
    result = c.run()
    assert result.state is ControllerState.ABSTAIN and result.terminal_reason == "GOAL_NOT_VERIFIED"
    assert not c.capabilities.list_capabilities()


@pytest.mark.parametrize("decision", [
    action("OBSERVE", tool="shell"),
    action("OBSERVE", tool="inspect", experience="unverified-candidate"),
    action("RUN_EXPERIMENT", patch="change_evaluator", layer="R"),
    action("RUN_EXPERIMENT", patch="quality", layer="R", evidence=["invented-image"]),
])
def test_unregistered_operations_and_unverified_evidence_fail_closed(tmp_path, decision):
    c, backend = make_runtime(tmp_path, [decision])
    result = c.run()
    assert result.state is ControllerState.ABSTAIN and result.terminal_reason == "RSI_PLANNER_FAILED"
    assert result.operator_calls == 1 and len(backend.calls) == 1


def test_domain_compiler_rejects_threshold_change_disguised_as_recipe(tmp_path):
    seq = decisions()
    seq[1]["patch"]["parameters_json"] = '{"quality_patch":"improve","threshold":0}'
    c, _ = make_runtime(tmp_path, seq)
    result = c.run()
    assert result.state is ControllerState.ABSTAIN and result.terminal_reason == "INVALID_RSI_INTERVENTION"
    assert result.operator_calls == 2 and not c.capabilities.list_capabilities()


def test_failed_trial_rolls_back_and_requests_a_new_runtime_decision(tmp_path):
    c, backend = make_runtime(tmp_path, decisions("ABSTAIN"),
                              candidate_metrics={"quality": .3, "safety": 1.0})
    result = c.run()
    assert result.state is ControllerState.ABSTAIN
    assert backend.calls[2]["history"][0]["decision"]["decision"] == "ROLLBACK"
    assert not c.capabilities.list_capabilities()


def test_recipe_patch_cannot_be_relabelled_as_method_improvement(tmp_path):
    seq = decisions()
    seq[1]["layer"] = "M"
    c, _ = make_runtime(tmp_path, seq)
    result = c.run()
    assert result.terminal_reason == "INVALID_RSI_INTERVENTION"


def test_zero_policy_limit_stops_before_planning_and_replay_has_no_effect(tmp_path):
    from dataevolver.harness.feedback import PolicySpec
    policy = PolicySpec('bounded', '1', 'evolution', {'candidate_limit': 0})
    c, backend = make_runtime(tmp_path, decisions(), policy=policy)
    state = c.run()
    assert state.terminal_reason == 'POLICY_CANDIDATE_LIMIT'
    assert state.experiments_used == 0 and state.operator_calls == 1
    assert backend.calls == []
    events = c.layout.events_path.read_bytes()
    assert c.run() == state
    assert c.layout.events_path.read_bytes() == events and backend.calls == []


def test_no_improvement_stops_on_committed_rollback_without_more_calls(tmp_path):
    from dataevolver.harness.feedback import PolicySpec
    policy = PolicySpec('bounded', '1', 'evolution', {'stop_rule': 'no_improvement'})
    c, backend = make_runtime(tmp_path, decisions('ABSTAIN'), policy=policy,
                              candidate_metrics={'quality': .3, 'safety': 1.})
    state = c.run()
    assert state.terminal_reason == 'POLICY_NO_IMPROVEMENT'
    assert state.data['rsi_history'][-1]['decision']['decision'] == 'ROLLBACK'
    assert state.experiments_used == 1 and len(backend.calls) == 2
    assert state.operator_calls == 3


def test_policy_limit_preserves_verified_goal_completion_after_adoption(tmp_path):
    from dataevolver.harness.feedback import PolicySpec
    policy = PolicySpec('bounded', '1', 'evolution',
                        {'candidate_limit': 1, 'stop_rule': 'no_improvement'})
    c, backend = make_runtime(tmp_path, decisions(), policy=policy)
    state = c.run()
    assert state.state is ControllerState.COMPLETE and state.experiments_used == 1
    assert len(backend.calls) == 3
    assert backend.calls[-1]['allowed_actions'] == ['COMPLETE', 'ABSTAIN']
    assert state.data['rsi_history'][-1]['decision']['decision'] == 'ADOPT_CANDIDATE'


def test_observation_only_policy_keeps_post_rollback_replanning(tmp_path):
    from dataevolver.harness.feedback import PolicySpec
    policy = PolicySpec('ordered', '1', 'evolution', {'observation_order': ['inspect']})
    c, backend = make_runtime(tmp_path, decisions('ABSTAIN'), policy=policy,
                              candidate_metrics={'quality': .3, 'safety': 1.})
    state = c.run()
    assert state.terminal_reason == 'RSI_AGENT_ABSTAINED' and len(backend.calls) == 3
    assert 'policy_stop_reason' not in backend.calls[-1]


def test_policy_cannot_expand_the_original_experiment_budget(tmp_path):
    from dataevolver.harness.feedback import PolicySpec
    policy = PolicySpec('bounded', '1', 'evolution', {'candidate_limit': 99})
    c, backend = make_runtime(tmp_path, decisions(), policy=policy,
                              budgets=BudgetSpec(4, 6, 0, 0, 60, 5))
    state = c.run()
    assert state.terminal_reason == 'BUDGET_EXHAUSTED'
    assert c.goal.budgets.max_experiments == 0 and state.experiments_used == 0
    assert len(backend.calls) == 2


def test_persisted_nonterminal_decision_cannot_bypass_spent_policy_limit(tmp_path):
    from dataevolver.harness.feedback import PolicySpec
    policy = PolicySpec('bounded', '1', 'evolution', {'candidate_limit': 1})
    c, backend = make_runtime(tmp_path, decisions(), policy=policy)
    for _ in range(50):
        state = c.status()
        if state.state is ControllerState.PROPOSE and state.data.get('rsi_history'):
            break
        c.tick()
    else:
        pytest.fail('did not reach the committed post-comparison state')
    assert state.experiments_used == 1 and len(backend.calls) == 2
    pending = action('OBSERVE', tool='inspect')
    c._commit(state, ControllerState.PROPOSE, 'TEST_PENDING_DECISION', 'FIXTURE',
              data={**state.data, 'rsi_pending_decision': pending})
    # Reopen the serialized state; neither an observation nor a fresh proposal may run.
    reopened = EvolutionController.open(repo_root=tmp_path, campaigns_root=tmp_path/'campaigns',
        goal=c.goal, registry=c.registry, planner=c.planner,
        evaluator_contract=c.evaluator.contract, execution_paths=(tmp_path/'operator.py',))
    state = reopened.run()
    assert state.terminal_reason == 'POLICY_CANDIDATE_LIMIT'
    assert state.experiments_used == 1 and state.operator_calls == 3
    assert len(backend.calls) == 2
