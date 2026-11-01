"""Read committed engineering feedback before a bounded candidate revision.

Feedback is development evidence, not independent qualification or authority to
activate code. Historical readers never run a Controller or candidate program.
"""
import json
import os
import subprocess
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.harness.models import CampaignState, EvidenceStatus, OperatorCall, OperatorReceipt


class ControllerContractFeedbackAuthority:
    def __init__(self, controller, operator, call_id):
        from .adapters.operator_contract_qualification import CandidateContractQualificationOperator
        if not isinstance(operator, CandidateContractQualificationOperator):
            raise ValueError("revision feedback requires the registered contract qualifier")
        self.controller, self.operator, self.call_id = controller, operator, call_id

    def identity(self):
        c = self.controller
        return {"campaign_id": c.goal.campaign_id, "call_id": self.call_id,
                "frozen_sha256": file_sha256(c.layout.frozen_contract_path),
                "operator_spec_sha256": self.operator.spec.sha256(),
                "protocol_sha256": payload_sha256(self.operator.protocol),
                "package_sha256": self.operator.package.compute_hash()}

    def read(self):
        c, op = self.controller, self.operator
        state = CampaignState.from_dict(json.loads(c.layout.state_path.read_text()))
        if (not c._frozen()[0] or c.replay().to_dict() != state.to_dict()
                or state.state.value not in {"ABSTAIN", "COMPLETE"} or state.data.get("pending_call")):
            raise ValueError("qualification feedback must be frozen, replayable and terminal")
        events = c.store.events.read()
        planned = [e for e in events if e.event_type == "ACTION_PLANNED" and e.payload.get("call_id") == self.call_id]
        committed = [e for e in events if e.event_type == "ACTION_COMMITTED" and e.payload.get("call_id") == self.call_id]
        if len(planned) != 1 or len(committed) != 1 or planned[0].sequence >= committed[0].sequence:
            raise ValueError("feedback needs an exact committed qualification invocation")
        call = OperatorCall.from_dict(planned[0].payload["operator_call"])
        receipt = OperatorReceipt.from_dict(committed[0].payload["receipt"])
        if (call.operator_id != op.spec.operator_id
                or c.registry.get(call.operator_id).spec.sha256() != op.spec.sha256()
                or c.goal.objective.get("candidate_package_sha256") != op.package.compute_hash()
                or c.goal.objective.get("contract_qualification_protocol_sha256") != payload_sha256(op.protocol)):
            raise ValueError("feedback protocol and package were not prebound by the Controller")
        verification = op.verify_effect(call, receipt)
        if (verification.status is not EvidenceStatus.VERIFIED
                or verification.artifact_hashes != committed[0].payload["effect_verification"]["artifact_hashes"]):
            raise ValueError("committed qualification effects no longer verify")
        report = json.loads(op._path(call).read_text())
        if report["qualification_scope"] != "synthetic_authority_contract_engineering":
            raise ValueError("independent/heldout evidence is unavailable to the revision builder")
        cases = []
        for index, case in enumerate(op.protocol["cases"]):
            execution = json.loads((op._path(call).parent/"executions"/(str(index)+".json")).read_text())
            _, expected, payload = op._case_data(case)
            cases.append({"case_id": case["case_id"], "kind": case["kind"],
                "checks": report["contract_checks"][index],
                # Passed vectors stay sealed here; a failed engineering vector
                # is a declared regression, never an independent test result.
                **({"input": payload, "expected": expected.to_dict(),
                    "actual": execution["output"], "status": execution["status"]}
                   if not report["contract_checks"][index]["passed"] else {})})
        return {"schema_version": "dataevolver.candidate_revision_feedback.v1",
                "scope": "synthetic_authority_contract_engineering", "origin": self.identity(),
                "package": op.package.to_dict(), "cases": cases,
                "report_sha256": file_sha256(op._path(call)),
                "pure_data_engineering_qualification": report["pure_data_engineering_qualification"],
                "lifecycle_total": report["lifecycle_total"],
                "resource_deadline_epoch": op.budget.resource_deadline_epoch,
                "runtime": op.runtime, "source_bytes_unchanged": True,
                "production_qualification": "NOT_ASSESSED", "production_adoption_allowed": False}


class FrozenContractFeedbackAuthority:
    """Host-selected historical config reader with no execute/run path."""
    def __init__(self, *, qualification_config, locations, python):
        self.config = Path(qualification_config).resolve()
        self.locations = {k: str(Path(v).resolve()) for k, v in locations.items()}
        self.python = Path(python).absolute()
        self._identity = self._current_identity()

    def _current_identity(self):
        goal = json.loads((self.config/"protected/qualification-goal.json").read_text())
        campaign = self.config/"output/campaigns"/goal["campaign_id"]
        return {"campaign_id": goal["campaign_id"], "config": str(self.config),
                "frozen_sha256": file_sha256(campaign/"frozen_contract.json"),
                "protocol_sha256": file_sha256(self.config/"protected/contract-qualification-protocol.json"),
                "source_commit": (self.config/"source-commit").read_text().strip(),
                "python": str(self.python), "python_sha256": file_sha256(self.python),
                "reader_sha256": file_sha256(Path(__file__)), "locations": self.locations}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError("historical qualification identity changed")
        return dict(self._identity)

    def read(self):
        before = self.identity()
        env = {**os.environ, "PYTHONPATH": str(self.config/"source/src"),
               "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0", "CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run([str(self.python), str(Path(__file__).resolve()), "--read-frozen",
            str(self.config), json.dumps(self.locations), str(self.python)],
            capture_output=True, text=True, timeout=180, env=env)
        if done.returncode:
            raise ValueError("historical qualification read failed: "+done.stderr[-2000:])
        result = json.loads(done.stdout)
        if (result["origin"]["campaign_id"] != before["campaign_id"]
                or result["origin"]["frozen_sha256"] != before["frozen_sha256"]):
            raise ValueError("historical reader returned a different qualification")
        self.identity()
        return result


def _read_frozen(config, locations, python):
    # PYTHONPATH selects only this trusted historical snapshot. Candidate
    # source is data resolved by ArtifactStore and is never imported here.
    from .operator_development import read_development_plan, CandidatePackage, FrozenSourceRequestAuthority
    from .frozen_candidate_request import FrozenCandidateReadinessRequestAuthority
    from .candidate_sandbox import TrustedContextRunner, SandboxLimits
    from .adapters.operator_contract_qualification import CandidateContractQualificationOperator
    from .controller import EvolutionController
    from .models import GoalSpec
    from .operators import OperatorRegistry
    from .planner import DeterministicPlanner
    from .evaluator import EvaluatorContract
    config = Path(config)
    loc = {k: Path(v) for k, v in locations.items()}
    plan = read_development_plan(json.loads((config/"protected/original-development-plan.json").read_text()))
    manifest = next((loc["recovery_config"]/"output").glob("campaigns/*/rounds/*/execute/candidate-package.json"))
    package = CandidatePackage.from_dict(json.loads(manifest.read_text()))
    store = ArtifactStore({"request": loc["request_origin"], "protocol": config/"protected",
        "candidate": loc["recovery_config"]/"candidate", "qualification_request": loc["readiness_config"]})
    protocol = json.loads((config/"protected/contract-qualification-protocol.json").read_text())
    op = CandidateContractQualificationOperator(plan=plan, package=package, store=store,
        request_authority=FrozenSourceRequestAuthority(config_root=loc["request_origin"], python=python),
        runner=TrustedContextRunner(python=python, limits=SandboxLimits(output_bytes=4096)),
        check_protocol=protocol,
        qualification_request_authority=FrozenCandidateReadinessRequestAuthority(**loc, python=python),
        output_root=config/"output")
    registry = OperatorRegistry(); registry.register(op)
    goal = GoalSpec.from_dict(json.loads((config/"protected/qualification-goal.json").read_text()))
    c = EvolutionController.open(repo_root=config/"source", campaigns_root=config/"output/campaigns",
        goal=goal, registry=registry, planner=DeterministicPlanner(()),
        evaluator_contract=EvaluatorContract.from_path(config/"evaluator.json"),
        schema_paths=tuple((config/"protected").rglob("*.json")),
        execution_paths=tuple((config/"source/src/dataevolver").rglob("*.py")))
    call = next(e.payload["call_id"] for e in c.store.events.read() if e.event_type == "ACTION_COMMITTED")
    return ControllerContractFeedbackAuthority(c, op, call).read()


if __name__ == "__main__":
    import sys
    # The module loaded as __main__ uses absolute historical imports below.
    __package__ = "dataevolver.harness"
    if len(sys.argv) != 5 or sys.argv[1] != "--read-frozen": raise SystemExit(2)
    print(json.dumps(_read_frozen(sys.argv[2], json.loads(sys.argv[3]), sys.argv[4])))
