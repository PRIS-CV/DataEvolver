"""Scope-bound operator versions, committed adoption and conservative revocation.

The catalog stores artifacts, not Python registrations or a second controller.
An adoption entry alone grants nothing: resolution also requires the original
Controller's frozen, committed effect and Evaluator acceptance. Revocation is
fail closed even when its producer is interrupted before committing a receipt.
"""
import copy
import json
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from .models import CampaignState, EvidenceStatus, OperatorCall, OperatorReceipt, StageResult
from .state_store import CampaignLease, atomic_write_json


SYNTHETIC_SCOPE = "synthetic_authority_contract_independent"
EMPTY_HEAD = "0" * 64


def engineering_origin_matches(origin, current_identity, historical_reader_sha256, report_sha256, package_sha256):
    """Keep the reader identity of the independent stage's original frozen core."""
    if set(origin) != {"authority", "report_sha256", "package_sha256"}:
        return False
    original = copy.deepcopy(origin["authority"])
    if "env_source_commit" in current_identity and "env_source_commit" not in original:
        # Older readers constrained env.sh to an execution-root parent. Derive
        # its commit from that already-bound source root, never caller text.
        env_source = str(Path(original["descriptor"]["env_path"]).parent / "source")
        if env_source not in original["source_commits"]:
            return False
        original["env_source_commit"] = original["source_commits"][env_source]
    return (original.get("reader_sha256") == historical_reader_sha256
            and {k: v for k, v in original.items() if k != "reader_sha256"}
            == {k: v for k, v in current_identity.items() if k != "reader_sha256"}
            and origin["report_sha256"] == report_sha256 and origin["package_sha256"] == package_sha256)


class OperatorVersionStore:
    """Append-only, hash-linked extension of capability storage for code versions."""

    def __init__(self, root):
        self.root = Path(root).absolute()

    def _safe(self):
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError("operator catalog cannot contain symlinks")

    def lease(self):
        self._safe()
        return CampaignLease(self.root / ".versions.lease", ttl_seconds=300)

    def events(self):
        self._safe()
        result = []
        head = EMPTY_HEAD
        for index, path in enumerate(sorted(self.root.glob("*.json")), 1):
            if path.is_symlink() or path.name != f"{index:08d}.json":
                raise ValueError("operator version history has a gap or unexpected file")
            event = json.loads(path.read_text())
            digest = event.get("sha256")
            body = {k: v for k, v in event.items() if k != "sha256"}
            if (body.get("schema_version") != "dataevolver.operator_version_event.v1"
                    or body.get("sequence") != index or body.get("previous_sha256") != head
                    or digest != payload_sha256(body) or body.get("action") not in {"ADOPTED", "REVOKED"}):
                raise ValueError("operator version history changed")
            snapshot = body["snapshot"]
            if body["snapshot_sha256"] != payload_sha256(snapshot):
                raise ValueError("operator adoption snapshot changed")
            key = (snapshot["operator_id"], snapshot["version"])
            previous = [x for x in result if (x["snapshot"]["operator_id"], x["snapshot"]["version"]) == key]
            if body["action"] == "ADOPTED":
                if previous or body["reason"] is not None:
                    raise ValueError("operator versions are immutable and cannot be readopted")
            elif (len(previous) != 1 or previous[0]["action"] != "ADOPTED"
                    or previous[0]["snapshot_sha256"] != body["snapshot_sha256"]
                    or not isinstance(body["reason"], str) or not body["reason"].strip()):
                raise ValueError("revocation must bind an active exact version and reason")
            result.append(event)
            head = digest
        return result

    def head(self):
        rows = self.events()
        return rows[-1]["sha256"] if rows else EMPTY_HEAD

    def append(self, *, action, snapshot, owner, expected_head, reason=None):
        """Caller holds lease; registered mutation operators are the write path."""
        rows = self.events()
        for row in rows:
            if row["owner"] == owner:
                if (row["action"], row["snapshot"], row["reason"]) != (action, snapshot, reason):
                    raise ValueError("a committed catalog owner cannot change its effect")
                return row
        if self.head() != expected_head:
            raise ValueError("operator catalog changed since the mutation was frozen")
        key = (snapshot["operator_id"], snapshot["version"])
        previous = [x for x in rows if (x["snapshot"]["operator_id"], x["snapshot"]["version"]) == key]
        if action == "ADOPTED" and (previous or reason is not None):
            raise ValueError("operator version already exists")
        if action == "REVOKED" and (len(previous) != 1 or previous[0]["action"] != "ADOPTED"
                or previous[0]["snapshot"] != snapshot or not isinstance(reason, str) or not reason.strip()):
            raise ValueError("only an exact active version can be revoked with a reason")
        if action not in {"ADOPTED", "REVOKED"}:
            raise ValueError("unknown operator lifecycle action")
        body = {"schema_version": "dataevolver.operator_version_event.v1", "sequence": len(rows) + 1,
                "previous_sha256": expected_head, "action": action, "snapshot": snapshot,
                "snapshot_sha256": payload_sha256(snapshot), "owner": owner, "reason": reason}
        event = {**body, "sha256": payload_sha256(body)}
        path = self.root / f"{len(rows) + 1:08d}.json"
        if path.exists():
            raise ValueError("operator event path already exists")
        atomic_write_json(path, event)
        self.events()
        return event

    def included(self, event):
        rows = self.events()
        if event not in rows:
            raise ValueError("operator lifecycle effect is absent from its catalog")
        return event

    def active(self, event, *, scope, head, qualification_authority=None):
        self.included(event)
        if (event["action"] != "ADOPTED" or event["snapshot"]["scope"] != scope
                or self.head() != head):
            raise ValueError("operator scope or catalog snapshot is incompatible")
        if scope != SYNTHETIC_SCOPE:
            from .catalog_candidate import CatalogQualificationAuthority, SCOPE
            from .selection_candidate import SelectionQualificationAuthority, SCOPE as SELECTION_SCOPE
            from .policy_qualification import PolicyQualificationAuthority, SCOPE as POLICY_SCOPE
            from .repair_qualification import RepairQualificationAuthority, SCOPE as REPAIR_SCOPE
            expected = {SCOPE: CatalogQualificationAuthority, SELECTION_SCOPE: SelectionQualificationAuthority,
                        POLICY_SCOPE: PolicyQualificationAuthority, REPAIR_SCOPE: RepairQualificationAuthority}
            if scope not in expected or not isinstance(qualification_authority, expected[scope]):
                raise ValueError('operator scope requires a matching qualification authority')
            qualification_authority.verify(event['snapshot'])
        if any(x["snapshot_sha256"] == event["snapshot_sha256"] and x["action"] == "REVOKED"
               for x in self.events()):
            raise ValueError("operator version is revoked")
        return event["snapshot"]


def qualification_snapshot(*, package, store, engineering, independent, runtime):
    """Read actual host-bound qualifications; summaries supplied as JSON cannot adopt."""
    from .contract_execution_authority import FrozenContractExecutionAuthority
    from .models import OperatorSpec
    from .operator_contracts import OperatorContract
    if not all(isinstance(x, FrozenContractExecutionAuthority) for x in (engineering, independent)):
        raise ValueError("adoption requires host-bound historical Controller readers")
    eng, ind = engineering.read(), independent.read()
    package_hash = package.compute_hash()
    if (eng.get("engineering_evaluator_accepted") is not True
            or ind.get("independent_evaluator_accepted") is not True
            or eng.get("pure_data_engineering_qualification") != "PASSED"
            or ind.get("pure_data_independent_qualification") != "PASSED"
            or eng.get("scope") != "synthetic_authority_contract_engineering"
            or ind.get("scope") != SYNTHETIC_SCOPE
            or any(x.get("production_adoption_allowed") is not False for x in (eng, ind))
            or any(x["origin"]["package_sha256"] != package_hash for x in (eng, ind))
            or not engineering_origin_matches(ind["engineering_origin"], engineering.identity(),
                file_sha256(independent.module_root/"src/dataevolver/harness/contract_execution_authority.py"),
                eng["report_sha256"], package_hash)
            or len(ind["fixture_family_ids"]) < 2
            or set(ind["fixture_family_ids"]) & set(eng["fixture_family_ids"])
            or set(ind["input_sha256s"]) & set(eng["input_sha256s"])
            or runtime != ind["runtime"]):
        raise ValueError("same-package engineering and independent qualification must authorize this exact runtime")
    for ref in package.files:
        store.resolve(ArtifactRef.from_dict(ref))
    read = lambda ref: json.loads(store.resolve(ArtifactRef.from_dict(ref)).read_text())
    spec = OperatorSpec.from_dict(read(package.operator_spec))
    contract = OperatorContract.from_dict(read(package.operator_contract)).bound_to(spec)
    return {"schema_version": "dataevolver.scope_bound_operator_adoption.v1",
        "operator_id": spec.operator_id, "version": spec.version, "scope": SYNTHETIC_SCOPE,
        "package_sha256": package_hash, "implementation_sha256": package.implementation["sha256"],
        "operator_spec_sha256": spec.sha256(), "interface_sha256": contract.compute_hash(),
        "runtime": runtime, "engineering": {"authority": engineering.identity(), "origin": eng["origin"],
            "report_sha256": eng["report_sha256"]}, "independent": {"authority": independent.identity(),
            "origin": ind["origin"], "report_sha256": ind["report_sha256"]},
        "lifecycle_total": ind["lifecycle_total"], "resource_deadline_epoch": ind["resource_deadline_epoch"],
        "qualification_inputs": sorted(eng["input_sha256s"] + ind["input_sha256s"]),
        "qualification_families": sorted(eng["fixture_family_ids"] + ind["fixture_family_ids"]),
        "historical_engineering_origin": ind["engineering_origin"],
        "production_adoption_allowed": False, "model_authority": False, "image_authority": False}


class CommittedOperatorAdoptionAuthority:
    """Resolve only the exact adopted version accepted by its frozen Controller."""
    def __init__(self, controller, operator, call_id):
        self.controller, self.operator, self.call_id = controller, operator, call_id
        self._identity = {"campaign_id": controller.goal.campaign_id, "call_id": call_id,
            "frozen_sha256": file_sha256(controller.layout.frozen_contract_path),
            "operator_spec_sha256": operator.spec.sha256(), "catalog": str(operator.catalog.root)}

    def identity(self):
        if file_sha256(self.controller.layout.frozen_contract_path) != self._identity["frozen_sha256"]:
            raise ValueError("adoption frozen contract changed")
        return copy.deepcopy(self._identity)

    def read(self):
        c, op = self.controller, self.operator
        self.identity()
        state = CampaignState.from_dict(json.loads(c.layout.state_path.read_text()))
        if (not c._frozen()[0] or c.replay().to_dict() != state.to_dict()
                or state.state.value != "COMPLETE" or state.data.get("pending_call")):
            raise ValueError("adoption Controller must be complete, frozen and replayable")
        planned = [e for e in c.store.events.read() if e.event_type == "ACTION_PLANNED" and e.payload.get("call_id") == self.call_id]
        committed = [e for e in c.store.events.read() if e.event_type == "ACTION_COMMITTED" and e.payload.get("call_id") == self.call_id]
        if len(planned) != 1 or len(committed) != 1 or planned[0].sequence >= committed[0].sequence:
            raise ValueError("exact committed adoption is required")
        call = OperatorCall.from_dict(planned[0].payload["operator_call"])
        receipt = OperatorReceipt.from_dict(committed[0].payload["receipt"])
        verification = op.verify_effect(call, receipt)
        if (call.operator_id != "evolution.adopt_operator_version"
                or receipt.status.value != "SUCCESS" or receipt.return_code != 0
                or c.registry.get(call.operator_id).spec.sha256() != self._identity["operator_spec_sha256"]
                or verification.status is not EvidenceStatus.VERIFIED
                or verification.artifact_hashes != committed[0].payload["effect_verification"]["artifact_hashes"]
                or not c._goal_satisfied(c.evaluator.evaluate(StageResult.from_dict(receipt.stage_result),
                    effect_status=EvidenceStatus.VERIFIED))):
            raise ValueError("adoption effects or frozen Evaluator acceptance changed")
        event = json.loads(op._path(call).read_text())["event"]
        if event["action"] != "ADOPTED":
            raise ValueError("revocation cannot authorize adoption")
        return op.catalog.included(event)

    def usage(self):
        self.read()
        rows = [e for e in self.controller.store.events.read() if e.event_type == "ACTION_PLANNED"
                and e.payload.get("call_id") == self.call_id]
        call = OperatorCall.from_dict(rows[0].payload["operator_call"])
        return json.loads(self.operator._path(call).read_text())["lifecycle_total"]
