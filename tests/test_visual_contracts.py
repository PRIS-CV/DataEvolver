import hashlib
import json
from dataclasses import replace

import pytest

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore
from dataevolver.harness.task_contracts import DatasetIntent, NodeSpec, NodeSpecGraph, PROTECTED_GATES, QualityPolicy, adapt_raw_image_pair_contracts


def intent():
    return DatasetIntent("example", "generate", "image_caption", "caption.v1", "Generate a bounded example.")


def policy():
    return QualityPolicy("quality", tuple(sorted(PROTECTED_GATES)), ("model.invoke",), max_model_calls=2)


def node(name="first", dependencies=()):
    return NodeSpec(name, "model.invoke", "1", "planner", dependencies)


def test_general_contract_round_trip_and_unknown_fields():
    original = intent()
    assert DatasetIntent.from_dict(original.to_dict()).compute_hash() == original.compute_hash()
    with pytest.raises(ValueError, match="unknown"):
        DatasetIntent.from_dict({**original.to_dict(), "shell": "anything"})
    with pytest.raises(ValueError, match="schema"):
        DatasetIntent.from_dict({**original.to_dict(), "schema_version": "future"})
    with pytest.raises(ValueError):
        replace(original, parameters={"threshold": float("nan")})


@pytest.mark.parametrize("mode", ["curate", "hybrid"])
def test_source_dependent_modes_require_hash_bound_inputs(mode):
    with pytest.raises(ValueError, match="source"):
        replace(intent(), mode=mode)
    source = ArtifactRef("source", "catalog.json", "a" * 64)
    bounded = replace(intent(), mode=mode, sources=(source,))
    assert DatasetIntent.from_dict(bounded.to_dict()).sources == (source,)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "a\\b", "C:drive", "./x", "."])
def test_artifact_ref_rejects_unsafe_paths(path):
    with pytest.raises(ValueError):
        ArtifactRef("source", path, "a" * 64)


def test_artifact_store_checks_content_and_symlink_escape(tmp_path):
    store_root = tmp_path / "source"
    store_root.mkdir()
    content = store_root / "asset.json"
    content.write_text("{}")
    ref = ArtifactRef("source", "asset.json", hashlib.sha256(content.read_bytes()).hexdigest())
    store = ArtifactStore({"source": store_root})
    assert store.resolve(ref) == content
    content.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        store.resolve(ref)
    outside = tmp_path / "outside"
    outside.write_text("{}")
    (store_root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        store.resolve(ArtifactRef("source", "link", hashlib.sha256(outside.read_bytes()).hexdigest()))


def test_graph_dependency_order_cycles_and_policy_permissions():
    graph = NodeSpecGraph((node("second", ("first",)), node()))
    assert [x.node_id for x in graph.ordered()] == ["first", "second"]
    assert NodeSpecGraph.from_dict(graph.to_dict()).compute_hash() == graph.compute_hash()
    graph.validate_policy(policy(), {"planner"})
    with pytest.raises(ValueError, match="cycle"):
        NodeSpecGraph((node("first", ("second",)), node("second", ("first",))))
    with pytest.raises(ValueError, match="unique"):
        NodeSpecGraph((node(), node()))
    with pytest.raises(ValueError, match="missing"):
        NodeSpecGraph((node("first", ("missing",)),))
    with pytest.raises(ValueError, match="budget"):
        graph.validate_policy(replace(policy(), max_model_calls=1), {"planner"})
    with pytest.raises(ValueError, match="approved"):
        graph.validate_policy(policy(), {"reviewer"})


def test_protected_gates_cannot_be_weakened_by_a_plan():
    with pytest.raises(ValueError, match="protected"):
        replace(policy(), required_gates=("integrity",))
    with pytest.raises(ValueError):
        replace(policy(), max_model_calls=True)


def test_local_raw_image_pair_bridge_preserves_intent_without_private_paths():
    legacy_intent = {"schema_version": "raw_image_pairs.dataset_intent.v1", "dataset_id": "editing",
                     "source_root": "/private/source", "training_goal": "Improve material edits",
                     "task_types": ["attribute_material"], "preserved_content": ["identity"],
                     "target_width": 1024, "budget": {"max_rounds": 3}}
    legacy_policy = {"schema_version": "raw_image_pairs.quality_policy.v1", "min_width": 512,
                     "promotion_min_rate_delta": 0.02}
    source = ArtifactRef("catalog", "input.jsonl", "b" * 64)
    adapted, quality = adapt_raw_image_pair_contracts(legacy_intent, legacy_policy, source=source)
    assert adapted.mode == "hybrid" and adapted.sources == (source,)
    assert adapted.parameters["preserved_content"] == ["identity"]
    assert "/private/" not in json.dumps(adapted.to_dict())
    assert "promotion_min_rate_delta" not in quality.task_criteria
    with pytest.raises(ValueError, match="warning"):
        adapt_raw_image_pair_contracts({**legacy_intent, "license_policy": "warn"}, legacy_policy, source=source)
