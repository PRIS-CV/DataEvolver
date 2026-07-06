from dataevolver.workflows.multimodal.t2i_constraints import (
    constraint_improvement,
    get_prompt_bank,
    has_failure_then_improvement,
    normalize_constraint_review,
    prompt_bank_counts,
    select_prompt_specs,
)


def test_prompt_bank_has_required_three_tiers():
    counts = prompt_bank_counts()
    assert counts == {
        "simple_object": 10,
        "compositional_relation": 20,
        "hard_constraints": 20,
    }
    assert len(get_prompt_bank()) == 50


def test_hard_prompts_cover_checkable_constraint_types():
    required = {
        "count",
        "spatial_lr",
        "spatial_depth",
        "occlusion",
        "material",
        "text_ban",
        "transparent_reflective",
        "part_detail",
        "same_color_distractor",
    }
    for spec in select_prompt_specs(tier="hard_constraints"):
        kinds = {row["type"] for row in spec["constraints"]}
        assert required <= kinds


def test_constraint_review_normalization_fails_missing_items_as_uncertain():
    constraints = [
        {"constraint_id": "c_count", "type": "count", "description": "Exactly three bottles."},
        {"constraint_id": "c_no_text", "type": "text_ban", "description": "No text."},
    ]
    review = normalize_constraint_review(
        {
            "constraint_checklist": [
                {
                    "constraint_id": "c_count",
                    "type": "count",
                    "description": "Exactly three bottles.",
                    "status": "fail",
                    "evidence": "Only two are visible.",
                    "confidence": "high",
                }
            ]
        },
        constraints,
    )
    assert review["failed_constraints"] == ["c_count"]
    assert review["uncertain_constraints"] == ["c_no_text"]
    assert review["vlm_route"] == "needs_fix"
    assert review["constraint_status_counts"] == {"pass": 0, "fail": 1, "uncertain": 1, "total": 2}


def test_failure_then_improvement_requires_specific_constraint_repair():
    round0 = normalize_constraint_review(
        {
            "constraint_checklist": [
                {"constraint_id": "c_count", "type": "count", "description": "Exactly three.", "status": "fail"},
                {"constraint_id": "c_no_text", "type": "text_ban", "description": "No text.", "status": "pass"},
            ]
        },
        [
            {"constraint_id": "c_count", "type": "count", "description": "Exactly three."},
            {"constraint_id": "c_no_text", "type": "text_ban", "description": "No text."},
        ],
    )
    round1 = normalize_constraint_review(
        {
            "constraint_checklist": [
                {"constraint_id": "c_count", "type": "count", "description": "Exactly three.", "status": "pass"},
                {"constraint_id": "c_no_text", "type": "text_ban", "description": "No text.", "status": "pass"},
            ]
        },
        [
            {"constraint_id": "c_count", "type": "count", "description": "Exactly three."},
            {"constraint_id": "c_no_text", "type": "text_ban", "description": "No text."},
        ],
    )
    improvement = constraint_improvement(round0, round1)
    assert improvement["has_improvement"] is True
    assert "c_count" in improvement["improved_constraints"]
    ok, evidence = has_failure_then_improvement([{"review": round0}, {"review": round1}])
    assert ok is True
    assert evidence["from_round"] == 0
    assert evidence["to_round"] == 1
