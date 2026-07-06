from pipeline.multimodal.edit_review import normalize_edit_review, default_edit_constraints


def test_edit_review_pass_requires_all_constraints_pass():
    constraints = default_edit_constraints("Change the blue mug to red.")
    review = normalize_edit_review(
        {
            "edit_checklist": [
                {
                    "constraint_id": row["constraint_id"],
                    "type": row["type"],
                    "description": row["description"],
                    "status": "pass",
                    "evidence": "visible",
                    "confidence": "high",
                }
                for row in constraints
            ]
        },
        constraints,
    )
    assert review["vlm_route"] == "pass"
    assert review["edit_status_counts"] == {"pass": 6, "fail": 0, "uncertain": 0, "total": 6}
    assert review["issue_tags"] == ["none"]


def test_edit_review_missing_items_are_uncertain_not_passed():
    constraints = default_edit_constraints("Change the blue mug to red.")
    review = normalize_edit_review(
        {
            "edit_checklist": [
                {
                    "constraint_id": "c_instruction_following",
                    "type": "instruction_following",
                    "description": constraints[0]["description"],
                    "status": "pass",
                }
            ]
        },
        constraints,
    )
    assert review["vlm_route"] == "needs_fix"
    assert review["edit_status_counts"] == {"pass": 1, "fail": 0, "uncertain": 5, "total": 6}
    assert "c_source_preservation" in review["uncertain_edit_constraints"]
