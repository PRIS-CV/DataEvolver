from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VALID_CONSTRAINT_STATUS = {"pass", "fail", "uncertain"}
PASS_STATUS = "pass"
FAIL_STATUS = "fail"
UNCERTAIN_STATUS = "uncertain"


def _constraint(constraint_id: str, kind: str, description: str) -> Dict[str, str]:
    return {
        "constraint_id": constraint_id,
        "type": kind,
        "description": description,
    }


def _spec(prompt_id: str, tier: str, prompt: str, constraints: List[Dict[str, str]]) -> Dict[str, object]:
    return {
        "id": prompt_id,
        "tier": tier,
        "prompt": prompt,
        "constraints": constraints,
    }


def _simple_specs() -> List[Dict[str, object]]:
    rows = [
        ("simple_red_cube", "a single matte red cube centered on a light gray studio surface", "matte red cube"),
        ("simple_blue_mug", "a single blue ceramic mug with one visible handle on a wooden tabletop", "blue ceramic mug"),
        ("simple_yellow_ball", "a single yellow tennis ball with visible fuzzy texture on a clean table", "yellow tennis ball"),
        ("simple_black_lamp", "a single black adjustable desk lamp with a round base on a plain desk", "black desk lamp"),
        ("simple_glass_bottle", "a single transparent glass bottle with clear reflections on a neutral background", "transparent glass bottle"),
        ("simple_green_toy_car", "a single green toy car with four visible wheels on a white tabletop", "green toy car"),
        ("simple_silver_key", "a single silver metal key lying diagonally on dark fabric", "silver metal key"),
        ("simple_orange_cone", "a single orange traffic cone with a white reflective band on gray concrete", "orange traffic cone"),
        ("simple_wooden_spoon", "a single wooden spoon with visible grain on a linen cloth", "wooden spoon"),
        ("simple_white_bowl", "a single white porcelain bowl casting a soft shadow on a beige table", "white porcelain bowl"),
    ]
    return [
        _spec(
            prompt_id,
            "simple_object",
            f"Create {prompt}. No readable text, logos, or watermark.",
            [
                _constraint("c_object_identity", "object_identity", f"The image shows {identity} as the main subject."),
                _constraint("c_single_subject", "count", "There is exactly one main subject."),
                _constraint("c_no_text", "text_ban", "There is no readable text, logo, label, or watermark."),
            ],
        )
        for prompt_id, prompt, identity in rows
    ]


def _relation_spec(prompt_id: str, prompt: str, constraints: List[Tuple[str, str, str]]) -> Dict[str, object]:
    return _spec(
        prompt_id,
        "compositional_relation",
        f"Create {prompt}. Keep the layout easy to inspect and avoid readable text, logos, or watermarks.",
        [_constraint(cid, kind, desc) for cid, kind, desc in constraints]
        + [_constraint("c_no_text", "text_ban", "There is no readable text, logo, label, or watermark.")],
    )


def _relation_specs() -> List[Dict[str, object]]:
    rows = [
        (
            "rel_mug_book_key",
            "a blue mug to the left of a closed red book, with a silver key in front of the book",
            [
                ("c_left_right", "spatial_lr", "The blue mug is left of the red book."),
                ("c_front", "spatial_depth", "The silver key is in front of the red book."),
                ("c_objects", "object_identity", "The mug, red book, and silver key are all visible."),
            ],
        ),
        (
            "rel_bowl_lemon_spoon",
            "a white bowl behind two yellow lemons, with a wooden spoon on the right side",
            [
                ("c_count", "count", "Exactly two yellow lemons are visible."),
                ("c_depth", "spatial_depth", "The bowl is behind the lemons."),
                ("c_right", "spatial_lr", "The wooden spoon is on the right side."),
            ],
        ),
        (
            "rel_camera_notebook_pen",
            "a black camera in front of an open notebook, with a red pen crossing the notebook diagonally",
            [
                ("c_depth", "spatial_depth", "The black camera is in front of the open notebook."),
                ("c_part", "part_detail", "The red pen crosses the notebook diagonally."),
                ("c_objects", "object_identity", "The camera, notebook, and red pen are visible."),
            ],
        ),
        (
            "rel_three_vases",
            "three small vases arranged left to right as blue, white, then green",
            [
                ("c_count", "count", "Exactly three small vases are visible."),
                ("c_order", "spatial_lr", "The vases are ordered left to right as blue, white, then green."),
                ("c_material", "material", "The vases look ceramic or glazed."),
            ],
        ),
        (
            "rel_lamp_toolbox_mug",
            "a brass desk lamp behind a red toolbox, with a blue mug on the right",
            [
                ("c_depth", "spatial_depth", "The brass lamp is behind the red toolbox."),
                ("c_right", "spatial_lr", "The blue mug is on the right side."),
                ("c_material", "material", "The lamp reads as brass or warm metal."),
            ],
        ),
        (
            "rel_blocks_stack",
            "four wooden blocks stacked as two on the bottom, one in the middle, and one on top",
            [
                ("c_count", "count", "Exactly four wooden blocks are visible."),
                ("c_stack", "spatial_depth", "The blocks form a 2-1-1 vertical stack."),
                ("c_material", "material", "The blocks have visible wood material."),
            ],
        ),
        (
            "rel_plate_fork_glass",
            "a round white plate centered between a fork on the left and a clear glass on the right",
            [
                ("c_center", "composition", "The white plate is centered."),
                ("c_left_right", "spatial_lr", "The fork is left of the plate and the glass is right of it."),
                ("c_transparent", "transparent_reflective", "The glass is transparent with visible highlights."),
            ],
        ),
        (
            "rel_train_trees_bridge",
            "a red toy train on a bridge, with pine trees behind the bridge and snow in the foreground",
            [
                ("c_depth", "spatial_depth", "Pine trees are behind the bridge."),
                ("c_foreground", "spatial_depth", "Snow is visible in the foreground."),
                ("c_subject", "object_identity", "The red toy train is on the bridge."),
            ],
        ),
        (
            "rel_robot_plants",
            "a small white robot between two tall green plants inside a glass dome",
            [
                ("c_count", "count", "Exactly two tall green plants flank the robot."),
                ("c_between", "spatial_lr", "The robot is between the two plants."),
                ("c_glass", "transparent_reflective", "The glass dome is transparent or reflective."),
            ],
        ),
        (
            "rel_teapot_kettle_lemons",
            "a white teapot in front of a copper kettle, with three lemon slices on the left",
            [
                ("c_depth", "spatial_depth", "The white teapot is in front of the copper kettle."),
                ("c_count", "count", "Exactly three lemon slices are visible."),
                ("c_left", "spatial_lr", "The lemon slices are on the left side."),
            ],
        ),
        (
            "rel_submarine_turtle_coral",
            "a yellow submarine below a sea turtle and above red coral",
            [
                ("c_vertical", "spatial_depth", "The submarine is below the sea turtle and above the coral."),
                ("c_objects", "object_identity", "The submarine, sea turtle, and red coral are all visible."),
                ("c_water", "material", "The scene reads as underwater."),
            ],
        ),
        (
            "rel_tablet_window_astronaut",
            "an astronaut near a window, with a glowing tablet floating to the astronaut's right",
            [
                ("c_right", "spatial_lr", "The glowing tablet is to the astronaut's right."),
                ("c_window", "object_identity", "A window is visible near the astronaut."),
                ("c_anatomy", "part_detail", "The astronaut has coherent limbs and helmet."),
            ],
        ),
        (
            "rel_terrarium_cottage_mushrooms",
            "a glass terrarium containing a tiny cottage behind two red mushrooms",
            [
                ("c_count", "count", "Exactly two red mushrooms are visible."),
                ("c_depth", "spatial_depth", "The tiny cottage is behind the mushrooms."),
                ("c_glass", "transparent_reflective", "The terrarium glass edge or reflection is visible."),
            ],
        ),
        (
            "rel_cart_lanterns_steam",
            "a food cart with two orange lanterns above it and steam rising behind the counter",
            [
                ("c_count", "count", "Exactly two orange lanterns are above the cart."),
                ("c_depth", "spatial_depth", "Steam rises behind the counter area."),
                ("c_subject", "object_identity", "The food cart is clearly visible."),
            ],
        ),
        (
            "rel_shelf_boxes_bottle",
            "a shelf with a glass bottle in front of three cardboard boxes",
            [
                ("c_count", "count", "Exactly three cardboard boxes are visible."),
                ("c_depth", "spatial_depth", "The glass bottle is in front of the boxes."),
                ("c_transparent", "transparent_reflective", "The bottle is transparent or reflective."),
            ],
        ),
        (
            "rel_chair_table_laptop",
            "a wooden chair behind a small table, with a closed silver laptop on the table",
            [
                ("c_depth", "spatial_depth", "The chair is behind the table."),
                ("c_tabletop", "spatial_depth", "The silver laptop is on the table."),
                ("c_material", "material", "The chair reads as wood."),
            ],
        ),
        (
            "rel_clock_books",
            "a round clock leaning against two stacked books, with a candle on the left",
            [
                ("c_count", "count", "Exactly two stacked books are visible."),
                ("c_left", "spatial_lr", "The candle is on the left side."),
                ("c_contact", "spatial_depth", "The clock leans against the books."),
            ],
        ),
        (
            "rel_shoes_box",
            "a pair of black shoes in front of an open cardboard box, with one shoe partly inside the box",
            [
                ("c_count", "count", "Exactly two black shoes are visible."),
                ("c_depth", "spatial_depth", "The shoes are in front of the open box."),
                ("c_occlusion", "occlusion", "One shoe is partly inside or occluded by the box."),
            ],
        ),
        (
            "rel_pencils_cup",
            "five colored pencils inside a clear cup, with the red pencil at the front",
            [
                ("c_count", "count", "Exactly five colored pencils are visible."),
                ("c_front", "spatial_depth", "The red pencil is at the front."),
                ("c_cup", "transparent_reflective", "The cup is clear or transparent."),
            ],
        ),
        (
            "rel_watch_strap",
            "a silver wristwatch with the strap open, placed below a folded blue cloth",
            [
                ("c_depth", "spatial_depth", "The watch is below the folded blue cloth."),
                ("c_part", "part_detail", "The watch strap is open and visible."),
                ("c_material", "material", "The watch reads as silver metal."),
            ],
        ),
    ]
    return [_relation_spec(*row) for row in rows]


_COUNT_WORDS = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
}


def _hard_spec(
    prompt_id: str,
    subject: str,
    count: int,
    material: str,
    left_obj: str,
    right_obj: str,
    front_obj: str,
    back_obj: str,
    occluder: str,
    part_detail: str,
    distractor: str,
) -> Dict[str, object]:
    count_word = _COUNT_WORDS[count]
    other_count = _COUNT_WORDS.get(count - 1, str(count - 1))
    prompt = (
        f"Create a high-difficulty verification scene with exactly {count_word} {material} {subject}s as the primary subjects. "
        f"Place {left_obj} on the left and {right_obj} on the right; put {front_obj} in front of the {subject}s and {back_obj} behind them. "
        f"The {occluder} must partially occlude only the rightmost {subject}, leaving the other {other_count} fully visible. "
        f"Each {subject} must show {part_detail}. Include {distractor} in the background as same-color distractors that must not be counted as primary {subject}s. "
        "Show clear transparent or reflective behavior where appropriate, use realistic material cues, and avoid readable text, logos, labels, or watermarks."
    )
    return _spec(
        prompt_id,
        "hard_constraints",
        prompt,
        [
            _constraint("c_count_primary", "count", f"Exactly {count_word} primary {subject}s are visible."),
            _constraint("c_left_right_anchors", "spatial_lr", f"{left_obj} is on the left and {right_obj} is on the right."),
            _constraint("c_front_back_anchors", "spatial_depth", f"{front_obj} is in front of the {subject}s and {back_obj} is behind them."),
            _constraint("c_controlled_occlusion", "occlusion", f"The {occluder} occludes only the rightmost {subject}."),
            _constraint("c_primary_material", "material", f"The primary {subject}s read as {material}."),
            _constraint("c_no_text", "text_ban", "No readable text, logos, labels, or watermarks are present."),
            _constraint("c_transparent_reflective", "transparent_reflective", "Transparent or reflective behavior is visible where appropriate."),
            _constraint("c_local_parts", "part_detail", f"Each primary {subject} shows {part_detail}."),
            _constraint("c_same_color_distractor", "same_color_distractor", f"The {distractor} are visible but not counted as primary {subject}s."),
        ],
    )


def _hard_specs() -> List[Dict[str, object]]:
    rows = [
        ("hard_glass_bottles", "bottle", 3, "cobalt-blue transparent glass", "a matte black ruler", "a white ceramic cup", "a silver coin", "a folded gray notebook", "translucent tracing paper", "two narrow neck rings and a cork stopper", "blue glass marbles"),
        ("hard_chrome_keys", "key", 4, "reflective chrome metal", "a red wax seal", "a black pen cap", "a small brass screw", "a dark velvet pouch", "a clear plastic strip", "round bow holes and individual teeth", "silver paper clips"),
        ("hard_acrylic_blocks", "block", 3, "clear acrylic", "a green sticky note", "a blue eraser", "a copper washer", "a white grid card", "a smoky transparent sheet", "beveled edges and internal reflections", "clear ice-cube props"),
        ("hard_ceramic_cups", "cup", 3, "glossy white ceramic", "a yellow lemon wedge", "a steel spoon", "a cinnamon stick", "a beige napkin", "a transparent glass plate", "visible handles and dark inner rims", "white ceramic saucers"),
        ("hard_brass_lamps", "lamp", 2, "brushed brass metal", "a black notebook", "a red screwdriver", "a round washer", "a wooden block", "a translucent fabric strip", "visible hinges and power cords", "brass drawer knobs"),
        ("hard_amber_vials", "vial", 4, "amber transparent glass", "a white dropper", "a blue cap", "a folded pipette wrapper", "a gray lab tray", "a clear measuring strip", "black caps and tiny shoulders", "amber glass beads"),
        ("hard_black_cameras", "camera", 2, "matte black plastic and glass", "a red lens cloth", "a silver memory card", "a lens cap", "a white calibration card", "a transparent acrylic shield", "lens rings and shutter buttons", "black camera batteries"),
        ("hard_metal_gears", "gear", 5, "brushed steel", "a blue caliper", "a red handled file", "a brass washer", "a dark tool roll", "a clear oil dropper", "distinct teeth and center holes", "steel washers"),
        ("hard_green_bottles", "bottle", 3, "emerald green reflective glass", "a cork coaster", "a white funnel", "a silver bottle opener", "a linen towel", "a translucent plastic sleeve", "long necks and lip ridges", "green glass beads"),
        ("hard_porcelain_bowls", "bowl", 3, "glossy porcelain", "a black chopstick rest", "a copper spoon", "a folded recipe card face-down", "a wooden tray", "a clear glass lid", "thin rims and visible inner curves", "white porcelain sauce dishes"),
        ("hard_silver_watches", "watch", 2, "polished silver metal", "a navy cloth", "a black strap tool", "a tiny screw", "a gray display stand", "a transparent plastic cover", "crowns, lugs, and open straps", "silver coins"),
        ("hard_red_toolboxes", "toolbox", 2, "glossy red painted metal", "a yellow tape measure", "a black clamp", "a loose screw", "a wooden crate", "a clear safety visor", "handles, latches, and corner wear", "red metal tins"),
        ("hard_blue_tiles", "tile", 4, "glossy blue ceramic", "a white grout float", "a red sponge", "a silver spacer", "a gray backing board", "a transparent ruler", "beveled edges and grout gaps", "blue ceramic shards"),
        ("hard_clear_terrariums", "terrarium", 2, "transparent glass", "a green moss tray", "a brass spray bottle", "a tiny mushroom", "a wooden desk rail", "a sheer cloth curtain", "glass seams, refraction, and lids", "clear glass jars"),
        ("hard_copper_kettles", "kettle", 2, "reflective copper", "a white teapot", "a blue towel", "a lemon slice", "a dark stove grate", "a transparent steam guard", "spouts, handles, and black knobs", "copper measuring cups"),
        ("hard_white_robots", "robot", 3, "smooth white plastic", "a green plant marker", "a blue watering can", "a small pebble", "a curved glass wall", "a translucent leaf", "round camera eyes and jointed arms", "white plant pots"),
        ("hard_yellow_submarines", "submarine", 2, "glossy yellow metal", "a red coral branch", "a blue shell", "a small anchor", "a dark reef arch", "a translucent bubble curtain", "round portholes and tail fins", "yellow reef fish"),
        ("hard_silver_tablets", "tablet", 3, "reflective silver glass", "a black stylus", "a blue cable", "a microfiber cloth", "a white docking stand", "a clear screen protector", "thin bezels and side buttons", "silver metal plates"),
        ("hard_crystal_perfume", "perfume bottle", 3, "transparent crystal glass", "a pink ribbon", "a black atomizer bulb", "a pearl bead", "a mirrored tray", "a translucent silk scarf", "spray nozzles and faceted stoppers", "clear crystal beads"),
        ("hard_orange_cones", "traffic cone", 4, "orange rubber with reflective bands", "a gray wrench", "a blue glove", "a pebble", "a concrete curb", "a transparent rain sheet", "white bands and square bases", "orange rubber caps"),
    ]
    return [_hard_spec(*row) for row in rows]


def get_prompt_bank() -> List[Dict[str, object]]:
    return _simple_specs() + _relation_specs() + _hard_specs()


def prompt_bank_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for spec in get_prompt_bank():
        tier = str(spec["tier"])
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def select_prompt_specs(tier: Optional[str] = None, prompt_id: Optional[str] = None) -> List[Dict[str, object]]:
    specs = get_prompt_bank()
    if tier and tier != "all":
        specs = [spec for spec in specs if spec["tier"] == tier]
    if prompt_id:
        specs = [spec for spec in specs if spec["id"] == prompt_id]
    return deepcopy(specs)


def write_prompt_bank(path: Path, tier: Optional[str] = None) -> None:
    specs = select_prompt_specs(tier=tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_constraints(constraints: Optional[Iterable[dict]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    seen = set()
    for idx, item in enumerate(constraints or []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("constraint_id") or item.get("id") or f"c_{idx + 1:02d}").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        kind = str(item.get("type") or item.get("kind") or "prompt_alignment").strip()
        desc = str(item.get("description") or item.get("constraint") or "").strip()
        if not desc:
            continue
        normalized.append(_constraint(cid, kind, desc))
    return normalized


def infer_constraints_from_prompt(prompt: str) -> List[Dict[str, str]]:
    text = " ".join((prompt or "").split())
    lower = text.lower()
    constraints = [
        _constraint("c_prompt_alignment", "prompt_alignment", "The image follows the full original prompt."),
    ]
    if re.search(r"\b(exactly|only)\s+(one|two|three|four|five|six|\d+)\b", lower):
        constraints.append(_constraint("c_count", "count", "The exact requested object count is satisfied."))
    if " left " in f" {lower} " or " right " in f" {lower} ":
        constraints.append(_constraint("c_left_right", "spatial_lr", "The requested left/right relationship is satisfied."))
    if any(term in lower for term in ["in front of", "behind", "front", "back"]):
        constraints.append(_constraint("c_front_back", "spatial_depth", "The requested front/back relationship is satisfied."))
    if any(term in lower for term in ["occlude", "occludes", "occluded", "partially hidden", "partly inside"]):
        constraints.append(_constraint("c_occlusion", "occlusion", "The requested occlusion relationship is satisfied."))
    if any(term in lower for term in ["glass", "transparent", "reflective", "reflection", "mirror", "chrome", "metal"]):
        constraints.append(_constraint("c_transparent_reflective", "transparent_reflective", "Transparent or reflective material cues are visible."))
    if any(term in lower for term in ["handle", "ring", "cap", "button", "strap", "wheel", "edge", "rim", "teeth", "porthole"]):
        constraints.append(_constraint("c_local_parts", "part_detail", "The requested local part details are visible."))
    if any(term in lower for term in ["no text", "no readable", "no logos", "no logo", "watermark", "avoid text"]):
        constraints.append(_constraint("c_no_text", "text_ban", "No readable text, logos, labels, or watermarks are present."))
    return constraints


def constraints_for_prompt(prompt: str, explicit_constraints: Optional[Iterable[dict]] = None) -> List[Dict[str, str]]:
    explicit = normalize_constraints(explicit_constraints)
    return explicit if explicit else infer_constraints_from_prompt(prompt)


def normalize_constraint_review(review: Optional[dict], constraints: List[Dict[str, str]]) -> Dict[str, object]:
    review = review if isinstance(review, dict) else {}
    by_id = {item["constraint_id"]: item for item in constraints}
    checklist_rows = review.get("constraint_checklist")
    if not isinstance(checklist_rows, list):
        checklist_rows = []
    normalized_rows = []
    seen = set()
    for item in checklist_rows:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("constraint_id") or item.get("id") or "").strip()
        if not cid:
            continue
        base = by_id.get(cid, _constraint(cid, str(item.get("type") or "prompt_alignment"), str(item.get("description") or "")))
        status = str(item.get("status") or "").strip().lower()
        if status not in VALID_CONSTRAINT_STATUS:
            status = UNCERTAIN_STATUS
        normalized_rows.append(
            {
                "constraint_id": cid,
                "type": str(item.get("type") or base["type"]),
                "description": str(item.get("description") or base["description"]),
                "status": status,
                "evidence": str(item.get("evidence") or "").strip(),
                "confidence": str(item.get("confidence") or "medium").strip().lower()
                if str(item.get("confidence") or "medium").strip().lower() in {"low", "medium", "high"}
                else "medium",
            }
        )
        seen.add(cid)
    for constraint in constraints:
        if constraint["constraint_id"] not in seen:
            normalized_rows.append(
                {
                    **constraint,
                    "status": UNCERTAIN_STATUS,
                    "evidence": "Constraint was not explicitly judged by the reviewer.",
                    "confidence": "low",
                }
            )
    failed = [row for row in normalized_rows if row["status"] == FAIL_STATUS]
    uncertain = [row for row in normalized_rows if row["status"] == UNCERTAIN_STATUS]
    passed = [row for row in normalized_rows if row["status"] == PASS_STATUS]
    route = str(review.get("vlm_route") or "").strip().lower()
    if route not in {"pass", "needs_fix", "reject"}:
        route = "pass" if not failed and not uncertain else "needs_fix"
    if failed and route == "pass":
        route = "needs_fix"
    score = round(len(passed) / max(1, len(normalized_rows)), 4)
    return {
        **review,
        "schema_version": "t2i_constraint_review_v1",
        "vlm_route": route,
        "constraint_checklist": normalized_rows,
        "passed_constraints": [row["constraint_id"] for row in passed],
        "failed_constraints": [row["constraint_id"] for row in failed],
        "uncertain_constraints": [row["constraint_id"] for row in uncertain],
        "constraint_status_counts": {
            "pass": len(passed),
            "fail": len(failed),
            "uncertain": len(uncertain),
            "total": len(normalized_rows),
        },
        "constraint_pass_rate": score,
        "issue_tags": _issue_tags_from_constraints(failed, uncertain),
        "scores": _scores_from_constraint_rate(score, failed, uncertain),
    }


def _issue_tags_from_constraints(failed: List[dict], uncertain: List[dict]) -> List[str]:
    if not failed and not uncertain:
        return ["none"]
    tags = []
    kinds = {str(row.get("type")) for row in failed + uncertain}
    if "text_ban" in kinds:
        tags.append("text_artifact")
    if kinds - {"text_ban"}:
        tags.append("prompt_constraint_miss")
    return tags[:3] or ["prompt_constraint_miss"]


def _scores_from_constraint_rate(score: float, failed: List[dict], uncertain: List[dict]) -> Dict[str, int]:
    overall = max(1, min(5, round(score * 5)))
    if failed:
        overall = min(overall, 3)
    elif uncertain:
        overall = min(overall, 4)
    return {
        "lighting": 3,
        "object_integrity": overall,
        "composition": overall,
        "render_quality_semantic": overall,
        "overall": overall,
    }


def failed_constraint_rows(review: dict) -> List[Dict[str, object]]:
    checklist = review.get("constraint_checklist") or []
    return [row for row in checklist if isinstance(row, dict) and row.get("status") == FAIL_STATUS]


def uncertain_constraint_rows(review: dict) -> List[Dict[str, object]]:
    checklist = review.get("constraint_checklist") or []
    return [row for row in checklist if isinstance(row, dict) and row.get("status") == UNCERTAIN_STATUS]


def constraint_status_map(review: dict) -> Dict[str, str]:
    return {
        str(row.get("constraint_id")): str(row.get("status"))
        for row in review.get("constraint_checklist", [])
        if isinstance(row, dict) and row.get("constraint_id")
    }


def constraint_improvement(prev_review: dict, current_review: dict) -> Dict[str, object]:
    prev_status = constraint_status_map(prev_review)
    current_status = constraint_status_map(current_review)
    previously_bad = {cid for cid, status in prev_status.items() if status in {FAIL_STATUS, UNCERTAIN_STATUS}}
    improved = sorted(cid for cid in previously_bad if current_status.get(cid) == PASS_STATUS)
    regressed = sorted(
        cid
        for cid, status in prev_status.items()
        if status == PASS_STATUS and current_status.get(cid) in {FAIL_STATUS, UNCERTAIN_STATUS}
    )
    prev_bad_count = sum(1 for status in prev_status.values() if status in {FAIL_STATUS, UNCERTAIN_STATUS})
    current_bad_count = sum(1 for status in current_status.values() if status in {FAIL_STATUS, UNCERTAIN_STATUS})
    return {
        "improved_constraints": improved,
        "regressed_constraints": regressed,
        "prev_bad_count": prev_bad_count,
        "current_bad_count": current_bad_count,
        "bad_count_delta": prev_bad_count - current_bad_count,
        "has_improvement": bool(improved) and current_bad_count < prev_bad_count,
    }


def has_failure_then_improvement(events: List[dict]) -> Tuple[bool, Dict[str, object]]:
    for idx in range(1, len(events)):
        prev = events[idx - 1].get("review") or {}
        current = events[idx].get("review") or {}
        improvement = constraint_improvement(prev, current)
        if improvement["has_improvement"]:
            return True, {"from_round": idx - 1, "to_round": idx, **improvement}
    return False, {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export the built-in T2I constraint prompt bank")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tier", default="all", choices=["all", "simple_object", "compositional_relation", "hard_constraints"])
    args = parser.parse_args()
    write_prompt_bank(Path(args.output), tier=args.tier)
