"""
Stage 1: Text-to-Text — LLM semantic expansion via Claude API (Anthropic).

Replaces the hardcoded static prompts with LLM-generated prompts that are
richer in material, color, and geometric detail — better for T2I (Stage 2).

This version calls the Anthropic Claude API (no local GPU needed).
Run this locally (not on server) to generate prompts.json, then upload to server.

Usage:
    python pipeline/stage1_text_expansion.py [--dry-run] [--model MODEL]

Environment:
    ANTHROPIC_API_KEY  — required for real API calls

Output:
    pipeline/data/prompts.json  (downstream-compatible schema)
"""

import json
import re
import sys
import os
import argparse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

OUTPUT_FILE = Path(__file__).parent / "data/prompts.json"

# Seed concepts — (obj_id, concept_name, category)
SEED_CONCEPTS = [
    ("obj_001", "wooden_chair",   "daily_item"),
    ("obj_002", "ceramic_mug",    "daily_item"),
    ("obj_003", "desk_lamp",      "daily_item"),
    ("obj_004", "sedan_car",      "vehicle"),
    ("obj_005", "motorcycle",     "vehicle"),
    ("obj_006", "bicycle",        "vehicle"),
    ("obj_007", "cat",            "animal"),
    ("obj_008", "dinosaur",       "animal"),
    ("obj_009", "gazebo",         "architecture"),
    ("obj_010", "fountain",       "architecture"),
]

SYSTEM_PROMPT = """You are a T2I prompt engineer specializing in 3D object rendering.
For each object concept, generate:
1. A detailed T2I prompt (50-80 words) for a single isolated object on a white background with photorealistic studio lighting.
2. A JSON block with keys: material, color, style.

Always respond in this exact format (no preamble, no thinking tags):
PROMPT: <your T2I prompt here>
JSON: {"material": "...", "color": "...", "style": "..."}"""

USER_TEMPLATE = """Object concept: "{concept}"

Requirements for the T2I prompt:
- Single isolated {concept} on a pure white background
- End with: "studio lighting, white background, photorealistic, 8K"
- Include specific material (texture, finish), color, and geometric details
- 50-80 words, no people, no scene

Generate the prompt and JSON metadata now."""

REPAIR_SYSTEM_PROMPT = """You are a T2I prompt engineer specializing in repairing failed 3D asset prompts for downstream 3D reconstruction.
You will receive an object concept, the previous prompt, and smoke-gate failure signals from a render-review loop.

Rewrite the prompt so it keeps the same object concept but materially improves geometry clarity, silhouette readability, material realism, and white-background isolatability.

Always respond in this exact format (no preamble, no thinking tags):
PROMPT: <your repaired T2I prompt here>
JSON: {"material": "...", "color": "...", "style": "..."}"""


SCENE_PROFILE_4BLEND = {
    "name": "4blend_roadside_autumn",
    "environment": "an overcast autumn roadside scene with damp asphalt, muted earthy colors, and soft diffuse outdoor light",
    "palette": "muted brown, walnut, oak, asphalt gray, autumn foliage tones",
    "material_goal": "realistic material response that will still look plausible when later inserted into a moody outdoor road scene",
    "negative_style": "avoid showroom gloss, plastic toy look, oversaturated colors, and bright studio-commercial styling",
}


TEMPLATE_LIBRARY = {
    "wooden_chair": {
        "material": "solid oak or walnut wood with visible grain and a subtle varnished sheen",
        "color": "medium-dark natural brown wood",
        "style": "minimal Scandinavian dining chair",
        "geometry": "straight backrest, flat seat, sturdy square legs, clean joinery, realistic proportions",
    },
    "ceramic_mug": {
        "material": "glazed ceramic with subtle speckled surface variation and believable reflections",
        "color": "deep cobalt or earthy blue glaze with lighter glaze variation",
        "style": "practical everyday mug with clear readable silhouette",
        "geometry": "a complete rounded handle, a clearly open thick rim, believable ceramic wall thickness, and a smooth base ring",
    },
    "desk_lamp": {
        "material": "powder-coated metal with brushed aluminum joints",
        "color": "matte charcoal black",
        "style": "practical industrial desk lamp",
        "geometry": "round base, articulated arm, conical shade, readable mechanical hinges",
    },
    "sedan_car": {
        "material": "painted automotive metal with clean panel seams, realistic glass, and restrained reflections",
        "color": "deep graphite gray with neutral metallic undertones",
        "style": "compact modern four-door sedan",
        "geometry": "four readable wheels, clear windshield and windows, defined headlights, proper wheel arches, and realistic road-car proportions",
    },
    "motorcycle": {
        "material": "painted metal bodywork with matte black mechanical parts and clean chrome accents",
        "color": "deep red body panels with black frame details",
        "style": "practical street motorcycle",
        "geometry": "two aligned wheels, visible handlebars, readable seat, compact fuel tank, and believable fork and swingarm structure",
    },
    "bicycle": {
        "material": "painted aluminum frame with rubber tires and brushed metal drivetrain details",
        "color": "muted forest green frame with black components",
        "style": "simple city bicycle",
        "geometry": "two thin wheels, straight handlebar, readable frame triangle, visible pedals, saddle, and chain area with correct proportions",
    },
    "backpack": {
        "material": "woven nylon fabric with subtle stitching, durable straps, and matte zipper hardware",
        "color": "charcoal black with muted olive accents",
        "style": "practical everyday backpack",
        "geometry": "soft rectangular body, front pocket, top carry handle, clear shoulder straps, and believable volume without collapsing",
    },
    "suitcase": {
        "material": "hard-shell polycarbonate with fine surface texture and satin-finish metal hardware",
        "color": "muted navy blue",
        "style": "compact carry-on suitcase",
        "geometry": "clean rectangular shell, telescopic handle housing, four caster wheels, readable zipper seam, and stable upright proportions",
    },
    "traffic_cone": {
        "material": "matte flexible safety plastic with a worn reflective band",
        "color": "safety orange with white reflective stripe",
        "style": "standard roadside traffic cone",
        "geometry": "tapered cone body, square weighted base, crisp silhouette, and realistic thickness at the rim and base",
    },
    "fire_hydrant": {
        "material": "painted cast metal with subtle weathering and solid industrial surface detail",
        "color": "strong hydrant red with slightly darker caps",
        "style": "classic urban fire hydrant",
        "geometry": "cylindrical center body, side nozzles, domed top cap, base flange, and sturdy symmetrical proportions",
    },
    "mailbox": {
        "material": "painted sheet metal with matte finish and slight edge wear",
        "color": "desaturated blue-gray",
        "style": "curved-top roadside mailbox",
        "geometry": "arched top shell, front door, flag arm, and a stable support post with readable proportions",
    },
    "park_bench": {
        "material": "painted cast metal supports with varnished wood slats",
        "color": "warm brown wood with dark iron supports",
        "style": "simple public park bench",
        "geometry": "slatted seat and backrest, two clear side supports, grounded legs, and realistic seat-back proportions",
    },
    "potted_plant": {
        "material": "unglazed terracotta pot with matte green leaves and subtle natural variation",
        "color": "earthy terracotta with medium green foliage",
        "style": "compact decorative houseplant",
        "geometry": "round pot with readable rim and base, dense upright leaf cluster, and believable plant volume without messy overgrowth",
    },
    "skateboard": {
        "material": "painted maple deck with matte grip tape, metal trucks, and rubber wheels",
        "color": "natural wood underside with muted teal accents",
        "style": "street skateboard",
        "geometry": "slim deck with upward-curved nose and tail, four wheels, visible trucks, and correct board-to-wheel proportions",
    },
    "bicycle_helmet": {
        "material": "matte molded shell with subtle vent edges and woven interior straps",
        "color": "matte white with gray trim",
        "style": "modern commuter bicycle helmet",
        "geometry": "rounded protective shell, clear vent openings, chin straps, and believable thickness without collapsing",
    },
    "watering_can": {
        "material": "powder-coated metal with soft satin reflections",
        "color": "muted sage green",
        "style": "traditional garden watering can",
        "geometry": "rounded body, arched top handle, long spout, and a readable fill opening with realistic proportions",
    },
    "acoustic_guitar": {
        "material": "varnished tonewood body with dark bridge, subtle grain, and satin-finish neck",
        "color": "warm honey brown wood with darker sides",
        "style": "classic acoustic guitar",
        "geometry": "curved hollow body, round sound hole, long neck, readable headstock, and straight string path with believable proportions",
    },
    "office_trash_bin": {
        "material": "matte injection-molded plastic with mild surface texture",
        "color": "neutral dark gray",
        "style": "simple office waste bin",
        "geometry": "tapered cylindrical bin, open rim, sturdy base, and clean vertical profile without deformation",
    },
    "wooden_stool": {
        "material": "solid wood with visible grain and softly worn edges",
        "color": "light natural oak",
        "style": "simple utility stool",
        "geometry": "flat round seat, four sturdy legs, clear leg spacing, and readable stretchers with realistic proportions",
    },
    "tool_box": {
        "material": "painted steel with matte black handle grips and latch hardware",
        "color": "muted industrial red with black details",
        "style": "portable mechanic toolbox",
        "geometry": "rectangular storage body, hinged lid, top handle, side latches, and solid box-like proportions",
    },
    "folding_chair": {
        "material": "powder-coated tubular steel frame with woven synthetic seat fabric",
        "color": "matte dark gray frame with muted khaki fabric",
        "style": "portable outdoor folding chair",
        "geometry": "X-shaped folding support structure, slightly reclined backrest, taut seat panel, and four readable feet with believable proportions",
    },
    "side_table": {
        "material": "painted metal frame with a lightly textured wood or composite top",
        "color": "warm walnut top with matte black frame",
        "style": "compact modern side table",
        "geometry": "round or square tabletop, slim support frame, stable base, and clear tabletop thickness without warped proportions",
    },
    "electric_kettle": {
        "material": "brushed stainless steel body with matte black handle and lid trim",
        "color": "silver metal with black accents",
        "style": "modern countertop electric kettle",
        "geometry": "rounded kettle body, arched handle, short pouring spout, top lid, and stable base ring with realistic appliance proportions",
    },
    "table_fan": {
        "material": "painted metal grill with molded plastic blades and base housing",
        "color": "soft white body with gray grill details",
        "style": "compact oscillating table fan",
        "geometry": "circular front grill, central hub, readable fan blades, short neck, and weighted round base with clear silhouette",
    },
    "camping_lantern": {
        "material": "painted metal cage with frosted translucent light chamber and rubberized handle details",
        "color": "muted olive green with warm off-white diffuser",
        "style": "portable camping lantern",
        "geometry": "cylindrical light body, top carry handle, protective frame, and stable base with readable outdoor utility proportions",
    },
    "storage_crate": {
        "material": "sturdy molded plastic with subtle ribbing and matte surface texture",
        "color": "desaturated charcoal gray",
        "style": "industrial stackable storage crate",
        "geometry": "rectangular bin shape, open top, reinforced side walls, readable hand holds, and straight stackable edges",
    },
    "hand_truck": {
        "material": "painted steel frame with hard rubber wheels and textured plastic handle grips",
        "color": "industrial blue frame with black wheels",
        "style": "upright warehouse hand truck",
        "geometry": "tall tubular frame, curved top handles, two aligned wheels, and a flat bottom toe plate with realistic dolly proportions",
    },
    "wheelbarrow": {
        "material": "painted metal basin with steel handles and a rubber front wheel",
        "color": "muted green basin with black wheel and dark steel frame",
        "style": "garden wheelbarrow",
        "geometry": "single front wheel, deep tray basin, long handles, and two support legs with believable outdoor tool proportions",
    },
    "step_ladder": {
        "material": "brushed aluminum frame with textured anti-slip steps and plastic foot caps",
        "color": "silver aluminum with dark gray step treads",
        "style": "portable household step ladder",
        "geometry": "A-frame stance, evenly spaced steps, top platform, and clear hinge structure with realistic ladder proportions",
    },
    "picnic_cooler": {
        "material": "hard molded plastic with a lightly textured shell and latch hardware",
        "color": "off-white body with muted blue lid",
        "style": "portable picnic cooler",
        "geometry": "rectangular insulated box, hinged lid, side handles, and sturdy base corners with believable cooler thickness",
    },
    "pickup_truck": {
        "material": "painted automotive metal with realistic glass, rubber tires, and restrained metallic reflections",
        "color": "deep graphite blue with black trim",
        "style": "mid-size modern pickup truck",
        "geometry": "double-cab body, open cargo bed, four readable wheels, defined headlights, and realistic road-vehicle proportions",
    },
    "scooter": {
        "material": "painted plastic body panels with matte black trim and metal fork details",
        "color": "muted teal body with black seat",
        "style": "compact city scooter",
        "geometry": "step-through body, front handlebar column, two aligned wheels, readable seat, and compact commuter proportions",
    },
    "delivery_van": {
        "material": "painted metal body with realistic glass, matte trim, and mild panel seam detail",
        "color": "clean white body with gray bumper trim",
        "style": "compact urban delivery van",
        "geometry": "boxy cargo body, short hood, large windshield, sliding side-door seam, and four clear wheels with believable van proportions",
    },
    "golf_cart": {
        "material": "painted fiberglass body with matte metal frame supports and vinyl seating",
        "color": "off-white body with beige seats",
        "style": "practical two-seat golf cart",
        "geometry": "roof canopy, open sides, bench seat, steering area, and four small wheels with correct cart proportions",
    },
    "traffic_barrier": {
        "material": "molded safety plastic with worn reflective panels and a matte surface",
        "color": "orange and white striped safety colors",
        "style": "portable roadside traffic barrier",
        "geometry": "long rectangular barrier body, weighted feet, crisp silhouette, and readable reflective striping with realistic thickness",
    },
    "road_bollard": {
        "material": "painted steel with subtle weathering and reflective band details",
        "color": "dark gray post with yellow reflective accents",
        "style": "fixed roadside safety bollard",
        "geometry": "upright cylindrical post, rounded or flat cap, stable base flange, and clean vertical silhouette with believable street scale",
    },
    "parking_meter": {
        "material": "painted cast metal with matte display housing and subtle edge wear",
        "color": "blue-gray body with black display trim",
        "style": "single-space parking meter",
        "geometry": "narrow post, rounded meter head, front display area, payment slot details, and realistic curbside proportions",
    },
    "street_sign": {
        "material": "painted metal sign plate with galvanized pole and low-gloss printed surface",
        "color": "blue sign face with white markings and gray pole",
        "style": "urban roadside street sign",
        "geometry": "rectangular or square sign panel, slim vertical pole, readable plate thickness, and clean upright proportions",
    },
    "planter_box": {
        "material": "painted wood or composite planter body with matte foliage and soil detail",
        "color": "warm brown planter with medium green plants",
        "style": "rectangular outdoor planter box",
        "geometry": "box-shaped planter body, visible rim thickness, dense but controlled foliage, and grounded rectangular proportions",
    },
    "street_trash_can": {
        "material": "painted perforated steel with matte finish and slight outdoor wear",
        "color": "dark green metal with black base trim",
        "style": "public street trash can",
        "geometry": "upright cylindrical bin, open top rim, perforated side walls, and sturdy base with realistic sidewalk scale",
    },
    "basketball": {
        "material": "pebbled rubber with realistic channel grooves and matte sports-surface response",
        "color": "classic orange-brown basketball color",
        "style": "standard outdoor basketball",
        "geometry": "clean spherical form, readable seam channels, and believable sports-ball proportions without distortion",
    },
    "soccer_ball": {
        "material": "synthetic leather panels with subtle seam detail and matte surface finish",
        "color": "white base with black panel accents",
        "style": "standard match soccer ball",
        "geometry": "clean spherical shape, readable stitched panel pattern, and realistic ball roundness without deformation",
    },
    "tennis_racket": {
        "material": "painted composite frame with woven string bed and wrapped handle grip",
        "color": "white frame with dark navy accents",
        "style": "modern tennis racket",
        "geometry": "oval racket head, taut string grid, slim shaft, and wrapped handle with correct sports-equipment proportions",
    },
    "baseball_bat": {
        "material": "varnished wood or satin-finish alloy with subtle wear at the grip area",
        "color": "natural wood brown with dark grip tape",
        "style": "full-size baseball bat",
        "geometry": "long tapered cylindrical form, thicker barrel end, narrow handle, and smooth readable silhouette",
    },
    "surfboard": {
        "material": "gloss-coated fiberglass with subtle wax texture and soft specular reflections",
        "color": "off-white board with muted blue accent stripe",
        "style": "classic short surfboard",
        "geometry": "long streamlined board shape, pointed nose, rounded tail, visible fin cluster, and believable board thickness",
    },
    "life_ring": {
        "material": "coated foam safety material with wrapped rope detail and weather-resistant finish",
        "color": "bright rescue orange with white rope accents",
        "style": "marine safety life ring",
        "geometry": "clean torus ring shape, readable inner opening, wrapped rope around the perimeter, and realistic flotation proportions",
    },
    "barbecue_grill": {
        "material": "painted metal body with dark grilling surfaces and light heat discoloration",
        "color": "matte black body with silver handles",
        "style": "compact backyard barbecue grill",
        "geometry": "rounded or boxy grill chamber, lid handle, side shelf hints, support legs, and two small wheels with stable proportions",
    },
    "portable_generator": {
        "material": "painted steel frame with molded plastic housing panels and matte control surfaces",
        "color": "industrial yellow body with black frame",
        "style": "portable jobsite generator",
        "geometry": "boxy generator core, tubular carry frame, control panel face, and two wheels with realistic equipment proportions",
    },
    "shopping_cart": {
        "material": "chrome-coated metal wire frame with plastic handle trim and rubber caster wheels",
        "color": "silver metal with red handle detail",
        "style": "standard supermarket shopping cart",
        "geometry": "wire basket body, child seat flap, four caster wheels, and lower rack shelf with believable retail-cart proportions",
    },
    "fire_extinguisher": {
        "material": "painted steel cylinder with matte hose and metal valve hardware",
        "color": "safety red body with black hose",
        "style": "portable fire extinguisher",
        "geometry": "upright pressure cylinder, top handle and nozzle assembly, curved hose, and stable bottom ring with realistic industrial scale",
    },
    "cat": {
        "material": "real fur with visible strand direction and subtle tonal variation",
        "color": "warm orange tabby with muted cream accents",
        "style": "real domestic short-haired cat",
        "geometry": "upright seated pose, alert ears, compact body, readable paws and tail",
    },
}

GEOMETRY_SIGNAL_TERMS = (
    "geometry",
    "shape",
    "silhouette",
    "structure",
    "distortion",
    "mismatch",
    "flat",
    "non-flat",
    "deformed",
    "collapsed",
    "proportion",
    "rim",
    "handle",
    "wheel",
    "leg",
    "arm",
    "tail",
)

MATERIAL_SIGNAL_TERMS = (
    "material",
    "texture",
    "plastic",
    "gloss",
    "glossy",
    "specular",
    "matte",
    "muddy",
    "surface",
    "ceramic",
    "glaze",
    "fur",
    "metal",
)

LIGHTING_SIGNAL_TERMS = (
    "underexposed",
    "overexposed",
    "flat_lighting",
    "flat lighting",
    "lighting",
    "shadow",
    "highlight",
    "dark",
    "black",
)

FRAMING_SIGNAL_TERMS = (
    "object_too_small",
    "object_too_large",
    "off_center",
    "cutoff",
    "occlusion",
    "full object",
    "fully visible",
    "cropped",
)

MASK_SIGNAL_TERMS = (
    "mask",
    "boundary",
    "background",
    "spill",
    "hole",
)


def get_seed_concept(obj_id: Optional[str] = None, concept_name: Optional[str] = None) -> tuple[str, str, str]:
    for item_id, item_name, category in SEED_CONCEPTS:
        if obj_id and item_id == obj_id:
            return item_id, item_name, category
        if concept_name and item_name == concept_name:
            return item_id, item_name, category
    if obj_id:
        fallback_name = obj_id
    else:
        fallback_name = str(concept_name or "object").strip().replace(" ", "_")
    return str(obj_id or ""), fallback_name, "unknown"


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_coerce_text(item) for item in value if _coerce_text(item)]
    text = _coerce_text(value)
    if not text:
        return []
    return [part.strip(" -*") for part in re.split(r"[;\n]", text) if part.strip(" -*")]


def load_seed_concepts(seed_concepts_file: Optional[str] = None) -> list[tuple[str, str, str]]:
    if not seed_concepts_file:
        return list(SEED_CONCEPTS)

    payload = json.loads(Path(seed_concepts_file).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("[Stage 1] --seed-concepts-file must contain a JSON list")

    loaded: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if isinstance(item, dict):
            obj_id = _coerce_text(item.get("id"))
            concept_name = _coerce_text(item.get("name"))
            category = _coerce_text(item.get("category")) or "unknown"
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            obj_id = _coerce_text(item[0])
            concept_name = _coerce_text(item[1])
            category = _coerce_text(item[2]) or "unknown"
        else:
            raise SystemExit(f"[Stage 1] Invalid seed concept entry at index {index}: {item!r}")

        if not obj_id or not concept_name:
            raise SystemExit(f"[Stage 1] Seed concept entry missing id/name at index {index}: {item!r}")
        if obj_id in seen_ids:
            raise SystemExit(f"[Stage 1] Duplicate object id in seed concepts file: {obj_id}")
        seen_ids.add(obj_id)
        loaded.append((obj_id, concept_name, category))

    if not loaded:
        raise SystemExit("[Stage 1] --seed-concepts-file produced no seed concepts")
    return loaded


def _shorten(text: str, max_words: int = 24) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(needle.lower() in lowered for needle in needles)


def _merge_template_meta(concept_name: str, features: Optional[dict]) -> dict:
    template = dict(TEMPLATE_LIBRARY.get(concept_name, {
        "material": "photorealistic material with realistic surface response",
        "color": "natural muted color palette",
        "style": "realistic product-scale object",
        "geometry": f"clean readable geometry for a {concept_name.replace('_', ' ')}",
    }))
    features = features or {}
    for key in ("material", "color", "style"):
        value = _coerce_text(features.get(key))
        if value:
            template[key] = value
    return template


def _build_repair_focus_clauses(failure_summary: dict) -> list[str]:
    failure_summary = failure_summary or {}
    structure_consistency = _coerce_text(failure_summary.get("structure_consistency")).lower()
    issue_tags = [tag.lower() for tag in _coerce_list(failure_summary.get("issue_tags"))]
    issue_text = " ".join(issue_tags)
    major_issues = " ".join(_coerce_list(failure_summary.get("major_issues")))
    suggested_fixes = " ".join(_coerce_list(failure_summary.get("suggested_fixes")))
    combined = " ".join(
        part for part in (
            issue_text,
            major_issues.lower(),
            suggested_fixes.lower(),
            _coerce_text(failure_summary.get("trace_text_excerpt")).lower(),
            _coerce_text(failure_summary.get("reason")).lower(),
        )
        if part
    )

    clauses: list[str] = []
    if structure_consistency == "major_mismatch" or _contains_any(combined, GEOMETRY_SIGNAL_TERMS):
        clauses.append(
            "Accurate object geometry, correct proportions, clear component separation, and believable thickness."
        )
    if _contains_any(combined, MATERIAL_SIGNAL_TERMS):
        clauses.append(
            "Crisp material response, readable surface detail, and natural specular variation instead of a flat, plastic, or muddy look."
        )
    if _contains_any(combined, LIGHTING_SIGNAL_TERMS):
        clauses.append(
            "Readable midtones and highlights so the object does not read as too dark, too black, or visually flat."
        )
    if _contains_any(combined, FRAMING_SIGNAL_TERMS):
        clauses.append(
            "Full object centered, fully visible, and clearly separated from the background."
        )
    if _contains_any(combined, MASK_SIGNAL_TERMS):
        clauses.append(
            "Clean outer contour with no thin fragments, holes, or ambiguous edges."
        )
    if not clauses:
        clauses.append(
            "Complete readable silhouette, realistic material response, and reconstruction-friendly geometry."
        )
    return clauses[:4]


def build_failure_aware_prompt(repair_spec: dict, scene_profile: dict) -> dict:
    obj_id, concept_name, category = get_seed_concept(
        obj_id=_coerce_text(repair_spec.get("id")) or None,
        concept_name=_coerce_text(repair_spec.get("name")) or None,
    )
    human_name = concept_name.replace("_", " ")
    previous_features = repair_spec.get("previous_features") if isinstance(repair_spec.get("previous_features"), dict) else {}
    meta = _merge_template_meta(concept_name, previous_features)
    failure_summary = repair_spec.get("failure_summary") if isinstance(repair_spec.get("failure_summary"), dict) else {}
    focus_clauses = _build_repair_focus_clauses(failure_summary)

    geometry = meta["geometry"]
    combined = " ".join(
        [
            _coerce_text(failure_summary.get("trace_text_excerpt")).lower(),
            " ".join(_coerce_list(failure_summary.get("major_issues"))).lower(),
            " ".join(_coerce_list(failure_summary.get("suggested_fixes"))).lower(),
        ]
    )
    if concept_name == "ceramic_mug":
        if "handle" in combined and "rounded handle" not in geometry:
            geometry += ", complete rounded handle with a clear open gap"
        if "rim" in combined and "rim" not in geometry:
            geometry += ", clearly open thick rim"
        geometry += ", standard everyday mug proportions"

    focus_sentence = " ".join(focus_clauses[:3])
    geometry_connector = ", " if geometry.lower().startswith(("single ", human_name.lower())) else " with "
    scene_environment = _coerce_text(scene_profile.get("environment")) or "an outdoor scene"
    scene_negative_style = _coerce_text(scene_profile.get("negative_style")) or "avoid toy-like plastic and muddy texture"
    scene_palette = _coerce_text(scene_profile.get("palette"))
    style_sentence = f"Muted photorealistic styling that stays believable in {scene_environment}."
    negative_sentence = scene_negative_style.rstrip(".")
    if negative_sentence:
        negative_sentence = negative_sentence[0].upper() + negative_sentence[1:]
    if scene_palette:
        negative_sentence = f"{negative_sentence}; favor {scene_palette}".strip()
    if negative_sentence and not negative_sentence.endswith("."):
        negative_sentence += "."
    prompt_parts = [
        f"A single isolated {human_name}{geometry_connector}{geometry}, made of {meta['material']}, in {meta['color']}, {meta['style']}.",
        "Keep the full object centered, fully visible, and easy to segment on white for stable 3D reconstruction.",
        focus_sentence,
        style_sentence,
        negative_sentence,
        "No background scene elements, no extra objects, pure white background, soft studio lighting, photorealistic, 8K",
    ]
    prompt = " ".join(part.strip() for part in prompt_parts if _coerce_text(part))
    return {
        "id": obj_id,
        "name": concept_name,
        "category": category,
        "prompt": prompt,
        "features": {
            "material": meta["material"],
            "color": meta["color"],
            "style": meta["style"],
        },
    }


def build_repair_user_prompt(repair_spec: dict) -> str:
    obj_id, concept_name, category = get_seed_concept(
        obj_id=_coerce_text(repair_spec.get("id")) or None,
        concept_name=_coerce_text(repair_spec.get("name")) or None,
    )
    failure_summary = repair_spec.get("failure_summary") if isinstance(repair_spec.get("failure_summary"), dict) else {}
    previous_prompt = _coerce_text(repair_spec.get("previous_prompt"))
    previous_features = repair_spec.get("previous_features") if isinstance(repair_spec.get("previous_features"), dict) else {}

    issue_tags = ", ".join(_coerce_list(failure_summary.get("issue_tags"))) or "none"
    major_issues = "\n".join(f"- {item}" for item in _coerce_list(failure_summary.get("major_issues"))) or "- none provided"
    suggested_fixes = "\n".join(f"- {item}" for item in _coerce_list(failure_summary.get("suggested_fixes"))) or "- none provided"

    return (
        f"Object id: {obj_id or 'unknown'}\n"
        f"Object concept: {concept_name.replace('_', ' ')}\n"
        f"Category: {category}\n"
        f"Previous prompt:\n{previous_prompt or '[missing previous prompt]'}\n\n"
        f"Previous features JSON:\n{json.dumps(previous_features or {}, ensure_ascii=False)}\n\n"
        "Latest smoke failure summary:\n"
        f"- verdict: {_coerce_text(failure_summary.get('detected_verdict')) or 'unknown'}\n"
        f"- asset_viability: {_coerce_text(failure_summary.get('asset_viability')) or 'unknown'}\n"
        f"- hybrid_score: {_coerce_text(failure_summary.get('hybrid_score')) or 'unknown'}\n"
        f"- structure_consistency: {_coerce_text(failure_summary.get('structure_consistency')) or 'unknown'}\n"
        f"- issue_tags: {issue_tags}\n"
        f"- failure_reason: {_coerce_text(failure_summary.get('reason')) or 'unknown'}\n"
        f"- abandon_reason: {_coerce_text(failure_summary.get('abandon_reason')) or 'none'}\n"
        "Major issues:\n"
        f"{major_issues}\n"
        "Suggested fixes:\n"
        f"{suggested_fixes}\n"
        f"Trace excerpt:\n{_shorten(_coerce_text(failure_summary.get('trace_text_excerpt')), max_words=120)}\n\n"
        "Requirements for the repaired prompt:\n"
        "- Keep the same semantic object concept. Do not change the object category.\n"
        "- Materially rewrite the prompt instead of lightly editing a few words.\n"
        "- Directly address the failure signals above, especially geometry readability, material realism, and silhouette clarity.\n"
        "- Keep it as a single isolated object on a pure white background.\n"
        '- End with: "soft studio lighting, white background, photorealistic, 8K"\n'
        "- Include concise but concrete material, color, and geometry detail.\n"
        "- 60-95 words, no people, no scene.\n\n"
        "Generate the repaired prompt and JSON metadata now."
    )


def build_scene_conditioned_prompt(concept_name: str, scene_profile: dict) -> dict:
    """Template-based prompt for isolated-object generation with scene-aware material styling."""
    human_name = concept_name.replace("_", " ")
    meta = TEMPLATE_LIBRARY.get(concept_name, {
        "material": "photorealistic material with realistic surface response",
        "color": "natural muted color palette",
        "style": "realistic product-scale object",
        "geometry": f"clean readable geometry for a {human_name}",
    })
    prompt = (
        f"A single isolated {human_name}, {meta['geometry']}, made of {meta['material']}, "
        f"in {meta['color']}, {meta['style']}. "
        f"Keep the object fully visible, centered, with a clean readable silhouette for 3D reconstruction. "
        f"Use a restrained, photorealistic material style that would still remain believable after later insertion into "
        f"{scene_profile['environment']}, with {scene_profile['material_goal']}. "
        f"Favor {scene_profile['palette']}; {scene_profile['negative_style']}. "
        f"No background scene elements, no extra objects, pure white background, soft studio lighting, photorealistic, 8K"
    )
    return {
        "prompt": prompt,
        "features": {
            "material": meta["material"],
            "color": meta["color"],
            "style": meta["style"],
        },
    }


def parse_response(response: str) -> dict:
    """Extract PROMPT and JSON from LLM response, robust to CoT preamble."""
    # Strip <think>...</think> blocks (Qwen-style CoT)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Extract PROMPT line
    prompt_match = re.search(r"PROMPT:\s*(.+?)(?=\nJSON:|$)", response, re.DOTALL)
    prompt_text = prompt_match.group(1).strip() if prompt_match else ""

    # Fallback: take first non-empty line
    if not prompt_text:
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        prompt_text = lines[0] if lines else "photorealistic studio photo, white background, 8K"

    # Extract JSON block
    json_match = re.search(r"JSON:\s*(\{[^}]+\})", response, re.DOTALL)
    if json_match:
        try:
            features = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            features = {"material": "unknown", "color": "unknown", "style": "unknown"}
    else:
        features = {"material": "unknown", "color": "unknown", "style": "unknown"}

    return {"prompt": prompt_text, "features": features}


def _create_anthropic_client():
    try:
        import anthropic
    except ImportError:
        print("[Stage 1] ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[Stage 1] ERROR: ANTHROPIC_API_KEY not set in environment.")
        sys.exit(1)

    return anthropic.Anthropic(api_key=api_key)


def _stage1_api_key() -> Optional[str]:
    return os.environ.get("STAGE1_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


def _anthropic_messages_endpoint(base_url: str) -> str:
    endpoint = str(base_url).rstrip("/")
    if endpoint.endswith("/v1/messages"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/messages"
    return endpoint + "/v1/messages"


def _call_anthropic_messages_http(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    base_url: str,
    timeout: float,
) -> str:
    api_key = _stage1_api_key()
    if not api_key:
        raise SystemExit("[Stage 1] STAGE1_API_KEY or ANTHROPIC_API_KEY is required for relay API calls")

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    request = urllib.request.Request(
        _anthropic_messages_endpoint(base_url),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"[Stage 1] Anthropic-compatible relay failed: HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"[Stage 1] Anthropic-compatible relay failed: {exc}") from exc

    parts = []
    for item in payload.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]))
    text = "".join(parts) or str(payload.get("text") or "")
    if not text:
        raise SystemExit("[Stage 1] Relay returned no text content")
    return text


def call_claude_api(concept_name: str, model: str, api_base_url: Optional[str] = None, api_timeout: float = 300) -> dict:
    """Call Anthropic Claude API for one concept."""
    human_name = concept_name.replace("_", " ")
    user_prompt = USER_TEMPLATE.format(concept=human_name)

    if api_base_url:
        response_text = _call_anthropic_messages_http(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=512,
            base_url=api_base_url,
            timeout=api_timeout,
        )
        print(f"  [Relay API response snippet] {response_text[:200]}")
        return parse_response(response_text)

    client = _create_anthropic_client()
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    response_text = message.content[0].text
    print(f"  [API response snippet] {response_text[:200]}")
    return parse_response(response_text)


def call_claude_repair_api(
    repair_spec: dict,
    model: str,
    api_base_url: Optional[str] = None,
    api_timeout: float = 300,
) -> dict:
    user_prompt = build_repair_user_prompt(repair_spec)
    if api_base_url:
        response_text = _call_anthropic_messages_http(
            model=model,
            system_prompt=REPAIR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=700,
            base_url=api_base_url,
            timeout=api_timeout,
        )
        print(f"  [Repair relay API response snippet] {response_text[:200]}")
        return parse_response(response_text)

    client = _create_anthropic_client()
    message = client.messages.create(
        model=model,
        max_tokens=700,
        system=REPAIR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    response_text = message.content[0].text
    print(f"  [Repair API response snippet] {response_text[:200]}")
    return parse_response(response_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API, use placeholder text (for testing pipeline)")
    parser.add_argument("--model", default="claude-haiku-4-5",
                        help="Anthropic model to use (default: claude-haiku-4-5)")
    parser.add_argument("--api-provider", choices=["anthropic"], default=os.environ.get("STAGE1_API_PROVIDER"))
    parser.add_argument("--api-base-url", default=os.environ.get("STAGE1_API_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL"))
    parser.add_argument("--api-timeout", type=float, default=float(os.environ.get("STAGE1_API_TIMEOUT", "300")))
    parser.add_argument("--ids", default=None,
                        help="Comma-separated object IDs to generate (default: all seed concepts)")
    parser.add_argument("--output-file", default=str(OUTPUT_FILE),
                        help="Where to write prompts JSON")
    parser.add_argument("--seed-concepts-file", default=None,
                        help="Optional JSON file overriding the default seed concept list")
    parser.add_argument("--scene-conditioned", action="store_true",
                        help="Generate prompts that keep isolated white background but are styled for the 4.blend roadside scene")
    parser.add_argument("--template-only", action="store_true",
                        help="Use local templates instead of calling the API")
    parser.add_argument("--repair-spec-file", default=None,
                        help="JSON file describing a failure-aware prompt rewrite request")
    args = parser.parse_args()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    repair_spec = None
    if args.repair_spec_file:
        repair_spec = json.loads(Path(args.repair_spec_file).read_text(encoding="utf-8"))
        if isinstance(repair_spec, list):
            repair_spec = repair_spec[0] if repair_spec else {}
        if not isinstance(repair_spec, dict):
            raise SystemExit("[Stage 1] --repair-spec-file must contain a JSON object or single-element list")

    if repair_spec is not None:
        mode_label = "failure-aware local rewrite" if (args.template_only or args.scene_conditioned or args.dry_run) else f"failure-aware API rewrite ({args.model})"
        print(f"[Stage 1] Using {mode_label}")
        if args.template_only or args.scene_conditioned or args.dry_run:
            repaired = build_failure_aware_prompt(repair_spec, SCENE_PROFILE_4BLEND)
        else:
            parsed = call_claude_repair_api(
                repair_spec,
                args.model,
                api_base_url=args.api_base_url,
                api_timeout=args.api_timeout,
            )
            obj_id, concept_name, category = get_seed_concept(
                obj_id=_coerce_text(repair_spec.get("id")) or None,
                concept_name=_coerce_text(repair_spec.get("name")) or None,
            )
            repaired = {
                "id": obj_id,
                "name": concept_name,
                "category": category,
                "prompt": parsed["prompt"],
                "features": parsed["features"],
            }
        output_file.write_text(json.dumps([repaired], indent=2, ensure_ascii=False))
        print(f"  -> {repaired['id']}: {repaired['prompt'][:160]}...")
        print(f"[Stage 1] Wrote 1 repaired prompt -> {output_file}")
        return

    selected = load_seed_concepts(args.seed_concepts_file)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        selected = [row for row in selected if row[0] in wanted]
        if not selected:
            raise SystemExit(f"[Stage 1] No matching seed concepts for ids={sorted(wanted)}")

    if args.dry_run:
        print("[Stage 1] DRY RUN mode — using placeholder prompts")
        results = []
        for obj_id, name, category in selected:
            human = name.replace("_", " ")
            results.append({
                "id": obj_id,
                "name": name,
                "category": category,
                "prompt": (f"A photorealistic {human} on a pure white background, "
                           "studio lighting, white background, photorealistic, 8K"),
                "features": {
                    "material": "placeholder",
                    "color": "placeholder",
                    "style": "placeholder"
                },
            })
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[Stage 1] Wrote {len(results)} dry-run prompts -> {output_file}")
        return

    if args.template_only or args.scene_conditioned:
        mode = "scene-conditioned templates" if args.scene_conditioned else "local templates"
        print(f"[Stage 1] Using {mode}")
        results = []
        for obj_id, name, category in selected:
            expanded = build_scene_conditioned_prompt(name, SCENE_PROFILE_4BLEND)
            results.append({
                "id": obj_id,
                "name": name,
                "category": category,
                "prompt": expanded["prompt"],
                "features": expanded["features"],
            })
            print(f"  -> {obj_id}: {expanded['prompt'][:120]}...")
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[Stage 1] Wrote {len(results)} prompts -> {output_file}")
        return

    print(f"[Stage 1] Using Claude API model: {args.model}")
    results = []
    for obj_id, name, category in selected:
        human_name = name.replace("_", " ")
        print(f"\n[Stage 1] Expanding concept: {human_name} ({obj_id})")
        expanded = call_claude_api(
            name,
            args.model,
            api_base_url=args.api_base_url,
            api_timeout=args.api_timeout,
        )
        results.append({
            "id":       obj_id,
            "name":     name,
            "category": category,
            "prompt":   expanded["prompt"],
            "features": expanded["features"],
        })
        wc = len(expanded["prompt"].split())
        print(f"  -> prompt ({wc} words): {expanded['prompt'][:80]}...")
        print(f"  -> features: {expanded['features']}")

    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[Stage 1] Done. Wrote {len(results)} prompts -> {output_file}")


if __name__ == "__main__":
    main()
