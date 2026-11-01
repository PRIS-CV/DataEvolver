"""Frozen recoloring tasks and bounded execution recipes for the G2 pilot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .contracts import ArtifactRef, Contract, canonical_json, identifier, positive_int


CONSTRAINTS = ("target_color", "target_shape", "other_objects", "background", "no_artifacts")
RECIPE_FIELDS = ("target_description", "spatial_expression", "preservation_instruction")


@dataclass(frozen=True)
class G2Task(Contract):
    schema: ClassVar[str] = "dataevolver.g2_task.v1"
    task_id: str
    source: ArtifactRef
    source_family: str
    split: str
    original_instruction: str
    target: str
    color: str
    target_box: tuple[float, float, float, float]
    seed: int
    provenance: dict
    constraints: tuple[str, ...] = CONSTRAINTS

    def __post_init__(self):
        identifier(self.task_id)
        identifier(self.source_family)
        if self.split not in {"pilot", "comparison", "experience_validation", "reuse_test"}:
            raise ValueError("unregistered G2 split")
        if not isinstance(self.source, ArtifactRef) or not self.source.media_type.startswith("image/"):
            raise ValueError("G2 requires a hash-bound source image")
        for s in (self.original_instruction, self.target, self.color):
            if not isinstance(s, str) or not s.strip() or len(s) > 1500:
                raise ValueError("task instruction/target/color must be bounded text")
        x0, y0, x1, y1 = self.target_box
        if (any(type(x) not in (int, float) for x in self.target_box) or
                not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1) or
                (x1 - x0) * (y1 - y0) > .85):
            raise ValueError("target ROI must be normalized and leave preservation pixels")
        positive_int(self.seed, "seed", zero=True)
        if tuple(self.constraints) != CONSTRAINTS:
            raise ValueError("original evaluation constraints cannot be changed by a recipe")
        canonical_json(self.provenance)

    @classmethod
    def from_dict(cls, value):
        values = cls.read_fields(value)
        values["source"] = ArtifactRef.from_dict(values["source"])
        values["target_box"] = tuple(values["target_box"])
        values["constraints"] = tuple(values.get("constraints", CONSTRAINTS))
        return cls(**values)

    def to_dict(self):
        value = super().to_dict()
        value["source"] = self.source.to_dict()
        return value


def validate_recipe(recipe, allowed):
    if set(recipe) != set(RECIPE_FIELDS) or set(allowed) != set(RECIPE_FIELDS):
        raise ValueError("recipe must contain exactly the three approved execution fields")
    for name in RECIPE_FIELDS:
        if (not isinstance(allowed[name], list) or not allowed[name] or
                any(not isinstance(x, str) or len(x) > 800 for x in allowed[name]) or
                recipe[name] not in allowed[name]):
            raise ValueError(f"recipe value outside frozen allowlist: {name}")
    canonical_json(recipe)


def compile_prompt(task: G2Task, recipe: dict, allowed: dict) -> str:
    validate_recipe(recipe, allowed)
    # Color and original requirements come only from TaskSpec, never from R.
    return (f"Change only {recipe['target_description']} {recipe['spatial_expression']} "
            f"to {task.color}. {recipe['preservation_instruction']} "
            f"Original task requirements: {task.original_instruction}")


def validate_split_boundaries(tasks):
    ids, family_splits, hash_splits = set(), {}, {}
    for task in tasks:
        if task.task_id in ids:
            raise ValueError("duplicate task identity")
        ids.add(task.task_id)
        for registry, key in ((family_splits, task.source_family), (hash_splits, task.source.sha256)):
            if key in registry and registry[key] != task.split:
                raise ValueError("source family or source bytes cross a frozen split boundary")
            registry[key] = task.split


def validate_g2_config(config):
    required = {"schema_version", "tasks", "allowed_recipe_values", "initial_recipe", "inference",
                "evaluation", "budget", "models", "stores", "dataset_store", "method", "split_policy"}
    if set(config) != required or config["schema_version"] != "dataevolver.g2_config.v1":
        raise ValueError("unexpected G2 configuration fields")
    tasks = tuple(G2Task.from_dict(x) for x in config["tasks"])
    if not 1 <= len(tasks) <= 32:
        raise ValueError("G2 pilot task count must be bounded")
    validate_split_boundaries(tasks)
    for task in tasks:
        allowed = config["allowed_recipe_values"][task.task_id]
        validate_recipe(config["initial_recipe"][task.task_id], allowed)
    if set(config["initial_recipe"]) != {t.task_id for t in tasks} or set(config["allowed_recipe_values"]) != {t.task_id for t in tasks}:
        raise ValueError("task recipe membership changed")
    inference = config["inference"]
    if set(inference) != {"width", "height", "steps", "true_cfg_scale", "guidance_scale", "negative_prompt", "timeout_seconds"}:
        raise ValueError("unregistered editing parameters")
    if inference["width"] != 512 or inference["height"] != 512 or not 1 <= inference["steps"] <= 40:
        raise ValueError("first G2 pilot binds 512 px images and at most 40 steps")
    for name in ("true_cfg_scale", "guidance_scale"):
        if type(inference[name]) not in (int, float) or not 0 <= inference[name] <= 10:
            raise ValueError("invalid guidance scale")
    positive_int(inference["timeout_seconds"], "timeout_seconds")
    evaluation = config["evaluation"]
    if set(evaluation) != {"max_outside_mae", "min_target_mae", "review_max_tokens", "review_timeout_seconds", "calibration_status"}:
        raise ValueError("unregistered evaluation fields")
    if not (0 < evaluation["max_outside_mae"] < 1 and 0 <= evaluation["min_target_mae"] < 1):
        raise ValueError("invalid frozen pixel thresholds")
    if evaluation["calibration_status"] not in {"pilot_uncalibrated", "frozen_after_pilot"}:
        raise ValueError("evaluation calibration must be explicit")
    budget = config["budget"]
    if set(budget) != {"max_model_calls", "max_wall_seconds", "max_output_bytes", "max_edit_calls", "max_probe_calls"}:
        raise ValueError("unregistered budget fields")
    for k, v in budget.items():
        positive_int(v, k, zero=k == "max_probe_calls")
    if budget["max_edit_calls"] < len(tasks) or budget["max_model_calls"] < 2 * len(tasks):
        raise ValueError("budget cannot cover the declared pilot")
    if config["method"] != {"id": "g2-fixed-m-v1", "mutable": False, "knowledge_snapshot": None}:
        raise ValueError("Step 1 fixes M and does not claim knowledge reuse")
    if config["split_policy"] != {"selection_splits": ["pilot", "experience_validation"], "benefit_split": "reuse_test", "freeze_k_during_test": True}:
        raise ValueError("experience selection/test access boundary changed")
    canonical_json(config)
    return tasks
