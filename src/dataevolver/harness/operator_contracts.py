"""Discoverable operator declarations; declarations never establish qualification.

The existing Controller owns execution and the existing operator verifies effects.
This contract adds bounded parameters, named artifact schemas and explicit effect
and recovery semantics without importing a candidate or granting new authority.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import ClassVar

from jsonschema import Draft202012Validator

from dataevolver.dataset.contracts import Contract, canonical_json, identifier


def _check_schema(value):
    if not isinstance(value, dict):
        raise ValueError("operator schemas must be JSON objects")
    canonical_json(value)
    Draft202012Validator.check_schema(value)
    def check_refs(item):
        if isinstance(item, dict):
            for key, nested in item.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                    not isinstance(nested, str) or not nested.startswith("#")
                ):
                    raise ValueError("operator schemas cannot fetch external references")
                check_refs(nested)
        elif isinstance(item, list):
            for nested in item:
                check_refs(nested)
    check_refs(value)


@dataclass(frozen=True)
class OperatorContract(Contract):
    schema: ClassVar[str] = "dataevolver.operator_contract.v1"
    operator_id: str
    version: str
    spec_sha256: str
    category: str
    input_schemas: dict[str, dict]
    output_schemas: dict[str, dict]
    parameter_schema: dict
    modalities: tuple[str, ...]
    effect_semantics: str
    recovery_semantics: str
    limitations: tuple[str, ...]

    def __post_init__(self):
        identifier(self.operator_id, "operator_id")
        identifier(self.version, "operator version")
        if (len(self.spec_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in self.spec_sha256)):
            raise ValueError("contract must bind the exact OperatorSpec SHA-256")
        if self.category not in {"basic", "evolution", "evaluation"}:
            raise ValueError("unknown operator category")
        if self.effect_semantics not in {"pure_data", "model", "external", "composite"}:
            raise ValueError("unknown effect semantics")
        if self.recovery_semantics not in {"verify_then_reuse", "verify_then_retry", "reconcile_only"}:
            raise ValueError("unknown recovery semantics")
        for name in ("input_schemas", "output_schemas"):
            ports = getattr(self, name)
            if not isinstance(ports, dict) or (name == "output_schemas" and not ports):
                raise ValueError("named artifact ports are required")
            for port, value in ports.items():
                identifier(port, "artifact port")
                _check_schema(value)
            object.__setattr__(self, name, copy.deepcopy(ports))
        _check_schema(self.parameter_schema)
        if (self.parameter_schema.get("type") != "object" or
                self.parameter_schema.get("additionalProperties") is not False):
            raise ValueError("operator parameters need a closed object schema")
        object.__setattr__(self, "parameter_schema", copy.deepcopy(self.parameter_schema))
        for name in ("modalities", "limitations"):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list)) or not values or
                    any(not isinstance(x, str) or not x.strip() for x in values) or
                    len(values) != len(set(values))):
                raise ValueError(f"{name} must contain distinct nonempty declarations")
            object.__setattr__(self, name, tuple(values))

    def validate_parameters(self, value):
        canonical_json(value)
        Draft202012Validator(self.parameter_schema).validate(value)

    def bound_to(self, spec):
        if ((spec.operator_id, spec.version) != (self.operator_id, self.version) or
                spec.sha256() != self.spec_sha256):
            raise ValueError("operator declaration does not match the registered implementation")
        if self.recovery_semantics == "verify_then_retry" and not spec.retry_safe:
            raise ValueError("retry declaration cannot grant retry authority")
        return self
