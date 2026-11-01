"""Runtime implementation requests are evidence, never execution authority."""
from __future__ import annotations

from jsonschema import validate


CAPABILITY_REQUEST_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['problem', 'proposed_capability', 'implementation_kind',
                 'required_inputs', 'required_outputs', 'acceptance_checks', 'validation_design'],
    'properties': {
        **{key: {'type': 'string', 'minLength': 1, 'maxLength': 3000}
           for key in ('problem', 'proposed_capability', 'validation_design')},
        'implementation_kind': {'enum': ['registered_operator', 'planner_method', 'artifact_contract']},
        **{key: {'type': 'array', 'minItems': 1, 'maxItems': 12, 'uniqueItems': True,
                 'items': {'type': 'string', 'minLength': 1, 'maxLength': 600}}
           for key in ('required_inputs', 'required_outputs', 'acceptance_checks')},
    },
}


def validate_capability_request(value):
    validate(value, CAPABILITY_REQUEST_SCHEMA)
    return value
