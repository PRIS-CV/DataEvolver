"""Terminal review accounting cannot reinterpret abstention as a new request."""
import copy

import pytest

from dataevolver.harness.production_feedback_request import _validate_review_terminal


def state():
    return {'state': 'ABSTAIN', 'terminal_reason': 'RSI_AGENT_ABSTAINED',
        'planner_calls': 1, 'operator_calls': 1, 'experiments_used': 0, 'data': {'pending_call': None}}


def test_terminal_abstention_is_auditable_without_implementation_authority():
    value = state(); before = copy.deepcopy(value)
    assert not _validate_review_terminal(value, {'action': 'ABSTAIN'}, False)
    assert value == before
    with pytest.raises(ValueError): _validate_review_terminal(value, {'action': 'ABSTAIN'}, True)
    value['terminal_reason'] = 'CAPABILITY_REQUESTED'
    assert _validate_review_terminal(value, {'action': 'REQUEST_CAPABILITY'}, True)


@pytest.mark.parametrize('change', ['live', 'reason', 'planner', 'operator', 'image', 'pending', 'decision'])
def test_incomplete_or_mismatched_review_cannot_be_read_as_terminal(change):
    value = state(); decision = {'action': 'ABSTAIN'}
    if change == 'live': value['state'] = 'PROPOSE'
    if change == 'reason': value['terminal_reason'] = 'BUDGET_EXHAUSTED'
    if change == 'planner': value['planner_calls'] = 2
    if change == 'operator': value['operator_calls'] = 0
    if change == 'image': value['experiments_used'] = 1
    if change == 'pending': value['data']['pending_call'] = {'call_id': 'unfinished'}
    if change == 'decision': decision['action'] = 'REQUEST_CAPABILITY'
    with pytest.raises(ValueError): _validate_review_terminal(value, decision, False)
