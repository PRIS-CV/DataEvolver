"""Readiness-only evidence/action boundaries; fixture requests are not RSI gain."""
import json
import os
import platform

import pytest

from dataevolver.dataset.contracts import file_sha256
from dataevolver.harness.operator_readiness import CandidateReadinessDomain, ControllerCandidateChecksAuthority
from dataevolver.harness.runtime_rsi import validate_decision
from dataevolver.testing.test_operator_candidate_checks import setup
from dataevolver.testing.test_runtime_capability_requests import decision

pytestmark = pytest.mark.skipif(platform.system() != 'Linux' or os.geteuid() != 0,
                                reason='fixture checks use actual Linux confinement')


class FixtureChecksAuthority:
    def __init__(self,operator,call):self.operator,self.call,self.changed=operator,call,False
    def identity(self):return {'fixture_identity_changed':self.changed}
    def read(self):
        path=self.operator._path(self.call)
        return json.loads(path.read_text()),{'package_sha256':self.operator.package.compute_hash(),
                                           'report_sha256':file_sha256(path)}


def domain_fixture(tmp_path):
    operator,call=setup(tmp_path);operator.execute(call)
    authority=FixtureChecksAuthority(operator,call)
    domain=CandidateReadinessDomain(plan=operator.plan,package=operator.package,store=operator.store,
        request_authority=operator.request_authority,checks_authority=authority)
    return domain,authority


def test_context_is_only_a_bound_readiness_review(tmp_path):
    domain,_=domain_fixture(tmp_path)
    context=domain.context(None)
    assert context['allowed_actions']==['REQUEST_CAPABILITY','ABSTAIN']
    assert context['allowed_patches']==context['allowed_observations']==[]
    assert context['capability_request_format']['layer']=='M'
    assert context['capability_request_format']['patch_kind']=='capability_gap'
    assert 'required_inputs' in context['capability_request_format']['request_schema']['required']
    assert context['evidence']['candidate_source']['content'].startswith('def resolve')
    assert context['evidence']['isolated_behavior_checks']['engineering_qualification']=='NOT_ASSESSED'
    with pytest.raises(ValueError):domain.compile_intervention({},None)
    with pytest.raises(ValueError):domain.compile_observation({},None)


def test_request_must_cite_actual_checks_and_cannot_grant_execution(tmp_path):
    domain,_=domain_fixture(tmp_path)
    d=decision();d['evidence_ids']=['candidate_source']
    with pytest.raises(ValueError):domain.compile_capability_request(d,None)
    d['evidence_ids']=['candidate_source','isolated_behavior_checks']
    validate_decision(d,domain.context(None))
    result=domain.compile_capability_request(d,None)
    assert result['decision']==d and result['proposal_actor']=='runtime_agent'
    assert result['production_effects']==0 and not result['implementation_authorized_by_request']
    assert not result['method_improvement_verified'] and not result['knowledge_promoted']


def test_authority_and_captured_execution_drift_rejected(tmp_path):
    domain,authority=domain_fixture(tmp_path)
    authority.changed=True
    with pytest.raises(ValueError,match='authority changed'):domain.context(None)
    authority.changed=False
    report=json.loads(authority.operator._path(authority.call).read_text())
    path=next(iter(report['execution_artifacts']))
    with open(path,'w') as stream:stream.write('{}')
    with pytest.raises(ValueError,match='execution changed'):domain.context(None)


def test_readiness_cannot_turn_a_check_into_a_production_experiment(tmp_path):
    domain,_=domain_fixture(tmp_path)
    d=decision();d['action']='RUN_EXPERIMENT';d['evidence_ids']=['isolated_behavior_checks']
    with pytest.raises(ValueError,match='frozen execution mode'):validate_decision(d,domain.context(None))


def test_unapproved_validator_is_not_a_controller_evidence_authority():
    with pytest.raises(ValueError,match='approved isolated-check operator'):
        ControllerCandidateChecksAuthority(None,object(),'fake-call')
