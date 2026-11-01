"""Registered independent synthetic contract checks; production stays gated."""
import json
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from ..operator_contracts import OperatorContract
from .operator_candidate_checks import CandidateChecksOperator
from .operator_contract_qualification import CandidateContractQualificationOperator


class IndependentContractQualificationOperator(CandidateContractQualificationOperator):
    def __init__(self, *, engineering_authority, **kwargs):
        self.engineering_authority=engineering_authority
        self.engineering_identity=engineering_authority.identity()
        super().__init__(**kwargs)
        self.spec=replace(self.spec,operator_id='evolution.qualify_independent_contract',
            source_path=str(Path(__file__).resolve()),code_revision=payload_sha256({
                'contract_qualifier':self.spec.code_revision,'adapter':file_sha256(Path(__file__)),
                'engineering_authority':self.engineering_identity}))
        self.operator_contract=OperatorContract(self.spec.operator_id,'1',self.spec.sha256(),'evaluation',
            {'candidate_package':{'type':'object'},'independent_protocol':{'type':'object'}},
            {'independent_contract_checks':{'type':'object'}},
            {'type':'object','additionalProperties':False,'properties':{}},
            ('json','source_code'),'pure_data','verify_then_reuse',(
                'Two new synthetic source families must be isolated from disclosed development evidence.',
                'Engineering must already pass for the same immutable candidate package.',
                'This scope does not qualify production images, adopt code, promote K or prove method benefit.'))

    def _engineering(self):
        if self.engineering_authority.identity()!=self.engineering_identity:
            raise ValueError('engineering authority changed')
        e=self.engineering_authority.read()
        if (e['scope']!='synthetic_authority_contract_engineering'
                or e['pure_data_engineering_qualification']!='PASSED'
                or e.get('engineering_evaluator_accepted') is not True
                or e['origin']['package_sha256']!=self.package.compute_hash()
                or e['production_adoption_allowed'] is not False):
            raise ValueError('same-package committed engineering qualification must pass first')
        return e

    def _check_protocol(self):
        p=self.protocol
        extras={'engineering_origin','family_ids'}
        if (p.get('schema_version')!='dataevolver.independent_contract_qualification.v1'
                or p.get('qualification_scope')!='synthetic_authority_contract_independent'
                or not isinstance(p.get('family_ids'),list) or len(p['family_ids'])<2
                or len(set(p['family_ids']))!=len(p['family_ids'])):
            raise ValueError('independent qualification needs two declared new source families')
        eng=self._engineering()
        binding={'authority':self.engineering_identity,'report_sha256':eng['report_sha256'],
                 'package_sha256':eng['origin']['package_sha256']}
        if p['engineering_origin']!=binding:
            raise ValueError('predeclared engineering origin changed')
        base={k:v for k,v in p.items() if k not in extras}
        base.update(schema_version='dataevolver.candidate_contract_qualification.v1',
                    qualification_scope='synthetic_authority_contract_engineering')
        self.protocol=base
        try:super()._check_protocol()
        finally:self.protocol=p
        if set(p['family_ids']) & set(eng['fixture_family_ids']):
            raise ValueError('development families cannot serve as independent sources')
        inputs=[case['input']['sha256'] for case in p['cases']]
        if len(set(inputs))!=len(inputs) or set(inputs)&set(eng['input_sha256s']):
            raise ValueError('independent inputs must be distinct and unseen in development')
        kinds={family:set() for family in p['family_ids']}
        for case in p['cases']:
            context,expected,_=self._case_data(case)
            if context.fixture_family_id not in kinds:
                raise ValueError('undeclared independent source family')
            kinds[context.fixture_family_id].add(case['kind'])
        if any(value!={'positive','negative'} for value in kinds.values()):
            raise ValueError('each independent family needs authenticated positive and adversarial checks')
        if (self.budget.resource_deadline_epoch!=eng['resource_deadline_epoch']
                or any(self.budget.prior_usage[k]<v for k,v in eng['lifecycle_total'].items())):
            raise ValueError('independent qualification must retain all prior usage and deadline')

    def _report(self,call):
        report=super()._report(call)
        passed=report['all_checks_passed']
        report.update(pure_data_engineering_qualification='PASSED',
            pure_data_independent_qualification='PASSED' if passed else 'FAILED',
            engineering_origin=self.protocol['engineering_origin'],
            independent_family_ids=self.protocol['family_ids'],source_families_isolated=True,
            production_adoption_allowed=False,independent_qualification='NOT_ASSESSED')
        return report

    def _receipt(self,call,path,reused):
        receipt=CandidateChecksOperator._receipt(self,call,path,reused)
        report=json.loads(path.read_text())
        return replace(receipt,stage_result={**receipt.stage_result,'metrics':{
            **receipt.stage_result['metrics'],'independent_contract_checks_passed':float(
                report['pure_data_independent_qualification']=='PASSED')}})
