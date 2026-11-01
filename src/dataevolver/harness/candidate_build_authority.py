"""Read an actual model-built package from its committed Controller effect."""
import json

from dataevolver.dataset.contracts import payload_sha256
from dataevolver.adapters.model_contracts import ModelReceipt
from .artifact_operator import CommittedEffectAuthority
from .adapters.operator_candidate_build import CandidateBuildOperator
from .operator_development import CandidatePackage


class CommittedCandidateBuildAuthority(CommittedEffectAuthority):
    def __init__(self, controller, operator):
        if not isinstance(operator, CandidateBuildOperator):
            raise TypeError('candidate origin requires the registered model builder')
        super().__init__(controller, operator)

    def read(self):
        call, receipt = self.read_effect()
        operator = self.operator
        work = operator._validate(call)
        effect = json.loads((work/'candidate_build.json').read_text())
        package = CandidatePackage.from_dict(json.loads((work/'candidate-package.json').read_text()))
        wrapper = json.loads((operator.gateway.root/(effect['request']['request_id']+'.receipt.json')).read_text())
        model = ModelReceipt.from_dict(wrapper['receipt'])
        if (receipt.return_code != 0 or effect['status'] != 'BUILT_UNQUALIFIED'
                or package.compute_hash() != effect['package_sha256']
                or wrapper['sha256'] != payload_sha256(wrapper['receipt'])
                or model.without_reuse_hash() != effect['model_receipt_sha256']
                or model.backend == 'mock' or not effect['native_inventory']
                or effect['model_usage']['model_calls'] != 1):
            raise ValueError('candidate has no real committed model-build origin')
        return {'package': package.to_dict(), 'origin': self.identity(),
                'receipt_sha256': receipt.sha256(), 'model_receipt_sha256': model.without_reuse_hash(),
                'request': effect['request'], 'native_inventory': effect['native_inventory'],
                'model_usage': effect['model_usage']}
