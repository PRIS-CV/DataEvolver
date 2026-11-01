"""Read an existing native training result into a next-run feedback context."""
import json

from dataevolver.dataset.contracts import ArtifactRef
from ..artifact_operator import ArtifactWorkloadOperator
from ..training_feedback import ExternalTrainingRunAdapter


class TrainingFeedbackInspectionOperator(ArtifactWorkloadOperator):
    def __init__(self, *, adapter, job, receipt, output_root):
        if not isinstance(adapter, ExternalTrainingRunAdapter) or not isinstance(receipt, ArtifactRef):
            raise TypeError('feedback inspection needs the registered training adapter and exact receipt')
        if not adapter._path(job).is_file():
            raise ValueError('feedback cannot register another external training job')
        if job.evaluation_split=='sealed':
            raise ValueError('sealed feedback cannot enter a next-run context')
        self.adapter,self.job,self.receipt=adapter,job,receipt
        super().__init__(operator_id='data.inspect_training_feedback',output_root=output_root,
            configuration={'job':job.to_dict(),'receipt':receipt.to_dict(),
                'stores':{k:str(v) for k,v in adapter.store.roots.items()},'training_registry':str(adapter.root)})
        self.spec=self.spec.with_updates(allowed_write_roots=(*self.spec.allowed_write_roots,str(adapter.root)))
        self._declare_contract()

    def _report(self):
        feedback=self.adapter.collect(self.job,self.receipt)
        context=self.adapter.next_data_plan_context(self.job,self.receipt)
        native=json.loads(self.adapter.store.resolve(self.receipt).read_text())
        return {'job_sha256':self.job.compute_hash(),'feedback':feedback.to_dict(),'next_run_context':context,
            'native_exit_code':native['exit_code'],'dataset_frozen':True,
            'next_run_required':True,'new_training_jobs':0,'training_benefit_verified':False}

    def workload(self,call):return self._report()

    def validate_report(self,call,report):
        if report!=self._report():raise ValueError('native training feedback or its next-run binding changed')

    def result(self,call,report):
        value=super().result(call,report);value['metrics']['feedback_verified']=1.
        return value
