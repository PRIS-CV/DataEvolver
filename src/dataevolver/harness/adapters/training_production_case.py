"""Produce the baseline side of a production case from real training receipts.

This registered workload consumes an original committed training trial, its
Gateway journals, real media and independent loading results. It never submits
an image or executes a candidate. The candidate side remains explicitly absent;
these baseline reports cannot qualify or reactivate an operator version.
"""
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.dataset.supervision import artifact_closure
from dataevolver.dataset.training_pipeline import QUALITY_CHECKS
from ..models import (canonical_json, EffectVerification, EvidenceStatus, ExpectedOutputSpec,
                      OperatorReceipt, OperatorSpec, RecoveryResult, RecoveryStatus, StageStatus)
from ..operator_contracts import OperatorContract
from ..operator_development import QualificationProtocol
from ..production_qualification import EVIDENCE_ROLES
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import CampaignLease, atomic_write_json
from ..training_workflow_delivery import capture_execution_audit, require_visual_acceptance, verify_trial_bundle


DECODE_SCRIPT = '''import hashlib,json,sys
from pathlib import Path
from PIL import Image
p=Path(sys.argv[1]);m=json.loads(p.read_text());rows=[]
for entry in m:
 f=p.parent/entry['path']
 if f.is_symlink() or not f.resolve().is_relative_to(p.parent.resolve()):raise ValueError('media path escaped')
 if hashlib.sha256(f.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('media bytes changed')
 with Image.open(f) as image:
  image.load();pixels=image.convert('RGBA')
  rows.append({'path':entry['path'],'sha256':entry['sha256'],'size':list(image.size),
               'rgba_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()})
print(json.dumps({'isolated_python':bool(sys.flags.isolated),'decoded_media':rows}))
'''


class TrainingProductionCaseOperator:
    def __init__(self, *, plan, package, store, request_authority, trial_authority,
                 family_id, stage, output_root, evidence_store, matched_context, python,
                 audit_config_roots=(), max_output_bytes=536870912):
        self.plan, self.package, self.stage, self.family_id = plan, package, stage, family_id
        self.store = ArtifactStore(store.roots)
        self.request_authority, self.trial_authority = request_authority, trial_authority
        self.request_identity, self.trial_identity = request_authority.identity(), trial_authority.identity()
        self.root = Path(output_root).resolve()
        self.evidence_store, self.python = evidence_store, Path(python).absolute()
        self.audit_config_roots = tuple(str(Path(p).resolve()) for p in audit_config_roots)
        if (type(max_output_bytes) is not int or not 1048576 <= max_output_bytes <= 1073741824
                or self.store.roots.get(evidence_store) != self.root
                or evidence_store in {r['store'] for r in package.files}):
            raise ValueError('a bounded host-owned evidence store is required')
        self.max_output_bytes = max_output_bytes
        if package.development_plan_sha256 != plan.compute_hash():
            raise ValueError('candidate package belongs to a different development plan')
        for name, root in self.store.roots.items():
            if name != evidence_store and (root.is_relative_to(self.root) or self.root.is_relative_to(root)):
                raise ValueError('production evidence overlaps a protected input store')
        self.protocol = QualificationProtocol.from_dict(json.loads(self.store.resolve(
            ArtifactRef.from_dict(plan.qualification_protocol)).read_text()))
        matches = [s for s in self.protocol.sources if s['family_id']==family_id]
        if stage != 'engineering' or family_id not in self.protocol.families(stage) or len(matches)!=1:
            raise ValueError('baseline receipt capture requires an original engineering family')
        self.source = matches[0]['artifact']
        self.matched_context = ArtifactRef.from_dict(matched_context).to_dict()
        if self.matched_context['sha256'] != self.protocol.matched_context_sha256:
            raise ValueError('original matched context changed')
        self.binding = {'mode':'registered_training_baseline_receipt', 'stage':stage,
            'family_id':family_id, 'source':self.source, 'package_sha256':package.compute_hash(),
            'design_protocol_sha256':self.protocol.compute_hash(),
            'matched_context_sha256':self.protocol.matched_context_sha256,
            'execution_contract_sha256':None, 'runtime':{'trust_namespace':'registered_training_baseline_receipt'}}
        self.spec = OperatorSpec('data.training_production_case','1',retry_safe=True,
            source_path=str(Path(__file__).resolve()),timeout_seconds=900,
            code_revision=payload_sha256({'adapter':file_sha256(Path(__file__)),
                'authority':self.trial_identity,'request_authority':self.request_identity,
                'delivery':file_sha256(Path(__file__).parents[1]/'training_workflow_delivery.py'),
                'binding':self.binding,'stores':{k:str(v) for k,v in self.store.roots.items()},
                'python':str(self.python),'python_sha256':file_sha256(self.python),
                'audit_config_roots':self.audit_config_roots,'max_output_bytes':max_output_bytes}),
            allowed_write_roots=(str(self.root),),input_artifact_refs=(self.source,),
            expected_output_specs=(ExpectedOutputSpec('production-case.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id,'1',self.spec.sha256(),'evaluation',
            {'committed_training_trial':{'type':'object'}},{'production_case':{'type':'object'}},
            {'type':'object','additionalProperties':False,'required':['binding_sha256'],
             'properties':{'binding_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'}}},
            ('json','image'), 'composite','verify_then_reuse',(
                'Only the receipt-backed baseline side is enabled; no models or candidate programs are executed.',
                'Original five model checks and runtime Agent outcomes remain unchanged, including failures.',
                'A missing production candidate invocation keeps every qualification check unpassed.',
                'Historical native costs are reconciled separately from this new CPU workload usage.'))

    def parameters(self):
        return {'binding_sha256':payload_sha256(self.binding)}

    def _path(self, call):
        self.operator_contract.validate_parameters(call.parameters)
        work = Path(call.work_dir)
        if ((call.operator_id,call.operator_version)!=(self.spec.operator_id,'1')
                or call.parameters!=self.parameters() or work.resolve()==self.root
                or tuple(call.input_artifact_refs)!=(self.source,)
                or not work.resolve().is_relative_to(self.root)
                or any(p.is_symlink() for p in (work,*work.parents))):
            raise ValueError('production workload identity or owner changed')
        if work.exists() and any(p.is_symlink() or (p.is_file() and p.stat().st_nlink!=1)
                                 for p in work.rglob('*')):
            raise ValueError('production evidence cannot contain linked files')
        return work/'production-case.json'

    def _source(self):
        if (self.request_authority.identity()!=self.request_identity
                or self.trial_authority.identity()!=self.trial_identity):
            raise ValueError('production workload source authority changed')
        self.request_authority.verify(self.plan,self.store)
        request=json.loads(self.store.resolve(ArtifactRef.from_dict(self.plan.capability_request)).read_text())
        source=json.loads(self.store.resolve(ArtifactRef.from_dict(self.source)).read_text())
        if (source.get('kind')!='existing-failed-source-regression'
                or source.get('request_sha256')!=self.plan.capability_request['sha256']):
            raise ValueError('this workload must bind the original runtime-request regression')
        value=self.trial_authority.read();trial=value['trial']
        if (trial['source']['sha256']!=request['source']['sha256']
                or trial['candidate']['sha256']!=request['rejected_candidate']['sha256']
                or any(str(self.store.roots.get(k))!=v for k,v in value['store_roots'].items())):
            raise ValueError('training receipt is not the exact requested source/candidate')
        for raw in self.package.files:
            self.store.resolve(ArtifactRef.from_dict(raw))
        self.store.resolve(ArtifactRef.from_dict(self.matched_context))
        return value

    def _ref(self,path,media_type='application/json'):
        return ArtifactRef(self.evidence_store,Path(path).relative_to(self.root).as_posix(),file_sha256(path),media_type).to_dict()

    def _scoped_copies(self, work, origin):
        size=sum(Path(row['matches'][0]['path']).stat().st_size for row in origin['scoped_reference_closure'])
        size+=sum(f.stat().st_size for f in work.rglob('*') if f.is_file())
        if size>self.max_output_bytes:
            raise ValueError('historical closure exceeds its finite output budget')
        folder=work/'historical-reference-closure';folder.mkdir(exist_ok=True)
        rows=[]
        for index,row in enumerate(origin['scoped_reference_closure']):
            matches=row['matches']
            if not matches:raise ValueError('historical reference lost its original scope')
            for match in matches:
                if file_sha256(Path(match['path']))!=row['sha256']:
                    raise ValueError('historical scoped artifact changed')
            source=Path(matches[0]['path']);target=folder/(str(index)+source.suffix)
            if not target.exists():shutil.copy2(source,target)
            if file_sha256(target)!=row['sha256']:raise ValueError('historical scoped copy changed')
            # Historical JSON keeps its original namespace and exact bytes.
            # Its recursively verified members are explicit separate parts.
            media_type=('application/vnd.dataevolver.scoped-json' if row['media_type']=='application/json'
                        else row['media_type'])
            rows.append({**row,'relocated_artifact':self._ref(target,media_type)})
        return rows

    @staticmethod
    def _save(path,value):
        if path.exists():
            if json.loads(path.read_text())!=value:
                raise ValueError('completed production evidence changed')
        else:
            atomic_write_json(path,value)

    def _load_media(self, work, trial):
        total=sum(self.store.resolve(ArtifactRef.from_dict(item['artifact'])).stat().st_size for item in trial['media'])
        if total > self.max_output_bytes:
            raise ValueError('relocated media exceeds its finite output budget')
        folder=work/'relocated-media';folder.mkdir(exist_ok=True)
        rows=[]
        for index,item in enumerate(trial['media']):
            ref=ArtifactRef.from_dict(item['artifact']);source=self.store.resolve(ref)
            target=folder/(str(index)+source.suffix)
            if not target.exists():shutil.copy2(source,target)
            if file_sha256(target)!=ref.sha256:raise ValueError('relocated media changed')
            rows.append({'path':target.name,'sha256':ref.sha256})
        manifest=folder/'manifest.json';self._save(manifest,rows)
        output=work/'independent-media-decode.json'
        if not output.exists():
            process=subprocess.run([str(self.python),'-I','-c',DECODE_SCRIPT,str(manifest)],
                cwd=folder,capture_output=True,text=True,timeout=120,
                env={'LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
            if process.returncode:raise ValueError('independent media loading failed: '+process.stderr[-1000:])
            value=json.loads(process.stdout)
            if value.get('isolated_python') is not True or len(value['decoded_media'])!=len(rows):
                raise ValueError('independent decoder did not read every real media file')
            atomic_write_json(output,value)
        value=json.loads(output.read_text())
        if (value.get('isolated_python') is not True
                or [(row['path'],row['sha256']) for row in value['decoded_media']]
                   !=[(row['path'],row['sha256']) for row in rows]
                or any(len(row['rgba_sha256'])!=64 or len(row['size'])!=2 for row in value['decoded_media'])):
            raise ValueError('independent decoder evidence changed')
        return value

    def _assemble(self, call):
        path=self._path(call);work=path.parent
        origin=self._source();trial=origin['trial']
        self._save(work/'original-training-authority.json',origin)
        scoped=self._scoped_copies(work,origin)
        if len(trial.get('media',[]))<6 or 'model_review' not in trial:
            raise ValueError('a complete training-media receipt is required, including failed quality')
        if (trial['media'][0]['artifact']['sha256']!=trial['source']['sha256']
                or trial['media'][1]['artifact']['sha256']!=trial['candidate']['sha256']):
            raise ValueError('trial media do not match the original source and candidate')
        # This independently checks the real composition operation, rather than
        # treating its visual appearance as authority or a passing quality gate.
        from PIL import Image
        def image(index,mode):
            with Image.open(self.store.resolve(ArtifactRef.from_dict(trial['media'][index]['artifact']))) as value:
                return value.convert(mode)
        source,candidate,raw,blend=image(0,'RGB'),image(1,'RGB'),image(2,'RGB'),image(4,'L')
        composed=Image.composite(raw,source,blend)
        composition_exact=composed.size==candidate.size and composed.tobytes()==candidate.tobytes()
        decoded=self._load_media(work,trial)
        load={'baseline_dataset_accepted':trial.get('qualified') is True,
              'decoded_media':decoded,'loaded_dataset_records':0,'eligible_for_delivery':False}
        if trial.get('qualified') is True:
            require_visual_acceptance(trial)
            proof=verify_trial_bundle(trial,work/'independent-dataset',receipt_roots=(origin['config'],))
            load.update(dataset_proof=proof,loaded_dataset_records=proof['loaded_records'],eligible_for_delivery=True)
        audit=capture_execution_audit((origin['config'],self.request_authority.config_root),
            work/'native-audit',config_roots=self.audit_config_roots,
            max_output_bytes=self.max_output_bytes-sum(f.stat().st_size for f in work.rglob('*') if f.is_file()))
        values={
            'input_closure':{'source':self.source,'matched_context':self.matched_context,
                'request':self.plan.capability_request,'trial_sha256':origin['trial_sha256'],
                'original_training_authority':self._ref(work/'original-training-authority.json',
                    'application/vnd.dataevolver.scoped-training-origin+json'),
                'historical_scoped_reference_closure':scoped,
                'source_image':trial['source'],'candidate_image':trial['candidate'],'media':trial['media'],
                'original_controller':{'state':origin['state'],'frozen':origin['frozen'],'replay':origin['replay'],
                    'receipt_sha256':origin['receipt_sha256']}},
            'matched_comparison':{'status':'BASELINE_ONLY','candidate_execution_performed':False,
                'matched_context':self.matched_context,'source_image':trial['source'],
                'baseline_candidate':trial['candidate'],'new_candidate_result':None,
                'original_models_sha256':payload_sha256(origin['contract'].get('models',{})),
                'deployment_metadata_comparison':'UNASSESSED_BASELINE_ONLY',
                'method_improvement_verified':False,'knowledge_promoted':False},
            'quality_checks':{'model_review':trial['model_review'],
                'required_checks':list(QUALITY_CHECKS),'runtime_agent_review':trial['runtime_visual_review'],
                'runtime_visual_actor':trial['runtime_visual_actor'],'baseline_qualified':trial['qualified'],
                'human_review_performed':False,'composition_exact':composition_exact,
                'new_candidate_quality_assessed':False},
            'cost_ledger':{'historical_training_usage':origin['historical_usage'],
                'historical_union_usage':audit['usage'],'historical_attempts_recharged':False,
                'new_model_calls':0,'new_image_calls':0,'new_planner_calls':0,'monetary_cost':None},
            'native_audit':{'audit':self._ref(work/'native-audit/audit.json'),
                'source_journals_unchanged':audit['source_journals_unchanged'],
                'model_calls_submitted_by_audit':audit['model_calls_submitted_by_audit']},
            'independent_load':load}
        artifacts={}
        for name,value in values.items():
            f=work/(name+'.json');self._save(f,value);artifacts[name]=self._ref(f)
        return origin,artifacts

    def _case(self, path):
        report=json.loads(path.read_text())
        return {'binding':self.binding,
            'checks':{name:False for name in self.protocol.stages[self.stage]['check_ids']},
            'artifacts':report['artifacts'],'usage':report['usage'],
            'candidate_invocation':{'actual_execution':False,'package_sha256':self.package.compute_hash(),
                'implementation_sha256':self.package.implementation['sha256'],'execution_artifact':None,
                'reason':'NO_BOUND_PRODUCTION_CANDIDATE'}}

    def _receipt(self, call, path, reused=False):
        value=json.loads(path.read_text())
        return OperatorReceipt(call.call_id,self.spec.operator_id,'1',StageStatus.SUCCESS,
            value['started_at'],value['finished_at'],return_code=0,reused=reused,effect_hash=file_sha256(path),
            output_artifacts=({'path':str(path),'sha256':file_sha256(path)},),
            stage_result={'operator_id':self.spec.operator_id,'operator_version':'1','status':'SUCCESS',
                'metrics':{'production_qualification_passed':0.0,'production_baseline_captured':1.0},
                'hard_constraints':{'artifact_valid':True},'production_case':self._case(path),
                'evidence_refs':[str(path)]})

    def verify_effect(self, call, receipt):
        try:
            path=self._path(call)
            if not path.exists():return EffectVerification(EvidenceStatus.UNOBSERVED,'production case absent')
            value=json.loads(path.read_text());origin=self._source()
            if (value['call_sha256']!=payload_sha256(call.to_dict()) or value['binding']!=self.binding
                    or value['original_trial_sha256']!=origin['trial_sha256']
                    or value['production_qualification_passed'] is not False
                    or value['new_candidate_execution'] is not False
                    or set(value['artifacts']) != EVIDENCE_ROLES):
                raise ValueError('production baseline owner or original trial changed')
            if json.loads((self.root/'single-case.json').read_text())!={'call_sha256':value['call_sha256']}:
                raise ValueError('production baseline durable owner changed')
            files={str(f) for f in path.parent.rglob('*') if f.is_file()
                   and '.receipts' not in f.parts and f!=path}
            if files!=set(value['files']):
                raise ValueError('production baseline evidence inventory changed')
            usage=value['usage']
            if (usage['operator_calls']!=1 or any(usage[k]!=0 for k in ('model_calls','planner_calls','image_experiments'))
                    or usage['output_bytes']!=sum(Path(f).stat().st_size for f in files)+path.stat().st_size
                         +(self.root/'single-case.json').stat().st_size
                    or usage['output_bytes']>self.max_output_bytes):
                raise ValueError('production baseline usage or output bound changed')
            hashes={str(path):file_sha256(path),**origin['artifact_hashes']}
            for name,digest in value['files'].items():
                f=Path(name)
                if not f.resolve().is_relative_to(path.parent) or file_sha256(f)!=digest:
                    raise ValueError('production baseline evidence changed')
                hashes[name]=digest
            quality=json.loads(self.store.resolve(ArtifactRef.from_dict(value['artifacts']['quality_checks'])).read_text())
            if quality['model_review']!=origin['trial']['model_review'] or quality['runtime_agent_review']!=origin['trial']['runtime_visual_review']:
                raise ValueError('original failed quality cannot be replaced')
            for ref in artifact_closure(list(value['artifacts'].values())+[self.source,self.matched_context],self.store):
                f=self.store.resolve(ref);hashes[str(f)]=ref.sha256
            if receipt is not None and (receipt.effect_hash!=file_sha256(path)
                    or receipt.stage_result!=self._receipt(call,path).stage_result):
                raise ValueError('production baseline receipt changed')
            return EffectVerification(EvidenceStatus.VERIFIED,'real training receipts/media/native and baseline evidence verified',
                                      artifact_hashes=hashes)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.SubprocessError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED,str(exc))

    def recover(self, call):
        path=self._path(call)
        if not path.exists():return RecoveryResult(RecoveryStatus.RETRY_SAFE,reason='read-only CPU baseline capture may resume')
        checked=self.verify_effect(call,_load_receipt(_receipt_path(call)))
        if checked.status is not EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.AMBIGUOUS,reason=checked.reason)
        return RecoveryResult(RecoveryStatus.RECOVERED,_write_receipt(_receipt_path(call),self._receipt(call,path,True)))

    def execute(self, call):
        path=self._path(call);self.root.mkdir(parents=True,exist_ok=True)
        with CampaignLease(self.root/'.training-production-case.lease',ttl_seconds=3600):
            recovered=self.recover(call)
            if recovered.status is RecoveryStatus.RECOVERED:return recovered.receipt
            if recovered.status is not RecoveryStatus.RETRY_SAFE:raise ValueError(recovered.reason)
            owner=self.root/'single-case.json';binding=payload_sha256(call.to_dict())
            self._save(owner,{'call_sha256':binding})
            intent=path.parent/'intent.json'
            if not intent.exists():atomic_write_json(intent,{'call_sha256':binding,
                'started_at':datetime.now(timezone.utc).isoformat()})
            data=json.loads(intent.read_text())
            if data['call_sha256']!=binding:raise ValueError('production baseline owner changed')
            origin,artifacts=self._assemble(call)
            files={str(f):file_sha256(f) for f in path.parent.rglob('*')
                   if f.is_file() and '.receipts' not in f.parts and f!=path}
            ended=datetime.now(timezone.utc).isoformat()
            usage={'operator_calls':1,'planner_calls':0,'model_calls':0,'image_experiments':0,
                'wall_seconds':math.ceil((datetime.fromisoformat(ended)-datetime.fromisoformat(data['started_at'])).total_seconds()),
                'output_bytes':0}
            report={'call_sha256':binding,'binding':self.binding,'original_trial_sha256':origin['trial_sha256'],
                'started_at':data['started_at'],'finished_at':ended,'artifacts':artifacts,'files':files,'usage':usage,
                'production_qualification_passed':False,'new_candidate_execution':False}
            actual=sum(Path(f).stat().st_size for f in files)+owner.stat().st_size
            for _ in range(12):
                size=actual+len((canonical_json(report)+'\n').encode())
                if size==usage['output_bytes']:break
                usage['output_bytes']=size
            else:raise ValueError('baseline workload byte accounting did not stabilize')
            if usage['output_bytes']>self.max_output_bytes:raise ValueError('baseline evidence exceeds its finite output budget')
            atomic_write_json(path,report)
            checked=self.verify_effect(call,None)
            if checked.status is not EvidenceStatus.VERIFIED:raise ValueError(checked.reason)
            return _write_receipt(_receipt_path(call),self._receipt(call,path))
