"""Scene-decomposition dataset Operator owned by EvolutionController."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelPolicy,ModelProfile
from dataevolver.adapters.model_gateway import ModelGateway,RoutedModelGateway,ModelError
from dataevolver.adapters.training_media_worker import TrainingMediaProvider,MODELS,validate_worker,cpu_observation
from dataevolver.adapters.training_prompt_enhancement import (PromptEnhancementProvider,PE_MODELS,PE_ROLES,
    validate_pe_worker,pe_observation)
from dataevolver.dataset.contracts import ArtifactStore,file_sha256,payload_sha256
from dataevolver.dataset.supervision import load_dataset,validate_record
from dataevolver.dataset.training_pipeline import TrainingOperations,QualityRejected,validate_production_contract,REVIEW_SCHEMA,negative_rejected
from ..models import (OperatorSpec,ExpectedOutputSpec,OperatorReceipt,StageStatus,EffectVerification,
                      EvidenceStatus,RecoveryResult,RecoveryStatus)
from ..operators import _receipt_path,_write_receipt,_load_receipt
from ..state_store import atomic_write_json


OPERATOR_ID="data.training_supervision"


class TrainingSupervisionOperator:
    def __init__(self,*,allowed_write_root,contract,roots=None,providers=None,wait_for_role=None):
        self.contract=json.loads(json.dumps(contract));validate_production_contract(self.contract)
        self.root=Path(allowed_write_root).resolve()
        self.fixture=self.contract["production"]["evidence_kind"]=="fixture"
        if roots is None:
            roots={name:os.environ[env] for name,env in self.contract["stores"].items()}
        if set(roots)!=set(self.contract["stores"]):raise ValueError("registered store membership changed")
        self.store=ArtifactStore(roots)
        self.dataset_store,self.bundle_store=self.contract["dataset_store"],self.contract["bundle_store"]
        self.dataset_root,self.bundle_root=self.store.roots[self.dataset_store],self.store.roots[self.bundle_store]
        writable=[self.root,self.dataset_root,self.bundle_root]
        if any(a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(writable) for b in writable[i+1:]):
            raise ValueError("campaign, dataset and portable roots must be disjoint")
        for name,root in self.store.roots.items():
            if name not in {self.dataset_store,self.bundle_store} and any(root.is_relative_to(w) or w.is_relative_to(root) for w in writable):
                raise ValueError("read-only source store overlaps a writable root")
        model=self.contract["models"]
        self.profiles=tuple(ModelProfile.from_dict(p) for p in model["profiles"])
        self.policy=ModelPolicy.from_dict(model["policy"])
        if self.policy.usage_mode!="accounting" or self.policy.max_calls is not None:
            raise ValueError("new training-data production uses explicit usage accounting")
        required={"training.generate","training.background","training.resolve","training.segment","training.review"}
        if set(self.policy.bindings)!=required:raise ValueError("training model role scope changed")
        self.prompt_policy=ModelPolicy.from_dict(model['prompt_policy']) if 'prompt_policy' in model else None
        if self.prompt_policy is not None:
            if (set(self.prompt_policy.bindings)!=set(PE_ROLES) or self.prompt_policy.usage_mode!='accounting'
                    or self.prompt_policy.max_calls is not None):raise ValueError('PE role policy must retain both explicit prompt components')
        if any(self.contract['production']['recipe'].get(k,False) for k in ('refine_scene_prompt','refine_background_prompt')) and self.prompt_policy is None:
            raise ValueError('enabled prompt refinement requires its independent model policy')
        self.providers,self.custom_wait=providers,wait_for_role
        if not self.fixture:
            by_id={p.profile_id:p for p in self.profiles}
            if any(p.backend!="local_python" for p in self.profiles):raise ValueError("real production requires registered offline workers")
            for p in self.profiles:
                config=model["workers"][p.profile_id]
                (validate_pe_worker if config['kind'] in PE_MODELS else validate_worker)(config)
                selected=PE_MODELS if config['kind'] in PE_MODELS else MODELS
                if p.model!=selected[config["kind"]] or p.revision!="sha256:"+payload_sha256(config["file_hashes"]):
                    raise ValueError("frozen model/worker identity mismatch")
            expected={"training.generate":"Qwen/Qwen-Image-2.1","training.background":"Qwen/Qwen-Image-2.1",
                      "training.resolve":"Qwen/Qwen3.8-27B","training.review":"Qwen/Qwen3.8-27B","training.segment":"SAM3"}
            if any(by_id[self.policy.bindings[r]].model!=m for r,m in expected.items()):
                raise ValueError("a selected production model was substituted")
            if self.prompt_policy is not None and any(by_id[self.prompt_policy.bindings[role]].model!=PE_MODELS[kind] for role,kind in PE_ROLES.items()):
                raise ValueError('a selected prompt component was substituted')
        self.spec=OperatorSpec(operator_id=OPERATOR_ID,version="1",retry_safe=True,source_path=str(Path(__file__)),
            code_revision=file_sha256(Path(__file__)),model_config_hash=payload_sha256({"config":self.contract,"roots":{k:str(v) for k,v in self.store.roots.items()}}),
            allowed_write_roots=tuple(str(p) for p in writable),expected_output_specs=(ExpectedOutputSpec("effect.json"),ExpectedOutputSpec("production.json")))

    def gateway(self,call):
        factories=self.providers
        if factories is None and self.fixture:
            factories={}
        if factories is None:
            factories={p.profile_id:(lambda p=p:(PromptEnhancementProvider if self.contract['models']['workers'][p.profile_id]['kind'] in PE_MODELS else TrainingMediaProvider)(
                p,self.contract["models"]["workers"][p.profile_id],self.store,self.dataset_store,
                Path(call.work_dir)/"native"/p.profile_id)) for p in self.profiles}
        main=ModelGateway(profiles=self.profiles,policy=self.policy,receipt_root=Path(call.work_dir)/"model_receipts",
                            artifact_store=self.store,providers=factories)
        if self.prompt_policy is None:return main
        prompt=ModelGateway(profiles=self.profiles,policy=self.prompt_policy,receipt_root=Path(call.work_dir)/'pe_model_receipts',
            artifact_store=self.store,providers=factories)
        return RoutedModelGateway(main,prompt)

    def _validate(self,call):
        if (call.operator_id,call.operator_version)!=(OPERATOR_ID,"1") or payload_sha256(call.parameters)!=payload_sha256(self.contract):
            raise ValueError("call differs from frozen training production contract")
        if not Path(call.work_dir).resolve().is_relative_to(self.root):raise ValueError("call work directory escaped its campaign")

    def _wait(self,call,role):
        if self.custom_wait is not None:return self.custom_wait(role)
        bindings=self.prompt_policy.bindings if role in PE_ROLES and self.prompt_policy is not None else self.policy.bindings
        config=self.contract["models"]["workers"][bindings[role]]
        if config.get('kind') in PE_MODELS:
            stable=0;deadline=self.contract['production']['resource_deadline_epoch']
            while True:
                try:placement=pe_observation(config);stable+=1
                except ModelError as exc:
                    if exc.code not in {'RESOURCE_BUSY','RESOURCE_UNAVAILABLE'}:raise
                    placement={'error_code':exc.code,'reason':str(exc)};stable=0
                atomic_write_json(Path(call.work_dir)/'resource-status.json',{'state':'READY_PE' if stable>=2 else 'WAITING_PE',
                    'role':role,'deployment':placement,'stable_observations':stable,'at':time.time(),'model_request_submitted':False})
                if stable>=2:return
                if stable==0 and time.time()>=deadline:raise TimeoutError('PE resource wait expired before submission')
                time.sleep(30 if stable else min(30,max(1,deadline-time.time())))
        if config.get("resident"):
            from dataevolver.adapters.training_media_resident import identity,resident_observation
            deployment=identity(config)
            os.environ["CUDA_VISIBLE_DEVICES"]=",".join(deployment["gpu_uuids"])
            stable=0;deadline=self.contract["production"]["resource_deadline_epoch"]
            while True:
                try:placement=resident_observation(config);stable+=1
                except ModelError as exc:
                    if exc.code not in {"RESOURCE_BUSY","RESOURCE_UNAVAILABLE"}:raise
                    placement={"error_code":exc.code,"reason":str(exc)};stable=0
                atomic_write_json(Path(call.work_dir)/"resource-status.json",{
                    "state":"READY_RESIDENT" if stable>=2 else "WAITING_RESIDENT_ACTIVATIONS",
                    "role":role,"deployment":placement,"stable_observations":stable,
                    "updated_at":time.time(),"model_request_submitted":False})
                if stable>=2:return
                if stable==0 and time.time()>=deadline:raise TimeoutError("resident activation resource wait expired before submission")
                time.sleep(30 if stable else min(30,max(1,deadline-time.time())))
            return
        if config.get("execution_device")=="cpu":
            os.environ["CUDA_VISIBLE_DEVICES"]=""
            placement=cpu_observation(config)
            atomic_write_json(Path(call.work_dir)/"resource-status.json",{"state":"READY_CPU","role":role,"deployment":placement,
                "updated_at":time.time(),"model_request_submitted":False})
            return
        required=config["resource_profile"]
        limit=Path(call.work_dir)/"resource-deadline.json"
        if not limit.exists():
            deadline=time.time()+self.contract["production"]["resource_wait_seconds"]
            atomic_write_json(limit,{"deadline":min(deadline,self.contract["production"].get("resource_deadline_epoch",deadline))})
        deadline=json.loads(limit.read_text())["deadline"]
        stable,selected=0,None
        while True:
            result=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,memory.free,utilization.gpu","--format=csv,noheader,nounits"],
                                  check=True,text=True,capture_output=True,timeout=20)
            rows=[]
            for line in result.stdout.splitlines():
                index,uuid,free,util=[x.strip() for x in line.split(",")]
                rows.append({"index":int(index),"uuid":uuid,"free_mib":int(free),"utilization":int(util)})
            if config.get("shard_max_memory_mib"):
                rows=sorted(rows,key=lambda r:r["index"])
                suitable=(len(rows)==2 and all(r["free_mib"]>=minimum and r["utilization"]<=required["max_utilization"]
                    for r,minimum in zip(rows,config["shard_min_free_mib"])))
                identity=tuple(r["uuid"] for r in rows) if suitable else None
                stable=stable+1 if identity and identity==selected else int(identity is not None)
                selected=identity
                atomic_write_json(Path(call.work_dir)/"resource-status.json",{"state":"WAITING_GPU","role":role,
                    "gpu_snapshot":rows,"shard_min_free_mib":config["shard_min_free_mib"],
                    "shard_max_memory_mib":config["shard_max_memory_mib"],"stable_observations":stable,"updated_at":time.time(),"model_request_submitted":False})
                if stable>=2:
                    os.environ["CUDA_VISIBLE_DEVICES"]=",".join(identity)
                    atomic_write_json(Path(call.work_dir)/"resource-status.json",{"state":"READY_SHARDED","role":role,
                        "selected_gpus":rows,"process_devices":["cuda:0","cuda:1"],"updated_at":time.time()})
                    return
                if time.time()>=deadline:raise TimeoutError("sharded GPU resource wait expired before submission")
                time.sleep(min(30,max(1,deadline-time.time())))
                continue
            suitable=sorted([r for r in rows if r["free_mib"]>=required["min_free_mib"] and r["utilization"]<=required["max_utilization"]],
                            key=lambda r:(r["utilization"],-r["free_mib"]))
            candidate=suitable[0] if suitable else None
            stable=stable+1 if candidate and selected and candidate["uuid"]==selected["uuid"] else int(candidate is not None)
            selected=candidate
            atomic_write_json(Path(call.work_dir)/"resource-status.json",{"state":"WAITING_GPU","role":role,"gpu_snapshot":rows,
                "profile":required,"stable_observations":stable,"updated_at":time.time(),"model_request_submitted":False})
            if stable>=2:
                os.environ["CUDA_VISIBLE_DEVICES"]=selected["uuid"]
                atomic_write_json(Path(call.work_dir)/"resource-status.json",{"state":"READY","role":role,"selected_gpu":selected,
                    "process_device":"cuda:0","updated_at":time.time()})
                return
            if time.time()>=deadline:raise TimeoutError("model-specific GPU resource wait expired before submission")
            time.sleep(min(30,max(1,deadline-time.time())))

    def _checker_proof(self):
        if self.fixture or self.contract["production"]["checker_precheck"]:return
        proof=self.contract["production"].get("checker_proof")
        if not proof:raise ValueError("main production requires the real canary checker proof")
        source=self.store.roots[proof["store"]]
        if file_sha256(source/"manifest.json")!=proof["manifest_sha256"]:raise ValueError("checker proof manifest changed")
        loaded=load_dataset(source)
        profile=next(p for p in self.profiles if p.profile_id==self.policy.bindings["training.review"])
        verified=False
        for row,_ in loaded:
            if row["family_id"] in {t["slot_id"] for t in self.contract["production"]["tasks"]}:raise ValueError("canary family reused in production")
            for model in row["model_receipts"]:
                request,receipt=model["request"],model["receipt"]
                if model.get("purpose")!="checker_negative":continue
                artifacts=request["artifacts"]
                verified|=(len(artifacts)>=2 and artifacts[0]==artifacts[1] and receipt["model"]==profile.model and
                    receipt["revision"]==profile.revision and request["response_schema"]==REVIEW_SCHEMA and
                    negative_rejected(receipt["output"]))
        if not verified:raise ValueError("negative proof did not reject the unchanged selected-object background")

    def verify_effect(self,call,receipt):
        self._validate(call);work=Path(call.work_dir)
        if not (work/"effect.json").exists():return EffectVerification(EvidenceStatus.UNOBSERVED,"training batch is incomplete")
        try:
            effect=json.loads((work/"effect.json").read_text());manifest=json.loads((work/"production.json").read_text())
            if effect["binding"]!=payload_sha256(call.to_dict()) or effect["production_sha256"]!=file_sha256(work/"production.json"):
                raise ValueError("training batch effect binding changed")
            gateway=self.gateway(call)
            if manifest["usage"]!=gateway.audit_usage():raise ValueError("model-attempt accounting differs from its journal")
            for row in manifest["accepted"]:
                validate_record(row,self.store,allow_fixture=self.fixture)
                for model in row["model_receipts"]:
                    from dataevolver.adapters.model_contracts import ModelRequest
                    cached=gateway.recover(ModelRequest.from_dict(model["request"]))
                    if cached is None or cached.without_reuse_hash()!=model["receipt_hash"]:raise ValueError("accepted record lost its model receipt")
            if manifest["complete"]:
                loaded=load_dataset(self.bundle_root,allow_fixture=self.fixture)
                if [r for r,_ in loaded]!=manifest["accepted"]:raise ValueError("portable dataset differs from accepted records")
            if receipt is not None and receipt.effect_hash!=file_sha256(work/"effect.json"):raise ValueError("Operator effect receipt changed")
            return EffectVerification(EvidenceStatus.VERIFIED,"batch membership, media, receipts, costs and independent loading verified",
                artifact_hashes={"effect.json":file_sha256(work/"effect.json"),"production.json":file_sha256(work/"production.json")})
        except (ValueError,OSError,KeyError,TypeError,ModelError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED,str(exc))

    def recover(self,call):
        verification=self.verify_effect(call,_load_receipt(_receipt_path(call)))
        if verification.status is EvidenceStatus.VERIFIED:return RecoveryResult(RecoveryStatus.RECOVERED,self._receipt(call,True))
        if verification.status is EvidenceStatus.UNOBSERVED:return RecoveryResult(RecoveryStatus.RETRY_SAFE,reason="recover bound steps and model journals before any new request")
        return RecoveryResult(RecoveryStatus.AMBIGUOUS,reason=verification.reason)

    def _receipt(self,call,reused):
        work=Path(call.work_dir);value=json.loads((work/"production.json").read_text());complete=value["complete"]
        count=self.contract["production"]["count"]
        receipt=OperatorReceipt(call_id=call.call_id,operator_id=OPERATOR_ID,operator_version="1",
            status=StageStatus.SUCCESS if complete else StageStatus.FAILED,started_at=value["started_at"],finished_at=value["finished_at"],
            return_code=0 if complete else 1,effect_hash=file_sha256(work/"effect.json"),reused=reused,
            output_artifacts=({"path":str(work/"production.json"),"sha256":file_sha256(work/"production.json")},),
            stage_result={"operator_id":OPERATOR_ID,"operator_version":"1","status":"SUCCESS" if complete else "FAILED",
                "metrics":{"qualified_ratio":len(value["accepted"])/count,"local_model_calls":value["usage"]["model_calls"]},
                "hard_constraints":{"artifact_valid":True,"all_tasks_recorded":len(value["slots"])==count,"all_qualified":complete},
                "failure_code":"" if complete else value["stop_reason"],"evidence_refs":[str(work/"production.json")],
                "cost":{"model_calls":value["usage"]["model_calls"],"observed_wall_seconds":value["usage"]["observed_wall_seconds"]}})
        return _write_receipt(_receipt_path(call),receipt)

    def execute(self,call):
        self._validate(call);recovery=self.recover(call)
        if recovery.status is RecoveryStatus.RECOVERED:return recovery.receipt
        if recovery.status is not RecoveryStatus.RETRY_SAFE:raise ValueError(recovery.reason)
        self._checker_proof()
        work=Path(call.work_dir);work.mkdir(parents=True,exist_ok=True)
        started=work/"started.json"
        if not started.exists():atomic_write_json(started,{"started_at":datetime.now(timezone.utc).isoformat()})
        gateway=self.gateway(call)
        ops=TrainingOperations(self.contract,self.store,self.dataset_store,gateway,work,call,lambda r:self._wait(call,r))
        accepted,failures,family_outputs,slots=[],[],[],[]
        consecutive=0;stop="";seen=set()
        for task in self.contract["production"]["tasks"]:
            slot={"slot_id":task["slot_id"],"split":task["split"],"status":"not_run","family_id":None};slots.append(slot)
            if stop:continue
            attempt=0
            while True:
                path=work/"families"/f"{task['slot_id']}-a{attempt}.json"
                if path.exists():outcome=json.loads(path.read_text())
                else:
                    try:
                        row=ops.run_family(task,attempt)
                        identity=row["target"]["image"]["sha256"]
                        if row["status"]=="pass" and identity in seen:raise QualityRejected("duplicate target image across source families")
                        outcome={"binding":payload_sha256([self.contract,task,attempt]),"record":row,"outputs":ops.last_outputs,
                                 "status":row["status"],"attempts":ops.attempts,"model_requests":ops.records}
                    except QualityRejected as exc:
                        outcome={"binding":payload_sha256([self.contract,task,attempt]),"status":"fail","error":str(exc),
                                 "attempts":ops.attempts,"model_requests":ops.records}
                    except ModelError as exc:
                        outcome={"binding":payload_sha256([self.contract,task,attempt]),"status":"fail" if exc.code=="QUALITY_REJECTED" else "blocked",
                                 "error_code":exc.code,"error":str(exc),"attempts":ops.attempts,"model_requests":ops.records}
                    except Exception as exc:
                        outcome={"binding":payload_sha256([self.contract,task,attempt]),"status":"blocked","error_code":type(exc).__name__,
                                 "error":str(exc),"attempts":ops.attempts,"model_requests":ops.records}
                    atomic_write_json(path,outcome)
                if outcome["binding"]!=payload_sha256([self.contract,task,attempt]):raise ValueError("frozen family task changed")
                if outcome["status"]=="pass":
                    row=outcome["record"];validate_record(row,self.store,allow_fixture=self.fixture)
                    accepted.append(row);family_outputs.append(outcome["outputs"]);seen.add(row["target"]["image"]["sha256"])
                    slot.update(status="pass",family_id=row["family_id"]);consecutive=0;break
                failures.append(outcome);consecutive+=1
                if outcome["status"]=="blocked":stop=outcome.get("error_code","INFRASTRUCTURE_BLOCKED");slot["status"]="blocked";break
                if consecutive>=self.contract["production"]["max_consecutive_failures"]:
                    stop="NO_PROGRESS";slot["status"]="failed";break
                attempt+=1
        complete=len(accepted)==self.contract["production"]["count"]
        export=None
        if complete:
            try:export={k:v.to_dict() for k,v in ops.export(accepted,self.bundle_root,family_outputs).items()}
            except Exception as exc:
                complete=False;stop="EXPORT_FAILED"
                failures.append({"status":"blocked","operation":"dataset.export","error_code":type(exc).__name__,"error":str(exc)})
        value={"schema_version":"dataevolver.training_production.v1","started_at":json.loads(started.read_text())["started_at"],
            "finished_at":datetime.now(timezone.utc).isoformat(),"complete":complete,"stop_reason":stop,"slots":slots,
            "accepted":accepted,"failures":failures,"export":export,"usage":gateway.audit_usage(),
            "evidence_kind":self.contract["production"]["evidence_kind"],"quality_benefit_verified":False,"training_benefit_verified":False}
        atomic_write_json(work/"production.json",value)
        atomic_write_json(work/"effect.json",{"binding":payload_sha256(call.to_dict()),"production_sha256":file_sha256(work/"production.json")})
        verified=self.verify_effect(call,None)
        if verified.status is not EvidenceStatus.VERIFIED:raise ValueError(verified.reason)
        return self._receipt(call,False)
