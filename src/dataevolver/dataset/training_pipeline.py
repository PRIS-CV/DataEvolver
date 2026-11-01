"""Nine scene-decomposition operations beneath the existing Harness.

This module owns media transformations and step evidence. EvolutionController
owns the campaign; ModelGateway owns model attempts and recovery.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.adapters.model_gateway import ModelError
from dataevolver.harness.data_contracts import DataNodeRun
from dataevolver.harness.models import OperatorReceipt, StageStatus
from dataevolver.harness.state_store import atomic_write_json
from .contracts import ArtifactRef, canonical_json, file_sha256, identifier, payload_sha256
from .supervision import (refs, write_artifact, extract_subject_references, completion_region,
    complete_background_composite, masked_background_condition, rectangular_condition_region,
    completion_blend_region, validate_record, export_dataset)


PORTS = {
    "scene.plan": ({"training_goal","task_config"},{"scene_plan"}),
    "image.generate": ({"scene_plan"},{"image"}),
    "subjects.resolve": ({"image","subject_list"},{"instance_bindings"}),
    "subjects.segment": ({"image","instance_bindings"},{"mask_list"}),
    "subjects.extract": ({"image","mask_list"},{"subject_references"}),
    "background.complete": ({"image","mask_list"},{"background_visible","background_completed"}),
    "sample.assemble": ({"background","subject_references","target_image","instance_bindings","scene_plan"},{"training_record"}),
    "sample.review": ({"training_record","mask_list","background_visible"},{"review_evidence"}),
    "dataset.export": ({"training_records","review_evidence","family_assignments","task_config"},{"dataset_bundle"}),
}
QUALITY_CHECKS = ("subject_correspondence","reference_identity","selected_removal",
                  "unselected_preservation","background_completion")
STATUS = {"type":"string","enum":["pass","fail","uncertain"]}
REVIEW_SCHEMA = {"type":"object","properties":{
    "status":STATUS,"summary":{"type":"string"},"criteria":{"type":"object","properties":{
        k:{"type":"object","properties":{"status":STATUS,"reason":{"type":"string"}},
           "required":["status","reason"],"additionalProperties":False} for k in QUALITY_CHECKS},
        "required":list(QUALITY_CHECKS),"additionalProperties":False}},
    "required":["status","summary","criteria"],"additionalProperties":False}
RESOLVE_SCHEMA = {"type":"object","properties":{
    "status":STATUS,"reason":{"type":"string"},
    "coordinate_system":{"type":"string","enum":["normalized_0_1","normalized_0_1000"]},
    "subjects":{"type":"array","items":{
        "type":"object","properties":{"subject_id":{"type":"string"},"target_instance_id":{"type":"string"},
            "description":{"type":"string"},"box_xyxy_normalized":{"type":"array","items":{"type":"number","minimum":0,"maximum":1000},"minItems":4,"maxItems":4}},
        "required":["subject_id","target_instance_id","description","box_xyxy_normalized"],"additionalProperties":False}}},
    "required":["status","reason","coordinate_system","subjects"],"additionalProperties":False}
IMAGE_SCHEMA = {"type":"object","properties":{"artifact":{"type":"object"},"actual_parameters":{"type":"object"},
    "image_size":{"type":"array","items":{"type":"integer"}}},"required":["artifact","actual_parameters","image_size"],"additionalProperties":False}
MASK_SCHEMA = {"type":"object","properties":{"instances":{"type":"array","items":{"type":"object"}}},
               "required":["instances"],"additionalProperties":False}
SHADOW_SCHEMA={"type":"object","properties":{
    "status":STATUS,"reason":{"type":"string"},
    "coordinate_system":{"type":"string","enum":["normalized_0_1","normalized_0_1000"]},
    "shadows":{"type":"array","items":{"type":"object","properties":{
        "subject_id":{"type":"string"},"box_xyxy_normalized":{"type":"array","items":{"type":"number","minimum":0,"maximum":1000},"minItems":4,"maxItems":4},
        "reason":{"type":"string"}},"required":["subject_id","box_xyxy_normalized","reason"],"additionalProperties":False}},
    "no_shadow_subject_ids":{"type":"array","items":{"type":"string"}}},
    "required":["status","reason","coordinate_system","shadows","no_shadow_subject_ids"],"additionalProperties":False}
REUSE_ROLES={"generate":"training.generate","resolve":"training.resolve","segment":"training.segment",
             "background":"training.background","negative-check":"training.review","review":"training.review",
             "shadow-probe":"training.resolve","refine-generate":"training.refine_t2i","refine-background":"training.refine_i2i"}


class QualityRejected(RuntimeError):
    """A completed observation rejected this scene; it is not infrastructure failure."""


def negative_rejected(result):
    """The unchanged background must fail object removal specifically."""
    return result["status"] in {"fail","uncertain"} and result["criteria"]["selected_removal"]["status"]=="fail"


def normalize_instance_observation(result):
    """Convert only explicitly declared units, retaining the actual raw output."""
    import math
    unit=result.get("coordinate_system")
    if unit not in {"normalized_0_1","normalized_0_1000"}:
        raise ValueError("observed box coordinate system must be explicitly declared")
    divisor=1 if unit=="normalized_0_1" else 1000
    subjects=[]
    for subject in result["subjects"]:
        box=subject["box_xyxy_normalized"]
        if (len(box)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=divisor for v in box)
                or box[0]>=box[2] or box[1]>=box[3]):
            raise ValueError("observed box does not match its declared coordinate system")
        subjects.append({**subject,"box_xyxy_normalized":[v/divisor for v in box]})
    return {**result,"coordinate_system":"normalized_0_1","subjects":subjects,
            "raw_observation":result,"coordinate_conversion":{"declared":unit,"divisor":divisor}}


def normalize_shadow_observation(result,selected_ids,preserved_ids=()):
    selected=set(selected_ids);found={s['subject_id'] for s in result['shadows']};absent=set(result['no_shadow_subject_ids'])
    preserved=set(preserved_ids)
    if (selected & preserved or result['status']!='pass' or found & absent
            or (found | absent)-selected-preserved or (found | absent)&selected!=selected):
        raise QualityRejected('shadow attribution is missing, uncertain, or refers to an unselected subject')
    normalized=normalize_instance_observation({'coordinate_system':result['coordinate_system'],'subjects':result['shadows']})
    return {'status':'pass','reason':result['reason'],
        'shadows':[s for s in normalized['subjects'] if s['subject_id'] in selected],
        'preserved_shadow_observations':[s for s in normalized['subjects'] if s['subject_id'] in preserved],
        'no_shadow_subject_ids':[s for s in result['no_shadow_subject_ids'] if s in selected],
        'preserved_no_shadow_subject_ids':[s for s in result['no_shadow_subject_ids'] if s in preserved],
        'coordinate_system':'normalized_0_1',
        'coordinate_conversion':normalized['coordinate_conversion'],'raw_observation':result}


def validate_shadow_correction(value, target, selected_ids):
    """Bind a recorded Agent observation to the actual failed source image."""
    if (value.get('kind') not in {'agent_visual_observation', 'runtime_agent_observation'} or
            value.get('source_image') != target.to_dict() or
            not value.get('reason') or not value.get('rejected_candidate')):
        raise ValueError('shadow correction must identify the observed source and rejected candidate')
    candidate = ArtifactRef.from_dict(value['rejected_candidate'])
    if not candidate.media_type.startswith('image/') or candidate.sha256 == target.sha256:
        raise ValueError('shadow correction requires a distinct failed completion')
    return normalize_shadow_observation(value, selected_ids)


def _field(value,path):
    for part in path.split("."):
        if not isinstance(value,dict) or part not in value:
            raise ValueError("missing compiled goal/config field: "+path)
        value=value[part]
    return value


def validate_production_contract(config):
    goal,plan=config["planning"]["goal"],config["planning"]["plan"]
    if (goal["task_type"]!="multi_subject_insertion" or goal["construction_mode"]!="scene_decomposition" or
            goal.get("materials") or goal["task_config"].get("planning_only")):
        raise ValueError("this producer compiles only the authorized no-material scene-decomposition route")
    nodes=plan["dataset_plan"]["nodes"]
    if {n["operator"] for n in nodes}!=set(PORTS) or len(nodes)!=len(PORTS):
        raise ValueError("production requires an explicit adapter for every planned operation")
    for node in nodes:
        expected_inputs,expected_outputs=PORTS[node["operator"]]
        if {b["port"] for b in node["input_bindings"]}!=expected_inputs or {p["name"] for p in node["outputs"]}!=expected_outputs:
            raise ValueError("planned media ports need a new compiler adapter: "+node["operator"])
    production=config["production"]
    for option in ('refine_scene_prompt','refine_background_prompt','probe_visible_effects','pe_t2i_enable_thinking','pe_i2i_enable_thinking'):
        if type(production['recipe'].get(option,False)) is not bool:raise ValueError('prompt refinement and observation modes must be explicit booleans')
    if type(production['recipe'].get('pe_max_tokens',4096)) is not int or not 128<=production['recipe'].get('pe_max_tokens',4096)<=24000:
        raise ValueError('PE generation must have a bounded token limit')
    if "background_negative_prompt" in production["recipe"] and not isinstance(production["recipe"]["background_negative_prompt"],str):
        raise ValueError("background negative prompt must be explicit text")
    if production["recipe"].get("background_conditioning","source_image") not in {"source_image","masked_region"}:
        raise ValueError("background conditioning must be an explicit registered transformation")
    condition_shape=production['recipe'].get('background_conditioning_shape','region')
    if condition_shape not in {'region','bounding_rectangle'}:
        raise ValueError('unknown background conditioning shape')
    if condition_shape!='region' and production['recipe'].get('background_conditioning')!='masked_region':
        raise ValueError('rectangular conditioning requires masked-region editing')
    feather=production["recipe"].get("background_feather_pixels",0)
    if type(feather) is not int or not 0<=feather<=32:
        raise ValueError("background feather radius must be between zero and 32")
    if type(production['recipe'].get('probe_shadows',False)) is not bool:
        raise ValueError('shadow probing must be explicitly enabled or disabled')
    if not isinstance(production['recipe'].get('shadow_probe_instruction',''),str):
        raise ValueError('shadow probe response instruction must be explicit text')
    corrections = production.get('shadow_region_corrections', {})
    if not isinstance(corrections, dict):
        raise ValueError('shadow corrections must be explicit family-to-evidence bindings')
    for family, ref in corrections.items():
        identifier(family)
        ArtifactRef.from_dict(ref)
    if production["evidence_kind"] not in {"real_execution","fixture"}:
        raise ValueError("unknown production evidence kind")
    if production["count"]!=sum(production["split_counts"].values()) or production["count"]<1:
        raise ValueError("production count and splits differ")
    if type(production["max_consecutive_failures"]) is not int or production["max_consecutive_failures"]<1:
        raise ValueError("a no-progress stop rule is required")
    tasks=production["tasks"]
    if len(tasks)!=production["count"] or len({t["slot_id"] for t in tasks})!=len(tasks):
        raise ValueError("frozen task slots do not match dataset target")
    if {k:sum(t["split"]==k for t in tasks) for k in production["split_counts"]}!=production["split_counts"]:
        raise ValueError("frozen task split membership changed")
    for task in tasks:
        identifier(task["slot_id"])
        for field in ('scene_refinement_request','background_refinement_request'):
            if field in task and (not isinstance(task[field],str) or not task[field].strip()):
                raise ValueError('PE brief request must be nonempty text')
        background_cfg = task.get('background_true_cfg_scale', production['recipe']['true_cfg_scale'])
        if type(background_cfg) not in (int, float) or not 1 <= background_cfg <= 8:
            raise ValueError('background CFG must be a finite explicit value between one and eight')
        edge_margin = task.get('selected_edge_margin_pixels', 0)
        if type(edge_margin) is not int or not 0 <= edge_margin <= 8:
            raise ValueError('selected edge margin must be between zero and eight pixels')
        identities=[s["subject_id"] for s in task["subjects"]]
        if task["selection_mode"] not in {"all","selected"} or len(identities)<2:
            raise ValueError("unknown subject selection mode or insufficient references")
        if not task["scene_prompt"] or not task["insertion_instruction"]:
            raise ValueError("natural scene and training instruction must be explicit")
        if len(set(identities))!=len(identities) or not set(task["selected_subject_ids"]).issubset(identities) or not task["selected_subject_ids"]:
            raise ValueError("task subject inventory/selection is not bound")
        if task["selection_mode"]=="selected" and set(task["selected_subject_ids"])==set(identities):
            raise ValueError("selected-subject test must preserve an unselected instance")
        if task["selection_mode"]=="all" and set(task["selected_subject_ids"])!=set(identities):
            raise ValueError("all-subject test dropped a planned instance")
    # Deployment handoff may adopt a committed generation, never resubmit it.
    reuse=production.get("generation_reuse",{})
    allowed={t["slot_id"]+"-a0":t for t in tasks}
    if not isinstance(reuse,dict) or set(reuse)-set(allowed):
        raise ValueError("generation reuse must bind an existing initial task family")
    for family,value in reuse.items():
        if set(value)!={"request","receipt_hash","source_config_sha256"}:
            raise ValueError("generation reuse requires its request, receipt and source contract")
        request=ModelRequest.from_dict(value["request"])
        if request.role!="training.generate" or request.seed!=allowed[family]["seed"]:
            raise ValueError("generation reuse changed the model role or task seed")
        for name in ("receipt_hash","source_config_sha256"):
            ArtifactRef("evidence",name,value[name])
    surface_contracts=production.get('surface_effect_contracts',{})
    if not isinstance(surface_contracts,dict) or set(surface_contracts)-set(allowed):
        raise ValueError('surface support contracts must bind frozen task families')
    if set(surface_contracts)&set(corrections):
        raise ValueError('surface support cannot mix with legacy box correction')
    for ref in surface_contracts.values():ArtifactRef.from_dict(ref)
    model_reuse=production.get("model_reuse",{})
    if not isinstance(model_reuse,dict) or set(model_reuse)-set(allowed):
        raise ValueError("model reuse must bind an existing initial task family")
    for family,phases in model_reuse.items():
        if not isinstance(phases,dict) or set(phases)-set(REUSE_ROLES) or (family in reuse and "generate" in phases):
            raise ValueError("model reuse phase is unknown or duplicated")
        for phase,value in phases.items():
            if set(value)!={"request","receipt_hash","source_config_sha256"}:
                raise ValueError("model reuse requires its request, receipt and source contract")
            request=ModelRequest.from_dict(value["request"])
            if request.role!=REUSE_ROLES[phase] or request.seed!=allowed[family]["seed"]:
                raise ValueError("model reuse changed the phase role or task seed")
            for name in ("receipt_hash","source_config_sha256"):ArtifactRef("evidence",name,value[name])
    return nodes


class TrainingOperations:
    def __init__(self,config,store,dataset_store,gateway,work_dir,parent_call,wait_for_role):
        self.config,self.store,self.dataset_store,self.gateway=config,store,dataset_store,gateway
        self.work,self.parent,self.wait=Path(work_dir),parent_call,wait_for_role
        self.nodes={n["operator"]:n for n in validate_production_contract(config)}
        self.recipe=config["production"]["recipe"]
        self.namespace=payload_sha256(parent_call.to_dict())[:16]
        self.revision=file_sha256(Path(__file__))
        self.allow_fixture=config["production"]["evidence_kind"]=="fixture"
        self.records,self.attempts=[],[]
        self.task=None;self.family_id="";self.operation=""

    def _write(self,name,value):
        return write_artifact(self.store,self.dataset_store,f"families/{self.family_id}/{name}.json",
                              canonical_json(value).encode(),"application/json")

    def _read(self,ref):
        value=json.loads(self.store.resolve(ref).read_text())
        for artifact in refs(value):self.store.resolve(artifact)
        return value

    def _append_model(self,record):
        if not any(r["request"]["request_id"]==record["request"]["request_id"] for r in self.records):
            self.records.append(record)

    def _model(self,role,parameters,schema,artifacts=(),*,phase,max_tokens=None,purpose="production"):
        identity=f"train-{self.namespace}-{self.family_id}-{phase}";identifier(identity)
        reuse=self.config["production"].get("generation_reuse",{}).get(self.family_id) if phase=="generate" else None
        reuse=self.config["production"].get("model_reuse",{}).get(self.family_id,{}).get(phase,reuse)
        if reuse is not None:identity=reuse["request"]["request_id"]
        request=ModelRequest(identity,role,({"role":"user","content":canonical_json(parameters)},),schema,
            artifacts=tuple(artifacts),max_tokens=max_tokens or self.recipe["review_max_tokens"],
            timeout_seconds=self.recipe["model_timeout_seconds"],max_attempts=1,seed=self.task["seed"])
        if reuse is not None and request.to_dict()!=reuse["request"]:
            raise ModelError("REQUEST_CONFLICT","adopted effect differs from the actual planned request")
        attempt={"request":request.to_dict(),"purpose":purpose,"operation":self.operation,
                 "task_hash":payload_sha256(self.task),"started_at":time.time(),"status":"LOOKUP"}
        location=self.work/"attempts"/(identity+".json")
        try:
            receipt=self.gateway.recover(request)
            if reuse is not None and (receipt is None or receipt.without_reuse_hash()!=reuse["receipt_hash"]):
                raise ModelError("PRIOR_EFFECT_MISSING","committed effect must be recovered; resubmission is forbidden")
            if receipt is None:
                self.wait(role)
                attempt["status"]="SUBMITTING";atomic_write_json(location,attempt)
                receipt=self.gateway.invoke(request)
            record={"request":request.to_dict(),"receipt":receipt.to_dict(),"receipt_hash":receipt.without_reuse_hash(),
                    "purpose":purpose,"operation":self.operation}
            self._append_model(record)
            attempt.update(status="COMPLETED",reused=receipt.reused,receipt_hash=record["receipt_hash"])
            return json.loads(canonical_json(receipt.output))
        except BaseException as exc:
            attempt.update(status="FAILED",error_code=getattr(exc,"code",type(exc).__name__),error=str(exc)[:2000])
            raise
        finally:
            attempt["finished_at"]=time.time();atomic_write_json(location,attempt);self.attempts.append(attempt)

    def step(self,operation,inputs,function):
        self.operation=operation
        node=self.nodes[operation];directory=self.work/"steps"/self.family_id/node["id"]
        path=directory/"effect.json"
        binding=payload_sha256({"node":node,"task":self.task,"config":self.config,"inputs":{k:v.to_dict() for k,v in inputs.items()}})
        if path.exists():
            value=json.loads(path.read_text())
            if value["binding"]!=binding or value["operator_revision"]!=self.revision:
                raise ValueError("completed step input or implementation changed")
            outputs={k:ArtifactRef.from_dict(v) for k,v in value["outputs"].items()}
            for ref in refs(value):self.store.resolve(ref)
            for model in value["model_requests"]:
                recovered=self.gateway.recover(ModelRequest.from_dict(model["request"]))
                if recovered is None or recovered.without_reuse_hash()!=model["receipt_hash"]:
                    raise ValueError("step lost its model receipt")
                self._append_model(model)
            return outputs
        atomic_write_json(directory/"started.json",{"operation":operation,"binding":binding,"inputs":{k:v.to_dict() for k,v in inputs.items()},"started_at":time.time()})
        start=len(self.records);started=time.time()
        outputs=function(inputs)
        if set(outputs)!=PORTS[operation][1]:raise ValueError("operation output ports changed")
        for ref in outputs.values():self.store.resolve(ref)
        receipt=OperatorReceipt(call_id=f"{self.parent.call_id}-{self.family_id}-{node['id']}",operator_id=operation,operator_version="1",
            status=StageStatus.SUCCESS,started_at=datetime.fromtimestamp(started,timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),return_code=0,
            output_artifacts=tuple(ref.to_dict() for ref in outputs.values()),stage_result={"outputs":{k:v.to_dict() for k,v in outputs.items()}})
        value={"schema_version":"dataevolver.training_step_effect.v1","binding":binding,"operator_revision":self.revision,
               "outputs":{k:v.to_dict() for k,v in outputs.items()},"model_requests":self.records[start:],"receipt":receipt.to_dict()}
        atomic_write_json(path,value)
        run=DataNodeRun(self.parent.campaign_id,"action-"+self.namespace,self.family_id+"-"+node["id"],payload_sha256(node),
            self.revision,inputs,outputs,receipt.sha256(),tuple(r["receipt_hash"] for r in self.records[start:]))
        atomic_write_json(directory/"node_run.json",run.to_dict())
        return outputs

    def scene_plan(self,inputs):
        return {"scene_plan":self._write("scene-plan",{"task":self.task,"original_goal":self.config["planning"]["goal"],
            "bound_training_goal":self._read(inputs["training_goal"]),"bound_task_config":self._read(inputs["task_config"])})}

    def refine_prompt(self,prompt,*,mode,artifacts=()):
        from dataevolver.adapters.training_prompt_enhancement import PE_T2I_SCHEMA,PE_I2I_SCHEMA
        role='training.refine_t2i' if mode=='generate' else 'training.refine_i2i'
        key='scene_refinement_request' if mode=='generate' else 'background_refinement_request'
        brief=self.task.get(key)
        refinement_input=prompt if brief is None else brief+'\nFixed execution requirements:\n'+prompt
        pe_parameters={'prompt':refinement_input}
        thinking_key='pe_t2i_enable_thinking' if mode=='generate' else 'pe_i2i_enable_thinking'
        if thinking_key in self.recipe:pe_parameters['enable_thinking']=self.recipe[thinking_key]
        result=self._model(role,pe_parameters,PE_T2I_SCHEMA if mode=='generate' else PE_I2I_SCHEMA,
            artifacts,phase='refine-'+mode,max_tokens=self.recipe.get('pe_max_tokens',4096),purpose='prompt_refinement')
        import re
        if any(int(i)>len(artifacts) for i in re.findall(r'<image(\d+)>',result['rewritten_prompt'])):
            raise QualityRejected('PE prompt refers to an image not supplied to the image model')
        model=self.records[-1]
        evidence=self._write(mode+'-prompt-refinement',{'original_prompt':prompt,'refined_prompt':result['rewritten_prompt'],
            'refinement_input':refinement_input,'brief_request':brief,
            'brief_request_origin':self.task.get('refinement_request_origin','task_configuration') if brief is not None else None,
            'pe_request_id':model['request']['request_id'],'pe_receipt_hash':model['receipt_hash'],
            'enable_thinking':pe_parameters.get('enable_thinking',True),
            'aspect_ratio_proposal':{k:v for k,v in result.items() if k in {'wh_ratio','ratio_follow'}},
            **({'raw_text_artifact':result['raw_text_artifact']} if 'raw_text_artifact' in result else {}),
            'applied_resolution':[self.recipe['width'],self.recipe['height']],
            'resolution_authority':'frozen task configuration','original_goal_sha256':payload_sha256(self.config['planning']['goal'])})
        return result['rewritten_prompt'],evidence

    def generate(self,inputs):
        value=self._read(inputs["scene_plan"]);task=value["task"]
        prompt=task["scene_prompt"]+"\nRequired subjects: "+json.dumps([s["description"] for s in task["subjects"]],ensure_ascii=False)+\
            "\nPhotorealistic natural context, physically plausible scale, support, illumination and contact. No isolated-object canvas, labels or text."
        refinement=None
        if self.recipe.get('refine_scene_prompt',False):prompt,refinement=self.refine_prompt(prompt,mode='generate')
        output=self._model("training.generate",{"prompt":prompt,**{k:self.recipe[k] for k in ("width","height","steps","true_cfg_scale")}},
                           IMAGE_SCHEMA,phase="generate",max_tokens=1)
        if not self.allow_fixture and output["actual_parameters"]!=json.loads(self.records[-1]["request"]["messages"][0]["content"]):
            raise ValueError("image backend parameters differ from the actual request")
        return {"image":self._write("image",{"image":output["artifact"],"actual_parameters":output["actual_parameters"],
            "family_id":self.family_id,"origin":"fixture" if self.allow_fixture else "synthetic",
            "request_id":self.records[-1]["request"]["request_id"],
            **({'prompt_refinement':refinement.to_dict()} if refinement is not None else {})})}

    def resolve(self,inputs):
        image=self._read(inputs["image"]);task=self._read(inputs["subject_list"])["task"]
        prompt="Inspect the actual image. Match every requested subject to exactly one actual instance. Reject missing, duplicate or fused subjects. "\
            "Do not assume the generation instruction was followed. Use the supplied subject_id, assign unique target_instance_id, "\
            "and report an observed description and xyxy box relative to image width/height. Explicitly declare coordinate_system: "\
            "normalized_0_1000 for values in [0,1000], or normalized_0_1 for fractions in [0,1]. Never return pixel coordinates. "\
            "A prior integration returned 0-to-1000 values to a 0-to-1 consumer; the declared units now determine an explicit conversion. "\
            "Expected inventory: "+json.dumps(task["subjects"],ensure_ascii=False)
        result=self._model("training.resolve",{"prompt":prompt},RESOLVE_SCHEMA,[ArtifactRef.from_dict(image["image"])],phase="resolve")
        result=normalize_instance_observation(result)
        found=result["subjects"]
        if (result["status"]!="pass" or {s["subject_id"] for s in found}!={s["subject_id"] for s in task["subjects"]} or
                len(found)!=len(task["subjects"]) or len({s["target_instance_id"] for s in found})!=len(found)):
            self._write("rejected-instance-observation",result)
            raise QualityRejected("actual subject correspondence failed or remained uncertain: "+result["reason"])
        return {"instance_bindings":self._write("instance-bindings",result)}

    def segment(self,inputs):
        image=self._read(inputs["image"]);observed=self._read(inputs["instance_bindings"])
        result=self._model("training.segment",{"subjects":observed["subjects"],"threshold":self.recipe["sam_threshold"],
            "mask_threshold":self.recipe["sam_mask_threshold"],"min_box_iou":self.recipe["sam_min_box_iou"]},MASK_SCHEMA,
            [ArtifactRef.from_dict(image["image"])],phase="segment",max_tokens=1)
        expected={s["subject_id"]:s["target_instance_id"] for s in observed["subjects"]}
        if (len(result["instances"])!=len(expected) or
                {s["subject_id"]:s["target_instance_id"] for s in result["instances"]}!=expected):
            raise ValueError("segmentation changed subject-to-instance correspondence")
        return {"mask_list":self._write("mask-list",result)}

    def extract(self,inputs):
        image=self._read(inputs["image"]);all_instances=self._read(inputs["mask_list"])["instances"]
        chosen=[x for x in all_instances if x["subject_id"] in self.task["selected_subject_ids"]]
        subjects,transforms=extract_subject_references(self.store,self.dataset_store,self.family_id,ArtifactRef.from_dict(image["image"]),chosen)
        return {"subject_references":self._write("references",{"subjects":subjects,"reference_transforms":transforms,"target_instances":chosen})}

    def background(self,inputs):
        image=self._read(inputs["image"]);target=ArtifactRef.from_dict(image["image"])
        instances=self._read(inputs["mask_list"])["instances"]
        chosen=[x for x in instances if x["subject_id"] in self.task["selected_subject_ids"]]
        kept=[x for x in instances if x["subject_id"] not in self.task["selected_subject_ids"]]
        shadow=None;extra_boxes=[]
        if self.recipe.get('probe_shadows',False):
            def describe(values):
                width,height=self.recipe['width'],self.recipe['height']
                return [{'subject_id':v['subject_id'],'description':v['description'],
                    'object_box_normalized':[v['box_xyxy'][0]/width,v['box_xyxy'][1]/height,
                        v['box_xyxy'][2]/width,v['box_xyxy'][3]/height]} for v in values]
            probe_prompt="Observe the actual image and locate ONLY cast shadows or reflections attributable to the selected subjects. "\
                "Give tight rectangles on the supporting surface, not boxes around the objects themselves. "\
                "Exclude shadows of unselected objects, windows, furniture and other scene illumination. "\
                "Every selected subject_id must have observed shadow regions or appear in no_shadow_subject_ids if none is visible. "\
                "Use uncertain rather than guessing attribution. Declare normalized_0_1000 or normalized_0_1 coordinates. "\
                "Observed object boxes below use normalized_0_1 coordinates. "\
                "Selected observed instances: "+json.dumps(describe(chosen),ensure_ascii=False)+\
                "\nPreserve these instances and their shadows: "+json.dumps(describe(kept),ensure_ascii=False)
            probe_prompt+='\n'+self.recipe.get('shadow_probe_instruction','')
            if self.recipe.get('probe_visible_effects',False):
                probe_prompt=probe_prompt.replace('ONLY cast shadows or reflections','ALL visible effects including steam, vapor, cast shadows or reflections')
                probe_prompt=probe_prompt.replace('Give tight rectangles on the supporting surface, not boxes around the objects themselves.',
                    'Give tight rectangles around each visible effect on any surface or above the objects. Exclude solid object bodies already handled by segmentation.')
                probe_prompt=probe_prompt.replace('observed shadow regions','observed effect regions')
                probe_prompt+='\nInclude effects above or beside each selected object, not only effects on its support. Steam or vapor attributable to a removed object must be covered. Keep all unselected effects separate.'
            observed=self._model('training.resolve',{'prompt':probe_prompt},SHADOW_SCHEMA,[target],
                phase='shadow-probe',max_tokens=1024,purpose='active_shadow_probe')
            observed=normalize_shadow_observation(observed,self.task['selected_subject_ids'],
                preserved_ids=[s['subject_id'] for s in kept])
            observed.update(probe_request_id=self.records[-1]['request']['request_id'],source_image=target.to_dict())
            observed['observation_scope']='visible_selected_object_effects' if self.recipe.get('probe_visible_effects',False) else 'cast_shadows_and_reflections'
            shadow=self._write('shadow-observation',observed)
            extra_boxes=[s['box_xyxy_normalized'] for s in observed['shadows']]
        correction_ref = self.config['production'].get('shadow_region_corrections', {}).get(self.family_id)
        if correction_ref is not None:
            correction = self._read(ArtifactRef.from_dict(correction_ref))
            observed = validate_shadow_correction(correction, target, self.task['selected_subject_ids'])
            self.store.resolve(ArtifactRef.from_dict(correction['rejected_candidate']))
            extra_boxes.extend(s['box_xyxy_normalized'] for s in observed['shadows'])
        surface_ref=self.config['production'].get('surface_effect_contracts',{}).get(self.family_id)
        surface_artifact=None
        if surface_ref is not None:
            from .surface_effects import compile_surface_artifacts
            surface=self._read(ArtifactRef.from_dict(surface_ref))
            visible,region,surface_condition,surface_artifact=compile_surface_artifacts(
                self.store,self.dataset_store,self.family_id,target,chosen,kept,surface,
                edge_margin=self.task.get('selected_edge_margin_pixels',0),
                model_size=(self.recipe['width'],self.recipe['height']))
        else:
            visible,region=completion_region(self.store,self.dataset_store,self.family_id,target,chosen,kept,
                dilation_pixels=self.recipe["mask_dilation_pixels"],extra_boxes=extra_boxes)
        prompt=self.recipe["background_instruction"]+"\nRemove only these observed instances: "+json.dumps(
            [{"description":x["description"],"box_xyxy":x["box_xyxy"]} for x in chosen],ensure_ascii=False)+\
            "\nPreserve these other instances: "+json.dumps([x["description"] for x in kept],ensure_ascii=False)+\
            "\nComplete occluded surfaces naturally, including residual cast shadows/reflections of removed objects. Preserve camera and all other scene content."
        parameters={"prompt":prompt,**{k:self.recipe[k] for k in ("width","height","steps","true_cfg_scale")}}
        parameters['true_cfg_scale'] = self.task.get('background_true_cfg_scale', parameters['true_cfg_scale'])
        if "background_negative_prompt" in self.recipe:parameters["negative_prompt"]=self.recipe["background_negative_prompt"]
        conditioning=target
        mode=self.recipe.get("background_conditioning","source_image")
        condition_shape=self.recipe.get('background_conditioning_shape','region')
        condition_region=region
        if surface_artifact is not None:
            mode='masked_region';condition_shape='surface_effect_contract'
        if mode=="masked_region":
            if condition_shape=='bounding_rectangle':
                condition_region=rectangular_condition_region(self.store,self.dataset_store,self.family_id,region,kept)
            conditioning=surface_condition if surface_artifact is not None else masked_background_condition(self.store,self.dataset_store,self.family_id,target,condition_region)
            parameters["prompt"]="Complete the neutral gray missing regions with the natural underlying background. "\
                "These regions must contain background surfaces only, with no removed objects, object fragments, shadows or reflections. "\
                "Match the surrounding surface texture, perspective, illumination and color. "\
                "Keep all visible content outside the missing regions unchanged.\n"+self.recipe["background_instruction"]+\
                "\nKeep these visible unselected objects unchanged: "+json.dumps([x["description"] for x in kept],ensure_ascii=False)
        transform=None
        if surface_artifact is not None:
            if surface.get('reconstruction'):
                from .context_reconstruction import prepare_context_condition
                conditioning,transform=prepare_context_condition(self.store,self.dataset_store,self.family_id,
                    target,conditioning,region,surface['reconstruction'],model_size=(self.recipe['width'],self.recipe['height']))
                condition_shape='surface_context_crop'
                # Polygon coordinates remain in source space in the audit
                # contract. The model sees the transformed crop, so only surface
                # descriptions (no misleading original coordinates) enter text.
                parameters['prompt']+='\nThis is an actual square context crop of the source. Reconstruct missing surfaces using the visible local context; preserve perspective and surface continuity. Observed surfaces: '+json.dumps([
                    x['description'] for x in surface['observation']['surfaces']],ensure_ascii=False)
            else:
                parameters['prompt']+='\nObserved receiving surfaces and attributable effects: '+json.dumps({
                    'surfaces':surface['observation']['surfaces'],'effects':surface['observation']['effects'],
                    'preservation_regions':surface['observation']['preservation_regions'],
                    'uncertain_regions':surface['observation']['uncertain_regions']},ensure_ascii=False)
        refinement=None
        if self.recipe.get('refine_background_prompt',False):
            parameters['prompt'],refinement=self.refine_prompt(parameters['prompt'],mode='background',artifacts=[conditioning])
        output=self._model("training.background",parameters,
                           IMAGE_SCHEMA,[conditioning],phase="background",max_tokens=1)
        if not self.allow_fixture and output["actual_parameters"]!=json.loads(self.records[-1]["request"]["messages"][0]["content"]):
            raise ValueError("background backend parameters differ from the actual request")
        raw=ArtifactRef.from_dict(output["artifact"])
        model_raw=raw;projection=None
        if transform is not None:
            from .context_reconstruction import project_context_output
            raw,projection=project_context_output(self.store,self.dataset_store,self.family_id,target,raw,transform)
        unharmonized=raw;harmonization=None
        if surface_artifact is not None and surface.get('implementation',{}).get('composition'):
            if projection is None:raise ValueError('surface composition requires a recorded context projection')
            from .surface_composition import harmonize_surface_edit
            raw,harmonization=harmonize_surface_edit(self.store,self.dataset_store,self.family_id,
                target,raw,surface_artifact,projection)
        blend=region if surface_artifact is not None else completion_blend_region(self.store,self.dataset_store,self.family_id,region,
            feather_pixels=self.recipe.get("background_feather_pixels",0),selected=chosen,opaque_boxes=extra_boxes,
            selected_edge_margin_pixels=self.task.get('selected_edge_margin_pixels',0))
        completed=complete_background_composite(self.store,self.dataset_store,self.family_id,target,raw,blend)
        surface_audit=None
        if surface_artifact is not None:
            from .surface_effects import audit_surface_composition
            surface_audit=audit_surface_composition(self.store,target,raw,completed,region,
                ArtifactRef.from_dict(self._read(surface_artifact)['protected']))
        return {"background_visible":visible,"background_completed":self._write("background",{"image":completed.to_dict(),
            "background_visible":visible.to_dict(),"completion_region":region.to_dict(),"raw_edit":raw.to_dict(),
            "conditioning_mode":mode,"conditioning_image":conditioning.to_dict(),
            "conditioning_shape":condition_shape,"conditioning_region":condition_region.to_dict(),
            "completion_blend":blend.to_dict(),"feather_pixels":0 if surface_artifact is not None else self.recipe.get("background_feather_pixels",0),
            "selected_edge_margin_pixels":self.task.get('selected_edge_margin_pixels',0),
            "shadow_observation":shadow.to_dict() if shadow is not None else None,
            "shadow_region_correction":correction_ref,
            **({'surface_effect_contract':surface_artifact.to_dict(),'surface_composition_audit':surface_audit}
               if surface_artifact is not None else {}),
            **({'context_reconstruction':projection.to_dict(),'raw_model_output':model_raw.to_dict()} if projection is not None else {}),
            **({'surface_harmonization':harmonization.to_dict(),'unharmonized_edit':unharmonized.to_dict()} if harmonization is not None else {}),
            **({'prompt_refinement':refinement.to_dict()} if refinement is not None else {}),
            "source_request_id":self.records[-1]["request"]["request_id"],
            "inferred":True,"dilation_pixels":0 if surface_artifact is not None else self.recipe["mask_dilation_pixels"],"preserved_instances":kept})}

    def assemble(self,inputs):
        background=self._read(inputs["background"]);reference=self._read(inputs["subject_references"])
        target=self._read(inputs["target_image"]);scene=self._read(inputs["scene_plan"])
        observed=self._read(inputs["instance_bindings"])
        expected={s["subject_id"]:s["target_instance_id"] for s in observed["subjects"]}
        if any(expected.get(s["subject_id"])!=s["target_instance_id"] for s in reference["target_instances"]):
            raise ValueError("reference extraction changed observed target identities")
        row={"schema_version":"dataevolver.supervision.v1","evidence_kind":self.config["production"]["evidence_kind"],
            "id":self.family_id,"family_id":self.family_id,"split":self.task["split"],"task_type":"multi_subject_insertion",
            "construction_mode":"scene_decomposition","status":"pending",
            "input":{"background":background["image"],"subjects":reference["subjects"],"instruction":self.task["insertion_instruction"]},
            "target":{"image":target["image"]},"annotations":{"target_instances":reference["target_instances"],
                "reference_transforms":reference["reference_transforms"],"background_visible":background["background_visible"],
                "instance_bindings":observed,
                "completion_region":background["completion_region"],"background_completion":background,
                "selection_mode":self.task["selection_mode"],"original_goal":scene["original_goal"],
                "provenance":{"origin":"fixture" if self.allow_fixture else "synthetic","source_family_id":self.family_id,"source_image":target["image"],
                              "source_request_id":target["request_id"],
                              "scene_task":self.task,"target_kind":"original_complete_scene","hidden_background":"inferred"},
                "recipe":self.recipe,"review":{"status":"pending","criteria":{}}},"model_receipts":list(self.records)}
        return {"training_record":self._write("training-record",row)}

    def _review(self,row,*,phase,negative=False):
        target=ArtifactRef.from_dict(row["target"]["image"])
        background=target if negative else ArtifactRef.from_dict(row["input"]["background"])
        artifacts=[target,background]+[ArtifactRef.from_dict(x["reference"]) for x in row["input"]["subjects"]]
        prompt="Images: 0=original complete target scene,1=candidate input background,remaining=subject references in listed order. "\
            "Judge the actual images independently of generation instructions. Required criteria: correct subject-to-instance mapping; "\
            "clean references preserving identity; all selected objects removed from background; unselected objects and other content preserved; "\
            "natural background completion without holes, seams, leftover object edges, cast shadows or reflections. "\
            "Use uncertain if a required property cannot be confirmed. Original immutable requirement: "+json.dumps(self.config["planning"]["goal"],ensure_ascii=False)+\
            "\nSubject reference order: "+json.dumps([x["id"] for x in row["input"]["subjects"]])+\
            "\nSelection and original instance mapping: "+json.dumps(row["annotations"]["target_instances"],ensure_ascii=False)
        result=self._model("training.review",{"prompt":prompt},REVIEW_SCHEMA,artifacts,phase=phase,
                           purpose="checker_negative" if negative else "quality_review")
        for criterion in result["criteria"].values():criterion["evidence"]=[r.to_dict() for r in artifacts]
        return result

    def review(self,inputs):
        import numpy as np
        from PIL import Image
        row=self._read(inputs["training_record"])
        instances=self._read(inputs["mask_list"])["instances"]
        if inputs["background_visible"].to_dict()!=row["annotations"]["background_visible"]:
            raise ValueError("review background evidence changed")
        target=np.asarray(Image.open(self.store.resolve(ArtifactRef.from_dict(row["target"]["image"]))).convert("RGB"))
        background=np.asarray(Image.open(self.store.resolve(ArtifactRef.from_dict(row["input"]["background"]))).convert("RGB"))
        region=np.asarray(Image.open(self.store.resolve(ArtifactRef.from_dict(row["annotations"]["completion_region"]))).convert("L"))>127
        if target.shape!=background.shape or region.shape!=target.shape[:2] or not np.array_equal(target[~region],background[~region]):
            raise ValueError("background completion changed protected image content")
        for instance in instances:
            if instance["subject_id"] not in self.task["selected_subject_ids"]:
                mask=np.asarray(Image.open(self.store.resolve(ArtifactRef.from_dict(instance["mask"]))).convert("L"))>127
                if not np.array_equal(target[mask],background[mask]):
                    raise ValueError("an unselected instance was modified")
        # A real unchanged-background negative case costs a complete review call.
        if self.config["production"]["checker_precheck"]:
            negative=self._review(row,phase="negative-check",negative=True)
            if not negative_rejected(negative):
                raise ValueError("reviewer did not confirm failed removal in the unchanged background")
            row["annotations"]["checker_negative"]={"rejected":True,"review":negative,
                "scope":"negative rejection only; not a calibration or quality accuracy claim"}
        result=self._review(row,phase="review")
        passed=result["status"]=="pass" and all(result["criteria"][k]["status"]=="pass" for k in QUALITY_CHECKS)
        row["annotations"]["review"]=result;row["status"]="pass" if passed else ("uncertain" if result["status"]=="uncertain" else "fail")
        row["model_receipts"]=list(self.records)
        if passed:validate_record(row,self.store,allow_fixture=self.allow_fixture)
        return {"review_evidence":self._write("review-evidence",{"record":row,"review":result})}

    def run_family(self,task,attempt_index):
        self.task={**task,"seed":task["seed"]+attempt_index*1000003,"attempt_index":attempt_index}
        self.family_id=f"{task['slot_id']}-a{attempt_index}";self.records,self.attempts=[],[]
        out={};mapping={"scene.plan":self.scene_plan,"image.generate":self.generate,"subjects.resolve":self.resolve,
            "subjects.segment":self.segment,"subjects.extract":self.extract,"background.complete":self.background,
            "sample.assemble":self.assemble,"sample.review":self.review}
        pending=[n for n in self.config["planning"]["plan"]["dataset_plan"]["nodes"] if n["operator"]!="dataset.export"]
        while pending:
            ready=[n for n in pending if all(dep in out for dep in n["depends_on"])]
            if not ready:raise ValueError("unresolved production graph dependency")
            for node in ready:
                inputs={}
                for binding in node["input_bindings"]:
                    source=binding["source"]
                    if source.startswith("node:"):
                        parent,port=source[5:].split(".",1);inputs[binding["port"]]=out[parent][port]
                    else:
                        goal=self.config["planning"]["goal"]
                        if source.startswith("goal:"):
                            value=_field({"request":goal["original_request"],**goal},source[5:])
                        elif source.startswith("config:"):
                            config=goal["task_config"];value=_field({"task_config":config,**config},source[7:])
                        else:raise ValueError("unsupported material binding for no-source production")
                        inputs[binding["port"]]=self._write("bound-"+binding["port"],value)
                out[node["id"]]=self.step(node["operator"],inputs,mapping[node["operator"]]);pending.remove(node)
        review_id=self.nodes["sample.review"]["id"]
        self.last_outputs={n:{k:r.to_dict() for k,r in ports.items()} for n,ports in out.items()}
        return self._read(out[review_id]["review_evidence"])["record"]

    def export(self,rows,destination,family_outputs):
        self.family_id="dataset-aggregate";self.task={"families":[r["family_id"] for r in rows],"split_counts":self.config["production"]["split_counts"]}
        self.records=[]
        assignments={"training_records":("sample.assemble","training_record"),"review_evidence":("sample.review","review_evidence"),
                     "family_assignments":("scene.plan","scene_plan")}
        inputs={name:self._write("aggregate-"+name,[f[self.nodes[op]["id"]][port] for f in family_outputs])
                for name,(op,port) in assignments.items()}
        inputs["task_config"]=self._write("aggregate-task-config",self.config["planning"]["goal"]["task_config"])
        def execute(bound):
            drafts=[self._read(ArtifactRef.from_dict(ref)) for ref in self._read(bound["training_records"])]
            reviews=[self._read(ArtifactRef.from_dict(ref))["record"] for ref in self._read(bound["review_evidence"])]
            if reviews!=rows or len(drafts)!=len(rows):raise ValueError("export rows differ from their bound review outputs")
            for draft,row in zip(drafts,rows):
                if any(draft[k]!=row[k] for k in ("id","family_id","input","target","construction_mode","split")):
                    raise ValueError("review changed original supervision or split")
            export_dataset(rows,self.store,destination,expected_counts=self.config["production"]["split_counts"],
                metadata={"planning_goal":self.config["planning"]["goal"],"plan":self.config["planning"]["plan"],
                    "config_hash":payload_sha256(self.config),"usage":self.gateway.audit_usage(),
                    "evidence_kind":self.config["production"]["evidence_kind"],"quality_scope":"production and loading acceptance only"},
                allow_fixture=self.allow_fixture)
            return {"dataset_bundle":ArtifactRef(self.config["bundle_store"],"manifest.json",file_sha256(Path(destination)/"manifest.json"))}
        return self.step("dataset.export",inputs,execute)
