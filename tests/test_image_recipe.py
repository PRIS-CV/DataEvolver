import copy
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
from dataevolver.adapters.model_gateway import ModelGateway
from dataevolver.dataset.contracts import ArtifactRef, canonical_json, file_sha256, payload_sha256
from dataevolver.dataset.curation_loader import load_export
from dataevolver.dataset.quality import counts, fit_blur_calibration, map_rows, select_balanced
from dataevolver.harness.adapters import image_recipe
from dataevolver.harness.adapters.data_graph import DataGraphOperator
from dataevolver.harness.adapters.image_recipe import ImageRecipeOperator, RecipePlanner
from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.data_contracts import DataInput, DataNodeGraph, DataNodeSpec, DataRunBudget
from dataevolver.harness.evaluator import EvaluatorContract
from dataevolver.harness.models import EvidenceStatus, GoalSpec, OperatorCall, RecoveryStatus
from dataevolver.harness.operators import OperatorRegistry
from dataevolver.harness.task_contracts import DatasetIntent, QualityPolicy, PROTECTED_GATES


def pool_fixture(tmp_path):
    raw, dataset = tmp_path/"raw", tmp_path/"pool-data"
    raw.mkdir()
    license_path = raw/"license.txt"; license_path.write_text("Fixture MIT")
    evidence = ArtifactRef("raw", license_path.name, file_sha256(license_path))
    sources, mapping, quotas = [], {"schema_version": "dataevolver.label_mapping.v1", "sources": {}}, {}
    for si, source in enumerate(("a", "b")):
        values, mapping["sources"][source], quotas[source] = [], {}, {}
        for label in range(3):
            name = source+"-"+str(label)
            mapping["sources"][source][str(label)] = {"label_id": name, "name": name, "evidence": "fixture known labels"}
            quotas[source][name] = 11
            for i in range(16):
                rng = np.random.default_rng(si*1000+label*100+i)
                pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                image = Image.fromarray(pixels)
                out = io.BytesIO(); image.save(out, "PNG")
                values.append({"image": {"bytes": out.getvalue(), "path": "ignored"}, "label": label, "family": f"{label}-{i}"})
        path = raw/(source+".parquet"); pq.write_table(pa.Table.from_pylist(values), path)
        sources.append({"source_id":source,"dataset":"fixture/"+source,"revision":"v1","source_split":"train",
                        "file":ArtifactRef("raw",path.name,file_sha256(path)).to_dict(),"license":"mit",
                        "license_evidence":[evidence.to_dict()],"image_column":"image","label_column":"label","family_column":"family"})
    path = raw/"sources.json"; path.write_text(canonical_json({"schema_version":"dataevolver.image_sources.v1","sources":sources}))
    source_ref=ArtifactRef("raw",path.name,file_sha256(path),"application/json")
    path=raw/"mapping.json";path.write_text(canonical_json(mapping))
    mapping_ref=ArtifactRef("raw",path.name,file_sha256(path),"application/json")
    nodes=[]
    bindings=[("audit",{"max_rows_per_source":48,"min_side":16,"max_pixels":100000}),
              ("labels",{}),("families",{"hamming_distance":0}),
              ("partition",{"quotas":quotas,"per_label_roles":{"development":4,"calibration":2,"canary":2,"reuse":2,"sealed":1},"seed":17}),
              ("clean",{"max_side":64}), ("calibrate_quality",{"blur_radius":4,"minimum_validation_accuracy":.9}),
              ("export",{"require_semantic_match":False})]
    previous=None
    for name,parameters in bindings:
        inputs={"sources":DataInput(artifact=source_ref)} if previous is None else {"catalog":DataInput(node=previous,output="catalog")}
        if name=="labels":inputs["mapping"]=DataInput(artifact=mapping_ref)
        nodes.append(DataNodeSpec(name,"images."+name,"1","data",inputs,parameters));previous=name
    contract={"intent":DatasetIntent("fixture-pool","curate","image_curation","curated_images.v1","Fixture pool",(source_ref,mapping_ref)).to_dict(),
              "quality_policy":QualityPolicy("fixture",tuple(sorted(PROTECTED_GATES)),tuple(n.operator for n in nodes),{"allowed_licenses":["mit"]},1).to_dict(),
              "graph":DataNodeGraph(tuple(nodes)).to_dict(),"budget":DataRunBudget(9,0,0,96,10000000,120).to_dict(),
              "stores":{"raw":"RAW","dataset":"DATASET"},"dataset_store":"dataset","models":{}}
    op=DataGraphOperator(allowed_write_root=tmp_path/"pool-campaign",contract=contract,roots={"raw":raw,"dataset":dataset})
    call=OperatorCall("pool","pool","pool","data.image_curation","2",0,str(tmp_path/"pool-campaign/run"),parameters=contract)
    receipt=op.execute(call)
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    run=json.loads((Path(call.work_dir)/"node_runs/export/node_run.json").read_text())
    ref=ArtifactRef.from_dict(run["outputs"]["dataset"])
    return op.store.resolve(ref).parent, {s:{label:1 for label in labels} for s,labels in quotas.items()}


def recipe_fixture(tmp_path, monkeypatch):
    pool,quotas=pool_fixture(tmp_path)
    profile=ModelProfile("fixture","mock","fixture-local","v1",("json","vision"))
    contract={"pool":ArtifactRef("pool","dataset.json",file_sha256(pool/"dataset.json"),"application/json").to_dict(),
              "stores":{"pool":"POOL","dataset":"DATASET"},"dataset_store":"dataset","quotas":quotas,"seed":23,
              "baseline":{"ranking":"hash"},"models":{"profiles":[profile.to_dict()],
              "policy":ModelPolicy({"planner":"fixture","reviewer":"fixture"},32).to_dict(),"workers":{}},
              "budget":DataRunBudget(6,32,96,66,10000000,120).to_dict(),
              "review":{"batch_size":4,"max_tokens":256,"timeout_seconds":20,"minimum_pair_accuracy":.8}}
    op=ImageRecipeOperator(allowed_write_root=tmp_path/"campaigns/recipe",contract=contract,
                           roots={"pool":pool,"dataset":tmp_path/"candidates"})
    calls=[]
    class Provider:
        def lookup(self,request,request_hash):
            p=op.root/"fixture-native"/(request.request_id+".json")
            if not p.exists():raise RuntimeError("ambiguous missing fixture effect")
            v=json.loads(p.read_text());assert v["request_hash"]==request_hash
            return v["response"]
        def invoke(self,request,request_hash):
            calls.append(request.request_id)
            if request.role=="planner":
                output={"ranking":request.response_schema["properties"]["ranking"]["enum"][0]}
            elif request.request_id.startswith("calibration"):
                output={"clearer":next(i for i,r in enumerate(request.artifacts) if "-blur" not in r.path)}
            else:
                output={"scores":[int(r.sha256[0],16)%5 for r in request.artifacts]}
            response={"model":profile.model,"output":output,"finish_reason":"stop","usage":{"images":len(request.artifacts),"input_tokens":1,"output_tokens":1}}
            p=op.root/"fixture-native"/(request.request_id+".json");p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(canonical_json({"request_hash":request_hash,"response":response}))
            return response
    monkeypatch.setattr(op,"gateway",lambda:ModelGateway(profiles=op.profiles,policy=op.policy,artifact_store=op.store,
        receipt_root=op.root/"model_receipts",providers={"fixture":Provider}))
    return op,calls


def test_mapping_is_explicit_and_preserves_raw_label():
    mapping={"schema_version":"dataevolver.label_mapping.v1","sources":{"a":{"0":{"label_id":"a-healthy","name":"healthy","evidence":"source card"}}}}
    result=map_rows([{"source_id":"a","source_label":0}],mapping)[0]
    assert result["source_label"]==0 and result["label_id"]=="a-healthy"
    with pytest.raises(ValueError,match="unmapped"):map_rows([{"source_id":"a","source_label":1}],mapping)


def test_quota_matching_finds_feasible_shared_family_assignment():
    rows=[{"id":"a1","source_id":"a","label_id":"one","family_id":"shared"},
          {"id":"a2","source_id":"a","label_id":"one","family_id":"a-only"},
          {"id":"b1","source_id":"b","label_id":"two","family_id":"shared"}]
    quotas={"a":{"one":1},"b":{"two":1}}
    chosen=select_balanced(rows,quotas)
    assert counts(chosen)==quotas and len({r["family_id"] for r in chosen})==2
    with pytest.raises(ValueError,match="infeasible"):select_balanced(rows,{"a":{"one":2},"b":{"two":1}})


def test_calibration_validation_never_changes_fitted_threshold():
    pairs=[{"family_id":"fit","partition":"fit","reference":.8,"blurred":.1},
           {"family_id":"held","partition":"validation","reference":.7,"blurred":.2}]
    a=fit_blur_calibration(pairs)
    pairs[1]["reference"],pairs[1]["blurred"]=.01,.99
    b=fit_blur_calibration(pairs)
    assert a["threshold"]==b["threshold"] and a["validation_accuracy"]>b["validation_accuracy"]
    assert a["human_gold"] is False and a["semantic_quality_calibrated"] is False


def test_pool_has_all_labels_and_disjoint_roles(tmp_path):
    pool,quotas=pool_fixture(tmp_path);rows=load_export(pool)
    assert len(rows)==66 and len({r["family_id"] for r in rows})==66
    assert {r["evaluation_role"] for r in rows}=={"development","calibration","canary","reuse","sealed"}
    assert all("quality" not in r for r in rows if r["evaluation_role"]=="sealed")
    assert len({r["label_id"] for r in rows})==6


def test_recipe_recovery_after_effect_does_not_repeat_models(tmp_path,monkeypatch):
    op,calls=recipe_fixture(tmp_path,monkeypatch)
    call=OperatorCall("recipe","initial","initial",op.spec.operator_id,"1",0,str(op.root/"initial"),parameters={"recipe":{"ranking":"hash"},"phase":"search"})
    original=image_recipe._write_receipt
    def crash(*args):raise RuntimeError("after effect")
    monkeypatch.setattr(image_recipe,"_write_receipt",crash)
    with pytest.raises(RuntimeError,match="after effect"):op.execute(call)
    count=len(calls);journal=json.loads((op.root/"recipe_budget.json").read_text())
    monkeypatch.setattr(image_recipe,"_write_receipt",original)
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    assert op.execute(call).reused and len(calls)==count
    assert json.loads((op.root/"recipe_budget.json").read_text())==journal
    sealed={r["image_sha256"] for r in op.rows if r["evaluation_role"]=="sealed"}
    review_requests=[entry["request"] for p in op.root.glob("review*.json") for entry in json.loads(p.read_text())["requests"]]
    assert review_requests
    assert all(not sealed.intersection(i["sha256"] for i in request["artifacts"]) for request in review_requests)
    with pytest.raises(ValueError,match="only"):op.compile_recipe({"ranking":"clarity","quotas":{}})
    with pytest.raises(ValueError,match="duplicate"):op.compile_recipe({"ranking":"clarity"},["clarity"])


@pytest.mark.parametrize("minimum_improvement",[0.,2.])
def test_controller_bounded_comparison_adoption_rollback_and_replay(tmp_path,monkeypatch,minimum_improvement):
    op,calls=recipe_fixture(tmp_path,monkeypatch)
    evaluator={"schema_version":"dataevolver.harness.evaluator.v1","evaluator_id":"fixture","revision":"1",
               "hard_constraints":{n:{"required":True} for n in ("artifact_valid","quotas_met","pool_fixed","sealed_unread")},
               "target_metrics":{"clarity_proxy":{"required":True,"minimum":0.,"minimum_improvement":minimum_improvement}},
               "protected_metrics":{"class_coverage":{"required":True,"max_regression":0.}}}
    path=tmp_path/"evaluator.json";path.write_text(canonical_json(evaluator))
    key={"input_artifact_hash":op.pool_ref.sha256,"parent_artifact_hash":"none","seed":23,"operator_id":op.spec.operator_id,
         "operator_version":"1","model_config_hash":op.spec.model_config_hash,"preprocessing_hash":op.contract_hash,
         "evaluator_hash":file_sha256(path),"environment_contract_hash":"fixture"}
    goal=GoalSpec.from_dict({"schema_version":"dataevolver.harness.goal.v1","campaign_id":"recipe",
         "objective":{"kind":"data_recipe_rsi","real_data_execution":True,"initial_trial_key":key,"canary_trial_key":key,"reuse_trial_key":{**key,"input_artifact_hash":"reuse"}},
         "initial_operator_id":op.spec.operator_id,"initial_operator_parameters":{"recipe":{"ranking":"hash"},"phase":"search"},
         "route_candidates":["image_recipe"],"budgets":{"max_rounds":4,"max_operator_calls":5,"max_experiments":3,"max_retries":0,"max_wall_seconds":120,"max_planner_calls":3},
         "resource_policy":{"allow_gpu":False,"allow_network":False,"allowed_operators":[op.spec.operator_id]},
         "frozen_evaluator":{"path":str(path),"sha256":file_sha256(path)},"planner":{"kind":"recipe_rsi"},"interventions":[]})
    registry=OperatorRegistry();registry.register(op)
    controller=EvolutionController.create(repo_root=tmp_path,campaigns_root=op.root.parent,goal=goal,registry=registry,
          planner=RecipePlanner(op),evaluator_contract=EvaluatorContract.from_path(path),idempotency_key="fixture")
    state=controller.run()
    assert state.state.value=="COMPLETE" and state.experiments_used==3 and state.planner_calls==3 and state.operator_calls==5
    assert len(state.data["recipe_history"])==3
    if minimum_improvement==2.:
        assert state.data["incumbent_recipe"]=={"ranking":"hash"} and not state.data["recipe_improvement_verified"]
        assert all(h["decision"]=="ROLLBACK" for h in state.data["recipe_history"])
    else:
        assert any(h["decision"]=="ADOPT_CANDIDATE" for h in state.data["recipe_history"])
    count=len(calls);seq=state.last_event_seq
    assert controller.resume().last_event_seq==seq and len(calls)==count
    assert controller.replay().to_dict()==controller.status().to_dict()
    planner_requests=[json.loads(p.read_text())["request"] for p in (op.root/"planner").glob("*.json")]
    assert len(planner_requests)==3 and all(not r["artifacts"] for r in planner_requests)


def test_missing_shared_budget_fails_closed(tmp_path,monkeypatch):
    op,calls=recipe_fixture(tmp_path,monkeypatch)
    op.reviewer_calibration(); count=len(calls)
    (op.root/"recipe_budget.json").unlink()
    with pytest.raises(ValueError,match="budget missing"):op.journal()
    assert len(calls)==count


def test_shared_image_limit_rejects_before_model_and_failure_can_resume(tmp_path,monkeypatch):
    op,calls=recipe_fixture(tmp_path,monkeypatch)
    from dataclasses import replace
    op.budget=replace(op.budget,max_model_images=1)
    call=OperatorCall("recipe","failed","failed",op.spec.operator_id,"1",0,str(op.root/"failed"),parameters={"recipe":{"ranking":"hash"},"phase":"search"})
    receipt=op.execute(call)
    assert receipt.status.value=="FAILED" and not calls
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    assert op.execute(call).reused and not calls


def test_sealed_image_cannot_be_given_to_review_model(tmp_path,monkeypatch):
    op,calls=recipe_fixture(tmp_path,monkeypatch)
    row=next(r for r in op.rows if r["evaluation_role"]=="sealed")
    request=op._model_task("illegal-sealed","test",{"type":"object"},[op.image_ref(row)])
    with pytest.raises(ValueError,match="non-sealed"):op.invoke(request)
    assert not calls and not (op.root/"recipe_budget.json").exists()
