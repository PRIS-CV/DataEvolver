"""Explicit media fixtures test contracts; no real-model quality evidence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway
from dataevolver.dataset.contracts import ArtifactStore, ArtifactRef
from dataevolver.dataset.supervision import (write_image, extract_subject_references, completion_region,
    complete_background_composite, masked_background_condition, rectangular_condition_region,
    completion_blend_region, dilate_binary_region, rectangle_region, validate_record, export_dataset, load_dataset)


@pytest.fixture
def fixture_media(tmp_path):
    store = ArtifactStore({"data": tmp_path/"data"})
    target = write_image(store, "data", "fixture/target.png", Image.new("RGB", (32, 24), "white"))
    selected = np.zeros((24, 32), dtype="uint8"); selected[4:12, 4:12] = 255
    kept = np.zeros_like(selected); kept[4:12, 12:20] = 255
    instances = [{"subject_id":"s1","target_instance_id":"i1","mask":write_image(store,"data","fixture/s1.png",Image.fromarray(selected)).to_dict()},
                 {"subject_id":"s2","target_instance_id":"i2","mask":write_image(store,"data","fixture/s2.png",Image.fromarray(kept)).to_dict()}]
    return store, target, instances


def test_rectangular_condition_hides_silhouette_without_widening_composite(fixture_media):
    store,target,instances=fixture_media
    shape=np.zeros((24,32),dtype='uint8')
    shape[3:18,3:7]=255;shape[14:18,3:25]=255
    region=write_image(store,'data','fixture/completion.png',Image.fromarray(shape))
    protected_before=Image.open(store.resolve(region)).tobytes()
    rectangular=rectangular_condition_region(store,'data','rectangle',region,[instances[1]])
    condition=masked_background_condition(store,'data','rectangle',target,rectangular)
    pixels=np.asarray(Image.open(store.resolve(condition)))
    assert list(pixels[10,9])==[127,127,127]  # The shape's interior cue is hidden.
    assert list(pixels[6,14])==[255,255,255]  # A preserved object remains visible.
    proposal=write_image(store,'data','fixture/raw.png',Image.new('RGB',(32,24),'red'))
    result=complete_background_composite(store,'data','rectangle',target,proposal,region)
    assert list(np.asarray(Image.open(store.resolve(result)))[10,9])==[255,255,255]
    assert Image.open(store.resolve(region)).tobytes()==protected_before


def test_rectangular_condition_rejects_preserved_content_in_completion(fixture_media):
    store,_,instances=fixture_media
    mask=Image.open(store.resolve(ArtifactRef.from_dict(instances[1]['mask'])))
    region=write_image(store,'data','fixture/overlap.png',mask)
    with pytest.raises(ValueError,match='overlaps a preserved'):
        rectangular_condition_region(store,'data','invalid',region,[instances[1]])


def test_references_keep_subject_binding_and_exact_coordinates(fixture_media):
    store, target, instances = fixture_media
    subjects, transforms = extract_subject_references(store,"data","fixture",target,instances)
    assert [s["id"] for s in subjects] == ["s1","s2"]
    assert transforms[0]["target_bbox_xyxy"] == [4,4,12,12]
    ref = Image.open(store.resolve(ArtifactRef.from_dict(subjects[0]["reference"])))
    assert ref.mode == "RGBA" and ref.size == (8,8)
    with pytest.raises(ValueError,match="duplicate"):
        extract_subject_references(store,"data","fixture",target,[instances[0],instances[0]])


def test_masked_condition_removes_only_selected_region_and_preserves_target(fixture_media):
    store,target,instances=fixture_media
    visible,region=completion_region(store,'data','condition-fixture',target,[instances[0]],[instances[1]],dilation_pixels=3)
    condition=masked_background_condition(store,'data','condition-fixture',target,region)
    original=np.asarray(Image.open(store.resolve(target)).convert('RGB'))
    mask=np.asarray(Image.open(store.resolve(region)))>127
    actual=np.asarray(Image.open(store.resolve(condition)).convert('RGB'))
    assert np.all(actual[mask]==127)
    assert np.array_equal(original[~mask],actual[~mask])
    kept=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(instances[1]['mask']))))>127
    assert np.array_equal(original[kept],actual[kept])
    assert condition.sha256!=target.sha256 and np.all(original==255)


def test_feather_stays_inside_authorized_region_and_keeps_generated_core(tmp_path):
    store=ArtifactStore({'data':tmp_path/'data'})
    original=write_image(store,'data','original.png',Image.new('RGB',(64,64),'red'))
    proposal=write_image(store,'data','proposal.png',Image.new('RGB',(64,64),'blue'))
    mask=np.zeros((64,64),dtype='uint8');mask[8:56,8:56]=255;mask[8:15,28:36]=0
    region=write_image(store,'data','mask.png',Image.fromarray(mask))
    blend=completion_blend_region(store,'data','fixture',region,feather_pixels=4)
    alpha=np.asarray(Image.open(store.resolve(blend)))
    assert np.all(alpha[mask==0]==0) and alpha[32,32]==255 and np.any((alpha>0)&(alpha<255))
    result=complete_background_composite(store,'data','fixture',original,proposal,blend)
    pixels=np.asarray(Image.open(store.resolve(result)))
    assert np.array_equal(pixels[mask==0],np.asarray(Image.open(store.resolve(original)))[mask==0])
    assert list(pixels[32,32])==[0,0,255]


def test_feather_cannot_blend_selected_object_back_into_completed_background(fixture_media):
    store,target,instances=fixture_media
    _,region=completion_region(store,'data','blend-fixture',target,[instances[0]],[instances[1]],dilation_pixels=1)
    blend=completion_blend_region(store,'data','blend-fixture',region,feather_pixels=4,selected=[instances[0]])
    alpha=np.asarray(Image.open(store.resolve(blend)))
    selected=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(instances[0]['mask']))))>127
    kept=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(instances[1]['mask']))))>127
    assert np.all(alpha[selected]==255) and np.all(alpha[kept]==0)


def test_one_pixel_selected_edge_margin_removes_feather_leak_and_protects_other_pixels(fixture_media):
    store,target,instances=fixture_media
    _,region=completion_region(store,'data','edge-fixture',target,[instances[0]],[instances[1]],dilation_pixels=3)
    original=completion_blend_region(store,'data','edge-before',region,feather_pixels=4,selected=[instances[0]])
    corrected=completion_blend_region(store,'data','edge-after',region,feather_pixels=4,
        selected=[instances[0]],selected_edge_margin_pixels=1)
    before=np.asarray(Image.open(store.resolve(original)))
    after=np.asarray(Image.open(store.resolve(corrected)))
    assert before[7,3]<255 and after[7,3]==255
    allowed=np.asarray(Image.open(store.resolve(region)))>127
    kept=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(instances[1]['mask']))))>127
    assert np.all(after[~allowed]==0) and np.all(after[kept]==0) and after[7,12]==0
    proposal=write_image(store,'data','edge-proposal.png',Image.new('RGB',(32,24),'blue'))
    result=complete_background_composite(store,'data','edge-result',target,proposal,corrected)
    pixels=np.asarray(Image.open(store.resolve(result)))
    assert list(pixels[7,3])==[0,0,255]
    assert np.array_equal(pixels[~allowed],np.asarray(Image.open(store.resolve(target)))[~allowed])


@pytest.mark.parametrize('radius',[0,1,3,8])
def test_fast_binary_dilation_matches_rank_filter_including_image_edges(radius):
    from PIL import ImageFilter
    mask=np.zeros((17,23),dtype='uint8');mask[0,0]=255;mask[8,12]=255;mask[-1,-1]=255
    expected=mask>127 if radius==0 else np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(2*radius+1)))>127
    assert np.array_equal(dilate_binary_region(mask>127,radius),expected)


def test_observed_shadow_regions_are_completed_while_preserved_instances_stay_protected(fixture_media):
    store,target,instances=fixture_media;box=[0,.5,.5,.9]
    _,region=completion_region(store,'data','shadow-fixture',target,[instances[0]],[instances[1]],dilation_pixels=1,extra_boxes=[box])
    blend=completion_blend_region(store,'data','shadow-fixture',region,feather_pixels=3,selected=[instances[0]],opaque_boxes=[box])
    alpha=np.asarray(Image.open(store.resolve(blend)))
    shadow=rectangle_region(alpha.shape,[box]);kept=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(instances[1]['mask']))))>127
    assert np.all(alpha[shadow & ~kept]==255) and np.all(alpha[kept]==0)
    with pytest.raises(ValueError,match='normalized'):rectangle_region(alpha.shape,[[0,0,1000,1000]])


def test_reference_does_not_hide_neighbor_rgb_under_transparency(fixture_media):
    store,target,instances=fixture_media
    mask=np.zeros((24,32),dtype="uint8");mask[4:12,4:12]=255;mask[6:8,6:8]=0
    changed=copy.deepcopy(instances[0]);changed["mask"]=write_image(store,"data","fixture/hole-mask.png",Image.fromarray(mask)).to_dict()
    subjects,_=extract_subject_references(store,"data","clean-reference",target,[changed])
    pixels=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(subjects[0]["reference"]))))
    assert (pixels[pixels[:,:,3]==0,:3]==0).all()


def test_completion_protects_unselected_object_and_outside_pixels(fixture_media):
    store,target,instances=fixture_media
    visible,region=completion_region(store,"data","fixture",target,instances[:1],instances[1:],dilation_pixels=4)
    mask=np.asarray(Image.open(store.resolve(region)))
    assert not mask[4:12,12:20].any()
    proposal=write_image(store,"data","fixture/proposal.png",Image.new("RGB",(32,24),"red"))
    completed=complete_background_composite(store,"data","fixture",target,proposal,region)
    pixels=np.asarray(Image.open(store.resolve(completed)))
    assert (pixels[mask==0]==255).all()
    assert tuple(pixels[6,6])==(255,0,0)
    assert Image.open(store.resolve(visible)).mode=="RGBA"


def fixture_record(tmp_path, fixture_media):
    store,target,instances=fixture_media
    subjects,transforms=extract_subject_references(store,"data","fixture",target,instances)
    visible,region=completion_region(store,"data","fixture",target,instances,[],dilation_pixels=1)
    profile=ModelProfile("fixture","mock","fixture", "1",mock_output={"ok":True})
    gateway=ModelGateway(profiles=(profile,),policy=ModelPolicy({"fixture":"fixture"},1),receipt_root=tmp_path/"receipts")
    req=ModelRequest("fixture-call","fixture",({"role":"user","content":"fixture only"},),
       {"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":False})
    receipt=gateway.invoke(req)
    checks={k:{"status":"pass","evidence":["fixture:test"]} for k in
       ("subject_correspondence","reference_identity","selected_removal","unselected_preservation","background_completion")}
    row={"schema_version":"dataevolver.supervision.v1","evidence_kind":"fixture","id":"fixture-record",
      "family_id":"fixture","split":"train","task_type":"multi_subject_insertion","construction_mode":"scene_decomposition",
      "status":"pass","input":{"background":target.to_dict(),"subjects":subjects,"instruction":"fixture insert"},
      "target":{"image":target.to_dict()},"annotations":{"target_instances":instances,"reference_transforms":transforms,
        "background_visible":visible.to_dict(),"completion_region":region.to_dict(),"background_completion":{"inferred":True},
        "provenance":{"origin":"fixture"},"review":{"status":"pass","criteria":checks}},
      "model_receipts":[{"request":req.to_dict(),"receipt":receipt.to_dict(),"receipt_hash":receipt.without_reuse_hash()}]}
    return row,store


def test_bundle_loads_independently_and_fixture_cannot_be_real(tmp_path, fixture_media):
    row,store=fixture_record(tmp_path,fixture_media)
    with pytest.raises(ValueError,match="fixture"):
        validate_record(row,store)
    validate_record(row,store,allow_fixture=True)
    bundle=tmp_path/"bundle"
    export_dataset([row],store,bundle,expected_counts={"train":1},allow_fixture=True)
    loaded=load_dataset(bundle,allow_fixture=True)
    assert loaded[0][0]["input"]["subjects"][0]["id"]=="s1"
    with pytest.raises(ValueError,match="fixture"):
        load_dataset(bundle)
    assert export_dataset([row],store,bundle,expected_counts={"train":1},allow_fixture=True)["reused"]
    (bundle/"manifest.json").write_text("changed")
    with pytest.raises(ValueError,match="hash changed"):
        load_dataset(bundle,allow_fixture=True)


def test_portable_bundle_includes_media_referenced_inside_json_evidence(tmp_path,fixture_media):
    from dataevolver.dataset.supervision import write_artifact
    from dataevolver.dataset.contracts import canonical_json
    row,store=fixture_record(tmp_path,fixture_media)
    store.roots['baseline']=tmp_path/'baseline'
    image=write_image(store,'baseline','failed-background.png',Image.new('RGB',(32,24),'gray'))
    audit=write_artifact(store,'data','fixture/audit.json',
        canonical_json({'fixture_only':True,'rejected_candidate':image.to_dict()}).encode(),'application/json')
    row['annotations']['audit']=audit.to_dict()
    bundle=tmp_path/'bundle'
    export_dataset([row],store,bundle,expected_counts={'train':1},allow_fixture=True)
    loaded,new_store=load_dataset(bundle,allow_fixture=True)[0]
    assert loaded==row
    assert new_store.resolve(image).is_relative_to(bundle)
    # Even recomputing the outer checksums cannot hide an omitted nested source.
    import json
    checksums=json.loads((bundle/'checksums.json').read_text())
    new_store.resolve(image).unlink()
    checksums.pop('stores/baseline/failed-background.png')
    (bundle/'checksums.json').write_text(canonical_json(checksums))
    with pytest.raises(ValueError,match='missing'):
        load_dataset(bundle,allow_fixture=True)


@pytest.mark.parametrize("mutation,match",[
    (lambda r:r["input"].update(target_masks=[]),"whitelist"),
    (lambda r:r["annotations"]["target_instances"].pop(),"one to one"),
    (lambda r:r["annotations"]["background_completion"].update(inferred=False),"provenance"),
    (lambda r:r["annotations"]["review"]["criteria"]["selected_removal"].update(status="uncertain"),"quality criteria"),
    (lambda r:r["model_receipts"][0].update(receipt_hash="0"*64),"receipt identity"),
])
def test_invalid_supervision_is_rejected(tmp_path,fixture_media,mutation,match):
    row,store=fixture_record(tmp_path,fixture_media)
    mutation(row)
    with pytest.raises(ValueError,match=match):validate_record(row,store,allow_fixture=True)


def test_source_family_cannot_cross_splits(tmp_path,fixture_media):
    row,store=fixture_record(tmp_path,fixture_media);other=copy.deepcopy(row);other.update(id="fixture-other",split="test")
    with pytest.raises(ValueError,match="source-family leakage"):
        export_dataset([row,other],store,tmp_path/"bundle",expected_counts={"train":1,"test":1},allow_fixture=True)


def test_valid_hash_cannot_hide_a_wrong_reference_pair(tmp_path,fixture_media):
    row,store=fixture_record(tmp_path,fixture_media)
    row["input"]["subjects"][0]["reference"]=write_image(store,"data","fixture/wrong-reference.png",Image.new("RGBA",(8,8),"red")).to_dict()
    with pytest.raises(ValueError,match="reference pixels"):
        validate_record(row,store,allow_fixture=True)
