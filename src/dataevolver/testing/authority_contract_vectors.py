"""Independent synthetic test-vector producer, never image/task ground truth.

Expected pixels are declared below without importing or executing a candidate.
Signing uses cryptography, independently of the generated operator's RSA code.
Freeze the resulting files before any candidate execution; do not regenerate an
existing corpus after observing a failure. The private key exists only here in
memory and never appears in a payload, context, manifest or candidate process.
"""
import base64
import copy
import hashlib
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from dataevolver.dataset.contracts import ArtifactRef, file_sha256
from dataevolver.harness.authority_qualification import AuthorityFixtureExpectation
from dataevolver.harness.candidate_trust import CandidateTrustContext, canonical_bytes, canonical_digest
from dataevolver.harness.state_store import atomic_write_json


def ordered(points):
    return sorted(points, key=lambda p:(p[1],p[0]))


def make_vectors(root, *, package_sha256, implementation_sha256, request_sha256,
                 family_id="synthetic-aperture-plane-engineering", store_name="protocol",
                 operator_id="data.boundary_effect_authority",operator_version="1"):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    public_root=root.parent
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    public=key.public_key().public_numbers()
    def digest(label):return hashlib.sha256(("synthetic-contract-only:"+label).encode()).hexdigest()
    mask=[[2,2],[3,2],[2,3]]
    effect=[[4,4]]
    prior=ordered(mask+[[6,5]])
    full=ordered(mask+effect)
    transform={"kind":"integer_translation_crop","crop":[1,1,7,6],"inverse_projection":"add_crop_origin"}
    manifest={"assets":{name:digest(name) for name in ("original","rejected_candidate","projected_edit","conditioning_image","composition_support")},
        "width":9,"height":8,"instances_sha256":"0"*64,"prior_support_sha256":"0"*64,
        "protected_sha256":"0"*64,"receiving_surfaces_sha256":"0"*64,
        "frozen":{"models":[{"id":"synthetic-frozen-model-identity","sha256":digest("model")}],
          "seed":31,"generation_settings":{"synthetic_fixture":True},"quality_thresholds":{"required":1},
          "required_model_checks":["identity","source-binding","preservation","removal","completion"],
          "independent_visual_audit_required":True},"transform":transform}
    instance={"id":"subject-a","mask":mask,"mask_sha256":canonical_digest(mask),"reference_sha256":digest("reference"),
              "required_boundary":copy.deepcopy(mask),"effects":[{"id":"effect-a","surface":"plane-a","pixels":effect}]}
    observation=lambda oid,kind,points,basis,fx=None,surface=None:{"id":oid,"kind":kind,"manifest_sha256":"0"*64,
        "subject":"subject-a","mask_sha256":instance["mask_sha256"],"reference_sha256":instance["reference_sha256"],
        "pixels":points,"effect":fx,"surface":surface,"basis":basis}
    payload={"evidence":{"source":{"registrar":"independent-fixture-registrar","signature":"","manifest":manifest},
        "instances":[instance],"observations":[
            observation("boundary-a","boundary_witness",copy.deepcopy(mask),"registered_removal_boundary"),
            observation("attribution-a","attributed_effect",copy.deepcopy(effect),"registered_separate_effect_attribution","effect-a","plane-a"),
            observation("aperture-preserve","preservation_witness",[[3,3]],"registered_preservation")],
        "prior_support":{"conditioning":copy.deepcopy(prior),"composition":copy.deepcopy(prior),"transform_sha256":canonical_digest(transform)},
        "protected":{"pixels":[[3,3]]},"uncertainty":[],
        "receiving_surfaces":[{"id":"plane-a","pixels":[[4,4],[5,4],[6,4]]}]},"parameters":{}}
    def bind(value):
        ev=value["evidence"];m=ev["source"]["manifest"]
        ev["prior_support"]["transform_sha256"]=canonical_digest(m["transform"])
        for field in ("instances","prior_support","protected","receiving_surfaces"):
            m[field+"_sha256"]=canonical_digest(ev[field])
        for obs in ev["observations"]:obs["manifest_sha256"]=canonical_digest(m)
        return value
    def sign(value):
        ev=value["evidence"];unsigned=copy.deepcopy(ev);unsigned["source"].pop("signature",None)
        ev["source"]["signature"]=base64.b64encode(key.sign(canonical_bytes(unsigned),padding.PKCS1v15(),hashes.SHA256())).decode()
        return value
    sign(bind(payload))
    baseline_manifest=canonical_digest(payload["evidence"]["source"]["manifest"])
    cases=[];contexts={};expectations={}
    def ref(file):return ArtifactRef(store_name,file.relative_to(public_root).as_posix(),file_sha256(file)).to_dict()
    def emit(name,value,*,status="unsupported",resolved=(),added=(),removed=(),reasons=(),conflicts=(),
             receipt=False,auth=True,pinned=None):
        folder=root/name;folder.mkdir()
        inp=folder/"input.json";atomic_write_json(inp,value)
        input_ref=ref(inp)
        context=CandidateTrustContext(name,package_sha256,implementation_sha256,request_sha256,input_ref,
            canonical_digest(value),family_id,{"independent-fixture-registrar":{"n_hex":format(public.n,"x"),"e":public.e,
                "manifest_sha256":pinned or canonical_digest(value["evidence"]["source"]["manifest"])}})
        expected=AuthorityFixtureExpectation(family_id,canonical_digest(value),auth,status,ordered(list(resolved)),receipt,
            ordered(list(added)),ordered(list(removed)),list(reasons),list(conflicts),operator_id,operator_version)
        cp=folder/"host-context.json";ep=folder/"expected.json"
        atomic_write_json(cp,context.to_dict());atomic_write_json(ep,expected.to_dict())
        cases.append({"case_id":name,"kind":"positive" if status=="executable" else "negative",
            "input":input_ref,"expected_subset":{"status":status,"image_submission_allowed":False,"candidate_adoption_allowed":False}})
        contexts[name]=ref(cp);expectations[name]=ref(ep)
    emit("authenticated-aperture",copy.deepcopy(payload),status="executable",resolved=full,added=effect,removed=[[6,5]],receipt=True)
    value=copy.deepcopy(payload)
    value['evidence']['source']['manifest'].update(width=1024,height=1024)
    value['evidence']['source']['manifest']['transform']['crop']=[0,0,1024,1024]
    sign(bind(value))
    emit("sparse-standard-square-1024",value,status="executable",resolved=full,added=effect,removed=[[6,5]],receipt=True)
    value=copy.deepcopy(payload);signature=bytearray(base64.b64decode(value["evidence"]["source"]["signature"]));signature[-1]^=1
    value["evidence"]["source"]["signature"]=base64.b64encode(signature).decode()
    emit("invalid-signature",value,auth=False,reasons=["unverified_registered_evidence"])
    value=copy.deepcopy(payload);value["evidence"]["source"]["manifest"]["assets"]["original"]=digest("stale-original")
    sign(value);emit("stale-current-manifest",value,auth=False,pinned=baseline_manifest,reasons=["stale_or_unpinned_current_manifest"])
    value=copy.deepcopy(payload);value["evidence"]["instances"][0]["mask"][0]=[1,2]
    value["evidence"]["instances"][0]["reference_sha256"]=digest("stale-reference")
    sign(value);emit("stale-immutable-mask-and-reference",value,reasons=["stale_binding:instances"])
    value=copy.deepcopy(payload);value["evidence"]["observations"][0]["pixels"]=[[2,2],[3,2]]
    sign(value);emit("missing-boundary-witness",value,receipt=True,resolved=[[2,2],[3,2],[4,4]],added=effect,
        removed=[[2,3],[6,5]],reasons=["missing_boundary_authority"])
    value=copy.deepcopy(payload);value["evidence"]["observations"][1]["pixels"]=[]
    sign(value);emit("missing-effect-witness",value,receipt=True,resolved=mask,removed=[[6,5]],reasons=["missing_effect_authority"])
    value=copy.deepcopy(payload);value["evidence"]["protected"]["pixels"].append([2,2]);sign(bind(value))
    emit("preservation-conflict",value,receipt=True,resolved=[[3,2],[2,3],[4,4]],added=effect,removed=[[2,2],[6,5]],
         conflicts=["required_coverage_intersects_preservation"])
    value=copy.deepcopy(payload);value["evidence"]["uncertainty"]=[{"kind":"attribution","subject":"subject-a","pixels":[],"required":True}]
    sign(value);emit("empty-required-uncertainty",value,receipt=True,resolved=full,added=effect,removed=[[6,5]],reasons=["attribution"])
    value=copy.deepcopy(payload);value["evidence"]["uncertainty"]=[{"kind":"penumbra","subject":"subject-a","pixels":[[4,4]],"required":True}]
    sign(value);emit("uncertain-penumbra",value,receipt=True,resolved=mask,removed=[[6,5]],reasons=["penumbra","uncertain_required_coverage"])
    value=copy.deepcopy(payload);value["evidence"]["instances"][0]["effects"][0]["pixels"]=[[5,5]]
    value["evidence"]["observations"][1]["pixels"]=[[5,5]];sign(bind(value))
    emit("receiving-surface-edge",value,reasons=["effect_outside_bound_receiving_surface"])
    value=copy.deepcopy(payload);value["evidence"]["source"]["manifest"]["transform"]["crop"]=[1,1,2,2];sign(bind(value))
    emit("outside-frozen-crop",value,receipt=True,resolved=full,added=effect,removed=[[6,5]],reasons=["support_outside_frozen_crop"])
    value=copy.deepcopy(payload);value["parameters"]["TRUSTED_REGISTRARS"]={"payload-key":{"e":65537,"n":1}}
    emit("payload-trust-override",value,reasons=["invalid_object_fields:"])
    inventory={str(file.relative_to(root)):file_sha256(file) for file in sorted(root.rglob('*.json'))}
    atomic_write_json(root/"fixture-corpus.json",{"scope":"synthetic_authority_contract","family_id":family_id,
        "candidate_implementation_not_loaded":True,"fixture_expectations_derived_from_candidate":False,
        "signing_private_key_exported":False,"cases":cases,"trust_contexts":contexts,"expectations":expectations,"files":inventory})
    return {"cases":cases,"trust_contexts":contexts,"expectations":expectations}
