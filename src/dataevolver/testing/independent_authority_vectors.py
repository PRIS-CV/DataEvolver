"""Two unseen synthetic geometry families for post-revision contract checks.

No candidate import, output-derived labels, task media or production authority.
The fixed declarations are frozen before executing the immutable candidate.
"""
import base64
import copy
import hashlib
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from dataevolver.dataset.contracts import ArtifactRef, file_sha256
from dataevolver.harness.authority_qualification import AuthorityFixtureExpectation, validate_fixture
from dataevolver.harness.candidate_trust import CandidateTrustContext, canonical_bytes, canonical_digest
from dataevolver.harness.state_store import atomic_write_json


def ordered(points):
    return [list(p) for p in sorted({tuple(p) for p in points},key=lambda p:(p[1],p[0]))]


def _make_geometry_vectors(root, *, package_sha256, implementation_sha256, request_sha256,
                           definitions, include_adversarial, store_name='protocol'):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);public_root=root.parent
    cases=[];contexts={};expectations={};families=[]
    def ref(path):return ArtifactRef(store_name,path.relative_to(public_root).as_posix(),file_sha256(path)).to_dict()
    def digest(label):return hashlib.sha256(('independent-synthetic-contract:'+label).encode()).hexdigest()
    for family,width,height,crop,objects,protected,extra in definitions:
        families.append(family)
        key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        public=key.public_key().public_numbers();registrar=family+'-registrar'
        transform={'kind':'integer_translation_crop','crop':crop,'inverse_projection':'add_crop_origin'}
        manifest={'assets':{n:digest(family+':'+n) for n in ['original','rejected_candidate','projected_edit','conditioning_image','composition_support']},
            'width':width,'height':height,'instances_sha256':'0'*64,'prior_support_sha256':'0'*64,
            'protected_sha256':'0'*64,'receiving_surfaces_sha256':'0'*64,
            'transform':transform,'frozen':{'models':[{'id':'independent-synthetic-only','sha256':digest(family+':model')}],
                'seed':73,'generation_settings':{'synthetic_fixture':True},'quality_thresholds':{'required':1},
                'required_model_checks':['identity','source-binding','preservation','removal','completion'],
                'independent_visual_audit_required':True}}
        instances=[];observations=[];surfaces=[]
        for name,mask,effect in objects:
            instance={'id':name,'mask':ordered(mask),'mask_sha256':canonical_digest(ordered(mask)),
                'reference_sha256':digest(family+':reference:'+name),'required_boundary':ordered(mask),
                'effects':[{'id':name+'-effect','surface':name+'-plane','pixels':ordered(effect)}]}
            instances.append(instance);surfaces.append({'id':name+'-plane','pixels':ordered(effect)})
            for kind,points,basis,fx,surface in [
                ('boundary_witness',mask,'registered_removal_boundary',None,None),
                ('attributed_effect',effect,'registered_separate_effect_attribution',name+'-effect',name+'-plane')]:
                observations.append({'id':name+'-'+kind,'kind':kind,'manifest_sha256':'0'*64,'subject':name,
                    'mask_sha256':instance['mask_sha256'],'reference_sha256':instance['reference_sha256'],
                    'pixels':ordered(points),'effect':fx,'surface':surface,'basis':basis})
        first=instances[0]
        observations.append({'id':family+'-preservation','kind':'preservation_witness','manifest_sha256':'0'*64,
            'subject':first['id'],'mask_sha256':first['mask_sha256'],'reference_sha256':first['reference_sha256'],
            'pixels':ordered(protected),'effect':None,'surface':None,'basis':'registered_preservation'})
        masks=ordered([point for _,mask,_ in objects for point in mask])
        effects=ordered([point for _,_,effect in objects for point in effect])
        full=ordered(masks+effects)
        payload={'evidence':{'source':{'registrar':registrar,'signature':'','manifest':manifest},
            'instances':instances,'observations':observations,'prior_support':{'conditioning':ordered(masks+extra),
                'composition':ordered(masks+extra),'transform_sha256':canonical_digest(transform)},
            'protected':{'pixels':ordered(protected)},'uncertainty':[],'receiving_surfaces':surfaces},'parameters':{}}
        def bind_sign(value):
            ev=value['evidence'];m=ev['source']['manifest']
            for field in ['instances','prior_support','protected','receiving_surfaces']:
                m[field+'_sha256']=canonical_digest(ev[field])
            for obs in ev['observations']:obs['manifest_sha256']=canonical_digest(m)
            unsigned=copy.deepcopy(ev);unsigned['source'].pop('signature',None)
            ev['source']['signature']=base64.b64encode(key.sign(canonical_bytes(unsigned),padding.PKCS1v15(),hashes.SHA256())).decode()
            return value
        def emit(name,value,*,status='unsupported',resolved=(),added=(),removed=(),reasons=(),conflicts=(),receipt=False,auth=True):
            folder=root/name;folder.mkdir();inp=folder/'input.json';atomic_write_json(inp,value)
            context=CandidateTrustContext(name,package_sha256,implementation_sha256,request_sha256,ref(inp),
                canonical_digest(value),family,{registrar:{'n_hex':format(public.n,'x'),'e':public.e,
                    'manifest_sha256':canonical_digest(value['evidence']['source']['manifest'])}})
            expected=AuthorityFixtureExpectation(family,canonical_digest(value),auth,status,ordered(resolved),receipt,
                ordered(added),ordered(removed),list(reasons),list(conflicts))
            validate_fixture(value,context,expected)
            cp=folder/'host-context.json';ep=folder/'expected.json'
            atomic_write_json(cp,context.to_dict());atomic_write_json(ep,expected.to_dict())
            cases.append({'case_id':name,'kind':'positive' if status=='executable' else 'negative','input':ref(inp),
                'expected_subset':{'status':status,'image_submission_allowed':False,'candidate_adoption_allowed':False}})
            contexts[name]=ref(cp);expectations[name]=ref(ep)
        bind_sign(payload)
        emit(family+'-positive',copy.deepcopy(payload),status='executable',resolved=full,added=effects,removed=extra,receipt=True)
        if not include_adversarial:
            continue
        if family=='independent-lattice-aperture':
            value=copy.deepcopy(payload);blocked=masks[0];value['evidence']['protected']['pixels'].append(blocked)
            bind_sign(value)
            emit(family+'-preservation-conflict',value,resolved=[p for p in full if p!=blocked],added=effects,
                 removed=extra+[blocked],receipt=True,conflicts=['required_coverage_intersects_preservation'])
            value=copy.deepcopy(payload);value['evidence']['uncertainty']=[{'kind':'attribution','subject':first['id'],'pixels':[],'required':True}]
            bind_sign(value)
            emit(family+'-unresolved-empty',value,resolved=full,added=effects,removed=extra,receipt=True,reasons=['attribution'])
        else:
            value=copy.deepcopy(payload);new_point=[250,181]
            value['evidence']['instances'][0]['effects'][0]['pixels']=[new_point]
            value['evidence']['observations'][1]['pixels']=[new_point];bind_sign(value)
            emit(family+'-surface-crossing',value,reasons=['effect_outside_bound_receiving_surface'])
            value=copy.deepcopy(payload);raw=bytearray(base64.b64decode(value['evidence']['source']['signature']));raw[0]^=1
            value['evidence']['source']['signature']=base64.b64encode(raw).decode()
            emit(family+'-invalid-signature',value,auth=False,reasons=['unverified_registered_evidence'])
    atomic_write_json(root/'independent-corpus.json',{'family_ids':families,'cases':cases,'trust_contexts':contexts,
        'expectations':expectations,'candidate_loaded':False,'candidate_output_derived_labels':False,
        'private_key_exported':False,'production_image_quality_labels':False})
    return {'cases':cases,'trust_contexts':contexts,'expectations':expectations,'family_ids':families}


def make_independent_vectors(root, *, package_sha256, implementation_sha256, request_sha256, store_name='protocol'):
    return _make_geometry_vectors(root, package_sha256=package_sha256, implementation_sha256=implementation_sha256,
        request_sha256=request_sha256, store_name=store_name, include_adversarial=True, definitions=[
            ('independent-lattice-aperture',1536,1152,[2,1,24,16],
             [('lattice',[[5,4],[6,4],[7,4],[5,5],[7,5],[5,6],[6,6],[7,6]],[[12,8],[13,8]])], [[6,5]],[[19,12]]),
            ('independent-disconnected-receivers',2048,1536,[64,32,1200,1000],
             [('island-a',[[230,170],[231,170],[230,171]],[[250,180],[251,180]]),
              ('island-b',[[900,700],[901,700],[902,700],[900,701]],[[920,710]])], [[901,701],[500,500]],[[1100,800]])])


def make_subsequent_request(root, *, package_sha256, implementation_sha256, request_sha256, store_name='protocol'):
    """A new compatible signed request, never additional qualification or images."""
    return _make_geometry_vectors(root, package_sha256=package_sha256, implementation_sha256=implementation_sha256,
        request_sha256=request_sha256, store_name=store_name, include_adversarial=False, definitions=[
            ('subsequent-angled-aperture',1280,960,[24,20,220,190],
             [('angled',[[40,36],[41,36],[42,36],[40,37],[42,37],[40,38],[41,38],[42,38],[43,39]],
               [[55,43],[56,43],[57,44]])], [[41,37],[80,75]],[[170,130]])])
