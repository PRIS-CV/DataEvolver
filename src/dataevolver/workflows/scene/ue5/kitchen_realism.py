"""Material scales are centimetres; edits apply only to the isolated UE copy."""
import hashlib
import math
from pathlib import Path
import unreal

LIB = unreal.MaterialEditingLibrary
ASSETS = unreal.AssetToolsHelpers.get_asset_tools()
DEST = '/Game/KitchenRealism02'
NOISE = '''
struct Surface {
 float hash(float3 p) { return frac(sin(dot(p,float3(127.1,311.7,74.7)))*43758.5453); }
 float noise(float3 p) {
  float3 i=floor(p),f=frac(p); f=f*f*(3-2*f);
  return lerp(lerp(lerp(hash(i),hash(i+float3(1,0,0)),f.x),
                   lerp(hash(i+float3(0,1,0)),hash(i+float3(1,1,0)),f.x),f.y),
              lerp(lerp(hash(i+float3(0,0,1)),hash(i+float3(1,0,1)),f.x),
                   lerp(hash(i+float3(0,1,1)),hash(i+1),f.x),f.y),f.z);
 }
 float fbm(float3 p) { return .57*noise(p)+.29*noise(p*2.07+17)+.14*noise(p*4.19+39); }
}; Surface s;
'''
GRADIENT = '''
float3 dx=ddx(P),dy=ddy(P),r1=cross(dy,N),r2=cross(N,dx);
float det=dot(dx,r1);
return normalize(max(abs(det),1e-8)*N-sign(det)*(ddx(H)*r1+ddy(H)*r2));
'''


def node(mat, kind):
    return LIB.create_material_expression(mat, getattr(unreal, kind))


def link(a, b, pin, output=''):
    assert LIB.connect_material_expressions(a, output, b, pin), (a, b, pin)


def scalar(mat, value):
    n = node(mat, 'MaterialExpressionConstant')
    n.set_editor_property('r', value)
    return n


def custom(mat, code, inputs, kind=unreal.CustomMaterialOutputType.CMOT_FLOAT4):
    n = node(mat, 'MaterialExpressionCustom')
    n.set_editor_property('output_type', kind)
    sockets = []
    for name in inputs:
        socket = unreal.CustomInput()
        socket.set_editor_property('input_name', name)
        sockets.append(socket)
    n.set_editor_property('inputs', sockets)
    n.set_editor_property('code', code)
    for name, source in inputs.items():
        link(source, n, name)
    return n


def material(name, program, height, metallic=0, coat=0, coat_roughness=.2, extras=None, parameters=None, cloth_color=None):
    path = DEST + '/M_' + name
    m = unreal.load_asset(path) or ASSETS.create_asset('M_' + name, DEST, unreal.Material, unreal.MaterialFactoryNew())
    assert m
    LIB.delete_all_material_expressions(m)
    m.set_editor_property('tangent_space_normal', False)
    m.set_editor_property('use_material_attributes', True)
    attributes = node(m, 'MaterialExpressionMakeMaterialAttributes')
    assert LIB.connect_material_property(attributes, '', unreal.MaterialProperty.MP_MATERIAL_ATTRIBUTES)
    input_names = {str(n).replace(' ', '').replace('_', '').lower():str(n)
                   for n in LIB.get_material_expression_input_names(attributes)}
    pins = {'MP_BASE_COLOR':'basecolor', 'MP_ROUGHNESS':'roughness', 'MP_NORMAL':'normal',
            'MP_METALLIC':'metallic', 'MP_SPECULAR':'specular',
            'MP_CUSTOM_DATA_0':'clearcoat', 'MP_CUSTOM_DATA_1':'clearcoatroughness',
            'MP_SUBSURFACE_COLOR':'subsurfacecolor'}

    def output(n, prop, pin=''):
        assert pins[prop] in input_names, input_names
        link(n, attributes, input_names[pins[prop]], pin)

    if coat:
        m.set_editor_property('shading_model', unreal.MaterialShadingModel.MSM_CLEAR_COAT)
        output(scalar(m, coat), 'MP_CUSTOM_DATA_0')
        output(scalar(m, coat_roughness), 'MP_CUSTOM_DATA_1')
    if cloth_color:
        m.set_editor_property('shading_model', unreal.MaterialShadingModel.MSM_CLOTH)
        fuzz = node(m, 'MaterialExpressionConstant3Vector')
        fuzz.set_editor_property('constant', unreal.LinearColor(*cloth_color, 1))
        output(fuzz, 'MP_SUBSURFACE_COLOR')
        output(scalar(m, .6), 'MP_CUSTOM_DATA_0')
    p = node(m, 'MaterialExpressionWorldPosition')
    local = node(m, 'MaterialExpressionTransformPosition')
    local.set_editor_property('transform_source_type', unreal.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_WORLD)
    local.set_editor_property('transform_type', unreal.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_LOCAL)
    link(p, local, '')
    normal = node(m, 'MaterialExpressionVertexNormalWS')
    pivot = node(m, 'MaterialExpressionObjectPositionWS')
    inputs = {'P': p, 'L': local, 'N': normal, 'Pivot': pivot}
    for key, default in (parameters or {}).items():
        value = node(m, 'MaterialExpressionVectorParameter')
        value.set_editor_property('parameter_name', key)
        value.set_editor_property('default_value', unreal.LinearColor(*default, 1))
        inputs[key] = value
    for key, texture in (extras or {}).items():
        t = node(m, 'MaterialExpressionTextureObject')
        t.set_editor_property('texture', texture)
        inputs[key] = t
    result = custom(m, NOISE + program, inputs)
    rgb = node(m, 'MaterialExpressionComponentMask')
    for key, value in {'r': True, 'g': True, 'b': True, 'a': False}.items():
        rgb.set_editor_property(key, value)
    link(result, rgb, '')
    output(rgb, 'MP_BASE_COLOR')
    alpha = node(m, 'MaterialExpressionComponentMask')
    for key, value in {'r': False, 'g': False, 'b': False, 'a': True}.items():
        alpha.set_editor_property(key, value)
    link(result, alpha, '')
    output(alpha, 'MP_ROUGHNESS')
    output(scalar(m, metallic), 'MP_METALLIC')
    output(scalar(m, .5), 'MP_SPECULAR')
    h = custom(m, NOISE + height, inputs, unreal.CustomMaterialOutputType.CMOT_FLOAT1)
    perturbed = custom(m, GRADIENT, {'P': p, 'N': normal, 'H': h}, unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    output(perturbed, 'MP_NORMAL')
    LIB.recompile_material(m)
    assert unreal.EditorAssetLibrary.save_loaded_asset(m)
    return m


BOARD = '''
float row=floor((L.y+13.5)/4.5);
float col=floor((L.x+19.0)/4.22+fmod(row,2)*.28);
float3 id=float3(col,row,7);
float2 q=float2((L.x+19.0)/4.22+fmod(row,2)*.28,(L.y+13.5)/4.5);
q=(frac(q)-.5)*float2(4.22,4.5);
float2 center=float2(s.hash(id+2),s.hash(id+19))*38-19;
float radius=length((q-center)*float2(.85+s.hash(id+4)*.3,1));
radius += .055*s.fbm(float3(q*.8, s.hash(id)*23));
float spacing=.16+.12*s.hash(id+3);
float ring=pow(saturate(.5+.5*sin(radius/spacing*6.28318)),9);
float porous=s.noise(float3(L.xy*24,12));
float macro=s.fbm(float3(L.xy*.65,29));
float scratches=0;
for(int k=0;k<18;k++) {
 float a=(s.hash(float3(k,2,9))-.5)*1.8;
 float2 d=float2(cos(a),sin(a)); float2 v=L.xy-float2((s.hash(float3(k,3,2))-.5)*25,(s.hash(float3(k,7,5))-.5)*16);
 float across=abs(dot(v,float2(-d.y,d.x)));
 float along=abs(dot(v,d));
 float w=max(.006, fwidth(across));
 scratches+=exp(-across*across/(w*w))*smoothstep(7.5,4.0,along);
}
scratches=saturate(scratches)*smoothstep(.8,.95,abs(N.z));
'''


def apply_lookdev(actors, manifest):
    labels = {a.get_actor_label(): a for a in actors}
    changes = []
    board = material('Endgrain', BOARD + '''
float tone=.93+.13*s.hash(id)+.055*(macro-.5)-.095*ring-.025*porous;
float3 base=float3(.30,.165,.075)*tone + scratches*float3(.012,.009,.005);
return float4(base,clamp(.34+.06*porous+.025*ring+scratches*.10,.29,.54));
''', BOARD + 'return .0035*(porous-.5)-.0015*ring-.002*scratches;', coat=.12, coat_roughness=.32)
    tile = material('GlazedTile', '''
float v=s.hash(Pivot*.173);
float edge=max(0,min(9.8-abs(L.x),5.05-abs(L.z)));
float pool=exp(-edge/.14);
float clay=s.fbm(L*.32+v*21);
float3 base=float3(.66,.635,.565)*(.966+.046*v+.014*(clay-.5)-.035*pool);
return float4(base,.19+.035*clay+.018*v);
''', 'return .014*s.fbm(L*.20)+.003*s.noise(L*1.8);', coat=.85, coat_roughness=.12)
    quartz = material('Quartz', '''
float fine=s.noise(P*14);
float crystal=smoothstep(.74,.88,fine);
float broad=s.fbm(P*.11);
float3 base=float3(.66,.63,.555)*(1+.018*(broad-.5)-.10*crystal);
return float4(base,.245+.055*broad+.025*crystal);
''', 'return .00055*(s.noise(P*14)-.5);')
    bowl = material('Stoneware', '''
float cloud=s.fbm(L*.72);
float iron=smoothstep(.81,.89,s.noise(L*14));
float turning=.5+.5*sin(L.z*15 + s.fbm(L*1.7)*1.8);
float glaze=.91+.12*cloud-.10*iron+.008*turning;
float rim=smoothstep(8.6,8.9,L.z);
float3 color=float3(.17,.235,.165)*glaze*(1-rim*.065);
float foot=1-smoothstep(.14,.37,L.z);
color=lerp(color,float3(.36,.29,.21),foot);
return float4(color,lerp(.27+.045*cloud,.67,foot));
''', '''
float turning=sin(L.z*15+s.fbm(L*1.7)*1.8);
return .00018*turning+.0007*s.fbm(L*4);
''', coat=.90, coat_roughness=.18)
    grout = material('Grout', 'return float4(float3(.43,.405,.35)*( .97+.06*s.noise(P*6)),.84);',
                     'return .0022*s.noise(P*9);')
    chrome = material('Chrome', 'return float4(float3(.68,.69,.70),.10+.035*s.fbm(P*.6));',
                      'return .00025*s.noise(P*50);', metallic=1)
    sink = material('BrushedSteel', '''
float smudge=s.fbm(P*.22);
return float4(float3(.58,.60,.61)*(.985+.025*smudge),.24+.055*smudge);
''', 'return .00045*s.noise(float3(P.x*70,P.y*.55,P.z*.55));', metallic=1)
    oak_texture = unreal.load_asset('/Game/KitchenPBRAssets01/kitchen-frame-001/Textures/oak_veneer_01_diff_2k')
    oak_rough = unreal.load_asset('/Game/KitchenPBRAssets01/kitchen-frame-001/Textures/oak_veneer_01_rough_2k')
    assert oak_texture and oak_rough
    wood = {}
    for vertical in [False, True]:
        mapping = '''
float2 uv=float2(P.x,P.z)/183.0;
if(abs(N.x)>.7) uv=float2(P.y,P.z)/183.0;
float h=s.hash(Pivot*.037);
uv+=float2(h,s.hash(Pivot*.039+7))*2.7;
'''
        if not vertical:
            mapping += 'uv=uv.yx;\n'
        wood[vertical] = material('OakVertical' if vertical else 'OakHorizontal', mapping + '''
float3 color=Texture2DSample(Oak,OakSampler,uv).rgb;
float value=dot(color,float3(.2126,.7152,.0722));
color=lerp(value.xxx,color,.78)*(.93+.10*h);
float rough=Texture2DSample(Rough,RoughSampler,uv).g;
return float4(color,clamp(.29+.27*rough,.30,.56));
''', mapping + 'return .0018*Texture2DSample(Rough,RoughSampler,uv).g;',
        coat=.12, coat_roughness=.3, extras={'Oak':oak_texture,'Rough':oak_rough})
    replacements = {'M_EndGrain_OiledOak': board, 'M_IvoryGlazedTile': tile,
                    'M_WarmQuartz': quartz, 'M_SageGlazedStoneware': bowl,
                    'M_TileGrout': grout, 'M_ChromedSteel': chrome}
    source_by_name = {o['name']: o for o in manifest['objects']}
    for actor in actors:
        name = actor.get_actor_label()
        source = source_by_name.get(name.replace('_UECurve',''))
        if not source:
            continue
        for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
            for slot, original in enumerate(source.get('materials', [])):
                mat = replacements.get(original)
                if original == 'M_oak_veneer_01':
                    bounds = source['world_bounds_m']
                    sizes = [bounds[1][i]-bounds[0][i] for i in range(3)]
                    mat = wood[sizes[2] > sizes[0] or name.endswith('Inset')]
                if name == 'IG_StainlessSinkShell':
                    mat = sink
                if mat:
                    assert slot < comp.get_num_materials(), (name, slot)
                    comp.set_material(slot, mat)
                    changes.append({'actor': name, 'slot':slot, 'material':mat.get_path_name()})
    assert any(x['actor']=='IG_GlazedBowl' for x in changes)
    assert any(x['actor']=='IG_EndGrainBoard' for x in changes)
    # Set the grout face 0.7 mm behind the tile faces, rather than a 15 mm void.
    grout_source = source_by_name['IG_GroutBed']['matrix_world']
    labels['IG_GroutBed'].set_actor_location(unreal.Vector(grout_source[0][3]*100,
        -grout_source[1][3]*100+1.43,grout_source[2][3]*100),False,False)
    geometry = tile_variants(labels, source_by_name)
    return {'material_assignments': changes, 'geometry': geometry,
            'shader_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'texture_coverage_cm': 183, 'endgrain_block_cm':[4.22,4.5],
            'plugins_exercised':['GeometryScripting'], 'photographic_acceptance':'pending_visual_review'}


def tile_variants(labels, source_by_name):
    """Keep the installed grid; vary the tile bodies below one millimetre."""
    template = labels['IG_Tile_0_0'].get_components_by_class(unreal.StaticMeshComponent)[0].static_mesh
    variants = []
    qa = []
    query = unreal.GeometryScript_MeshQueries
    for i in range(8):
        path = DEST + '/SM_TileWelded_' + str(i)
        mesh = unreal.load_asset(path)
        if not mesh:
            mesh = unreal.EditorAssetLibrary.duplicate_asset(template.get_path_name(), path)
            assert mesh
            dynamic = unreal.DynamicMesh()
            dynamic, outcome = unreal.GeometryScript_AssetUtils.copy_mesh_from_static_mesh(
                template, dynamic, unreal.GeometryScriptCopyMeshFromAssetOptions(), unreal.GeometryScriptMeshReadLOD())
            assert outcome == unreal.GeometryScriptOutcomePins.SUCCESS, str(outcome)
            # Interchange duplicated coincident positions at UV/normal seams.
            unreal.GeometryScript_MeshRepair.weld_mesh_edges(dynamic,
                unreal.GeometryScriptWeldEdgesOptions(tolerance=.0001, only_unique_pairs=True))
            unreal.GeometryScript_MeshRepair.compact_mesh(dynamic)
            assert query.get_is_closed_mesh(dynamic), 'Coincident seam weld did not close the tile'
            assert query.get_is_dense_mesh(dynamic)
            for vertex in range(query.get_vertex_count(dynamic)):
                position, valid = query.get_vertex_position(dynamic, vertex)
                assert valid, 'Expected compact source mesh vertex IDs'
                x,y,z = position.to_tuple()
                position = unreal.Vector(x+.014*math.sin(z*.54+i*1.73),
                    y+.025*math.sin(x*.31+i)*math.cos(z*.47-i),
                    z+.015*math.sin(x*.67+i*2.1))
                _, valid = unreal.GeometryScript_MeshEdits.set_vertex_position(dynamic, vertex, position, True)
                assert valid
            unreal.GeometryScript_Normals.recompute_normals(dynamic, unreal.GeometryScriptCalculateNormalsOptions())
            _, outcome = unreal.GeometryScript_AssetUtils.copy_mesh_to_static_mesh(
                dynamic, mesh, unreal.GeometryScriptCopyMeshToAssetOptions(), unreal.GeometryScriptMeshWriteLOD())
            assert outcome == unreal.GeometryScriptOutcomePins.SUCCESS
            assert unreal.EditorAssetLibrary.save_loaded_asset(mesh)
        dynamic = unreal.DynamicMesh()
        dynamic, outcome = unreal.GeometryScript_AssetUtils.copy_mesh_from_static_mesh(
            mesh,dynamic,unreal.GeometryScriptCopyMeshFromAssetOptions(),unreal.GeometryScriptMeshReadLOD())
        assert outcome == unreal.GeometryScriptOutcomePins.SUCCESS
        record = {'asset':path, 'vertices':query.get_vertex_count(dynamic),
            'triangles':dynamic.get_triangle_count(), 'closed':query.get_is_closed_mesh(dynamic),
            'open_edges':query.get_num_open_border_edges(dynamic)}
        assert record['closed'] and record['open_edges']==0 and record['triangles']>0,record
        qa.append(record)
        variants.append(mesh)
    assigned = []
    for name, actor in sorted(labels.items()):
        if not name.startswith('IG_Tile_'):
            continue
        source = source_by_name[name]
        bounds = source['world_bounds_m']
        if abs(bounds[1][0]-bounds[0][0]-.196)>.001:
            continue
        number = int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
        comp = actor.get_components_by_class(unreal.StaticMeshComponent)[0]
        mat = comp.get_material(0)
        comp.set_static_mesh(variants[number%8])
        comp.set_material(0, mat)
        original = source['matrix_world']
        dx=(number%101/100-.5)*.035
        dz=((number//101)%101/100-.5)*.035
        dy=((number//10201)%101/100-.5)*.055
        actor.set_actor_location(unreal.Vector(original[0][3]*100+dx,-original[1][3]*100+dy,original[2][3]*100+dz),False,False)
        assigned.append(name)
    assert assigned
    return {'variants':qa,'assigned_tiles':len(assigned), 'max_vertex_deformation_cm':.033,
            'max_installation_offset_cm':.0275, 'original_bevels_preserved':True}
