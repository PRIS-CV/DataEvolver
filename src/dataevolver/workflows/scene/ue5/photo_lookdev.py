"""Native look development on isolated scene copies; lengths are centimetres."""
import hashlib
import os
from pathlib import Path
import unreal
from kitchen_realism import material, LIB

DEST = '/Game/PhotoProduction01'
ASSETS = unreal.AssetToolsHelpers.get_asset_tools()


def seed(name):
    data = hashlib.sha256(name.encode()).digest()
    return [int.from_bytes(data[i:i+4], 'big') / 4294967295 for i in (0, 4, 8)]


def instance(parent, name):
    path = DEST + '/MI_' + name.replace('.', '_').replace('-', '_')
    result = unreal.load_asset(path) or ASSETS.create_asset(path.rsplit('/', 1)[1], DEST,
        unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    assert result
    LIB.set_material_instance_parent(result, parent)
    LIB.set_material_instance_vector_parameter_value(result, 'Variation', unreal.LinearColor(*seed(name), 1))
    assert unreal.EditorAssetLibrary.save_loaded_asset(result)
    return result


def texture(suffix):
    paths = unreal.EditorAssetLibrary.list_assets('/Game', recursive=True)
    matches = [p for p in paths if p.rsplit('/', 1)[-1].split('.')[0] == suffix]
    if not matches and suffix in ['oak_veneer_01_diff_2k', 'oak_veneer_01_rough_2k']:
        directory = Path(os.environ['DATAEVOLVER_LOOKDEV_TEXTURE_ROOT'])
        sources = list(directory.glob('*-' + suffix + '.jpg'))
        assert len(sources) == 1
        task = unreal.AssetImportTask()
        task.filename = str(sources[0])
        task.destination_path, task.destination_name = DEST, suffix
        task.automated = task.save = True
        ASSETS.import_asset_tasks([task])
        matches = task.imported_object_paths
    assert len(matches) == 1, (suffix, matches)
    result = unreal.load_asset(matches[0])
    assert isinstance(result, unreal.Texture2D)
    if 'rough' in suffix:
        result.set_editor_property('srgb', False)
        assert unreal.EditorAssetLibrary.save_loaded_asset(result)
    return result


def wood_material(name, axes, diffuse, rough, planks=False):
    mapping = f'''
float2 uv=({axes})/183.0;
uv.x*=Variation.x<.5 ? -1 : 1;
uv.y*=Variation.y<.5 ? -1 : 1;
uv*=.93+.14*Variation.z;
uv+=Variation.xy*11.37;
'''
    if planks:
        # The scan has horizontal boards; keep each mesh inside one gap-free row.
        mapping += f'''float2 crossSection=({axes})/183.0;
float rows[9]={{.06,.19,.31,.42,.52,.635,.747,.85,.945}};
uv.y=rows[min((int)(Variation.x*9),8)]+clamp(crossSection.y*.35,-.027,.027);
'''
    return material(name, mapping + '''
float3 col=Texture2DSample(Oak,OakSampler,uv).rgb;
float lum=dot(col,float3(.2126,.7152,.0722));
col=lerp(lum.xxx,col,.67)*(.86+.26*Variation.z);
float rough=Texture2DSample(Rough,RoughSampler,uv).g;
float handling=s.fbm(L*.16+Variation*31);
return float4(col,clamp(.43+.20*rough+.035*handling,.43,.70));
''', mapping + 'return .006*Texture2DSample(Rough,RoughSampler,uv).g;',
        extras={'Oak': diffuse, 'Rough': rough}, parameters={'Variation': (.3, .4, .5)})


def detail_meshes():
    path = os.environ['DATAEVOLVER_LOOKDEV_DETAIL_MESH']
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    previous = set(subsystem.get_all_level_actors())
    manager = unreal.InterchangeManager.get_interchange_manager_scripted()
    parameters = unreal.ImportAssetParameters()
    parameters.set_editor_property('is_automated', True)
    assert manager.import_scene('/Game/PhotoDetails01', manager.create_source_data(path), parameters)
    created = [a for a in subsystem.get_all_level_actors() if a not in previous]
    result = {a.get_actor_label():a.static_mesh_component.static_mesh for a in created
              if isinstance(a, unreal.StaticMeshActor)}
    assert set(result) == {'PhotoHead0','PhotoHead1','PhotoHead2','PhotoHead3','PhotoPlatformPlank','PhotoBowl','PhotoOrange'}, result
    for actor in reversed(created):
        subsystem.destroy_actor(actor)
    return result


ENDGRAIN = '''
float row=floor((L.y+13.5)/4.5);
float2 cell=float2((L.x+19)/4.22+fmod(row,2)*.5,(L.y+13.5)/4.5);
float3 id=float3(floor(cell),9);
float2 q=(frac(cell)-.5)*float2(4.22,4.5);
float2 centre=float2(s.hash(id+3),s.hash(id+17))*12-6;
float a=s.hash(id+1)*6.28318;
float2 v=q-centre;
float2 grain=float2(cos(a)*v.x-sin(a)*v.y,sin(a)*v.x+cos(a)*v.y);
float radius=length(grain*float2(.65+s.hash(id+7)*.5,1));
radius+=.045*s.fbm(float3(q*1.1,s.hash(id)*29));
float phase=radius/(.11+.14*s.hash(id+5))*6.28318;
float rings=pow(saturate(.5+.5*sin(phase)),6);
float pores=pow(saturate(s.noise(float3(L.xy*36,19))),9);
float edge=min(min(frac(cell.x),1-frac(cell.x))*4.22,min(frac(cell.y),1-frac(cell.y))*4.5);
float join=1-smoothstep(0,.008,edge);
float tone=.85+.30*s.hash(id+31);
'''


def kitchen(actors, manifest):
    labels = {a.get_actor_label(): a for a in actors}
    details = detail_meshes()
    for source, replacement in [('IG_GlazedBowl', 'PhotoBowl'), ('IG_Orange', 'PhotoOrange')]:
        component = labels[source].static_mesh_component
        original_materials = [component.get_material(i) for i in range(component.get_num_materials())]
        component.set_static_mesh(details[replacement])
        for index, original in enumerate(original_materials):
            component.set_material(index, original)
    for i in range(4):
        labels['IG_UtensilHead'+str(i)].static_mesh_component.set_static_mesh(details['PhotoHead'+str(i)])
    oak, rough = texture('oak_veneer_01_diff_2k'), texture('oak_veneer_01_rough_2k')
    woods = {vertical: wood_material('PhotoOakVertical' if vertical else 'PhotoOakHorizontal',
        'float2(abs(N.x)>.7?L.y:L.x,L.z)' if vertical else 'float2(L.z,L.x)', oak, rough) for vertical in [True, False]}
    board = material('PhotoEndgrain', ENDGRAIN + '''
float3 col=float3(.33,.185,.083)*tone*(1-.18*rings-.11*pores-.16*join);
return float4(col,.43+.07*rings+.05*pores+.03*s.fbm(L*.7));
''', ENDGRAIN + 'return -.003*rings-.005*pores-.006*join;')
    tile = material('PhotoTile', '''
float cloud=s.fbm(L*.25+Variation*19);
float edge=min(9.8-abs(L.x),5.05-abs(L.z));
float pooling=exp(-max(edge,0)/.20);
float3 col=float3(.67,.642,.585)*(.91+.16*Variation.z+.022*(cloud-.5)-.03*pooling);
return float4(col,.19+.065*cloud+.07*Variation.x);
''', '''
return .025*s.fbm(L*.14+Variation*11)+.004*s.noise(L*1.8);
''', parameters={'Variation': (.3,.4,.5)})
    bowl = material('PhotoStoneware', '''
float clay=s.fbm(L*.33);
float speck=pow(s.noise(L*21),11);
float foot=1-smoothstep(.15,.55,L.z);
float lip=smoothstep(8.1,8.8,L.z);
float3 col=float3(.135,.192,.127)*(.9+.18*clay)-speck*.09;
col=lerp(col,float3(.34,.265,.18),foot);
return float4(col,lerp(lerp(.24+.075*clay+.04*speck,.18,lip),.76,foot));
''', 'return .00045*sin(L.z*15)+.001*s.fbm(L*3);')
    quartz = material('PhotoQuartz', '''
float cloud=s.fbm(P*.085);
float grain=s.noise(P*9);
float crystal=pow(grain,8);
float3 col=float3(.64,.612,.55)*(.985+.027*cloud)-crystal*.065;
return float4(col,.30+.07*cloud+.035*grain);
''', 'return .0008*s.noise(P*9);')
    steel = material('PhotoBrushedSteel', '''
float smudge=s.fbm(L*.18);
return float4(float3(.53,.55,.57)*(.97+.045*smudge),.30+.075*smudge);
''', 'return .00035*s.noise(float3(L.x*65,L.y*.3,L.z*.3));', metallic=1)
    ceramic = material('PhotoCrock', '''
float cloud=s.fbm(L*.5);
float speck=pow(s.noise(L*18),10);
return float4(float3(.58,.535,.44)*(.96+.06*cloud)-speck*.07,.32+.09*cloud);
''', 'return .001*s.fbm(L*2)+.0004*sin(L.z*12);')
    hardware = material('PhotoSatinHardware',
        'return float4(float3(.57,.585,.60),.24+.035*s.fbm(L*.7));',
        'return .00015*s.noise(float3(L.x*60,L.y*.3,L.z*.3));', metallic=1)
    assignments = []
    replacements = {'M_EndGrain_OiledOak': board, 'M_IvoryGlazedTile': tile,
        'M_SageGlazedStoneware': bowl, 'M_WarmQuartz': quartz, 'M_UtensilStoneware': ceramic,
        'M_ChromedSteel': hardware}
    for item in manifest['objects']:
        name = item['name']
        if name not in labels:
            continue
        actor = labels[name]
        for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
            for index, original in enumerate(item.get('materials', [])):
                mat = replacements.get(original)
                if original == 'M_oak_veneer_01':
                    low, high = item['world_bounds_m']
                    vertical = high[2]-low[2] > high[0]-low[0] or name.endswith('Inset')
                    mat = instance(woods[vertical], name)
                if original == 'M_IvoryGlazedTile':
                    mat = instance(tile, name)
                    values = seed(name)
                    rotation = actor.get_actor_rotation()
                    rotation.pitch += (values[0]-.5)*.24
                    rotation.roll += (values[1]-.5)*.24
                    actor.set_actor_rotation(rotation, False)
                    # This frozen kitchen's tile faces point toward world +Y; avoid the grout backing.
                    actor.add_actor_world_offset(unreal.Vector(0, .06+values[2]*.02, 0), False, False)
                if name == 'IG_StainlessSinkShell':
                    mat = steel
                if mat:
                    comp.set_material(index, mat)
                    assignments.append(dict(actor=name, slot=index, material=mat.get_path_name()))
    assert any(r['actor']=='IG_EndGrainBoard' for r in assignments)
    return dict(profile='photo-v1', assignments=assignments,
        geometry='Four utensil heads and subdivided closed bowl/orange; original local bounds retained',
        variation='SHA256 of source actor name; stable across attempts',
        tile_max_rotation_degrees=.12, tile_outward_offset_cm=[.06,.08], collision_and_trajectory_changed=False)


def sphere(actor, radius):
    path = DEST + '/SM_' + actor.get_actor_label()
    comp = actor.static_mesh_component
    mesh = unreal.load_asset(path) or unreal.EditorAssetLibrary.duplicate_asset(comp.static_mesh.get_path_name(), path)
    dynamic = unreal.DynamicMesh()
    unreal.GeometryScript_Primitives.append_sphere_lat_long(dynamic, unreal.GeometryScriptPrimitiveOptions(),
        unreal.Transform(), radius, 64, 128)
    query = unreal.GeometryScript_MeshQueries
    unreal.GeometryScript_MeshRepair.weld_mesh_edges(dynamic,
        unreal.GeometryScriptWeldEdgesOptions(tolerance=.00001, only_unique_pairs=True))
    unreal.GeometryScript_Normals.recompute_normals(dynamic, unreal.GeometryScriptCalculateNormalsOptions())
    qa = dict(actor=actor.get_actor_label(), radius_cm=radius, vertices=query.get_vertex_count(dynamic),
        triangles=dynamic.get_triangle_count(), closed=query.get_is_closed_mesh(dynamic),
        open_edges=query.get_num_open_border_edges(dynamic))
    assert qa['closed'] and qa['open_edges']==0 and qa['triangles']>10000, qa
    _, outcome = unreal.GeometryScript_AssetUtils.copy_mesh_to_static_mesh(dynamic, mesh,
        unreal.GeometryScriptCopyMeshToAssetOptions(), unreal.GeometryScriptMeshWriteLOD())
    assert outcome == unreal.GeometryScriptOutcomePins.SUCCESS
    assert unreal.EditorAssetLibrary.save_loaded_asset(mesh)
    comp.set_static_mesh(mesh)
    return qa


BASKETBALL = '''
float3 n=normalize(L);
float meridian=min(abs(n.x),abs(n.y))*12;
float belt=abs(n.z)*12;
float arc=abs(n.z-.52*(n.x*n.x-n.y*n.y))*8.7;
float distance=min(min(meridian,belt),arc);
float aa=max(fwidth(distance),.008);
float seam=1-smoothstep(.075-aa,.075+aa,distance);
float3 cell=floor(L*8.5);
float pebble=0;
for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++) for(int z=-1;z<=1;z++) {
 float3 c=cell+float3(x,y,z);
 float3 grainPoint=c+float3(s.hash(c),s.hash(c+13),s.hash(c+29));
 float d=length(L*8.5-grainPoint);
 pebble=max(pebble,exp(-d*d*12));
}
'''


def outdoor(actors, manifest):
    labels = {a.get_actor_label(): a for a in actors}
    details = detail_meshes()
    basketball = material('PhotoBasketball', BASKETBALL + '''
float wear=s.fbm(L*.8);
float3 rubber=float3(.37,.094,.022)*(.89+.17*pebble+.08*wear);
return float4(lerp(rubber,float3(.009,.007,.005),seam),lerp(.72+.12*(1-pebble),.9,seam));
''', BASKETBALL + 'return .014*pebble-.045*seam;')
    tennis_code = '''
float3 n=normalize(L);
float angle=atan2(n.y,n.x);
float d=abs(n.z-.46*cos(2*angle))*3.35;
float seam=1-smoothstep(.066-max(fwidth(d),.006),.066+max(fwidth(d),.006),d);
float fibre=s.noise(L*57);
'''
    tennis = material('PhotoTennis', tennis_code + '''
float3 felt=float3(.43,.56,.022)*(.88+.18*fibre);
return float4(lerp(felt,float3(.67,.65,.46),seam),.94);
''', tennis_code + 'return .0025*fibre-.007*seam;', cloth_color=(.35,.40,.10))
    geometry = [sphere(labels[name], radius) for name, radius in [('Collision_A',12.0),('Collision_B',3.35)]]
    labels['Collision_A'].static_mesh_component.set_material(0, basketball)
    labels['Collision_B'].static_mesh_component.set_material(0, tennis)
    hidden = []
    for name, actor in labels.items():
        if name.startswith(('Collision_A_Seam','Collision_B_Seam')):
            actor.set_actor_hidden_in_game(True)
            hidden.append(name)
    brick = material('PhotoOutdoorBrick', '''
float pore=s.fbm(L*8+Variation*33);
float broad=s.fbm(L*.2+Variation*11);
float3 col=float3(.57,.588,.59)*(.89+.19*Variation.z+.035*broad);
return float4(col,.62+.18*pore);
''', 'return .018*s.fbm(L*3+Variation*17)+.003*s.noise(L*35);',
        parameters={'Variation': (.3,.4,.5)})
    assignments = []
    oak, rough = texture('wood_planks_diff_2k'), texture('wood_planks_rough_2k')
    woods = {axis:wood_material('PhotoOutdoorWood' + str(axis), axes, oak, rough, planks=True)
        for axis, axes in enumerate(['float2(L.x,L.y)','float2(L.y,L.x)','float2(L.z,L.x)'])}
    for item in manifest['objects']:
        name = item['name']
        if name not in labels:
            continue
        actor = labels[name]
        for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
            for index, original in enumerate(item.get('materials', [])):
                if original in ['M_FiredBrick','M_DistantBrick']:
                    comp.set_material(index, instance(brick, name))
                    assignments.append(name)
                if original.startswith('LD_AxisWood') or original == 'M_wood_planks':
                    dimensions = item['dimensions_m']
                    axis = max(range(3), key=dimensions.__getitem__)
                    comp.set_material(index, instance(woods[axis], name))
                    assignments.append(name)
    assert len(hidden)==4 and assignments
    platform = labels['Collision_DirectPlatform']
    platform.set_actor_hidden_in_game(True)
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for i in range(3):
        position = platform.get_actor_location() + unreal.Vector(0,(i-1)*22,-1.8)
        actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor,position,platform.get_actor_rotation())
        actor.set_actor_label('PhotoPlatformPlank'+str(i))
        actor.static_mesh_component.set_static_mesh(details['PhotoPlatformPlank'])
        actor.static_mesh_component.set_material(0,instance(woods[0],actor.get_actor_label()))
        actor.set_actor_enable_collision(False)
    return dict(profile='photo-v1', sphere_qa=geometry, hidden_obsolete_seams=hidden,
        platform='Three 170 x 21.92 x 3.6 cm closed Blender planks; 1 mm bevel, 0.8 mm seams',
        visual_collision_difference='Original continuous support proxy retained; gaps and edge bevels <= 1 mm',
        varied_surface_bindings=len(assignments), physics_recomputed=False,
        radii_match_source_analytic_collision=True)


def locked_camera(actor):
    component = actor.get_components_by_class(unreal.CineCameraComponent)[0]
    t = actor.get_actor_transform()
    settings = component.get_editor_property('post_process_settings')
    return dict(location_cm=list(t.translation.to_tuple()), quaternion=list(t.rotation.to_tuple()),
        scale=list(t.scale3d.to_tuple()), focal_length_mm=component.current_focal_length,
        filmback={'width_mm':component.filmback.sensor_width,'height_mm':component.filmback.sensor_height},
        aperture=component.current_aperture,
        exposure={k:str(settings.get_editor_property(k)) for k in ['auto_exposure_method',
            'auto_exposure_apply_physical_camera_exposure','camera_iso','camera_shutter_speed',
            'depth_of_field_fstop','auto_exposure_bias']}, blend=component.post_process_blend_weight)


if __name__ == '__main__':
    assert all(0 <= v <= 1 for v in seed('fixture'))
    assert seed('fixture') == seed('fixture') and seed('fixture') != seed('other')
