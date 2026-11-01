"""Import a frozen GLB scene and replay its original rigid-body poses in UE.

shortcut: glTF procedural materials remain import approximations; validate and
rebuild their appearance before promoting this engineering recipe.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import traceback

import unreal

sys.path.insert(0, str(Path(__file__).parent))
from trajectory import camera_coverage, check_trace, ue_pose
from measure_hdri import measure
from ue_lifecycle import quit_when_idle

output = Path(os.environ['DATAEVOLVER_RENDER_OUTPUT'])
config = json.loads(Path(os.environ['DATAEVOLVER_SCENE_CONFIG']).read_text())
package = Path(config['package'])
manifest = json.loads((package / 'manifest.json').read_text())
trace_path = package / 'trace.json'
trace = json.loads(trace_path.read_text())
camera_name = os.environ['DATAEVOLVER_CAMERA']
resolution = json.loads(os.environ['DATAEVOLVER_RESOLUTION'])
samples = int(os.environ['DATAEVOLVER_SPATIAL_SAMPLES'])
count = int(os.environ['DATAEVOLVER_ANIMATION_FRAMES'])
prefix = 'scene_' + camera_name
receipt = dict(engine=unreal.SystemLibrary.get_engine_version(), renderer='UE Movie Render Queue',
    camera=camera_name, resolution=resolution, spatial_samples=samples,
    source_sha256=manifest['source_sha256'], glb_sha256=manifest['export']['sha256'],
    scene_family=manifest['scene_family'], hardware_ray_tracing=False, success=False,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    material_limit=manifest['material_limit'], physical_truth='unknown', benchmark_validated=False)
runtime_poses = []
runtime_errors = []
tick_handle = None


def sample_runtime(delta):
    try:
        pipeline = executor.get_editor_property('active_movie_pipeline')
        if not pipeline or unreal.MoviePipelineLibrary.get_current_segment_state(pipeline) != unreal.MovieRenderShotState.RENDERING:
            return
        frame = unreal.MoviePipelineLibrary.get_root_frame_number(pipeline).value
        if not 0 <= frame < count:
            return
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
        if not world:
            return
        runtime_actors = {a.get_actor_label(): a for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)}
        record = dict(output_frame=frame, actors={})
        for name in trace['actor_names']:
            transform = runtime_actors[name].get_actor_transform()
            rotation = [transform.rotation.rotate_vector(unreal.Vector(*(int(i == axis) for i in range(3)))).to_tuple() for axis in range(3)]
            record['actors'][name] = dict(location_cm=transform.translation.to_tuple(), rotation_columns=rotation,
                                         scale=transform.scale3d.to_tuple())
        runtime_poses.append(record)
    except Exception:
        if not runtime_errors:
            runtime_errors.append(traceback.format_exc())


def save():
    (output / 'render-receipt.json').write_text(json.dumps(receipt, indent=2))


def finished(executor, success):
    if tick_handle is not None:
        unreal.unregister_slate_post_tick_callback(tick_handle)
    if count > 1:
        (output / 'runtime-poses.json').write_text(json.dumps(runtime_poses))
        position_error = rotation_error = scale_error = 0.0
        for record in runtime_poses:
            for name, actual in record['actors'].items():
                xyz, _, matrix = ue_pose(trace['frames'][record['output_frame']]['objects'][name])
                position_error = max(position_error, max(abs(actual['location_cm'][i] - xyz[i]) for i in range(3)))
                rotation_error = max(rotation_error, max(abs(actual['rotation_columns'][axis][i] - matrix[i][axis]) for axis in range(3) for i in range(3)))
                scale_error = max(scale_error, max(abs(s - 1) for s in actual['scale']))
        covered = sorted({r['output_frame'] for r in runtime_poses})
        receipt['runtime_pose_capture'] = dict(observed_frames=covered, records=len(runtime_poses), errors=runtime_errors,
            max_translation_error_cm=position_error, max_rotation_matrix_error=rotation_error, max_scale_error=scale_error,
            full_frame_coverage=covered == list(range(count)),
            scope='PIE world sampled at Slate post tick with MRQ root frame number; not a render-pass callback')
    receipt['executor_success'] = success
    receipt['native_frames'] = []
    try:
        for path in sorted(output.glob('*.png')):
            data = path.read_bytes()
            assert data[:8] == b'\x89PNG\r\n\x1a\n'
            assert list(struct.unpack('>II', data[16:24])) == resolution
            receipt['native_frames'].append(dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        receipt['success'] = bool(success and len(receipt['native_frames']) == count)
    except Exception:
        receipt['exception'] = traceback.format_exc()
    save()
    quit_when_idle(receipt, save)


def errored(executor, pipeline, fatal, message):
    receipt.setdefault('errors', []).append(dict(fatal=fatal, message=str(message)))
    save()


try:
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == manifest['trace_sha256']
    receipt['trace_validation'] = check_trace(trace)
    source = package / manifest['export']['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['export']['sha256']
    assert count in (1, len(trace['frames'])) and trace['fps'] == 24
    source_camera = next(o for o in manifest['objects'] if o['name'] == camera_name)
    override = config.get('camera_override')
    if override and override['camera'] == camera_name:
        from trajectory import aimed_camera
        source_camera = aimed_camera(source_camera, override)
        receipt['camera_override'] = override
    receipt['camera_coverage'] = camera_coverage(source_camera, trace, resolution[0] / resolution[1],
        config.get('camera_required_window', [0, 1]))
    assert receipt['camera_coverage']['required_window_all_centres_in_view'], 'Source camera misses a target inside its required observation window'
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    stamp = Path(os.environ['DATAEVOLVER_WORKING_PROJECT']).parent / 'workflow-source.json'
    identity = {k: receipt[k] for k in ('source_sha256', 'glb_sha256', 'scene_family')}
    if stamp.exists():
        assert json.loads(stamp.read_text()) == identity, 'Prepared project belongs to another source'
        assert levels.load_level('/Game/WorkflowFrozen')
        receipt['scene_import'] = 'reused_hashed_prepared_project'
    else:
        assert levels.new_level('/Game/WorkflowFrozen'), 'Cannot create isolated map'
        manager = unreal.InterchangeManager.get_interchange_manager_scripted()
        parameters = unreal.ImportAssetParameters()
        parameters.set_editor_property('is_automated', True)
        assert manager.import_scene('/Game/WorkflowFrozenAssets', unreal.InterchangeManager.create_source_data(str(source)), parameters)
        receipt['scene_import'] = 'fresh_glb'
    inventory = list(actors.get_all_level_actors())
    labels = {a.get_actor_label(): a for a in inventory}
    mesh_actors = [a for a in inventory if isinstance(a, unreal.StaticMeshActor)]
    # Editor camera actors also own visualization meshes; they are not exported geometry.
    mesh_count = len(mesh_actors)
    assert mesh_count == manifest['exported_mesh_objects'], (mesh_count, manifest['exported_mesh_objects'])
    receipt['mesh_actor_count'] = mesh_count
    missing_materials = []
    for actor in mesh_actors:
        for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
            if any(m is None for m in comp.get_materials()):
                missing_materials.append(actor.get_actor_label())
    assert not missing_materials, missing_materials
    if config.get('material_rebuild') == 'outdoor-ball-v1':
        from kitchen_realism import material
        rebuilt = {
            'M_BasketballRubber': material('WorkflowBasketball',
                'float t=.88+.12*s.noise(L*7); return float4(float3(.42,.105,.025)*t,.83+.08*s.noise(L*4));',
                'return .018*(s.noise(L*7)-.5);'),
            'M_CollisionTennisFelt': material('WorkflowTennisFelt',
                'return float4(float3(.48,.65,.055)*(.93+.07*s.noise(L*24)),.96);',
                'return .006*(s.noise(L*24)-.5);'),
            'M_CollisionBallSeam': material('WorkflowRubberSeam',
                'return float4(float3(.012,.010,.008),.88);', 'return 0;'),
            'M_CollisionTennisSeam': material('WorkflowTennisSeam',
                'return float4(float3(.72,.70,.58),.96);', 'return .003*s.noise(L*20);')}
        bindings = []
        for item in manifest['objects']:
            if item['type'] != 'MESH' or item['hide_render']:
                continue
            for index, name in enumerate(item['materials']):
                if name in rebuilt:
                    labels[item['name']].static_mesh_component.set_material(index, rebuilt[name])
                    bindings.append(dict(actor=item['name'], slot=index, source_material=name))
        assert len(bindings) == 6, bindings
        receipt['material_rebuild'] = dict(profile='outdoor-ball-v1', bindings=bindings,
            note='Procedural rubber/felt approximation; not calibrated material measurements')
    if os.environ.get('DATAEVOLVER_PHOTO_PROFILE') == 'photo-v1':
        from photo_lookdev import outdoor
        receipt['photo_lookdev'] = outdoor(inventory, manifest)
    imported_camera = labels[camera_name]
    camera = actors.spawn_actor_from_class(unreal.CineCameraActor, imported_camera.get_actor_location(), imported_camera.get_actor_rotation())
    camera.set_actor_label('WF_' + camera_name)
    if 'camera_override' in receipt:
        xyz = lambda values: unreal.Vector(values[0]*100, -values[1]*100, values[2]*100)
        location, target = xyz(override['location_m']), xyz(override['target_m'])
        camera.set_actor_location(location, False, True)
        camera.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(location, target), False)
    comp = camera.get_cine_camera_component()
    width = source_camera['sensor_width_mm']
    comp.set_editor_property('filmback', unreal.CameraFilmbackSettings(sensor_width=width, sensor_height=width * resolution[1] / resolution[0]))
    comp.set_editor_property('current_focal_length', source_camera['lens_mm'])
    comp.set_editor_property('current_aperture', 5.6)
    receipt['camera_optics'] = dict(focal_length_mm=source_camera['lens_mm'], sensor_width_mm=width, sensor_height_mm=width * resolution[1] / resolution[0])
    receipt['skylights_before_cleanup'] = [a.get_actor_label() for a in inventory if isinstance(a, unreal.SkyLight)]
    for actor in inventory:
        if isinstance(actor, (unreal.Light, unreal.SkyLight, unreal.SkyAtmosphere)) or actor.get_actor_label().startswith('WF_'):
            actors.destroy_actor(actor)
    sky = actors.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 300))
    sky.set_actor_label('WF_OutdoorSky')
    sky.light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    sky.light_component.set_editor_property('source_type', unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
    cubemap = unreal.load_asset('/Game/KitchenRealism/daylight')
    if not cubemap:
        task = unreal.AssetImportTask()
        task.filename = config['hdri']
        task.destination_path = '/Game/KitchenRealism'
        task.destination_name = 'daylight'
        task.automated = task.save = True
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        assert task.imported_object_paths
        cubemap = unreal.load_asset(task.imported_object_paths[0])
    assert isinstance(cubemap, unreal.TextureCube)
    sky.light_component.set_cubemap(cubemap)
    sky.light_component.set_editor_property('real_time_capture', False)
    hdri_measurement = measure(config['hdri'])
    # A relative HDR panorama needs an authored scale before physical camera exposure.
    sky_reference = 8000.0
    sky_scale = sky_reference / hdri_measurement['horizontal_diffuse_irradiance_relative']
    assert 0 < sky_scale < 100000
    sky.light_component.set_intensity(sky_scale)
    sky.light_component.recapture_sky()
    active_skies = [a for a in actors.get_all_level_actors() if isinstance(a, unreal.SkyLight)]
    assert len(active_skies) == 1 and active_skies[0] == sky
    receipt['skylights_after_cleanup'] = [a.get_actor_label() for a in active_skies]
    sun = actors.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 500), unreal.Rotator(-55, -30, 0))
    sun.light_component.set_intensity(3500)
    sun.light_component.set_editor_property('light_source_angle', 5.0)
    sun.light_component.set_editor_property('atmosphere_sun_light', True)
    actors.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0))
    settings = comp.get_editor_property('post_process_settings')
    for key, value in dict(override_auto_exposure_method=True, auto_exposure_method=unreal.AutoExposureMethod.AEM_MANUAL,
        override_auto_exposure_apply_physical_camera_exposure=True, auto_exposure_apply_physical_camera_exposure=True,
        override_camera_iso=True, camera_iso=200.0, override_camera_shutter_speed=True, camera_shutter_speed=100.0,
        override_depth_of_field_fstop=True, depth_of_field_fstop=5.6).items():
        settings.set_editor_property(key, value)
    comp.set_editor_property('post_process_settings', settings)
    comp.set_editor_property('post_process_blend_weight', 1.0)
    receipt['lighting'] = dict(sun_lux=3500, sky_intensity=sky_scale,
        sky_irradiance_reference=sky_reference, hdri_measurement=hdri_measurement,
        note='Authored relative HDR normalization with one recaptured SkyLight; not measured lux or source-radiometry calibration')
    receipt['exposure'] = dict(iso=200, fstop=5.6, shutter_speed=100)
    from photo_lookdev import locked_camera
    receipt['locked_camera'] = locked_camera(camera)
    from kitchen_realism import node, custom, LIB
    backdrop = unreal.load_asset('/Game/M_WorkflowSky') or unreal.AssetToolsHelpers.get_asset_tools().create_asset('M_WorkflowSky', '/Game', unreal.Material, unreal.MaterialFactoryNew())
    assert backdrop
    LIB.delete_all_material_expressions(backdrop)
    backdrop.set_editor_property('shading_model', unreal.MaterialShadingModel.MSM_UNLIT)
    backdrop.set_editor_property('two_sided', True)
    backdrop.set_editor_property('is_sky', True)
    env = node(backdrop, 'MaterialExpressionTextureObject')
    env.set_editor_property('texture', cubemap)
    view = node(backdrop, 'MaterialExpressionCameraVectorWS')
    emission = custom(backdrop, 'return TextureCubeSample(Env,EnvSampler,-V).rgb * ' + repr(sky_scale) + ';',
                      {'Env': env, 'V': view}, unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    assert LIB.connect_material_property(emission, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    LIB.recompile_material(backdrop)
    assert unreal.EditorAssetLibrary.save_loaded_asset(backdrop)
    dome = actors.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0))
    dome.set_actor_label('WF_EnvironmentBackdrop')
    dome.static_mesh_component.set_static_mesh(unreal.load_asset('/Engine/BasicShapes/Sphere'))
    dome.static_mesh_component.set_material(0, backdrop)
    dome.static_mesh_component.set_cast_shadow(False)
    dome.static_mesh_component.set_editor_property('affect_distance_field_lighting', False)
    dome.set_actor_enable_collision(False)
    dome.set_actor_scale3d(unreal.Vector(2000, 2000, 2000))
    receipt['environment_backdrop'] = dict(cubemap=cubemap.get_path_name(), scale=sky_scale,
        source='same relative HDRI as SkyLight', collisions=False, casts_shadow=False,
        scope='Visual background only; not additional physical scene geometry')

    assets = unreal.AssetToolsHelpers.get_asset_tools()
    sequence = unreal.load_asset('/Game/WorkflowSequence') or assets.create_asset('WorkflowSequence', '/Game', unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    assert sequence
    for binding in sequence.get_bindings():
        binding.remove()
    for track in sequence.get_tracks():
        sequence.remove_track(track)
    sequence.set_display_rate(unreal.FrameRate(24, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(count)
    binding = sequence.add_possessable(camera)
    section = sequence.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    section.set_range(0, count)
    section.set_camera_binding_id(sequence.get_binding_id(binding))
    if count > 1:
        position_error = rotation_error = key_error = 0.0
        key_count = 0
        sources = {o['name']: o for o in manifest['objects']}
        for name in trace['actor_names']:
            actor = labels[name]
            actor.detach_from_actor(unreal.DetachmentRule.KEEP_WORLD, unreal.DetachmentRule.KEEP_WORLD, unreal.DetachmentRule.KEEP_WORLD)
            for child_name, item in sources.items():
                ancestor = item
                while ancestor.get('parent') and ancestor['parent'] != name:
                    ancestor = sources[ancestor['parent']]
                if child_name == name or ancestor.get('parent') == name:
                    for component in labels[child_name].get_components_by_class(unreal.StaticMeshComponent):
                        component.set_simulate_physics(False)
                        component.set_mobility(unreal.ComponentMobility.MOVABLE)
            initial = actor.get_actor_transform()
            section = sequence.add_possessable(actor).add_track(unreal.MovieScene3DTransformTrack).add_section()
            section.set_range(0, count)
            channels = section.get_channels_by_type(unreal.MovieSceneScriptingDoubleChannel)
            assert len(channels) == 9
            expected = []
            for index, frame in enumerate(trace['frames']):
                assert frame['frame'] == index + 1 and abs(frame['time'] - index / 24) < 1e-8
                xyz, q, matrix = ue_pose(frame['objects'][name])
                rotation = unreal.Quat(*q).rotator()
                actor.set_actor_transform(unreal.Transform(location=unreal.Vector(*xyz), rotation=rotation, scale=unreal.Vector(1, 1, 1)), False, True)
                actual = actor.get_actor_transform()
                position_error = max(position_error, max(abs(v - xyz[k]) for k, v in enumerate(actual.translation.to_tuple())))
                for axis in range(3):
                    unit = [0, 0, 0]
                    unit[axis] = 1
                    rotated = actual.rotation.rotate_vector(unreal.Vector(*unit)).to_tuple()
                    rotation_error = max(rotation_error, max(abs(rotated[k] - matrix[k][axis]) for k in range(3)))
                values = xyz + [rotation.roll, rotation.pitch, rotation.yaw] + [1, 1, 1]
                expected.append(values)
                for channel, value in zip(channels, values):
                    channel.add_key(time=unreal.FrameNumber(index), new_value=value, time_unit=unreal.MovieSceneTimeUnit.DISPLAY_RATE, interpolation=unreal.MovieSceneKeyInterpolation.CONSTANT)
                    key_count += 1
            actor.set_actor_transform(initial, False, True)
            for c, channel in enumerate(channels):
                keys = channel.get_keys()
                assert len(keys) == count
                for index, key in enumerate(keys):
                    assert key.get_time(unreal.MovieSceneTimeUnit.DISPLAY_RATE).frame_number.value == index
                    key_error = max(key_error, abs(key.get_value() - expected[index][c]))
        assert position_error < .0001 and rotation_error < .00001 and key_error < .0001
        receipt['animation'] = dict(frames=count, fps=24, physics_recomputed=False, actors=trace['actor_names'],
            source_sha256=manifest['trace_sha256'], key_count=key_count,
            max_native_actor_translation_error_cm=position_error, max_native_actor_rotation_matrix_error=rotation_error,
            max_channel_key_readback_error=key_error, evidence_scope='Sequence construction; per-frame MRQ poses not measured')

    assert levels.save_current_level()
    unreal.EditorAssetLibrary.save_loaded_asset(sequence)
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
    stamp.write_text(json.dumps(identity, indent=2))
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    queue.delete_all_jobs()
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = prefix
    job.map = unreal.SoftObjectPath('/Game/WorkflowFrozen.WorkflowFrozen')
    job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
    settings = job.get_configuration()
    target = settings.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    target.output_directory = unreal.DirectoryPath(str(output))
    target.file_name_format = prefix + '_{frame_number}'
    target.output_resolution = unreal.IntPoint(*resolution)
    target.use_custom_playback_range = True
    target.custom_start_frame, target.custom_end_frame = 0, count
    render_pass = unreal.PhotoPosePass if os.environ.get('DATAEVOLVER_RENDER_PASS_POSE')=='1' else unreal.MoviePipelineDeferredPassBase
    settings.find_or_add_setting_by_class(render_pass)
    receipt['render_pass_class'] = str(render_pass)
    settings.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    settings.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    cvars = settings.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    receipt['console_variables'] = {'r.MotionBlurQuality': 0, 'r.DepthOfFieldQuality': 0,
        'r.ScreenPercentage': 100, 'sg.GlobalIlluminationQuality': 4, 'sg.ReflectionQuality': 4,
        'r.Lumen.Reflections.DownsampleFactor': 1, 'r.Lumen.ScreenProbeGather.DownsampleFactor': 8,
        'r.Lumen.ScreenProbeGather.TracingOctahedronResolution': 16}
    for name, value in receipt['console_variables'].items():
        assert cvars.add_or_update_console_variable(name, value)
    aa = settings.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.spatial_sample_count, aa.temporal_sample_count = samples, 1
    aa.override_anti_aliasing = True
    aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_NONE
    aa.engine_warm_up_count = aa.render_warm_up_count = 128
    aa.render_warm_up_frames = True
    if os.environ.get('DATAEVOLVER_PREPARE_ONLY') == '1':
        manifest_result = unreal.MoviePipelineEditorLibrary.save_queue_to_manifest_file(queue)
        assert manifest_result and manifest_result[0]
        queue_path = Path(manifest_result[1]).resolve()
        assert queue_path.is_file()
        receipt.update(prepared=True, queue_manifest=str(queue_path))
        save()
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)
    else:
        executor = unreal.MoviePipelinePIEExecutor()
        executor.on_executor_finished_delegate.add_callable_unique(finished)
        executor.on_executor_errored_delegate.add_callable_unique(errored)
        if count > 1:
            tick_handle = unreal.register_slate_post_tick_callback(sample_runtime)
        save()
        subsystem.render_queue_with_executor_instance(executor)
except Exception:
    receipt['exception'] = traceback.format_exc()
    save()
    unreal.log_error(receipt['exception'])
    unreal.SystemLibrary.quit_editor()
