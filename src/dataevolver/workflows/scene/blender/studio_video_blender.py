"""Blender-side rigid-body sampling, frozen replay and authored-camera rendering."""
import argparse
import hashlib
import json
import math
import sys
import time
from array import array
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from dataevolver.workflows.scene.blender import render_quality as rq


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def depth(obj):
    n = 0
    while obj.parent:
        obj = obj.parent
        n += 1
    return n


def moving_controller(obj):
    return bool(obj.animation_data and (obj.animation_data.action or
                obj.animation_data.drivers or obj.animation_data.nla_tracks)) or bool(obj.constraints)


def deformation_state(scene, expected, frame):
    """Verify evaluated float32 local coordinates previously audited against the solver cache."""
    if not expected:
        return {}
    dg = bpy.context.evaluated_depsgraph_get()
    result = {}
    for name, hashes in expected.items():
        obj = scene.objects[name].evaluated_get(dg)
        if obj.type != 'MESH' or not 1 <= frame <= len(hashes):
            raise ValueError('Invalid deformation replay contract: ' + name)
        coords = array('f', [0.0]) * (len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get('co', coords)
        if sys.byteorder != 'little':
            coords.byteswap()
        digest = hashlib.sha256(coords.tobytes()).hexdigest()
        if digest != hashes[frame - 1]:
            raise ValueError('Deformation differs from audited solver replay: ' + name)
        result[name] = digest
    return result


def snapshot(config, run):
    sc = bpy.context.scene
    world = sc.rigidbody_world
    all_active = [o for o in sc.objects if o.rigid_body and o.rigid_body.type == 'ACTIVE']
    passive = [o for o in sc.objects if o.rigid_body and o.rigid_body.type == 'PASSIVE']
    if not world or not world.enabled or not all_active or not passive:
        raise ValueError('场景需要启用刚体世界、动态物体和静态碰撞表面。')
    if not sc.use_gravity or Vector(sc.gravity).length < 1e-6:
        raise ValueError('请在物理源场景启用重力。')
    if abs(sc.unit_settings.scale_length - 1.0) > 1e-6:
        raise ValueError('请使用 1 Blender 单位 = 1 米的源场景。')
    unsupported = [o.name for o in sc.objects if any(m.type in ('CLOTH','FLUID','SOFT_BODY') for m in o.modifiers)]
    if unsupported:
        raise ValueError('当前自动烘焙支持刚体；检测到尚不支持的变形或流体仿真：' + ', '.join(unsupported[:8]))
    actors = []
    for obj in all_active:
        if obj.rigid_body.kinematic or moving_controller(obj):
            raise ValueError('动态物体必须由求解器驱动，请移除运动动画、约束或运动学设置：' + obj.name)
        ancestor = obj.parent
        compound_child = False
        while ancestor:
            if moving_controller(ancestor):
                raise ValueError('动态物体的父对象存在动画或约束：' + ancestor.name)
            if ancestor.rigid_body and ancestor.rigid_body.type == 'ACTIVE':
                if ancestor.rigid_body.collision_shape != 'COMPOUND':
                    raise ValueError('嵌套动态刚体仅支持 COMPOUND 子形状：' + obj.name)
                compound_child = True
            ancestor = ancestor.parent
        if not compound_child:
            actors.append(obj)
    if not actors:
        raise ValueError('未找到可独立求解的动态物体。')
    actors.sort(key=lambda o: (depth(o), o.name))
    cameras = [o for o in sc.objects if o.type == 'CAMERA' and not o.hide_render]
    names = config.get('cameras') or ([sc.camera.name] if sc.camera in cameras else [])
    names = list(dict.fromkeys(names + sorted(o.name for o in cameras)))[:config['views']]
    if len(names) != config['views'] or any(n not in {o.name for o in cameras} for n in names):
        raise ValueError(f"需要 {config['views']} 个有效相机，当前可用 {len(cameras)} 个。")
    sc.frame_start = 1
    sc.frame_end = config['frames']
    sc.render.fps = config['fps']
    sc.render.fps_base = 1.0
    world.point_cache.frame_start = 1
    world.point_cache.frame_end = config['frames']
    # Blender 5.2 can crash in the RNA setter even when assigning False to
    # an already-disabled disk cache. Keep unsolved memory-cache sources intact.
    if getattr(world.point_cache, 'use_disk_cache', False) or world.point_cache.is_baked:
        raise ValueError('请提供未烘焙、未启用磁盘缓存的物理源场景。')
    sc.frame_set(1)
    bpy.context.view_layer.update()
    manifest = dict(blender_version=bpy.app.version_string, build_hash=bpy.app.build_hash.decode(),
        solver='Blender Bullet', fps=config['fps'], frames=config['frames'],
        duration_s=config['duration_s'], gravity=list(sc.gravity), units='meters',
        substeps_per_frame=world.substeps_per_frame, solver_iterations=world.solver_iterations,
        time_scale=world.time_scale, cameras=names,
        actors=[dict(name=o.name, mass=o.rigid_body.mass, collision_shape=o.rigid_body.collision_shape,
            friction=o.rigid_body.friction, restitution=o.rigid_body.restitution,
            collision_margin=o.rigid_body.collision_margin, dimensions=list(o.dimensions)) for o in actors],
        active_shapes=len(all_active), passive_shapes=len(passive), objects=len(sc.objects),
        meshes=sum(o.type == 'MESH' for o in sc.objects), materials=len(bpy.data.materials),
        source=str(run/'physics.blend'), cache=str(run/'render.blend'),
        limits=['Rigid bodies only; no fluid or fracture calibration.',
                'Trajectory replay proves cache consistency, not independent physical truth.',
                'Motion detection does not establish semantic interactions or scene quality.'])
    bpy.ops.wm.save_as_mainfile(filepath=str(run/'physics.blend'))
    actor_names = [o.name for o in actors]
    rows = []
    for frame in range(1, config['frames'] + 1):
        sc.frame_set(frame)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        row = {n: [list(r) for r in bpy.data.objects[n].evaluated_get(dg).matrix_world] for n in actor_names}
        if not all(math.isfinite(v) for m in row.values() for r in m for v in r):
            raise ValueError('物理求解返回非有限姿态。')
        rows.append(dict(frame=frame, time=(frame-1)/config['fps'], objects=row))
        write(run/'simulation-progress.json', dict(frame=frame, frames=config['frames']))
    trace = dict(fps=config['fps'], frames=rows, actor_names=actor_names,
                 source='sequential Bullet evaluated matrices; no authored dynamic motion')
    write(run/'trace.json', trace)
    motions = {}
    for n in actor_names:
        initial = Matrix(rows[0]['objects'][n])
        motions[n] = dict(max_displacement_m=max((Matrix(r['objects'][n]).translation-initial.translation).length for r in rows),
            max_rotation_rad=max(initial.to_quaternion().rotation_difference(Matrix(r['objects'][n]).to_quaternion()).angle for r in rows))
    manifest['motion'] = motions
    manifest['motion_observed'] = any(d['max_displacement_m'] > .001 or d['max_rotation_rad'] > .01 for d in motions.values())
    write(run/'simulation.json', manifest)
    if not manifest['motion_observed']:
        raise ValueError('未观察到超过 1 mm 或 0.01 rad 的物体运动；已保留物理源和轨迹。')
    # Source stays unsolved. Reopen it to avoid relying on ephemeral point caches.
    bpy.ops.wm.open_mainfile(filepath=str(run/'physics.blend'))
    sc = bpy.context.scene
    for obj in sorted([o for o in sc.objects if o.rigid_body], key=depth, reverse=True):
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj], selected_editable_objects=[obj]):
            bpy.ops.rigidbody.object_remove()
    bpy.ops.rigidbody.world_remove()
    for row in rows:
        for n in actor_names:
            obj = bpy.data.objects[n]
            obj.rotation_mode = 'QUATERNION'
            obj.matrix_world = Matrix(row['objects'][n])
            for prop in ('location', 'rotation_quaternion', 'scale'):
                obj.keyframe_insert(data_path=prop, frame=row['frame'])
    for action in bpy.data.actions:
        curves = action.fcurves if hasattr(action, 'fcurves') else [f for layer in action.layers
            for strip in layer.strips for bag in getattr(strip, 'channelbags', []) for f in bag.fcurves]
        for curve in curves:
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'
    error = 0.0
    for row in rows:
        sc.frame_set(row['frame'])
        bpy.context.view_layer.update()
        for n in actor_names:
            actual = bpy.data.objects[n].matrix_world
            target = row['objects'][n]
            error = max(error, *(abs(actual[i][j]-target[i][j]) for i in range(4) for j in range(4)))
    manifest['replay_max_matrix_error'] = error
    manifest['trace_sha256'] = hashlib.sha256((run/'trace.json').read_bytes()).hexdigest()
    write(run/'simulation.json', manifest)
    if error > 1e-4:
        raise ValueError('运动缓存与求解轨迹不一致：' + str(error))
    sc['studio_trace_sha256'] = manifest['trace_sha256']
    sc['studio_source'] = str(run/'physics.blend')
    sc.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(run/'render.blend'))


def render(config, run, camera, index):
    started = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=str(run/'render.blend'))
    sc = bpy.context.scene
    sc.camera = bpy.data.objects[camera]
    sc.render.engine = 'CYCLES'
    overrides = dict(config.get('render_overrides') or {})
    if not config.get('render_quality'):
        overrides.update(samples=config['samples'], percent=100, persistent_data=True)
    state = rq.apply(sc, config.get('render_quality') or 'quality', config['resolution'], overrides,
                     device=config.get('render_backend', 'AUTO'))
    state['simulation'] = json.loads((run/'simulation.json').read_text())
    state['source'] = str(run/'render.blend')
    state['source_sha256'] = hashlib.sha256((run/'render.blend').read_bytes()).hexdigest()
    state['ai_processing'] = dict(interpolation=False, super_resolution=False, video_redraw=False)
    source_start = config.get('source_frame_start', 1)
    if (source_start < sc.frame_start or source_start + config['frames'] - 1 > sc.frame_end
            or sc.render.fps != config['fps'] or sc.render.fps_base != 1):
        raise ValueError('Requested frames/fps do not match the frozen scene')
    contract = dict(config={k:v for k,v in config.items() if k != 'resume'},
                    source_sha256=state['source_sha256'], camera=camera,
                    scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__),Path(rq.__file__))})
    state['resume_contract'] = contract
    receipt = run/f'camera-{index:02d}-render.json'
    previous = {}
    if receipt.exists():
        if not config.get('resume'):
            raise FileExistsError('Render receipt exists; use explicit resume')
        old = json.loads(receipt.read_text())
        if old.get('resume_contract') != contract:
            raise ValueError('Resume source, scripts or settings changed')
        history = run/'attempts';history.mkdir(exist_ok=True)
        backup = history/f'camera-{index:02d}-{time.time_ns()}.json'
        backup.write_bytes(receipt.read_bytes())
        previous = {r.get('output_frame', r['frame']):r for r in old['frames']}
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.image_settings.color_depth = '16' if config.get('render_quality') in ('gt','reference') else '8'
    sc.render.use_file_extension = True
    bpy.ops.wm.save_as_mainfile(filepath=str(run/f'camera-{index:02d}-configured.blend'))
    directory = run/'frames'/f'camera-{index:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    state['startup_s'] = time.perf_counter() - started
    state['source_frame_range'] = [source_start, source_start + config['frames'] - 1]
    for output_frame in range(1, config['frames'] + 1):
        frame = source_start + output_frame - 1
        sc.frame_set(frame)
        if 'task_seed' in config:
            sc.cycles.seed = rq.frame_seed(config['task_seed'], frame)
            sc.cycles.use_animated_seed = False
        path = directory/f'{output_frame:05d}.png'
        try:
            # Check the frozen replay even when a completed frame is reused.
            actors = state['simulation'].get('actors', [])
            transforms = {a['name']:[list(row) for row in bpy.data.objects[a['name']].matrix_world]
                          for a in actors}
            deforming = deformation_state(sc, state['simulation'].get('deformation_vertex_sha256'), frame)
            if config.get('resume') and rq.reusable_frame(path, previous.get(output_frame)):
                record = dict(previous[output_frame], reused=True)
                state['frames'].append(record)
            else:
                if path.exists():
                    if not config.get('resume'):
                        raise FileExistsError('Frame exists; use explicit resume')
                    quarantine = run/'quarantine';quarantine.mkdir(exist_ok=True)
                    path.rename(quarantine/f'{index:02d}-{output_frame:05d}-{time.time_ns()}.png')
                record = rq.render_frame(sc, path, state)
                record.update(output_frame=output_frame, output_time_s=(output_frame-1)/config['fps'],
                              actor_transforms=transforms, deformation_sha256=deforming, reused=False)
            if deforming and record.get('deformation_sha256') != deforming:
                raise ValueError('Rendered frame lacks its audited deformation receipt')
            state['actual'] = rq.actual(sc,state)
        finally:
            rq.write_receipt(receipt, state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=['simulate','render'], required=True)
    p.add_argument('--camera')
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    config = json.loads(args.config.read_text())
    run = args.config.parent
    if args.stage == 'simulate':
        snapshot(config, run)
    else:
        render(config, run, args.camera, args.index)


if __name__ == '__main__':
    main()
