"""Portable Cycles profiles. Copied beside the headless skill's render_batch.py.

Pure settings validation works without bpy; hardware decisions run inside Blender.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time


PROFILES = {
    'draft': dict(percent=50, samples=32, threshold=.10, denoise_quality='FAST'),
    'balanced': dict(percent=75, samples=128, threshold=.03, denoise_quality='BALANCED'),
    'quality': dict(percent=100, samples=256, threshold=.01, denoise_quality='HIGH'),
    'final': dict(percent=100, samples=512, threshold=.005, denoise_quality='HIGH'),
    'preview': dict(percent=50, samples=64, threshold=.05, denoise_quality='FAST'),
    'gt': dict(percent=100, samples=256, threshold=.01, denoise_quality='HIGH'),
    'reference': dict(percent=100, samples=1024, threshold=0., denoise_quality='HIGH',
                      denoise=False, adaptive=False),
}
BACKENDS = ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI')
_FAILED_BACKENDS = {}


def frame_seed(task_seed, frame):
    return int.from_bytes(hashlib.sha256(f'{task_seed}:{frame}'.encode()).digest()[:4], 'big') % 2147483648


def reusable_frame(path, record):
    path = Path(path)
    return bool(record and path.is_file() and path.stat().st_size == record['bytes']
                and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'])


def settings(profile, overrides=None):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('Unknown render quality: ' + str(profile))
    value = dict(denoise=True, adaptive=True, persistent_data=False, seed=0,
                 animated_seed=False, motion_blur=False) | PROFILES[profile]
    overrides = overrides or {}
    if not isinstance(overrides, dict) or set(overrides) - set(value):
        raise ValueError('Unknown render override; use: ' + ', '.join(sorted(value)))
    value.update(overrides)
    for key, low, high in [('samples', 1, 16777216), ('percent', 1, 100), ('seed', 0, 2147483647)]:
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError('Invalid render override: ' + key)
    if (type(value['threshold']) not in (int, float) or not math.isfinite(value['threshold'])
            or not 0 <= value['threshold'] <= 1):
        raise ValueError('threshold must be finite and within [0, 1]')
    for key in ('denoise', 'adaptive', 'persistent_data', 'animated_seed', 'motion_blur'):
        if type(value[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    if value['denoise_quality'] not in ('FAST', 'BALANCED', 'HIGH'):
        raise ValueError('denoise_quality must be FAST, BALANCED or HIGH')
    if profile in ('gt', 'reference') and value['percent'] != 100:
        raise ValueError('GT/reference must render at the target resolution')
    if profile == 'reference' and (value['denoise'] or value['samples'] < 1024):
        raise ValueError('reference needs >=1024 samples and no denoising')
    return value


def target_resolution(scene, resolution=None):
    target = resolution if resolution is not None else scene.get('render_target_resolution')
    if target is None:
        target = [scene.render.resolution_x, scene.render.resolution_y]
    if len(target) != 2 or any(type(v) is not int or not 4 <= v <= 65536 for v in target):
        raise ValueError('Target resolution needs two positive integer dimensions (4..65536)')
    scene['render_target_resolution'] = list(target)
    scene.render.resolution_x, scene.render.resolution_y = target
    return list(target)


def _bpy():
    import bpy
    return bpy


def _devices(prefs, backend):
    prefs.compute_device_type = backend
    prefs.get_devices_for_type(backend)
    # prefs.devices retains entries from previously queried backends.
    return [d for d in prefs.devices if d.type == backend]


def capabilities():
    bpy = _bpy()
    prefs = bpy.context.preferences.addons['cycles'].preferences
    available, errors = {}, {}
    previous = prefs.compute_device_type
    for backend in BACKENDS:
        try:
            available[backend] = [dict(name=d.name, id=d.id, type=d.type)
                                  for d in _devices(prefs, backend)]
        except (TypeError, RuntimeError) as e:
            errors[backend] = str(e)
    prefs.compute_device_type = previous
    import _cycles
    from cycles.properties import enum_denoiser, enum_preview_denoiser
    cy = bpy.context.scene.cycles
    enums = {p: [e.identifier for e in cy.bl_rna.properties[p].enum_items]
             for p in ('denoising_quality', 'denoising_input_passes') if hasattr(cy, p)}
    enums['denoiser'] = [e[0] for e in enum_denoiser(cy, bpy.context)]
    enums['viewport_denoiser'] = [e[0] for e in enum_preview_denoiser(cy, bpy.context)]
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.free',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        gpu = dict(exit_code=query.returncode, csv=query.stdout.strip(), error=query.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as e:
        gpu = dict(error=str(e))
    return dict(blender_version=bpy.app.version_string, build_hash=bpy.app.build_hash.decode(),
                background=bpy.app.background, backends=available, backend_errors=errors,
                denoisers=enums, oidn_compiled=bool(_cycles.with_openimagedenoise),
                oidn_gpu_property=hasattr(cy, 'denoising_use_gpu'), gpu=gpu,
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                dlss_final_render='not validated; never selected by these profiles')


def _select_device(scene, state, preferred='AUTO', excluded=()):
    bpy = _bpy()
    prefs = bpy.context.preferences.addons['cycles'].preferences
    if preferred not in ('AUTO', 'GPU', 'CPU', *BACKENDS):
        raise ValueError('Unknown render backend: ' + str(preferred))
    candidates = [] if preferred == 'CPU' else list(BACKENDS)
    if preferred in BACKENDS:
        candidates.remove(preferred)
        candidates.insert(0, preferred)
    for backend in candidates:
        if backend in excluded:
            continue
        if backend in _FAILED_BACKENDS:
            state['fallbacks'].append(dict(stage='device discovery', backend=backend,
                reason='Failed earlier in this process: ' + _FAILED_BACKENDS[backend]))
            continue
        try:
            devices = _devices(prefs, backend)
        except (TypeError, RuntimeError) as e:
            state['fallbacks'].append(dict(stage='device discovery', backend=backend, reason=str(e)))
            continue
        if not devices:
            continue
        for device in prefs.devices:
            device.use = device.type == backend
        scene.cycles.device = 'GPU'
        state['backend'] = backend
        state['devices'] = [dict(name=d.name, id=d.id, type=d.type) for d in devices]
        return
    for device in prefs.devices:
        device.use = device.type == 'CPU'
    scene.cycles.device = 'CPU'
    state.update(backend='CPU', devices=[dict(name=d.name, id=d.id, type=d.type)
                                       for d in prefs.devices if d.type == 'CPU'])
    if preferred != 'CPU':
        state['fallbacks'].append(dict(stage='device discovery', reason='No usable GPU backend; using CPU'))


def _denoise(scene, state):
    cy = scene.cycles
    if not cy.use_denoising:
        return
    bpy = _bpy()
    prefs = bpy.context.preferences.addons['cycles'].preferences
    available = state['capabilities']['denoisers']['denoiser']
    denoiser = 'OPENIMAGEDENOISE' if 'OPENIMAGEDENOISE' in available else 'OPTIX'
    if denoiser not in available or (denoiser == 'OPTIX' and state['backend'] != 'OPTIX'):
        cy.use_denoising = False
        state['fallbacks'].append(dict(stage='denoising', reason='No compatible denoiser; retained native noisy output'))
        return
    cy.denoiser = denoiser
    cy.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    if hasattr(cy, 'denoising_prefilter'):
        cy.denoising_prefilter = 'ACCURATE'
    if hasattr(cy, 'denoising_quality'):
        cy.denoising_quality = state['requested']['denoise_quality']
    else:
        state['fallbacks'].append(dict(stage='denoising', reason='Quality enum unavailable; using build default'))
    if hasattr(cy, 'denoising_use_gpu'):
        cy.denoising_use_gpu = (state['backend'] != 'CPU' and denoiser == 'OPENIMAGEDENOISE'
                               and hasattr(prefs, 'has_oidn_gpu_devices') and prefs.has_oidn_gpu_devices())
        if denoiser == 'OPENIMAGEDENOISE' and not cy.denoising_use_gpu:
            state['fallbacks'].append(dict(stage='denoising', reason='OIDN GPU unsupported on selected device; using CPU OIDN'))
    for layer in scene.view_layers:
        if hasattr(layer.cycles, 'denoising_store_passes'):
            layer.cycles.denoising_store_passes = True
        if hasattr(layer.cycles, 'use_denoising'):
            layer.cycles.use_denoising = True


def apply(scene, profile, resolution=None, overrides=None, device='AUTO'):
    requested = settings(profile, overrides)
    state = dict(profile=profile, requested=requested, target_resolution=target_resolution(scene, resolution),
                 capabilities=capabilities(), fallbacks=[], frames=[], failed_attempts=[])
    scene.render.engine = 'CYCLES'
    _select_device(scene, state, device)
    cy, r = scene.cycles, scene.render
    cy.samples = requested['samples']
    cy.use_adaptive_sampling = requested['adaptive']
    cy.adaptive_threshold = requested['threshold']
    cy.adaptive_min_samples = 0
    cy.time_limit = 0
    cy.use_denoising = requested['denoise']
    cy.seed, cy.use_animated_seed = requested['seed'], requested['animated_seed']
    r.resolution_percentage = requested['percent']
    r.use_persistent_data = requested['persistent_data']
    r.use_motion_blur = requested['motion_blur']
    if profile in ('gt', 'reference'):
        r.use_sequencer = False
        r.use_compositing = False
        r.dither_intensity = 0.
    _denoise(scene, state)
    state['actual'] = actual(scene, state)
    return state


def actual(scene, state):
    cy, r = scene.cycles, scene.render
    camera = scene.camera
    return dict(backend=state['backend'], devices=state['devices'], device=cy.device,
        target_resolution=list(scene['render_target_resolution']), percent=r.resolution_percentage,
        resolution=[r.resolution_x*r.resolution_percentage//100, r.resolution_y*r.resolution_percentage//100],
        samples=cy.samples, adaptive=cy.use_adaptive_sampling, threshold=cy.adaptive_threshold,
        adaptive_min_samples=cy.adaptive_min_samples, time_limit=cy.time_limit,
        denoise=cy.use_denoising, denoiser=cy.denoiser,
        denoise_quality=getattr(cy, 'denoising_quality', None),
        denoise_gpu=getattr(cy, 'denoising_use_gpu', False) if cy.use_denoising else False,
        denoise_guides=getattr(cy, 'denoising_input_passes', None),
        denoise_prefilter=getattr(cy, 'denoising_prefilter', None),
        persistent_data=r.use_persistent_data, seed=cy.seed, animated_seed=cy.use_animated_seed,
        fps=r.fps, fps_base=r.fps_base, frame_range=[scene.frame_start, scene.frame_end],
        time_origin_frame=scene.frame_start, motion_blur=r.use_motion_blur,
        motion_blur_shutter=getattr(r, 'motion_blur_shutter', None),
        camera=dict(name=camera.name, matrix_world=[list(row) for row in camera.matrix_world],
            lens_mm=camera.data.lens, sensor_width_mm=camera.data.sensor_width,
            sensor_height_mm=camera.data.sensor_height, sensor_fit=camera.data.sensor_fit,
            shift=[camera.data.shift_x,camera.data.shift_y], clip=[camera.data.clip_start,camera.data.clip_end],
            dof=camera.data.dof.use_dof, fstop=camera.data.dof.aperture_fstop,
            focus_distance=camera.data.dof.focus_distance) if camera else None,
        pixel_aspect=[r.pixel_aspect_x,r.pixel_aspect_y],
        color=dict(display=scene.display_settings.display_device, view=scene.view_settings.view_transform,
                   look=scene.view_settings.look, exposure=scene.view_settings.exposure, gamma=scene.view_settings.gamma,
                   white_balance={k:getattr(scene.view_settings,k) for k in ('use_white_balance','temperature','tint')
                                  if hasattr(scene.view_settings,k)}),
        output=dict(format=r.image_settings.file_format, depth=r.image_settings.color_depth,
                    mode=r.image_settings.color_mode, exr_codec=r.image_settings.exr_codec),
        compositing=r.use_compositing, sequencer=r.use_sequencer, dither_intensity=r.dither_intensity,
        view_layers=[dict(name=layer.name, denoising_data=getattr(layer.cycles,'denoising_store_passes',False),
                          depth=layer.use_pass_z, object_index=layer.use_pass_object_index) for layer in scene.view_layers],
        light_paths={k:getattr(cy,k) for k in ('max_bounces','diffuse_bounces','glossy_bounces','transmission_bounces',
                                            'transparent_max_bounces','sample_clamp_direct','sample_clamp_indirect')})


def render_frame(scene, path, state):
    """Retry only recognized device/memory failures, keeping resolution and samples."""
    bpy = _bpy()
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to overwrite rendered frame: ' + str(path))
    if state['profile'] in ('gt', 'reference'):
        image = scene.render.image_settings
        if image.file_format not in ('PNG', 'OPEN_EXR', 'OPEN_EXR_MULTILAYER'):
            raise ValueError('GT/reference requires lossless PNG or EXR frame sequences')
        if image.file_format.startswith('OPEN_EXR') and (image.color_depth != '32' or image.exr_codec != 'ZIP'):
            raise ValueError('GT/reference numerical EXR needs float32 and lossless ZIP')
    scene.render.filepath = str(path)
    for _ in range(5):
        started = time.perf_counter()
        before = actual(scene, state)
        try:
            if bpy.ops.render.render(write_still=True) != {'FINISHED'}:
                raise RuntimeError('Blender render did not finish')
            if not path.is_file():
                raise RuntimeError('Blender did not write the requested frame')
        except RuntimeError as e:
            seconds = time.perf_counter()-started
            message = str(e).lower()
            state['failed_attempts'].append(dict(frame=scene.frame_current, seconds=seconds, error=str(e), actual=before))
            known = any(w in message for w in ('out of memory','failed to allocate','system is out of gpu','cuda','optix','denois'))
            if not known:
                raise
            memory = any(w in message for w in ('out of memory','failed to allocate','system is out of gpu'))
            if ((memory or ('denois' in message and 'kernel' not in message))
                    and scene.cycles.use_denoising and getattr(scene.cycles, 'denoising_use_gpu', False)):
                scene.cycles.denoising_use_gpu = False
                scene.render.use_persistent_data = False
                action = 'Retry with CPU OIDN and Persistent Data off'
            elif state['backend'] != 'CPU':
                previous = state['backend']
                if not memory:
                    _FAILED_BACKENDS[previous] = str(e)
                excluded = [f['backend'] for f in state['fallbacks'] if f.get('failed_backend')]
                _select_device(scene, state, excluded=excluded+[previous])
                if memory:
                    scene.render.use_persistent_data = False
                _denoise(scene, state)
                action = 'Retry on ' + state['backend']
                state['fallbacks'].append(dict(backend=previous, failed_backend=True, reason=str(e)))
            else:
                raise
            state['fallbacks'].append(dict(stage='render', frame=scene.frame_current, reason=str(e), action=action))
            print('RENDER_FALLBACK ' + json.dumps(state['fallbacks'][-1]), flush=True)
            continue
        record = dict(frame=scene.frame_current, time_s=(scene.frame_current-scene.frame_start)*scene.render.fps_base/scene.render.fps,
                      path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      actual=actual(scene, state))
        state['frames'].append(record)
        state['actual'] = record['actual']
        return record
    raise RuntimeError('Render device fallback limit reached; inspect failed_attempts')


def write_receipt(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(path)
