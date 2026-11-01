"""Serial, isolated frozen-scene renders with durable attempts and native-file checks.

shortcut: rigid transforms only; validate a separate cache and renderer before
admitting deformation, fluids, or other physical mechanisms.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(plan):
    if plan.get('schema_version') != 1 or plan.get('recipe') not in ('kitchen-rigid-v1', 'frozen-scene-v1'):
        raise ValueError('Expected schema 1 and a supported frozen rigid recipe')
    generic = plan['recipe'] == 'frozen-scene-v1'
    if plan.get('allocator', 'ansi') not in ('ansi', 'binned2'):
        raise ValueError('Unsupported pinned UE allocator')
    if plan.get('display_backend', 'offscreen') not in ('offscreen', 'xvfb'):
        raise ValueError('Unsupported pinned display backend')
    if plan.get('execution_mode', 'pie') not in ('pie', 'runtime_cli'):
        raise ValueError('Unsupported UE execution mode')
    if plan.get('execution_mode') == 'runtime_cli' and plan.get('runtime_executor') not in plan.get('dependencies', []):
        raise ValueError('Runtime executor must be a pinned dependency')
    if plan.get('lookdev_profile') not in (None, 'photo-v1'):
        raise ValueError('Unsupported look development profile')
    if plan.get('lookdev_profile') == 'photo-v1':
        assets = plan.get('lookdev_assets', {})
        if (set(assets) != {'texture_directory', 'detail_mesh'} or
                any(not isinstance(p, str) or not Path(p).is_absolute() or '..' in Path(p).parts for p in assets.values())):
            raise ValueError('Photo look development requires explicit local texture and detail assets')
    if plan.get('pose_capture') not in (None, 'render_pass'):
        raise ValueError('Unsupported pose capture boundary')
    if plan.get('pose_capture') and plan.get('pose_trace') not in plan.get('dependencies', []):
        raise ValueError('Render-pass pose trace must be pinned')
    if 'container_image' in plan and not re.fullmatch(r'sha256:[a-f0-9]{64}', plan['container_image']):
        raise ValueError('Pin the complete Docker image ID')
    if generic:
        value = plan.get('scene_config')
        if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('Missing scene configuration path')
        if not re.fullmatch(r'[a-f0-9]{64}', plan.get('trace_sha256', '')):
            raise ValueError('Missing frozen trajectory hash')
    for key in ('root', 'project', 'engine', 'renderer', 'gpu_launcher'):
        value = plan.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('Invalid absolute path: ' + key)
    for key in ('container', 'user', 'expected_host', 'gpu_uuid', 'engine_version'):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError('Missing ' + key)
    for key, low, high in [('min_free_mib', 20000, 1000000), ('max_utilization', 0, 100),
                           ('timeout_s', 60, 3600)]:
        if type(plan.get(key)) is not int or not low <= plan[key] <= high:
            raise ValueError('Invalid ' + key)
    if type(plan.get('admission_timeout_s', 60)) is not int or not 60 <= plan.get('admission_timeout_s', 60) <= 900:
        raise ValueError('Admission timeout must be 60..900 seconds')
    if not isinstance(plan.get('dependencies'), list) or not plan['dependencies']:
        raise ValueError('Pin renderer and input dependencies')
    for value in plan['dependencies']:
        if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('Invalid dependency path')
    if not re.fullmatch(r'[a-f0-9]{64}', plan.get('source_sha256', '')):
        raise ValueError('Missing frozen source hash')
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 50:
        raise ValueError('Expected 1..50 render tasks')
    seen = set()
    for task in tasks:
        name = task.get('id', '')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name) or name in seen:
            raise ValueError('Task IDs must be unique directory names')
        seen.add(name)
        if generic and (not isinstance(task.get('camera'), str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,79}', task['camera'])):
            raise ValueError('Invalid scene camera name')
        if not generic and task.get('camera') not in ('Reference_Main', 'Contact_Side'):
            raise ValueError('Camera is outside this validated recipe')
        if type(task.get('frames')) is not int or task['frames'] not in (range(1, 601) if generic else (1, 48)) or task.get('fps') != 24:
            raise ValueError('Unsupported frozen frame count or FPS')
        if generic and task.get('file_prefix') != 'scene_' + task['camera']:
            raise ValueError('Generic frame prefix must match the source camera')
        if type(task.get('samples')) is not int or not 1 <= task['samples'] <= 64:
            raise ValueError('Samples must be 1..64')
        resolution = task.get('resolution')
        if not isinstance(resolution, list) or len(resolution) != 2 or any(type(n) is not int or not 64 <= n <= 4096 for n in resolution):
            raise ValueError('Resolution must have two integer sides, 64..4096')
    return plan


def package_contract(plan):
    project = Path(plan['project'])
    files = [project, Path(plan['renderer']), Path(plan['engine']), Path(plan['gpu_launcher']), Path(__file__)]
    files.extend(Path(p) for p in plan['dependencies'])
    if (project.parent / 'workflow-source.json').exists():
        files.append(project.parent / 'workflow-source.json')
    if 'scene_config' in plan:
        files.append(Path(plan['scene_config']))
    if plan.get('lookdev_profile') == 'photo-v1':
        files.append(Path(plan['lookdev_assets']['detail_mesh']))
        textures = list(Path(plan['lookdev_assets']['texture_directory']).glob('*.jpg'))
        if not textures:
            raise ValueError('Photo look development texture inventory is empty')
        files.extend(textures)
    for directory in ('Content', 'Config', 'Plugins'):
        files.extend(p for p in (project.parent / directory).rglob('*') if p.is_file())
    hashes = {str(p.resolve(strict=True)): sha256(p) for p in sorted(set(files))}
    value = {'plan': plan, 'file_hashes': hashes}
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return value


def verify_native(run, task, source_sha256, engine_version, trace_sha256=None, pose_trace=None):
    """Recompute integrity. A render-success flag cannot override a failed exit."""
    run = Path(run).resolve()
    if int((run / 'process.exit').read_text()) != 0:
        raise ValueError('UE process did not exit zero')
    receipt = read(run / 'render-receipt.json')
    if receipt.get('success') is not True or receipt.get('executor_success') is not True or receipt.get('exception') or receipt.get('errors'):
        raise ValueError('MRQ did not complete without errors')
    for key, expected in [('camera', task['camera']), ('resolution', task['resolution']),
                           ('spatial_samples', task['samples']), ('source_sha256', source_sha256)]:
        if receipt.get(key) != expected:
            raise ValueError('Render contract mismatch: ' + key)
    if not receipt.get('engine', '').startswith(engine_version + '-'):
        raise ValueError('Engine version changed')
    records = receipt.get('native_frames', [])
    if len(records) != task['frames'] or len(list(run.glob('*.png'))) != task['frames']:
        raise ValueError('Incomplete or extra native frames')
    files = []
    for index, record in enumerate(records):
        prefix = task.get('file_prefix', 'kitchen_' + task['camera'])
        path = run / f'{prefix}_{index:04d}.png'
        if Path(record['path']).resolve() != path or not path.is_relative_to(run):
            raise ValueError('Native frame path mismatch')
        with path.open('rb') as stream:
            header = stream.read(33)
        if header[:8] != b'\x89PNG\r\n\x1a\n' or list(struct.unpack('>II', header[16:24])) != task['resolution']:
            raise ValueError('Invalid native PNG dimensions')
        actual = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if any(actual[k] != record[k] for k in ('bytes', 'sha256')):
            raise ValueError('Native frame hash or size mismatch')
        files.append(actual)
    log = (run / 'stdout.log').read_text(errors='replace')
    if (run / 'prepare.log').exists():
        log += (run / 'prepare.log').read_text(errors='replace')
    if re.search(r'Failed to compile Material|LogShaderCompilers: Error|Fatal error:|Fatal Python error', log):
        raise ValueError('UE log contains a shader or fatal error')
    if task['frames'] > 1:
        animation = receipt.get('animation', {})
        if animation.get('frames') != task['frames'] or animation.get('fps') != 24 or animation.get('physics_recomputed') is not False:
            raise ValueError('Frozen animation contract mismatch')
        if trace_sha256 is not None and animation.get('source_sha256') != trace_sha256:
            raise ValueError('Frozen animation trajectory hash mismatch')
        for field, limit in [('max_native_actor_translation_error_cm', .0001),
                              ('max_native_actor_rotation_matrix_error', .00001),
                              ('max_channel_key_readback_error', .0001 if trace_sha256 else .00001)]:
            value = animation.get(field)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value < limit:
                raise ValueError('Invalid animation conversion evidence: ' + field)
    proof = {'status': 'passed', 'frames': files, 'source_sha256': source_sha256,
            'render_receipt_sha256': sha256(run / 'render-receipt.json'),
            'process_exit': 0, 'physical_truth': 'unknown', 'photorealism': 'not_reviewed',
            'runtime_pose_validation': 'unknown', 'production_ready': False,
            'benchmark_validated': False}
    if pose_trace is not None:
        from dataevolver.workflows.scene.ue5.render_pose_qa import assess
        observed = assess(run / 'render-pass-poses.jsonl', read(pose_trace), task['frames'], task['samples'])
        if receipt.get('render_pass_pose_capture') != observed:
            raise ValueError('Render-pass pose receipt changed')
        proof.update(runtime_pose_validation='passed', render_pass_pose=observed,
                     render_pass_pose_sha256=sha256(run / 'render-pass-poses.jsonl'))
    return proof


@contextmanager
def lock(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def preflight(plan, *, enforce_load=True):
    if socket.gethostname() != plan['expected_host']:
        raise ValueError('Wrong execution host')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = []
    for line in query.splitlines():
        index, uuid, free, usage = [s.strip() for s in line.split(',')]
        devices.append(dict(index=int(index), uuid=uuid, free_mib=int(free), utilization=int(usage)))
    gpu = next(d for d in devices if d['uuid'] == plan['gpu_uuid'])
    if gpu['free_mib'] < plan['min_free_mib'] or (enforce_load and gpu['utilization'] > plan['max_utilization']):
        raise RuntimeError('Bound GPU is busy or has insufficient memory')
    container = json.loads(subprocess.check_output(['docker', 'inspect', plan['container']], text=True))[0]
    if 'container_image' in plan:
        if container['Image'] != plan['container_image']:
            raise RuntimeError('Container image differs from the pinned image ID')
        if container['HostConfig'].get('Init') is not True:
            raise RuntimeError('Pinned runtime requires Docker init for child-process reaping')
    libraries = subprocess.check_output(['docker', 'exec', plan['container'], 'ldd', plan['engine']], text=True)
    if 'not found' in libraries:
        raise RuntimeError('UE runtime has unresolved shared libraries')
    visible = subprocess.check_output(['docker', 'exec', plan['container'], 'nvidia-smi',
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
    if visible != [gpu['uuid']]:
        raise RuntimeError('Container must expose exactly the pinned GPU')
    processes = subprocess.check_output(['docker', 'exec', plan['container'], 'ps', '-eo', 'pid,comm'], text=True)
    if 'UnrealEditor' in processes:
        raise RuntimeError('Container already has an UnrealEditor process; inspect before retry')
    return {'hostname': socket.gethostname(), 'gpu': gpu, 'all_devices': devices,
            'container_image': container['Image'], 'container_init': container['HostConfig'].get('Init'),
            'engine_shared_libraries': 'resolved',
            'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0]}


def acquire_gpu(plan, api):
    # GPU utilization can still include the previous render immediately after UE exits.
    attempts = plan.get('admission_timeout_s', 60) // 2
    for attempt in range(attempts + 1):
        try:
            with lock(Path(plan['root']) / 'runtime/mcp-allocation.lock'):
                discovered = preflight(plan)
                return api['lease_gpu'](Path(plan['root']), [discovered['gpu']], api['reserved_gpu_uuids'](Path(plan['root'])),
                                        max_utilization=plan['max_utilization'])
        except (RuntimeError, BlockingIOError) as error:
            if isinstance(error, RuntimeError) and not str(error).startswith(('Bound GPU is busy', 'No GPU lease available')):
                raise
            if attempt == attempts:
                raise
            time.sleep(2)


@contextmanager
def virtual_display(plan, run):
    if plan.get('display_backend', 'offscreen') != 'xvfb':
        yield
        return
    base = ['docker', 'exec', '--user', plan['user'], plan['container']]
    socket_test = base + ['test', '-S', '/tmp/.X11-unix/X99']
    if subprocess.run(socket_test, capture_output=True).returncode == 0:
        yield
        return
    bootstrap = ('import os; from pathlib import Path; Path(' + repr(str(run / 'xvfb.pid')) +
                 ').write_text(str(os.getpid())); os.execlp("Xvfb", "Xvfb", ":99", "-screen", "0", '
                 '"4096x2160x24", "-nolisten", "tcp", "-ac")')
    with (run / 'xvfb.log').open('w') as log:
        display = subprocess.Popen(base + ['python3', '-c', bootstrap], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(15):
                if display.poll() is not None:
                    raise RuntimeError('Xvfb exited before display readiness; inspect xvfb.log')
                if subprocess.run(socket_test, capture_output=True).returncode == 0:
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Xvfb socket did not become ready')
            yield
        finally:
            if display.poll() is None and (run / 'xvfb.pid').exists():
                subprocess.run(base + ['kill', '-TERM', str(int((run / 'xvfb.pid').read_text()))],
                               capture_output=True, timeout=15)
            display.wait(timeout=15)


def render_attempt(plan, task, run, gpu_lock):
    write(run / 'preflight.json', preflight(plan, enforce_load=False))
    source = Path(plan['project'])
    project = run / 'project' / source.name
    shutil.copytree(source.parent, project.parent,
                    ignore=shutil.ignore_patterns('Saved', 'Intermediate', 'DerivedDataCache', '.git'))
    for path in [run, *run.rglob('*')]:
        os.chown(path, 10001, 10001)
    root = Path(plan['root'])
    home = root / 'envs/ue5/home'
    environment = {'HOME': str(home), 'XDG_RUNTIME_DIR': str(home / 'runtime'),
        'XDG_CACHE_HOME': str(home / 'cache'), 'TMPDIR': str(home / 'runtime/tmp'),
        'UE-LocalDataCachePath': str(home / 'cache/DerivedDataCache'), 'UE-ZenDataPath': str(home / 'cache/Zen'),
        'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json', 'CUDA_VISIBLE_DEVICES': '0',
        'DATAEVOLVER_WORKING_PROJECT': str(project),
        'DATAEVOLVER_RENDER_OUTPUT': str(run), 'DATAEVOLVER_RESOLUTION': json.dumps(task['resolution']),
        'DATAEVOLVER_CAMERA': task['camera'], 'DATAEVOLVER_SPATIAL_SAMPLES': str(task['samples']),
        'DATAEVOLVER_ANIMATION_FRAMES': str(task['frames'])}
    if 'scene_config' in plan:
        environment['DATAEVOLVER_SCENE_CONFIG'] = plan['scene_config']
    if plan.get('lookdev_profile') == 'photo-v1':
        environment['DATAEVOLVER_LOOKDEV_TEXTURE_ROOT'] = plan['lookdev_assets']['texture_directory']
        environment['DATAEVOLVER_LOOKDEV_DETAIL_MESH'] = plan['lookdev_assets']['detail_mesh']
    if plan.get('lookdev_profile'):
        environment['DATAEVOLVER_PHOTO_PROFILE'] = plan['lookdev_profile']
    if plan.get('pose_capture'):
        environment['DATAEVOLVER_RENDER_PASS_POSE'] = '1'
        environment['DATAEVOLVER_POSE_TRACE'] = plan['pose_trace']
    runtime_cli = plan.get('execution_mode') == 'runtime_cli'
    if runtime_cli:
        environment['DATAEVOLVER_PREPARE_ONLY'] = '1'
        init = project.parent / 'Content/Python/init_unreal.py'
        init.parent.mkdir(exist_ok=True)
        init.write_text('import sys\nsys.path.insert(0, ' + repr(str(Path(plan['runtime_executor']).parent)) + ')\nimport ue_cli_executor\n')
        os.chown(init.parent, 10001, 10001)
        os.chown(init, 10001, 10001)
    if plan.get('display_backend') == 'xvfb':
        environment.update(DISPLAY=':99', SDL_VIDEO_DRIVER='x11')
    command = ['docker', 'exec', '--user', plan['user']]
    for name, value in environment.items():
        command += ['-e', name + '=' + value]
    # An in-container timeout owns the UE process group if the SSH client exits.
    display_flags = ['-windowed', '-ResX=1280', '-ResY=720', '-sdlvideodriver=x11'] if plan.get('display_backend') == 'xvfb' else ['-RenderOffscreen']
    command += [plan['container'], 'timeout', '--signal=TERM', '--kill-after=15s', str(plan['timeout_s']),
        plan['engine'], str(project), *display_flags,
        '-ansimalloc' if plan.get('allocator', 'ansi') == 'ansi' else '-binnedmalloc2', '-vulkan', '-norhithread',
        '-norenderthread', '-unattended', '-nosound', '-nosplash', '-NoP4', '-nocef', '-corelimit=8',
        '-stdout', '-FullStdOutLogOutput', '-abslog=' + str(run / 'Unreal.log'),
        '-ExecutePythonScript=' + plan['renderer']]
    with virtual_display(plan, run):
        if runtime_cli:
            prepare = command[:command.index(plan['engine']) + 2] + ['-run=pythonscript', '-NullRHI', '-unattended', '-nosound', '-nop4',
                '-stdout', '-FullStdOutLogOutput', '-script=' + plan['renderer']]
            write(run / 'prepare-command.json', prepare)
            with (run / 'prepare.log').open('w') as log:
                setup = subprocess.run(prepare, stdout=log, stderr=subprocess.STDOUT, pass_fds=(gpu_lock,))
            (run / 'prepare.process.exit').write_text(str(setup.returncode) + '\n')
            prepared = read(run / 'render-receipt.json')
            if setup.returncode or not prepared.get('prepared') or prepared.get('exception'):
                raise ValueError('Runtime scene preparation failed')
            manifest = Path(prepared['queue_manifest'])
            if not manifest.is_relative_to(project.parent) or not manifest.is_file():
                raise ValueError('Missing prepared queue manifest')
            map_name = '/Game/WorkflowFrozen' if plan['recipe']=='frozen-scene-v1' else '/Game/KitchenPBR01'
            command = command[:command.index(plan['engine']) + 2] + [map_name + '?game=/Script/MovieRenderPipelineCore.MoviePipelineGameMode',
                '-game', *display_flags, '-vulkan', '-ansimalloc' if plan.get('allocator', 'ansi') == 'ansi' else '-binnedmalloc2',
                '-unattended', '-nosound', '-nosplash', '-NoP4', '-nocef', '-corelimit=8', '-stdout', '-FullStdOutLogOutput',
                '-abslog=' + str(run / 'Unreal.log'), '-MoviePipelineConfig=' + str(manifest),
                '-MoviePipelineLocalExecutorClass=/Script/MovieRenderPipelineCore.MoviePipelinePythonHostExecutor',
                '-ExecutorPythonClass=/Engine/PythonTypes.WorkflowRuntimeExecutor']
        write(run / 'command.json', command)
        with (run / 'stdout.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, pass_fds=(gpu_lock,))
        (run / 'process.exit').write_text(str(result.returncode) + '\n')
    proof = verify_native(run, task, plan['source_sha256'], plan['engine_version'], plan.get('trace_sha256'),
                          plan.get('pose_trace') if plan.get('pose_capture') == 'render_pass' else None)
    decode = ['docker', 'exec', plan['container'], 'ffmpeg', '-nostdin', '-v', 'error', '-xerror',
        '-framerate', str(task['fps']), '-start_number', '0', '-i',
        str(run / (task.get('file_prefix', 'kitchen_' + task['camera']) + '_%04d.png')), '-frames:v', str(task['frames']), '-f', 'null', '-']
    result = subprocess.run(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    write(run / 'decode.json', {'command': decode, 'exit_code': result.returncode, 'stderr': result.stderr,
                               'native_frame_hashes': [f['sha256'] for f in proof['frames']]})
    if result.returncode != 0 or result.stderr.strip():
        raise ValueError('Native PNG sequence decode failed')
    proof['full_native_decode'] = True
    write(run / 'native-qa.json', proof)
    return proof


def verified_resume(plan, task, run):
    run = Path(run)
    proof = verify_native(run, task, plan['source_sha256'], plan['engine_version'], plan.get('trace_sha256'),
                          plan.get('pose_trace') if plan.get('pose_capture') == 'render_pass' else None)
    saved = read(run / 'native-qa.json')
    decode = read(run / 'decode.json')
    if saved != dict(proof, full_native_decode=True) or decode['exit_code'] != 0 or decode['stderr'].strip() or decode['native_frame_hashes'] != [f['sha256'] for f in proof['frames']]:
        raise ValueError('Saved native QA or decode contract changed')


def run_batch(plan, out, *, resume=False, retry_failed=False):
    validate_manifest(plan)
    out = Path(out).resolve()
    root = Path(plan['root']).resolve()
    if not out.is_relative_to(root / 'experiments'):
        raise ValueError('Batch output must be under server experiments/')
    contract = package_contract(plan)
    if out.exists() and not resume:
        raise ValueError('Existing batch requires --resume')
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / 'batch.lock'):
        if (out / 'contract.json').exists():
            if read(out / 'contract.json') != contract:
                raise ValueError('Inputs, scripts, project or recipe changed; use a new batch')
        else:
            write(out / 'contract.json', contract)
        previous = read(out / 'progress.json') if (out / 'progress.json').exists() else {'tasks': []}
        old = {r['id']: r for r in previous['tasks']}
        progress = {'status': 'running', 'contract_sha256': contract['sha256'], 'started_at': time.time(),
                    'tasks': [], 'production_ready': False, 'benchmark_validated': False}
        for task in plan['tasks']:
            record = dict(old.get(task['id'], {'id': task['id'], 'status': 'queued'}))
            if record['status'] == 'completed':
                verified_resume(plan, task, record['run'])
                record['reused'] = True
                progress['tasks'].append(record)
                continue
            if record['status'] not in ('queued', 'failed'):
                raise ValueError('Unfinished attempt has unknown outcome; inspect its native process and receipts')
            if record['status'] == 'failed' and not retry_failed:
                raise ValueError('Inspect incomplete/failed task, then explicitly use --retry-failed')
            directory = out / task['id']
            directory.mkdir(exist_ok=True)
            attempt = max([int(p.name) for p in directory.iterdir() if p.is_dir() and p.name.isdigit()] or [0]) + 1
            run = directory / f'{attempt:03d}'
            run.mkdir()
            record.pop('error', None)
            record.pop('process_exit', None)
            record.update(status='running', run=str(run), reused=False, started_at=time.time())
            progress['tasks'].append(record)
            write(out / 'progress.json', progress)
            started = time.monotonic()
            try:
                # Reuse the same GPU lease as the existing Blender/Studio launcher.
                import runpy
                api = runpy.run_path(plan['gpu_launcher'])
                (root / 'runtime').mkdir(exist_ok=True)
                _, gpu_lock = acquire_gpu(plan, api)
                try:
                    render_attempt(plan, task, run, gpu_lock)
                finally:
                    os.close(gpu_lock)
                record.update(status='completed', process_exit=0)
            except Exception as error:
                record.update(status='failed', error=f'{type(error).__name__}: {error}')
            finally:
                record.update(elapsed_s=time.monotonic() - started, finished_at=time.time())
                write(run / 'attempt-exit.json', record)
                write(out / 'progress.json', progress)
        progress['status'] = 'completed' if all(r['status'] == 'completed' for r in progress['tasks']) else 'failed'
        progress['finished_at'] = time.time()
        progress['counts'] = {status: sum(r['status'] == status for r in progress['tasks']) for status in ('completed', 'failed')}
        write(out / 'progress.json', progress)
        write(out / 'batch-exit.json', progress)
        return 0 if progress['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    try:
        return run_batch(read(args.manifest), args.out, resume=args.resume, retry_failed=args.retry_failed)
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
