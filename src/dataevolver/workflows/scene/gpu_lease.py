"""Reuse the native Blender/UE GPU lease without its standalone launcher."""
import fcntl
import json
import os
from pathlib import Path


def reserved_gpu_uuids(root, proc_root=Path('/proc')):
    reserved = set()
    for path in (root / 'experiments/mcp-sessions').glob('*/session.json'):
        try:
            session = json.loads(path.read_text())
            command = (proc_root / str(int(session['pid'])) / 'cmdline').read_bytes().split(b'\0')
            server = Path(session['server']).resolve()
            if any(arg and Path(os.fsdecode(arg)).is_absolute()
                   and Path(os.fsdecode(arg)).resolve() == server for arg in command):
                reserved.update(gpu['uuid'] for gpu in session.get('gpus',[session['gpu']]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return reserved


def sharing_policy(root):
    path = root / 'runtime/blender-gpu-sharing.json'
    if not path.exists():
        return dict(gpu_uuids=[], min_free_mib=16384)
    policy = json.loads(path.read_text())
    if (not isinstance(policy.get('gpu_uuids'), list)
            or not all(isinstance(u, str) and u.startswith('GPU-') for u in policy['gpu_uuids'])
            or type(policy.get('min_free_mib')) is not int or policy['min_free_mib'] < 16384
            or type(policy.get('max_peer_slots', 2)) is not int
            or policy.get('max_peer_slots', 2) not in (2, 3)):
        raise ValueError('Invalid Blender GPU sharing policy; minimum headroom is 16384 MiB')
    return policy


def lease_gpu(root, devices, reserved, *, allow_shared=False, max_utilization=5):
    if type(max_utilization) is not int or not 0 <= max_utilization <= 100:
        raise ValueError('GPU utilization ceiling must be an integer from 0 to 100')
    configured = sharing_policy(root)
    policy = configured if allow_shared else dict(gpu_uuids=[], min_free_mib=16384)
    for gpu in sorted(devices, key=lambda d: (-d['free_mib'], d['utilization'], d['index'])):
        primary = root / 'runtime' / ('mcp-gpu-' + gpu['uuid'] + '.lock')
        secondary = primary.with_name(primary.stem + '-shared.lock')
        paths = [primary, secondary]
        if configured.get('max_peer_slots', 2) == 3:
            paths.append(primary.with_name(primary.stem + '-shared-2.lock'))
        share = gpu['uuid'] in policy['gpu_uuids'] and gpu['free_mib'] >= policy['min_free_mib']
        slots = list(zip(paths, range(len(paths)))) if share and gpu['uuid'] in reserved else []
        if gpu['uuid'] not in reserved and gpu['free_mib'] >= 4096 and gpu['utilization'] <= max_utilization:
            probes = []
            try:
                for path in paths[1:]:
                    probe = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                    probes.append(probe)
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                slots = [(primary, 0)]
            except BlockingIOError:
                slots = []
            finally:
                for probe in probes:
                    os.close(probe)
        for path, slot in slots:
            lock = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(lock)
                continue
            os.set_inheritable(lock, True)
            return dict(gpu, lease_slot=slot, shared_peer=gpu['uuid'] in reserved), lock
    raise RuntimeError('No GPU lease available; check reservations, sharing policy and free memory')
