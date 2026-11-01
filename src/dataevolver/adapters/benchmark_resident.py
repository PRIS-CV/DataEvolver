"""Load frozen benchmark models once and serve the existing native transport."""
import argparse
import json
import os
from pathlib import Path
import socket
import time

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json


def serve(deployment):
    import torch
    from . import benchmark_worker, training_media_resident
    from .training_media_worker import actual_device_map, gpu_observation
    root = Path(deployment['state_root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    endpoint = root/'worker.sock'
    if len(os.fsencode(endpoint)) >= 108 or endpoint.exists():
        raise ValueError('benchmark socket must be a new, short owned path')
    configs = deployment['configs']
    if not configs or len({c['threads'] for c in configs}) != 1:
        raise ValueError('benchmark host needs one frozen thread configuration')
    torch.set_num_threads(configs[0]['threads']); torch.set_num_interop_threads(1)
    input_roots = {}
    for config in configs:
        for name, env in config.get('input_store_envs', {}).items():
            path = Path(os.environ[env]).resolve(strict=True)
            if not path.is_dir() or (name in input_roots and input_roots[name] != str(path)):
                raise ValueError('benchmark input store mapping differs across models')
            input_roots[name] = str(path)
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(visible) != torch.cuda.device_count():
        raise ValueError('benchmark CUDA mapping changed')
    models, identities = {}, {}
    for config in configs:
        benchmark_worker.validate_worker(config)
        if config.get('resident') or config['kind'] in models:
            raise ValueError('benchmark startup requires unique ephemeral model configs')
        observed = dict(config)
        if config.get('loading'):
            observed.update(shard_max_memory_mib=config['loading']['max_memory_mib'],
                            shard_min_free_mib=config['loading']['min_free_mib'])
        else:
            # A small model shares process device 0 with the sharded model.
            os.environ['CUDA_VISIBLE_DEVICES'] = visible[0]
        try:
            preflight = gpu_observation(observed)
        finally:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(visible)
        model_path = benchmark_worker.verify_assets(config)
        entry = benchmark_worker.load_components(config, model_path)
        mapping = actual_device_map(entry['model'])
        if any(str(v) in {'cpu', 'disk', 'meta'} for v in mapping.values()):
            raise ValueError('benchmark resident requires actual CUDA weights')
        kind = config['kind']
        models[kind] = {'components': entry, 'model_path': model_path}
        identities[kind] = {'model': benchmark_worker.benchmark_model(config), 'path': model_path,
            'assets_sha256': payload_sha256(config['file_hashes']), 'device_map': mapping,
            'ready': True, 'preload_observation': preflight}
    handler = Path(benchmark_worker.__file__).resolve()
    identity = {'deployment_id': payload_sha256([identities, os.getpid(), time.time()]),
        'models': identities, 'pid': os.getpid(), 'socket_path': str(endpoint),
        'gpu_uuids': visible,
        'input_store_roots': input_roots,
        'process_start_ticks': Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19],
        'handler_path': str(handler), 'handler_sha256': file_sha256(handler),
        'resident_implementation_sha256': file_sha256(Path(training_media_resident.__file__)),
        'authorized_job_roots': [str(Path(p).resolve()) for p in deployment['job_roots']],
        'authorized_dataset_roots': [str(Path(p).resolve()) for p in deployment['dataset_roots']]}
    atomic_write_json(root/'identity.json', identity)
    status = {'pid': os.getpid(), 'models': identities,
              'cuda_visible_devices': ','.join(visible), 'inference_requests': 0}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(endpoint)); server.listen(4)
        training_media_resident.serve_resident(server, models, status, root, root/'identity.json')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--deployment', required=True)
    serve(json.loads(Path(parser.parse_args().deployment).read_text()))


if __name__ == '__main__':
    main()
