"""Registered serial host for selected visual models; no planning or retries."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json


def serve(configs, state_root, *, job_roots, dataset_roots):
    import torch
    from . import local_visual_worker, training_media_resident
    from .training_media_worker import actual_device_map, gpu_observation
    from .visual_models import MODELS, validate_visual_worker, verify_assets
    if torch.cuda.device_count() != 1 or ',' in os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise ValueError('visual resident requires one explicitly mapped GPU')
    if not configs or len({c['threads'] for c in configs}) != 1:
        raise ValueError('resident models must share one explicit host thread configuration')
    torch.set_num_threads(configs[0]['threads'])
    torch.set_num_interop_threads(1)
    root = Path(state_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    endpoint = root/'worker.sock'
    if len(os.fsencode(endpoint)) >= 108:
        raise ValueError('resident Unix socket path exceeds the Linux limit; choose a shorter state root')
    if endpoint.exists():
        raise ValueError('resident endpoint already exists; do not replace an owned host')
    models, identities = {}, {}
    for config in configs:
        validate_visual_worker(config)
        if config.get('resident') or config['kind'] == 'h3_turbo' or config['kind'] in models:
            raise ValueError('startup needs unique ephemeral visual configs')
        gpu_observation(config)
        assets = verify_assets(config)
        kind = config['kind']
        runtime = {}
        if kind == 'sam3':
            from transformers import Sam3Processor, Sam3VideoModel
            from .resident_components import sam_components
            processor = Sam3Processor.from_pretrained(assets['model'], local_files_only=True)
            video, info = Sam3VideoModel.from_pretrained(assets['model'], dtype=torch.float32,
                local_files_only=True, output_loading_info=True)
            video = video.to('cuda:0').eval()
            components = sam_components(processor=processor, video_model=video, loading_info=info)
            placement = actual_device_map(video)
        else:
            components = local_visual_worker._load_qwen({'model_path': assets['model'], 'config': config}, runtime)
            if kind == 'qwen21':
                components.to('cuda:0')
                placement = actual_device_map(components)
            else:
                placement = actual_device_map(components['model'])
        models[kind] = {'components': components, 'model_path': assets['model'], 'loading_runtime': runtime}
        identities[kind] = {'model': MODELS[kind], 'path': assets['model'],
            'assets_sha256': payload_sha256(config['file_hashes']), 'device_map': placement, 'ready': True}
    handler = Path(local_visual_worker.__file__).resolve()
    identity = {'deployment_id': payload_sha256([identities, os.getpid(), time.time()]),
        'models': identities, 'pid': os.getpid(), 'socket_path': str(endpoint),
        'gpu_uuids': os.environ['CUDA_VISIBLE_DEVICES'].split(','),
        'process_start_ticks': Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19],
        'handler_path': str(handler), 'handler_sha256': file_sha256(handler),
        'resident_implementation_sha256': file_sha256(Path(training_media_resident.__file__)),
        'authorized_job_roots': [str(Path(p).resolve()) for p in job_roots],
        'authorized_dataset_roots': [str(Path(p).resolve()) for p in dataset_roots]}
    atomic_write_json(root/'identity.json', identity)
    status = {'pid': os.getpid(), 'models': identities, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
              'inference_requests': 0}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(endpoint)); server.listen(4)
        training_media_resident.serve_resident(server, models, status, root, root/'identity.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.deployment).read_text())
    serve(value['configs'], value['state_root'], job_roots=value['job_roots'], dataset_roots=value['dataset_roots'])


if __name__ == '__main__':
    main()
