"""Local contract/recovery tests; these fixtures do not run UE or validate physics."""
import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import zlib

SPEC = importlib.util.spec_from_file_location('ue5_batch', Path(__file__).resolve().parents[1] / 'src/dataevolver/workflows/scene/ue5/render_batch.py')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/dataevolver/workflows/scene/ue5'))
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def png(width, height):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress((b'\0' + b'\0' * width * 3) * height)) + chunk(b'IEND', b'')


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'template' / 'NativeRender.uproject'
        self.project.parent.mkdir()
        self.project.write_text('{}')
        for name in ('engine', 'render.py', 'launcher.py', 'input.json'):
            (self.root / name).write_text('fixture')
        self.plan = dict(schema_version=1, recipe='kitchen-rigid-v1', root=str(self.root), project=str(self.project),
            engine=str(self.root / 'engine'), renderer=str(self.root / 'render.py'), gpu_launcher=str(self.root / 'launcher.py'),
            dependencies=[str(self.root / 'input.json')], container='fixture', user='fixture', expected_host='fixture',
            gpu_uuid='GPU-fixture', min_free_mib=24000, max_utilization=5, timeout_s=60,
            engine_version='5.8.4', source_sha256='a' * 64,
            tasks=[dict(id='one', camera='Reference_Main', frames=1, fps=24, samples=8, resolution=[128, 72])])
        self.task = self.plan['tasks'][0]

    def native(self, run, task=None):
        task = task or self.task
        run.mkdir(parents=True, exist_ok=True)
        (run / 'process.exit').write_text('0\n')
        (run / 'stdout.log').write_text('LogMovieRenderPipeline: completed\n')
        prefix = task.get('file_prefix', 'kitchen_' + task['camera'])
        path = run / f"{prefix}_0000.png"
        path.write_bytes(png(*task['resolution']))
        receipt = dict(success=True, executor_success=True, engine='5.8.4-build', camera=task['camera'],
            resolution=task['resolution'], spatial_samples=task['samples'], source_sha256=self.plan['source_sha256'],
            native_frames=[dict(path=str(path), bytes=path.stat().st_size, sha256=batch.sha256(path))])
        batch.write(run / 'render-receipt.json', receipt)
        return receipt

    def test_exit139_rejected_even_when_mrq_and_png_succeeded(self):
        run = self.root / 'run'
        self.native(run)
        (run / 'process.exit').write_text('139\n')
        with self.assertRaisesRegex(ValueError, 'exit zero'):
            batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4')

    def test_modified_missing_and_extra_frames_rejected(self):
        for mutation in ('changed', 'missing', 'extra'):
            with self.subTest(mutation=mutation):
                run = self.root / mutation
                receipt = self.native(run)
                path = Path(receipt['native_frames'][0]['path'])
                if mutation == 'changed':
                    path.write_bytes(path.read_bytes() + b'changed')
                elif mutation == 'missing':
                    path.unlink()
                else:
                    (run / 'extra.png').write_bytes(path.read_bytes())
                with self.assertRaises(ValueError):
                    batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4')

    def test_render_success_does_not_certify_physics_or_realism(self):
        run = self.root / 'run'
        self.native(run)
        proof = batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4')
        self.assertEqual(proof['status'], 'passed')
        self.assertFalse(proof['production_ready'])
        self.assertFalse(proof['benchmark_validated'])
        self.assertEqual(proof['physical_truth'], 'unknown')
        self.assertEqual(proof['runtime_pose_validation'], 'unknown')

    def test_shader_error_bad_source_and_camera_rejected(self):
        for field in ('shader', 'prepare_shader', 'source_sha256', 'camera'):
            with self.subTest(field=field):
                run = self.root / field
                receipt = self.native(run)
                if field == 'shader':
                    (run / 'stdout.log').write_text('Failed to compile Material M_Test')
                elif field == 'prepare_shader':
                    (run / 'prepare.log').write_text('Failed to compile Material M_Test')
                else:
                    receipt[field] = 'wrong'
                    batch.write(run / 'render-receipt.json', receipt)
                with self.assertRaises(ValueError):
                    batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4')

    def test_duplicate_ids_invalid_frames_and_traversal_rejected(self):
        for kind in ('duplicate', 'frames', 'path', 'boolean'):
            plan = copy.deepcopy(self.plan)
            if kind == 'duplicate':
                plan['tasks'].append(copy.deepcopy(self.task))
            elif kind == 'path':
                plan['tasks'][0]['id'] = '../escape'
            else:
                plan['tasks'][0]['frames'] = 2 if kind == 'frames' else True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                batch.validate_manifest(plan)

    def test_runtime_executor_is_a_pinned_dependency(self):
        plan = copy.deepcopy(self.plan)
        plan.update(execution_mode='runtime_cli', runtime_executor=str(self.root / 'runtime.py'))
        with self.assertRaisesRegex(ValueError, 'pinned dependency'):
            batch.validate_manifest(plan)
        plan['dependencies'].append(plan['runtime_executor'])
        batch.validate_manifest(plan)
        plan['execution_mode'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'execution mode'):
            batch.validate_manifest(plan)

    def test_runtime_image_init_and_missing_library_rejected_before_render(self):
        self.plan['container_image'] = 'sha256:' + 'b' * 64
        container = dict(Image=self.plan['container_image'], HostConfig=dict(Init=True))
        gpu = '0, GPU-fixture, 100000, 0\n'
        for problem in ('image', 'init', 'library', 'none'):
            actual = copy.deepcopy(container)
            libraries = 'libc.so => /lib/libc.so\n'
            if problem == 'image': actual['Image'] = 'sha256:' + 'c' * 64
            if problem == 'init': actual['HostConfig']['Init'] = False
            if problem == 'library': libraries = 'libatk-1.0.so.0 => not found\n'
            outputs = [gpu, json.dumps([actual]), libraries, 'GPU-fixture\n', '1 docker-init\n', '590.48\n']
            with self.subTest(problem=problem), patch.object(batch.socket, 'gethostname', return_value='fixture'), \
                    patch.object(batch.subprocess, 'check_output', side_effect=outputs):
                if problem == 'none':
                    result = batch.preflight(self.plan)
                    self.assertEqual(result['container_image'], container['Image'])
                    self.assertEqual(result['engine_shared_libraries'], 'resolved')
                else:
                    with self.assertRaises(RuntimeError): batch.preflight(self.plan)

    def test_resume_recomputes_raw_render_pass_pose_evidence(self):
        run = self.root / 'pose'
        receipt = self.native(run)
        matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]
        trace = dict(actor_names=['ball'], frames=[dict(objects=dict(ball=matrix))])
        trace_path = self.root / 'trace.json'
        batch.write(trace_path, trace)
        pose = dict(location_cm=[0,0,100], scale=[1,1,1], rotation_columns=[[1,0,0],[0,1,0],[0,0,1]])
        records = [dict(output_frame=0, source_frame=0, spatial_sample=s, temporal_sample=0,
            spatial_count=8, temporal_count=1, view_location_cm=[0,0,0], view_rotation_degrees=[0,0,0],
            actors=dict(ball=dict(pose, components=[copy.deepcopy(pose)]))) for s in range(8)]
        path = run / 'render-pass-poses.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in records))
        from render_pose_qa import assess
        receipt['render_pass_pose_capture'] = assess(path, trace, 1, 8)
        batch.write(run / 'render-receipt.json', receipt)
        self.plan.update(pose_capture='render_pass', pose_trace=str(trace_path))
        proof = batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4', pose_trace=trace_path)
        batch.write(run / 'native-qa.json', dict(proof, full_native_decode=True))
        batch.write(run / 'decode.json', dict(exit_code=0, stderr='', native_frame_hashes=[f['sha256'] for f in proof['frames']]))
        batch.verified_resume(self.plan, self.task, run)
        records[-1]['actors']['ball']['components'][0]['location_cm'][0] = 1
        path.write_text('\n'.join(json.dumps(r) for r in records))
        with self.assertRaises(AssertionError): batch.verified_resume(self.plan, self.task, run)

    def test_resume_rechecks_files_skips_valid_jobs_and_rejects_recipe_drift(self):
        out = self.root / 'experiments/batch'
        out.mkdir(parents=True)
        contract = batch.package_contract(self.plan)
        batch.write(out / 'contract.json', contract)
        run = out / 'one/001'
        self.native(run)
        proof = batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4')
        batch.write(run / 'native-qa.json', dict(proof, full_native_decode=True))
        batch.write(run / 'decode.json', dict(exit_code=0, stderr='', native_frame_hashes=[f['sha256'] for f in proof['frames']]))
        batch.write(out / 'progress.json', dict(tasks=[dict(id='one', status='completed', run=str(run))]))
        with patch.object(batch, 'render_attempt') as render:
            self.assertEqual(batch.run_batch(self.plan, out, resume=True), 0)
            render.assert_not_called()
        self.assertTrue(batch.read(out / 'progress.json')['tasks'][0]['reused'])
        (self.root / 'input.json').write_text('different frozen input')
        with self.assertRaisesRegex(ValueError, 'changed'):
            batch.run_batch(self.plan, out, resume=True)

    def test_unknown_inflight_attempt_not_replayed_by_retry_flag(self):
        out = self.root / 'experiments/batch'
        out.mkdir(parents=True)
        batch.write(out / 'contract.json', batch.package_contract(self.plan))
        batch.write(out / 'progress.json', dict(tasks=[dict(id='one', status='running')]))
        with patch.object(batch, 'render_attempt') as render:
            with self.assertRaisesRegex(ValueError, 'unknown outcome'):
                batch.run_batch(self.plan, out, resume=True, retry_failed=True)
            render.assert_not_called()

    def test_harness_render_index_reads_native_frames_and_rejects_input_drift(self):
        from dataevolver.workflows.scene.ue5_worker import verify_render_index
        directory = self.root / 'native/nested'
        directory.mkdir(parents=True)
        out = directory / 'render'
        out.mkdir()
        manifest = directory / 'render-manifest.json'
        batch.write(manifest, self.plan)
        contract = batch.package_contract(self.plan)
        batch.write(out / 'contract.json', contract)
        run = out / 'one/001'
        self.native(run)
        proof = batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4')
        batch.write(run / 'native-qa.json', dict(proof, full_native_decode=True))
        batch.write(run / 'decode.json', dict(exit_code=0, stderr='', native_frame_hashes=[f['sha256'] for f in proof['frames']]))
        batch.write(out / 'batch-exit.json', dict(status='completed', contract_sha256=contract['sha256'],
            tasks=[dict(id='one', status='completed', run=str(run))]))
        index = directory / 'render-index.json'
        batch.write(index, dict(schema_version='dataevolver.ue5_render.v1',
            manifest=dict(path=str(manifest), sha256=batch.sha256(manifest)), batch=str(out)))
        verify_render_index(index)
        (self.root / 'input.json').write_text('changed dependency')
        with self.assertRaises(ValueError):
            verify_render_index(index)

    def test_outdoor_recipe_binds_camera_prefix_and_trace(self):
        self.plan.update(recipe='frozen-scene-v1', scene_config=str(self.root / 'input.json'), trace_sha256='b' * 64)
        self.task.update(camera='Collision_Main', frames=96, file_prefix='scene_Collision_Main')
        batch.validate_manifest(self.plan)
        run = self.root / 'outdoor'
        receipt = self.native(run)
        original = Path(receipt['native_frames'][0]['path'])
        for index in range(1, 96):
            path = run / f'scene_Collision_Main_{index:04d}.png'
            path.write_bytes(original.read_bytes())
            receipt['native_frames'].append(dict(path=str(path), bytes=path.stat().st_size, sha256=batch.sha256(path)))
        receipt['animation'] = dict(frames=96, fps=24, physics_recomputed=False, source_sha256='b' * 64,
            max_native_actor_translation_error_cm=0, max_native_actor_rotation_matrix_error=0, max_channel_key_readback_error=0)
        batch.write(run / 'render-receipt.json', receipt)
        batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4', 'b' * 64)
        with self.assertRaisesRegex(ValueError, 'trajectory hash'):
            batch.verify_native(run, self.task, self.plan['source_sha256'], '5.8.4', 'c' * 64)
        self.task['file_prefix'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'prefix'):
            batch.validate_manifest(self.plan)

    def test_admission_waits_for_recent_gpu_work_but_not_wrong_container(self):
        api = dict(lease_gpu=lambda *args, **kwargs: ('gpu', 123), reserved_gpu_uuids=lambda root: set())
        with patch.object(batch, 'lock', return_value=nullcontext()), patch.object(batch.time, 'sleep') as sleep:
            with patch.object(batch, 'preflight', side_effect=[RuntimeError('Bound GPU is busy or has insufficient memory'), {'gpu': {}}]):
                self.assertEqual(batch.acquire_gpu(self.plan, api), ('gpu', 123))
                sleep.assert_called_once_with(2)
            with patch.object(batch, 'preflight', side_effect=RuntimeError('Container already has an UnrealEditor process')):
                with self.assertRaisesRegex(RuntimeError, 'already has'):
                    batch.acquire_gpu(self.plan, api)

    def test_virtual_display_cleanup_on_render_exception_preserves_existing_display(self):
        self.plan['display_backend'] = 'xvfb'
        run = self.root / 'display'
        run.mkdir()
        process = Mock()
        process.poll.return_value = None
        (run / 'xvfb.pid').write_text('12345')
        with patch.object(batch.subprocess, 'run', side_effect=[Mock(returncode=1), Mock(returncode=0), Mock(returncode=0)]) as command, patch.object(batch.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'render failed'):
                with batch.virtual_display(self.plan, run):
                    raise RuntimeError('render failed')
            self.assertEqual(command.call_args_list[-1].args[0][-3:], ['kill','-TERM','12345'])
            process.wait.assert_called_once_with(timeout=15)
        with patch.object(batch.subprocess, 'run', return_value=Mock(returncode=0)) as command, patch.object(batch.subprocess, 'Popen') as spawn:
            with batch.virtual_display(self.plan, run):
                pass
            spawn.assert_not_called()
            self.assertEqual(command.call_count, 1)


if __name__ == '__main__':
    unittest.main()
