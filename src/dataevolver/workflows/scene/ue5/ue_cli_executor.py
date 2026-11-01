"""One-job native runtime MRQ executor, following Epic's bundled runtime example."""
import hashlib
import json
import os
from pathlib import Path
import struct
import traceback
import unreal
from trajectory import assess_runtime_poses

pose_records = []
pose_errors = []
pose_actors = {}
pose_trace = None


@unreal.uclass()
class WorkflowRuntimeExecutor(unreal.MoviePipelinePythonHostExecutor):
    active_pipeline = unreal.uproperty(unreal.MoviePipeline)

    @unreal.ufunction(override=True)
    def execute_delayed(self, queue):
        try:
            global pose_trace, pose_actors
            assert queue and len(queue.get_jobs()) == 1
            world = self.get_last_loaded_world()
            mode = unreal.GameplayStatics.get_game_mode(world)
            pawns = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Pawn)
            output = Path(os.environ['DATAEVOLVER_RENDER_OUTPUT'])
            receipt = json.loads((output / 'render-receipt.json').read_text())
            spawned = [dict(name=p.get_name(), class_name=p.get_class().get_path_name()) for p in pawns]
            # Imported GLB scenes contain no Pawn; remove only runtime game avatars.
            for pawn in pawns:
                pawn.destroy_actor()
            remaining = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Pawn)
            receipt['runtime_world'] = dict(game_mode=mode.get_class().get_path_name(), pawn_count=len(remaining), removed_game_pawns=spawned)
            (output / 'render-receipt.json').write_text(json.dumps(receipt, indent=2))
            assert not remaining, 'Runtime game avatars were not removed'
            if int(os.environ['DATAEVOLVER_ANIMATION_FRAMES']) > 1:
                if 'DATAEVOLVER_SCENE_CONFIG' in os.environ:
                    scene = json.loads(Path(os.environ['DATAEVOLVER_SCENE_CONFIG']).read_text())
                    source = Path(scene['package']) / 'trace.json'
                else:
                    source = Path(receipt['animation']['source'])
                pose_trace = json.loads(source.read_text())
                inventory = unreal.GameplayStatics.get_all_actors_of_class(self.get_last_loaded_world(), unreal.Actor)
                pose_actors = {a.get_actor_label(): a for a in inventory if a.get_actor_label() in pose_trace['actor_names']}
                assert set(pose_actors) == set(pose_trace['actor_names'])
            job = queue.get_jobs()[0]
            job.get_configuration().initialize_transient_settings()
            self.active_pipeline = unreal.new_object(self.target_pipeline_class, outer=self.get_last_loaded_world(), base_type=unreal.MoviePipeline)
            self.active_pipeline.on_movie_pipeline_work_finished_delegate.add_function_unique(self, 'render_finished')
            self.active_pipeline.initialize(job)
        except Exception:
            output = Path(os.environ['DATAEVOLVER_RENDER_OUTPUT'])
            receipt = json.loads((output / 'render-receipt.json').read_text())
            receipt.update(success=False, executor_success=False, exception=traceback.format_exc())
            (output / 'render-receipt.json').write_text(json.dumps(receipt, indent=2))
            unreal.log_error(receipt['exception'])
            self.on_executor_errored()
            self.on_executor_finished_impl()

    @unreal.ufunction(override=True)
    def is_rendering(self):
        return self.active_pipeline is not None

    @unreal.ufunction(override=True)
    def on_begin_frame(self):
        super(WorkflowRuntimeExecutor, self).on_begin_frame()
        try:
            pipeline = self.active_pipeline
            if not pose_trace or not pipeline or unreal.MoviePipelineLibrary.get_current_segment_state(pipeline) != unreal.MovieRenderShotState.RENDERING:
                return
            frame = unreal.MoviePipelineLibrary.get_root_frame_number(pipeline).value
            if not 0 <= frame < len(pose_trace['frames']):
                return
            record = dict(output_frame=frame, actors={})
            for name, actor in pose_actors.items():
                transform = actor.get_actor_transform()
                rotation = [list(transform.rotation.rotate_vector(unreal.Vector(*(int(i == axis) for i in range(3)))).to_tuple()) for axis in range(3)]
                record['actors'][name] = dict(location_cm=list(transform.translation.to_tuple()), rotation_columns=rotation,
                                             scale=list(transform.scale3d.to_tuple()))
            pose_records.append(record)
        except Exception:
            if not pose_errors:
                pose_errors.append(traceback.format_exc())

    @unreal.ufunction(ret=None, params=[unreal.MoviePipelineOutputData])
    def render_finished(self, results):
        output = Path(os.environ['DATAEVOLVER_RENDER_OUTPUT'])
        receipt = json.loads((output / 'render-receipt.json').read_text())
        receipt.update(execution_mode='runtime_cli', executor_success=bool(results.success),
                       runtime_executor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       native_frames=[], success=False)
        try:
            if pose_trace:
                (output / 'runtime-poses.json').write_text(json.dumps(pose_records))
                receipt['runtime_pose_capture'] = assess_runtime_poses(pose_records, pose_trace, pose_errors)
                receipt['runtime_pose_capture']['scope'] = 'Game world at runtime executor on_begin_frame; not a render-pass callback'
            resolution = json.loads(os.environ['DATAEVOLVER_RESOLUTION'])
            for path in sorted(output.glob('*.png')):
                data = path.read_bytes()
                assert data[:8] == b'\x89PNG\r\n\x1a\n'
                assert list(struct.unpack('>II', data[16:24])) == resolution
                receipt['native_frames'].append(dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            receipt['success'] = bool(results.success and len(receipt['native_frames']) == int(os.environ['DATAEVOLVER_ANIMATION_FRAMES']))
            if os.environ.get('DATAEVOLVER_RENDER_PASS_POSE') == '1':
                from render_pose_qa import assess
                trace = json.loads(Path(os.environ['DATAEVOLVER_POSE_TRACE']).read_text())
                receipt['render_pass_pose_capture'] = assess(output / 'render-pass-poses.jsonl', trace,
                    int(os.environ['DATAEVOLVER_ANIMATION_FRAMES']), int(os.environ['DATAEVOLVER_SPATIAL_SAMPLES']))
        except Exception:
            receipt['success'] = False
            receipt['exception'] = traceback.format_exc()
        (output / 'render-receipt.json').write_text(json.dumps(receipt, indent=2))
        unreal.log('WORKFLOW_RUNTIME_MRQ_FINISHED success=' + str(receipt['success']))
        self.active_pipeline = None
        self.on_executor_finished_impl()
