import tempfile
import unittest
from pathlib import Path

from dataevolver.runtime.gpu_scheduler import (
    GPUInfo,
    JobRequest,
    ShardSpec,
    classify_failure,
    infer_gpu_reservations,
    plan_backfill_cycle,
    parse_gpu_spec,
    parse_nvidia_smi_gpu_csv,
    plan_gpu_leases,
    recommended_action,
)


class GPUShardSchedulerTests(unittest.TestCase):
    def test_parse_gpu_spec_supports_ranges_and_exclusions(self):
        includes, excludes = parse_gpu_spec("0-6,!5", total_gpus=8)
        self.assertEqual(includes, {0, 1, 2, 3, 4, 5, 6})
        self.assertEqual(excludes, {5})

    def test_parse_nvidia_smi_csv_handles_names_and_pstate(self):
        rows = "0, NVIDIA H100 80GB HBM3, 42383 MiB, 81559 MiB, 78 %, P0\n1, 4 MiB, 81559 MiB, 0 %\n"
        gpus = parse_nvidia_smi_gpu_csv(rows)
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[0].memory_used_mib, 42383)
        self.assertEqual(gpus[0].memory_total_mib, 81559)
        self.assertEqual(gpus[0].utilization_gpu, 78)
        self.assertEqual(gpus[1].free_mib, 81555)

    def test_planner_reserves_vlm_and_rejects_unsafe_high_resolution_pano(self):
        gpus = [
            GPUInfo(index=0, memory_used_mib=4, memory_total_mib=81559, utilization_gpu=0),
            GPUInfo(index=1, memory_used_mib=4, memory_total_mib=81559, utilization_gpu=0),
            GPUInfo(index=5, memory_used_mib=64955, memory_total_mib=81559, utilization_gpu=0),
        ]
        jobs = [
            JobRequest("pano_a", "hyworld_pano_h100_80gb", priority=1),
            JobRequest("pano_b", "hyworld_pano_h100_80gb", priority=2),
            JobRequest("scene_gate", "blender_render", priority=3),
        ]
        plan = plan_gpu_leases(gpus, jobs, include_spec="0-6,!5")
        self.assertEqual(plan.leases[0].job_id, "scene_gate")
        self.assertEqual(plan.leases[0].gpu_ids, (0,))
        self.assertEqual([item.job_id for item in plan.rejected], ["pano_a", "pano_b"])
        self.assertIn(">= 83000 MiB free", plan.rejected[0].reason)
        self.assertNotIn(5, plan.idle_gpu_ids)

    def test_lowres_pano_profile_allows_one_retry_only(self):
        gpus = [
            GPUInfo(index=0, memory_used_mib=4, memory_total_mib=81559, utilization_gpu=0),
            GPUInfo(index=1, memory_used_mib=4, memory_total_mib=81559, utilization_gpu=0),
        ]
        jobs = [
            JobRequest("pano_low_a", "hyworld_pano_lowres_h100_80gb", priority=1),
            JobRequest("pano_low_b", "hyworld_pano_lowres_h100_80gb", priority=2),
        ]
        plan = plan_gpu_leases(gpus, jobs, include_spec="0-1")
        self.assertEqual(plan.leases[0].job_id, "pano_low_a")
        self.assertEqual(plan.rejected[0].job_id, "pano_low_b")
        self.assertIn("max_parallel=1", plan.rejected[0].reason)

    def test_failure_classifier_returns_actionable_hyworld_advice(self):
        failure = "torch.OutOfMemoryError: CUDA out of memory in hyworld2/panogen/hunyuan_image_3/autoencoder_kl_3d.py"
        failure_class = classify_failure(failure)
        self.assertEqual(failure_class, "hyworld_pano_oom")
        self.assertIn("Do not launch more same-resolution HY-Pano", recommended_action(failure_class))
        self.assertEqual(classify_failure("/usr/bin/xvfb-run: line 184: blender: command not found"), "blender_missing")

    def test_dynamic_backfill_reserves_vllm_and_busy_unknown_gpu(self):
        gpus = [
            GPUInfo(index=0, memory_used_mib=4, memory_total_mib=81559, utilization_gpu=0),
            GPUInfo(index=1, memory_used_mib=4, memory_total_mib=81559, utilization_gpu=0),
            GPUInfo(index=5, memory_used_mib=64955, memory_total_mib=81559, utilization_gpu=0),
            GPUInfo(index=6, memory_used_mib=4, memory_total_mib=81559, utilization_gpu=0),
            GPUInfo(index=7, memory_used_mib=37741, memory_total_mib=81559, utilization_gpu=100),
        ]
        reservations = infer_gpu_reservations(gpus, include_spec="0-7")
        self.assertEqual({item.gpu_id for item in reservations}, {5, 7})

        shards = [
            ShardSpec("scene101_gate", "scene_view_gate", ("python", "gate.py"), priority=1),
            ShardSpec("scene102_rot", "rotation8_blender", ("python", "rot.py"), priority=2),
            ShardSpec("audit", "quality_audit_cpu", ("python", "audit.py"), priority=3),
        ]
        plan = plan_backfill_cycle(gpus, shards, include_spec="0-7")
        by_id = {lease.job_id: lease.gpu_ids for lease in plan.leases}
        self.assertEqual(by_id["scene101_gate"], (0,))
        self.assertEqual(by_id["scene102_rot"], (1,))
        self.assertEqual(by_id["audit"], ())
        self.assertNotIn(5, plan.idle_gpu_ids)
        self.assertNotIn(7, plan.idle_gpu_ids)

    def test_backfill_skips_completed_shards_by_expected_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "done.txt").write_text("ok", encoding="utf-8")
            gpus = [GPUInfo(index=0, memory_used_mib=4, memory_total_mib=81559, utilization_gpu=0)]
            shards = [
                ShardSpec("already_done", "rotation8_blender", ("python", "x.py"), expected_outputs=("done.txt",)),
                ShardSpec("pending", "rotation8_blender", ("python", "y.py")),
            ]
            plan = plan_backfill_cycle(gpus, shards, include_spec="0", base_dir=root)
            self.assertEqual(plan.skipped_completed, ("already_done",))
            self.assertEqual([lease.job_id for lease in plan.leases], ["pending"])


if __name__ == "__main__":
    unittest.main()
