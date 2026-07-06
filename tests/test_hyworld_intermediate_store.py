import json
import tempfile
import unittest
from pathlib import Path

from pipeline.hyworld_intermediate_store import HYWorldIntermediateStore


class HYWorldIntermediateStoreTests(unittest.TestCase):
    def test_preserves_source_panorama_and_only_recopies_changed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_dir = Path(directory) / "scene_001"
            scene_dir.mkdir()
            panorama = scene_dir / "panorama.png"
            panorama.write_bytes(b"panorama-v1")

            store = HYWorldIntermediateStore(
                scene_dir / "intermediates",
                scene_id="scene_001",
                run_id="test-run",
            )
            source = store.capture(
                "00_source_panorama",
                [("scene", scene_dir, ["panorama.png"])],
                required=[("panorama", panorama)],
            )
            self.assertEqual(source["file_count"], 1)

            depth = scene_dir / "render_results" / "full_depth_prediction.pt"
            depth.parent.mkdir()
            depth.write_bytes(b"depth-v1")
            first_stage = store.capture("traj_generate", [("scene", scene_dir, ["**/*"])])
            self.assertEqual(first_stage["file_count"], 1)
            self.assertGreaterEqual(first_stage["unchanged_file_count"], 1)

            depth.write_bytes(b"depth-version-2")
            second_stage = store.capture("traj_render", [("scene", scene_dir, ["**/*"])])
            self.assertEqual(second_stage["file_count"], 1)
            store.finish("success")

            snapshot = (
                store.run_dir
                / "stages"
                / "00_source_panorama"
                / "scene"
                / "panorama.png"
            )
            self.assertEqual(snapshot.read_bytes(), b"panorama-v1")
            manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "success")
            self.assertEqual([stage["stage_id"] for stage in manifest["stages"]], [
                "00_source_panorama",
                "traj_generate",
                "traj_render",
            ])
            self.assertFalse(any("intermediates" in artifact["relative_path"] for artifact in first_stage["artifacts"]))

    def test_missing_required_artifact_is_recorded_before_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_dir = Path(directory) / "scene_002"
            scene_dir.mkdir()
            store = HYWorldIntermediateStore(
                scene_dir / "intermediates",
                scene_id="scene_002",
                run_id="failed-run",
            )
            panorama = scene_dir / "panorama.png"
            with self.assertRaises(FileNotFoundError):
                store.capture(
                    "00_source_panorama",
                    [("scene", scene_dir, ["panorama.png"])],
                    required=[("panorama", panorama)],
                )
            store.finish("failure", error="missing panorama")

            manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failure")
            self.assertEqual(manifest["stages"][0]["status"], "incomplete")
            self.assertEqual(manifest["stages"][0]["missing_required"][0]["label"], "panorama")


if __name__ == "__main__":
    unittest.main()
