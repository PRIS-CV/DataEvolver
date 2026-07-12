import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


from dataevolver.runtime.hyworld_scene_contract import (
    SceneContractError,
    depth_statistics,
    extract_support_surface,
    load_depth_payload,
    load_scene_contract,
    radial_statistics,
    validate_non_shell_geometry,
)


class FakeMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        triangles = self.vertices[self.faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        self.area_faces = lengths * 0.5
        self.face_normals = cross / lengths[:, None]


class SceneContractTests(unittest.TestCase):
    def test_constant_radius_shell_is_rejected(self):
        theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        vertices = np.stack([8 * np.cos(theta), 8 * np.sin(theta), np.zeros_like(theta)], axis=1)
        radial = radial_statistics(vertices)
        with self.assertRaisesRegex(SceneContractError, "constant-radius"):
            validate_non_shell_geometry(radial)

    def test_constant_depth_is_rejected(self):
        radial = {
            "coefficient_of_variation": 0.2,
            "relative_range": 0.8,
        }
        depth = depth_statistics(np.full((32, 64), 8.0, dtype=np.float32))
        with self.assertRaisesRegex(SceneContractError, "effectively constant"):
            validate_non_shell_geometry(radial, depth)

    def test_loads_worldmirror_depth_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "depth_0000.npy", np.array([[1.0, 2.0]], dtype=np.float32))
            np.save(root / "depth_0001.npy", np.array([[3.0, 4.0]], dtype=np.float32))
            values, stats = load_depth_payload(root)
        self.assertEqual(values.size, 4)
        self.assertEqual(stats["source_file_count"], 2)
        self.assertGreater(stats["coefficient_of_variation"], 0.0)

    def test_extracts_real_mesh_support_surface(self):
        vertices = [
            [-2, -2, 0],
            [2, -2, 0],
            [2, 2, 0],
            [-2, 2, 0],
            [-1, -1, 2],
            [1, -1, 2],
            [0, 1, 3],
        ]
        faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6]]
        support = extract_support_surface(
            FakeMesh(vertices, faces),
            object_name="HYWorldSceneMesh",
            minimum_area=1.0,
            minimum_span=1.0,
        )
        self.assertEqual(support.object_name, "HYWorldSceneMesh")
        self.assertAlmostEqual(support.height, 0.0)
        self.assertGreaterEqual(support.area, 16.0)

    def test_support_area_is_preserved_when_faces_are_sampled(self):
        mesh = FakeMesh(
            [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]],
            [[0, 1, 2], [0, 2, 3]],
        )
        support = extract_support_surface(
            mesh,
            object_name="HYWorldSceneMesh",
            minimum_area=1.0,
            minimum_span=1.0,
            max_faces=1,
        )
        self.assertGreaterEqual(support.area, 16.0)

    def test_contract_rejects_artificial_support(self):
        payload = {
            "schema": "dataevolver.hyworld_scene.v1",
            "status": "ready",
            "selected_support_surface_id": "support_000",
            "support_surfaces": [{"id": "support_000", "artificial": True}],
            "render_policy": {"artificial_support_allowed": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SceneContractError, "reconstructed support"):
                load_scene_contract(path)


if __name__ == "__main__":
    unittest.main()
