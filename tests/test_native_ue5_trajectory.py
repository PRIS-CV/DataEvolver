"""Coordinate/timeline checks, without claiming physical calibration."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('trajectory', Path(__file__).resolve().parents[1] / 'src/dataevolver/workflows/scene/ue5/trajectory.py')
trajectory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trajectory)


class TrajectoryTests(unittest.TestCase):
    def test_aimed_camera_centres_target_and_rejects_singular_views(self):
        camera = trajectory.aimed_camera(dict(lens_mm=40, sensor_width_mm=36),
            dict(location_m=[0,-5,3], target_m=[0,0,1]))
        target = [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]
        trace = dict(actor_names=['A'], frames=[dict(objects={'A':target})])
        self.assertTrue(trajectory.camera_coverage(camera, trace, 16/9)['first_frame_all_centres_in_view'])
        for position in ([0,0,1], [0,0,3]):
            with self.assertRaises(AssertionError):
                trajectory.aimed_camera({}, dict(location_m=position, target_m=[0,0,1]))

    def test_runtime_pose_coverage_and_empty_evidence(self):
        matrix = [[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]
        trace = dict(actor_names=['A'], frames=[dict(objects={'A':matrix})])
        self.assertIsNone(trajectory.assess_runtime_poses([], trace, [])['max_translation_error_cm'])
        actual = dict(location_cm=[100,-200,300], scale=[1,1,1], rotation_columns=[[1,0,0],[0,1,0],[0,0,1]])
        records = [dict(output_frame=0, actors={'A':actual})]
        result = trajectory.assess_runtime_poses(records, trace, [])
        self.assertTrue(result['full_frame_coverage'])
        self.assertEqual(result['max_translation_error_cm'], 0)
        actual['location_cm'][1] += 2
        self.assertEqual(trajectory.assess_runtime_poses(records, trace, [])['max_translation_error_cm'], 2)

    def test_camera_rejects_behind_and_outside_centres(self):
        identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        camera = dict(matrix_world=identity, lens_mm=50, sensor_width_mm=36)
        pose = copy.deepcopy(identity)
        pose[2][3] = -5
        trace = dict(actor_names=['A'], frames=[dict(objects={'A':pose})])
        self.assertTrue(trajectory.camera_coverage(camera, trace, 16/9)['first_frame_all_centres_in_view'])
        for x, z in ((0, 5), (3, -5)):
            pose[0][3], pose[2][3] = x, z
            result = trajectory.camera_coverage(camera, trace, 16/9)
            self.assertFalse(result['first_frame_all_centres_in_view'])
            self.assertEqual(result['outside_frames_zero_based'], {'A':[0]})

    def test_units_handedness_and_half_turn(self):
        xyz, q, matrix = trajectory.ue_pose([[-1,0,0,1],[0,-1,0,2],[0,0,1,3],[0,0,0,1]])
        self.assertEqual(xyz, [100,-200,300])
        self.assertEqual(q, [0,0,1,0])
        self.assertEqual(matrix, [[-1,0,0],[0,-1,0],[0,0,1]])

    def test_followup_window_keeps_earlier_camera_misses(self):
        camera = dict(matrix_world=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], lens_mm=50, sensor_width_mm=36)
        frames = [dict(objects={'A':[[1,0,0,x],[0,1,0,0],[0,0,1,-5],[0,0,0,1]]}) for x in (3, 0, 0)]
        trace = dict(actor_names=['A'], frames=frames)
        default = trajectory.camera_coverage(camera, trace, 16/9)
        self.assertFalse(default['required_window_all_centres_in_view'])
        result = trajectory.camera_coverage(camera, trace, 16/9, [1,3])
        self.assertTrue(result['required_window_all_centres_in_view'])
        self.assertFalse(result['first_frame_all_centres_in_view'])
        self.assertEqual(result['outside_frames_zero_based'], {'A':[0]})
        for window in ([0,4], [1,1], [-1,2], [True,2]):
            with self.subTest(window=window), self.assertRaises(AssertionError):
                trajectory.camera_coverage(camera, trace, 16/9, window)

    def test_rejects_scale_reflection_and_nonfinite(self):
        for x in (2, -1, float('nan')):
            with self.subTest(x=x), self.assertRaises(AssertionError):
                trajectory.ue_pose([[x,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    def test_two_actor_trace_rejects_time_and_identity_drift(self):
        matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        trace = dict(fps=24, actor_names=['A','B'], frames=[dict(frame=i+1, time=i/24, objects={'A':matrix,'B':matrix}) for i in range(96)])
        self.assertEqual(trajectory.check_trace(trace)['frames'], 96)
        bad = copy.deepcopy(trace)
        bad['frames'][48]['time'] += 1/24
        with self.assertRaises(AssertionError):
            trajectory.check_trace(bad)
        del trace['frames'][50]['objects']['B']
        with self.assertRaises(AssertionError):
            trajectory.check_trace(trace)


if __name__ == '__main__':
    unittest.main()
