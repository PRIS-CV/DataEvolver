"""Frozen rigid transforms: Blender metres/right-handed to UE centimetres."""
import math


def aimed_camera(camera, override):
    """Build a Blender camera matrix from a fixed location and look target."""
    location, target = override['location_m'], override['target_m']
    assert all(len(v) == 3 and all(math.isfinite(x) for x in v) for v in (location, target))
    direction = [b-a for a, b in zip(location, target)]
    length = math.sqrt(sum(x*x for x in direction))
    assert length > .001
    forward = [x/length for x in direction]
    right = [forward[1], -forward[0], 0]
    length = math.sqrt(sum(x*x for x in right))
    assert length > .001, 'Vertical views require an explicit camera up vector'
    right = [x/length for x in right]
    up = [right[(i+1)%3]*forward[(i+2)%3]-right[(i+2)%3]*forward[(i+1)%3] for i in range(3)]
    matrix = [[right[i], up[i], -forward[i], location[i]] for i in range(3)] + [[0,0,0,1]]
    ue_pose(matrix)
    result = dict(camera, matrix_world=matrix)
    if 'lens_mm' in override:
        assert math.isfinite(override['lens_mm']) and override['lens_mm'] > 0
        result['lens_mm'] = override['lens_mm']
    return result


def ue_pose(matrix):
    assert len(matrix) == 4 and all(len(row) == 4 for row in matrix)
    assert all(math.isfinite(v) for row in matrix for v in row)
    assert max(abs(matrix[3][i] - (i == 3)) for i in range(4)) < 1e-6
    signs = [1, -1, 1]
    r = [[matrix[i][j] * signs[i] * signs[j] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            assert abs(sum(r[k][i] * r[k][j] for k in range(3)) - (i == j)) < 1e-5, 'Apply source scale before freezing'
    determinant = sum(r[0][i] * (r[1][(i+1)%3] * r[2][(i+2)%3] - r[1][(i+2)%3] * r[2][(i+1)%3]) for i in range(3))
    assert abs(determinant - 1) < 1e-5, 'Expected a proper rigid rotation'
    trace = sum(r[i][i] for i in range(3))
    if trace > 0:
        s = 2 * math.sqrt(1 + trace)
        q = [(r[2][1]-r[1][2])/s, (r[0][2]-r[2][0])/s, (r[1][0]-r[0][1])/s, s/4]
    else:
        i = max(range(3), key=lambda k: r[k][k])
        j, k = (i+1) % 3, (i+2) % 3
        s = 2 * math.sqrt(1+r[i][i]-r[j][j]-r[k][k])
        q = [0.0, 0.0, 0.0, (r[k][j]-r[j][k])/s]
        q[i], q[j], q[k] = s/4, (r[j][i]+r[i][j])/s, (r[k][i]+r[i][k])/s
    norm = math.sqrt(sum(v*v for v in q))
    return [matrix[i][3]*100*signs[i] for i in range(3)], [v/norm for v in q], r


def check_trace(trace):
    assert trace['fps'] == 24 and trace['actor_names'] and trace['frames']
    assert len(set(trace['actor_names'])) == len(trace['actor_names'])
    for index, frame in enumerate(trace['frames']):
        assert frame['frame'] == index + 1 and abs(frame['time'] - index/24) < 1e-8
        assert set(frame['objects']) == set(trace['actor_names'])
        for matrix in frame['objects'].values():
            ue_pose(matrix)
    return {'frames': len(trace['frames']), 'actors': trace['actor_names']}


def camera_coverage(camera, trace, aspect, required_window=(0, 1)):
    """Project source actor centres; this cannot establish visible contact or occlusion."""
    matrix = camera['matrix_world']
    ue_pose(matrix)
    assert camera['lens_mm'] > 0 and camera['sensor_width_mm'] > 0 and aspect > 0
    assert camera.get('shift', [0, 0]) == [0, 0], 'Shifted cameras need a separate projection'
    assert len(required_window) == 2 and all(type(n) is int for n in required_window)
    start, end = required_window
    assert 0 <= start < end <= len(trace['frames']), 'Invalid required camera window'
    half_width = camera['sensor_width_mm'] / (2 * camera['lens_mm'])
    outside = {name: [] for name in trace['actor_names']}
    for index, frame in enumerate(trace['frames']):
        for name, pose in frame['objects'].items():
            delta = [pose[i][3] - matrix[i][3] for i in range(3)]
            x, y, back = [sum(matrix[i][j] * delta[i] for i in range(3)) for j in range(3)]
            if back >= 0 or abs(x) > -back * half_width or abs(y) > -back * half_width / aspect:
                outside[name].append(index)
    return dict(outside_frames_zero_based=outside, first_frame_all_centres_in_view=all(0 not in frames for frames in outside.values()),
                required_window_zero_based_end_exclusive=[start, end],
                required_window_all_centres_in_view=all(not any(start <= n < end for n in frames) for frames in outside.values()),
                scope='Source camera projection of actor centres only; no silhouette, occlusion or contact visibility proof')


def assess_runtime_poses(records, trace, errors):
    position_error = rotation_error = scale_error = 0.0
    for record in records:
        frame = record['output_frame']
        assert 0 <= frame < len(trace['frames']) and set(record['actors']) == set(trace['actor_names'])
        for name, actual in record['actors'].items():
            xyz, _, matrix = ue_pose(trace['frames'][frame]['objects'][name])
            values = actual['location_cm'] + actual['scale'] + [v for column in actual['rotation_columns'] for v in column]
            assert all(math.isfinite(v) for v in values)
            position_error = max(position_error, max(abs(actual['location_cm'][i] - xyz[i]) for i in range(3)))
            rotation_error = max(rotation_error, max(abs(actual['rotation_columns'][axis][i] - matrix[i][axis]) for axis in range(3) for i in range(3)))
            scale_error = max(scale_error, max(abs(s - 1) for s in actual['scale']))
    covered = sorted({r['output_frame'] for r in records})
    return dict(observed_frames=covered, records=len(records), errors=errors,
        max_translation_error_cm=position_error if records else None,
        max_rotation_matrix_error=rotation_error if records else None,
        max_scale_error=scale_error if records else None,
        full_frame_coverage=covered == list(range(len(trace['frames']))))
