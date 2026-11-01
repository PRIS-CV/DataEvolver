"""Validate poses at the deferred pass boundary, including every spatial sample."""
import json
import math
from pathlib import Path
if __package__:
    from .trajectory import assess_runtime_poses
else:
    from trajectory import assess_runtime_poses


def assess(path, trace, frames, samples):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    expected = {(f,s,0) for f in range(frames) for s in range(samples)}
    observed = [(r['output_frame'],r['spatial_sample'],r['temporal_sample']) for r in records]
    assert len(observed) == len(expected) and set(observed) == expected, 'Missing or duplicate render samples'
    assert all(r['source_frame']==r['output_frame'] and r['spatial_count']==samples
               and r['temporal_count']==1 for r in records), 'Native frame or sample timing changed'
    assert all(math.isfinite(v) for r in records for v in r['view_location_cm']+r['view_rotation_degrees'])
    subset = dict(trace, frames=trace['frames'][:frames])
    result = assess_runtime_poses(records, subset, [])
    components = [dict(output_frame=r['output_frame'],actors={name:component for name,pose in r['actors'].items()
        for component in pose['components']}) for r in records]
    assert all(len(pose['components'])==1 for r in records for pose in r['actors'].values()), 'Expected one rigid mesh component'
    component_result = assess_runtime_poses(components, subset, [])
    for item in [result,component_result]:
        assert item['full_frame_coverage'] and item['max_translation_error_cm']<=.001, item
        assert item['max_rotation_matrix_error']<=.00001 and item['max_scale_error']<=.000001, item
    return dict(status='passed',frames=frames,samples_per_frame=samples,records=len(records),
        actor=result,component=component_result,translation_tolerance_cm=.001,
        rotation_matrix_tolerance=.00001,
        scope='Native deferred RenderSample_GameThreadImpl immediately before the same Super render call; actor and component transforms',
        pixel_visibility='Requires separate native image, silhouette and occlusion checks')
