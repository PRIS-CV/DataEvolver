"""Export a frozen Blender scene and original trace; never save over the source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--request', type=Path)
parser.add_argument('--output', type=Path)
parser.add_argument('--out', type=Path)
parser.add_argument('--trace', type=Path)
parser.add_argument('--simulation', type=Path)
parser.add_argument('--expected-source-sha256')
parser.add_argument('--expected-trace-sha256')
parser.add_argument('--family')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


request = json.loads(args.request.read_text()) if args.request else None
if request is not None:
    from dataevolver.dataset.contracts import payload_sha256
    from dataevolver.harness.state_store import atomic_write_json
    if request['stage'] != 'blender' or args.output is None:
        raise ValueError('scene export requires a Blender request and native output directory')
    inputs = request['inputs']
    for value in inputs.values():
        if digest(value['path']) != value['sha256']:
            raise ValueError('scene export input changed')
    if Path(bpy.data.filepath).resolve() != Path(inputs['scene']['path']).resolve():
        raise ValueError('Blender did not open the declared frozen source')
    args.out, args.trace, args.simulation = args.output / 'package', Path(inputs['trace']['path']), Path(inputs['simulation']['path'])
    args.expected_source_sha256, args.expected_trace_sha256 = inputs['scene']['sha256'], inputs['trace']['sha256']
    args.family = json.loads(Path(inputs['scene_spec']['path']).read_text())['family']
if any(getattr(args, name) is None for name in ('out', 'trace', 'simulation', 'expected_source_sha256', 'expected_trace_sha256', 'family')):
    raise ValueError('provide a native request or complete explicit frozen-scene export arguments')


source = Path(bpy.data.filepath)
assert digest(source) == args.expected_source_sha256
assert digest(args.trace) == args.expected_trace_sha256
trace = json.loads(args.trace.read_text())
scene = bpy.context.scene
assert scene.rigidbody_world is None, 'Export the frozen render source, not the simulation source'
assert trace['fps'] == scene.render.fps / scene.render.fps_base
assert trace['frames'][0]['frame'] == 1
scene.frame_set(1)
bpy.context.view_layer.update()
args.out.mkdir(parents=True, exist_ok=False)
objects = []
for obj in scene.objects:
    item = dict(name=obj.name, type=obj.type, matrix_world=[list(r) for r in obj.matrix_world],
                hide_render=obj.hide_render, parent=obj.parent.name if obj.parent else None)
    if obj.type in ('MESH', 'CURVE'):
        item.update(materials=[m.name if m else None for m in obj.data.materials], dimensions_m=list(obj.dimensions))
    if obj.type == 'CAMERA':
        item.update(lens_mm=obj.data.lens, sensor_width_mm=obj.data.sensor_width,
                    sensor_height_mm=obj.data.sensor_height, shift=[obj.data.shift_x, obj.data.shift_y])
    if obj.type == 'LIGHT':
        item.update(light_type=obj.data.type, energy=obj.data.energy)
    objects.append(item)
    if obj.hide_render:
        obj.hide_set(True)

curves = [o for o in scene.objects if o.type == 'CURVE' and not o.hide_render]
graph = bpy.context.evaluated_depsgraph_get()
for obj in curves:
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(graph), preserve_all_data_layers=True, depsgraph=graph)
    replacement = bpy.data.objects.new(obj.name + '_UECurve', mesh)
    scene.collection.objects.link(replacement)
    replacement.matrix_world = obj.matrix_world.copy()
    obj.hide_set(True)
    obj.hide_render = True

glb = args.out / 'scene.glb'
result = bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', use_visible=True,
    export_apply=True, export_animations=False, export_cameras=True, export_lights=True, export_extras=False)
assert result == {'FINISHED'}
data = glb.read_bytes()
assert struct.unpack('<4sII', data[:12]) == (b'glTF', 2, len(data))
length = struct.unpack_from('<I', data, 12)[0]
gltf = json.loads(data[20:20 + length])
actual = {n['name'] for n in gltf['nodes'] if 'mesh' in n}
expected = {o['name'] for o in objects if o['type'] == 'MESH' and not o['hide_render']}
expected.update(o.name + '_UECurve' for o in curves)
assert actual == expected, {'missing': sorted(expected - actual), 'extra': sorted(actual - expected)}
assert set(trace['actor_names']).issubset({n['name'] for n in gltf['nodes']})
for name in ('trace', 'simulation'):
    shutil.copyfile(getattr(args, name), args.out / (name + '.json'))
manifest = dict(scene_family=args.family, source=str(source), source_sha256=args.expected_source_sha256,
    trace_sha256=args.expected_trace_sha256, simulation_sha256=digest(args.simulation),
    blender=bpy.app.version_string, source_units='meters', source_frame=1, objects=objects,
    fps=trace['fps'], frames=len(trace['frames']), actor_names=trace['actor_names'],
    exported_mesh_objects=len(actual), packed_images=sum(bool(i.packed_file) for i in bpy.data.images),
    export=dict(path='scene.glb', bytes=len(data), sha256=digest(glb)),
    material_limit='Procedural nodes unsupported by glTF need baking or UE material reconstruction; not a photorealism acceptance.',
    motion='Original frozen poses; no physics recomputation.', physical_truth='unknown', benchmark_validated=False)
(args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
assert digest(source) == args.expected_source_sha256
assert digest(args.trace) == args.expected_trace_sha256
print(json.dumps({k: v for k, v in manifest.items() if k != 'objects'}))
if request is not None:
    atomic_write_json(args.output / 'result.json', {'status': 'completed', 'request_sha256': payload_sha256(request)})
