"""Apply a declared rigid contract, then reuse the audited Blender snapshot path."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

from dataevolver.workflows.scene.blender.studio_video_blender import snapshot
from dataevolver.workflows.scene.blender.rigid_contract import mass_properties
from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json as write

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mode", choices=("nominal", "refined"), required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
request = json.loads(args.request.read_text())
record = request["inputs"]["simulation_spec"]
if request["stage"] != "blender" or file_sha256(Path(record["path"])) != record["sha256"]:
    raise ValueError("simulation contract changed")
spec = json.loads(Path(record["path"]).read_text())
run = args.output
run.mkdir(parents=True, exist_ok=True)
source = Path(bpy.data.filepath)
source_hash = file_sha256(source)
if request["inputs"]["scene"] != {"path": str(source.resolve()), "sha256": source_hash}:
    raise ValueError("Blender did not open the frozen scene input")
sc = bpy.context.scene
if any(any(mod.type in ("CLOTH", "SOFT_BODY", "FLUID") for mod in obj.modifiers) for obj in sc.objects):
    raise ValueError("The rigid adapter cannot transport deformation")
if abs(sc.unit_settings.scale_length-1) > 1e-8:
    raise ValueError("Author the assembly in metres")
expected = set(spec["instances"]) | set(spec["environment"]["supports"])
actual = {obj.name for obj in sc.objects if obj.rigid_body}
if actual != expected:
    raise ValueError("Rigid inventory differs: " + str(actual.symmetric_difference(expected)))
sc.frame_set(1)
sc.use_gravity = True
sc.gravity = spec["environment"]["gravity_m_s2"]
world = sc.rigidbody_world
if not world or abs(world.time_scale-1) > 1e-8:
    raise ValueError("Expected an unscaled rigid world")
factor = 1 if args.mode == "nominal" else spec["solver"]["refinement_factor"]
world.substeps_per_frame = spec["solver"]["substeps"] * factor
world.solver_iterations = spec["solver"]["iterations"]
substeps = world.substeps_per_frame
for name, item in {**spec["environment"]["supports"], **spec["instances"]}.items():
    obj = sc.objects[name]
    if obj.parent or obj.constraints or obj.animation_data:
        raise ValueError("Rigid contract objects must have no authored controllers or parents: " + name)
    if any(abs(x-1) > 1e-6 for x in obj.scale):
        raise ValueError("Apply assembly object scale before simulation: " + name)
    if max(abs(x-y) for x, y in zip(obj.dimensions, item["dimensions_m"])) > 1e-4:
        raise ValueError("Assembly dimensions differ from the contract: " + name)
    bounds_center = sum((Vector(p) for p in obj.bound_box), Vector()) / 8
    if bounds_center.length > 1e-5:
        raise ValueError("Center the declared ideal proxy origin: " + name)
    obj.location = item["position_m"]
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = item.get("rotation_rad", [0, 0, 0])
    rigid = obj.rigid_body
    rigid.friction, rigid.restitution = item["friction"], item.get("restitution", 0)
    rigid.use_margin = True
    rigid.collision_margin = 0
    if name in spec["instances"]:
        if any(abs(v) > 1e-10 for v in item.get("velocity_m_s", [0, 0, 0])):
            raise ValueError("Nonzero initial velocity needs a qualified impulse adapter")
        if rigid.type != "ACTIVE" or rigid.kinematic or rigid.collision_shape != item["collider"]["shape"]:
            raise ValueError("Actor proxy does not match the contract")
        rigid.mass = mass_properties(item)["mass_kg"]
        rigid.linear_damping = item.get("linear_damping", 0)
        rigid.angular_damping = item.get("angular_damping", 0)
        rigid.use_deactivation = False
    elif rigid.type != "PASSIVE" or rigid.collision_shape != "BOX":
        raise ValueError("Native supports require passive box proxies")
fps, frames = spec["timeline"]["fps"], spec["timeline"]["frames"]
snapshot({"fps": fps, "frames": frames, "duration_s": (frames-1)/fps,
          "views": 1, "cameras": []}, run)
write(run / "source-check.json", {"source_sha256": source_hash,
                                  "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
                                  "substeps": substeps})

write(run / "result.json", {"status": "completed", "request_sha256": payload_sha256(request)})
