"""Build a component scene from an Agent-authored JSON recipe in background Blender.

shortcut: this recipe covers primitive components; bind a reviewed modeling script
through scene.blender for sculpting, custom topology or asset-specific modeling.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def main():
    from dataevolver.dataset.contracts import file_sha256, payload_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    request = json.loads(args.request.read_text())
    source = request["inputs"]["scene_spec"]
    if request["stage"] != "blender" or file_sha256(Path(source["path"])) != source["sha256"]:
        raise ValueError("modeling request or recipe changed")
    recipe = json.loads(Path(source["path"]).read_text())
    objects = recipe["objects"]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 128:
        raise ValueError("modeling recipe requires 1..128 components")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    builders = {"cube": bpy.ops.mesh.primitive_cube_add,
                "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
                "cylinder": bpy.ops.mesh.primitive_cylinder_add,
                "cone": bpy.ops.mesh.primitive_cone_add,
                "torus": bpy.ops.mesh.primitive_torus_add}
    names = set()
    for item in objects:
        if not isinstance(item["name"], str) or not item["name"] or item["name"] in names:
            raise ValueError("component names must be nonempty and unique")
        names.add(item["name"])
        vectors = {name: item.get(name, default) for name, default in
                   (("location", [0, 0, 0]), ("rotation", [0, 0, 0]), ("scale", [1, 1, 1]))}
        for name, vector in vectors.items():
            if len(vector) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                raise ValueError("component transforms require three finite numbers")
            if name == "scale" and any(v <= 0 for v in vector):
                raise ValueError("component scale must be positive")
        builders[item["shape"]](location=vectors["location"], rotation=vectors["rotation"], scale=vectors["scale"])
        obj = bpy.context.object
        obj.name = item["name"]
        material = bpy.data.materials.new("Material_" + item["name"])
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = item.get("color", [0.6, 0.6, 0.6, 1.0])
        shader.inputs["Roughness"].default_value = item.get("roughness", 0.5)
        shader.inputs["Metallic"].default_value = item.get("metallic", 0.0)
        obj.data.materials.append(material)
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / request["outputs"]["scene"]))
    bpy.ops.export_scene.gltf(filepath=str(args.output / request["outputs"]["mesh"]), export_format="GLB")
    (args.output / "result.json").write_text(json.dumps({"status": "completed",
        "request_sha256": payload_sha256(request), "objects": len(objects), "blender_version": bpy.app.version_string}))


if __name__ == "__main__":
    main()
