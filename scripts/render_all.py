import bpy
import os
import sys
import math
from mathutils import Vector

### === CLI 参数解析 === ###
# 在 blender 调用中，"--" 后面的参数传给脚本
argv = sys.argv
try:
    sep = argv.index("--")
    script_args = argv[sep + 1:]
except ValueError:
    script_args = []

import argparse
parser = argparse.ArgumentParser(description="Blender 参数化渲染脚本")
parser.add_argument("--model-id", type=int, required=True, help="模型编号（对应 data/models/{id}.blend）")
parser.add_argument("--scene-id", type=int, required=True, help="场景编号（对应 data/sence/{id}.blend）")
parser.add_argument("--gpu-index", type=int, default=0, help="使用的 GPU 索引")
parser.add_argument("--num-frames", type=int, default=5, help="每方向帧数（default 5，完整渲染用 30）")
parser.add_argument("--num-cameras-per-ring", type=int, default=3, help="每圈相机数（default 3，完整渲染用 10）")
args = parser.parse_args(script_args)

### === 路径配置 === ###
BASE_DIR = "<blender-workspace>"
object_blend_path = os.path.join(BASE_DIR, "data/models", f"{args.model_id}.blend")
scene_blend_path  = os.path.join(BASE_DIR, "data/sence",  f"{args.scene_id}.blend")

target_gpu_index    = args.gpu_index
num_frames          = args.num_frames
num_cameras_per_ring = args.num_cameras_per_ring
move_step           = [0.05, 0.05, 0]

# 输出目录：output/{scene}_{model}/
output_subdir = f"{args.scene_id}_{args.model_id}"
output_path   = os.path.join(BASE_DIR, "output", output_subdir)
output_mask_dir = os.path.join(output_path, "mask")
os.makedirs(output_path,     exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

print(f"=== render_all.py ===")
print(f"model : {object_blend_path}")
print(f"scene : {scene_blend_path}")
print(f"gpu   : {target_gpu_index}")
print(f"frames: {num_frames}  cameras/ring: {num_cameras_per_ring}")
print(f"output: {output_path}")

### === 工具函数 === ###
def direction_to_euler(direction_vec):
    target = Vector(direction_vec)
    rot_quat = target.to_track_quat('-Z', 'Y')
    return rot_quat.to_euler()

def get_scene_bounds():
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.hide_get():
            bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            for v in bbox_world:
                min_x = min(min_x, v.x); min_y = min(min_y, v.y); min_z = min(min_z, v.z)
                max_x = max(max_x, v.x); max_y = max(max_y, v.y); max_z = max(max_z, v.z)
    return Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))

def auto_adjust_object_size(obj, bounds_min, bounds_max):
    size = bounds_max - bounds_min
    avg_dim = (size.x + size.y + size.z) / 3
    target_size = avg_dim * 0.1
    current_size = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
    if current_size == 0:
        print("❌ 物体尺寸为0，跳过缩放")
        return
    scale_factor = target_size / current_size
    obj.scale *= scale_factor
    print(f"📏 物体缩放因子: {scale_factor:.4f}")

def adjust_lighting_to_target_brightness(target_brightness=0.4, preview_resolution=(128, 72), camera_list=None):
    """调整光照亮度，PIL 不可用时 graceful fallback"""
    scene = bpy.context.scene
    if camera_list:
        scene.camera = camera_list[0]

    try:
        from PIL import Image
        import numpy as np
        import tempfile

        old_x, old_y = scene.render.resolution_x, scene.render.resolution_y
        old_samples  = scene.cycles.samples
        scene.render.resolution_x, scene.render.resolution_y = preview_resolution
        scene.cycles.samples = 32
        scene.render.image_settings.file_format = 'PNG'
        temp_file = os.path.join(tempfile.gettempdir(), "preview_brightness.png")
        scene.render.filepath = temp_file
        bpy.ops.render.render(write_still=True)

        img = Image.open(temp_file).convert('RGB')
        avg_brightness = np.asarray(img).astype(float).mean() / 255.0
        if avg_brightness > 0:
            scale = target_brightness / avg_brightness
            for light in bpy.data.lights:
                light.energy *= scale
            if scene.world and scene.world.node_tree:
                for node in scene.world.node_tree.nodes:
                    if node.type == 'BACKGROUND':
                        node.inputs[1].default_value *= scale

        scene.render.resolution_x, scene.render.resolution_y = old_x, old_y
        scene.cycles.samples = old_samples
        print(f"💡 亮度调整完成 (avg={avg_brightness:.3f} → 目标={target_brightness})")

    except Exception as e:
        print(f"⚠️  亮度自动调整跳过（{e}），应用固定 0.3× 衰减")
        for light in bpy.data.lights:
            light.energy *= 0.3
        if scene.world and scene.world.node_tree:
            for node in scene.world.node_tree.nodes:
                if node.type == 'BACKGROUND':
                    node.inputs[1].default_value *= 0.3

def find_ground_object_auto():
    candidates = [obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.hide_get()]
    for obj in candidates:
        if obj.name.lower() == "ground":
            return obj
    lowest = float('inf')
    ground_obj = None
    for obj in candidates:
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        z = min(v.z for v in bbox)
        if z < lowest:
            lowest = z
            ground_obj = obj
    return ground_obj

def fix_ground_orientation(ground_obj):
    if ground_obj.scale.z < 0:
        ground_obj.scale.z = abs(ground_obj.scale.z)
        print("⚠️ 修复地面负缩放")
    rx = math.degrees(ground_obj.rotation_euler.x)
    if abs(rx) in [90, 180, 270]:
        ground_obj.rotation_euler.x = 0.0
        print("⚠️ 修复地面旋转角度")

def adjust_object_to_ground_with_ray(obj, ground_obj):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z    = min(v.z for v in bbox_world)
    center_x = sum(v.x for v in bbox_world) / 8.0
    center_y = sum(v.y for v in bbox_world) / 8.0
    ray_origin = Vector((center_x, center_y, min_z + 0.01))
    ray_dir    = Vector((0, 0, -1))
    hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, ray_origin, ray_dir, distance=2.0)
    if hit and hit_obj == ground_obj:
        delta_z = location.z - min_z
        obj.location.z += delta_z
        print(f"✅ 精确贴合地面，提升: {delta_z:.4f}")
    else:
        print("⚠️ 未命中地面，跳过贴地")

def get_all_meshes(obj):
    meshes = []
    if obj.type == 'MESH':
        meshes.append(obj)
    for o in bpy.context.collection.objects:
        if o.parent == obj:
            meshes.extend(get_all_meshes(o))
    return meshes

### === mask 渲染函数 === ###
def render_mask_for_main(scene, main_meshes, cam, mask_path, res_x, res_y):
    # 记录原始设置
    orig_lights = {light: light.energy for light in bpy.data.lights}
    world = scene.world
    orig_world_color = None
    orig_world_strength = None
    if world and world.node_tree:
        for n in world.node_tree.nodes:
            if n.type == 'BACKGROUND':
                try:
                    orig_world_color    = tuple(n.inputs[0].default_value)
                    orig_world_strength = n.inputs[1].default_value
                except Exception:
                    pass
    orig_view_transform = scene.view_settings.view_transform
    orig_look           = scene.view_settings.look
    orig_gamma          = scene.view_settings.gamma

    # 隐藏所有非主物体 mesh
    hidden_objs = []
    for o in bpy.data.objects:
        if o not in main_meshes and o.type == 'MESH' and not o.hide_render:
            o.hide_render = True
            hidden_objs.append(o)

    # 主物体全白 Emission
    orig_materials = []
    mask_mat = bpy.data.materials.new(name="__mask_mat__")
    mask_mat.use_nodes = True
    nodes = mask_mat.node_tree.nodes
    for n in list(nodes):
        nodes.remove(n)
    out_node  = nodes.new(type='ShaderNodeOutputMaterial')
    emit_node = nodes.new(type='ShaderNodeEmission')
    emit_node.inputs[0].default_value = (1, 1, 1, 1)
    emit_node.inputs[1].default_value = 1.0
    mask_mat.node_tree.links.new(emit_node.outputs[0], out_node.inputs[0])

    for mesh in main_meshes:
        if len(mesh.material_slots) == 0:
            mesh.data.materials.append(mask_mat)
        orig_materials.append([slot.material for slot in mesh.material_slots])
        for i in range(len(mesh.material_slots)):
            mesh.material_slots[i].material = mask_mat

    # 关灯 + 黑色背景 + Standard 色彩管理
    for light in bpy.data.lights:
        light.energy = 0.0
    if world and world.node_tree:
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                node.inputs[0].default_value = (0, 0, 0, 1)
                node.inputs[1].default_value = 1.0
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look           = 'None'
    scene.view_settings.gamma          = 1.0

    # 渲染 mask
    scene.camera = cam
    scene.render.filepath = mask_path
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'BW'
    bpy.ops.render.render(write_still=True)

    # === 恢复 ===
    for mesh, mats in zip(main_meshes, orig_materials):
        for i, mat in enumerate(mats):
            mesh.material_slots[i].material = mat
    bpy.data.materials.remove(mask_mat)

    for o in hidden_objs:
        o.hide_render = False
    for light, energy in orig_lights.items():
        light.energy = energy
    if world and world.node_tree and orig_world_color is not None:
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                node.inputs[0].default_value = orig_world_color
                node.inputs[1].default_value = orig_world_strength
    scene.view_settings.view_transform = orig_view_transform
    scene.view_settings.look           = orig_look
    scene.view_settings.gamma          = orig_gamma

### === 创建多圈相机 === ###
def create_cameras_rings(bounds_min, bounds_max, target_obj, num_cameras_per_ring):
    """三圈相机，每圈 num_cameras_per_ring 个"""
    camera_rings_config = [
        (0.01, 0.005, num_cameras_per_ring),
        (0.02, 0.01,  num_cameras_per_ring),
        (0.03, 0.015, num_cameras_per_ring),
    ]
    bbox_world = [target_obj.matrix_world @ Vector(c) for c in target_obj.bound_box]
    center     = sum(bbox_world, Vector((0, 0, 0))) / 8.0
    scene_size = bounds_max - bounds_min
    max_span   = max(scene_size.x, scene_size.y)
    scene_height = scene_size.z

    camera_list = []
    for ring_idx, (radius_ratio, height_ratio, cam_count) in enumerate(camera_rings_config):
        radius = max_span    * radius_ratio
        height = scene_height * height_ratio
        print(f"🎥 创建第{ring_idx+1}圈相机: radius={radius:.3f} height={height:.3f} count={cam_count}")
        for i in range(cam_count):
            angle   = 2 * math.pi * i / cam_count
            cam_loc = Vector((center.x + radius * math.cos(angle),
                              center.y + radius * math.sin(angle),
                              center.z + height))
            cam_dir  = center - cam_loc
            cam_data = bpy.data.cameras.new(name=f"Camera_{ring_idx}_{i}")
            cam_obj  = bpy.data.objects.new(name=f"Camera_{ring_idx}_{i}", object_data=cam_data)
            cam_obj.location       = cam_loc
            cam_obj.rotation_euler = direction_to_euler(cam_dir)
            cam_data.lens         = 50
            cam_data.sensor_width = 36
            bpy.context.collection.objects.link(cam_obj)
            camera_list.append(cam_obj)
    return camera_list

### === GPU 全局偏好设置（在 open_mainfile 前设，持久生效）=== ###
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
cuda_devices = [d for d in prefs.devices if d.type == 'CUDA']
if target_gpu_index >= len(cuda_devices):
    raise IndexError(f"GPU index {target_gpu_index} 超出范围（共 {len(cuda_devices)} 个 CUDA 设备）")
for i, d in enumerate(cuda_devices):
    d.use = (i == target_gpu_index)
    print(f"{'✅' if d.use else '🚫'} GPU {i}: {d.name}")

### === 加载场景 === ###
bpy.ops.wm.open_mainfile(filepath=scene_blend_path)
scene = bpy.context.scene

# 在 open_mainfile 之后重新应用所有 scene 级设置
scene.render.engine = 'CYCLES'
scene.render.resolution_x           = 1024
scene.render.resolution_y           = 1024
scene.render.resolution_percentage  = 100

scene.cycles.device                 = 'GPU'
scene.cycles.use_adaptive_sampling  = True
scene.cycles.adaptive_threshold     = 0.01
scene.cycles.samples                = 512
scene.cycles.use_denoising          = True
try:
    scene.cycles.denoiser           = 'OPENIMAGEDENOISE'
except TypeError:
    try:
        scene.cycles.denoiser       = 'NLM'
    except TypeError:
        pass   # 旧版 Blender 无此属性，忽略
scene.cycles.tile_size              = 512

scene.view_settings.view_transform  = 'Filmic'
try:
    scene.view_settings.look        = 'None'
except TypeError:
    pass
scene.view_settings.gamma           = 0.5

for light in bpy.data.lights:
    light.energy *= 0.3
if scene.world and scene.world.node_tree:
    for node in scene.world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            node.inputs[1].default_value *= 0.3

### === 导入物体 === ###
with bpy.data.libraries.load(object_blend_path) as (data_from, data_to):
    data_to.objects = data_from.objects

imported_objs = []
for o in data_to.objects:
    if o:
        bpy.context.collection.objects.link(o)
        imported_objs.append(o)
        print(f"导入: {o.name} ({o.type})")

main_obj = next((o for o in imported_objs if not o.hide_get()), None)
if main_obj is None:
    raise RuntimeError("未找到主物体（所有导入对象均隐藏）")
print(f"主物体: {main_obj.name}")

main_meshes = get_all_meshes(main_obj)
if not main_meshes:
    raise RuntimeError(f"主物体 {main_obj.name} 没有任何 mesh")

### === 场景准备 === ###
ground_obj = find_ground_object_auto()
if ground_obj:
    fix_ground_orientation(ground_obj)
bounds_min, bounds_max = get_scene_bounds()
auto_adjust_object_size(main_obj, bounds_min, bounds_max)
if ground_obj:
    adjust_object_to_ground_with_ray(main_obj, ground_obj)
start_location = list(main_obj.location)

camera_list = create_cameras_rings(bounds_min, bounds_max, main_obj, num_cameras_per_ring)

adjust_lighting_to_target_brightness(0.4, camera_list=camera_list)

# 文件名前缀
scene_name  = str(args.scene_id)
object_name = str(args.model_id)

### === 收集渲染任务 === ###
render_tasks = []
# forward: frame 0..num_frames-1
for frame in range(num_frames):
    pos = (
        start_location[0] + move_step[0] * frame,
        start_location[1] + move_step[1] * frame,
        start_location[2] + move_step[2] * frame,
    )
    for cam in camera_list:
        img_name = f"{scene_name}_{object_name}_{cam.name}_forward_{frame+1:02d}.png"
        render_tasks.append((pos, cam, img_name))
# reverse
for frame in range(num_frames):
    pos = (
        start_location[0] - move_step[0] * frame,
        start_location[1] - move_step[1] * frame,
        start_location[2] - move_step[2] * frame,
    )
    for cam in camera_list:
        img_name = f"{scene_name}_{object_name}_{cam.name}_reverse_{frame+1:02d}.png"
        render_tasks.append((pos, cam, img_name))

total = len(render_tasks)
print(f"\n📋 总任务数: {total} 主图 + {total} mask = {total*2} 张")

### === 主图渲染（skip existing） === ###
print("\n=== 开始主图渲染 ===")
skipped_main = 0
rendered_main = 0
for idx, (pos, cam, img_name) in enumerate(render_tasks, 1):
    out_file = os.path.join(output_path, img_name)
    if os.path.exists(out_file):
        print(f"[{idx}/{total}] ⏭️  跳过（已存在）: {img_name}")
        skipped_main += 1
        continue
    main_obj.location = pos
    scene.camera      = cam
    scene.render.filepath = out_file
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGBA'
    print(f"[{idx}/{total}] 🎬 主图: {img_name}")
    bpy.ops.render.render(write_still=True)
    rendered_main += 1

print(f"主图完成: 渲染 {rendered_main}，跳过 {skipped_main}")

### === Mask 渲染（skip existing） === ###
print("\n=== 开始 mask 渲染 ===")
skipped_mask  = 0
rendered_mask = 0
for idx, (pos, cam, img_name) in enumerate(render_tasks, 1):
    mask_file = os.path.join(output_mask_dir, img_name)
    if os.path.exists(mask_file):
        print(f"[{idx}/{total}] ⏭️  跳过 mask（已存在）: {img_name}")
        skipped_mask += 1
        continue
    main_obj.location = pos
    print(f"[{idx}/{total}] 🎭 mask: {img_name}")
    render_mask_for_main(scene, main_meshes, cam, mask_file,
                         scene.render.resolution_x, scene.render.resolution_y)
    rendered_mask += 1

print(f"mask 完成: 渲染 {rendered_mask}，跳过 {skipped_mask}")
print("\n✅ 全部渲染完成！")
