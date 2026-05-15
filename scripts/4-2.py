import bpy
import os
import math
from mathutils import Vector
from PIL import Image
import tempfile
import numpy as np

### === 参数配置 === ###
target_gpu_index = 1
object_blend_path = "<object-blend-path>"
scene_blend_path = "<scene-blend-path>"
output_path = "<output-root>"
os.makedirs(output_path, exist_ok=True)

num_frames = 30
move_step = [0.05, 0.05, 0]

### === 工具函数 === ###
def direction_to_euler(direction_vec):
    target = Vector(direction_vec)
    rot_quat = target.to_track_quat('-Z', 'Y')
    return rot_quat.to_euler()

# ===== 1.py功能集成 =====
def get_scene_bounds():
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.hide_get():
            bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            for v in bbox_world:
                min_x = min(min_x, v.x)
                min_y = min(min_y, v.y)
                min_z = min(min_z, v.z)
                max_x = max(max_x, v.x)
                max_y = max(max_y, v.y)
                max_z = max(max_z, v.z)
    return Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))

def auto_adjust_camera_positions(camera_list, scene_bounds_min, scene_bounds_max, target_obj):
    bbox_world = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
    center = sum(bbox_world, Vector((0, 0, 0))) / 8.0
    scene_size = scene_bounds_max - scene_bounds_min
    max_span = max(scene_size.x, scene_size.y)
    scene_height = scene_size.z
    radius = max_span * 0.02  # 0.5倍场景跨度
    height = scene_height * 0.01  # 0.3倍场景高度
    count = len(camera_list)
    for i, cam in enumerate(camera_list):
        angle = 2 * math.pi * i / count
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        z = center.z + height
        cam_loc = Vector((x, y, z))
        cam_dir = center - cam_loc
        cam.location = cam_loc
        cam.rotation_euler = direction_to_euler(cam_dir)
        cam.data.lens = 50
        cam.data.sensor_width = 36

def auto_adjust_object_size(obj, scene_bounds_min, scene_bounds_max):
    size = scene_bounds_max - scene_bounds_min
    avg_dim = (size.x + size.y + size.z) / 3
    target_size = avg_dim * 0.1
    current_size = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
    if current_size == 0:
        print("❌ 物体尺寸为0，无法缩放。")
        return
    scale_factor = target_size / current_size
    obj.scale *= scale_factor
    print(f"📏 物体缩放因子: {scale_factor:.4f} (目标尺寸: {target_size:.2f}, 当前尺寸: {current_size:.2f})")

def adjust_lighting_to_target_brightness(target_brightness=0.4, preview_resolution=(128, 72), camera_list=None):
    scene = bpy.context.scene
    if camera_list:
        scene.camera = camera_list[0]
    old_x, old_y = scene.render.resolution_x, scene.render.resolution_y
    old_samples = scene.cycles.samples
    scene.render.resolution_x, scene.render.resolution_y = preview_resolution
    scene.cycles.samples = 32
    scene.render.image_settings.file_format = 'PNG'
    temp_file = os.path.join(tempfile.gettempdir(), "preview.png")
    scene.render.filepath = temp_file
    bpy.ops.render.render(write_still=True)
    img = Image.open(temp_file).convert('RGB')
    avg_brightness = np.asarray(img).astype(np.float32).mean() / 255.0
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
    min_z = min(v.z for v in bbox_world)
    center_x = sum(v.x for v in bbox_world) / 8.0
    center_y = sum(v.y for v in bbox_world) / 8.0
    ray_origin = Vector((center_x, center_y, min_z + 0.01))
    ray_dir = Vector((0, 0, -1))
    hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, ray_origin, ray_dir, distance=2.0)
    if hit and hit_obj == ground_obj:
        delta_z = location.z - min_z
        obj.location.z += delta_z
        print(f"✅ 精确贴合地面，提升高度: {delta_z:.4f}")
    else:
        print("⚠️ 未命中地面")
# ===== 1.py功能集成结束 =====

### === 渲染设置 === ###
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# GPU 设置
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()

cuda_devices = [d for d in prefs.devices if d.type == 'CUDA']
if target_gpu_index >= len(cuda_devices):
    raise IndexError("指定 GPU index 超出范围")
for i, d in enumerate(cuda_devices):
    d.use = (i == target_gpu_index)
    print(f"{'✅ 启用' if d.use else '🚫 禁用'} GPU: {d.name}")

# 高显存使用配置
scene.cycles.device = 'GPU'
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.01
scene.cycles.samples = 512  # 提高采样数
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.cycles.use_bvh_spatial_split = True
scene.cycles.debug_use_spatial_splits = False
scene.cycles.debug_use_qbvh = False
scene.cycles.use_progressive_refine = False
scene.cycles.tile_size = 512

# 曝光和色彩管理，防止过曝
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'None'
# scene.view_settings.exposure = 1
scene.view_settings.gamma = 0.5

### === 加载场景文件 === ###
bpy.ops.wm.open_mainfile(filepath=scene_blend_path)
scene = bpy.context.scene

# 在这里设置分辨率，确保覆盖场景文件的设置
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100

# 降低所有灯光强度
for light in bpy.data.lights:
    light.energy *= 0.3

# 降低环境光（如果使用了背景贴图）
if scene.world and scene.world.node_tree:
    for node in scene.world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            node.inputs[1].default_value *= 0.3


### === 导入物体 === ###
with bpy.data.libraries.load(object_blend_path) as (data_from, data_to):
    data_to.objects = data_from.objects

# 导入后，自动选择第一个非隐藏的对象为主物体
imported_objs = []
for o in data_to.objects:
    if o:
        bpy.context.collection.objects.link(o)
        imported_objs.append(o)
        print(f"导入对象: {o.name}, 类型: {o.type}")

# 选择第一个非隐藏的对象为主物体
main_obj = None
for o in imported_objs:
    if not o.hide_get():
        main_obj = o
        break
if main_obj is None:
    raise RuntimeError("未找到主物体")

print(f"选用主物体: {main_obj.name}, 类型: {main_obj.type}")

# 获取主物体的所有mesh（自身或递归子mesh）
def get_all_meshes(obj):
    meshes = []
    if obj.type == 'MESH':
        meshes.append(obj)
    # 递归查找子对象
    for o in bpy.context.collection.objects:
        if o.parent == obj:
            meshes.extend(get_all_meshes(o))
    return meshes

main_meshes = get_all_meshes(main_obj)
if not main_meshes:
    raise RuntimeError(f"主物体 {main_obj.name} 没有任何mesh，无法渲染mask")
print("主物体包含的mesh:", [m.name for m in main_meshes])

# === 1.py功能：地面检测、修正、物体缩放、贴地 === #
ground_obj = find_ground_object_auto()
if ground_obj:
    fix_ground_orientation(ground_obj)
bounds_min, bounds_max = get_scene_bounds()
auto_adjust_object_size(main_obj, bounds_min, bounds_max)
if ground_obj:
    adjust_object_to_ground_with_ray(main_obj, ground_obj)
start_location = list(main_obj.location)

# ====== 三圈相机参数 ======
# 相机圈配置：每圈包含 (半径比例, 高度比例, 相机数量)
camera_rings_config = [
    (0.01, 0.005, 10),   # 圈1：半径比例0.1，高度比例0.05，10个相机
    (0.02, 0.01, 10),    # 圈2：半径比例0.2，高度比例0.1，10个相机
    (0.03, 0.015, 10),   # 圈3：半径比例0.3，高度比例0.15，10个相机
]

output_mask_dir = os.path.join(output_path, 'mask')
os.makedirs(output_mask_dir, exist_ok=True)

def create_cameras_rings(camera_list, bounds_min, bounds_max, target_obj, rings_config):
    """
    创建多圈相机

    Args:
        camera_list: 相机列表，用于存储创建的相机
        bounds_min, bounds_max: 场景边界
        target_obj: 目标物体
        rings_config: 相机圈配置列表，每个元素为 (半径比例, 高度比例, 相机数量)
    """
    bbox_world = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
    center = sum(bbox_world, Vector((0, 0, 0))) / 8.0
    scene_size = bounds_max - bounds_min
    max_span = max(scene_size.x, scene_size.y)
    scene_height = scene_size.z

    for ring_idx, (radius_ratio, height_ratio, cameras_count) in enumerate(rings_config):
        radius = max_span * radius_ratio
        height = scene_height * height_ratio

        print(f"🎥 创建第{ring_idx + 1}圈相机: 半径={radius:.2f}, 高度={height:.2f}, 数量={cameras_count}")

        for i in range(cameras_count):
            angle = 2 * math.pi * i / cameras_count
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            z = center.z + height
            cam_loc = Vector((x, y, z))
            cam_dir = center - cam_loc

            cam_data = bpy.data.cameras.new(name=f"Camera_{ring_idx}_{i}")
            cam_obj = bpy.data.objects.new(name=f"Camera_{ring_idx}_{i}", object_data=cam_data)
            cam_obj.location = cam_loc
            cam_obj.rotation_euler = direction_to_euler(cam_dir)
            bpy.context.collection.objects.link(cam_obj)
            camera_list.append(cam_obj)

# mask渲染函数
def render_mask_for_main(scene, main_meshes, cam, mask_path, res_x, res_y):
    # 1. 记录原始设置
    orig_lights = {light: light.energy for light in bpy.data.lights}
    world = scene.world
    orig_world_nodes = None
    if world and world.node_tree:
        orig_world_nodes = []
        for n in world.node_tree.nodes:
            node_info = (n.name, n.type)
            # 安全地获取输入值
            if hasattr(n, 'inputs') and len(n.inputs) > 0 and hasattr(n.inputs[0], 'default_value'):
                try:
                    input_value = n.inputs[0].default_value[:] if hasattr(n.inputs[0].default_value, '__getitem__') else n.inputs[0].default_value
                    node_info += (input_value,)
                except:
                    node_info += (None,)
            else:
                node_info += (None,)
            orig_world_nodes.append(node_info)
    orig_view_transform = scene.view_settings.view_transform
    orig_look = scene.view_settings.look
    orig_gamma = scene.view_settings.gamma

    # 2. 隐藏所有非main_meshes
    hidden_objs = []
    for o in bpy.data.objects:
        if o not in main_meshes and o.type == 'MESH':
            if not o.hide_render:
                o.hide_render = True
                hidden_objs.append(o)
    # 3. main_meshes全白Emission
    orig_materials = []
    mask_mat = bpy.data.materials.new(name="mask_mat")
    mask_mat.use_nodes = True
    nodes = mask_mat.node_tree.nodes
    for n in nodes:
        nodes.remove(n)
    output = nodes.new(type='ShaderNodeOutputMaterial')
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[0].default_value = (1,1,1,1)
    emission.inputs[1].default_value = 1.0
    mask_mat.node_tree.links.new(emission.outputs[0], output.inputs[0])
    for mesh in main_meshes:
        if len(mesh.material_slots) == 0:
            mesh.data.materials.append(mask_mat)
        orig_materials.append([slot.material for slot in mesh.material_slots])
        for i in range(len(mesh.material_slots)):
            mesh.material_slots[i].material = mask_mat
    # 4. 关闭光照
    for light in bpy.data.lights:
        light.energy = 0.0
    # 5. 背景全黑
    if world and world.node_tree:
        # 移除所有非BACKGROUND节点
        for node in list(world.node_tree.nodes):
            if node.type != 'BACKGROUND':
                world.node_tree.nodes.remove(node)
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                node.inputs[0].default_value = (0,0,0,1)
                node.inputs[1].default_value = 1.0
    # 6. 色彩管理设为Standard
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.gamma = 1.0
    # 7. 渲染
    scene.camera = cam
    scene.render.filepath = mask_path
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    bpy.ops.render.render(write_still=True)
    # 8. 恢复材质
    for mesh, mats in zip(main_meshes, orig_materials):
        for i, mat in enumerate(mats):
            mesh.material_slots[i].material = mat
    # 9. 恢复隐藏
    for o in hidden_objs:
        o.hide_render = False
    # 10. 恢复灯光
    for light, energy in orig_lights.items():
        light.energy = energy
    # 11. 恢复世界背景节点
    if world and world.node_tree and orig_world_nodes is not None:
        # 先清空所有节点
        for node in list(world.node_tree.nodes):
            world.node_tree.nodes.remove(node)
        # 恢复节点
        for name, ntype, val in orig_world_nodes:
            node = world.node_tree.nodes.new(type='ShaderNodeBackground' if ntype == 'BACKGROUND' else 'ShaderNodeBackground')
            node.name = name
            if val is not None and hasattr(node, 'inputs') and len(node.inputs) > 0:
                try:
                    # 确保输入值的维度正确
                    if hasattr(node.inputs[0], 'default_value'):
                        expected_dim = len(node.inputs[0].default_value)
                        if val is not None:
                            if len(val) == 3 and expected_dim == 4:
                                # RGB转RGBA，添加alpha=1.0
                                node.inputs[0].default_value = (*val, 1.0)
                            elif len(val) == 4 and expected_dim == 4:
                                # 直接设置RGBA
                                node.inputs[0].default_value = val
                            elif len(val) == expected_dim:
                                # 维度匹配，直接设置
                                node.inputs[0].default_value = val
                            else:
                                # 维度不匹配，使用默认值
                                print(f"⚠️ 跳过节点 {name} 的输入值恢复，维度不匹配")
                except Exception as e:
                    print(f"⚠️ 恢复节点 {name} 时出错: {e}")
    # 12. 恢复色彩管理
    scene.view_settings.view_transform = orig_view_transform
    scene.view_settings.look = orig_look
    scene.view_settings.gamma = orig_gamma

### === 创建三圈相机 === #
camera_list = []
create_cameras_rings(camera_list, bounds_min, bounds_max, main_obj, camera_rings_config)

adjust_lighting_to_target_brightness(0.4, camera_list=camera_list)

# === 变量定义 ===
scene_name = os.path.splitext(os.path.basename(scene_blend_path))[0]
object_name = os.path.splitext(os.path.basename(object_blend_path))[0]


### === 渲染任务收集 ===
render_tasks = []
for frame in range(num_frames):
    pos = (
        start_location[0] + move_step[0] * frame,
        start_location[1] + move_step[1] * frame,
        start_location[2] + move_step[2] * frame
    )
    for cam in camera_list:
        img_name = f"{scene_name}_{object_name}_{cam.name}_forward_{frame+1:02d}.png"
        render_tasks.append((pos, cam, img_name))
for frame in range(num_frames):
    pos = (
        start_location[0] - move_step[0] * frame,
        start_location[1] - move_step[1] * frame,
        start_location[2] - move_step[2] * frame
    )
    for cam in camera_list:
        img_name = f"{scene_name}_{object_name}_{cam.name}_reverse_{frame+1:02d}.png"
        render_tasks.append((pos, cam, img_name))

# === 主图渲染 ===
for pos, cam, img_name in render_tasks:
    main_obj.location = pos
    scene.camera = cam
    scene.render.filepath = os.path.join(output_path, img_name)
    print(f"🎬 渲染主图: {scene.render.filepath}")
    bpy.ops.render.render(write_still=True)

# === mask渲染 ===
for pos, cam, img_name in render_tasks:
    main_obj.location = pos
    mask_path = os.path.join(output_mask_dir, img_name)
    print(f"🎬 渲染mask: {mask_path}")
    render_mask_for_main(scene, main_meshes, cam, mask_path, scene.render.resolution_x, scene.render.resolution_y)
