# ARIS Pipeline Configuration Reference

> 最后更新: 2026-05-09

---

## 1. 服务器环境

| 项目 | 值 |
|------|-----|
| SSH 别名 | `<ssh-alias>`（用户 <user>，密钥免密） |
| GPU | 3 x A800 80GB |
| Python | `python`（3.10） |
| Blender | `blender`（4.2.4） |
| 代码目录 | `~/ARIS`（**唯一**，禁止引用 `~/aris2/ARIS`） |
| 运行输出 | `<runtime-root>`（symlink → `~/ARIS/runtime`） |
| tmux | `tmux new -s <name>`（screen 不可用） |

### 1.1 API 配置

| 项目 | 值 |
|------|-----|
| API 代理 | `https://dragoncode.codes/v1`（provider=anthropic） |
| API Key | `sk-***` |
| 备注 | dragoncode 可能 503，stage1 可用 `--template-only` 替代 |

### 1.2 VLM 模型

| 版本 | 路径 | 类型 | 状态 |
|------|------|------|------|
| Qwen3.5-35B-A3B | `/data/wuwenzhuo/Qwen3.5-35B-A3B` | MoE | 已弃用 |
| **Qwen3.6-27B** | `/huggingface/model_hub/Qwen3.6-27B` | Dense | **当前使用** |

切换涉及文件: `pipeline/stage5_5_vlm_review.py`（L47-48 路径, L481 类名用 `Qwen3_5ForConditionalGeneration` 非 Moe 版, L497 加载）。

---

## 2. Pipeline 架构（full_pipeline_v2）

```
Phase A-D: 资产生成
  Step 01  seed concept 生成（Claude Haiku API）
  Step 02  T2I → SAM2 → 3D 重建
  Step 03  VLM quality gate（内部循环，max N 轮）
  Step 04  build accepted pair root
  Step 05  rotation8 一致性导出（Blender Cycles 8 角度）
  Step 06  angle gate（可选）
  Step 07  trainready 数据集构建
  Step 08  object-disjoint split

Phase E: VLM 数据集评估（外部循环，max M 轮）
  → eval_dataset_vlm.py
  → analyze_feedback_for_dataset.py
  → retire_and_replace.py

Phase F: 反馈改进
  → 重新生成替换资产 → 更新数据集
```

### 2.1 启动命令

```bash
# 全流程（推荐）
python scripts/run_full_pipeline.py \
  --seed-count 1 \
  --device cuda:0 \
  --gate-max-rounds 3 \
  --max-feedback-rounds 3

# 资产生成子流程
bash scripts/launch_new_asset_force_rot8.sh \
  --seed-count N --device cuda:X --gate-max-rounds 5
```

### 2.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gate-max-rounds` | 3 | 内部 VLM gate 最大轮数 |
| `--gate-threshold` | 0.78 | gate 通过阈值 |
| `--max-feedback-rounds` | 3 | 外部评估反馈最大轮数 |
| `--retire-score-floor` | 0.40 | 低于此分数无条件淘汰 |
| `--retire-max-per-round` | 5 | 每轮最多淘汰物体数 |
| `--step-scale` | **1.0** | VLM gate 动作步长乘数（之前是 0.25，过小已修复） |
| `--gpus` | — | 多 GPU 并行，如 `0,1,2` |

---

## 3. GPU 并行

各阶段严格串行，阶段内部做**物体级并行**（subprocess + `CUDA_VISIBLE_DEVICES`）。

| 阶段 | 并行方式 | 模型 | 显存 |
|------|---------|------|------|
| Stage 1: 文本扩展 | 无（API 调用） | Claude Haiku | 0 |
| Stage 2: T2I | 单卡 | Qwen-Image-2512 | ~56 GB |
| Stage 2.5: SAM | 单卡 | SAM3 | ~3 GB |
| Stage 3: 3D 重建 | **多卡并行** | Hunyuan3D-2.1 | ~40-60 GB |
| VLM Gate | **多卡并行** | Qwen3.6-27B + Blender | ~20-35 GB |
| Rotation8 导出 | **多卡并行** | Blender CYCLES | ~2-4 GB |
| Angle Gate | 单卡 | Qwen3.6-27B + Blender | ~20-35 GB |

### 分片规则

- 物体 round-robin 分配到各 GPU
- 每个 worker 通过 `CUDA_VISIBLE_DEVICES=N` 限制为单卡
- worker 内部统一用 `cuda:0`（映射到物理 GPU N）
- VLM gate worker 额外设置 `VLM_FORCE_SINGLE_GPU=1`

### 耗时估算

- 3D 重建 + VLM Gate 是最大瓶颈
- 3 卡 → 1 卡: 每轮约 3x 慢（6 物体 ~1.8h → ~5h，18 物体 ~10-12h）

---

## 4. 场景配置

### 4.1 场景池（`configs/scene_pool.json`）

当前 7 个可用场景:

| 场景 | 类型 | blend 文件 | 备注 |
|------|------|-----------|------|
| indoor_4blend | indoor | `sence/4.blend` | 室内，常用基线 |
| outdoor7 | outdoor | `sence_ourdoor/outdoor7.blend` | 直升机停机坪，纯几何，深度全覆盖 |
| outdoor9 | outdoor | `sence_ourdoor/outdoor9.blend` | 高速公路，焦距 50mm |
| outdoor11 | outdoor | `sence_ourdoor/outdoor11.blend` | 工业停车场，纯几何，深度全覆盖 |
| outdoor13 | outdoor | `sence_ourdoor/outdoor13.blend` | 超跑场景，焦距 50mm |
| outdoor14 | outdoor | `sence_ourdoor/outdoor14.blend` | 公园，lens_override=35mm |
| outdoor15 | outdoor | `sence_ourdoor/outdoor15.blend` | BezierCurve 189m 地面 |
| outdoor16 | outdoor | `sence_ourdoor/outdoor16.blend` | 城市街道，广角 30mm |

Blend 文件路径前缀: `<large-file-root>

### 4.2 摄像头距离调优（`scene_camera_distance_scale`）

值越大 = 相机越远。

| 场景 | 当前值 | 备注 |
|------|--------|------|
| 4.blend | 1.18 | 基线室内 |
| outdoor7 | 0.503 | 2026-05-03 从 0.4025 拉远 25% |
| outdoor9 | 0.55 | — |
| outdoor11 | 继承默认 | — |
| outdoor13 | 1.1375 | 从 0.65 拉远 75% |
| outdoor14 | 1.8 | lens_override=35mm |
| outdoor15 | 0.4025 | — |
| outdoor16 | 1.225 | 从 0.7 拉远 75% |

### 4.3 场景模板结构（`configs/scene_template*.json`）

**共享参数**（所有场景一致）：

```json
{
  "use_existing_world": true,
  "disable_existing_lights": false,
  "add_key_light": true,
  "scale_existing_lights": 1.5,
  "scale_world_background": 1.2,
  "adaptive_brightness": true,
  "adaptive_brightness_damping": 0.45,
  "adaptive_brightness_min_gain": 0.9,
  "adaptive_brightness_max_gain": 2.5,
  "adaptive_world_gain_scale": 0.28,
  "render_engine": "CYCLES",
  "render_resolution": 1024,
  "cycles_samples": 512,         // outdoor14 例外: 128
  "cycles_device": "GPU",
  "cycles_compute_device_type": "CUDA",
  "filmic_gamma": 0.54,
  "target_brightness": 0.385,
  "default_object_value_scale": 1.5,
  "default_object_roughness_add": -0.03,
  "default_object_specular_add": 0.05,
  "object_scale_mode": "height_range",
  "target_bbox_height_range": [0.5, 2.0],
  "camera_mode": "scene_camera_match",
  "render_depth_normal": true,
  "depth_output_format": "PNG"    // 统一 PNG，不要用 EXR
}
```

**场景差异参数**：

| 参数 | 说明 | 差异场景 |
|------|------|---------|
| `blend_path` | blend 文件路径 | 每场景不同 |
| `support_object_name` | 地面物体名 | 4.blend="Plane", outdoor7="DOME_..." |
| `scene_camera_distance_scale` | 相机距离缩放 | 见 4.2 节 |
| `camera_lens_override` | 焦距覆盖(mm) | outdoor14=35.0, 其余无 |
| `cycles_samples` | 渲染采样数 | outdoor14=128, 其余=512 |

### 4.4 深度图

```
渲染流程: Blender Compositor → EXR(float32) → Python 后处理 → 16-bit PNG
  1. compositor 输出 EXR（保留世界坐标距离值）
  2. 读 EXR → 排除 infinity(>1e9) → min-max 归一化
  3. 近处=亮(65535), 远处=暗, 背景 infinity=黑(0)
  4. 使用 convert_depth_exr_to_png() 函数
```

> **注意**: 直接渲染 PNG 会丢失精度（值域压缩只剩黑白），必须先 EXR 再转换。

**深度覆盖率**:
- 纯几何背景（全覆盖）: outdoor7, outdoor11
- HDRI 背景: 仅几何体区域有深度，天空无深度

---

## 5. VLM 动作空间

### 5.1 `action_space.json` — 34 动作离散空间

用于 VLM gate 的渲染参数微调。

| 分组 | 动作 | 目标参数 | 类型 | delta | 范围 |
|------|------|---------|------|-------|------|
| **lighting** | L_KEY_UP/DOWN | key_scale | mul | 1.4 / 0.65 | [0.3, 3.0] |
| | L_FILL_UP/DOWN | fill_scale | mul | 1.4 / 0.65 | [0.3, 3.0] |
| | L_RIM_UP/DOWN | rim_scale | mul | 1.4 / 0.65 | [0.3, 3.0] |
| | L_WORLD_EV_UP/DOWN | world_ev | add | +0.5 / -0.5 | [-2.0, 2.0] |
| | L_KEY_SHADOW_SOFT/HARD | key_shadow_size | mul | 1.5 / 0.6 | [0.3, 4.0] |
| **camera** | C_DOLLY_IN/OUT | distance_scale | mul | 0.9 / 1.1 | [0.7, 1.3] |
| | C_PAN_LEFT/RIGHT | pan_deg | add | -4 / +4 | [-10, 10] |
| | C_TILT_UP/DOWN | tilt_deg | add | +4 / -4 | [-10, 10] |
| | C_FOV_NARROW/WIDE | fov_offset_deg | add | -5 / +5 | [-12, 12] |
| **object** | O_SHIFT_LEFT/RIGHT | offset_x | add | -0.04 / +0.04 | [-0.15, 0.15] |
| | O_SHIFT_UP/DOWN | offset_y | add | +0.04 / -0.04 | [-0.15, 0.15] |
| | O_LIFT/LOWER | offset_z | add | +0.04 / -0.04 | [-0.15, 0.15] |
| | O_SCALE_UP/DOWN | scale | mul | 1.15 / 0.85 | [0.7, 1.4] |
| **scene** | S_BG_BRIGHTER/DARKER | bg_ev | add | +0.4 / -0.4 | [-1.5, 1.5] |
| | S_BG_CONTRAST_UP/DOWN | bg_contrast | mul | 1.2 / 0.8 | [0.5, 2.0] |
| | S_CONTACT_SHADOW_UP/DOWN | contact_shadow_strength | mul | 1.4 / 0.65 | [0.0, 3.0] |
| | S_DOF_STRONGER/WEAKER | dof_strength | add | +0.1 / -0.1 | [0.0, 1.0] |
| **material** | M_SATURATION_DOWN | saturation_scale | mul | 0.8 | [0.3, 1.5] |
| | M_VALUE_UP | value_scale | mul | 1.2 | [0.3, 2.0] |

**复合动作**: L_RIM_FILL_UP, L_RIM_SHADOW_SOFT, L_KEY_FILL_UP（各含 2 个子动作，any_success 策略）

### 5.2 `scene_action_space.json` — 场景感知动作空间

场景级别的附加控制，用于 Blender 场景插入：

| 分组 | 动作 | 目标参数 | 类型 | delta |
|------|------|---------|------|-------|
| lighting | L_KEY_YAW_POS/NEG_15 | key_yaw_deg | add | ±15 |
| object | O_ROTATE_Z_POS/NEG_15 | yaw_deg | add | ±15 |
| | O_SCALE_UP/DOWN_10 | scale | mul | 1.2 / 0.8 |
| scene | ENV_ROTATE_30/NEG_30 | hdri_yaw_deg | add | ±30 |
| | ENV_STRENGTH_UP/DOWN | env_strength_scale | mul | 1.4 / 0.65 |
| material | M_VALUE_DOWN_STRONG | value_scale | mul | 0.65 |
| | M_HUE_WARM_NEG/POS | hue_offset | add | ±0.05 |
| | M_ROUGHNESS_UP/DOWN | roughness_add | add | ±0.15 |
| | M_SHEEN_UP | specular_add | add | +0.12 |

### 5.3 VLM 步长调优历史

| 版本 | delta 大小 | step_scale | 3 轮提升 |
|------|-----------|------------|---------|
| v1 | 原始（偏小） | 0.25 | +0.0009 |
| v2 | 增大 | 0.25 | +0.0044（~5x） |
| **v3** | 增大 | **1.0** | 待验证 |

> **三重衰减问题**: delta 本身小 × `get_step_scale()` 逐轮衰减 × `--step-scale 0.25` → 每步几乎无变化。已通过增大 delta + step_scale=1.0 修复。

---

## 6. VLM Review Schema（`vlm_review_schema.json`）

VLM 输出结构化评审结果，版本 `vlm_review_v1`：

### 6.1 评分维度

| 维度 | 类型 | 范围 |
|------|------|------|
| lighting | int | 1-5 |
| object_integrity | int | 1-5 |
| composition | int | 1-5 |
| render_quality_semantic | int | 1-5 |
| overall | int | 1-5 |

每个维度附带 confidence: `low` / `medium` / `high`

### 6.2 路由决策

`vlm_route`: `pass` / `needs_fix` / `reject`

### 6.3 Issue Tags（最多 3 个）

涵盖 30+ 标签:

| 类别 | 标签 |
|------|------|
| 曝光 | underexposed, overexposed |
| 光照 | flat_lighting, harsh_shadow, weak_subject_separation, scene_light_mismatch, shadow_missing |
| 背景 | background_too_bright/dark, background_distracting |
| 构图 | object_too_small/large, off_center_*, object_cutoff_* |
| 物理 | floating_object, ground_intersection, floating_visible, ground_intersection_visible |
| 几何 | geometry_distortion, mesh_interpenetration, partial_occlusion |
| 遮罩 | mask_boundary_error, mask_hole, mask_spill |
| 景深 | depth_of_field_too_strong/weak |
| 材质 | color_shift, scale_implausible |
| 物理 | physical_implausibility |

### 6.4 诊断字段

| 字段 | 取值 |
|------|------|
| lighting_diagnosis | flat_no_rim, flat_low_contrast, underexposed_global, underexposed_shadow, harsh_shadow_key, scene_light_mismatch, shadow_missing, good |
| structure_consistency | good, minor_mismatch, major_mismatch |
| color_consistency | good, minor_shift, major_shift |
| physics_consistency | good, minor_issue, major_issue |

### 6.5 Issue → Action 映射

VLM 检测到问题后自动映射到 action（优先级从左到右）：

| 问题 | 推荐动作 |
|------|---------|
| underexposed | L_KEY_FILL_UP → L_KEY_UP → L_WORLD_EV_UP |
| overexposed | L_KEY_DOWN → L_WORLD_EV_DOWN |
| flat_lighting | L_FILL_DOWN → L_KEY_SHADOW_SOFT → L_RIM_UP |
| harsh_shadow | L_KEY_SHADOW_SOFT → L_FILL_UP |
| object_too_small | C_DOLLY_IN → O_SCALE_UP |
| object_too_large | C_DOLLY_OUT → O_SCALE_DOWN |
| floating_object | O_LOWER → S_CONTACT_SHADOW_UP |
| color_shift | M_SATURATION_DOWN → M_VALUE_UP |
| mesh_interpenetration | __MESH_REJECT__（直接拒绝） |

---

## 7. Pipeline 优化决策

### 7.1 A 组跳过 Angle Gate

A 组（全角度组）物体已通过 VLM quality gate，rotation8 从同一 best state 渲染不同角度，质量有保障，无需逐角度 VLM 评估。省去 ~2h/轮。

### 7.2 动态 3:2 分组

`--group-ratio 3:2` 应用于 VLM gate 后的 accepted 数量:
- 13 accepted → A=8, B=5
- 不整除时 `ceil` 向上取整，优先满足 A

### 7.3 B 组 Angle Gate 多卡并行

`run_rotation8_angle_gate.py --gpus` 按物体分片，每卡一个 subprocess。

---

## 8. 已知 Bug 与修复

### 8.1 VLM 双加载 OOM（模块路径不一致）

**问题**: `run_rotation8_angle_gate.py` Phase 1 用 `from pipeline.stage5_5_vlm_review import ...`，Phase 2 用 `from stage5_5_vlm_review import ...`，Python 视为两个独立模块→ VLM 加载两次 → 68GB x2 OOM。

**修复**: 统一为 `from stage5_5_vlm_review import run_vlm_review`（一行修复）。

**教训**: 同一 .py 文件不能混用 `package.module` 和 `module` 两种导入路径。

### 8.2 VLM warmup OOM（MoE 预分配过大）

**问题**: `transformers.modeling_utils.caching_allocator_warmup` 按总参 35B 估算预分配 (~65GB)，但 MoE 实际只用 ~3B 活跃参数。

**修复**: monkey-patch 禁用 warmup（`stage5_5_vlm_review.py:load_vlm()`），仅影响加载速度慢几秒。

### 8.3 scene_assignments 多 GPU 覆盖

**问题**: 多 GPU 模式下 worker 子进程各自覆盖 `scene_assignments.json`，最后一个 worker 只留 1/3 条目。

**修复**: 加判断 `if not assign_path.exists() or args.gpus:` 防止 worker 覆盖，全局状态文件只由父进程写入。

---

## 9. 脚本速查

### 9.1 主流程脚本（`scripts/`）

| 脚本 | 作用 |
|------|------|
| `run_full_pipeline.py` | 端到端全流程（Phase A-F） |
| `stage1_generate_ai_seed_concepts.py` | AI 生成 seed concept |
| `build_scene_assets_from_stage1.py` | T2I → SAM2 → 3D |
| `run_vlm_quality_gate_loop.py` | VLM 质量门控循环 |
| `build_accepted_pair_root_from_gate.py` | 构建 accepted pair root |
| `export_rotation8_from_best_object_state.py` | rotation8 导出 |
| `build_rotation8_trainready_dataset.py` | trainready 数据集构建 |
| `build_object_split_for_rotation_dataset.py` | object-disjoint split |

### 9.2 Feedback Loop 脚本（`scripts/feedback_loop/`）

| 脚本 | 作用 |
|------|------|
| `eval_dataset_vlm.py` | VLM 评估 trainready 数据集 |
| `analyze_feedback_for_dataset.py` | 分析弱角度/弱物体 |
| `retire_and_replace.py` | 淘汰差物体 + 生成替换 |
| `build_augmented_rotation_dataset.py` | 合并新数据到 baseline |
| `validate_dataset.py` | 数据集完整性校验 |

---

## 10. 操作规范

| 规范 | 说明 |
|------|------|
| 代码同步 | 修改后优先 `scp` 到服务器，不主动 git |
| Git 推送 | v2ray 代理端口 10808，推送前确认代理已启动 |
| 多行命令 | 写成脚本文件，避免终端粘贴问题 |
| 核心脚本 | 使用现有参数控制行为，不改动核心脚本 |
| 深度图 | 代码默认 + 模板统一 PNG，新场景必须包含 `"depth_output_format": "PNG"` |
| 新场景模板 | 必须包含全部共享参数 + 正确的 `support_object_name` + `depth_output_format: PNG` |
