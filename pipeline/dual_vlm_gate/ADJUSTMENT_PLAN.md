# Dual VLM Gate Adjustment Plan

保存时间：2026-05-08

## 背景

当前 `obj_1709 + obj_1708` 在双物体 VLM gate 中，VLM 反复给出类似问题：

- `object_too_small`
- `floating_object`
- `flat_lighting`

现有动作空间主要会反复触发 `A_SCALE_UP_10` / `B_SCALE_UP_10`，但缺少更直接的贴地、重新布局、相机和光照控制，所以分数容易卡住。

## 待加入的调整参数

优先加入这些动作：

- `A_SNAP_TO_GROUND`
- `B_SNAP_TO_GROUND`
- `PAIR_FORCE_B_ON_GROUND`
- `A_MOVE_LEFT_SMALL`
- `A_MOVE_RIGHT_SMALL`
- `A_MOVE_FORWARD_SMALL`
- `A_MOVE_BACK_SMALL`
- `B_MOVE_LEFT_SMALL`
- `B_MOVE_RIGHT_SMALL`
- `B_MOVE_FORWARD_SMALL`
- `B_MOVE_BACK_SMALL`
- `CAMERA_ELEVATION_UP`
- `CAMERA_ELEVATION_DOWN`
- `CAMERA_FOCAL_WIDER`
- `CAMERA_FOCAL_TIGHTER`
- `CAMERA_CENTER_A`
- `CAMERA_CENTER_B`
- `CAMERA_CENTER_PAIR`
- `L_SPOT_HIGHER`
- `L_SPOT_LOWER`
- `L_SPOT_WIDER`
- `L_SPOT_NARROWER`
- `ENV_STRENGTH_UP`
- `ENV_STRENGTH_DOWN`

## 修改位置

需要同步修改三个文件：

1. `pipeline/dual_vlm_gate/dual_actions.py`
   - 增加 action spec
   - 增加 issue tag 到 action 的 fallback 映射
   - 必要时扩展 control state 字段

2. `pipeline/dual_vlm_gate/dual_action_space.json`
   - 把新增 action 暴露给 VLM prompt
   - 否则 VLM 输出会被过滤

3. `pipeline/dual_vlm_gate/dual_scene_render.py`
   - 真正消费新增 control state 字段
   - 映射到 Blender 中的对象位置、贴地、相机、焦距、灯光参数

## 设计原则

- 继续使用离散 action，不让 VLM 直接输出任意连续数值。
- 每个 action 必须有固定 delta 和 bounds，方便审计、回滚和复现。
- 对贴图 mesh 不默认开放颜色/材质改动，优先调整布局、尺度、相机和光照。
- 优先解决可视问题：贴地、尺度、相机 framing、光照，而不是修改纹理。

## Reminder

当前 7 场景批跑完成后，先提醒用户执行本计划，再决定是否重跑双物体 gate。
