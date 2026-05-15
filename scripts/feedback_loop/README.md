# External Feedback Loop v3

这个目录保存外部反馈 loop 的通用工具层。主入口仍是：

```bash
python scripts/run_external_dataset_quality_loop.py ...
```

## 设计

v3 loop 把两个反馈信号合并起来：

1. 质量 gate：Stage1-3 生成资产后，先经过 Blender/VLM gate，只保留 accepted 样本。
2. 评测反馈：训练和评测后，用 `compare.py` 找弱角度，再指导下一轮只补弱角度数据。

完整链路：

```text
eval metrics
-> compare.py
-> analyze_feedback_for_dataset.py
-> Stage1 AI seed concepts
-> Stage2/2.5/3 assets
-> VLM quality gate
-> accepted-only root
-> export rotation views
-> build train-ready
-> optional baseline augmentation with pinned split
-> validate final dataset
```

## 工具

- `compare.py`：读取 TestSet / SpatialEdit 指标，输出整体、角度、物体、物体-角度维度的 delta 和弱角度。
- `analyze_feedback_for_dataset.py`：把 compare 结果转成 `dataset_feedback_plan.json`。
- `build_pinned_split.py`：冻结已有 baseline 的 train/val/test object split。
- `build_augmented_rotation_dataset.py`：把新 accepted train pairs 合并进 baseline，新物体只进入 train。
- `validate_dataset.py`：检查 pair 字段、图片路径、object split 泄漏、是否误用 `bbox_views/`。
- `build_render_prior_library.py`：从高分 VLM control state 提取渲染 warm-start 先验。

## 推荐用法

如果已有上一轮评测结果，直接让总控脚本先生成反馈计划：

```bash
python scripts/run_external_dataset_quality_loop.py \
  --loop-root runtime/external_dataset_quality_loop/r02 \
  --baseline-eval-dir <baseline_eval_dir> \
  --current-eval-dir <current_eval_dir> \
  --baseline-mode ours_objinfo \
  --current-mode ours_feedback \
  --baseline-split-root <baseline_split_dataset> \
  --augment-baseline \
  --use-plan-seed-count \
  ...
```

如果已经有 `dataset_feedback_plan.json`：

```bash
python scripts/run_external_dataset_quality_loop.py \
  --loop-root runtime/external_dataset_quality_loop/r02 \
  --feedback-plan runtime/external_dataset_quality_loop/r01/feedback/dataset_feedback_plan.json \
  --baseline-split-root <baseline_split_dataset> \
  --augment-baseline \
  ...
```

输出里的 `final_dataset_root` 是下一轮训练入口。
