# 训练数据生产编译器

生产编译器 `data.training_supervision` 承接已验收的无素材原场景分解方案，保留完整规划、主体列表、媒体引用和来源族。通过 `training_run freeze` 固定配置、模型资产身份和 Git 源码快照；使用该快照执行 `training_run run`，再独立执行 `validate`。

只有全部任务记录、真实回执、媒体检查、可移植加载、Harness 回放和冻结合同均通过，才算数据生产验收。当前工程回归通过；真实小批仍在等待 GPU。视频方案已经生成，视频执行适配尚待完成。

| 入口 | 实现 |
| --- | --- |
| Harness Operator | `dataevolver.harness.adapters.training_supervision` |
| 九个媒体操作 | `dataevolver.dataset.training_pipeline` |
| 独立样本与数据集加载 | `dataevolver.dataset.supervision` |
| 离线模型适配 | `dataevolver.adapters.training_media_worker` |
| 冻结、执行、验收 | `dataevolver.harness.training_run` |

```bash
PYTHONPATH=src python -m dataevolver.harness.training_run freeze \
  --contract "$production_contract" --out "$run_config" \
  --campaign-id "$campaign_id" --wall-seconds "$wall_seconds"
source "$run_config/env.sh"
PYTHONPATH="$run_config/source/src" "$DATAEVOLVER_TRAIN_PYTHON" \
  -m dataevolver.harness.training_run run --config-dir "$run_config"
PYTHONPATH="$run_config/source/src" "$DATAEVOLVER_TRAIN_PYTHON" \
  -m dataevolver.harness.training_run validate --config-dir "$run_config"
```

`production_contract`: `planning.goal` + `planning.plan` + `production` + `models` + `stores`。

`env.sh`: 私有部署配置，提供合同指定的模型、解释器与存储环境变量。

```python
from dataevolver.dataset.supervision import load_dataset
from dataevolver.dataset.contracts import ArtifactRef

records = load_dataset("/path/to/portable_dataset")
record, artifact_store = records[0]
background = artifact_store.resolve(ArtifactRef.from_dict(record["input"]["background"]))
target = artifact_store.resolve(ArtifactRef.from_dict(record["target"]["image"]))
```

`evidence_kind=fixture`、`allow_fixture=True`: 工程测试专用。

`training_benefit_verified=False`、`rsi_benefit_verified=False`: 当前证据边界。
