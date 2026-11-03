# 图像数据 pipeline 服务器交接

当前交付是代码、配置模板和合成 fixture 验证。真实来源字段、图像链接、模型服务及阈值仍待服务器验收。当前 Mac 不下载模型、不部署服务，也不采集真实数据。

## 1．目录与预检

1. 在新服务器核对账户权限、可用磁盘、内存和 GPU 状态。
2. 为代码、输入证据、下载缓存、源包、清洗输出及 campaign 分配目录。
3. 保持输入 store、源包 store、清洗 store 和 campaign 目录互不嵌套。
4. 使用 Python 3.12，安装 `.[image-data,test]`。
5. 在服务器执行合成 fixture，确认软件环境。

路径通过环境变量配置。新服务器地址和实际目录不写入代码：

```bash
export IMAGE_INPUT_ROOT=<只读输入及证据目录>
export IMAGE_BUNDLE_ROOT=<源包和下载缓存目录>
export IMAGE_CURATED_ROOT=<清洗输出目录>
export IMAGE_RUNTIME_ROOT=<.dataevolver 运行目录>
export IMAGE_EXPERIMENT_ROOT=<experiments/dataprocess 下的实验目录>
export HF_ENDPOINT=https://hf-mirror.com
```

采集合同中将 `inputs`、`bundle` 两个 store 分别映射到 `IMAGE_INPUT_ROOT`、`IMAGE_BUNDLE_ROOT`。清洗合同再增加 `dataset` 到 `IMAGE_CURATED_ROOT`。campaign 放到 `IMAGE_RUNTIME_ROOT/campaigns`。源码不 import 实验目录。

数据预算会记录接收的文件字节和新写入的文件字节。`max_output_bytes` 需要覆盖这些预留。`CollectionSpec.max_download_bytes` 只约束下载字节。先核对分片大小，再设置预算；不要用入选图像数估计完整分片流量。

## 2．来源配置与 50 张链路试点

来源模板是 `configs/harness/image_sources.example.json`。首轮只使用 50 张候选，将两来源配额各设为 25，并把 `max_candidates` 设为 50。

1. 核对镜像端点上的精确仓库 revision 和文件清单。
2. 替换 `__SET_...__` 元数据文件名及 VG 外部 URL。
3. 核对表字段，填入 `field_mapping`。
4. 留存数据集声明、图像来源、图像许可及上游划分证据。
5. 将已知上游验证、测试和项目评测 COCO ID 写入排除列表。
6. 固定 CollectionSpec 文件，并构造收集 Goal。

`tables` 支持 Hub `path`、外部 `url`、JSON `key` 和 ZIP `member`。COCO 可使用图像 Parquet 分片，也可使用原始 `coco_url`。配置分片时，元数据中的 `image_shard` 可缩小下载范围；缺少该索引时会读取全部显式配置的分片。VG 外部图片不改成镜像域。

下面的公共 API 在实验侧构造合同。路径、预算和 revision 应先确认：

```python
import os
from pathlib import Path
from dataevolver.dataset.contracts import ArtifactRef, file_sha256
from dataevolver.harness.data_contracts import DataRunBudget
from dataevolver.harness.image_pipeline import collection_contract, image_pipeline_goal

inputs = Path(os.environ["IMAGE_INPUT_ROOT"])
spec_path = inputs / "collection.json"
source = ArtifactRef("inputs", spec_path.name, file_sha256(spec_path))
budget = DataRunBudget(8, 0, 0, 50, 40_000_000_000, 10_800)
contract = collection_contract(
    source, mode="server",
    stores={"inputs": "IMAGE_INPUT_ROOT", "bundle": "IMAGE_BUNDLE_ROOT"},
    dataset_store="bundle", budget=budget,
)
goal = image_pipeline_goal(
    contract, campaign_id="layout-collection-canary",
    evaluator_path="configs/harness/image_pipeline_evaluator.json",
)
output = Path(os.environ["IMAGE_RUNTIME_ROOT"]) / "collection-goal.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(goal.to_json())
```

示例预算是上限，不是资源用量测量。真实执行前按文件清单和磁盘核对。源包不足会报告缺口，不替换来源。

```bash
python -m dataevolver.harness.cli plan \
  --goal "$IMAGE_RUNTIME_ROOT/collection-goal.json"
python -m dataevolver.harness.cli evolve \
  --goal "$IMAGE_RUNTIME_ROOT/collection-goal.json" \
  --campaign-root "$IMAGE_RUNTIME_ROOT/campaigns" \
  --idempotency-key layout-collection-canary
```

真实命令应放在服务器独立 tmux 会话，并保存命令、源码状态、配置、日志与退出码。上面两条真实命令本轮未执行。

## 3．服务器模型与清洗合同

`image_models.server.example.json` 定义 annotator 和 reviewer 两个 ModelGateway 角色。模板不会启动模型。服务器部署时应记录实际模型 ID、revision、模型家族、部署 ID 和 endpoint origin。模型选择和 GPU 分配依据新服务器资源确认。

1. 部署视觉 JSON 服务，记录可复核的 checkpoint revision。
2. 将服务信息填入 `image_deployments.example.json`。
3. 将 profile、服务模型 ID、revision 和 origin 配成相同值。
4. 设置 `IMAGE_ANNOTATOR_BASE_URL` 和 `IMAGE_REVIEWER_BASE_URL`。
5. 在服务器用 `harness.cli models --config ... --check-endpoint` 检查服务。
6. 将部署记录作为 ArtifactRef，加入 `DatasetIntent.sources`。

允许列表填写 origin，例如 `http://127.0.0.1:8001`；BASE_URL 可以包含 `/v1`。跨主机服务需要同时修改允许列表和部署证据。缺少部署记录或服务身份不匹配时，真实清洗预检阻断。

采集完成后，从 `collect` 节点的 `node_run.json` 读取 `outputs.source_bundle`。调用 `layout_contract()`，传入该 ArtifactRef、task criteria、模型配置、stores 和 DataRunBudget。50 张候选的模型调用与图像上限各为 150，覆盖初审、最多一次修订及复审。随后调用 `image_pipeline_goal()`，生成独立的清洗 Goal。

清洗的 `allow_gpu=false` 表示 Harness 不在调用进程加载 GPU 模型。模型运行在已部署的服务器服务上。真实路线允许访问配置中的来源和模型端点；fixture 路线禁用网络。

## 4．约 200 张开发切片校准

50 张链路试点用于确认来源、坐标、服务和交付链路。校准另取约 200 张开发切片，覆盖：

- 自动通过、自动拒绝和证据不足的记录。
- 多主体数量、主体大小、遮挡及空间方向。
- 模糊、水印、噪声、曝光和 caption 错误。
- 两来源、主体类别和模型分歧。

校准样本与留出评估样本分离。保护证据、独立人工金标和服务 revision 都应固定为不可变输入。

`image_calibration.example.json` 的 `passed` 缺省为 false。阈值应从实际校准报告取得。初始报告留空时，模型的通过建议仍是 unknown。离线 `plan` 输出 `review_rubric_hash` 和已解析的 `profile_hash`，应与报告绑定。开发 ID 不应落入当前划分的 eval 切片。

`image_protected_evidence.example.json` 的 `samples` 按最终样本 ID 登记：

```json
{
  "样本ID": {
    "privacy": {"status": "unknown", "evidence": []},
    "safety": {"status": "unknown", "evidence": []},
    "deletion": {"status": "unknown", "evidence": []}
  }
}
```

有独立证据后才填 pass。`evidence` 使用不可变 ArtifactRef。真实报告的 `evidence_level` 为 `independent_reference`，fixture 报告为 `mock_verified`。同家族模型修订与复核仍视为耦合；若需要批准，应提供独立的 `independent_layout` 证据。模型输出不能创建这些保护证据。

## 5．2,000–5,000 张候选试点

链路和质量验收完成后，再创建新配置和新 campaign，最多收集 COCO/VG 各 2,500 张候选。复用匹配缓存，但不复用不同配置的运行真相。

验收时应报告获取率、完整分片字节、实际接收字节、各 gate 保留率、缺证据比例、分项错误、来源及能力覆盖、耗时和峰值资源。单来源无通过记录或整体零产出都保留原因；不降低门槛补足数量。

独立复制导出目录后运行 `curation_loader`，校验全部成员、哈希、图像、框、关系、计数和关联划分。只把证据闭合的记录标为 `training_candidate`。真实训练及独立评测仍属于后续工作。
