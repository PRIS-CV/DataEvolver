# 图像收集与布局清洗

本路线提供两条独立 pipeline。收集输出 `SourceBundle.v1`。清洗通过 ArtifactRef 读取源包，输出 `curated_images.v2`。当前证据级别是 `mock_verified`。

本机负责代码和合成 fixture 检查。真实采集、模型部署、推理和图像批处理放到新 Linux 服务器。既有 `data.image_curation@2` 保留离线 CPU 约束；原 `curated_images.v1` 加载器继续可用。

## 离线入口

使用 Python 3.12 和隔离环境：

```bash
python -m pip install -e ".[image-data,test]"
python -m dataevolver.tools.image_pipeline_fixture \
  --root .dataevolver/image-pipeline-fixture
```

fixture 命令生成 8 张合成源图，其中包含 1 张跨来源重复图。两次 campaign 分别执行 `plan`、`evolve --dry-run`、`resume`、`status` 和 `replay`。缺省导出 7 条 mock 记录。运行根目录保留 `harness_calls.jsonl` 和 `fixture_result.json`。

新运行使用新目录；中断恢复使用原 campaign，不改写原 Goal 和输入。

```bash
python -m dataevolver.harness.cli plan --goal <goal.json>
python -m dataevolver.dataset.curation_loader <导出目录>
python -m pytest -q tests/test_image_pipeline.py tests/test_data_curation.py \
  tests/test_image_recipe.py tests/test_visual_contracts.py \
  tests/test_model_gateway.py tests/test_visual_harness.py
```

`plan` 只读本地合同，报告节点、预算、解析后的配置和缺失条件。命令不创建 campaign、不查询端点，也不调用模型。

## 收集

`data.image_collection@1` 复用数据图、预算、回执和恢复机制。公共入口 `collection_contract()` 和 `image_pipeline_goal()` 构造 Harness 合同。来源规范由不可变的 CollectionSpec 输入引用。

HF 端点按以下顺序解析，并绑定到运行：

```text
CollectionSpec.hf_endpoint → HF_ENDPOINT → https://hf-mirror.com
```

元数据和 Hub 文件共用端点与精确 revision。镜像缺少指定 revision 时失败，不自动切换。默认下载并发为 4，单文件外层尝试最多 3 次；SDK 内部传输重试属于文件尝试内部。外部 URL 使用 ETag 和 Range 续传，VG 图片保留原域。

来源模板 `image_sources.example.json` 固定 COCO/VG 入口及 revision，默认配额各 2,500。`__SET_...__` 表示服务器待核对的字段和 URL，不能作为已验证来源配置。

COCO 按 image_id 合并图像信息、实例、caption 和图像许可。单独的 Hub `train` 或数字文件名不证明上游 `train2017`。VG 保留对象 ID、属性归属和关系边，并通过 COCO 对应 ID 排除已知评测图像。远程加载脚本不执行。

`tables` 的 Hub 文件用 `path`，外链用 `url`，JSON 容器用 `key`，ZIP 内部文件用 `member`。Fixture 使用 `artifact`。`field_mapping` 指定原字段到规范字段的映射。图像可来自 Parquet 分片或原 URL。元数据存在 `image_shard` 时只读所需分片；否则读取全部显式配置的分片。

源包包含嵌入图像字节与原标注的 Parquet、来源证据、已解析的采集规范、下载记录和成员哈希。源包文件完成后设为只读。原始缓存与源包分开保存。`load_source_bundle()` 可独立核验源包。

下载记录区分实际接收的文件负载字节、完整文件大小和缓存命中。计数不含 HTTP 头及元数据 API 响应体。少量入选图像仍可能需要下载完整分片。失败不自动替换来源或提高配额。

## 清洗

`data.image_curation@3` 使用固定节点顺序：

```text
audit → families → review → repair → rereview → export
```

审计实际解码图像，保留原图和原标注，将 EXIF 方向同步应用到像素与框，再导出 PNG。越界框不会静默裁剪。标注缺失或矛盾时最多修订一次，随后由 reviewer 独立复核。

`generic_image_v1` 与 `multi_subject_layout_v1` 编译到 `QualityPolicy.task_criteria`。参数变更通过新配置和新运行生效。保护条件不接受模型输出覆盖。

| 项目 | 缺省值 |
| --- | --- |
| 原图短边、最大长宽比、最大像素数 | 256 px；4；64 MP |
| 规范化图像长边上限 | 2,048 px |
| 精确哈希与 dHash | 精确 RGB 像素哈希；dHash 间隔 4 bit |
| 划分与固定种子 | train/dev/eval 为 90/5/5；20261103 |
| 主体数 | 至少 2 个 |
| 入选框 | 短边至少 32 输出像素；面积至少 1% |
| 非歧义方向 | 完整框间隔至少 0.02 |

这些数值是工程初值，不是 Boogu 公开阈值。`overrides` 可调整质量参数。类别合并只使用显式的 `category_mapping`。

重复组选择一个通过记录。同源关联组约束划分，保留不同照片。系统不把同一主体的不同照片直接当成重复。小切片不保证三个划分都有记录，也不保证比例精确。两两 dHash 分组限于 5,000 条候选。

`count_scope` 显式列出入选对象 ID。框使用方向统一后的归一化 xyxy。对象 ID、属性归属和关系引用都应有效。首版只从完整框生成 `left_of`、`right_of`、`above`、`below`。其他关系保留原证据。

独立复核分别检查主体、caption、属性、关系、模糊、水印、噪声和曝光。caption 声明应与数量、对象、属性和方向一致。阈值从独立校准报告读取；缺少校准时模型的通过建议仍为 unknown。修订者与复核者耦合时，需要独立人工证据才能批准修订。

## 证据、交付及后续

许可缺省允许列表为空。数据集声明与逐图条件分别核对。隐私、安全、删除条件和来源证据由不可变输入提供；模型不能自批。服务部署、校准和保护证据模板都在 `configs/harness/image_*.example.json`。

导出含 JSONL、PNG、原图字节、原 caption 与标注、核验事实、gate、review、模型请求与回执、来源证据、配置、隔离、数据说明及独立加载回执。加载器校验成员、哈希、图像、坐标、引用、计数、几何方向及关联划分。

- `mock_verified` 始终表示合成或 mock 证据，不能升级为真实候选。
- `layout_ready` 字段表示布局核验通过。
- `technical_candidate_pending_independent_review` 表示仍缺保护证据。
- `training_candidate` 表示当前检查及独立加载通过，训练收益尚未验证。

路线不生成 `train_ready`。单来源可零通过；整体零通过输出 `NO_QUALIFIED_OUTPUT` 和原因统计。执行 evaluator 只验证流程完成与产物有效性，不承诺筛选产量。

PE 的用户要求、`rewritten_prompt` 和比例字段由后续适配器生成。当前交付是通用布局中间语料。

服务器顺序与公共 API 示例见[服务器交接](IMAGE_DATA_SERVER_HANDOFF.md)。先验收 50 张真实候选，再用约 200 张开发切片校准，随后扩到 2,000–5,000 张候选。当前没有真实数据或训练验证结论。
