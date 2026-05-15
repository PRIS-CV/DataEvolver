# 新 AI 论文复现投喂模板

本文档用于在需要让一个新的 AI 代理复现某一篇论文时，直接复制粘贴给它。核心目标是：让新 AI 严格按照 `AI_REPRODUCTION_GUIDE.md` 的流程，从论文出发，完成资源发现、方法拆解、小规模复现、报告输出和复现等级判断。

## 1. 最推荐发送内容

复制下面模板，替换尖括号内容：

```text
请严格按照 D:\SeeThrough3D\AI_REPRODUCTION_GUIDE.md 的流程，复现以下论文。

论文：
<arXiv 链接或论文标题>

本地 PDF：
<如果有，填写 PDF 路径；如果没有，写“无”>

工作目录：
<希望代码放置的目录>

大文件目录：
<超过 1G 的数据、生成图片、候选结果、缓存放置的目录>

服务器：
<如果有，填写 ssh 别名，例如 ssh <别名>；如果没有，写“无”>

算力信息：
<例如：服务器有 3 张 A800；如果未知，写“请自行检查”>

限制：
- 不要默认下载全量数据，先用小样本跑通整体架构。
- 不要下载新模型，除非我明确同意。
- 不要修改已有项目或已有 conda 环境。
- 如果需要环境，请创建平行环境。
- 官方最终结果只能用于验证，不能算方法复现。
- 必须明确区分 official_downloaded、official_result_recomputed、self_generated、mock_or_proxy、synthetic_proxy、not_completed。
- 必须创建 docs/METHOD_CONSTRAINT_CARD.md、REPRO_STATUS.json、reports。
- 如果缺真实数据，可以做 real-small 或 synthetic proxy 分支，但必须标注 proxy，不能声称完整论文级复现。
- 如果遇到需要用户授权的大下载、闭源模型、付费 API、超过指定算力预算、或预计运行超过 N 小时的任务，必须暂停并请求确认。
- 每完成一个阶段，必须更新 REPRO_STATUS.json，不要只在最后写状态。
- 最终报告必须说明当前属于 evaluation-chain smoke reproduction、small-scale method reproduction、real-small method reproduction、synthetic-proxy construction branch 还是 full paper-level reproduction。
- 最终报告必须说明：如果要升级到更高复现等级，还需要哪些资源、预计成本和主要风险。

请开始执行，不要只给计划；先读论文、找官方资源、拆方法线，然后实现可运行的小规模闭环。
```

## 2. 最小可用版本

如果只想快速测试新 AI 是否能独立推进，可以发送：

```text
请按照 D:\SeeThrough3D\AI_REPRODUCTION_GUIDE.md 完整复现这篇论文：
<arXiv 链接>

要求：
1. 先读论文并寻找官方 GitHub、Hugging Face、项目页和补充材料。
2. 不下载全量大文件，只抽小样本跑通。
3. 代码放到 <代码目录>。
4. 超过 1G 的数据放到 <大文件目录>。
5. 如需算力使用 <ssh 别名>。
6. 明确区分官方产物复算、真实方法复现、proxy 复现。
7. 必须输出 METHOD_CONSTRAINT_CARD.md、REPRO_STATUS.json、报告和可视化证据。
8. 遇到大下载、闭源模型、付费 API、超过算力预算或预计运行超过 N 小时的任务时，必须先请求确认。
9. 每完成一个阶段都要更新 REPRO_STATUS.json，最终说明升到更高复现等级所需资源、预计成本和风险。
```

## 3. 如果论文有本地 PDF

可以这样发：

```text
请按照 D:\SeeThrough3D\AI_REPRODUCTION_GUIDE.md 复现以下论文。

论文标题：
<论文标题>

本地 PDF：
<PDF 绝对路径>

请先根据标题或 PDF 内容查找 arXiv / 官方链接，确认论文身份，再进行复现。

复现限制：
- 不要下载全量数据。
- 优先复用本地 PDF。
- 官方资源只用于验证和 schema 对齐。
- 必须实现小规模方法复现。
```

## 4. 如果有服务器和已有模型

可以追加：

```text
服务器可通过以下方式访问：
ssh <别名>

请先只读检查服务器：
- 可用 GPU；
- conda 环境；
- /huggingface/model_hub；
- /huggingface/dataset_hub；
- 相关已有项目目录。

不要下载新模型。
不要复制大模型权重。
不要修改已有项目。
如果需要环境，请克隆或创建平行 conda 环境。

代码目录：
<服务器代码目录>

大文件目录：
<服务器大文件目录>
```

## 5. 如果想测试提示词本身能力

为了测试 `AI_REPRODUCTION_GUIDE.md` 是否足够让新 AI 独立复现，不要提前告诉它太多中间结论。只给它：

1. `AI_REPRODUCTION_GUIDE.md`
2. 论文 arXiv 链接或 PDF
3. 工作目录
4. 大文件目录
5. 服务器和下载限制

不要提前给：

- 官方仓库分析结论；
- 数据集结构答案；
- 哪些文件一定有用；
- 上一次复现的中间判断；
- 你希望它得出的结论。

这样更能测出提示词的泛化能力。

## 6. 新 AI 必须交付的结果

最终至少应交付：

```text
docs/METHOD_CONSTRAINT_CARD.md
docs/REPRODUCTION_PLAN.md
REPRO_STATUS.json
reports/<paper>_repro_summary.md
reports/artifact_lineage.json
reports/contact_sheet.jpg
```

如果论文涉及图像、视频、3D 或可视化数据，还必须交付可视化证据。

如果论文涉及 benchmark 或 preference data，还必须交付：

```text
benchmark json/jsonl
candidate metadata
metrics
preference pairs
official-vs-local comparison report
```

## 7. 最终回复应包含的判断

新 AI 的最终回复不能只说“已完成”。必须明确：

```text
当前复现等级：
已完成的方法线：
复用官方产物的方法线：
proxy / synthetic 的方法线：
not_completed 的方法线：
关键输出路径：
已运行的校验：
仍然缺什么：
是否能称为完整论文级复现：
```

推荐结论格式：

```text
evaluation smoke reproduction: pass / fail
small-scale method reproduction: pass / fail
real-small method reproduction: pass / fail / not attempted
synthetic proxy branch: pass / fail / not attempted
paper-level full reproduction: completed / not completed
```

## 8. 一句话版本

如果只想极简投喂，可以用：

```text
请严格按照 D:\SeeThrough3D\AI_REPRODUCTION_GUIDE.md，从 <arXiv 链接或 PDF> 出发复现这篇论文；不要下载全量数据，先跑通小样本闭环；代码放 <代码目录>，大文件放 <大文件目录>；如需算力用 <ssh 别名>；必须明确区分官方产物、真实方法复现和 proxy 复现，并输出 METHOD_CONSTRAINT_CARD.md、REPRO_STATUS.json、报告和可视化证据；遇到大下载、闭源模型、付费 API、超过算力预算或预计运行超过 N 小时的任务时先请求确认；每完成一个阶段更新 REPRO_STATUS.json，最终说明升到更高复现等级所需资源、预计成本和风险。
```
