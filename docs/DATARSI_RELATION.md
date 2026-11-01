# DataEvolver and DataRSI / 仓库关系说明

## Repository roles

**DataEvolver is our engineering project.** The arXiv DataEvolver paper is our first technical report. Further iterations led to **DataRSI: A 3D Data Harness for Failure-to-Data Evolution**, submitted to **ICASSP 2027**.

- **[DataEvolver](https://github.com/PRIS-CV/DataEvolver)** maintains a portable Harness base: campaign control, registered operators, model/provider bindings, artifact validation and recovery. Asset generation, reconstruction, review and Blender/UE5 rendering are optional workflows.
- **[DataEvolver on arXiv](https://arxiv.org/abs/2605.01789)** is the first technical report, *Let Your Data Build and Improve Itself via Goal-Driven Loop Agents*.
- **[DataRSI](https://github.com/PRIS-CV/DataRSI)** is the subsequent research paper on registered protocols, failure-to-data requests, sample admission and regression-aware model admission. Its public repository currently contains a release notice; implementation and reproduction materials are being prepared for release.

The paper connects two loops through a Failure-to-Data Feedback interface:

1. **Data Self-Evolution Loop:** inspect candidate supervision, apply bounded repairs, and admit samples under a deterministic sample contract.
2. **Model Admission Loop:** evaluate a candidate update against the frozen champion, check target gains and protected-region regression, and return `PROMOTE` or `ROLLBACK`.

The research protocol freezes the budget, evaluators, trainer, and admission rules for each bounded revision. The manuscript results cover one bounded revision and do not establish repeated recursive improvement. Engineering defaults in DataEvolver are not a substitute for the paper's registered experimental settings.

## Choosing the right entry point

| Goal | Entry point |
|---|---|
| Configure and run the engineering pipeline | [DataEvolver Quick Start](../README.md#quick-start) · [Task lifecycle](../README.md#task-lifecycle) |
| Understand the DataRSI method and manuscript results | [Research update](../README.md#datarsi-research-update) · [Project website](https://pris-cv.github.io/DataEvolver/#results) |
| Follow DataRSI code, reproduction and citation releases | [DataRSI release notice](https://github.com/PRIS-CV/DataRSI#readme); a complete public reproduction package is not yet available. |
| Inspect historical repair examples | [Trace subpage](https://pris-cv.github.io/DataEvolver/traces/?lang=en) · [Evidence policy](../apps/project-page/frontend/traces/README.md) |
| Cite the first DataEvolver technical report | [DataEvolver citation](../README.md#citation) |

DataEvolver retains its repository name and `dataevolver` package namespace. Current execution uses the unified Harness CLI; private deployment paths belong to host configuration. The arXiv citation and existing workshop/poster recognition refer to the earlier DataEvolver work. DataRSI is labeled as submitted to ICASSP 2027; this does not claim acceptance or a complete public code release.

## 中文说明

**DataEvolver 是我们实际维护的工程项目。** arXiv 上的 DataEvolver 是第一版技术报告。项目进一步迭代后，我们形成了新论文 **DataRSI: A 3D Data Harness for Failure-to-Data Evolution**，并投稿至 **ICASSP 2027**。

本工程仓库维护可迁移的 Harness 基础版本，提供 Campaign 控制、注册算子、模型/Provider 绑定、产物验证与恢复。资产生成、重建、审查及 Blender/UE5 渲染是可选工作流。DataRSI 的公开仓库目前只有发布公告与许可证；实现代码和复现材料仍在整理。方法与论文数字见[中文研究更新](../README_zh.md#datarsi-研究更新)。

DataRSI 通过失败到数据反馈接口连接数据自进化循环与模型准入循环。前者依据确定性样本合同接纳修复后的数据，后者依据目标收益与受保护区域的回归检查，决定晋升或回滚候选模型。论文结果覆盖一次预算内的修订，尚未证明多轮递归改进。论文复现需遵守已注册的预算、评估器、训练配置和准入规则，完整公开复现包尚未发布。

工程仓库继续使用 `DataEvolver` 名称及 `dataevolver` 包名。现有 arXiv 引用和海报展示记录属于早期 DataEvolver 工作；[KDD Workshop 官网](https://usail-hkust.github.io/aidatasci/accepted-papers.html)当前将本项目列为 Poster。DataRSI 的状态是已投稿 ICASSP 2027。引用第一版技术报告时，使用[本仓库的引用条目](../README_zh.md#引用)。DataRSI 后续发布与引用信息以[论文仓库公告](https://github.com/PRIS-CV/DataRSI#readme)为准。
