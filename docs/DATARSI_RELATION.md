# DataEvolver and DataRSI / 仓库关系说明

## Repository roles

**DataEvolver is the engineering companion repository for DataRSI.** DataRSI builds on the synthetic-data construction infrastructure developed in DataEvolver and formalizes failure-driven dataset evolution as a bounded revision process.

- **[DataEvolver](https://github.com/PRIS-CV/DataEvolver)** maintains reusable engineering infrastructure: asset generation, 3D reconstruction, scene rendering, VLM review and repair, multimodal exports, and runtime tooling.
- **[DataRSI](https://github.com/PRIS-CV/DataRSI)** hosts the research framework and experimental artifacts for *DataRSI: A 3D Data Harness for Failure-to-Data Evolution*, including registered protocols, failure-to-data requests, sample admission, and regression-aware model admission.

The paper connects two loops through a Failure-to-Data Feedback interface:

1. **Data Self-Evolution Loop:** inspect candidate supervision, apply bounded repairs, and admit samples under a deterministic sample contract.
2. **Model Admission Loop:** evaluate a candidate update against the frozen champion, check target gains and protected-region regression, and return `PROMOTE` or `ROLLBACK`.

The research protocol freezes the budget, evaluators, trainer, and admission rules for each bounded revision. General-purpose engineering defaults in DataEvolver are not a substitute for the paper's registered experimental settings.

## Choosing the right entry point

| Goal | Entry point |
|---|---|
| Configure and run the engineering pipeline | [DataEvolver Quick Start](../README.md#quick-start) |
| Understand the DataRSI method and experimental scope | [DataRSI README](https://github.com/PRIS-CV/DataRSI#readme) |
| Reproduce or inspect paper results | [DataRSI reproduction guide](https://github.com/PRIS-CV/DataRSI/blob/main/docs/REPRODUCTION.md) |
| Check the relationship from the research repository | [DataRSI's relationship document](https://github.com/PRIS-CV/DataRSI/blob/main/docs/DATAEVOLVER_RELATION.md) |
| Cite the DataRSI work | [DataRSI citation metadata](https://github.com/PRIS-CV/DataRSI/blob/main/CITATION.cff) |
| Cite the earlier DataEvolver work | [DataEvolver citation](../README.md#citation) |

DataEvolver retains its repository name, `dataevolver` package namespace, CLI names, and existing deployment paths. The two repositories have distinct release histories and scopes. Existing DataEvolver publication links and recognition refer to the earlier DataEvolver work; the DataRSI research repository provides the corresponding paper materials.

## 中文说明

**DataEvolver 是 DataRSI 的配套工程仓库。** DataEvolver 提供长期维护的合成数据构建基础设施；DataRSI 在此基础上研究由模型失败驱动的有界监督数据修订过程。

本工程仓库主要用于资产生成、3D 重建、场景渲染、VLM 审查与修复、多模态数据导出和运行环境配置。DataRSI 论文仓库主要用于方法说明、固定协议、实验材料和结果复现。

DataRSI 通过失败到数据反馈接口连接数据自进化闭环与模型准入闭环：前者依据确定性样本合同接纳修复后的数据，后者依据目标收益和受保护区域的退化检查，决定晋升或回滚候选模型。论文复现应使用 DataRSI 注册的预算、评估器、训练配置和准入规则。

工程仓库继续使用 `DataEvolver` 名称及 `dataevolver` 包名，现有安装命令与路径保持兼容。引用 DataRSI 方法及实验时，请使用论文仓库的引用信息；引用既有 DataEvolver 工作时，请使用本仓库原有的引用信息。
