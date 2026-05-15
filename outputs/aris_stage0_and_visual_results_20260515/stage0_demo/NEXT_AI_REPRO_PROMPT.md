# Next AI Reproduction Prompt

请严格按照 `runtime/AI_REPRODUCTION_GUIDE.md` 的流程，从下面 Stage0
websearch 产出的候选论文中选择最适合复现目标的一篇或多篇，继续完成论文复现。

## Stage0 Research Task

derive an ARIS dataset construction method from multiple agentic test-time scaling and self-improvement papers

## Stage0 Summary

Synthesis from multiple agentic/test-time/self-improvement papers suggests ARIS should build datasets through budgeted candidate generation, trajectory logging, explicit feedback/critique, process-level quality gates, diversity preservation, and provenance-aware promotion. The construction loop should generate multiple candidate samples or policies, evaluate them with cheap validators or structured AI feedback, repair or reject with recorded reasons, and only promote batches that satisfy reproducibility, diversity, and downstream utility checks.

## Candidate arXiv Links

- https://arxiv.org/abs/2605.08083  # LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling
- https://arxiv.org/abs/2604.04373  # Decocted Experience: Reformatting Human Demonstrations for Longer-Horizon Tasks
- https://arxiv.org/abs/2604.16529  # Scaling Test-Time Compute for Agentic Coding
- https://arxiv.org/abs/2601.23228  # Scaling Multiagent Systems with Process Rewards
- https://arxiv.org/abs/2501.05707  # Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains

## Required Downstream Behavior

- 先确认论文身份，优先使用 arXiv URL，PDF 次之。
- 先拆解论文方法线，再写代码。
- 必须创建 `docs/METHOD_CONSTRAINT_CARD.md`。
- 必须明确区分 `self_generated`、`official_downloaded`、
  `official_result_recomputed`、`mock_or_proxy`、`synthetic_proxy`、
  `not_completed`。
- 官方最终产物只能用于验证，不能直接算作方法复现。
- 不要默认下载全量数据，先用小样本跑通整体架构。
- 如果需要大下载、闭源模型、付费 API、额外账号或超过预算的任务，
  必须暂停并请求用户确认。
- 每完成一个阶段，更新 `REPRO_STATUS.json`。
- 最终报告必须给出复现等级和升级到更高等级所需资源、成本、风险。

## Fill Before Running

```text
工作目录:
<代码目录>

大文件目录:
<超过 1G 的数据、生成图片、候选结果、缓存目录>

服务器:
<ssh 别名或“无”>

算力信息:
<例如 GPU 数量，未知则写“请自行检查”>

限制:
- 不要下载新模型，除非用户明确同意。
- 不要修改已有项目或已有 conda 环境。
- 如需环境，请创建平行环境。
```
