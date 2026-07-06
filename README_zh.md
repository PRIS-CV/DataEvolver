# DataEvolver

DataEvolver 正在整理为干净的 `src/` Python package。

这个公共分支只保留可复用 package 边界和项目元数据。本地实验、生成报告、特定数据集配置、旧 pipeline 脚本、网站输出和 agent 工作区文件都不再作为公共源码树的一部分。

## 源码结构

```text
src/dataevolver/
├── adapters/     # 模型和外部运行时适配器
├── agents/       # Agent 工作流和反馈闭环
├── annotation/   # VLM、几何和审核流程
├── dataset/      # 数据集契约、构建器和导出器
├── runtime/      # 运行时 profile、manifest、worker、observability
├── tools/        # 无状态工具和报告辅助函数
└── workflows/    # 可复用编排流程
```

## 仓库管理规则

- 新增可复用代码默认进入 `src/dataevolver/`。
- `pipeline/`、`scripts/`、`configs/`、`docs/`、`tests/`、`web/`、`.agents/`、`experiments/` 等本地整理前目录默认忽略。
- 生成数据集、报告、模型权重、复制的外部仓库和机器私有 profile 不进入公共 Git 树。
- 已经被 GitHub 追踪但应改为本地文件的 legacy 文件，需要用 `git rm --cached` 从 Git 追踪中移除，同时保留本地迁移副本。

## 当前状态

这个 PR 是仓库内容管理清理，不表示 package API 已经完整。下一步应把本地 legacy 目录中稳定、可复用的实现逐步迁移到上面的 package 模块中。
