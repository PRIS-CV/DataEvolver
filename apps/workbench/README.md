# DataEvolver Workbench

A private, single-user React/Node application around Pi RPC and DataEvolver's
existing Harness. From the checkout root, run `./apps/workbench/launch.sh`.
The launcher, its Node dependencies and the UI live in this optional app;
the core stays in `src/dataevolver/harness/`. The root `launch.sh` forwards
arguments to this launcher for compatibility.

- [English guide](../../docs/en/workbench.md)
- [中文指南](../../docs/zh-CN/workbench.md)
- [Candidate orchestration](../../docs/en/orchestration.md) / [候选方案编排](../../docs/zh-CN/orchestration.md)
- [Upstream references](THIRD_PARTY_NOTICES.md)

`backend/` owns configuration, sessions, subprocess jobs, image API and HTTP/SSE.
`extensions/` exposes the backend to Pi as tools. `frontend/` contains the UI.
The public `../project-page` app remains read-only and independent.

From `apps/workbench/`, development commands are `npm ci --ignore-scripts`, `npm run typecheck`,
`npm run build`, `npm start -- --no-open`. These commands do not start an agent
until a session is created. Contract tests are available with `npm test`.
Tests and runtime workflows were not executed for the initial static review.

Data collection/curation and exported-sample review: [English](../../docs/en/data-workspace.md) / [中文](../../docs/zh-CN/data-workspace.md).
