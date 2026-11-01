# DataEvolver Workbench

A private, single-user React/Node application around Pi RPC and DataEvolver's
existing Harness. Start it from the checkout root with `./launch.sh`.

- [English guide](../../docs/en/workbench.md)
- [中文指南](../../docs/zh-CN/workbench.md)
- [Upstream references](THIRD_PARTY_NOTICES.md)

`backend/` owns configuration, sessions, subprocess jobs, image API and HTTP/SSE.
`extensions/` exposes the backend to Pi as tools. `frontend/` contains the UI.
The public `../project-page` app remains read-only and independent.

From `apps/workbench/`, development commands are `npm ci --ignore-scripts`, `npm run typecheck`,
`npm run build`, `npm start -- --no-open`. These commands do not start an agent
until a session is created. Contract tests are available with `npm test`.
Tests and runtime workflows were not executed for the initial static review.
