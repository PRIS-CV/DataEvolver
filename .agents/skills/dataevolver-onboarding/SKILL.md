---
name: dataevolver-onboarding
description: Inspect a DataEvolver execution host and prepare explicit model and scene profiles for the unified Harness.
---

# DataEvolver onboarding

Reuse the user's existing execution host, model choices and budget. Ask only for
missing information required by the selected workflow. Read `AGENT_SETUP.md` and
the relevant workflow contract before installing dependencies or running work.

- Inspect Git status, Python, disk, existing environments and model directories.
- Before GPU work, inspect memory, utilization and processes; preserve other jobs.
- Start with `python -m dataevolver --help` and `operator-sdk catalog`.
- Bind data plans with `compile-plan`; bind configured native workers with `worker-compile`.
- Execute through the same `create`, `step`, `status`, `replay` commands.
- Bind the host's chosen models through explicit ModelGateway profiles; no model or tool is required by default.
- Keep private deployment information under ignored `.dataevolver/local/`.
- Do not infer model readiness, rendering quality or training benefit from a file
  inventory or a CPU fixture test. Preserve failures and do not resubmit unknown effects.

The legacy dual-loop bootstrap and shell runner are retired. For current scene
configuration use `docs/NATIVE_SCENE_WORKFLOW.md`; for model contracts use
`docs/LOCAL_VISUAL_MODELS.md`.
