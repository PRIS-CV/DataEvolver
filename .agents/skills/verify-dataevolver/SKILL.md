---
name: verify-dataevolver
description: Inspect DataEvolver source contracts and drive its public Harness CLI, scene compilation and native receipt recovery on an authorized execution host.
---

# Verify DataEvolver

Status: draft runtime scenarios. Source checks have run; CLI journeys, model
inference, Blender and UE5 have not run for this integration. The user requested
no tests during its preparation. Never promote that omission to a pass.

## Launch and doctor

Use Python 3.12 for the source checker. From the repository root:

```bash
python .agents/skills/verify-dataevolver/scripts/check_repository.py
git status --short --branch
git rev-parse HEAD
```

This reads the public source inventory and Python/JSON/TOML files. It checks
private paths/credential patterns, internal module paths, public entry points
and optional Blender launch flags; it does not import the application,
load models, create campaigns or establish runtime readiness.

Before runtime verification, obtain authorization for that scope. On the chosen
host install the editable checkout in its isolated environment:

```bash
python -m pip install -e '.[harness,test]'
python -m dataevolver --help
python -m dataevolver operator-sdk catalog
```

Both commands must exit zero before driving a campaign. Record the actual Python,
Git commit, dirty state, output and exit codes. For GPU work inspect host GPU
memory, utilization and processes, and select the frozen UUID before execution.
Use the user's existing host and budget; preserve other sessions and processes.

## Drive

The maintained scenarios are in `scenarios/README.md`. Use `compile-plan` for
data plans and `worker-compile` for configured native processes. Both reach the same
`create`, `step`, `status` and `replay` commands. Model inspection belongs to the
existing `visual-models` CLI; it does not prove inference readiness.

Never use the retired production or standalone multimodal CLIs. The retained
multimodal `run()` library serves registered operators. Do not invent a second
execution, budget or retry loop in this skill.

## Evidence

Store each authorized verification under an unused
`.dataevolver/local/verification/<campaign-id>/`. Preserve the command transcript,
exit codes, prepared goal, campaign events and source/model bindings. For native
work preserve `request.json`, `intent.json`, `worker.log`, `exit.json`,
`result.json`, `native.json` and the committed workload receipt. UE5 also needs its
render index, batch contract, task receipts, native frames and decode evidence.

Check the resulting state and re-read the output hashes. A process exit or a
completion label alone is insufficient. `quality_verified=false` remains false;
execution integrity does not prove mesh quality, realism or training benefit.

## Cleanup

The CLI has no persistent verification service to stop. Stop only a process or
tmux job that this verification created, using its recorded identity. Preserve
failed or ambiguous native attempts and all evidence. Never delete an intent to
make a retry pass; reconcile unknown effects first. New code or model bindings
require a new campaign, leaving historical hashes unchanged.
