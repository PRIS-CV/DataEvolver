# Agent integration

[Home](../../README.md) · [Documentation index](../../README.md#documentation) · [简体中文](../zh-CN/agents.md)

An outer agent prepares plans and code and calls the public Harness interfaces. An internal planner provider returns proposals.

Run command examples from the repository root unless a step changes directory.

## Codex and other outer agents

**Codex has two distinct integration roles.** An outer execution-capable agent can
write a plan, author a worker or modeling recipe, compile a goal, and call the
public CLI within its authorization. It reads JSON status and receipts to decide
whether to request another step. The optional
[`CodexCLIPlannerProvider`](../../src/dataevolver/adapters/codex_planner.py) is an inner
model transport: it returns schema-conforming proposals with shell, execution and
multi-agent tools disabled. It does not run the data workflow. Other supported
providers can supply those proposals through the same model contract.

For example, Codex can prepare a Blender component recipe, submit it as a native
worker, inspect the resulting artifacts, then prepare a separate UE5 rendering
task. Modeling, scene export and rendering have separate input/output contracts.
An agent-generated plan or script becomes executable only through its configured
operator binding; the Controller remains responsible for campaign state.

## Using with Claude Code

Claude Code is an optional outer agent. It can author plans and code and call the
same Harness CLI as Codex; the framework does not require it as a planner or tool.

### Step 1: Install Claude Code

Follow the [official setup instructions](https://code.claude.com/docs/en/overview)
for your host and chosen authentication method.

### Step 2: Create your `CLAUDE.md`

The root `CLAUDE.md` is ignored. Keep private host/profile references there and
point the agent to `AGENT_SETUP.md`, the relevant workflow contract and the public
CLI. Keep credentials in the private environment, outside the instruction text.

### Step 3: Use project skills (optional)

Reuse [onboarding](../../.agents/skills/dataevolver-onboarding/SKILL.md) and
[verification](../../.agents/skills/verify-dataevolver/SKILL.md) guidance in your agent's
supported instruction format. A new skill should explain a workflow rather than
implement another Controller.

### Step 4: Launch Claude Code

From the source checkout, start your configured `claude` client and ask it to
prepare the task through `compile-plan` or `worker-compile`. Use `create`, bounded
`step` and the observation commands for execution; authorization still governs
model/native workloads.
