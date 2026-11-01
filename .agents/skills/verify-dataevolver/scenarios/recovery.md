# Inspect and replay retained work

Goal: recover state from retained campaign evidence without resending an effect.
Precondition: the authorized scene scenario created `component-check` and the
original source and frozen inputs are available. Cost: CPU state reads; no model
or native process execution for these commands.

```bash
export DE_VERIFY_ROOT="$PWD/.dataevolver/local/verification"
python -m dataevolver status --campaign component-check --campaign-root "$DE_VERIFY_ROOT/campaigns"
python -m dataevolver replay --campaign component-check --campaign-root "$DE_VERIFY_ROOT/campaigns"
```

Capture both outputs and exits. Compare replayed state with current campaign
state and inspect the existing action/native receipts. These reads must not
increase operator/model call counts or create new native attempts. An intent
without verified completion is an unresolved effect; do not remove files or
execute the task to manufacture recovery evidence. Preserve all evidence.

Current status: not executed on this integration.
