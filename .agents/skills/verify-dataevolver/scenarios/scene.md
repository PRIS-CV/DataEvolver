# Native component scene

Goal: compile and execute the provided component-modeling recipe through the
public Harness. Precondition: authorization to run Blender, a compatible host
Blender installation and the editable checkout. The sample permits CPU work and
blocks network/GPU permission. It is a primitive recipe, not a realism benchmark.

Choose a new campaign ID for every changed worker or recipe. The following ID and
output directories must not already exist:

```bash
export DE_VERIFY_ROOT="$PWD/.dataevolver/local/verification"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m dataevolver scene-compile --profile configs/scene/blender.example.json \
  --output "$DE_VERIFY_ROOT/prepared/component-check" \
  --campaign component-check --campaign-root "$DE_VERIFY_ROOT/campaigns"
python -m dataevolver create --goal "$DE_VERIFY_ROOT/prepared/component-check/goal.json" \
  --campaign-root "$DE_VERIFY_ROOT/campaigns" --idempotency-key component-check
python -m dataevolver step --campaign component-check \
  --campaign-root "$DE_VERIFY_ROOT/campaigns" --max-actions 8
python -m dataevolver status --campaign component-check --campaign-root "$DE_VERIFY_ROOT/campaigns"
python -m dataevolver replay --campaign component-check --campaign-root "$DE_VERIFY_ROOT/campaigns"
```

Compilation must report no execution or inference. After execution, require a
completed campaign, matching replay state, a zero process exit, bound request and
result hashes, and nonempty declared `.blend`/GLB outputs with valid headers.
Record logs and receipts from the actual campaign; do not substitute test fixtures.
Retain the campaign after completion or failure. The worker timeout is 600 seconds;
unknown effects require reconciliation, not another `step` to repeat the effect.

UE5 requires host-specific frozen project, engine, GPU and source inputs. Prepare
its profile using `docs/NATIVE_SCENE_WORKFLOW.md` before authorizing a render;
the supplied UE5 template is not an immediately executable deployment.

Current status: not executed on this integration.
