# Mac Migration Notes

This branch is a sanitized migration snapshot for continuing ARIS dual-object pipeline development on macOS.

## What Is Included

- `pipeline/`: core ARIS pipeline code, including dual-object rendering and VLM gate modules.
- `pipeline/dual_vlm_gate/`: dual-object VLM loop, scene pool, action space, selected placements, and notes.
- `scripts/`: supporting build, render, Stage0, and dataset construction scripts.
- `configs/`: action spaces, scene pools, scene templates, and VLM review schema.
- `docs/` and `runtime_prompts/`: Stage0 and reproduction workflow documentation.
- `outputs/aris_stage0_and_visual_results_20260515/`: compact Stage0 demo and selected visualization results.

## What Is Excluded

- local `runtime/` data
- full `outputs/` except the compact handoff package
- `gallery/`
- `objects/`
- `.blend` scene files
- generated images outside the curated handoff package
- model checkpoints and caches
- local SSH aliases, server paths, API keys, and personal absolute paths

## macOS Setup Sketch

```bash
git clone -b codex/macos-dual-pipeline-migration https://github.com/PRIS-CV/DataEvolver.git
cd DataEvolver
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install the project dependencies according to the specific stage you want to run. Blender-dependent stages require a local Blender installation and may need `BLENDER_BIN`:

```bash
export BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender
```

API-driven stages should read credentials from environment variables only, for example:

```bash
export STAGE1_API_KEY=<your-key>
```

Do not commit local keys, server aliases, runtime outputs, or downloaded model/data artifacts.

## Current Migration Caveat

This is a code migration snapshot, not a complete runnable data bundle. To reproduce rendering on macOS, provide local mesh assets, scene `.blend` files, and model/runtime dependencies separately.
