# Public repair traces

Bilingual subpage for the existing DataEvolver project site. Query `?lang=zh`
selects Chinese. The default public mode uses the bundled archive without a
backend; an optional read-only API is provided. Use an HTTP service for ES modules.

From `apps/project-page/`: `npm ci`, then `npm run dev` or `npm run build`.
See [frontend and backend deployment](../../deploy/README.md).

## Evidence policy

The featured Qwen / FLUX section is a separate historical **configuration
comparison**, not a third continuous agent trace. It preserves six original
images (reference, depth, two Qwen arms, FLUX baseline/retry), model names and
source hashes in `model-cases.json`. Qwen-to-FLUX changes generators; the overall
experiment remained blocked. No per-case metric is inferred from a summary.
The read-only `/api/traces` contract still contains the original two traces;
the comparison is a bundled static asset, including when the API is connected.

To rebuild the comparison from an explicitly reviewed local archive:

```sh
node tooling/export-model-case.mjs PATH_TO_REVIEWED_ARCHIVE
```

This validates existing SHA256 records and copies original bytes. It does not
reach a server, invoke a model, edit images or expose private source paths.

- Original PNG bytes are preserved; source SHA256 hashes are included.
- Ground-contact crops are display-only SVG viewports with identical image-space
  bounds; full frames remain visible. Realized cameras differ across rounds.
- Historical gate acceptance is not an independent quality certification.
- Bottle-count recovery is a **partial repair after regression**, not a net
  improvement over the initial round. All four rounds remain available.
- Raw model transcripts and private host paths are excluded from public exports.
- The examples do not establish current Harness quality or trained-model gains.

## Rebuild and check

`build_showcase.py` requires two explicit archive roots; it never reads remote
experiments or calls models:

```sh
python3 frontend/traces/build_showcase.py --evidence-root PATH --archive-root PATH
python3 frontend/traces/test_showcase.py
node --check frontend/traces/trace.js
```

The evidence root contains the selected grounding case's metadata under
`grounding/`. The archive root contains the preserved scene frames and the
four-round `t2i_constraint_loop_hard_glass_antireg_20260622_142826` experiment.
Exports are deterministic for unchanged sources. The npm build retains the exact
original frames and creates `dist/` for the existing GitHub Pages deployment.
It does not rerun experiments. API and data-source checks run with `npm test`.
