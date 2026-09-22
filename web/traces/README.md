# Public repair traces

Bilingual subpage for the existing DataEvolver project site. Query `?lang=zh`
selects Chinese. The default public mode uses the bundled archive without a
backend; an optional read-only API is provided. Use an HTTP service for ES modules.

From the repository root: `npm ci`, then `npm run dev` or `npm run build`.
See [frontend and backend deployment](../../deploy/traces/README.md).

## Evidence policy

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
python3 web/traces/build_showcase.py --evidence-root PATH --archive-root PATH
python3 web/traces/test_showcase.py
node --check web/traces/trace.js
```

The evidence root contains the selected grounding case's metadata under
`grounding/`. The archive root contains the preserved scene frames and the
four-round `t2i_constraint_loop_hard_glass_antireg_20260622_142826` experiment.
Exports are deterministic for unchanged sources. The npm build retains the exact
original frames and creates `dist/` for the existing GitHub Pages deployment.
It does not rerun experiments. API and data-source checks run with `npm test`.
