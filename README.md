# DataEvolver

DataEvolver is being reorganized into a clean `src/`-based Python package.

This public branch intentionally keeps only the reusable package boundary and
project metadata. Local experiments, generated reports, dataset-specific
configs, legacy pipeline scripts, website output, and agent workspace files are
not part of the public source tree.

## Source Layout

```text
src/dataevolver/
├── adapters/     # Model and external runtime adapters
├── agents/       # Agent workflows and feedback loops
├── annotation/   # VLM, geometry, and review passes
├── dataset/      # Dataset contracts, builders, and exporters
├── runtime/      # Runtime profiles, manifests, workers, observability
├── tools/        # Stateless helpers and reporting utilities
└── workflows/    # Reusable orchestration flows
```

## Repository Policy

- New reusable code should be added under `src/dataevolver/`.
- Local-only folders such as `pipeline/`, `scripts/`, `configs/`, `docs/`,
  `tests/`, `web/`, `.agents/`, and `experiments/` are ignored by default.
- Generated datasets, reports, model weights, copied upstream repositories, and
  machine-specific profiles should stay outside the public Git tree.
- If a legacy file is already tracked on GitHub and should become local-only,
  remove it from Git tracking with `git rm --cached` while preserving any local
  copy needed for migration.

## Status

This PR is a repository hygiene cleanup. It does not claim that the package API
is complete yet; the next step is to migrate stable code from local legacy
folders into the package modules above.
