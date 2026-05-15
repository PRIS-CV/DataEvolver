# ARIS Project And Image Notes

## Project Snapshot

ARIS currently has two connected lines of work:

1. Stage0 paper-search / method-synthesis workflow.
2. Dual-object scene placement and VLM-guided visual optimization pipeline.

The Stage0 work is used as an upstream method discovery layer. It converts a research task into structured paper evidence, a synthesized method prior, and a downstream reproduction prompt.

The dual-object pipeline focuses on placing two mesh objects into selected scenes, rendering candidates, and using VLM feedback to adjust placement, camera, lighting, scale, material visibility, and contact quality.

## Stage0 Demo

The preserved demo is under:

```text
stage0_demo/
```

It uses multiple agentic / test-time scaling / self-improvement papers to derive a dataset construction method for ARIS.

Core output:

```text
generate -> validate -> critique -> repair -> distill -> gate -> promote
```

Interpretation:

- `generate`: create multiple candidates rather than one sample.
- `validate`: run cheap checks, VLM scoring, rules, or executable validators.
- `critique`: produce explicit failure analysis.
- `repair`: apply targeted corrections.
- `distill`: compress useful trajectories while preserving state, action, outcome, and failure context.
- `gate`: promote only samples that satisfy quality, diversity, and provenance requirements.
- `promote`: move accepted samples into the next pipeline stage.

## Visual Result Notes

### 01_outdoor4_best.png

Representative high-quality outdoor result. This scene has relatively stable lighting, object visibility, and spatial grounding. It is useful as a positive reference for the current dual-object pipeline.

### 02_outdoor5_best.png

Selected result showing usable object placement and scene integration. It is useful for comparing object scale and camera framing across outdoor scenes.

### 03_outdoor6_best.png

Selected from the outdoor6 runs after manual/parameter corrections. This scene was difficult because terrain height and object-ground contact were unstable, so this image should be treated as a usable but sensitive reference.

### 04_outdoor7_best.png

Selected outdoor7 result. This scene previously showed background edge distortion, so it is mainly useful for reviewing whether the current crop/camera configuration remains acceptable.

### 05_outdoor11_best.png

Selected outdoor11 result. This scene was affected by camera/scene-angle variance in earlier runs, so the image is useful as a reference for stable placement-state reuse.

### 06_fix_round00_vs_best.jpg

Comparison sheet showing initial output versus optimized/best result after fixes. Use it to explain the effect of VLM-guided adjustment.

### 07_old_vs_fix_best.jpg

Before/after comparison for earlier lighting/material/placement fixes. Use it to describe why normalization and stable placement are necessary.

### 08_visual_review_contact_sheet_1.jpg

Contact sheet for quick visual review across multiple outputs. Useful for scanning common failure modes rather than presenting a single final image.

### 09_group_meeting_onepage_preview.png

One-page English group-meeting preview emphasizing visual progress. This is useful as a presentation snapshot rather than a raw experiment artifact.

## Current Strengths

- The selected scene pool can produce visually usable dual-object results.
- Outdoor4 is currently the strongest visual reference.
- Stage0 can synthesize a reusable construction method from several related papers.
- The pipeline now has a clearer conceptual bridge from paper search to data construction.

## Known Limitations

- Some scenes remain sensitive to terrain height, camera angle, or background distortion.
- VLM adjustment improves outputs but still needs stronger structured logging and quality gates.
- The Stage0 demo is not yet a full end-to-end reproduction run.
- The current visual package contains selected representative outputs, not a statistically complete benchmark.

## Recommended Next Step

Turn the Stage0 synthesis into concrete ARIS pipeline fields:

- trajectory log
- critique / repair record
- validation score
- diversity tags
- promotion decision
- artifact provenance

This would connect the paper-derived method directly to dual-object dataset construction.
