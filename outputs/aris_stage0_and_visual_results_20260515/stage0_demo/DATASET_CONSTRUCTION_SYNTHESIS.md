# Dataset Construction Synthesis

Task: derive an ARIS dataset construction method from multiple agentic test-time scaling and self-improvement papers.

## Papers Reviewed

- [LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling](https://arxiv.org/abs/2605.08083)
- [Decocted Experience: Reformatting Human Demonstrations for Longer-Horizon Tasks](https://arxiv.org/abs/2604.04373)
- [Scaling Test-Time Compute for Agentic Coding](https://arxiv.org/abs/2604.16529)
- [Scaling Multiagent Systems with Process Rewards](https://arxiv.org/abs/2601.23228)
- [Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains](https://arxiv.org/abs/2501.05707)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

## Common Pattern

The papers point to one shared recipe:

1. Generate multiple candidates, not one.
2. Run cheap validation or structured feedback.
3. Keep trajectory and rejection history.
4. Distill or compress only after preserving causal decision information.
5. Preserve diversity before filtering.
6. Promote only items that pass explicit quality gates.

## Recommended ARIS Construction Method

### 1. Candidate generation

For each task, create several candidate examples or policies with different:

- scenario types
- reasoning paths
- constraint sets
- failure modes
- object relations or domain assumptions

### 2. Trajectory logging

Store per-item or per-batch trace data:

- generation prompt
- candidate set
- critique result
- repair step
- accept / reject decision
- reason for rejection

### 3. Process-level feedback

Do not score only final outputs. Add step-level feedback where possible:

- proposal quality
- constraint satisfaction
- diversity contribution
- repair usefulness
- provenance completeness

### 4. Distillation

If raw trajectories are long, compress them only after preserving:

- before / after state
- action rationale
- failure context
- outcome

### 5. Quality gate

Promote a batch only if it satisfies:

- task-grounded validation
- diversity coverage
- reproducibility trace
- provenance labeling
- budget constraints

## ARIS-Specific Output Labeling

Recommended labels:

- `official_downloaded`
- `official_result_recomputed`
- `self_generated`
- `mock_or_proxy`
- `synthetic_proxy`
- `not_completed`

## Practical Summary

The strongest synthesis is not “one better prompt.” It is a small production loop:

```text
generate -> validate -> critique -> repair -> distill -> gate -> promote
```

That is the shape ARIS should adopt for dataset construction.
