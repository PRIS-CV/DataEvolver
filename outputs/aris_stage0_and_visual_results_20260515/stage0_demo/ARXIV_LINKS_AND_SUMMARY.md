# ArXiv Links And Summary

## Task

derive an ARIS dataset construction method from multiple agentic test-time scaling and self-improvement papers

Prior id: `20260515-agentic-dataset-construction`

## Selected arXiv Papers

| # | Paper | arXiv | Why it matters |
|---|---|---|---|
| 1 | LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling | https://arxiv.org/abs/2605.08083 | AutoTTS frames test-time scaling design as agentic discovery: sample candidate strategies, execute/evaluate them in a task environment, feed back execution traces, and iterate toward better strategies. Useful abstraction for ARIS: represent dataset construction as a search environment with actions, observations, feedback, trajectory logs, and explicit stopping criteria. |
| 2 | Decocted Experience: Reformatting Human Demonstrations for Longer-Horizon Tasks | https://arxiv.org/abs/2604.04373 | Decocted Experience compresses and reformats long demonstrations into more useful experience units for agent learning, emphasizing trajectory transformation rather than raw data accumulation. For ARIS, raw examples should be decomposed into state/action/outcome/failure annotations and then distilled into reusable construction patterns. Prefer seeds that include before/after state, action rationale, and observable consequence. |
| 3 | Scaling Test-Time Compute for Agentic Coding | https://arxiv.org/abs/2604.16529 | This work studies test-time search for coding agents, comparing ways to spend inference budget across candidate generation, validation, and selection. The useful pattern is budgeted candidate generation plus executable validation, not blind one-shot generation. Seed examples should have machine-checkable constraints where possible. Ask Stage1 to generate N candidates per item and preserve validator outputs and selection rationale. |
| 4 | Scaling Multiagent Systems with Process Rewards | https://arxiv.org/abs/2601.23228 | This work studies multiagent systems with per-action process rewards from AI feedback, addressing credit assignment and sample efficiency in expensive multiagent rollouts. For ARIS dataset construction, assign credit at the step/action level rather than only at final sample acceptance; this makes multi-role generation, critique, repair, and arbitration auditable. Seed tasks should expose intermediate decisions that can receive process feedback. |
| 5 | Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains | https://arxiv.org/abs/2501.05707 | Multiagent finetuning emphasizes diverse reasoning chains and self-improvement from multiple generated solutions or perspectives. The transferable idea is diversity before filtering: construct datasets from varied solution paths, not just the most likely path. Force variation across scenario type, reasoning path, object relation, constraints, and failure mode. Ask Stage1 to report diversity dimensions covered by each batch. |

## Links For Next Step

- https://arxiv.org/abs/2605.08083
- https://arxiv.org/abs/2604.04373
- https://arxiv.org/abs/2604.16529
- https://arxiv.org/abs/2601.23228
- https://arxiv.org/abs/2501.05707

## Summary For Next Step

Synthesis from multiple agentic/test-time/self-improvement papers suggests ARIS should build datasets through budgeted candidate generation, trajectory logging, explicit feedback/critique, process-level quality gates, diversity preservation, and provenance-aware promotion. The construction loop should generate multiple candidate samples or policies, evaluate them with cheap validators or structured AI feedback, repair or reject with recorded reasons, and only promote batches that satisfy reproducibility, diversity, and downstream utility checks.
