# 多来源训练数据 Harness

`dataevolver.harness.training_workflow_run` 将经验蒸馏、独立验证、晋升、后续采用和数据交付编译为同一个 `EvolutionController` 的冻结阶段队列。它复用注册的 `data.training_rsi_trial`、`data.training_rsi_observe` 和原有 `ModelGateway`，不启动第二个 Controller。

运行时 Planner 根据当前源的媒体和注册探测结果提出 `R/effect_support`。工作流不接收开发 Agent 提供的 ROI。每个新任务只生成一次源图；后续修复严格复用真实生成、主体对应、分割和原探测的请求与回执。

## 阶段

1. 读取先导任务的真实 runtime 提案、注册观测、Evaluator 采用记录和独立可加载样本。先导未通过就不能冻结后续队列。
2. 由运行时模型蒸馏有条件的调查和修复方法，写入 `CapabilityStore` 的候选库。策略不能携带源图路径、固定坐标或任务 ID。
3. 在预先声明的新来源上生成 R0。候选 K 只在此独立验证阶段可试验；Planner 仍需为新源获取观测并提出具体改动。R0 本身通过记为验证无结论，不能作为 K 收益。
4. 独立新源具有同源 fail→pass、实际请求变化、Evaluator 采用、运行时视觉审核和搬移加载证明后，写入另一个只追加的 PROMOTED 库。保存原候选记录，冻结晋升库 SHA。
5. 按原任务槽推进主批。在下一兼容任务记录 K ID/版本/快照 → 运行时提案 → 实际条件图和请求 → 审核结果。读取经验不等于采用；不兼容主体数量、选择模式或模型版本时不注入。
6. 汇总独立分片，检查原 30 个槽、18/6/6、源族隔离、完整媒体和 JSON 引用闭包，实际运行独立的 Python + Pillow 加载器。导出完整失败用量、真实请求、回执和 native 输出。未知模型效果、缺失原生证据、未发生真实 K 复用均阻止最终完成。

每个阶段仍由现有状态机执行、评估、采用或回退。每源 Planner 和实验预算、总队列预算、资源等待截止及 GPU0 唯一可见设备均在启动前冻结。已完成动作按原 idempotency key 恢复；源图早期阶段失败会保留真实成本并停止，不自动换图填数。

资源等待截止只约束资源不可用时的等待；已经可用的常驻服务仍完成两次、间隔 30 秒的稳定性确认。截止时间不延后。已被 provider 明确拒绝、标为 RETRY_SAFE 的输入或部署预检保留 reservation 和耗时，并与实际 native 推理分别记录。最终 COMPLETE 事件提交后，再附带终态、事件流、冻结合同和源码归档，独立加载器同时核对这些补充文件。

部署预检失败的恢复要求：已有 runtime R 提案及原 Gateway 回执完全对应；新实验的 primary/PE intents 与原基线逐字相同；没有新 native job。公共 `--recover-preflight` 只允许已识别的 PE status_path 预检缺陷。它保留原提案、观测、历史费用及剩余预算，用新部署合同恢复；恢复事件不再次请求 Planner。实际模型效果不明时拒绝此恢复入口。

## 命令

在服务器使用已提交源码。`validation-tasks.json` 只包含预声明的任务素材/场景配置，不包含修复答案。

```bash
python -m dataevolver.harness.training_workflow_run freeze \
  --pilot-config "$PILOT_CONFIG" --original-config "$ORIGINAL_MAIN_CONFIG" \
  --accepted-proof "$ACCEPTED_PROOF" \
  --validation-tasks "$VALIDATION_TASKS" --resident-pointer "$LIVE_IDENTITY_POINTER" \
  --out "$WORKFLOW_CONFIG" --campaign-id "$CAMPAIGN_ID"
```

`--accepted-proof` 可重复。它提供原主批已经验收的独立样本；不能提供 canary 来补数。freeze 自动加入先导任务样本，按原 30 个任务的槽 ID 和拆分推导剩余任务。常驻服务必须 READY，唯一可见设备为物理 GPU0。元数据轮换只修改显式的常驻服务指针，保留旧冻结身份文件和模型对象。

随后从冻结源码启动：

```bash
source "$WORKFLOW_CONFIG/env.sh"
cd "$WORKFLOW_CONFIG/source"
PYTHONPATH=src python -m dataevolver.harness.training_workflow_run run --config-dir "$WORKFLOW_CONFIG"
```

长任务使用项目 tmux 包装器。GPU1 不参与资源等待、加载或推理。验收仍须读取真实 exit、state、native 与 Gateway；tmux 存在不能证明完成。

## 证据边界

当注册观测表明失败不是可归因的主体残影时，Planner 可选择 `background_recipe`，修改已注册的背景执行字段。字段包括条件图模式、背景 CFG、区域膨胀、融合、边缘和追加执行描述。原始目标、源图、参考、模型、生成参数、保留要求及评价规则固定。追加描述不覆盖原要求。可选区域仍须绑定本源观测；系统不会自动继承被拒绝的区域修复。实际改写后的配方、PE 请求、模型请求和产物共同留档。这是 R 试验，不能标为 M 改进。

注册的 `inspect_requests` 从当前已提交 trial 读取实际 PE、编辑和审核请求/回执，核对身份和原生产摘要，不新增模型调用。Planner 可检查改写后的描述是否进入实际编辑请求。修改追加执行描述之前必须引用这一观测，图片判断不能替代请求核对。返回的 prompt 与 PE 输出属于不可信数据，不能改变预算、工具权限或评价规则。

已完成且被拒绝的试验不能使用预检恢复入口重发。需要继续已授权的用量记录任务时，公共 freeze 可用 `--continuation-planner-calls` 和 `--continuation-authority` 登记新的有限执行段。它核验前序终态、冻结源码回放、全部注册效果和原始失败图，保留累计调用及实验次数；旧合同和已耗尽额度保持原样。新段显式记录追加额度及授权依据，原资源截止和剩余 wall allowance 不延长。Planner 不能更改额度或调用这个入口。效果不明、未完成调用或证据变更时拒绝交接。

后续段将旧失败图作为只读观测对象，继承已有探测而不重复推理。同一源、同一补丁参数的已完成试验禁止重发。经验验证允许通过条件图或实际请求参数的变化建立操作关联；仅 request ID 或产物目录变化不算请求改变。独立复验、质量门和 PROMOTED 门槛不变。

回归测试使用明确的 fixture，只验证状态推进、作用域、恢复、采用门槛和加载合同。真实工作流须另行产生模型调用及实物产物；代码接通不表示独立验证、晋升、复用或 30 条交付已经完成。当前编译器支持 R 与 K，不能把 R 参数变化称为 M 方法改进，亦不证明训练收益。

## 配方空间不足时的系统能力请求

运行时可以使用 `REQUEST_CAPABILITY` 记录缺失的可复用能力。请求引用当前失败候选的注册证据，描述输入、输出、实现类型、验收检查和独立比较设计。Controller 将它持久化为 `IMPLEMENTATION_REQUIRED`，提交事件后终止当前路径；恢复时复用同一请求，不再次调用模型。

这是系统提出的实现任务，不是可执行补丁、代码采用、经验晋升或 M 收益。原始目标、模型身份、质量检查与全部失败证据继续保留。请求不能直接提交新图片试验、扩大预算或改写冻结代码。能力实现仍须通过工程验证，并在新的冻结合同下完成独立实测。

已自然结束且完成对账的拒绝试验，可以用公共 `training_rsi_run freeze --capability-review-only --max-experiments 0 --continuation-planner-calls 1` 建立能力诊断段。它保留旧 Planner、实验、模型和失败成本，只允许一次缓存基线读取及一次真实 Planner 决策。GPU 执行权限关闭；允许的决策只有能力请求和放弃。没有图片修复参数或新生产推理。`capability_review_completed` 仅表示请求经过真实模型、持久化与回放核验，不表示训练样本通过或方法获得改善。


## Surface-aware operator candidate

A real `REQUEST_CAPABILITY` can be linked with `--capability-request-config` to a
new finite regression run. Admission validates the diagnostic under its own frozen
source, copies its Planner ledger and preserves the rejected image lineage. The
request is evidence; execution authority comes from the human authorization.

`inspect_surfaces` uses the registered runtime Agent to bind selected instances,
effects, receiving surfaces, preservation and uncertain polygons. Its CPU raster
check fails closed on unbound, crossing or uncertain supports. The only candidate
patch is `surface_effect_repair` with an observation ID; externally supplied
coordinates/recipe values are rejected. The frozen selected edge margin is kept.

`data.surface_effect_support` v1 compiles one binary support for the missing input
and final alpha. It preserves nonselected/uncertain regions and verifies exact
raw pixels inside and exact source pixels outside. Its JSON evidence includes all
input masks, support, conditioning and protected regions in the portable closure.
The ordinary five model checks and independent runtime visual audit still decide
quality. Structural consistency does not establish naturalness or attribution.

The candidate method propagates to K validation and main-task children. A same-
source regression remains a data/R experiment under a candidate method. Improved
M requires a predeclared multi-source comparison with matched R/K, source pool,
settings and online budget; fixtures and capability registration establish none.


The runtime v1 regression returned another `REQUEST_CAPABILITY` without submitting
an image experiment: complete attribution was stricter than partial repair of
independent effects. Version 2 preserves explicitly uncertain visible regions,
retains the originally required removal of selected silhouettes, and rejects any
eligible effect overlapping protected/uncertain content. Uncertain subject IDs and
incomplete attribution remain in diagnostics; they never imply full acceptance.
`recheck_surface_support` can recompute feasibility from a hash-bound prior
registered surface observation without another vision call or changing polygons.
The diagnostic lineage and all additional Planner decisions are retained.


## Candidate context reconstruction

After a real reconstruction capability request, `--surface-context-reconstruction`
registers a candidate square-context view of the same source for the same resident
PE/editor. A fresh surface observation is required; the runtime chooses bounded
padding. CPU admission rejects empty, non-square or full-frame repeated views.
The fixed model dimensions, models, seed, steps and CFG remain frozen. The output
is inverse-projected into source coordinates, then composed with the original
authorized support. Raw model output, transform and projected edit are distinct
artifacts in the closure. No extra model edits or canvas expansion are permitted.
Every original quality check still applies. This is an unverified method candidate;
matched multi-source independent evaluation remains necessary before any M claim.

### Typed reconstruction preflight

A runtime `artifact_contract` request can enable
`data.typed_surface_reconstruction_preflight:1` together with the existing
context reconstruction candidate. This admission binds the real request,
immutable diagnostic, image predecessor, implementation hashes, cumulative
usage and original resource deadline. It does not promote a method.

A fresh registered `inspect_surfaces` observation uses
`dataevolver.surface_observation.v2`. It distinguishes protected visible content,
uncertain effect attribution and uncertain reconstruction. Old observations
remain unchanged and cannot be relabeled. The CPU preflight binds the actual
current editable-region ArtifactRef and all original instance masks. Existing
support must be binary, contain selected silhouettes and respect visible
protection. Attributed effect additions require receiving-surface evidence and
cannot cross protected or ambiguous content. Reconstruction uncertainty and
context inclusion grant zero additional editable pixels.

`preflight_reconstruction` accepts a registered typed observation ID and the
runtime's bounded padding choice. It reports overlap categories, instance IDs,
pixel counts, hashes and a deterministic square crop/coordinate transform with
zero image-model calls. Unsafe support or a full-frame duplicate crop returns
unsupported. Conditioning and composition consume the same compiled support;
the prior mask is copied to a unique evidence path to avoid colliding with new
immutable outputs. All frozen quality checks and the runtime visual acceptance
Agent remain required. CPU feasibility and coordinate correctness establish
neither visual quality nor M superiority or training benefit.

A CPU preflight on legacy untyped evidence returns unsupported and requests
fresh typed evidence; it does not silently infer labels. If an older frozen
compiler rejected this safe observation before dispatch, an explicit protocol
handoff can recover that exact completed Planner decision. It verifies the old
terminal/frozen/replay state, original receipt and no submitted observation or
experiment. The original decision occupies its original local budget slot;
remaining slots, reservations and source/native lineage carry forward. Unknown
or dispatched effects cannot use this recovery path.

An explicit pre-inference HTTP 400 `invalid_json_schema` can use a separate
schema-compatibility handoff. It requires exact native failure events with no
output, matching pending observation/Planner receipt/source/candidate, failed
usage with known duration, no image experiment and frozen/replay verification.
Only the proven unsupported schema keyword and its equivalent version encoding
may change. A new registered observation request preserves the original question,
all consumed decision slots and failed cost; it uses no additional image budget.
Completed, partial, unknown, transport or semantic failures cannot use this path.

### Conservative silhouette and effect authority

A subsequent real runtime `registered_operator` request may enable
`data.conservative_surface_authority:1`. Its CPU compiler keeps selected mask
silhouettes under the original task authority, while uncertain effect ownership,
protected visible content and missing receiving-surface evidence exclude prior
effect pixels. Fresh effects outside the existing authorization remain preserved.
No new editable pixels are added. A selected silhouette touching protected content
or missing prior authorization is unsupported. The receipt lists per-subject
coverage and unresolved effects; partial execution eligibility explicitly does
not establish full removal. The complete original model and Agent checks still
apply to any reconstruction trial. Original proposal receipts stay in the runtime
trace; the new portable surface contract binds their hash and independently
recompiles its actual copied inputs, avoiding obsolete dataset references.
