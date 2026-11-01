# Bounded image curation

The `data.image_curation` Operator executes a mixed data/model graph under the existing `EvolutionController`. Versioned model and dataset contracts and the `visual.contract_mock` route retain their existing behavior.

## Contracts and authority

- `DataInput` v2 binds an immutable `ArtifactRef` or a named upstream output.
- `DataNodeSpec` v2 declares a registered Operator, revision, kind, input ports, parameters and optional model role/call/image allowance. Pure data nodes have no model role or model allowance.
- `DataNodeGraph` v2 validates dependencies, cycles, output ports, registered steps and declared budgets before execution.
- `DataNodeRun` v2 binds inputs, outputs, the implementation hash and an Operator receipt. Model receipt hashes are optional and appear only for model nodes.
- `DataRunBudget` v2 reserves child Operator attempts, audited input units and dataset output bytes in a durable journal. One wall-time deadline survives resume. Failed attempts consume their reservations.
- `ModelGateway` keeps one shared model-attempt journal for the graph. Model failure and recovery never create a fresh call budget.

The image-curation CLI route permits one graph per campaign, one round, zero experiments and no interventions. Controller `max_operator_calls=1` counts this composite call. The graph's `max_operator_calls` counts child steps and retries; its model limit counts actual provider attempts. These are distinct units. Resume completes the same pending call.

Controller owns campaign states, permissions, the frozen Bundle, evaluation and terminal decisions. Steps implement the existing `execute/recover/verify_effect` protocol. They cannot adopt recipes or change campaign state.

## Registered steps

| Operator | Kind | Input → output | Behavior |
| --- | --- | --- | --- |
| `images.audit` | data | sources → catalog | Decode a bounded intake, record source/version/license evidence, exact pixel hashes and dHash. Original sources remain read-only. |
| `images.families` | data | catalog → catalog | Join cross-source exact/near duplicates and declared source families. |
| `images.split` | data | catalog → catalog | Assign family-disjoint train/dev/eval partitions before selection or derivatives. |
| `images.select` | data | catalog → catalog | Apply source quotas, cover available partitions and retain one member per duplicate family. |
| `images.clean` | data | catalog → catalog | Apply EXIF orientation, RGB conversion, bounded resizing and PNG encoding; preserve parent/family/split lineage. |
| `images.score` | model | catalog → catalog | Run one bounded image/text scoring batch through ModelGateway; attach uncalibrated semantic evidence. |
| `images.export` | data | catalog → dataset + report | Apply the declared semantic filter and export self-contained images, manifest, evidence, quarantine and data card. |

Operators accept only their declared parameters. Arbitrary code, shell and external row file paths are not node parameters. JSON catalogs are bounded in size. A node writes only to the injected dataset store and its approved runtime directory.

## Source catalog

`dataevolver.image_sources.v1` contains an explicit `sources` list. Each source declares:

- `source_id`, dataset identity, exact revision, upstream `source_split`;
- `file` as an `ArtifactRef` and a policy-approved `license`;
- `license_evidence` as immutable text/card references;
- `image_column`, `label_column`, `family_column`;
- optional `format` (`parquet` or `zip_indexed`) and an `index` reference.

Parquet input requires embedded image bytes. An optional `dataevolver.source_index.v1` lists unique row offsets. Archive input requires that index with `archive_path`, `label` and `family` rows. Members are read without extracting archive paths to the filesystem. Only upstream training data enters this image-curation route.

The supplied source index is a frozen intake scope, not a new evaluation set. All reported input counts refer to audited image units in that scope. Original dataset sizes and coverage limitations belong in the experiment's source report.

Duplicate grouping combines exact decoded pixels, dHash distance and available source families. Semantic duplicate detection is not implemented. Unknown physical subjects cannot be inferred from a perceptual hash; record this limitation for sources without family metadata. These partitions are engineering slices, not sealed benchmarks.

## Offline local model

`ModelProfile.backend=local_python` uses the registered `clip_cpu` worker. A worker config names `model_path_env` and `python_env`, limits threads/images, and binds `file_hashes` for model/processor files. Optional `runtime_versions` bind Python and package versions. Actual runtime versions are retained in the native model receipt.

Heavy model imports and loading occur only in a child process after gateway cache checks. The worker verifies the file inventory, uses local files only, sets `CUDA_VISIBLE_DEVICES=""`, and runs with an explicit subprocess timeout. It returns image/text cosine similarities and measured image/token use. JSON serialization is not reported as generated language tokens.

Native request receipts support recovery after inference completes but before the gateway receipt commits. An unknown timeout remains `AMBIGUOUS_EFFECT` and is not automatically reissued. A complete graph cache hit performs zero worker launches or model loads.

The image-curation route is offline CPU execution. Existing endpoint and external API interfaces remain in ModelGateway; external API dispatch stays disabled. It does not start a GPU service.

## Configuration and execution

Use an experiment-owned GoalSpec and explicit artifact-store environment references. The contract has these keys: `intent`, `quality_policy`, `graph`, `budget`, `stores`, `dataset_store`, `models`. Pure data graphs can use `models={}` and zero model budget. A model graph supplies `profiles`, `policy` and registered `workers` under `models`.

The writable dataset store, read-only source stores and campaign runtime root must be disjoint. Model paths, interpreters and roots are configuration values; reusable source contains no server paths or dataset IDs.

```bash
# Run on the authorized server after configuration and resource checks.
python -m dataevolver.harness.cli evolve --goal "$GOAL" \
  --campaign-root "$CAMPAIGNS" --idempotency-key "$RUN_KEY"
python -m dataevolver.harness.cli resume --campaign "$CAMPAIGN_ID" \
  --campaign-root "$CAMPAIGNS"
python -m dataevolver.harness.cli replay --campaign "$CAMPAIGN_ID" \
  --campaign-root "$CAMPAIGNS"
python -m dataevolver.dataset.curation_loader "$EXPORT_ROOT"
```

This route requires `objective.real_data_execution=true`, `allow_gpu=false`, `allow_network=false`, and an explicit allowed Operator. Do not pass `--dry-run` for real data processing. Prepare dependencies and model files in isolated server environments before creating the campaign. Optional dependencies are grouped as `curation` and `vision`; reuse a compatible existing PyTorch installation.

## Recovery and output meaning

Child intents, effects, Operator receipts and NodeRuns remain in the campaign runtime. Dataset assets/catalogs/exports remain in the injected dataset store. Before reusing a step, verify its identity, implementation, input/output hashes and any model receipts. Missing budget records, modified artifacts or ambiguous model effects fail closed. A lost NodeRun can be reconstructed from a verified step receipt; the step is not executed again.

Budget exhaustion, invalid data effects and typed model failures produce failed Operator receipts. Controller records the failed stage and reaches a terminal state under the frozen campaign budget. Reopening a failed call retains its receipt and reservations. An unconfirmed model result is not an invitation to send another inference request.

The standalone loader validates the exported members, image hashes, counts, source/license identity, unique families and lineage/split bindings. It reads no experiment directory, source archive, model cache or live service. Export metadata itself remains hash-bound by the Harness artifact reference.

Exports have governance `technical_candidate_pending_independent_review`. Deterministic integrity/provenance/license/duplicate/leakage evidence and uncalibrated model scores are retained separately. Unknown privacy/safety stays unknown. Route completion establishes processing and loading reliability; independent quality calibration, dataset promotion, RSI, training and downstream benefit require their corresponding later evidence.
