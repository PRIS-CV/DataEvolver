# Image collection and layout curation

This route prepares two independent Harness campaigns. Collection emits a portable
`SourceBundle.v1`. Curation consumes its `ArtifactRef` and emits `curated_images.v2`.
The existing `data.image_curation@2` CPU route and v1 loader keep their behavior.

The current evidence is **mock_verified**. Run synthetic fixtures on a development
computer. Run real acquisition, model serving and image processing on the new Linux
server. The operators reject real execution on macOS.

## Offline development

Use Python 3.12 and an isolated environment:

```bash
python -m pip install -e ".[image-data,test]"
python -m dataevolver.tools.image_pipeline_fixture \
  --root .dataevolver/image-pipeline-fixture
```

The fixture command creates synthetic source tables and images. It exercises
`plan`, `evolve --dry-run`, `resume`, `status` and `replay` for both campaigns.
It records the calls in `harness_calls.jsonl` and returns the source and export
references in `fixture_result.json`. The default fixture has eight source images
and one cross-source duplicate. The resulting export has seven mock records.

Use a fresh runtime root for a new run. Resume the exact existing campaign when
recovering an interruption. Do not edit a frozen goal or regenerate its inputs.

```bash
python -m dataevolver.harness.cli plan --goal <goal.json>
python -m dataevolver.dataset.curation_loader <export-directory>
python -m pytest -q tests/test_image_pipeline.py tests/test_data_curation.py \
  tests/test_image_recipe.py tests/test_visual_contracts.py \
  tests/test_model_gateway.py tests/test_visual_harness.py
```

`plan` reads local contracts and reports missing store configuration. It creates
no campaign, queries no endpoint and invokes no model. A valid offline plan does
not establish server readiness.

## Collection contract

`data.image_collection@1` uses the existing `DataNodeGraph` and one
`images.collect` node. Its input is an immutable `CollectionSpec` JSON artifact.
`collection_contract()` and `image_pipeline_goal()` in
`dataevolver.harness.image_pipeline` build the existing Harness contracts.

The examples in `configs/harness/image_sources.example.json` pin the approved
COCO/VG repository revisions and cap each source at 2,500 candidates. The file
contains explicit `__SET_...__` placeholders because real fields and VG metadata
URLs have not been exercised. Resolve them against the pinned inventories on
the server. Record actual field mappings and evidence before constructing a goal.

Source entries contain:

| Field | Meaning |
| --- | --- |
| `repo_id`, `revision` | Dataset repository and exact 40-character revision |
| `tables` | Named metadata tables; each entry uses `path` for a Hub file or `url` for an external file |
| `image_files` | Image Parquet shards; an empty list uses each record's original image URL |
| `field_mapping` | Per-table raw-column to canonical-column mapping |
| `dataset_license`, `dataset_evidence` | Separate dataset declaration and hash-bound evidence |
| `origin_type`, `origin_evidence` | Image origin and evidence, independent of caption generation |
| `allowed_hosts` | Explicit hosts for external URLs and redirects |
| `quota` | Upper bound; failed downloads do not trigger replacement sources |

Fixture entries use an `artifact` reference instead of a remote `path` or `url`.
JSON containers require `key`; ZIP files require an explicit `member`. Parquet
tables are read in batches. Loading scripts are never executed.

The canonical COCO tables are `image_info`, `instances`, `captions`, `categories`
and `licenses`. Image metadata uses `id`, `width`, `height`, `license` and
`coco_url` or an explicit upstream split. Instances use `image_id`, `id`,
`category_id` and `bbox` in pixel xywh. Captions use `image_id` and `caption`.
License rows use `id` and `name`; their complete records remain in the image
license evidence. A Hub split named `train` or a bare numeric filename does not
prove `train2017`. An explicit source split override also requires split evidence.

The canonical VG tables are `image_info`, `objects`, `attributes` and
`relationships`. Nested per-image lists are supported. The normalizer preserves
object IDs, attribute owners, relationship IDs and raw edges. `coco_id` links
exclude known evaluation/test/project benchmark images and relate shared images.
VG image URLs retain their original hosts.

HF endpoint precedence is:

```text
CollectionSpec.hf_endpoint → HF_ENDPOINT → https://hf-mirror.com
```

Registration resolves the endpoint and binds it to the operator contract.
Metadata and Hub downloads use that same endpoint and exact revision. A missing
revision fails without fallback. Explicitly switching endpoint keeps the revision
and creates a new run configuration.

The downloader uses the Hub SDK cache and resume support. External URLs use
ETag-bound Range resume. Concurrency defaults to four and whole-file attempts
are capped at three. SDK transport retries remain internal to a file attempt.
`downloads.json` records per-file attempts, cache hits, received file-payload
bytes and materialized file sizes. HTTP headers and metadata API response sizes
are outside the payload counter. SDK transfer callbacks require Hub SDK 1.33 or
later. A cache hit transfers zero file-payload bytes.

The raw cache remains outside the source bundle. The completed bundle has
read-only files, embedded image bytes and original annotations in Parquet,
source evidence, the resolved acquisition specification, download records and
a complete member hash index. `load_source_bundle()` verifies it independently.
Image batches contain at most 128 records; only required image shards are used
when metadata supplies `image_shard`. Without a shard index, every configured
image shard is inspected and its full download size is counted.

## Curation contract and gates

`data.image_curation@3` has six registered nodes:

```text
layout.audit → layout.families → layout.review
             → layout.repair → layout.rereview → layout.export
```

Audit decodes original bytes, checks dimensions, preserves the original image
and annotation, applies all EXIF orientations to pixels and boxes, and exports
bounded RGB PNGs. Boxes are never silently clipped. Invalid annotations remain
visible and can enter one repair attempt.

`generic_image_v1` and `multi_subject_layout_v1` compile into
`QualityPolicy.task_criteria`. `overrides` can change quality parameters for a
new run. They cannot remove provenance, license, privacy, safety or leakage
conditions. The defaults are engineering starting values, not Boogu thresholds.

| Parameter | Default |
| --- | --- |
| Minimum original short side; maximum aspect; maximum pixels | 256 px; 4; 64 MP |
| Maximum normalized long side | 2,048 px |
| dHash distance | 4 bits, alongside exact RGB pixel hashes |
| Splits and seed | train/dev/eval = 90/5/5; 20261103 |
| Selected subjects | At least two |
| Selected box | Short side at least 32 output pixels; area at least 1% |
| Directional separation | Complete boxes separated by a normalized margin of 0.02 |

Selected object IDs define `count_scope`. Normalized xyxy boxes, unique object
IDs, attribute owners and relationship references must be valid. Category
merging uses only `category_mapping`. Verified geometric edges are `left_of`,
`right_of`, `above` and `below`. Other original edges remain unverified evidence.

Duplicate groups and related-source groups are distinct. Exact/dHash duplicate
groups select one passing representative. Related groups constrain splits and
retain distinct photos. Assignment uses a deterministic hash; small samples need
not contain all three splits or achieve exact percentages. Grouping is bounded
to 5,000 candidates and uses pairwise dHash comparisons.

Review uses a calibrated independent role. It checks subject identity, caption
faithfulness, attributes, relationships, blur, watermark, noise and exposure
separately. Caption claims must reference verified objects and agree with counts,
attributes and directional geometry. Missing calibration leaves pass proposals
unknown. A model's score alone cannot grant a pass.

The annotator repairs missing or contradictory annotations once. The reviewer
then checks the new caption and boxes independently. Continued disagreement is
quarantined. Coupled producers/reviewers cannot independently approve a repair
without separate human layout evidence. Model calls, original requests and
receipts stay under the existing gateway. An unknown timeout consumes the
reservation and is never automatically resent.

## Evidence and governance

`image_layout_profile.example.json` deliberately starts with empty license
allowlists and missing evidence. Unknown terms cannot become training candidates.
Use immutable artifacts in `DatasetIntent.sources` for:

- `calibration`: `dataevolver.layout_calibration.v1`, independent authority,
  bound reviewer/revision, `rubric_hash`, `profile_hash`, per-criterion thresholds,
  gold evidence, development IDs and separate held-out IDs;
- `protected_evidence`: `dataevolver.image_protected_evidence.v1`, independent
  authority and per-sample privacy/safety decisions with evidence references;
- `deployments`: `dataevolver.model_deployments.v1`, server deployment IDs,
  model revisions/families and matching endpoint origins.

Every decision retains pass, fail or unknown, a reason and evidence. Defects
and unresolved quality checks are quarantined. Missing protected evidence may
leave a technically readable, independently reviewed record pending.
Governance follows evidence:

| State | Meaning |
| --- | --- |
| `mock_verified` | Synthetic/mock evidence only; promotion is prohibited |
| `layout_ready` | Per-record layout checks passed; the field does not certify training benefit |
| `technical_candidate_pending_independent_review` | Layout output still lacks protected evidence |
| `training_candidate` | All declared checks and the independent package loader pass |

No route emits `train_ready`. A source may yield zero accepted records. If all
records fail, export an empty verified package with `NO_QUALIFIED_OUTPUT` and
reason counts. The execution evaluator checks pipeline completion and artifact
validity; it does not promise a retained count or training benefit.

## Portable export and server handoff

The v2 package contains JSONL records, normalized PNGs, original bytes, source
evidence, original captions/annotations, verified facts, object counts,
split/group lineage, model requests/receipts, calibration references, quarantine,
resolved filtering configuration, a data card and an independent loader receipt.
The loader checks members, hashes, image decoding, object references, geometric
relations, count scopes, group isolation and mock governance without an
experiment directory or model service.

PE-specific instructions, `rewritten_prompt` and aspect-ratio fields belong to
the later PE adapter. The current export is the common annotated intermediate.

Use [the server checklist](IMAGE_DATA_SERVER_HANDOFF.md) for the real sequence:
resource/access preflight, 50 candidates, approximately 200 stratified
development calibration samples, then a 2,000–5,000-candidate pilot. Real source
field compatibility, VG URL availability, model reliability, calibration and
training value remain unverified.
