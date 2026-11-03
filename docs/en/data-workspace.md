# Data workspace

The optional Workbench adds Collect, Curate, and Review & export tabs. It uses the existing Harness Controller and image pipelines. It does not create a separate scheduler or change quality gates.

## Prepare the host

Use the [image pipeline guide](../IMAGE_DATA_PIPELINES.md) to prepare a collection Goal or a curation Goal. The Goal contains hashed contracts and references to immutable source, model, policy, and evidence inputs. The page accepts a Goal file path on the selected execution host. It does not edit these inputs or generate new Goals.

In Workbench Settings, select the local or OpenSSH execution target and its Python environment. Configure an **absolute dataset directory** in local Paths or remote Execution settings. Package directories in the viewer are relative to that directory. Each read checks the host/path configuration shown by the requesting tab. If another tab changes it, reload Workbench before reading again. A local directory does not configure or synchronize its SSH counterpart.

Set each store environment variable named by the Goal on the execution host. Also configure model endpoint and credential variables there. Local commands inherit the Workbench process environment. SSH commands use the remote command environment; when needed, configure `remotePython` as a host-side interpreter wrapper that loads the deployment environment and forwards arguments. Workbench does not copy local credentials or data to the remote host. Annotator/reviewer ModelProfile bindings are independent of Image studio generation/editing routes.

## Collect and curate

1. Enter the existing Goal path in Collect or Curate and select **Inspect plan**. This submits `plan --goal` through the existing jobs API. The command checks the Goal offline without downloading data, inference, or creating a Campaign.
2. Read the receipt. Collection plans include the resolved source specification when store variables are available. Curation plans include resolved quality criteria. Plans report nodes, budgets, missing store variables, and blockers. A succeeded process can still report a blocked plan.
3. Set the Campaign root and an explicit idempotency key. Enable **Fixture Goal (--dry-run)** only for a fixture image Goal; keep it off for a server Goal. Select **Create from Goal**. The native Goal determines the Campaign ID; enter that same ID for subsequent controls.
4. Use **Bounded step**, **Status**, **Explain**, **Replay**, **Pause**, or **Resume execution**. Step and resume execute work. Pause follows the native checkpoint semantics; it does not prove a running remote worker stopped immediately.

Receipts retain job inputs, status and bounded log tails. The page refreshes the selected queued/running receipt. Unknown effects remain unknown; inspect the native receipts before retrying. Jobs are still managed by the existing bounded Workbench queue. `plan`, like status inspection, bypasses that execution queue. Proposal sessions retain their existing bridge restriction to status/explain/replay.

## Review and export

Enter a relative directory containing `bundle.json` or `dataset.json`. **Package summary** displays SourceBundle or curated package metadata, including member references and download information when present. **Read samples** supports curated `manifest.jsonl`, `quarantine.jsonl`, and `reviews.jsonl` tables. It reads up to 20 complete records per page with byte cursors; it does not load Parquet into the browser.

Select a record to view its caption, normalized boxes, full annotations, gates, and review evidence. Quarantine records can contain reasons without an image. The package must index each displayed image. The viewer checks that its resolved path stays inside the package and its bytes match the member SHA-256.

The viewer reads existing exports. It does not create archives, rewrite records, or grant training readiness. The viewer displays metadata and JSONL as recorded. It does not independently validate the entire package. Use the existing native loader for that check. `mock_verified`, `layout_ready`, `technical_candidate_pending_independent_review`, and `training_candidate` retain their existing meanings. None proves training benefit.

The stdlib reader runs through the same local/SSH host and Python selection as Harness jobs. The reader bounds metadata to 4 MiB, each JSONL line to 512 KiB, and each table page to about 4 MiB. Images have an 8 MiB limit. Process output has a 16 MiB limit and a 30-second timeout. Oversized or malformed inputs produce an error; the viewer does not truncate a record into valid evidence. Do not change a package while paging through it; reload the first page after replacement.

Source/profile form builders, manual annotation edits, archive downloads, and pipeline-specific node dashboards are outside this first integration. The generic Harness page remains available for full command receipts.
