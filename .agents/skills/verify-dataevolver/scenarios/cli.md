# CLI and registry

Goal: find the current Harness commands and registered operators.

Precondition: an authorized host with the editable checkout and Harness extras.
Cost: CPU CLI processes; no model, native engine or dataset generation.

```bash
python -m dataevolver --help
python -m dataevolver operator-sdk catalog
dataevolver --help
dataevolver-harness --help
```

Capture stdout, stderr and the exit code for each command. All must exit zero;
the two installed aliases must expose the same commands as the module entry.
Missing dependencies or an unresolved module are failures. Do not silently use
another checkout. No service cleanup is required; retain the transcripts.

Current status: not executed on this integration.
