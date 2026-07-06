# Universal 3D Layout Scripts

This folder contains the lightweight generation and review entry points for the
DataEvolver universal 3D layout dataset workflow.

Scripts:

- `render_dataevolver_universal_contract_record.py`: Blender-side renderer for
  one dual-object record.
- `run_dataevolver_universal_contract_batch.py`: batch launcher for the
  universal artifact contract.
- `run_universal_vlm_inner_loop.py`: selected-record VLM loop runner.
- `run_vggt_omega_geometry_review.py`: VGGT-Omega proxy geometry review hook.

See `docs/UNIVERSAL_3D_LAYOUT_DATASET.md` for the public workflow summary and
server artifact locations.
