# Optional visual model adapters

These optional implementations are selected by explicit deployment profiles.
The base ModelGateway accepts the host's model and provider bindings. The identities
below constrain these adapters; they are not framework-wide model defaults.

The `visual_generation` domain compiles through `compile_plan` to the existing
`data.graph@1` operator. Its model steps use the existing `ModelGateway`, attempt
ledger, native receipts, artifact store and recovery protocol. Registering these
steps does not establish model quality, downstream training benefit or an RSI
method improvement.

## Exact models and precision

| Component | Registered roles | Precision implementation |
| --- | --- | --- |
| Qwen/Qwen-Image-2.1 | `visual.image.generate`, `visual.image.edit` | TorchAO dynamic FP8 activations/weights in the image transformer; other components retain released BF16/FP32 precision |
| Qwen/Qwen-Image-2.1-PE-T2I | `visual.prompt.t2i` | TorchAO dynamic FP8 language linear layers; vision encoder and output head excluded |
| Qwen/Qwen-Image-2.1-PE-I2I | `visual.prompt.i2i` | Same mixed precision; 1–10 ordered reference images |
| Qwen/Qwen3.8-27B | `visual.inspect`, `visual.inspect.video`, `visual.review.image`, `visual.review.video` | Same mixed precision; video review observes first, middle and last frames |
| SAM/sam3 | `visual.segment.image`, `visual.segment.video` | FP32 detector/tracker; this implementation makes no FP8 SAM claim |
| lightx2v/Minimax-h3-Turbo | `visual.video.generate` | LightX2V `fp8-torchao` DiT, BF16 LoRA/activations, BF16/FP32 sensitive weights |

PE components rewrite prompts; they do not render final images. Their root
`config.json` selects the language architecture, including when their directory
also contains an unrelated image `model_index.json`.

H3 requires a MiniMax-H3 base directory, the exact released 8-step Turbo LoRA,
derived FP8 DiT weights, an independently validated schedule-matched AdaLN cache
and the pinned LightX2V source revision. A provider uses one initial image,
8 denoising evaluations and 24 FPS. Text-to-video is composed as image generation
followed by this image-to-video operator. Reference-video conditioning is outside
this adapter's contract. It never imports an experiment's launcher or benchmark
answers, builds an alternative quantizer, or silently substitutes another model.

## Inventory and deployment freeze

Run asset inspection on the runtime host:

```bash
python -m dataevolver.harness.cli visual-models --kind pe_t2i --root "$PE_T2I_ROOT" \
  --hash-weights --output "$RUN_ROOT/pe-t2i.assets.json"
```

Without `--hash-weights`, inspection verifies architecture, component directories
and indexed shard membership, and hashes configuration/processor assets. With
that flag it also hashes every selected weight file. Both modes explicitly return
`inference_performed=false` and `execution_ready=false`.

Each `models.workers[profile_id]` has the following fields:

```python
worker = {
    "schema_version": "dataevolver.local_visual_worker.v1",
    "kind": "pe_t2i", "model_path_env": "PE_T2I_ROOT", "python_env": "VISUAL_PYTHON",
    "file_hashes": inventory["file_hashes"],   # requires weights_hashed=True
    "runtime_versions": runtime_versions,     # exact Python/package versions
    "implementation_sha256": local_visual_worker_sha256,
    "binding_sha256": visual_models_sha256,
    "transport_sha256": training_media_worker_sha256,
    "precision": "torchao-fp8-dynamic", "execution_device": "cuda",
    "gpu_uuid": discovered_gpu_uuid,
    "threads": 4, "max_images": 1, "image_side": 1024,
    "max_input_tokens": 4096, "max_frames": 124,
    "resource_profile": measured_deployment_resource_profile,
    "dependencies": {},
}
```

These values illustrate the schema, not measured capacity. Probe the GPU, runtime
and available memory before freezing a deployment. `resource_profile` contains
`min_free_mib`, `max_utilization`, `cpu_offload` and an explicit `estimate_basis`.
The provider requires `CUDA_VISIBLE_DEVICES` to equal the frozen physical UUID.
It rechecks load before execution and records UUID, process mapping and peak GPU
allocation. CPU is used for inspection, hashing, orchestration and H3 offloading;
this FP8 transport executes model kernels on one mapped GPU. GPU0 and GPU1 can host
separate frozen deployments.

Use `ModelProfile(..., backend="local_python", model=MODELS[kind],
revision="sha256:" + payload_sha256(worker), capabilities=("json", "vision"))`.
The revision binds assets, dependency hashes, precision, implementation, GPU and
limits together. Changing precision cannot reuse a receipt for another frozen
profile. Source/model paths remain environment references.

H3 `dependencies` has exactly `lora`, `fp8`, `backend` and `adaln`. Each dependency
contains `path_env` and `file_hashes` relative to that directory. Freeze the released
LoRA, derived FP8 file, all imported `lightx2v/**/*.py` sources, and the AdaLN
manifest/tables. Its runtime requires the existing pinned LightX2V environment;
the backend label `fp8-torchao` does not imply a separate `torchao` package.
Video decoding/encoding requires PyAV in the applicable worker environment.
Missing packages or unavailable precision kernels cause a failure; there is no
automatic BF16 fallback. The worker inspects inner tensor storage before claiming
FP8 and records actual H3 weight types, dynamic LoRA branches and NFE.

## Graph composition and provenance

The shared registry exposes eleven versioned steps:

```text
prompt -> visual.prompt.t2i -> visual.image.generate -> media
media + edit prompt -> visual.prompt.i2i -> visual.image.edit -> media
media + motion prompt -> visual.video.generate -> video
media -> visual.observe.image/video -> observation
media + observation -> visual.segment.image/video -> masks
media -> visual.review.image/video -> review + FeedbackSignal
```

PE steps are optional. A plain prompt artifact contains exactly `{"prompt":"..."}`.
Generation/edit/video steps have `prompt` and, where applicable, `media` input ports.
The observation steps have a `media` input; segmentation has `media` and
`observation`. A result envelope contains the exact request, receipt hash and
confirmed output. PE's output port is `prompt`, generation's is `media`,
observation's is `observation`, segmentation's is `masks`.

Each model node declares exactly one call, its exact role and its media budget.
Common parameters are `seed`, `max_tokens` and `timeout_seconds`. Generation adds
`sampling`; observation adds a nonempty unique `subject_ids` list. Image sampling
contains `width`, `height`, `steps`, `true_cfg_scale` and optional `negative_prompt`.
Video sampling contains `width`, `height`, `num_frames`, `steps=8`, `fps=24`.
The graph and gateway budgets remain explicit; compiling a graph grants no new
online window, candidate allowance, model calls or policy privileges.

Quality review adds an explicit unique `dimensions` list. It returns the native
review envelope and a `sample_quality` FeedbackSignal bound to that review and the
exact media version. Failures and uncertain judgments are retained. A passing
summary with a failed/uncertain criterion is rejected. The signal keeps
`calibration_verified=false` and `uncertainty=unknown`, including when a model
judgment passes. Independent model review is not automatically objective ground
truth. The trusted evaluator and existing RSI policy continue to decide whether
this evidence supports a change; the feedback step grants no adoption authority.

Before segmentation, the step recovers the original generation and observation
receipts without issuing new requests. It checks the media hash, generation
request hash, actual execution prompt, observed subject IDs and unique instance
IDs. Failed/uncertain observations, another image/video, a changed prompt or a
forged output cannot authorize segmentation. PE-I2I results must refer to the
same image being edited. Generated instructions are context rather than visual
ground truth.

Image SAM uses observed descriptions and boxes with mask/box correspondence
checks. Video SAM propagates observed text prompts across every bounded frame;
missing/ambiguous identities, identity changes, shared instances or missing frames
reject the result. Its text-only tracking does not establish fine-grained identity
for indistinguishable objects. Sampled video observations do not establish full
temporal quality. Both paths preserve these limits for later quality feedback.

The existing image/catalog sampling and `images.families` deduplication remain the
shared dataset-level quality operators. These adapters are engineering additions;
they are not counted as automatically evolved M candidates. Historical H3 native
receipts and new DataEvolver receipts must be reported separately. Real FP8 model
acceptance, quality calibration and training benefit require their own evidence.

Image editing also accepts 1–10 ordered references. The frozen worker capacity `max_images` may impose a lower deployment limit. Qwen Image 2.1 generation and editing both use `QwenImage21Pipeline`; Qwen3.8 and PE use the checkpoint-declared `qwen3_5` architecture.
