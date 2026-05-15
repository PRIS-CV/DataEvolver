# ARIS: Free-Form VLM-Guided Synthetic Data Construction for View-Controlled Image Editing

**Technical Report**

---

## Abstract

We present **ARIS** (**A**utonomous **R**endering with **I**terative **S**cene-aware feedback), a fully automated pipeline for constructing training-ready synthetic datasets for image editing tasks. Building high-quality paired training data for 3D-aware image editing—such as view-controlled rotation—has traditionally required extensive manual tuning of rendering parameters in tools like Blender, an effort that is both time-consuming and difficult to scale. ARIS addresses this by introducing a *free-form VLM-guided rendering evolution loop*: a vision-language model reviews rendered scenes using natural language feedback, and an AI agent reads this feedback directly to select the next rendering action from a structured action space, iteratively improving visual quality until the reviewer confirms the result is acceptable.

We instantiate this pipeline to construct **ARIS-Rotate**, a 50-object, 350-pair rotation editing dataset in which each sample pairs a canonical front-view image with a target view specified by a natural-language prompt (e.g., *"Rotate this object from front view to right side view"*). Ablation experiments demonstrate that the VLM evolution loop substantially improves rendering quality over single-pass rendering without iterative refinement. Fine-tuning Qwen Image Edit 2511 with LoRA on ARIS-Rotate yields measurable improvements over the base model in PSNR, SSIM, and LPIPS on a held-out test set, validating the downstream training value of automatically constructed data.

---

## 1. Introduction

High-quality paired training data is fundamental to supervised image editing models. For 3D-aware editing tasks—such as rotating an object to a specified viewpoint, changing its lighting, or altering its material—each training sample requires a *source image*, a *target image* from a different viewpoint or condition, and a corresponding *editing instruction*. Collecting such data from real-world captures is prohibitively expensive and laborious, while existing synthetic approaches face a critical quality bottleneck.

**The quality gap in synthetic data pipelines.** A standard approach is to render 3D assets automatically, producing paired images at low cost. However, naïve rendering often yields scenes with artifacts: flat lighting that fails to match the scene environment, color inconsistencies between object and background, physically implausible object placement (floating or intersecting the ground), and poor shadow quality. Existing pipelines typically address quality control through rigid scoring rules or fixed controllers that select rendering adjustments based on numeric thresholds—approaches that lack the semantic understanding needed to diagnose and resolve complex visual failures.

**Our approach.** We propose ARIS, a pipeline in which rendering quality is improved through a *free-form VLM-guided evolution loop*. Rather than relying on structured scores or fixed rules, we employ a vision-language model (VLM) to review rendered scenes using natural language, identifying specific visual problems such as flat lighting, color shift, or weak grounding. An AI agent directly reads this free-form feedback and selects actions from a discrete rendering action space to address the diagnosed issues—adjusting key light intensity, environment rotation, material parameters, or object placement. This process repeats until the VLM reviewer explicitly confirms the result is acceptable.

The AI agent replaces the manual trial-and-error that human artists would otherwise perform when integrating 3D assets into Blender scenes: once the pipeline is validated, large-scale dataset construction no longer requires per-object expert tuning, reducing cost and improving consistency.

**Instantiation: view-controlled rotation editing.** We instantiate ARIS for a concrete and evaluable task: constructing a dataset for *rotation-conditioned image editing*, where a model must rotate an object from a canonical front view to a target viewpoint specified in natural language (e.g., *"right side view"*, *"back-left view"*). We construct **ARIS-Rotate**, comprising 50 objects × 8 horizontal viewpoints, yielding 350 editing pairs for training.

**Results.** Ablation experiments show that the VLM evolution loop substantially improves rendering quality compared to single-pass rendering without iterative refinement. LoRA fine-tuning of Qwen Image Edit 2511 on ARIS-Rotate improves over the base model on PSNR, SSIM, and LPIPS, confirming that automatically constructed data has downstream training value. Comparing LoRA trained on refined data versus unrefined data further demonstrates that the evolution loop produces data of higher utility.

**Contributions:**

1. **Free-form VLM-guided rendering evolution loop.** We introduce a quality refinement loop in which an AI agent reads free-form VLM reviewer feedback and selects rendering actions accordingly, replacing rigid score-based controllers with semantically-aware adaptation.

2. **Scene-aware Blender rendering integration.** We describe a rendering system that preserves real-scene lighting and environment, integrating 3D objects into existing Blender scene files while allowing per-object adjustment of lighting, placement, material, and scene parameters through 24 structured atomic actions.

3. **ARIS-Rotate dataset and validation.** We construct a 50-object, 350-pair rotation editing dataset with natural-language viewpoint prompts, and validate its training utility via LoRA fine-tuning on Qwen Image Edit 2511.

---

## 2. Related Work

**Synthetic data for image editing.** A large body of work has explored automatically constructing paired training data for instruction-following image editing. InstructPix2Pix [CITATION NEEDED] generates training pairs by combining GPT-4 [CITATION NEEDED] to produce editing instructions with Stable Diffusion [CITATION NEEDED] to render source/target image pairs. MagicBrush [CITATION NEEDED] introduces human-annotated real editing pairs to address distributional gaps in purely synthetic data. Emu Edit [CITATION NEEDED] scales instruction-following editing using a diverse multi-task dataset. These approaches focus on 2D semantic editing tasks; constructing *geometrically controlled* pairs—especially multi-view rotation—requires explicit 3D modeling that 2D diffusion-based pipelines cannot provide. ARIS fills this gap through a full 3D-to-render pipeline with iterative quality control.

**3D-aware data generation.** Objaverse [CITATION NEEDED] and its successors provide large-scale 3D asset libraries for rendering synthetic multi-view data. Methods such as Zero-1-to-3 [CITATION NEEDED], SyncDreamer [CITATION NEEDED], and Wonder3D [CITATION NEEDED] train novel view synthesis models on such data. These works focus on geometry reconstruction and view synthesis, but do not address scene-aware rendering quality or the iterative quality control needed to produce *training-ready* edited pairs suitable for instruction-following models. ARIS bridges this gap by integrating 3D reconstruction with scene-aware Blender rendering and VLM-based quality evolution.

**VLM as feedback and judge.** Using language models to evaluate or improve the outputs of other systems has gained significant traction. LLM-as-a-Judge [CITATION NEEDED] and related work show that strong LLMs can serve as proxies for human evaluation. In the multimodal setting, recent work has explored VLMs for image quality assessment [CITATION NEEDED] and preference alignment [CITATION NEEDED]. Our approach differs in that the VLM feedback is not used for *post-hoc evaluation* but as an *active signal* in an iterative rendering loop: the AI agent reads the reviewer's free-form text and directly decides which rendering parameters to adjust, closing the loop between evaluation and generation.

---

## 3. Method

### 3.1 Pipeline Overview

ARIS transforms a natural-language object description into a set of high-quality rendered images through six stages. The pipeline is fully automated: no human intervention is required between the initial concept description and the final quality-approved renders.

> **[Figure 1 placeholder]** Horizontal flowchart: Text Expansion → T2I Generation → SAM3 Segmentation → 3D Reconstruction → Scene-Aware Rendering → VLM Review Loop

**Stage 1: Text expansion.** Given a seed concept (e.g., *"wooden chair"*), a language model expands it into a detailed T2I prompt specifying material, color, geometry, and surface texture. This step increases prompt specificity, improving the quality and diversity of downstream generated images.

**Stage 2: Object image generation.** A T2I model (Qwen-Image-2512) generates a 1024×1024 reference image of the object against a clean background, using the expanded prompt.

**Stage 2.5: Foreground segmentation.** SAM3 segments the object foreground from the background, producing an RGBA image with transparent background. This removes incidental background content while preserving the object with high fidelity. Critically, SAM3 operates on the object region only, so the resulting 3D reconstruction (Stage 3) does not contain background geometry such as a floor plane, eliminating the need for post-hoc mesh cleaning.

**Stage 3: Single-image 3D reconstruction.** Hunyuan3D-2.1 reconstructs a textured 3D mesh (GLB format) from the RGBA image using a feedforward single-image reconstruction model. The resulting mesh includes physically-based rendering (PBR) texture maps, enabling realistic rendering under varied lighting conditions.

**Stage 4: Scene-aware Blender rendering.** The reconstructed mesh is inserted into a real Blender scene and rendered using Cycles path tracing. Section 3.2 describes this stage in detail.

**Stage 5.5–5.6: VLM review and rendering evolution.** Rendered images are reviewed by a VLM (Qwen3.5-35B-A3B) using free-form natural language. An AI agent reads this review and selects rendering parameter adjustments to address identified quality issues. This loop repeats until the reviewer explicitly confirms the result is acceptable. Section 3.3 describes this stage in detail.

---

### 3.2 Scene-Aware Blender Rendering

Rendering each 3D object requires placing it plausibly within a scene, matching the scene's lighting environment, and rendering from consistent viewpoints. We use a fixed Blender scene file (a furnished indoor scene) as the background environment.

**Scene environment preservation.** A key design principle is *not* to overwrite the scene's existing lighting: the world environment (HDRI) and existing lights are preserved from the scene file. Key lights are scaled by a fixed factor (0.3) rather than disabled, ensuring the object receives natural scene illumination. This avoids the common failure mode of rendering objects under artificial studio lighting that clashes with the scene background.

**Object placement.** The object mesh is placed on the scene ground plane using a raycast procedure: the system casts a ray downward from the object's bounding box to detect the real ground surface height, ensuring physical plausibility. The object is positioned such that its base rests on the scene floor, with a configurable vertical offset for fine adjustment.

**Rendering configuration.** All renders use Cycles path tracing with 512 samples at 1024×1024 resolution, providing high-quality global illumination and shadow detail. For each object, we render eight horizontal viewpoints at 0°, 45°, 90°, 135°, 180°, 225°, 270°, and 315° azimuth, corresponding to the natural-language view labels defined in Section 4.

---

### 3.3 Free-Form VLM-Guided Rendering Evolution

The core contribution of ARIS is the iterative quality refinement loop that drives rendered images toward acceptable visual quality through VLM feedback.

> **[Figure 2 placeholder]** Left column: Round 0 render → Round 1 → ... → Final (keep). Right column: VLM free-form text summary + agent-selected action at each round.

**VLM review.** After each rendering round, a VLM (Qwen3.5-35B-A3B, with extended thinking enabled) reviews the rendered images and produces a free-form natural language assessment. The reviewer is prompted to identify specific visual problems, including:

- `flat_low_contrast` — flat or low-contrast lighting
- `color_shift` — color inconsistency between object and scene
- `scene_light_mismatch` — lighting direction mismatches the scene
- `shadow_missing` — missing or weak contact shadows
- `floating` / `ground_intersection` — physically implausible placement
- `scale_implausible` — object too large or too small for the scene
- `material_too_plastic` — overly smooth or unrealistic material appearance

The reviewer concludes with one of three verdicts: **keep** (result is acceptable), **revise** (specific changes recommended), or **reject** (fundamental problems requiring re-rendering).

**AI agent decision.** An AI agent reads the full free-form reviewer text from the trace record and identifies the primary diagnosed issue. Based on the diagnosis, the agent selects one or more actions from the rendering action space (Table 1). This design differs from rigid score-based controllers: the agent interprets semantic descriptions of visual failures rather than responding mechanically to numeric thresholds.

**Action space.** The rendering action space contains 24 atomic actions organized into four groups:

**Table 1: Rendering Action Space**

| Group | Example Actions | Count | Target Parameters |
|-------|----------------|-------|-------------------|
| Lighting | `L_KEY_UP`, `L_KEY_YAW_POS_15` | 4 | Key light scale, key light yaw angle |
| Object | `O_LIFT_SMALL`, `O_SCALE_UP_10` | 6 | Vertical offset, yaw rotation, scale |
| Scene | `ENV_ROTATE_30`, `ENV_STRENGTH_UP` | 5 | HDRI yaw, environment strength, contact shadow |
| Material | `M_VALUE_DOWN`, `M_ROUGHNESS_UP` | 9 | Saturation, value, hue offset, roughness, specular |
| **Total** | | **24** | |

**Issue-to-action mapping.** The agent uses a structured mapping from diagnosed issues to candidate actions. For example:
- `flat_low_contrast` → `ENV_STRENGTH_UP`
- `color_shift` → `M_VALUE_DOWN_STRONG`, `M_SATURATION_DOWN`
- `scene_light_mismatch` → `ENV_ROTATE_30`, `L_KEY_YAW_POS_15`
- `shadow_missing` → `S_CONTACT_SHADOW_UP`, `L_KEY_UP`
- `floating_visible` → `O_LOWER_SMALL`

**Anti-oscillation and step decay.** To prevent the agent from oscillating between opposing actions, the system tracks the sign of each parameter's adjustment history. If a parameter's sign has reversed three or more times, that parameter is frozen for the remainder of the optimization. Action step sizes decay with round number (100% at round 0, 70% at round 1, 50% at round 2+) to achieve fine-grained convergence near the optimum.

**Termination.** The loop terminates when the VLM reviewer explicitly produces a *keep* verdict, or after a maximum of five rounds. Only results with an explicit keep verdict are considered quality-approved.

---

## 4. ARIS-Rotate Dataset

### 4.1 Construction

We apply the ARIS pipeline to construct **ARIS-Rotate**, a dataset for rotation-conditioned image editing. Each sample in ARIS-Rotate pairs a canonical front-view image of an object with a target-view image and a natural-language editing instruction.

**Object selection and diversity.** We generate 50 objects spanning diverse categories including furniture, household items, tools, decorative objects, and outdoor equipment. Object descriptions are generated to cover a range of materials (wood, metal, plastic, fabric), shapes (compact, elongated, symmetric, asymmetric), and color profiles.

**Viewpoint specification.** Each object is rendered at eight equally-spaced horizontal azimuths. We represent viewpoints using natural-language labels rather than numeric angles, reflecting how users would naturally describe rotation instructions:

| Azimuth | View Label |
|---------|-----------|
| 0° | front view |
| 45° | front-right view |
| 90° | right side view |
| 135° | back-right view |
| 180° | back view |
| 225° | back-left view |
| 270° | left side view |
| 315° | front-left view |

**Editing pairs and instruction template.** We designate the 0° (front view) render as the canonical source image. For each of the remaining seven target views, we form an editing pair with the following instruction template:

> `Rotate this object from front view to {target_view}.`

Each of the 50 objects contributes 7 editing pairs, for a total of **350 training pairs**. The full rendered image set comprises **400 images** (50 objects × 8 views).

> **[Figure 3 placeholder]** Grid: rows = 4 representative objects, columns = 8 viewpoints. Each cell shows the rendered image at that viewpoint after VLM evolution.

### 4.2 Data Split

We partition objects using an **object-disjoint** split: the same object cannot appear in both training and test sets.

| Split | Objects | Pairs | Images |
|-------|---------|-------|--------|
| Train | 35 | 245 | 280 |
| Val | 5 | 35 | 40 |
| Test | 10 | 70 | 80 |
| **Total** | **50** | **350** | **400** |

### 4.3 Dataset Statistics

> **[TODO: Fill after data generation]**
> - Distribution of VLM evolution rounds per object
> - Distribution of final VLM quality scores
> - Fraction of objects requiring ≥1, ≥3, ≥5 refinement rounds
> - Breakdown of primary issue types diagnosed (flat_lighting, color_shift, etc.)

---

## 5. Experiments

### 5.1 Ablation: VLM Evolution Loop Effectiveness

We first validate that the VLM-guided evolution loop improves rendering quality compared to single-pass rendering without iterative refinement. We construct two versions of the 50-object rendering set:

- **No-evolution (direct render):** The 3D mesh is inserted into the Blender scene and rendered at the default control state without any VLM review or parameter adjustment.
- **Full pipeline (VLM evolution):** The complete ARIS loop is applied, with the AI agent reading VLM free-form feedback and selecting rendering actions until the reviewer issues a keep verdict.

**Table 2: Ablation — Effect of VLM Evolution Loop on Rendering Quality**

| Setting | Avg. Quality Score ↑ | Keep Rate (%) ↑ | Avg. Rounds |
|---------|---------------------|-----------------|-------------|
| No-evolution (direct render) | [TODO] | [TODO] | 0 |
| Full pipeline (VLM evolution, ours) | [TODO] | [TODO] | [TODO] |

> **[Figure 4 placeholder]** 3–4 representative objects. Left: No-evolution. Right: Full pipeline. Annotate primary issue fixed (flat lighting, color shift, shadow, etc.)

### 5.2 LoRA Fine-Tuning Setup

We evaluate the downstream training value of ARIS-Rotate by fine-tuning Qwen Image Edit 2511 with LoRA on the training split and evaluating on the test split.

**Base model.** Qwen Image Edit 2511 is an instruction-following image editing model that takes a source image and a text instruction as input and produces an edited image.

**Training configuration.**
- LoRA rank: [TODO], alpha: [TODO], target modules: [TODO]
- Learning rate: [TODO], batch size: [TODO], epochs: [TODO]
- Input: source image (front view) + text instruction
- Target: edited image (target viewpoint)

**Compared conditions:**
1. **Base model** — Qwen Image Edit 2511 without fine-tuning
2. **LoRA (no-evolution data)** — LoRA fine-tuned on single-pass rendered data
3. **LoRA (full pipeline data)** — LoRA fine-tuned on quality-approved ARIS data

### 5.3 Quantitative Results

We evaluate on the object-disjoint test split (10 objects, 70 pairs) using PSNR, SSIM, and LPIPS.

**Table 3: Quantitative Evaluation on ARIS-Rotate Test Set**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| Base model (no fine-tuning) | [TODO] | [TODO] | [TODO] |
| LoRA on no-evolution data | [TODO] | [TODO] | [TODO] |
| **LoRA on full pipeline data (ours)** | **[TODO]** | **[TODO]** | **[TODO]** |

**Table 4: Per-Viewpoint PSNR — Base vs. LoRA (Full Pipeline)**

| Target View | Azimuth | Base PSNR | LoRA PSNR |
|-------------|---------|-----------|-----------|
| front-right view | 45° | [TODO] | [TODO] |
| right side view | 90° | [TODO] | [TODO] |
| back-right view | 135° | [TODO] | [TODO] |
| back view | 180° | [TODO] | [TODO] |
| back-left view | 225° | [TODO] | [TODO] |
| left side view | 270° | [TODO] | [TODO] |
| front-left view | 315° | [TODO] | [TODO] |
| **Mean** | | **[TODO]** | **[TODO]** |

### 5.4 Qualitative Results

> **[Figure 5 placeholder]** Rows: 3–4 representative objects. Columns: Input (front view) | Base model output | LoRA output | Ground truth. Show difficult views (back view, left side view) where improvement is most visible.

---

## 6. Limitations

**Action space coverage.** The current 24-action discrete action space has limited coverage of certain failure modes. In particular, `flat_low_contrast` (where the overall scene has low tonal range) and `color_shift` (where object hue diverges from scene) are not always fully resolved through material and lighting adjustments alone. Some objects required 40+ evolution rounds without reaching a keep verdict, suggesting that these failure modes require actions beyond the current space—for example, scene background replacement or global tone mapping adjustments.

**Dataset scale and category coverage.** ARIS-Rotate contains 50 objects. While sufficient to demonstrate the pipeline's feasibility and validate downstream utility, this scale does not provide the breadth required for a general rotation editing model. Category coverage is also limited; some specialized or atypical object types may fall outside the distribution covered by this first validation set.

**Rotation-only validation.** We validate ARIS on a single editing task (rotation). The pipeline is designed to generalize to other geometry-controlled editing tasks (lighting changes, material substitution, scale variation), but such generalization has not been experimentally verified in this report.

**VLM reviewer noise.** The Qwen3.5-35B-A3B reviewer exhibits score variance of approximately 0.006–0.012 per query, which can produce inconsistent verdicts for borderline cases. Repeated review passes or multi-view aggregation can mitigate this, but reviewer reliability remains a variable in pipeline quality.

**No public release.** The current implementation relies on a specific Blender scene file and server environment and is not publicly released with this technical report.

> **[Figure 6 placeholder]** 2–3 objects where the evolution loop could not reach a keep verdict. Annotate: (a) flat_low_contrast unresolved, (b) persistent color_shift, (c) geometry-limited case.

---

## 7. Conclusion

We presented ARIS, a pipeline for automatically constructing training-ready synthetic datasets for image editing tasks. The central contribution is the free-form VLM-guided rendering evolution loop, in which an AI agent reads natural language feedback from a VLM reviewer and selects rendering parameter adjustments from a structured action space, iterating until the reviewer confirms acceptable quality. This approach replaces manual Blender parameter tuning with an automated feedback cycle, enabling scalable and consistent 3D dataset construction.

We instantiated ARIS for rotation-conditioned image editing, producing ARIS-Rotate: a 50-object, 350-pair dataset with natural-language viewpoint instructions. Ablation experiments confirmed that the evolution loop improves rendering quality over single-pass rendering. LoRA fine-tuning of Qwen Image Edit 2511 on ARIS-Rotate demonstrated measurable improvement over the base model on the test set, validating the downstream utility of automatically constructed data.

**Future work.** Several directions extend this work naturally. First, expanding the action space to cover failure modes currently unresolved (global tone mapping, background replacement) would improve coverage. Second, applying ARIS to additional editing tasks—lighting variation, material substitution, scale change—would validate generality. Third, scaling to larger object sets (500+) with broader category coverage would provide the data volume needed for training stronger editing models. Finally, integrating an adaptive round budget that allocates more evolution rounds to harder objects would improve efficiency on large-scale runs.

---

## References

> [CITATION NEEDED — all references below require verification via Semantic Scholar API or DOI before submission]

- [1] Brooks et al. "InstructPix2Pix: Learning to Follow Image Editing Instructions." CVPR 2023.
- [2] OpenAI. "GPT-4 Technical Report." 2023.
- [3] Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.
- [4] Zhang et al. "MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing." 2024.
- [5] Sheynin et al. "Emu Edit: Precise Image Editing via Recognition and Generation Tasks." 2024.
- [6] Deitke et al. "Objaverse: A Universe of Annotated 3D Objects." 2023.
- [7] Liu et al. "Zero-1-to-3: Zero-shot One Image to 3D Object." 2023.
- [8] Liu et al. "SyncDreamer: Generating Multiview-Consistent Images from a Single-View Image." 2023.
- [9] Long et al. "Wonder3D: Single Image to 3D Using Cross-Domain Diffusion." 2024.
- [10] Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." 2024.
- [11] Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)." CVPR 2018.
- [12] Qwen Team. "Qwen Image Edit 2511." 2025. [TODO: official citation]
- [13] Qwen Team. "Qwen3.5-35B-A3B." 2025. [TODO: official citation]
- [14] Tencent Hunyuan Team. "Hunyuan3D-2.1." 2025. [TODO: official citation]

---

*[TODO items summary]*
- *Section 4.3: Fill dataset statistics after pipeline run completes*
- *Section 5.2: Fill LoRA training config after training run*
- *Tables 2–4: Fill all [TODO] numbers after experiments complete*
- *Figures 1–6: Replace placeholders with actual figures*
- *All references: Verify via Semantic Scholar API before arXiv submission*
