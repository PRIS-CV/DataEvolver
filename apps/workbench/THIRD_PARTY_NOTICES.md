# Upstream references and dependencies

The independent DataEvolver Workbench follows DeepSeek Harness (DSH) patterns for
CLI-to-web startup, a session rail, settings cards, write-only credentials and a
protected local browser boundary. This implementation does not copy the Cordis
plugin runtime or DSH UI source.

- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), MIT;
  source reference `ce7073d51254c26cd63b22a60835b8f948deb447`.
- [Pi](https://github.com/earendil-works/pi), MIT. The workbench depends on the
  published package and public RPC client. Pi remains unmodified.
- [SoL-Pi](https://github.com/NVlabs/SoL-Pi), MIT, NVIDIA Corporation & Affiliates.
  The launcher obtains a separate source checkout; it does not vendor Pi or
  SoL-Pi into DataEvolver. Each mechanism retains its upstream implementation.
- React, MIT; Vite, MIT; TypeBox, MIT; TypeScript, Apache-2.0.
- [Base UI](https://github.com/mui/base-ui), MIT, Material-UI SAS. The frontend
  uses its accessible interaction components through local wrappers. The license
  text is included in [licenses/base-ui.txt](licenses/base-ui.txt).
- [Lucide](https://github.com/lucide-icons/lucide), ISC with MIT notices for
  Feather-derived icons. The frontend imports named React icons. The license
  texts are included in [licenses/lucide.txt](licenses/lucide.txt).

The Workbench home composition also takes visual inspiration from
[Canvas UI](https://github.com/DavidHDev/canvas-ui), source reference
`a04a99b235299ee79a123a16ccb2e6cf9da74b6b`. Canvas UI carries an MIT + Commons
Clause license. Its source, effects, styles and artwork are not included in this
implementation; this reference does not change DataEvolver's license.

The frontend refinement follows the app-screen and redesign guidance from
[Hallmark](https://github.com/Nutlope/hallmark), MIT, source reference
`4f6a46c6dc325d40ab268c93e25c7b96676ebccd`. It informs hierarchy, restrained
surfaces, typography and focus states; no Hallmark runtime or artwork is bundled.

Dependency packages and the separate SoL-Pi checkout retain their original
licenses. The workbench retains DataEvolver's repository license.

Candidate orchestration takes design ideas from [Hyra's official description](https://hy.tencent.com/research/hyra)
and the [Hyra AI4AI artifacts](https://github.com/Tencent-Hunyuan/Hyra-results/tree/8443f9a9aae1a0b72c79e0b2908c79026b8e1555/AI4AI).
No Hyra source, training recipe, model weights or runtime is bundled.
