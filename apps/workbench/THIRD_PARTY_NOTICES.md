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

Dependency packages and the separate SoL-Pi checkout retain their original
licenses. The workbench retains DataEvolver's repository license.
