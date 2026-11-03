import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readFile } from "node:fs/promises";

export default function (pi: ExtensionAPI) {
  const call = async (op: string, data: unknown, signal?: AbortSignal) => {
    const response = await fetch(process.env.DATAEVOLVER_WORKBENCH_BRIDGE!, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.DATAEVOLVER_WORKBENCH_CAPABILITY}`,
      },
      body: JSON.stringify({ op, data }),
      signal,
    });
    const result = (await response.json()) as Record<string, unknown>;
    if (!response.ok)
      throw new Error(
        (typeof result.error === "string"
          ? result.error
          : "Workbench operation failed") +
          (typeof result.receipt === "string"
            ? ` · receipt: ${result.receipt}`
            : ""),
      );
    return result;
  };
  const result = (value: unknown) => ({
    content: [{ type: "text" as const, text: JSON.stringify(value) }],
    details: value,
  });
  pi.registerTool({
    name: "dataevolver_harness",
    label: "DataEvolver Harness",
    description:
      "Compile or control an existing DataEvolver Campaign on the configured local/SSH host. Values are explicit CLI fields, not shell text. plan requires goal and inspects an image pipeline offline; compile-plan requires plan/profile/evaluator/output/campaign; worker-compile requires profile/output/campaign/campaignRoot; create requires goal/campaignRoot/idempotencyKey and accepts optional boolean dryRun (true only for fixture image Goals, false/omitted for server Goals); step requires campaign/campaignRoot/maxActions (1..100); inspection/pause/resume require campaign/campaignRoot. resume executes work when dispatched. Mutating commands wait in a bounded Workbench queue; plan/status/explain/replay/pause bypass it. Session abort/stop cancels waiting jobs, not running effects. The returned job receipt includes exit status and logs; unknown effects require inspection, not automatic resend.",
    parameters: Type.Object({
      command: Type.Union(
        [
          "plan",
          "compile-plan",
          "worker-compile",
          "create",
          "step",
          "status",
          "explain",
          "replay",
          "pause",
          "resume",
        ].map((value) => Type.Literal(value)),
      ),
      values: Type.Record(
        Type.String(),
        Type.Union([Type.String(), Type.Number(), Type.Boolean()]),
      ),
    }),
    async execute(_id, params, signal) {
      return result(await call("harness", params, signal));
    },
  });
  pi.registerTool({
    name: "dataevolver_image",
    label: "Configured image API",
    description:
      "Use the configured OpenAI-compatible image generation/editing API or chat/vision prompt enhancer. Input image paths refer to the workbench host; output and receipt are retained there. Calls are never automatically retried. Generation/editing endpoints must return b64_json. A different native protocol needs its own adapter.",
    parameters: Type.Object({
      kind: Type.Union([
        Type.Literal("generation"),
        Type.Literal("edit"),
        Type.Literal("promptEnhancer"),
      ]),
      prompt: Type.String(),
      image: Type.Optional(Type.String()),
    }),
    async execute(_id, params, signal) {
      const value = await call("image", params, signal);
      const response = result(value);
      if (typeof value.path === "string") {
        const mimeType = value.path.endsWith(".png")
          ? "image/png"
          : value.path.endsWith(".webp")
            ? "image/webp"
            : "image/jpeg";
        const bytes = await readFile(value.path);
        if (bytes.length <= 4 * 1024 * 1024)
          return {
            content: [
              ...response.content,
              {
                type: "image" as const,
                data: bytes.toString("base64"),
                mimeType,
              },
            ],
            details: value,
          };
      }
      return response;
    },
  });
  pi.registerTool({
    name: "dataevolver_delegate",
    label: "Delegate child task",
    description:
      "Delegate using the configured child provider/model. mode=proposal authors candidates in a fresh local workspace with a bounded snapshot of explicit contextJobIds; its bridge permits only Harness status/explain/replay. This is not an OS sandbox. Default mode=task preserves the shared workspace and tool access. Returns a child id; inspect/wait/stop with dataevolver_child. Stop completed children to release capacity. Only main sessions may delegate. Context logs are observations, not verified knowledge.",
    parameters: Type.Object({
      name: Type.String(), task: Type.String(),
      mode: Type.Optional(Type.Union([Type.Literal("task"), Type.Literal("proposal")])),
      contextJobIds: Type.Optional(Type.Array(Type.String(), { maxItems: 16 })),
    }),
    async execute(_id, params, signal) {
      return result(await call("delegate", params, signal));
    },
  });
  pi.registerTool({
    name: "dataevolver_child",
    label: "Inspect child session",
    description:
      "Inspect, wait for completion, or stop one of this main session's children. Child messages are evidence to review, not automatic acceptance. Stopping a child does not prove native effects were undone.",
    parameters: Type.Object({
      id: Type.String(),
      action: Type.Union([
        Type.Literal("inspect"),
        Type.Literal("wait"),
        Type.Literal("stop"),
      ]),
    }),
    async execute(_id, params, signal) {
      return result(await call("child", params, signal));
    },
  });
}
