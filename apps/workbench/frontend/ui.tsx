import React from "react";

export type Translate = (en: string, cn: string) => string;

export type SettingsCategory =
  | "models"
  | "images"
  | "paths"
  | "execution"
  | "runtime"
  | "credentials";

const svgProps = {
  width: 16,
  height: 16,
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true as const,
};

export function IconPlus() {
  return (
    <svg {...svgProps}>
      <path d="M8 3.25v9.5M3.25 8h9.5" />
    </svg>
  );
}

export function IconChat() {
  return (
    <svg {...svgProps}>
      <path d="M2.75 4.25A1.5 1.5 0 0 1 4.25 2.75h7.5a1.5 1.5 0 0 1 1.5 1.5v5.5a1.5 1.5 0 0 1-1.5 1.5H6.5l-3 2.25v-2.25h-.25a1.5 1.5 0 0 1-.5-1.12V4.25Z" />
    </svg>
  );
}

export function IconHarness() {
  return (
    <svg {...svgProps}>
      <rect x="2.5" y="2.75" width="4.25" height="4.25" rx="1" />
      <rect x="9.25" y="2.75" width="4.25" height="4.25" rx="1" />
      <rect x="5.85" y="9.25" width="4.3" height="4.25" rx="1" />
      <path d="M4.6 7v1.1c0 .6.5 1.15 1.25 1.15h4.3c.75 0 1.25-.55 1.25-1.15V7" />
    </svg>
  );
}

export function IconImage() {
  return (
    <svg {...svgProps}>
      <rect x="2.5" y="2.75" width="11" height="10.5" rx="1.5" />
      <circle cx="5.75" cy="6" r="1.1" />
      <path d="m3.5 11.75 3.1-3.1a1 1 0 0 1 1.4 0l1.65 1.65 1.35-1.35a1 1 0 0 1 1.4 0l1.1 1.1" />
    </svg>
  );
}

export function IconSettings() {
  return (
    <svg {...svgProps}>
      <path d="M2.75 4.5h1.65M7.1 4.5h6.15M2.75 8h6.15M11.6 8h1.65M2.75 11.5h2.65M8.1 11.5h5.15" />
      <circle cx="5.75" cy="4.5" r="1.35" />
      <circle cx="10.25" cy="8" r="1.35" />
      <circle cx="6.75" cy="11.5" r="1.35" />
    </svg>
  );
}

export function IconGlobe() {
  return (
    <svg {...svgProps}>
      <circle cx="8" cy="8" r="5.5" />
      <path d="M2.75 8h10.5M8 2.5c1.5 1.5 2.25 3.4 2.25 5.5S9.5 12 8 13.5C6.5 12 5.75 10.1 5.75 8S6.5 4 8 2.5Z" />
    </svg>
  );
}

export function IconSend() {
  return (
    <svg {...svgProps}>
      <path d="M8 12.75V3.5M4.25 7.25 8 3.5l3.75 3.75" />
    </svg>
  );
}

export function IconKey() {
  return (
    <svg {...svgProps}>
      <circle cx="5.75" cy="8.25" r="2.75" />
      <path d="M8.5 8.25h5M11.5 8.25v2M13.25 8.25v1.5" />
    </svg>
  );
}

export function IconClose() {
  return (
    <svg {...svgProps}>
      <path d="m4.25 4.25 7.5 7.5M11.75 4.25l-7.5 7.5" />
    </svg>
  );
}

export function IconArrowUpRight() {
  return (
    <svg {...svgProps}>
      <path d="M4.5 11.5 11.5 4.5M6 4.5h5.5V10" />
    </svg>
  );
}

export function getSettingsCategories(t: Translate): {
  id: SettingsCategory;
  label: string;
  summary: string;
}[] {
  return [
    {
      id: "models",
      label: t("Models and agents", "模型与 Agent"),
      summary: t("Providers, model limits & agent roles", "服务协议、模型参数与主子角色"),
    },
    {
      id: "images",
      label: t("Images", "图像服务"),
      summary: t("Generation, edit & prompt enhancer", "生成、编辑与提示增强路由"),
    },
    {
      id: "paths",
      label: t("Tools and paths", "工具与路径"),
      summary: t("Workspace, Python, Blender & UE5", "工作区与本地可执行文件路径"),
    },
    {
      id: "execution",
      label: t("Execution and SSH", "执行主机与 SSH"),
      summary: t("Local workstation or remote OpenSSH", "本地或远端服务器执行环境"),
    },
    {
      id: "runtime",
      label: t("Agent runtime", "Agent 运行时"),
      summary: t("Runtime CLI, reducer & compaction", "运行时扩展、日志归约与压缩"),
    },
    {
      id: "credentials",
      label: t("Credentials", "私有凭据"),
      summary: t("Write-only API keys & environment vars", "仅写入的环境变量密钥管理"),
    },
  ];
}

export function harnessCommandMeta(
  command: string,
  t: Translate,
): { title: string; summary: string } {
  switch (command) {
    case "compile-plan":
      return {
        title: t("Compile plan", "编译计划"),
        summary: t(
          "Compile a task plan with profile and evaluator into an executable artifact.",
          "结合运行配置与评估器，将任务计划编译为可执行产物。",
        ),
      };
    case "worker-compile":
      return {
        title: t("Compile worker", "编译执行器"),
        summary: t(
          "Build the worker bundle for a target campaign profile and output path.",
          "为目标 Campaign 配置与输出路径构建执行器包。",
        ),
      };
    case "create":
      return {
        title: t("Create campaign", "创建任务"),
        summary: t(
          "Initialize Campaign state from a goal specification file.",
          "从目标规范文件初始化 Campaign 状态目录。",
        ),
      };
    case "step":
      return {
        title: t("Bounded step", "有界推进"),
        summary: t(
          "Advance an existing Campaign by a bounded maximum number of actions.",
          "按指定的最大动作步数有界推进已有 Campaign。",
        ),
      };
    case "status":
      return {
        title: t("Inspect status", "查看状态"),
        summary: t(
          "Read the current Campaign status snapshot and progress counters.",
          "读取当前 Campaign 的状态快照与阶段进度。",
        ),
      };
    case "explain":
      return {
        title: t("Explain state", "解释状态"),
        summary: t(
          "Summarize recent state transitions, evidence, and active blockers.",
          "汇总近期状态流转、关键证据与当前阻塞原因。",
        ),
      };
    case "replay":
      return {
        title: t("Replay trace", "回放轨迹"),
        summary: t(
          "Replay recorded Campaign execution receipts and event history.",
          "回放已记录的 Campaign 执行回执与事件轨迹。",
        ),
      };
    case "pause":
      return {
        title: t("Pause campaign", "暂停任务"),
        summary: t(
          "Request an active Campaign to pause cleanly at a safe boundary.",
          "请求运行中的 Campaign 在安全边界处暂停。",
        ),
      };
    case "resume":
      return {
        title: t("Resume campaign", "恢复任务"),
        summary: t(
          "Resume native Campaign execution from its persisted checkpoint.",
          "从持久化检查点恢复 Campaign 的原生执行。",
        ),
      };
    default:
      return { title: command, summary: "" };
  }
}

export function harnessFieldLabel(key: string, t: Translate): string {
  switch (key) {
    case "plan":
      return t("Plan file path (plan)", "计划文件路径 (plan)");
    case "profile":
      return t("Execution profile (profile)", "运行配置 (profile)");
    case "evaluator":
      return t("Evaluator file path (evaluator)", "评估器文件路径 (evaluator)");
    case "output":
      return t("Compiled output path (output)", "编译输出路径 (output)");
    case "campaign":
      return t("Campaign ID (campaign)", "任务标识 (campaign)");
    case "campaignRoot":
      return t("Campaign root directory (campaignRoot)", "任务根目录 (campaignRoot)");
    case "goal":
      return t("Goal specification path (goal)", "目标规范文件路径 (goal)");
    case "idempotencyKey":
      return t("Idempotency key (idempotencyKey)", "幂等键 (idempotencyKey)");
    case "maxActions":
      return t("Max bounded actions (maxActions)", "最大推进步数 (maxActions)");
    default:
      return key;
  }
}

export function imageRouteMeta(
  kind: "generation" | "edit" | "promptEnhancer",
  t: Translate,
): { title: string; protocol: string } {
  switch (kind) {
    case "generation":
      return {
        title: t("Image generation", "图像生成"),
        protocol: "POST /images/generations (b64_json)",
      };
    case "edit":
      return {
        title: t("Image editing", "图像编辑"),
        protocol: "POST /images/edits (multipart · b64_json)",
      };
    case "promptEnhancer":
      return {
        title: t("Prompt enhancement", "提示词增强"),
        protocol: "POST /chat/completions",
      };
  }
}

export function pathFieldLabel(key: string, t: Translate): string {
  switch (key) {
    case "workspace":
      return t("Workspace directory (workspace)", "工作区目录 (workspace)");
    case "python":
      return t("Python executable (python)", "Python 可执行文件 (python)");
    case "blender":
      return t("Blender executable (blender)", "Blender 可执行文件 (blender)");
    case "ue5":
      return t("Unreal Engine 5 executable (ue5)", "UE5 可执行文件 (ue5)");
    case "ueProject":
      return t("Unreal project file (ueProject)", "UE 项目文件 (ueProject)");
    case "assets":
      return t("Asset directory (assets)", "资产目录 (assets)");
    case "datasets":
      return t("Dataset directory (datasets)", "数据集目录 (datasets)");
    case "output":
      return t("Output directory (output)", "输出目录 (output)");
    default:
      return key;
  }
}

export function executionFieldLabel(key: string, t: Translate): string {
  switch (key) {
    case "sshHost":
      return t("OpenSSH host alias (sshHost)", "OpenSSH 主机别名 (sshHost)");
    case "remoteWorkspace":
      return t("Remote workspace path (remoteWorkspace)", "远端工作区路径 (remoteWorkspace)");
    case "remotePython":
      return t("Remote Python executable (remotePython)", "远端 Python 路径 (remotePython)");
    case "remoteBlender":
      return t("Remote Blender executable", "远端 Blender 可执行文件");
    case "remoteUe5":
      return t("Remote UE5 executable", "远端 UE5 可执行文件");
    case "remoteUeProject":
      return t("Remote Unreal project file", "远端 UE 项目文件");
    case "remoteAssets":
      return t("Remote asset directory", "远端资产目录");
    case "remoteDatasets":
      return t("Remote dataset directory", "远端数据集目录");
    case "remoteOutput":
      return t("Remote output directory", "远端输出目录");
    default:
      return key;
  }
}

export function runtimeFieldLabel(key: string, t: Translate): string {
  switch (key) {
    case "cli":
      return t("Runtime Node.js entry file", "运行时 Node.js 入口文件");
    case "solExtension":
      return t("Runtime extension", "运行时扩展");
    default:
      return key;
  }
}

export function runtimeToggleMeta(
  key:
    | "enabled"
    | "actionFusion"
    | "observationPack"
    | "evidencePreservingReducer"
    | "onlineContextCompact",
  t: Translate,
): { title: string; hint: string } {
  switch (key) {
    case "enabled":
      return {
        title: t("Enable runtime pipeline", "启用运行时管线"),
        hint: t("Activate extended runtime hooks for new sessions", "为新会话启用扩展运行时处理管线"),
      };
    case "actionFusion":
      return {
        title: t("Action fusion", "动作融合 (Action fusion)"),
        hint: t("Combine sequential compatible tool actions", "合并连续且兼容的工具执行步骤"),
      };
    case "observationPack":
      return {
        title: t("Observation pack", "观测打包 (Observation pack)"),
        hint: t("Structure verbose tool observations compactly", "结构化整理冗长的环境观测输出"),
      };
    case "evidencePreservingReducer":
      return {
        title: t("Evidence-preserving reducer", "保真日志归约"),
        hint: t("Summarize long logs while retaining key receipts", "在保留关键证据回执的前提下归约长日志"),
      };
    case "onlineContextCompact":
      return {
        title: t("Online context compaction", "在线上下文压缩"),
        hint: t("Compact long session history between active turns", "在长任务轮次间自动压缩历史上下文"),
      };
  }
}

export function stateTone(
  raw: string,
): "active" | "success" | "warn" | "danger" | "neutral" {
  const s = (raw || "").toLowerCase();
  if (["running", "streaming", "busy", "working", "started"].includes(s))
    return "active";
  if (["completed", "succeeded", "done", "ok", "ready", "open", "idle"].includes(s))
    return "success";
  if (["paused", "pending", "queued", "reconnecting"].includes(s))
    return "warn";
  if (["failed", "error", "aborted"].includes(s)) return "danger";
  return "neutral";
}

export function formatState(raw: string, t: Translate): string {
  const s = (raw || "").toLowerCase();
  switch (s) {
    case "running":
      return t("running", "运行中");
    case "idle":
      return t("idle", "空闲");
    case "stopped":
      return t("stopped", "已停止");
    case "completed":
    case "succeeded":
      return t(raw, "已完成");
    case "failed":
    case "error":
      return t(raw, "失败");
    case "paused":
      return t("paused", "已暂停");
    case "queued":
      return t("queued", "排队中");
    default:
      return raw;
  }
}
