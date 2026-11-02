import React from "react";
import {
  Aperture,
  ArrowRight,
  ArrowUp,
  ArrowUpRight,
  Check,
  ChevronDown,
  Cpu,
  Folder,
  GitBranch,
  Globe,
  Image as LucideImage,
  KeyRound,
  Layers,
  type LucideIcon,
  type LucideProps,
  MessageSquare,
  Plus,
  RotateCcw,
  Server,
  SlidersHorizontal,
  Sparkles,
  Square,
  Terminal,
  X,
} from "lucide-react";

export type Translate = (en: string, cn: string) => string;

export type SettingsCategory =
  | "models"
  | "images"
  | "paths"
  | "execution"
  | "runtime"
  | "credentials";

export function isSettingsCategory(value: unknown): value is SettingsCategory {
  return (
    value === "models" ||
    value === "images" ||
    value === "paths" ||
    value === "execution" ||
    value === "runtime" ||
    value === "credentials"
  );
}

function wrapLucide(Icon: LucideIcon, defaultSize = 16) {
  return function WrappedLucideIcon({
    size = defaultSize,
    strokeWidth = 1.65,
    ...props
  }: LucideProps) {
    return (
      <Icon
        aria-hidden="true"
        focusable="false"
        size={size}
        strokeWidth={strokeWidth}
        {...props}
      />
    );
  };
}

export const IconPlus = wrapLucide(Plus);
export const IconArrowRight = wrapLucide(ArrowRight);
export const IconSend = wrapLucide(ArrowUp);
export const IconArrowUpRight = wrapLucide(ArrowUpRight);
export const IconCheck = wrapLucide(Check, 14);
export const IconChevronDown = wrapLucide(ChevronDown, 14);
export const IconChat = wrapLucide(MessageSquare);
export const IconHarness = wrapLucide(GitBranch);
export const IconImage = wrapLucide(LucideImage);
export const IconAperture = wrapLucide(Aperture);
export const IconSettings = wrapLucide(SlidersHorizontal);
export const IconGlobe = wrapLucide(Globe);
export const IconKey = wrapLucide(KeyRound);
export const IconClose = wrapLucide(X);
export const IconRefresh = wrapLucide(RotateCcw, 15);
export const IconStop = wrapLucide(Square, 14);
export const IconCpu = wrapLucide(Cpu);
export const IconServer = wrapLucide(Server);
export const IconFolder = wrapLucide(Folder);
export const IconLayers = wrapLucide(Layers);
export const IconTerminal = wrapLucide(Terminal);
export const IconSparkles = wrapLucide(Sparkles);

export function getSettingsCategories(t: Translate): {
  id: SettingsCategory;
  label: string;
  summary: string;
}[] {
  return [
    {
      id: "models",
      label: t("Models and agents", "模型与 Agent"),
      summary: t(
        "Providers, model limits & agent roles",
        "服务协议、模型参数与主子角色",
      ),
    },
    {
      id: "images",
      label: t("Images", "图像服务"),
      summary: t(
        "Generation, edit & prompt enhancer",
        "生成、编辑与提示增强路由",
      ),
    },
    {
      id: "paths",
      label: t("Tools and paths", "工具与路径"),
      summary: t(
        "Workspace, Python, Blender & UE5",
        "工作区与本地可执行文件路径",
      ),
    },
    {
      id: "execution",
      label: t("Execution and SSH", "执行主机与 SSH"),
      summary: t(
        "Local workstation or remote OpenSSH",
        "本地或远端服务器执行环境",
      ),
    },
    {
      id: "runtime",
      label: t("Agent runtime", "Agent 运行时"),
      summary: t(
        "Runtime CLI, reducer & compaction",
        "运行时扩展、日志归约与压缩",
      ),
    },
    {
      id: "credentials",
      label: t("Credentials", "私有凭据"),
      summary: t(
        "Write-only API keys & environment vars",
        "仅写入的环境变量密钥管理",
      ),
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
          "Initialize a new Campaign state directory from an explicit goal.",
          "基于明确的任务目标初始化新的 Campaign 状态目录。",
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
      return t(
        "Evaluator path or ID (evaluator)",
        "评估器标识或路径 (evaluator)",
      );
    case "output":
      return t("Compiled output path (output)", "编译输出路径 (output)");
    case "campaign":
      return t("Campaign ID (campaign)", "任务标识 (campaign)");
    case "campaignRoot":
      return t(
        "Campaign root directory (campaignRoot)",
        "任务根目录 (campaignRoot)",
      );
    case "goal":
      return t("Campaign goal (goal)", "任务目标 (goal)");
    case "idempotencyKey":
      return t("Idempotency key (idempotencyKey)", "幂等键 (idempotencyKey)");
    case "maxActions":
      return t(
        "Max bounded actions (maxActions)",
        "最大推进步数 (maxActions)",
      );
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
    default:
      return key;
  }
}

export function executionFieldLabel(key: string, t: Translate): string {
  switch (key) {
    case "sshHost":
      return t(
        "SSH host alias or address (sshHost)",
        "SSH 主机别名或地址 (sshHost)",
      );
    case "sshPort":
      return t("SSH port (sshPort)", "SSH 端口 (sshPort)");
    case "sshUser":
      return t("SSH user (sshUser)", "SSH 用户名 (sshUser)");
    case "remoteWorkspace":
      return t(
        "Remote workspace path (remoteWorkspace)",
        "远端工作区路径 (remoteWorkspace)",
      );
    case "remotePython":
      return t(
        "Remote Python executable (remotePython)",
        "远端 Python 路径 (remotePython)",
      );
    default:
      return key;
  }
}

export function runtimeFieldLabel(key: string, t: Translate): string {
  switch (key) {
    case "cli":
      return t("Runtime CLI executable", "运行时 CLI 可执行路径");
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
        hint: t(
          "Activate extended runtime hooks for new sessions",
          "为新会话启用扩展运行时处理管线",
        ),
      };
    case "actionFusion":
      return {
        title: t("Action fusion", "动作融合 (Action fusion)"),
        hint: t(
          "Combine sequential compatible tool actions",
          "合并连续且兼容的工具执行步骤",
        ),
      };
    case "observationPack":
      return {
        title: t("Observation pack", "观测打包 (Observation pack)"),
        hint: t(
          "Structure verbose tool observations compactly",
          "结构化整理冗长的环境观测输出",
        ),
      };
    case "evidencePreservingReducer":
      return {
        title: t("Evidence-preserving reducer", "保真日志归约"),
        hint: t(
          "Summarize long logs while retaining key receipts",
          "在保留关键证据回执的前提下归约长日志",
        ),
      };
    case "onlineContextCompact":
      return {
        title: t("Online context compaction", "在线上下文压缩"),
        hint: t(
          "Compact long session history between active turns",
          "在长任务轮次间自动压缩历史上下文",
        ),
      };
  }
}

export function stateTone(
  raw: string,
): "active" | "success" | "warn" | "danger" | "neutral" {
  const s = (raw || "").toLowerCase();
  if (["running", "streaming", "busy", "working", "started"].includes(s))
    return "active";
  if (
    ["completed", "succeeded", "done", "ok", "ready", "open", "idle"].includes(
      s,
    )
  )
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
    case "cancelled":
      return t("cancelled", "已取消");
    case "unknown":
      return t("unknown", "效果待核对");
    default:
      return raw;
  }
}
