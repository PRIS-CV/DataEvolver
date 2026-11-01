import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type {
  Agent,
  Bootstrap,
  Config,
  Job,
  Message,
  Provider,
  Session,
  ImageResult,
  WorkbenchEvent,
} from "./types";
import "./style.css";

async function api<T = unknown>(path: string, value?: unknown): Promise<T> {
  const response = await fetch(
    "/api/" + path,
    value === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(value),
        },
  );
  if (response.status === 401)
    throw new Error(
      "Open the startup URL printed by launch.sh / 请打开 launch.sh 输出的登录链接",
    );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      (data.error || response.statusText) +
        (data.receipt ? ` · receipt: ${data.receipt}` : ""),
    );
  return data;
}
function Field({
  label,
  value,
  set,
  type = "text",
  placeholder = "",
}: {
  label: string;
  value: string | number;
  set: (value: string) => void;
  type?: string;
  placeholder?: string;
}) {
  const id = React.useId();
  return (
    <label className="field" htmlFor={id}>
      <span>{label}</span>
      <input
        id={id}
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(e) => set(e.target.value)}
        autoComplete="off"
      />
    </label>
  );
}
function AgentSelect({
  agent,
  providers,
  update,
  label,
}: {
  agent: Agent;
  providers: Provider[];
  update: (agent: Agent) => void;
  label: string;
}) {
  return (
    <fieldset className="agent-select">
      <legend>{label}</legend>
      <label>
        Provider
        <select
          value={agent.provider}
          onChange={(e) =>
            update({ ...agent, provider: e.target.value, model: "" })
          }
        >
          <option value="">—</option>
          {providers.map((p) => (
            <option key={p.id}>{p.id}</option>
          ))}
        </select>
      </label>
      <label>
        Model
        <select
          value={agent.model}
          onChange={(e) => update({ ...agent, model: e.target.value })}
        >
          <option value="">—</option>
          {providers
            .find((p) => p.id === agent.provider)
            ?.models.map((m) => (
              <option key={m.id}>{m.id}</option>
            ))}
        </select>
      </label>
      <label>
        Thinking
        <select
          value={agent.thinking}
          onChange={(e) => update({ ...agent, thinking: e.target.value })}
        >
          {["off", "minimal", "low", "medium", "high", "xhigh", "max"].map(
            (x) => (
              <option key={x}>{x}</option>
            ),
          )}
        </select>
      </label>
    </fieldset>
  );
}
function Settings({
  initial,
  credentials,
  zh,
  save,
}: {
  initial: Config;
  credentials: string[];
  zh: boolean;
  save: (config: Config, secrets: Record<string, string>) => Promise<boolean>;
}) {
  const [config, setConfig] = useState(initial);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [keyName, setKeyName] = useState("");
  const [keyValue, setKeyValue] = useState("");
  const [saving, setSaving] = useState(false);
  const t = (en: string, cn: string) => (zh ? cn : en);
  const updateProvider = (index: number, value: Provider) =>
    setConfig({
      ...config,
      providers: config.providers.map((p, i) => (i === index ? value : p)),
    });
  const updateMap = (
    section: "paths" | "execution" | "pi",
    key: string,
    value: string,
  ) =>
    setConfig({ ...config, [section]: { ...config[section], [key]: value } });
  return (
    <form
      className="settings"
      onSubmit={async (e) => {
        e.preventDefault();
        setSaving(true);
        try {
          if (await save(config, secrets)) {
            setSecrets({});
            setKeyValue("");
          }
        } finally {
          setSaving(false);
        }
      }}
    >
      <header className="section-heading">
        <div>
          <span className="eyebrow">CONFIGURATION</span>
          <h1>{t("Your models. Your machines.", "你的模型，你的工作站。")}</h1>
          <p>
            {t(
              "Settings apply to new sessions. Existing sessions retain their configuration snapshot.",
              "新设置用于新会话，已有会话保留创建时的配置。",
            )}
          </p>
        </div>
        <button className="primary" disabled={saving}>
          {saving ? "…" : t("Save configuration", "保存配置")}
        </button>
      </header>
      <section className="card">
        <h2>{t("Model providers", "模型服务")}</h2>
        <p>
          {t(
            "Use a supported Pi protocol. Model IDs and endpoints are supplied by your deployment.",
            "选择 Pi 协议，模型 ID 和地址由你的部署提供。",
          )}
        </p>
        {config.providers.map((p, i) => (
          <fieldset className="provider" key={i}>
            <legend>{p.id || `Provider ${i + 1}`}</legend>
            <div className="fields">
              <Field
                label="Provider ID"
                value={p.id}
                set={(id) => updateProvider(i, { ...p, id })}
              />
              <label>
                API
                <select
                  value={p.api}
                  onChange={(e) =>
                    updateProvider(i, { ...p, api: e.target.value })
                  }
                >
                  {[
                    "openai-completions",
                    "openai-responses",
                    "anthropic-messages",
                    "google-generative-ai",
                  ].map((a) => (
                    <option key={a}>{a}</option>
                  ))}
                </select>
              </label>
              <Field
                label="Base URL"
                value={p.baseUrl}
                set={(baseUrl) => updateProvider(i, { ...p, baseUrl })}
              />
              <Field
                label={t(
                  "Key environment name (empty for no auth)",
                  "密钥环境变量名（无鉴权留空）",
                )}
                value={p.apiKeyEnv}
                set={(apiKeyEnv) => updateProvider(i, { ...p, apiKeyEnv })}
              />
            </div>
            {p.models.map((model, j) => {
              const patch = (value: Partial<typeof model>) =>
                updateProvider(i, {
                  ...p,
                  models: p.models.map((m, k) =>
                    j === k ? { ...m, ...value } : m,
                  ),
                });
              return (
                <div className="model-row" key={j}>
                  <Field
                    label="Model ID"
                    value={model.id}
                    set={(id) => patch({ id })}
                  />
                  <Field
                    label="Context tokens"
                    type="number"
                    value={model.contextWindow}
                    set={(n) => patch({ contextWindow: Number(n) })}
                  />
                  <Field
                    label="Max output"
                    type="number"
                    value={model.maxTokens}
                    set={(n) => patch({ maxTokens: Number(n) })}
                  />
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={model.reasoning}
                      onChange={(e) => patch({ reasoning: e.target.checked })}
                    />
                    Reasoning
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={model.input.includes("image")}
                      onChange={(e) =>
                        patch({
                          input: e.target.checked
                            ? ["text", "image"]
                            : ["text"],
                        })
                      }
                    />
                    Vision
                  </label>
                  <button
                    type="button"
                    onClick={() =>
                      updateProvider(i, {
                        ...p,
                        models: p.models.filter((_, k) => k !== j),
                      })
                    }
                  >
                    ×
                  </button>
                </div>
              );
            })}
            <div className="actions">
              <button
                type="button"
                onClick={() =>
                  updateProvider(i, {
                    ...p,
                    models: [
                      ...p.models,
                      {
                        id: "",
                        reasoning: true,
                        input: ["text", "image"],
                        contextWindow: 128000,
                        maxTokens: 16000,
                      },
                    ],
                  })
                }
              >
                {t("Add model", "添加模型")}
              </button>
              <button
                type="button"
                onClick={() =>
                  setConfig({
                    ...config,
                    providers: config.providers.filter((_, k) => i !== k),
                  })
                }
              >
                {t("Remove provider", "移除服务")}
              </button>
            </div>
          </fieldset>
        ))}
        <button
          type="button"
          onClick={() =>
            setConfig({
              ...config,
              providers: [
                ...config.providers,
                {
                  id: "",
                  api: "openai-completions",
                  baseUrl: "",
                  apiKeyEnv: "",
                  models: [
                    {
                      id: "",
                      reasoning: true,
                      input: ["text", "image"],
                      contextWindow: 128000,
                      maxTokens: 16000,
                    },
                  ],
                },
              ],
            })
          }
        >
          {t("Add provider", "添加服务")}
        </button>
      </section>
      <section className="card">
        <h2>{t("Main and child agents", "主 Agent 与子 Agent")}</h2>
        <div className="fields">
          {(["main", "child"] as const).map((role) => (
            <AgentSelect
              key={role}
              agent={config.agents[role]}
              providers={config.providers}
              label={role}
              update={(agent) =>
                setConfig({
                  ...config,
                  agents: { ...config.agents, [role]: agent },
                })
              }
            />
          ))}
        </div>
      </section>
      <section className="card">
        <h2>
          {t(
            "Image generation, editing and prompt enhancement",
            "图像生成、编辑与提示增强",
          )}
        </h2>
        <p>
          {t(
            "Image APIs use /images/generations and multipart /images/edits with b64_json responses. PE uses /chat/completions.",
            "图像接口使用 /images/generations 与 multipart /images/edits，响应需包含 b64_json。PE 使用 /chat/completions。",
          )}
        </p>
        {(["generation", "edit", "promptEnhancer"] as const).map((kind) => (
          <fieldset key={kind}>
            <legend>{kind}</legend>
            <div className="fields">
              {(["baseUrl", "model", "apiKeyEnv"] as const).map((key) => (
                <Field
                  key={key}
                  label={key}
                  value={config.image[kind][key]}
                  set={(value) =>
                    setConfig({
                      ...config,
                      image: {
                        ...config.image,
                        [kind]: { ...config.image[kind], [key]: value },
                      },
                    })
                  }
                />
              ))}
            </div>
          </fieldset>
        ))}
      </section>
      <section className="card">
        <h2>{t("Workbench host paths", "工作台主机路径")}</h2>
        <p>
          {t(
            "Use executable files for Python, Blender and UE5. These paths are available to the agent and native subprocesses.",
            "Python、Blender 和 UE5 填写可执行文件路径。Agent 与原生进程均可读取这些设置。",
          )}
        </p>
        <div className="fields">
          {Object.entries(config.paths).map(([key, value]) => (
            <Field
              key={key}
              label={key}
              value={value}
              set={(value) => updateMap("paths", key, value)}
            />
          ))}
        </div>
      </section>
      <section className="card">
        <h2>{t("Execution host / OpenSSH", "执行主机 / OpenSSH")}</h2>
        <label>
          Target
          <select
            value={config.execution.target}
            onChange={(e) => updateMap("execution", "target", e.target.value)}
          >
            <option value="local">local</option>
            <option value="ssh">ssh</option>
          </select>
        </label>
        <p>
          {t(
            "SSH uses your host config and ssh-agent, strict host keys and BatchMode. Remote paths refer to the server. Provision its checkout and environment separately.",
            "SSH 使用宿主机配置与 ssh-agent，启用严格主机密钥检查和 BatchMode。远端路径均指服务器路径；请先准备远端源码与环境。",
          )}
        </p>
        <div className="fields">
          {Object.entries(config.execution)
            .filter(([key]) => key !== "target")
            .map(([key, value]) => (
              <Field
                key={key}
                label={key}
                value={value}
                set={(value) => updateMap("execution", key, value)}
              />
            ))}
        </div>
      </section>
      <section className="card">
        <h2>Pi / SoL-Pi</h2>
        <div className="fields">
          {Object.entries(config.pi).map(([key, value]) => (
            <Field
              key={key}
              label={key}
              value={value}
              set={(value) => updateMap("pi", key, value)}
            />
          ))}
        </div>
        <div className="checks">
          {(
            [
              "enabled",
              "actionFusion",
              "observationPack",
              "evidencePreservingReducer",
              "onlineContextCompact",
            ] as const
          ).map((key) => (
            <label className="check" key={key}>
              <input
                type="checkbox"
                checked={config.sol[key]}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    sol: { ...config.sol, [key]: e.target.checked },
                  })
                }
              />
              {key}
            </label>
          ))}
        </div>
        <p>
          {t(
            "Reducer sends eligible logs to its configured model. Compaction can make model calls and continue the task. Set both explicitly.",
            "Reducer 会将符合条件的日志发送给配置的模型。压缩可能调用模型并继续任务，请显式配置。",
          )}
        </p>
        <div className="fields">
          {(["reducerProvider", "reducerModel"] as const).map((key) => (
            <Field
              key={key}
              label={key}
              value={config.sol[key]}
              set={(value) =>
                setConfig({ ...config, sol: { ...config.sol, [key]: value } })
              }
            />
          ))}
          <Field
            label="cacheWriteReadRatio"
            type="number"
            value={config.sol.cacheWriteReadRatio}
            set={(value) =>
              setConfig({
                ...config,
                sol: { ...config.sol, cacheWriteReadRatio: Number(value) },
              })
            }
          />
        </div>
      </section>
      <section className="card">
        <h2>{t("Private credentials", "私有凭据")}</h2>
        <p>
          {t(
            "Keys are write-only. Leave inputs empty to keep stored keys. SSH keys stay in OpenSSH, not this form.",
            "密钥仅写入，页面不会读取已保存的值。空白输入保留旧值；SSH 私钥由 OpenSSH 管理。",
          )}
        </p>
        <div className="fields">
          <Field label="Environment name" value={keyName} set={setKeyName} />
          <Field
            label="API key"
            type="password"
            value={keyValue}
            set={setKeyValue}
          />
          <button
            type="button"
            disabled={!keyName || !keyValue}
            onClick={() => {
              setSecrets({ ...secrets, [keyName]: keyValue });
              setKeyValue("");
              setKeyName("");
            }}
          >
            {t("Stage key for save", "暂存密钥")}
          </button>
        </div>
        <div className="tags">
          {[...new Set([...credentials, ...Object.keys(secrets)])].map(
            (name) => (
              <span key={name}>
                {name}:{" "}
                {secrets[name] === ""
                  ? t("remove on save", "保存时删除")
                  : t("configured", "已配置")}{" "}
                <button
                  type="button"
                  aria-label={`Remove ${name}`}
                  onClick={() => setSecrets({ ...secrets, [name]: "" })}
                >
                  ×
                </button>
              </span>
            ),
          )}
        </div>
      </section>
    </form>
  );
}
const commandFields: Record<string, string[]> = {
  "compile-plan": ["plan", "profile", "evaluator", "output", "campaign"],
  "worker-compile": ["profile", "output", "campaign", "campaignRoot"],
  create: ["goal", "campaignRoot", "idempotencyKey"],
  step: ["campaign", "campaignRoot", "maxActions"],
  status: ["campaign", "campaignRoot"],
  explain: ["campaign", "campaignRoot"],
  replay: ["campaign", "campaignRoot"],
  pause: ["campaign", "campaignRoot"],
  resume: ["campaign", "campaignRoot"],
};
function App() {
  const [zh, setZh] = useState(navigator.language.startsWith("zh"));
  const t = (en: string, cn: string) => (zh ? cn : en);
  const [data, setData] = useState<Bootstrap>();
  const [tab, setTab] = useState("chat");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [selected, setSelected] = useState("");
  const selectedRef = useRef("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [partial, setPartial] = useState("");
  const [state, setState] = useState("stopped");
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState("prompt");
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState<Agent>({
    provider: "",
    model: "",
    thinking: "high",
  });
  const [command, setCommand] = useState("status");
  const [values, setValues] = useState<Record<string, string>>({
    maxActions: "1",
    campaignRoot: ".dataevolver/campaigns",
  });
  const [job, setJob] = useState<Job>();
  const [imageKind, setImageKind] = useState("generation");
  const [imagePrompt, setImagePrompt] = useState("");
  const [imagePath, setImagePath] = useState("");
  const [imageResult, setImageResult] = useState<ImageResult>();
  const run = async (fn: () => Promise<unknown>) => {
    setError("");
    try {
      return await fn();
    } catch (error) {
      setError((error as Error).message);
    }
  };
  const load = async () => setData(await api<Bootstrap>("bootstrap"));
  const refreshSession = async (id: string) => {
    const details = await api<{
      messages: Message[];
      state: string;
      session: Session;
    }>(`sessions/${id}`);
    if (selectedRef.current === id) {
      setMessages(details.messages);
      setState(details.state);
      setModel(details.session.profile);
      setPartial("");
    }
  };
  useEffect(() => {
    void run(load);
    const events = new EventSource("/api/events");
    events.onmessage = (e) => {
      const item = JSON.parse(e.data) as WorkbenchEvent;
      if (item.type === "connected") {
        setNotice("");
        void run(async () => {
          await load();
          if (selectedRef.current) await refreshSession(selectedRef.current);
        });
      }
      if (item.type === "pi_event" && item.sessionId === selectedRef.current) {
        setState(item.state);
        if (
          item.event.type === "message_update" &&
          item.event.assistantMessageEvent.type === "text_delta"
        ) {
          const delta = item.event.assistantMessageEvent.delta;
          setPartial((text) => text + delta);
        }
        if (item.event.type === "message_end") {
          const message = item.event.message as Message;
          setMessages((previous) => [...previous, message]);
          if (message.role === "assistant") setPartial("");
        }
        if (item.event.type === "agent_settled")
          void run(() => refreshSession(item.sessionId));
      }
      if (item.type === "session_error") setError(item.error);
      if (
        [
          "session_created",
          "session_opened",
          "session_stopped",
          "job_started",
          "job_finished",
        ].includes(item.type)
      )
        void run(load);
      if (item.type === "job_finished")
        setJob((previous) =>
          previous && previous.id === item.job.id
            ? { ...previous, ...item.job }
            : previous,
        );
      if (item.type === "job_output")
        setJob((previous) =>
          previous && previous.id === item.jobId
            ? {
                ...previous,
                output: ((previous.output || "") + item.text).slice(-200000),
              }
            : previous,
        );
    };
    events.onerror = () => setNotice("Reconnecting… / 连接恢复中…");
    return () => events.close();
  }, []);
  const choose = async (id: string) => {
    selectedRef.current = id;
    setSelected(id);
    setTab("chat");
    setMessages([]);
    setPartial("");
    await refreshSession(id);
  };
  const sendCommand = (input: unknown) =>
    api(`sessions/${selected}/commands`, input);
  return (
    <div className="shell">
      <a className="skip" href="#workspace">
        {t("Skip to workspace", "跳转到工作区")}
      </a>
      <aside className="sidebar">
        <a className="brand" href="/">
          <img src="/logo.png" alt="" />
          <span>
            DataEvolver<small>WORKBENCH</small>
          </span>
        </a>
        <button
          className="new-session"
          onClick={() =>
            void run(async () => {
              const session = await api<Session>("sessions", {});
              await choose(session.id);
              await load();
            })
          }
        >
          ＋ {t("New conversation", "新建对话")}
        </button>
        <nav aria-label="Workspace">
          <button
            className={tab === "chat" ? "active" : ""}
            onClick={() => setTab("chat")}
          >
            {t("Conversations", "对话")}
          </button>
          <button
            className={tab === "harness" ? "active" : ""}
            onClick={() => setTab("harness")}
          >
            Harness
          </button>
          <button
            className={tab === "images" ? "active" : ""}
            onClick={() => setTab("images")}
          >
            {t("Image studio", "图像工作区")}
          </button>
          <button
            className={tab === "settings" ? "active" : ""}
            onClick={() => setTab("settings")}
          >
            {t("Settings", "设置")}
          </button>
        </nav>
        <span className="eyebrow">SESSIONS</span>
        <div className="session-list">
          {data?.sessions.map((s) => (
            <button
              className={s.id === selected ? "active" : ""}
              key={s.id}
              onClick={() => void run(() => choose(s.id))}
            >
              <span>
                {s.parentId ? "↳ " : ""}
                {s.name}
              </span>
              <small>
                {s.profile.model} · {s.state}
              </small>
            </button>
          ))}
        </div>
        <footer>
          <span>Pi + SoL-Pi</span>
          <button onClick={() => setZh(!zh)}>
            {zh ? "English" : "简体中文"}
          </button>
        </footer>
      </aside>
      <main id="workspace">
        <header className="topbar">
          <span>{t("Private workspace", "私有工作区")}</span>
          <span className="status">
            {notice ||
              t("Host permissions · single user", "宿主机权限 · 单用户")}
          </span>
        </header>
        {error && (
          <div className="error" role="alert">
            {error}
            <button onClick={() => setError("")} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}
        {!data ? (
          <p className="loading">{t("Loading configuration…", "读取配置…")}</p>
        ) : (
          <>
            {tab === "settings" && (
              <Settings
                key={JSON.stringify(data.config)}
                initial={data.config}
                credentials={data.credentials}
                zh={zh}
                save={async (config, secrets) => {
                  let saved = false;
                  await run(async () => {
                    const view = await api<
                      Pick<Bootstrap, "config" | "credentials">
                    >("config", { config, secrets });
                    setData((previous) =>
                      previous ? { ...previous, ...view } : previous,
                    );
                    setNotice(t("Configuration saved", "配置已保存"));
                    saved = true;
                  });
                  return saved;
                }}
              />
            )}
            {tab === "chat" && (
              <div className="conversation">
                <header className="chat-heading">
                  <div>
                    <span className="eyebrow">AGENT SESSION</span>
                    <h1>
                      {data.sessions.find((s) => s.id === selected)?.name ||
                        t("What will you build?", "这次构建什么？")}
                    </h1>
                  </div>
                  {selected && <span className="pill">{state}</span>}
                </header>
                {selected ? (
                  <>
                    <details className="model-switch">
                      <summary>
                        {model.provider} / {model.model} · {model.thinking}
                      </summary>
                      <AgentSelect
                        label={t("Session model", "会话模型")}
                        agent={model}
                        providers={data.config.providers}
                        update={setModel}
                      />
                      <button
                        onClick={() =>
                          void run(() =>
                            sendCommand({ command: "set_model", ...model }),
                          )
                        }
                      >
                        {t("Switch between turns", "在轮次之间切换")}
                      </button>
                    </details>
                    <div className="messages" aria-live="polite">
                      {messages.map((m, i) => (
                        <article key={i} className={`message ${m.role}`}>
                          <span className="message-role">
                            {m.role === "toolResult" ? m.toolName : m.role}
                            {m.isError ? " · error" : ""}
                          </span>
                          {typeof m.content === "string" ? (
                            <pre>{m.content}</pre>
                          ) : (
                            m.content?.map((block, k) =>
                              block.type === "text" ? (
                                <pre key={k}>{block.text}</pre>
                              ) : block.type === "toolCall" ? (
                                <p key={k}>↗ {block.name}</p>
                              ) : block.type === "image" ? (
                                <img
                                  key={k}
                                  className="message-image"
                                  src={`data:${block.mimeType};base64,${block.data}`}
                                  alt="Agent artifact"
                                />
                              ) : null,
                            )
                          )}
                        </article>
                      ))}
                      {partial && (
                        <article className="message assistant">
                          <span className="message-role">
                            assistant · streaming
                          </span>
                          <pre>{partial}</pre>
                        </article>
                      )}
                    </div>
                    <form
                      className="composer"
                      onSubmit={(e) => {
                        e.preventDefault();
                        setBusy(true);
                        void run(async () => {
                          await sendCommand({ command: mode, message: prompt });
                          setPrompt("");
                        }).finally(() => setBusy(false));
                      }}
                    >
                      <label className="sr-only" htmlFor="prompt">
                        Message
                      </label>
                      <textarea
                        id="prompt"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder={t(
                          "Describe a task or steer the current run…",
                          "描述任务，或调整当前任务方向…",
                        )}
                        required
                      />
                      <div className="composer-actions">
                        <select
                          aria-label="Message mode"
                          value={mode}
                          onChange={(e) => setMode(e.target.value)}
                        >
                          <option value="prompt">Prompt</option>
                          <option value="steer">Steer</option>
                          <option value="follow_up">Follow-up</option>
                        </select>
                        <button
                          type="button"
                          onClick={() =>
                            void run(() => sendCommand({ command: "resume" }))
                          }
                        >
                          {t("Reopen session", "恢复对话")}
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            void run(() => sendCommand({ command: "abort" }))
                          }
                        >
                          {t("Stop agent", "停止 Agent")}
                        </button>
                        <button
                          className="primary"
                          disabled={busy || !prompt.trim()}
                        >
                          {t("Send", "发送")} ↑
                        </button>
                      </div>
                    </form>
                  </>
                ) : (
                  <div className="welcome">
                    <div className="orbit">
                      D<span>↗</span>
                    </div>
                    <h2>
                      {t(
                        "A plan becomes a traceable workflow.",
                        "让计划成为有据可查的工作流。",
                      )}
                    </h2>
                    <p>
                      {t(
                        "Configure an agent, open a conversation, then compile and advance your task through the Harness.",
                        "配置 Agent 并开始对话，然后通过 Harness 编译与推进任务。",
                      )}
                    </p>
                    <div className="welcome-cards">
                      <button onClick={() => setTab("settings")}>
                        <b>01 / {t("Connect", "连接")}</b>
                        <span>
                          {t(
                            "Models, keys and workstations",
                            "模型、密钥和工作站",
                          )}
                        </span>
                      </button>
                      <button onClick={() => setTab("harness")}>
                        <b>02 / {t("Execute", "执行")}</b>
                        <span>
                          {t(
                            "Compile, create and bounded step",
                            "编译、创建与有界推进",
                          )}
                        </span>
                      </button>
                      <button onClick={() => setTab("images")}>
                        <b>03 / {t("Inspect", "查看")}</b>
                        <span>
                          {t(
                            "Images, receipts and artifacts",
                            "图像、回执与产物",
                          )}
                        </span>
                      </button>
                    </div>
                    <p className="quiet">
                      {t(
                        "Agents can edit files and run host commands. SoL plans are conversation plans; Campaign state belongs to the Harness.",
                        "Agent 可以编辑文件并执行宿主机命令。SoL 计划属于对话；Campaign 状态由 Harness 负责。",
                      )}
                    </p>
                  </div>
                )}
              </div>
            )}
            {tab === "harness" && (
              <div className="page">
                <header className="section-heading">
                  <div>
                    <span className="eyebrow">CAMPAIGN CONTROL</span>
                    <h1>Harness</h1>
                    <p>
                      {data.config.execution.target === "ssh"
                        ? `${data.config.execution.sshHost} · ${data.config.execution.remoteWorkspace}`
                        : data.config.paths.workspace}
                    </p>
                  </div>
                </header>
                <section className="card">
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      void run(async () => {
                        const inputs = Object.fromEntries(
                          commandFields[command].map((k) => [
                            k,
                            k === "maxActions"
                              ? Number(values[k] || 1)
                              : values[k] || "",
                          ]),
                        );
                        const row = await api<Job>("harness", {
                          command,
                          values: inputs,
                        });
                        setJob({ ...row, output: "" });
                        await load();
                      });
                    }}
                  >
                    <label>
                      Command
                      <select
                        value={command}
                        onChange={(e) => setCommand(e.target.value)}
                      >
                        {Object.keys(commandFields).map((c) => (
                          <option key={c}>{c}</option>
                        ))}
                      </select>
                    </label>
                    <div className="fields">
                      {commandFields[command].map((key) => (
                        <Field
                          key={key}
                          label={key}
                          value={values[key] || ""}
                          type={key === "maxActions" ? "number" : "text"}
                          set={(value) =>
                            setValues({ ...values, [key]: value })
                          }
                        />
                      ))}
                    </div>
                    {command === "resume" && (
                      <p className="warning">
                        {t(
                          "Campaign resume immediately calls run() and continues native work.",
                          "Campaign resume 会立即调用 run() 并继续原生任务。",
                        )}
                      </p>
                    )}
                    <button className="primary">
                      {t("Submit command", "提交命令")}
                    </button>
                  </form>
                </section>
                <section className="card">
                  <h2>{t("Jobs and receipts", "任务与回执")}</h2>
                  <div className="job-list">
                    {data.jobs.map((j) => (
                      <button
                        key={j.id}
                        onClick={() =>
                          void run(async () =>
                            setJob(await api<Job>("jobs/" + j.id)),
                          )
                        }
                      >
                        <span>{j.command}</span>
                        <span className="pill">{j.status}</span>
                        <small>
                          {j.id.slice(0, 8)} · {j.target}
                        </small>
                      </button>
                    ))}
                  </div>
                  {job && (
                    <>
                      <h3>
                        {job.command} · {job.status}
                      </h3>
                      <button
                        onClick={() =>
                          void run(async () =>
                            setJob(await api<Job>("jobs/" + job.id)),
                          )
                        }
                      >
                        {t("Refresh retained output", "刷新保留的输出")}
                      </button>
                      <pre className="log">{job.output || "…"}</pre>
                    </>
                  )}
                </section>
              </div>
            )}
            {tab === "images" && (
              <div className="page">
                <header className="section-heading">
                  <div>
                    <span className="eyebrow">IMAGE STUDIO</span>
                    <h1>
                      {t("Generate. Refine. Inspect.", "生成、修订与查看。")}
                    </h1>
                    <p>
                      {t(
                        "Paths and output refer to the workbench host. Calls retain an intent and receipt and are never retried automatically.",
                        "输入和产物路径均指工作台主机。调用会保留意图与回执，失败后不自动重试。",
                      )}
                    </p>
                  </div>
                </header>
                <section className="card">
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      setBusy(true);
                      void run(async () =>
                        setImageResult(
                          await api<ImageResult>("image", {
                            kind: imageKind,
                            prompt: imagePrompt,
                            image: imagePath || undefined,
                          }),
                        ),
                      ).finally(() => setBusy(false));
                    }}
                  >
                    <label>
                      Operation
                      <select
                        value={imageKind}
                        onChange={(e) => setImageKind(e.target.value)}
                      >
                        <option value="generation">Generation</option>
                        <option value="edit">Editing</option>
                        <option value="promptEnhancer">
                          Prompt enhancement
                        </option>
                      </select>
                    </label>
                    <Field
                      label="Input image (editing / vision PE)"
                      value={imagePath}
                      set={setImagePath}
                    />
                    <label>
                      Prompt
                      <textarea
                        value={imagePrompt}
                        onChange={(e) => setImagePrompt(e.target.value)}
                        required
                      />
                    </label>
                    <button className="primary" disabled={busy}>
                      {busy ? "…" : t("Submit", "提交")}
                    </button>
                  </form>
                </section>
                {imageResult && (
                  <section className="card">
                    {imageResult.path && (
                      <img
                        className="artifact"
                        src={
                          "/api/artifact?path=" +
                          encodeURIComponent(imageResult.path)
                        }
                        alt="Generated artifact"
                      />
                    )}
                    <pre className="log">
                      {JSON.stringify(imageResult, null, 2)}
                    </pre>
                  </section>
                )}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
