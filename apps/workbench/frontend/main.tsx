import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type {
  Agent,
  Bootstrap,
  Config,
  ImageResult,
  Job,
  Message,
  Provider,
  Session,
  WorkbenchEvent,
} from "./types";
import {
  executionFieldLabel,
  formatState,
  getSettingsCategories,
  harnessCommandMeta,
  harnessFieldLabel,
  IconArrowUpRight,
  IconChat,
  IconClose,
  IconGlobe,
  IconHarness,
  IconImage,
  IconKey,
  IconPlus,
  IconSend,
  IconSettings,
  imageRouteMeta,
  pathFieldLabel,
  runtimeFieldLabel,
  runtimeToggleMeta,
  type SettingsCategory,
  stateTone,
  type Translate,
} from "./ui";
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
  mono = false,
}: {
  label: string;
  value: string | number;
  set: (value: string) => void;
  type?: string;
  placeholder?: string;
  mono?: boolean;
}) {
  const id = React.useId();
  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{label}</span>
      <input
        id={id}
        className={mono ? "mono-input" : undefined}
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(e) => set(e.target.value)}
        autoComplete="off"
      />
    </label>
  );
}

const THINKING_LEVELS = [
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;

function AgentSelect({
  agent,
  providers,
  update,
  label,
  t = (en) => en,
}: {
  agent: Agent;
  providers: Provider[];
  update: (agent: Agent) => void;
  label: string;
  t?: Translate;
}) {
  const activeProvider = providers.find((p) => p.id === agent.provider);
  const hasProviderInList = providers.some((p) => p.id === agent.provider);
  const hasModelInList = Boolean(
    activeProvider?.models.some((m) => m.id === agent.model),
  );
  return (
    <fieldset className="agent-select">
      <legend>{label}</legend>
      <label className="field">
        <span className="field-label">{t("Provider", "服务提供方")}</span>
        <select
          value={agent.provider}
          onChange={(e) =>
            update({ ...agent, provider: e.target.value, model: "" })
          }
        >
          <option value="">—</option>
          {agent.provider && !hasProviderInList && (
            <option value={agent.provider}>{agent.provider}</option>
          )}
          {providers.map((p) => (
            <option key={p.id} value={p.id}>
              {p.id}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span className="field-label">{t("Model", "模型")}</span>
        <select
          value={agent.model}
          onChange={(e) => update({ ...agent, model: e.target.value })}
        >
          <option value="">—</option>
          {agent.model && !hasModelInList && (
            <option value={agent.model}>{agent.model}</option>
          )}
          {activeProvider?.models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span className="field-label">{t("Thinking", "推理深度")}</span>
        <select
          value={agent.thinking}
          onChange={(e) => update({ ...agent, thinking: e.target.value })}
        >
          {THINKING_LEVELS.map((x) => (
            <option key={x} value={x}>
              {x}
            </option>
          ))}
        </select>
      </label>
    </fieldset>
  );
}

function Settings({
  initial,
  credentials,
  inspection,
  defaultCategory = "models",
  onCategoryChange,
  zh,
  save,
}: {
  initial: Config;
  credentials: string[];
  inspection?: Bootstrap["inspection"];
  defaultCategory?: SettingsCategory;
  onCategoryChange?: (category: SettingsCategory) => void;
  zh: boolean;
  save: (config: Config, secrets: Record<string, string>) => Promise<boolean>;
}) {
  const [config, setConfig] = useState(initial);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [keyName, setKeyName] = useState("");
  const [keyValue, setKeyValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [category, setCategoryState] =
    useState<SettingsCategory>(defaultCategory);
  const setCategory = (next: SettingsCategory) => {
    setCategoryState(next);
    onCategoryChange?.(next);
  };
  useEffect(() => {
    setCategoryState(defaultCategory);
  }, [defaultCategory]);
  const t: Translate = (en, cn) => (zh ? cn : en);

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

  const categories = getSettingsCategories(t);
  const allCredentialNames = [
    ...new Set([...credentials, ...Object.keys(secrets)]),
  ];
  const activeKeyCount = allCredentialNames.filter(
    (name) => secrets[name] !== "",
  ).length;
  const referencedEnvNames = [
    ...new Set(
      [
        ...config.providers.map((p) => p.apiKeyEnv),
        config.image.generation.apiKeyEnv,
        config.image.edit.apiKeyEnv,
        config.image.promptEnhancer.apiKeyEnv,
      ]
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  ];
  const hasUnsaved =
    Object.keys(secrets).length > 0 ||
    JSON.stringify(config) !== JSON.stringify(initial);

  const addNewProvider = () =>
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
    });

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
      <header className="section-heading settings-header">
        <div>
          <span className="eyebrow">
            {t("CONFIGURATION", "工作台配置")}
          </span>
          <h1>{t("Your models. Your machines.", "你的模型，你的工作站。")}</h1>
          <p>
            {t(
              "Settings apply to new sessions. Existing sessions retain their configuration snapshot.",
              "新设置用于新会话，已有会话保留创建时的配置快照。",
            )}
          </p>
        </div>
        <div className="heading-actions">
          {hasUnsaved && (
            <span className="pill pill-warn">
              {t("Unsaved edits", "有未保存更改")}
            </span>
          )}
          <button type="submit" className="primary" disabled={saving}>
            {saving
              ? t("Saving…", "保存中…")
              : t("Save configuration", "保存配置")}
          </button>
        </div>
      </header>

      <div className="settings-layout">
        <nav
          className="settings-nav"
          aria-label={t("Settings categories", "设置分类")}
        >
          {categories.map((cat) => {
            const active = category === cat.id;
            const badge =
              cat.id === "models"
                ? String(config.providers.length)
                : cat.id === "execution"
                  ? config.execution.target || "local"
                  : cat.id === "credentials"
                    ? String(activeKeyCount)
                    : null;
            return (
              <button
                key={cat.id}
                type="button"
                className={active ? "settings-tab active" : "settings-tab"}
                aria-current={active ? "true" : undefined}
                onClick={() => setCategory(cat.id)}
              >
                <span className="settings-tab-top">
                  <span className="settings-tab-title">{cat.label}</span>
                  {badge !== null && (
                    <span className="settings-tab-badge">{badge}</span>
                  )}
                </span>
                <small className="settings-tab-desc">{cat.summary}</small>
              </button>
            );
          })}
        </nav>

        <div className="settings-panel">
          {category === "models" && (
            <>
              <section className="card">
                <div className="card-head">
                  <div>
                    <h2>
                      {t("Main and child agents", "主 Agent 与子 Agent")}
                    </h2>
                    <p>
                      {t(
                        "Assign default provider, model and thinking budget for primary conversations and delegated child sessions.",
                        "为主对话与派生的子会话分别设置默认服务、模型与推理深度。",
                      )}
                    </p>
                  </div>
                </div>
                <div className="fields">
                  {(["main", "child"] as const).map((role) => (
                    <AgentSelect
                      key={role}
                      t={t}
                      agent={config.agents[role]}
                      providers={config.providers}
                      label={
                        role === "main"
                          ? t("Main agent (main)", "主 Agent (main)")
                          : t("Child agent (child)", "子 Agent (child)")
                      }
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
                <div className="card-head">
                  <div>
                    <h2>{t("Model providers", "模型服务")}</h2>
                    <p>
                      {t(
                        "Select a supported API protocol. Model IDs and endpoints are supplied by your deployment.",
                        "选择支持的 API 协议，模型 ID 和服务地址由你的部署提供。",
                      )}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="secondary-btn"
                    onClick={addNewProvider}
                  >
                    <IconPlus />
                    <span>{t("Add provider", "添加服务")}</span>
                  </button>
                </div>

                {config.providers.length === 0 ? (
                  <p className="empty-box">
                    {t(
                      "No providers configured yet. Add a provider to register endpoints and models.",
                      "尚未配置模型服务。点击「添加服务」注册接口地址与模型。",
                    )}
                  </p>
                ) : (
                  <div className="provider-stack">
                    {config.providers.map((p, i) => (
                      <fieldset className="provider-card" key={i}>
                        <legend className="sr-only">
                          {p.id || `Provider ${i + 1}`}
                        </legend>
                        <div className="provider-card-header">
                          <div className="provider-title-group">
                            <strong>
                              {p.id ||
                                t(`Provider ${i + 1}`, `模型服务 ${i + 1}`)}
                            </strong>
                            <span className="pill">{p.api}</span>
                          </div>
                          <button
                            type="button"
                            className="danger-ghost-btn"
                            onClick={() =>
                              setConfig({
                                ...config,
                                providers: config.providers.filter(
                                  (_, k) => i !== k,
                                ),
                              })
                            }
                          >
                            {t("Remove provider", "移除服务")}
                          </button>
                        </div>

                        <div className="fields">
                          <Field
                            label={t("Provider ID", "服务标识 (Provider ID)")}
                            value={p.id}
                            mono
                            set={(id) => updateProvider(i, { ...p, id })}
                          />
                          <label className="field">
                            <span className="field-label">
                              {t("API protocol", "API 协议")}
                            </span>
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
                                <option key={a} value={a}>
                                  {a}
                                </option>
                              ))}
                            </select>
                          </label>
                          <Field
                            label={t("Base URL", "接口地址 (Base URL)")}
                            value={p.baseUrl}
                            mono
                            set={(baseUrl) =>
                              updateProvider(i, { ...p, baseUrl })
                            }
                          />
                          <div className="field-with-link">
                            <Field
                              label={t(
                                "Key environment name (empty for no auth)",
                                "密钥环境变量名（无鉴权留空）",
                              )}
                              value={p.apiKeyEnv}
                              mono
                              set={(apiKeyEnv) =>
                                updateProvider(i, { ...p, apiKeyEnv })
                              }
                            />
                            <button
                              type="button"
                              className="inline-link-btn"
                              onClick={() => setCategory("credentials")}
                            >
                              <IconKey />
                              <span>
                                {t(
                                  "Manage keys in Credentials",
                                  "前往私有凭据配置密钥",
                                )}
                              </span>
                            </button>
                          </div>
                        </div>

                        <div className="models-subsection">
                          <div className="models-subheader">
                            <span>
                              {t("Models", "模型列表")} ({p.models.length})
                            </span>
                            <button
                              type="button"
                              className="secondary-btn compact-btn"
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
                              <IconPlus />
                              <span>{t("Add model", "添加模型")}</span>
                            </button>
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
                                  label={t("Model ID", "模型 ID")}
                                  value={model.id}
                                  mono
                                  set={(id) => patch({ id })}
                                />
                                <Field
                                  label={t("Context tokens", "上下文窗口")}
                                  type="number"
                                  value={model.contextWindow}
                                  set={(n) =>
                                    patch({ contextWindow: Number(n) })
                                  }
                                />
                                <Field
                                  label={t("Max output", "最大输出")}
                                  type="number"
                                  value={model.maxTokens}
                                  set={(n) => patch({ maxTokens: Number(n) })}
                                />
                                <div className="model-capabilities">
                                  <label className="check">
                                    <input
                                      type="checkbox"
                                      checked={model.reasoning}
                                      onChange={(e) =>
                                        patch({ reasoning: e.target.checked })
                                      }
                                    />
                                    <span>{t("Reasoning", "推理")}</span>
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
                                    <span>{t("Vision", "视觉")}</span>
                                  </label>
                                  <button
                                    type="button"
                                    className="icon-remove-btn"
                                    aria-label={t(
                                      `Remove model ${model.id || j + 1}`,
                                      `移除模型 ${model.id || j + 1}`,
                                    )}
                                    onClick={() =>
                                      updateProvider(i, {
                                        ...p,
                                        models: p.models.filter(
                                          (_, k) => k !== j,
                                        ),
                                      })
                                    }
                                  >
                                    <IconClose />
                                  </button>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </fieldset>
                    ))}
                  </div>
                )}
              </section>
            </>
          )}

          {category === "images" && (
            <section className="card">
              <div className="card-head">
                <div>
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
                </div>
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => setCategory("credentials")}
                >
                  <IconKey />
                  <span>{t("Credentials", "管理密钥")}</span>
                </button>
              </div>

              <div className="route-stack">
                {(["generation", "edit", "promptEnhancer"] as const).map(
                  (kind) => {
                    const meta = imageRouteMeta(kind, t);
                    return (
                      <fieldset key={kind} className="route-card">
                        <legend>
                          {meta.title} · <code>{kind}</code>
                        </legend>
                        <p className="route-protocol">{meta.protocol}</p>
                        <div className="fields fields-3">
                          <Field
                            label={t("Base URL", "接口地址 (baseUrl)")}
                            value={config.image[kind].baseUrl}
                            mono
                            set={(value) =>
                              setConfig({
                                ...config,
                                image: {
                                  ...config.image,
                                  [kind]: {
                                    ...config.image[kind],
                                    baseUrl: value,
                                  },
                                },
                              })
                            }
                          />
                          <Field
                            label={t("Model ID", "模型 ID (model)")}
                            value={config.image[kind].model}
                            mono
                            set={(value) =>
                              setConfig({
                                ...config,
                                image: {
                                  ...config.image,
                                  [kind]: {
                                    ...config.image[kind],
                                    model: value,
                                  },
                                },
                              })
                            }
                          />
                          <Field
                            label={t(
                              "Key environment name",
                              "密钥变量名 (apiKeyEnv)",
                            )}
                            value={config.image[kind].apiKeyEnv}
                            mono
                            set={(value) =>
                              setConfig({
                                ...config,
                                image: {
                                  ...config.image,
                                  [kind]: {
                                    ...config.image[kind],
                                    apiKeyEnv: value,
                                  },
                                },
                              })
                            }
                          />
                        </div>
                      </fieldset>
                    );
                  },
                )}
              </div>
            </section>
          )}

          {category === "paths" && (
            <section className="card">
              <div className="card-head">
                <div>
                  <h2>{t("Workbench host paths", "工作台主机路径")}</h2>
                  <p>
                    {t(
                      "Use executable files for Python, Blender and UE5. These paths are available to the agent and native subprocesses.",
                      "Python、Blender 和 UE5 填写可执行文件路径。Agent 与原生子进程均可读取这些设置。",
                    )}
                  </p>
                </div>
              </div>
              <div className="fields">
                {Object.entries(config.paths).map(([key, value]) => {
                  const detected = inspection?.paths?.[key];
                  return (
                    <div key={key} className="field-with-link">
                      <Field
                        label={pathFieldLabel(key, t)}
                        value={value}
                        mono
                        set={(next) => updateMap("paths", key, next)}
                      />
                      {detected && (
                        <span className="detected-hint">
                          {t("Host inspection:", "主机检测：")}{" "}
                          <code>{detected}</code>
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {category === "execution" && (
            <section className="card">
              <div className="card-head">
                <div>
                  <h2>{t("Execution host / OpenSSH", "执行主机 / OpenSSH")}</h2>
                  <p>
                    {t(
                      "Choose where Harness jobs and remote commands run. Local runs directly on the workbench host.",
                      "选择 Harness 任务与命令的执行位置。本地模式直接在工作台宿主机运行。",
                    )}
                  </p>
                </div>
              </div>

              <label className="field target-select-field">
                <span className="field-label">
                  {t("Execution target", "执行目标 (target)")}
                </span>
                <select
                  value={config.execution.target}
                  onChange={(e) =>
                    updateMap("execution", "target", e.target.value)
                  }
                >
                  <option value="local">
                    {t("Local workstation (local)", "本地工作站 (local)")}
                  </option>
                  <option value="ssh">
                    {t("Remote OpenSSH host (ssh)", "远端 OpenSSH 主机 (ssh)")}
                  </option>
                </select>
              </label>

              <div className="callout-note">
                {t(
                  "SSH uses your host config and ssh-agent, strict host keys and BatchMode. Remote paths refer to the server. Provision its checkout and environment separately.",
                  "SSH 使用宿主机配置与 ssh-agent，启用严格主机密钥检查和 BatchMode。远端路径均指服务器路径；请先单独准备远端源码与运行环境。",
                )}
              </div>

              {Boolean(inspection?.sshHosts?.length) &&
                "sshHost" in config.execution && (
                  <div className="env-suggestions">
                    <span className="env-suggestions-label">
                      {t(
                        "Detected OpenSSH host aliases:",
                        "检测到的本机 OpenSSH 主机别名：",
                      )}
                    </span>
                    <div className="chip-row">
                      {inspection!.sshHosts.map((host) => (
                        <button
                          key={host}
                          type="button"
                          className={
                            config.execution.sshHost === host
                              ? "chip-btn active"
                              : "chip-btn"
                          }
                          onClick={() => updateMap("execution", "sshHost", host)}
                        >
                          {host}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

              <div className="fields">
                {Object.entries(config.execution)
                  .filter(([key]) => key !== "target")
                  .map(([key, value]) => (
                    <Field
                      key={key}
                      label={executionFieldLabel(key, t)}
                      value={value}
                      mono
                      set={(next) => updateMap("execution", key, next)}
                    />
                  ))}
              </div>
            </section>
          )}

          {category === "runtime" && (
            <section className="card">
              <div className="card-head">
                <div>
                  <h2>{t("Agent runtime", "Agent 运行时")}</h2>
                  <p>
                    {t(
                      "Configure the underlying runtime executable, extension path, context compaction and log reduction options.",
                      "配置底层运行时可执行程序、扩展路径、上下文压缩与日志归约策略。",
                    )}
                  </p>
                </div>
              </div>

              <div className="fields">
                {Object.entries(config.pi).map(([key, value]) => (
                  <Field
                    key={key}
                    label={runtimeFieldLabel(key, t)}
                    value={value}
                    mono
                    set={(next) => updateMap("pi", key, next)}
                  />
                ))}
              </div>

              <div className="toggle-grid">
                {(
                  [
                    "enabled",
                    "actionFusion",
                    "observationPack",
                    "evidencePreservingReducer",
                    "onlineContextCompact",
                  ] as const
                ).map((key) => {
                  const meta = runtimeToggleMeta(key, t);
                  return (
                    <label className="toggle-tile" key={key}>
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
                      <span className="toggle-copy">
                        <strong>{meta.title}</strong>
                        <small>{meta.hint}</small>
                      </span>
                    </label>
                  );
                })}
              </div>

              <div className="callout-note">
                {t(
                  "Reducer sends eligible logs to its configured model. Compaction can make model calls and continue the task. Set both explicitly.",
                  "Reducer 会将符合条件的日志发送给配置的模型。上下文压缩可能触发模型调用并继续任务，请显式配置这两项。",
                )}
              </div>

              <div className="fields fields-3">
                <Field
                  label={t(
                    "Reducer provider (reducerProvider)",
                    "归约服务提供方 (reducerProvider)",
                  )}
                  value={config.sol.reducerProvider}
                  mono
                  set={(value) =>
                    setConfig({
                      ...config,
                      sol: { ...config.sol, reducerProvider: value },
                    })
                  }
                />
                <Field
                  label={t(
                    "Reducer model (reducerModel)",
                    "归约模型 (reducerModel)",
                  )}
                  value={config.sol.reducerModel}
                  mono
                  set={(value) =>
                    setConfig({
                      ...config,
                      sol: { ...config.sol, reducerModel: value },
                    })
                  }
                />
                <Field
                  label={t(
                    "Cache write/read ratio (cacheWriteReadRatio)",
                    "缓存写读比 (cacheWriteReadRatio)",
                  )}
                  type="number"
                  value={config.sol.cacheWriteReadRatio}
                  set={(value) =>
                    setConfig({
                      ...config,
                      sol: {
                        ...config.sol,
                        cacheWriteReadRatio: Number(value),
                      },
                    })
                  }
                />
              </div>
            </section>
          )}

          {category === "credentials" && (
            <section className="card">
              <div className="card-head">
                <div>
                  <h2>{t("Private credentials", "私有凭据")}</h2>
                  <p>
                    {t(
                      "Keys are write-only. Leave inputs empty to keep stored keys. SSH keys stay in OpenSSH, not this form.",
                      "密钥仅写入，页面不会读取已保存的值。空白输入保留旧值；SSH 私钥由 OpenSSH 管理，无需在此填写。",
                    )}
                  </p>
                </div>
              </div>

              {referencedEnvNames.length > 0 && (
                <div className="env-suggestions">
                  <span className="env-suggestions-label">
                    {t(
                      "Quick-fill from configured routes:",
                      "从当前服务配置快速填入变量名：",
                    )}
                  </span>
                  <div className="chip-row">
                    {referencedEnvNames.map((envName) => (
                      <button
                        key={envName}
                        type="button"
                        className={
                          keyName === envName ? "chip-btn active" : "chip-btn"
                        }
                        onClick={() => setKeyName(envName)}
                      >
                        {envName}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="credential-stage-box">
                <div className="fields">
                  <Field
                    label={t("Environment name", "环境变量名")}
                    value={keyName}
                    placeholder="API_KEY_ENV"
                    mono
                    set={setKeyName}
                  />
                  <Field
                    label={t("API key (write-only)", "API 密钥（仅写入）")}
                    type="password"
                    value={keyValue}
                    set={setKeyValue}
                  />
                </div>
                <div className="credential-stage-actions">
                  <button
                    type="button"
                    className="secondary-btn"
                    disabled={!keyName || !keyValue}
                    onClick={() => {
                      setSecrets({ ...secrets, [keyName]: keyValue });
                      setKeyValue("");
                      setKeyName("");
                    }}
                  >
                    <IconKey />
                    <span>{t("Stage key for save", "暂存密钥")}</span>
                  </button>
                  {keyName && keyValue && (
                    <span className="stage-hint">
                      {t(
                        "Stage this key first, then click Save configuration.",
                        "请先点击「暂存密钥」，再点击右上角「保存配置」。",
                      )}
                    </span>
                  )}
                </div>
              </div>

              <div className="credentials-list-section">
                <h3>{t("Stored and staged keys", "已配置与待保存密钥")}</h3>
                {allCredentialNames.length === 0 ? (
                  <p className="empty-box">
                    {t(
                      "No private credentials stored yet.",
                      "当前尚未保存任何私有密钥。",
                    )}
                  </p>
                ) : (
                  <div className="credential-rows">
                    {allCredentialNames.map((name) => {
                      const pendingDelete = secrets[name] === "";
                      const pendingWrite =
                        typeof secrets[name] === "string" &&
                        secrets[name].length > 0;
                      return (
                        <div
                          key={name}
                          className={
                            pendingDelete
                              ? "credential-row removing"
                              : "credential-row"
                          }
                        >
                          <code className="credential-name">{name}</code>
                          <span
                            className={`pill ${
                              pendingDelete
                                ? "pill-warn"
                                : pendingWrite
                                  ? "pill-active"
                                  : "pill-success"
                            }`}
                          >
                            {pendingDelete
                              ? t("remove on save", "保存时删除")
                              : pendingWrite
                                ? t("staged · configured", "已暂存 · 待保存")
                                : t("configured", "已配置")}
                          </span>
                          <button
                            type="button"
                            className="icon-remove-btn"
                            aria-label={t(`Remove ${name}`, `移除 ${name}`)}
                            onClick={() =>
                              setSecrets({ ...secrets, [name]: "" })
                            }
                          >
                            <IconClose />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </section>
          )}
        </div>
      </div>
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
  const t: Translate = (en, cn) => (zh ? cn : en);
  const [data, setData] = useState<Bootstrap>();
  const [tab, setTab] = useState("chat");
  const [settingsCategory, setSettingsCategory] =
    useState<SettingsCategory>("models");
  const openSettings = (cat?: SettingsCategory) => {
    if (cat) setSettingsCategory(cat);
    setTab("settings");
  };
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

  useEffect(() => {
    document.documentElement.lang = zh ? "zh-CN" : "en";
  }, [zh]);

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

  const createNewConversation = () =>
    void run(async () => {
      const session = await api<Session>("sessions", {});
      await choose(session.id);
      await load();
    });

  const sendCommand = (input: unknown) =>
    api(`sessions/${selected}/commands`, input);

  const currentSession = data?.sessions.find((s) => s.id === selected);
  const selectedProviderObj = data?.config.providers.find(
    (p) => p.id === model.provider,
  );
  const providerInConfig = Boolean(
    data?.config.providers.some((p) => p.id === model.provider),
  );
  const modelInProvider = Boolean(
    selectedProviderObj?.models.some((m) => m.id === model.model),
  );
  const modelChanged = Boolean(
    currentSession &&
      (currentSession.profile.provider !== model.provider ||
        currentSession.profile.model !== model.model ||
        currentSession.profile.thinking !== model.thinking),
  );

  const workspaceScope = data
    ? data.config.execution.target === "ssh"
      ? [
          data.config.execution.sshHost,
          data.config.execution.remoteWorkspace,
        ]
          .filter(Boolean)
          .join(" · ")
      : data.config.paths.workspace || ""
    : "";

  const activeJob = job ?? data?.jobs[0];
  const currentCommandMeta = harnessCommandMeta(command, t);
  const activeImageRoute =
    data?.config.image[imageKind as keyof Config["image"]];

  return (
    <div className="shell">
      <a className="skip" href="#workspace">
        {t("Skip to workspace", "跳转到工作区")}
      </a>

      <aside className="sidebar">
        <div className="sidebar-header">
          <a className="brand" href="/">
            <span className="brand-mark">
              <img src="/logo.png" alt="" />
            </span>
            <span className="brand-copy">
              <span className="brand-title">DataEvolver</span>
              <small>WORKBENCH</small>
            </span>
          </a>
          <button
            type="button"
            className="new-session"
            onClick={createNewConversation}
          >
            <IconPlus />
            <span>{t("New conversation", "新建对话")}</span>
          </button>
        </div>

        <nav className="primary-nav" aria-label={t("Workspace", "工作区导航")}>
          <button
            type="button"
            className={tab === "chat" ? "active" : ""}
            aria-current={tab === "chat" ? "page" : undefined}
            onClick={() => setTab("chat")}
          >
            <IconChat />
            <span>{t("Chat", "对话")}</span>
          </button>
          <button
            type="button"
            className={tab === "harness" ? "active" : ""}
            aria-current={tab === "harness" ? "page" : undefined}
            onClick={() => setTab("harness")}
          >
            <IconHarness />
            <span>Harness</span>
          </button>
          <button
            type="button"
            className={tab === "images" ? "active" : ""}
            aria-current={tab === "images" ? "page" : undefined}
            onClick={() => setTab("images")}
          >
            <IconImage />
            <span>{t("Image studio", "图像工坊")}</span>
          </button>
          <button
            type="button"
            className={tab === "settings" ? "active" : ""}
            aria-current={tab === "settings" ? "page" : undefined}
            onClick={() => setTab("settings")}
          >
            <IconSettings />
            <span>{t("Settings", "设置")}</span>
          </button>
        </nav>

        <div className="session-section">
          <span className="eyebrow">
            {t("SESSION HISTORY", "会话历史")}
          </span>
          <div className="session-list">
            {!data ? (
              <span className="sidebar-empty">
                {t("Loading…", "读取中…")}
              </span>
            ) : data.sessions.length === 0 ? (
              <span className="sidebar-empty">
                {t("No sessions yet", "暂无会话记录")}
              </span>
            ) : (
              data.sessions.map((s) => {
                const isCurrent = s.id === selected && tab === "chat";
                return (
                  <button
                    type="button"
                    className={
                      s.id === selected ? "session-item active" : "session-item"
                    }
                    aria-current={isCurrent ? "true" : undefined}
                    key={s.id}
                    onClick={() => void run(() => choose(s.id))}
                  >
                    <span className="session-name">
                      {s.parentId ? "↳ " : ""}
                      {s.name}
                    </span>
                    <small className="session-meta">
                      <span>
                        {s.profile.model || t("Unassigned", "未选模型")}
                      </span>
                      <span aria-hidden="true">·</span>
                      <span>{formatState(s.state, t)}</span>
                    </small>
                  </button>
                );
              })
            )}
          </div>
        </div>

        <footer className="sidebar-footer">
          <button
            type="button"
            className="lang-btn"
            aria-label={t(
              "Switch interface language to Chinese",
              "切换界面语言为英文",
            )}
            onClick={() => setZh(!zh)}
          >
            <IconGlobe />
            <span>{zh ? "English" : "简体中文"}</span>
          </button>
          {data && (
            <span className="target-chip">
              {data.config.execution.target === "ssh"
                ? "SSH"
                : t("Local", "本地")}
            </span>
          )}
        </footer>
      </aside>

      <main id="workspace">
        <header className="topbar">
          <div className="topbar-left">
            <span className="topbar-label">
              {t("Private workspace", "私有研究工作区")}
            </span>
            {workspaceScope && (
              <span className="topbar-scope" title={workspaceScope}>
                {workspaceScope}
              </span>
            )}
          </div>
          <span className={notice ? "status status-active" : "status"}>
            {notice ||
              t("Host permissions · single user", "宿主机权限 · 单用户")}
          </span>
        </header>

        {error && (
          <div className="error" role="alert">
            <span>{error}</span>
            <button
              type="button"
              onClick={() => setError("")}
              aria-label={t("Dismiss", "关闭错误提示")}
            >
              <IconClose />
            </button>
          </div>
        )}

        {!data ? (
          <div className="loading-state">
            <p className="loading">
              {t("Loading configuration…", "正在读取工作台配置…")}
            </p>
          </div>
        ) : (
          <>
            {tab === "settings" && (
              <Settings
                key={JSON.stringify(data.config)}
                initial={data.config}
                credentials={data.credentials}
                inspection={data.inspection}
                defaultCategory={settingsCategory}
                onCategoryChange={setSettingsCategory}
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
                {selected ? (
                  <>
                    <header className="chat-header">
                      <div className="chat-heading">
                        <div className="chat-title-block">
                          <span className="eyebrow">
                            {currentSession?.parentId
                              ? t("CHILD SESSION", "派生子会话")
                              : t("AGENT SESSION", "Agent 会话")}
                          </span>
                          <h1>
                            {currentSession?.name ||
                              t("What will you build?", "这次构建什么？")}
                          </h1>
                        </div>
                        <span className={`pill pill-${stateTone(state)}`}>
                          {formatState(state, t)}
                        </span>
                      </div>

                      <div
                        className="model-bar"
                        role="group"
                        aria-label={t("Session model", "会话模型与推理控制")}
                      >
                        <label className="compact-control">
                          <span>{t("Provider", "服务")}</span>
                          <select
                            value={model.provider}
                            onChange={(e) =>
                              setModel({
                                ...model,
                                provider: e.target.value,
                                model: "",
                              })
                            }
                          >
                            <option value="">—</option>
                            {model.provider && !providerInConfig && (
                              <option value={model.provider}>
                                {model.provider}
                              </option>
                            )}
                            {data.config.providers.map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.id}
                              </option>
                            ))}
                          </select>
                        </label>

                        <label className="compact-control">
                          <span>{t("Model", "模型")}</span>
                          <select
                            value={model.model}
                            onChange={(e) =>
                              setModel({ ...model, model: e.target.value })
                            }
                          >
                            <option value="">—</option>
                            {model.model && !modelInProvider && (
                              <option value={model.model}>{model.model}</option>
                            )}
                            {selectedProviderObj?.models.map((m) => (
                              <option key={m.id} value={m.id}>
                                {m.id}
                              </option>
                            ))}
                          </select>
                        </label>

                        <label className="compact-control compact-thinking">
                          <span>{t("Thinking", "推理")}</span>
                          <select
                            value={model.thinking}
                            onChange={(e) =>
                              setModel({ ...model, thinking: e.target.value })
                            }
                          >
                            {THINKING_LEVELS.map((level) => (
                              <option key={level} value={level}>
                                {level}
                              </option>
                            ))}
                          </select>
                        </label>

                        <button
                          type="button"
                          className={
                            modelChanged
                              ? "model-apply-btn changed"
                              : "model-apply-btn"
                          }
                          onClick={() =>
                            void run(() =>
                              sendCommand({ command: "set_model", ...model }),
                            )
                          }
                        >
                          {t("Switch between turns", "在轮次之间切换")}
                        </button>
                      </div>
                    </header>

                    <div className="messages" aria-live="polite">
                      {messages.length === 0 && !partial && (
                        <div className="thread-empty">
                          <p>
                            {t(
                              "Conversation ready. Describe a research task, code change, or preparation plan below.",
                              "会话已就绪。请在下方输入研究任务、代码修改或计划准备指令。",
                            )}
                          </p>
                        </div>
                      )}

                      {messages.map((m, i) => {
                        const isTool = m.role === "toolResult";
                        const isUser = m.role === "user";
                        const roleLabel = isTool
                          ? m.toolName || t("Tool result", "工具输出")
                          : isUser
                            ? t("You", "用户")
                            : "DataEvolver";
                        return (
                          <article
                            key={i}
                            className={`message ${m.role}${
                              m.isError ? " is-error" : ""
                            }`}
                          >
                            <div className="message-head">
                              <span className="message-role">{roleLabel}</span>
                              {m.isError && (
                                <span className="pill pill-danger">
                                  {t("error", "错误")}
                                </span>
                              )}
                            </div>
                            {typeof m.content === "string" ? (
                              <pre className="message-body">{m.content}</pre>
                            ) : (
                              <div className="message-blocks">
                                {m.content?.map((block, k) =>
                                  block.type === "text" ? (
                                    <pre key={k} className="message-body">
                                      {block.text}
                                    </pre>
                                  ) : block.type === "toolCall" ? (
                                    <div key={k} className="tool-call-badge">
                                      <IconArrowUpRight />
                                      <span>
                                        {t("Tool call", "调用工具")}
                                      </span>
                                      <code>{block.name}</code>
                                    </div>
                                  ) : block.type === "image" ? (
                                    <figure key={k} className="message-figure">
                                      <img
                                        className="message-image"
                                        src={`data:${block.mimeType};base64,${block.data}`}
                                        alt={t(
                                          "Agent artifact",
                                          "Agent 图像产物",
                                        )}
                                      />
                                    </figure>
                                  ) : null,
                                )}
                              </div>
                            )}
                          </article>
                        );
                      })}

                      {partial && (
                        <article className="message assistant streaming">
                          <div className="message-head">
                            <span className="message-role">DataEvolver</span>
                            <span className="pill pill-active">
                              {t("streaming", "生成中")}
                            </span>
                          </div>
                          <pre className="message-body">{partial}</pre>
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
                        {t("Message", "输入消息")}
                      </label>
                      <textarea
                        id="prompt"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={(e) => {
                          if (
                            (e.metaKey || e.ctrlKey) &&
                            e.key === "Enter" &&
                            prompt.trim() &&
                            !busy
                          ) {
                            e.preventDefault();
                            e.currentTarget.form?.requestSubmit();
                          }
                        }}
                        placeholder={t(
                          "Describe a task or steer the current run…",
                          "描述研究任务，或引导当前执行方向…",
                        )}
                        required
                      />
                      <div className="composer-actions">
                        <label className="mode-select-wrap">
                          <span className="sr-only">
                            {t("Message mode", "消息发送模式")}
                          </span>
                          <select
                            aria-label={t("Message mode", "消息发送模式")}
                            value={mode}
                            onChange={(e) => setMode(e.target.value)}
                          >
                            <option value="prompt">
                              {t("Prompt", "指令 (Prompt)")}
                            </option>
                            <option value="steer">
                              {t("Steer", "转向 (Steer)")}
                            </option>
                            <option value="follow_up">
                              {t("Follow-up", "追问 (Follow-up)")}
                            </option>
                          </select>
                        </label>

                        <div className="session-controls">
                          <button
                            type="button"
                            className="ghost-btn"
                            onClick={() =>
                              void run(() => sendCommand({ command: "resume" }))
                            }
                          >
                            {t("Reopen session", "恢复对话")}
                          </button>
                          <button
                            type="button"
                            className="ghost-btn"
                            onClick={() =>
                              void run(() => sendCommand({ command: "abort" }))
                            }
                          >
                            {t("Stop agent", "停止 Agent")}
                          </button>
                        </div>

                        <button
                          type="submit"
                          className="primary send-btn"
                          disabled={busy || !prompt.trim()}
                        >
                          <span>{t("Send", "发送")}</span>
                          <IconSend />
                        </button>
                      </div>
                    </form>
                  </>
                ) : (
                  <div className="welcome">
                    <div className="welcome-mark" aria-hidden="true">
                      <img src="/logo.png" alt="" />
                    </div>
                    <h1>
                      {t(
                        "A plan becomes a traceable workflow.",
                        "让计划成为有据可查的工作流。",
                      )}
                    </h1>
                    <p className="welcome-lead">
                      {t(
                        "Configure an agent, open a conversation, then compile and advance your task through the Harness.",
                        "配置 Agent 并开启对话，随后通过 Harness 编译计划并有界推进任务。",
                      )}
                    </p>

                    <div className="welcome-cta-row">
                      <button
                        type="button"
                        className="primary"
                        onClick={createNewConversation}
                      >
                        <IconPlus />
                        <span>{t("New conversation", "新建对话")}</span>
                      </button>
                      <button
                        type="button"
                        className="secondary-btn"
                        onClick={() => openSettings("models")}
                      >
                        <IconSettings />
                        <span>
                          {data.config.agents.main.provider &&
                          data.config.agents.main.model
                            ? `${data.config.agents.main.provider} / ${data.config.agents.main.model}`
                            : t("Configure models", "配置模型服务")}
                        </span>
                      </button>
                    </div>

                    <div className="welcome-cards">
                      <button
                        type="button"
                        onClick={() => setTab("settings")}
                      >
                        <span className="step-index">01</span>
                        <b>{t("Connect", "连接配置")}</b>
                        <span>
                          {t(
                            "Models, credentials and execution workstations",
                            "模型服务、私有凭据与执行工作站",
                          )}
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={() => setTab("harness")}
                      >
                        <span className="step-index">02</span>
                        <b>{t("Execute", "有界执行")}</b>
                        <span>
                          {t(
                            "Compile plans, create Campaigns and run bounded steps",
                            "编译计划、创建 Campaign 与有界步进",
                          )}
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={() => setTab("images")}
                      >
                        <span className="step-index">03</span>
                        <b>{t("Inspect", "图像与核验")}</b>
                        <span>
                          {t(
                            "Generate images, refine prompts and inspect receipts",
                            "生成与编辑图像、增强提示并查看回执",
                          )}
                        </span>
                      </button>
                    </div>

                    <p className="quiet">
                      {t(
                        "Agents can edit files and run host commands. Conversation plans are separate from Campaign state, which belongs to the Harness.",
                        "Agent 可以编辑文件并执行宿主机命令。对话计划用于组织步骤；Campaign 状态由 Harness 独立管理。",
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
                    <span className="eyebrow">
                      {t("CAMPAIGN CONTROL", "CAMPAIGN 任务控制")}
                    </span>
                    <h1>Harness</h1>
                    <p className="mono-caption">
                      {data.config.execution.target === "ssh"
                        ? `${data.config.execution.sshHost} · ${data.config.execution.remoteWorkspace}`
                        : data.config.paths.workspace}
                    </p>
                  </div>
                  <span className="pill">
                    {data.config.execution.target === "ssh"
                      ? t("Remote OpenSSH", "远端 OpenSSH")
                      : t("Local host", "本地主机")}
                  </span>
                </header>

                <div className="workbench-split">
                  <section className="card split-primary">
                    <div className="card-head">
                      <div>
                        <h2>{t("Command setup", "命令配置")}</h2>
                        <p>
                          {t(
                            "Select a Campaign lifecycle operation and provide explicit arguments.",
                            "选择 Campaign 生命周期操作并填写显式执行参数。",
                          )}
                        </p>
                      </div>
                    </div>

                    <form
                      className="stack-form"
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
                      <label className="field">
                        <span className="field-label">
                          {t("Harness command", "Harness 命令")}
                        </span>
                        <select
                          value={command}
                          onChange={(e) => setCommand(e.target.value)}
                        >
                          {Object.keys(commandFields).map((c) => {
                            const meta = harnessCommandMeta(c, t);
                            return (
                              <option key={c} value={c}>
                                {meta.title} ({c})
                              </option>
                            );
                          })}
                        </select>
                      </label>

                      <div className="command-brief">
                        <div className="command-brief-head">
                          <strong>{currentCommandMeta.title}</strong>
                          <code>{command}</code>
                        </div>
                        <p>{currentCommandMeta.summary}</p>
                      </div>

                      <div className="fields fields-single">
                        {commandFields[command].map((key) => (
                          <Field
                            key={key}
                            label={harnessFieldLabel(key, t)}
                            value={values[key] || ""}
                            type={key === "maxActions" ? "number" : "text"}
                            mono={key !== "goal"}
                            set={(value) =>
                              setValues({ ...values, [key]: value })
                            }
                          />
                        ))}
                      </div>

                      {command === "resume" && (
                        <div className="warning-callout" role="note">
                          {t(
                            "Campaign resume immediately calls run() and continues native work.",
                            "Campaign resume 会立即调用 run() 并继续原生任务。",
                          )}
                        </div>
                      )}

                      <div className="form-submit-row">
                        <button type="submit" className="primary">
                          {t("Submit command", "提交命令")}
                        </button>
                      </div>
                    </form>
                  </section>

                  <section className="card split-secondary">
                    <div className="card-head">
                      <div>
                        <h2>{t("Jobs and receipts", "任务与回执")}</h2>
                        <p>
                          {t(
                            "Inspect execution history and retained standard output streams.",
                            "查看命令执行记录与保留的标准输出回执。",
                          )}
                        </p>
                      </div>
                    </div>

                    {data.jobs.length === 0 ? (
                      <div className="empty-box">
                        {t(
                          "No Harness jobs recorded yet. Submit a command to inspect its receipt and output.",
                          "暂无 Harness 任务记录。提交命令后可在此查看回执与输出。",
                        )}
                      </div>
                    ) : (
                      <div className="job-list">
                        {data.jobs.map((j) => {
                          const meta = harnessCommandMeta(j.command, t);
                          const isSelected = activeJob?.id === j.id;
                          return (
                            <button
                              type="button"
                              key={j.id}
                              className={
                                isSelected ? "job-item active" : "job-item"
                              }
                              onClick={() =>
                                void run(async () =>
                                  setJob(await api<Job>("jobs/" + j.id)),
                                )
                              }
                            >
                              <div className="job-item-main">
                                <strong>{meta.title}</strong>
                                <code>{j.command}</code>
                              </div>
                              <div className="job-item-meta">
                                <span
                                  className={`pill pill-${stateTone(j.status)}`}
                                >
                                  {formatState(j.status, t)}
                                </span>
                                <small>
                                  {j.id.slice(0, 8)} · {j.target}
                                </small>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    )}

                    {activeJob && (
                      <div className="receipt-panel">
                        <div className="receipt-header">
                          <div className="receipt-title">
                            <h3>
                              {harnessCommandMeta(activeJob.command, t).title}{" "}
                              <code>{activeJob.command}</code>
                            </h3>
                            <small>
                              {activeJob.id} · {activeJob.target}
                            </small>
                          </div>
                          <div className="receipt-actions">
                            <span
                              className={`pill pill-${stateTone(activeJob.status)}`}
                            >
                              {formatState(activeJob.status, t)}
                            </span>
                            <button
                              type="button"
                              className="secondary-btn compact-btn"
                              onClick={() =>
                                void run(async () =>
                                  setJob(
                                    await api<Job>("jobs/" + activeJob.id),
                                  ),
                                )
                              }
                            >
                              {t("Refresh retained output", "刷新保留的输出")}
                            </button>
                          </div>
                        </div>
                        <pre className="log">{activeJob.output || "…"}</pre>
                      </div>
                    )}
                  </section>
                </div>
              </div>
            )}

            {tab === "images" && (
              <div className="page">
                <header className="section-heading">
                  <div>
                    <span className="eyebrow">
                      {t("IMAGE STUDIO", "图像工坊")}
                    </span>
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

                <div className="workbench-split">
                  <section className="card split-primary">
                    <div className="card-head">
                      <div>
                        <h2>{t("Studio request", "图像请求配置")}</h2>
                        <p>
                          {t(
                            "Choose an operation route, optional host source image, and prompt.",
                            "选择操作路由、可选的主机源图像路径与提示词。",
                          )}
                        </p>
                      </div>
                    </div>

                    <form
                      className="stack-form"
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
                      <label className="field">
                        <span className="field-label">
                          {t("Operation route", "操作路由")}
                        </span>
                        <select
                          value={imageKind}
                          onChange={(e) => setImageKind(e.target.value)}
                        >
                          <option value="generation">
                            {t("Generation", "图像生成 (Generation)")}
                          </option>
                          <option value="edit">
                            {t("Editing", "图像编辑 (Editing)")}
                          </option>
                          <option value="promptEnhancer">
                            {t(
                              "Prompt enhancement",
                              "提示词增强 (Prompt enhancement)",
                            )}
                          </option>
                        </select>
                      </label>

                      <div className="route-summary-bar">
                        <div className="route-summary-copy">
                          <span className="route-summary-label">
                            {t("Configured target", "当前路由目标")}
                          </span>
                          <code>
                            {activeImageRoute?.model
                              ? `${activeImageRoute.model}${
                                  activeImageRoute.baseUrl
                                    ? ` · ${activeImageRoute.baseUrl}`
                                    : ""
                                }`
                              : t(
                                  "Not configured in Settings yet",
                                  "尚未在设置中配置该路由",
                                )}
                          </code>
                        </div>
                        <button
                          type="button"
                          className="inline-link-btn"
                          onClick={() => openSettings("images")}
                        >
                          {t("Edit routes", "配置路由")}
                        </button>
                      </div>

                      <Field
                        label={t(
                          "Input image path on host (editing / vision PE)",
                          "主机输入图像路径（用于编辑或视觉提示增强）",
                        )}
                        value={imagePath}
                        placeholder="/path/to/source.png"
                        mono
                        set={setImagePath}
                      />

                      <label className="field">
                        <span className="field-label">
                          {t("Prompt", "提示词")}
                        </span>
                        <textarea
                          value={imagePrompt}
                          onChange={(e) => setImagePrompt(e.target.value)}
                          placeholder={t(
                            "Describe visual subject, composition, edit instructions or prompt enhancement goals…",
                            "描述画面主体、构图细节、局部编辑指令或提示词增强目标…",
                          )}
                          required
                        />
                      </label>

                      <div className="form-submit-row">
                        <button
                          type="submit"
                          className="primary"
                          disabled={busy}
                        >
                          {busy
                            ? t("Submitting…", "请求处理中…")
                            : t("Submit", "提交")}
                        </button>
                      </div>
                    </form>
                  </section>

                  <section className="card split-secondary">
                    <div className="card-head">
                      <div>
                        <h2>{t("Output and receipt", "产物与回执")}</h2>
                        <p>
                          {t(
                            "Rendered host artifact, returned text and retained receipt metadata.",
                            "渲染的主机图像产物、返回文本与保留的校验回执。",
                          )}
                        </p>
                      </div>
                    </div>

                    {!imageResult ? (
                      <div className="studio-empty">
                        <IconImage />
                        <p>
                          {t(
                            "Submit a generation, editing, or prompt enhancement request to inspect the resulting artifact and receipt here.",
                            "提交图像生成、编辑或提示词增强请求后，产物预览与校验回执将显示在此处。",
                          )}
                        </p>
                      </div>
                    ) : (
                      <div className="studio-result-stack">
                        {imageResult.path && (
                          <figure className="artifact-stage">
                            <img
                              className="artifact"
                              src={
                                "/api/artifact?path=" +
                                encodeURIComponent(imageResult.path)
                              }
                              alt={t("Generated artifact", "生成的图像产物")}
                            />
                          </figure>
                        )}

                        {imageResult.text && (
                          <div className="pe-output-card">
                            <div className="pe-output-head">
                              <strong>
                                {t("Returned text", "返回文本结果")}
                              </strong>
                              <button
                                type="button"
                                className="inline-link-btn"
                                onClick={() =>
                                  setImagePrompt(imageResult.text || "")
                                }
                              >
                                {t("Copy to prompt", "填入提示词")}
                              </button>
                            </div>
                            <pre className="pe-output-body">
                              {imageResult.text}
                            </pre>
                          </div>
                        )}

                        <dl className="receipt-grid">
                          <div>
                            <dt>{t("Receipt", "回执标识")}</dt>
                            <dd>{imageResult.receipt}</dd>
                          </div>
                          {imageResult.path && (
                            <div>
                              <dt>{t("Artifact path", "产物路径")}</dt>
                              <dd>{imageResult.path}</dd>
                            </div>
                          )}
                          {imageResult.host && (
                            <div>
                              <dt>{t("Host", "主机")}</dt>
                              <dd>{imageResult.host}</dd>
                            </div>
                          )}
                          {imageResult.sha256 && (
                            <div>
                              <dt>SHA-256</dt>
                              <dd>{imageResult.sha256}</dd>
                            </div>
                          )}
                        </dl>

                        <pre className="log">
                          {JSON.stringify(imageResult, null, 2)}
                        </pre>
                      </div>
                    )}
                  </section>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
