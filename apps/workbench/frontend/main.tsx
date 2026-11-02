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
  ActionButton,
  CompactSwitch,
  DisclosureGroup,
  DisclosureItem,
  SelectField,
  SettingsTabPanel,
  SettingsTabs,
  SwitchField,
  TextField,
  TooltipProvider,
  withCurrentOption,
} from "./components/ui";
import {
  executionFieldLabel,
  formatState,
  getSettingsCategories,
  harnessCommandMeta,
  harnessFieldLabel,
  IconAperture,
  IconArrowRight,
  IconArrowUpRight,
  IconChat,
  IconClose,
  IconCpu,
  IconGlobe,
  IconHarness,
  IconImage,
  IconKey,
  IconPlus,
  IconRefresh,
  IconSend,
  IconSettings,
  IconStop,
  imageRouteMeta,
  isSettingsCategory,
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
      "Open the startup URL printed by apps/workbench/launch.sh / 请打开 apps/workbench/launch.sh 输出的登录链接",
    );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      (data.error || response.statusText) +
        (data.receipt ? ` · receipt: ${data.receipt}` : ""),
    );
  return data;
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

const API_PROTOCOLS = [
  "openai-completions",
  "openai-responses",
  "anthropic-messages",
  "google-generative-ai",
] as const;

function buildProviderOptions(providers: Provider[], current: string) {
  const base = providers
    .filter((p) => p.id !== "")
    .map((p) => ({ value: p.id, label: p.id }));
  return withCurrentOption(base, current, "—");
}

function buildModelOptions(
  providers: Provider[],
  providerId: string,
  currentModel: string,
) {
  const active = providers.find((p) => p.id === providerId);
  const base = (active?.models ?? [])
    .filter((m) => m.id !== "")
    .map((m) => ({ value: m.id, label: m.id }));
  return withCurrentOption(base, currentModel, "—");
}

function buildThinkingOptions(current: string) {
  const base = THINKING_LEVELS.map((level) => ({ value: level, label: level }));
  return withCurrentOption(base, current);
}

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
  return (
    <fieldset className="agent-select">
      <legend>{label}</legend>
      <SelectField
        label={t("Provider", "服务提供方")}
        value={agent.provider}
        options={buildProviderOptions(providers, agent.provider)}
        onChange={(provider) => update({ ...agent, provider, model: "" })}
      />
      <SelectField
        label={t("Model", "模型")}
        value={agent.model}
        options={buildModelOptions(providers, agent.provider, agent.model)}
        onChange={(model) => update({ ...agent, model })}
      />
      <SelectField
        label={t("Thinking", "推理深度")}
        value={agent.thinking}
        options={buildThinkingOptions(agent.thinking)}
        onChange={(thinking) => update({ ...agent, thinking })}
      />
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

  const tabsMeta = getSettingsCategories(t).map((cat) => ({
    ...cat,
    badge:
      cat.id === "models"
        ? String(config.providers.length)
        : cat.id === "execution"
          ? config.execution.target || "local"
          : cat.id === "credentials"
            ? String(activeKeyCount)
            : null,
  }));

  const runtimeStages = [
    "actionFusion",
    "observationPack",
    "evidencePreservingReducer",
    "onlineContextCompact",
  ] as const;
  const enabledStageCount = runtimeStages.filter((k) => config.sol[k]).length;

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
          <ActionButton type="submit" variant="primary" disabled={saving}>
            {saving
              ? t("Saving…", "保存中…")
              : t("Save configuration", "保存配置")}
          </ActionButton>
        </div>
      </header>

      <SettingsTabs
        value={category}
        onValueChange={(next) => {
          if (isSettingsCategory(next)) setCategory(next);
        }}
        ariaLabel={t("Settings categories", "设置分类")}
        tabs={tabsMeta}
      >
        <SettingsTabPanel value="models">
          <section className="card">
            <div className="card-head">
              <div>
                <h2>{t("Main and child agents", "主 Agent 与子 Agent")}</h2>
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
              <ActionButton variant="secondary" onClick={addNewProvider}>
                <IconPlus />
                <span>{t("Add provider", "添加服务")}</span>
              </ActionButton>
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
                          {p.id || t(`Provider ${i + 1}`, `模型服务 ${i + 1}`)}
                        </strong>
                        <span className="pill">{p.api}</span>
                      </div>
                      <ActionButton
                        variant="danger"
                        size="sm"
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
                      </ActionButton>
                    </div>

                    <div className="fields">
                      <TextField
                        label={t("Provider ID", "服务标识 (Provider ID)")}
                        value={p.id}
                        mono
                        set={(id) => updateProvider(i, { ...p, id })}
                      />
                      <SelectField
                        label={t("API protocol", "API 协议")}
                        value={p.api}
                        mono
                        options={withCurrentOption(
                          API_PROTOCOLS.map((a) => ({ value: a, label: a })),
                          p.api,
                        )}
                        onChange={(nextApi) =>
                          updateProvider(i, { ...p, api: nextApi })
                        }
                      />
                      <TextField
                        label={t("Base URL", "接口地址 (Base URL)")}
                        value={p.baseUrl}
                        mono
                        set={(baseUrl) => updateProvider(i, { ...p, baseUrl })}
                      />
                      <div className="field-with-link">
                        <TextField
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
                        <ActionButton
                          variant="link"
                          size="sm"
                          onClick={() => setCategory("credentials")}
                        >
                          <IconKey size={13} />
                          <span>
                            {t(
                              "Manage keys in Credentials",
                              "前往私有凭据配置密钥",
                            )}
                          </span>
                        </ActionButton>
                      </div>
                    </div>

                    <div className="models-subsection">
                      <div className="models-subheader">
                        <span>
                          {t("Models", "模型列表")} ({p.models.length})
                        </span>
                        <ActionButton
                          variant="secondary"
                          size="sm"
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
                          <IconPlus size={14} />
                          <span>{t("Add model", "添加模型")}</span>
                        </ActionButton>
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
                          <div className="model-card" key={j}>
                            <div className="model-card-top">
                              <TextField
                                label={t("Model ID", "模型 ID")}
                                value={model.id}
                                mono
                                set={(id) => patch({ id })}
                              />
                              <div className="model-capabilities">
                                <CompactSwitch
                                  label={t("Reasoning", "推理")}
                                  checked={model.reasoning}
                                  onChange={(reasoning) => patch({ reasoning })}
                                />
                                <CompactSwitch
                                  label={t("Vision", "视觉")}
                                  checked={model.input.includes("image")}
                                  onChange={(vision) =>
                                    patch({
                                      input: vision
                                        ? ["text", "image"]
                                        : ["text"],
                                    })
                                  }
                                />
                                <ActionButton
                                  variant="ghost"
                                  size="icon"
                                  aria-label={t(
                                    `Remove model ${model.id || j + 1}`,
                                    `移除模型 ${model.id || j + 1}`,
                                  )}
                                  tooltip={t(
                                    "Remove this model entry",
                                    "移除该模型条目",
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
                                  <IconClose size={15} />
                                </ActionButton>
                              </div>
                            </div>

                            <DisclosureGroup className="model-limits-accordion">
                              <DisclosureItem
                                value={`limits-${i}-${j}`}
                                title={t(
                                  "Token window & output limits",
                                  "上下文窗口与最大输出限制",
                                )}
                                summary={`${model.contextWindow} ctx / ${model.maxTokens} max`}
                              >
                                <div className="fields">
                                  <TextField
                                    label={t("Context tokens", "上下文窗口")}
                                    type="number"
                                    value={model.contextWindow}
                                    set={(n) =>
                                      patch({ contextWindow: Number(n) })
                                    }
                                  />
                                  <TextField
                                    label={t("Max output", "最大输出")}
                                    type="number"
                                    value={model.maxTokens}
                                    set={(n) => patch({ maxTokens: Number(n) })}
                                  />
                                </div>
                              </DisclosureItem>
                            </DisclosureGroup>
                          </div>
                        );
                      })}
                    </div>
                  </fieldset>
                ))}
              </div>
            )}
          </section>
        </SettingsTabPanel>

        <SettingsTabPanel value="images">
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
              <ActionButton
                variant="secondary"
                onClick={() => setCategory("credentials")}
              >
                <IconKey size={15} />
                <span>{t("Credentials", "管理密钥")}</span>
              </ActionButton>
            </div>

            <div className="route-stack">
              {(["generation", "edit", "promptEnhancer"] as const).map(
                (kind) => {
                  const meta = imageRouteMeta(kind, t);
                  return (
                    <fieldset key={kind} className="route-card">
                      <legend>
                        {meta.title} / <code>{kind}</code>
                      </legend>
                      <p className="route-protocol">{meta.protocol}</p>
                      <div className="fields fields-3">
                        <TextField
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
                        <TextField
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
                        <TextField
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
        </SettingsTabPanel>

        <SettingsTabPanel value="paths">
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
                    <TextField
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
        </SettingsTabPanel>

        <SettingsTabPanel value="execution">
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

            <SelectField
              className="target-select-field"
              label={t("Execution target", "执行目标 (target)")}
              value={config.execution.target || "local"}
              options={[
                {
                  value: "local",
                  label: t("Local workstation (local)", "本地工作站 (local)"),
                },
                {
                  value: "ssh",
                  label: t(
                    "Remote OpenSSH host (ssh)",
                    "远端 OpenSSH 主机 (ssh)",
                  ),
                },
              ]}
              onChange={(target) => updateMap("execution", "target", target)}
            />

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
                  <TextField
                    key={key}
                    label={executionFieldLabel(key, t)}
                    value={value}
                    mono
                    set={(next) => updateMap("execution", key, next)}
                  />
                ))}
            </div>
          </section>
        </SettingsTabPanel>

        <SettingsTabPanel value="runtime">
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
                <TextField
                  key={key}
                  label={runtimeFieldLabel(key, t)}
                  value={value}
                  mono
                  set={(next) => updateMap("pi", key, next)}
                />
              ))}
            </div>

            <div className="runtime-master-switch">
              <SwitchField
                title={runtimeToggleMeta("enabled", t).title}
                description={runtimeToggleMeta("enabled", t).hint}
                checked={config.sol.enabled}
                onChange={(enabled) =>
                  setConfig({
                    ...config,
                    sol: { ...config.sol, enabled },
                  })
                }
              />
            </div>

            <div className="callout-note">
              {t(
                "Reducer sends eligible logs to its configured model. Compaction can make model calls and continue the task. Set both explicitly.",
                "Reducer 会将符合条件的日志发送给配置的模型。上下文压缩可能触发模型调用并继续任务，请显式配置这两项。",
              )}
            </div>

            <DisclosureGroup
              defaultOpen={["stages", "reducer"]}
              className="runtime-accordion"
            >
              <DisclosureItem
                value="stages"
                title={t(
                  "Pipeline optimization stages",
                  "运行时优化与上下文阶段",
                )}
                summary={t(
                  `${enabledStageCount} / 4 active`,
                  `已开启 ${enabledStageCount} / 4`,
                )}
              >
                <div className="toggle-grid">
                  {runtimeStages.map((key) => {
                    const meta = runtimeToggleMeta(key, t);
                    return (
                      <SwitchField
                        key={key}
                        title={meta.title}
                        description={meta.hint}
                        checked={config.sol[key]}
                        onChange={(checked) =>
                          setConfig({
                            ...config,
                            sol: { ...config.sol, [key]: checked },
                          })
                        }
                      />
                    );
                  })}
                </div>
              </DisclosureItem>

              <DisclosureItem
                value="reducer"
                title={t(
                  "Reducer model & cache ratio",
                  "归约模型与缓存写读比参数",
                )}
                summary={
                  config.sol.reducerModel || t("Unassigned", "未指定模型")
                }
              >
                <div className="fields fields-3">
                  <TextField
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
                  <TextField
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
                  <TextField
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
              </DisclosureItem>
            </DisclosureGroup>
          </section>
        </SettingsTabPanel>

        <SettingsTabPanel value="credentials">
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
                <TextField
                  label={t("Environment name", "环境变量名")}
                  value={keyName}
                  placeholder="API_KEY_ENV"
                  mono
                  set={setKeyName}
                />
                <TextField
                  label={t("API key (write-only)", "API 密钥（仅写入）")}
                  type="password"
                  value={keyValue}
                  set={setKeyValue}
                />
              </div>
              <div className="credential-stage-actions">
                <ActionButton
                  variant="secondary"
                  disabled={!keyName || !keyValue}
                  onClick={() => {
                    setSecrets({ ...secrets, [keyName]: keyValue });
                    setKeyValue("");
                    setKeyName("");
                  }}
                >
                  <IconKey size={15} />
                  <span>{t("Stage key for save", "暂存密钥")}</span>
                </ActionButton>
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
                              ? t("staged, pending save", "已暂存，待保存")
                              : t("configured", "已配置")}
                        </span>
                        <ActionButton
                          variant="ghost"
                          size="icon"
                          aria-label={t(`Remove ${name}`, `移除 ${name}`)}
                          tooltip={t(
                            "Mark key for removal when saved",
                            "标记在保存配置时删除该密钥",
                          )}
                          onClick={() =>
                            setSecrets({ ...secrets, [name]: "" })
                          }
                        >
                          <IconClose size={15} />
                        </ActionButton>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </section>
        </SettingsTabPanel>
      </SettingsTabs>
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

function StudioHome({
  data,
  workspaceScope,
  t,
  onCreateSession,
  onChooseSession,
  onOpenSettings,
  onOpenHarness,
  onOpenImages,
}: {
  data: Bootstrap;
  workspaceScope: string;
  t: Translate;
  onCreateSession: () => void;
  onChooseSession: (id: string) => void;
  onOpenSettings: (category: SettingsCategory) => void;
  onOpenHarness: (presetCommand?: string) => void;
  onOpenImages: (presetRoute?: string) => void;
}) {
  const { config, credentials, sessions } = data;
  const isSsh = config.execution.target === "ssh";
  const mainLabel =
    config.agents.main.provider && config.agents.main.model
      ? `${config.agents.main.provider} / ${config.agents.main.model}`
      : t("Unassigned", "未指定主模型");
  const childLabel =
    config.agents.child.provider && config.agents.child.model
      ? `${config.agents.child.provider} / ${config.agents.child.model}`
      : t("Unassigned", "未指定子模型");

  return (
    <section className="studio-home">
      <div className="studio-hero">
        <div className="studio-hero-main">
          <h1 className="studio-title">
            {t(
              "Build data around your training goal.",
              "围绕训练目标，构建数据。",
            )}
          </h1>

          <p className="studio-lead">
            {t(
              "Plan image and 3D data tasks with an agent, then execute and review receipts in Harness.",
              "通过 Agent 规划图像与 3D 数据任务，在 Harness 中执行并查看回执。",
            )}
          </p>

          <div className="studio-actions">
            <ActionButton
              variant="primary"
              className="studio-cta-primary"
              onClick={onCreateSession}
            >
              <span>{t("New conversation", "新建对话")}</span>
              <IconArrowRight size={16} />
            </ActionButton>
            <ActionButton
              variant="secondary"
              className="studio-cta-secondary"
              onClick={() => onOpenSettings("models")}
            >
              <IconSettings size={15} />
              <span>
                {t("Configure models & host", "配置模型与执行主机")}
              </span>
            </ActionButton>
          </div>
        </div>

        <aside
          className="studio-hero-spec"
          aria-label={t("Active workbench snapshot", "当前工作台配置摘要")}
        >
          <h2>{t("Current environment", "当前环境")}</h2>
          <dl className="studio-spec-dl">
            <div>
              <dt>{t("Execution", "执行目标")}</dt>
              <dd>
                {isSsh
                  ? config.execution.sshHost || "ssh"
                  : t("Local process", "本地进程 (local)")}
              </dd>
            </div>
            <div>
              <dt>{t("Workspace", "工作区路径")}</dt>
              <dd>{workspaceScope || t("Not set", "未设置路径")}</dd>
            </div>
          </dl>
        </aside>
      </div>

      <div className="studio-bento-grid">
        <article className="bento-card bento-harness">
          <div className="bento-card-top">
            <h2>
              <IconHarness size={18} />
              Harness
            </h2>
            <ActionButton
              variant="ghost"
              size="sm"
              onClick={() => onOpenHarness()}
            >
              <span>{t("Open Harness", "打开 Harness")}</span>
              <IconArrowUpRight size={14} />
            </ActionButton>
          </div>
          <p>
            {t(
              "Separate persisted Campaign state from chat turns. Advance work with explicit action limits, inspect status, or replay retained receipts.",
              "将持久化 Campaign 状态与对话轮次分离，按显式步数上限推进任务、查看状态或回放保留回执。",
            )}
          </p>
          <div
            className="bento-pipeline"
            role="group"
            aria-label={t("Harness quick commands", "Harness 快捷命令跳转")}
          >
            {(["compile-plan", "create", "step", "replay"] as const).map(
              (cmd, idx) => (
                <button
                  key={cmd}
                  type="button"
                  className="pipeline-node"
                  onClick={() => onOpenHarness(cmd)}
                >
                  <span className="pipeline-step">0{idx + 1}</span>
                  <strong>{harnessCommandMeta(cmd, t).title}</strong>
                  <code>{cmd}</code>
                </button>
              ),
            )}
          </div>
        </article>

        <article className="bento-card bento-studio">
          <div className="bento-card-top">
            <h2>
              <IconAperture size={18} />
              {t("Image studio", "图像工坊")}
            </h2>
            <ActionButton
              variant="ghost"
              size="sm"
              onClick={() => onOpenImages()}
            >
              <span>{t("Open Studio", "进入工坊")}</span>
              <IconArrowUpRight size={14} />
            </ActionButton>
          </div>
          <p>
            {t(
              "Run host-path image generation, multipart edits, and prompt enhancement with retained receipts and no automatic retries.",
              "执行主机路径图像生成、多部分编辑与提示词增强，保留校验回执且失败不自动重试。",
            )}
          </p>
          <div className="reticle-routes">
            {(
              [
                ["generation", t("Generation", "图像生成")],
                ["edit", t("Multipart edit", "图像编辑")],
                ["promptEnhancer", t("Prompt enhancer", "提示词增强")],
              ] as const
            ).map(([routeId, label]) => (
              <button
                key={routeId}
                type="button"
                className="reticle-chip"
                onClick={() => onOpenImages(routeId)}
              >
                <span>{label}</span>
                <code>{routeId}</code>
              </button>
            ))}
          </div>
        </article>

        <article className="bento-card bento-dark">
          <div className="bento-card-top">
            <h2>
              <IconCpu size={18} />
              {t("Agents & credentials", "Agent 与凭据")}
            </h2>
          </div>
          <dl className="bento-dark-specs">
            <div>
              <dt>{t("Main", "主 Agent")}</dt>
              <dd>
                <code>{mainLabel}</code>
                <span>{config.agents.main.thinking}</span>
              </dd>
            </div>
            <div>
              <dt>{t("Child", "子 Agent")}</dt>
              <dd>
                <code>{childLabel}</code>
                <span>{config.agents.child.thinking}</span>
              </dd>
            </div>
            <div>
              <dt>{t("Keys", "私有凭据")}</dt>
              <dd>
                {credentials.length > 0
                  ? t(
                      `${credentials.length} stored env key(s)`,
                      `已保存 ${credentials.length} 项环境密钥`,
                    )
                  : t("No keys stored", "尚未保存密钥")}
              </dd>
            </div>
          </dl>
          <div className="bento-dark-actions">
            <ActionButton
              variant="on-dark"
              size="sm"
              onClick={() => onOpenSettings("models")}
            >
              <IconCpu size={14} />
              <span>{t("Models & agents", "模型与 Agent")}</span>
            </ActionButton>
            <ActionButton
              variant="on-dark"
              size="sm"
              onClick={() => onOpenSettings("credentials")}
            >
              <IconKey size={14} />
              <span>{t("Credentials", "私有凭据")}</span>
            </ActionButton>
          </div>
        </article>

        <article className="bento-card bento-sessions">
          <div className="bento-card-top">
            <h2>
              <IconChat size={18} />
              {t("Recent conversations", "近期对话")}
            </h2>
            <ActionButton
              variant="ghost"
              size="sm"
              onClick={onCreateSession}
            >
              <IconPlus size={14} />
              <span>{t("Start session", "新建对话")}</span>
            </ActionButton>
          </div>

          {sessions.length > 0 ? (
            <>
              <div className="bento-session-list">
                {sessions.slice(0, 3).map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    className="bento-session-row"
                    onClick={() => onChooseSession(s.id)}
                  >
                    <span className="bento-session-title">
                      {s.parentId ? "↳ " : ""}
                      {s.name}
                    </span>
                    <span className="bento-session-meta">
                      <code>
                        {s.profile.model || t("Unassigned", "未选模型")}
                      </code>
                      <span className={`pill pill-${stateTone(s.state)}`}>
                        {formatState(s.state, t)}
                      </span>
                    </span>
                  </button>
                ))}
              </div>
            </>
          ) : (
            <>
              <h3>
                {t("Your first conversation starts here", "从第一个训练目标开始")}
              </h3>
              <p>
                {t(
                  "No conversations yet. Start one to plan your data task.",
                  "还没有对话。新建对话，规划你的数据任务。",
                )}
              </p>
              <div className="bento-capability-strip">
                <span>{t("Prompt / Steer / Follow-up", "指令 / 转向 / 追问")}</span>
                <span>{t("Per-turn model switch", "轮次间模型切换")}</span>
                <span>{t("Python / Blender / UE5 paths", "原生子进程工具路径")}</span>
              </div>
            </>
          )}

          <p className="bento-boundary-note">
            {t(
              "Agents can edit files and run host commands. Conversation plans are separate from Campaign state, which belongs to the Harness.",
              "Agent 可以编辑文件并执行宿主机命令。对话计划用于组织步骤；Campaign 状态由 Harness 独立管理。",
            )}
          </p>
        </article>
      </div>
    </section>
  );
}

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
    <TooltipProvider>
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
            <ActionButton
              variant="secondary"
              className="new-session"
              onClick={createNewConversation}
            >
              <IconPlus size={15} />
              <span>{t("New conversation", "新建对话")}</span>
            </ActionButton>
          </div>

          <nav
            className="primary-nav"
            aria-label={t("Workspace", "工作区导航")}
          >
            <button
              type="button"
              className={tab === "chat" ? "nav-item active" : "nav-item"}
              aria-current={tab === "chat" ? "page" : undefined}
              onClick={() => setTab("chat")}
            >
              <IconChat size={16} />
              <span>{t("Chat", "对话")}</span>
            </button>
            <button
              type="button"
              className={tab === "harness" ? "nav-item active" : "nav-item"}
              aria-current={tab === "harness" ? "page" : undefined}
              onClick={() => setTab("harness")}
            >
              <IconHarness size={16} />
              <span>Harness</span>
            </button>
            <button
              type="button"
              className={tab === "images" ? "nav-item active" : "nav-item"}
              aria-current={tab === "images" ? "page" : undefined}
              onClick={() => setTab("images")}
            >
              <IconImage size={16} />
              <span>{t("Image studio", "图像工坊")}</span>
            </button>
            <button
              type="button"
              className={tab === "settings" ? "nav-item active" : "nav-item"}
              aria-current={tab === "settings" ? "page" : undefined}
              onClick={() => setTab("settings")}
            >
              <IconSettings size={16} />
              <span>{t("Settings", "设置")}</span>
            </button>
          </nav>

          <div className="session-section">
            <span className="section-label">
              {t("Session history", "会话历史")}
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
                        s.id === selected
                          ? "session-item active"
                          : "session-item"
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
                        <span>{formatState(s.state, t)}</span>
                      </small>
                    </button>
                  );
                })
              )}
            </div>
          </div>

          <footer className="sidebar-footer">
            <ActionButton
              variant="secondary"
              size="sm"
              className="lang-btn"
              aria-label={t(
                "Switch interface language to Chinese",
                "切换界面语言为英文",
              )}
              tooltip={t(
                "Switch language to Simplified Chinese",
                "切换界面语言为 English",
              )}
              onClick={() => setZh(!zh)}
            >
              <IconGlobe size={14} />
              <span>{zh ? "English" : "简体中文"}</span>
            </ActionButton>
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
                t("Host permissions / single user", "宿主机权限 / 单用户")}
            </span>
          </header>

          {error && (
            <div className="error" role="alert">
              <span>{error}</span>
              <ActionButton
                variant="ghost"
                size="icon"
                onClick={() => setError("")}
                aria-label={t("Dismiss", "关闭错误提示")}
                tooltip={t("Dismiss error message", "关闭错误横幅")}
              >
                <IconClose size={15} />
              </ActionButton>
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

              {tab === "chat" &&
                (selected ? (
                  <div className="conversation">
                    <header className="chat-header">
                      <div className="chat-heading">
                        <div className="chat-title-block">
                          <h1>
                            {currentSession?.name ||
                              t("What will you build?", "这次构建什么？")}
                          </h1>
                          <span className="section-label">
                            {currentSession?.parentId
                              ? t("Child session", "派生子会话")
                              : t("Agent session", "Agent 会话")}
                          </span>
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
                        <SelectField
                          className="compact-control"
                          label={t("Provider", "服务")}
                          inlineLabel
                          compact
                          value={model.provider}
                          options={buildProviderOptions(
                            data.config.providers,
                            model.provider,
                          )}
                          onChange={(provider) =>
                            setModel({ ...model, provider, model: "" })
                          }
                        />

                        <SelectField
                          className="compact-control"
                          label={t("Model", "模型")}
                          inlineLabel
                          compact
                          value={model.model}
                          options={buildModelOptions(
                            data.config.providers,
                            model.provider,
                            model.model,
                          )}
                          onChange={(nextModel) =>
                            setModel({ ...model, model: nextModel })
                          }
                        />

                        <SelectField
                          className="compact-control compact-thinking"
                          label={t("Thinking", "推理")}
                          inlineLabel
                          compact
                          value={model.thinking}
                          options={buildThinkingOptions(model.thinking)}
                          onChange={(thinking) =>
                            setModel({ ...model, thinking })
                          }
                        />

                        <ActionButton
                          variant="secondary"
                          size="sm"
                          className={
                            modelChanged
                              ? "model-apply-btn changed"
                              : "model-apply-btn"
                          }
                          tooltip={t(
                            "Apply selected provider, model and thinking budget for the next turn",
                            "在下一轮对话开始前切换为所选服务、模型与推理深度",
                          )}
                          onClick={() =>
                            void run(() =>
                              sendCommand({ command: "set_model", ...model }),
                            )
                          }
                        >
                          {t("Switch between turns", "在轮次之间切换")}
                        </ActionButton>
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
                                      <IconArrowUpRight size={13} />
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
                        <SelectField
                          className="mode-select-wrap"
                          label={t("Message mode", "消息发送模式")}
                          ariaLabel={t("Message mode", "消息发送模式")}
                          srOnlyLabel
                          compact
                          value={mode}
                          options={[
                            {
                              value: "prompt",
                              label: t("Prompt", "指令 (Prompt)"),
                            },
                            {
                              value: "steer",
                              label: t("Steer", "转向 (Steer)"),
                            },
                            {
                              value: "follow_up",
                              label: t("Follow-up", "追问 (Follow-up)"),
                            },
                          ]}
                          onChange={setMode}
                        />

                        <div className="session-controls">
                          <ActionButton
                            variant="ghost"
                            size="sm"
                            tooltip={t(
                              "Reopen stopped session process (resume)",
                              "重新唤醒已停止的会话进程 (resume)",
                            )}
                            onClick={() =>
                              void run(() => sendCommand({ command: "resume" }))
                            }
                          >
                            <IconRefresh size={14} />
                            <span>{t("Reopen session", "恢复对话")}</span>
                          </ActionButton>
                          <ActionButton
                            variant="ghost"
                            size="sm"
                            tooltip={t(
                              "Interrupt the current agent run (abort)",
                              "立即中断当前正在运行的 Agent 轮次 (abort)",
                            )}
                            onClick={() =>
                              void run(() => sendCommand({ command: "abort" }))
                            }
                          >
                            <IconStop size={13} />
                            <span>{t("Stop agent", "停止 Agent")}</span>
                          </ActionButton>
                        </div>

                        <ActionButton
                          type="submit"
                          variant="primary"
                          size="sm"
                          className="send-btn"
                          disabled={busy || !prompt.trim()}
                          tooltip={t(
                            "Send message (⌘/Ctrl + Enter)",
                            "发送消息（⌘/Ctrl + Enter）",
                          )}
                        >
                          <span>{t("Send", "发送")}</span>
                          <IconSend size={14} />
                        </ActionButton>
                      </div>
                    </form>
                  </div>
                ) : (
                  <StudioHome
                    data={data}
                    workspaceScope={workspaceScope}
                    t={t}
                    onCreateSession={createNewConversation}
                    onChooseSession={(id) => void run(() => choose(id))}
                    onOpenSettings={(cat) => openSettings(cat)}
                    onOpenHarness={(preset) => {
                      if (preset) setCommand(preset);
                      setTab("harness");
                    }}
                    onOpenImages={(preset) => {
                      if (preset) setImageKind(preset);
                      setTab("images");
                    }}
                  />
                ))}

              {tab === "harness" && (
                <div className="page">
                  <header className="section-heading">
                    <div>
                      <h1>Harness</h1>
                      <p className="mono-caption">
                        {data.config.execution.target === "ssh"
                          ? `${data.config.execution.sshHost}:${data.config.execution.remoteWorkspace}`
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
                        <SelectField
                          label={t("Harness command", "Harness 命令")}
                          value={command}
                          options={Object.keys(commandFields).map((c) => {
                            const meta = harnessCommandMeta(c, t);
                            return {
                              value: c,
                              label: `${meta.title} (${c})`,
                            };
                          })}
                          onChange={setCommand}
                        />

                        <div className="command-brief">
                          <div className="command-brief-head">
                            <strong>{currentCommandMeta.title}</strong>
                            <code>{command}</code>
                          </div>
                          <p>{currentCommandMeta.summary}</p>
                        </div>

                        <div className="fields fields-single">
                          {commandFields[command].map((key) => (
                            <TextField
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
                          <ActionButton type="submit" variant="primary">
                            {t("Submit command", "提交命令")}
                          </ActionButton>
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
                              <ActionButton
                                variant="secondary"
                                size="sm"
                                tooltip={t(
                                  "Reload retained output snapshot from host",
                                  "从主机重新拉取该任务保留的完整输出",
                                )}
                                onClick={() =>
                                  void run(async () =>
                                    setJob(
                                      await api<Job>("jobs/" + activeJob.id),
                                    ),
                                  )
                                }
                              >
                                <IconRefresh size={13} />
                                <span>
                                  {t(
                                    "Refresh retained output",
                                    "刷新保留的输出",
                                  )}
                                </span>
                              </ActionButton>
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
                      <h1>
                        {t("Image studio", "图像工坊")}
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
                        <SelectField
                          label={t("Operation route", "操作路由")}
                          value={imageKind}
                          options={[
                            {
                              value: "generation",
                              label: t(
                                "Generation",
                                "图像生成 (Generation)",
                              ),
                            },
                            {
                              value: "edit",
                              label: t("Editing", "图像编辑 (Editing)"),
                            },
                            {
                              value: "promptEnhancer",
                              label: t(
                                "Prompt enhancement",
                                "提示词增强 (Prompt enhancement)",
                              ),
                            },
                          ]}
                          onChange={setImageKind}
                        />

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
                          <ActionButton
                            variant="link"
                            size="sm"
                            onClick={() => openSettings("images")}
                          >
                            {t("Edit routes", "配置路由")}
                          </ActionButton>
                        </div>

                        <TextField
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
                          <ActionButton
                            type="submit"
                            variant="primary"
                            disabled={busy}
                          >
                            {busy
                              ? t("Submitting…", "请求处理中…")
                              : t("Submit", "提交")}
                          </ActionButton>
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
                          <IconImage size={28} />
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
                                alt={t(
                                  "Generated artifact",
                                  "生成的图像产物",
                                )}
                              />
                            </figure>
                          )}

                          {imageResult.text && (
                            <div className="pe-output-card">
                              <div className="pe-output-head">
                                <strong>
                                  {t("Returned text", "返回文本结果")}
                                </strong>
                                <ActionButton
                                  variant="link"
                                  size="sm"
                                  onClick={() =>
                                    setImagePrompt(imageResult.text || "")
                                  }
                                >
                                  {t("Copy to prompt", "填入提示词")}
                                </ActionButton>
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
    </TooltipProvider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
