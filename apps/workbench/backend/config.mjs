import { mkdir, readFile, rename, writeFile, chmod } from "node:fs/promises";
import { randomUUID, createHash } from "node:crypto";
import { homedir } from "node:os";
import { resolve, dirname, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";

export const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
export const appRoot = fileURLToPath(new URL("../", import.meta.url));
export const privateRoot = resolve(
  process.env.DATAEVOLVER_WORKBENCH_HOME ||
    resolve(
      homedir(),
      ".dataevolver",
      "workbench",
      createHash("sha256").update(repoRoot).digest("hex").slice(0, 12),
    ),
);
export const thinkingLevels = [
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
];
const idPattern = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$/;
const apis = [
  "openai-completions",
  "openai-responses",
  "anthropic-messages",
  "google-generative-ai",
];
export function check(condition, message) {
  if (!condition) throw new Error(message);
}
export function identifier(value) {
  check(
    typeof value === "string" && idPattern.test(value),
    "Invalid identifier",
  );
  return value;
}
export function endpoint(value, optional = true) {
  if (!value && optional) return "";
  const url = new URL(value);
  check(
    ["http:", "https:"].includes(url.protocol) &&
      !url.username &&
      !url.password &&
      !url.search &&
      !url.hash,
    "Endpoint must be HTTP(S) without credentials, query or fragment",
  );
  return url.href.replace(/\/$/, "");
}
export function defaults() {
  const route = { baseUrl: "", model: "", apiKeyEnv: "" };
  return {
    version: 1,
    providers: [],
    agents: {
      main: { provider: "", model: "", thinking: "high" },
      child: { provider: "", model: "", thinking: "medium" },
    },
    image: {
      generation: { ...route },
      edit: { ...route },
      promptEnhancer: { ...route },
    },
    paths: {
      workspace: repoRoot,
      python:
        process.env.DATAEVOLVER_WORKBENCH_PYTHON ||
        resolve(repoRoot, ".venv/bin/python"),
      blender: "",
      ue5: "",
      ueProject: "",
      assets: "",
      datasets: "",
      output: resolve(repoRoot, ".dataevolver/artifacts"),
    },
    execution: {
      target: "local",
      sshHost: "",
      remoteWorkspace: "",
      remotePython: ".venv/bin/python",
      remoteBlender: "",
      remoteUe5: "",
      remoteUeProject: "",
      remoteAssets: "",
      remoteDatasets: "",
      remoteOutput: ".dataevolver/artifacts",
    },
    pi: {
      cli: "",
      solExtension:
        process.env.DATAEVOLVER_SOL_EXTENSION ||
        resolve(
          repoRoot,
          ".dataevolver/workbench-deps/sol-pi/src/sol-pi/index.ts",
        ),
    },
    sol: {
      enabled: true,
      actionFusion: true,
      observationPack: true,
      evidencePreservingReducer: false,
      onlineContextCompact: false,
      cacheWriteReadRatio: 12.5,
      reducerProvider: "",
      reducerModel: "",
    },
  };
}
export function validateConfig(input) {
  check(
    input && input.version === 1,
    "Unsupported workbench configuration version",
  );
  const config = structuredClone(input);
  const keys = (value, names) =>
    check(
      value &&
        typeof value === "object" &&
        !Array.isArray(value) &&
        Object.keys(value).length === names.length &&
        Object.keys(value).every((key) => names.includes(key)),
      "Unknown or missing configuration field",
    );
  const shape = defaults();
  keys(config, Object.keys(shape));
  for (const section of ["agents", "image", "paths", "execution", "pi", "sol"])
    keys(config[section], Object.keys(shape[section]));
  check(
    Array.isArray(config.providers) && config.providers.length <= 32,
    "Expected at most 32 providers",
  );
  const ids = new Set();
  const envName = (value) =>
    check(
      value === "" ||
        (typeof value === "string" && /^[A-Z_][A-Z0-9_]*$/.test(value)),
      "Invalid credential environment name",
    );
  const text = (value) =>
    check(
      typeof value === "string" &&
        value.length <= 8192 &&
        !value.includes("\0"),
      "Expected bounded text",
    );
  for (const provider of config.providers) {
    keys(provider, ["id", "api", "baseUrl", "apiKeyEnv", "models"]);
    identifier(provider.id);
    check(!ids.has(provider.id), "Duplicate provider");
    ids.add(provider.id);
    check(apis.includes(provider.api), "Unsupported Pi protocol");
    provider.baseUrl = endpoint(provider.baseUrl, false);
    envName(provider.apiKeyEnv);
    check(
      Array.isArray(provider.models) &&
        provider.models.length > 0 &&
        provider.models.length <= 100,
      "Provider needs models",
    );
    const models = new Set();
    for (const model of provider.models) {
      keys(model, ["id", "reasoning", "input", "contextWindow", "maxTokens"]);
      text(model.id);
      check(
        model.id && !models.has(model.id) && !/[\r\n]/.test(model.id),
        "Duplicate or invalid model",
      );
      models.add(model.id);
      check(typeof model.reasoning === "boolean", "reasoning must be boolean");
      check(
        Array.isArray(model.input) &&
          model.input.includes("text") &&
          model.input.every((x) => ["text", "image"].includes(x)),
        "Invalid model inputs",
      );
      for (const key of ["contextWindow", "maxTokens"])
        check(
          Number.isSafeInteger(model[key]) && model[key] > 0,
          "Set positive model token limits",
        );
      check(
        model.maxTokens <= model.contextWindow,
        "maxTokens exceeds contextWindow",
      );
    }
  }
  for (const role of ["main", "child"]) {
    const agent = config.agents[role];
    text(agent.provider);
    text(agent.model);
    keys(agent, ["provider", "model", "thinking"]);
    check(thinkingLevels.includes(agent.thinking), "Invalid thinking level");
    check(
      (!agent.provider && !agent.model) ||
        config.providers.some(
          (p) =>
            p.id === agent.provider &&
            p.models.some((m) => m.id === agent.model),
        ),
      "Agent model is not registered",
    );
  }
  for (const kind of ["generation", "edit", "promptEnhancer"]) {
    const route = config.image[kind];
    route.baseUrl = endpoint(route.baseUrl);
    text(route.model);
    envName(route.apiKeyEnv);
    keys(route, ["baseUrl", "model", "apiKeyEnv"]);
    check(
      Boolean(route.baseUrl) === Boolean(route.model),
      "Image routes require both endpoint and model",
    );
  }
  for (const value of Object.values(config.paths)) text(value);
  check(
    isAbsolute(config.paths.workspace) && isAbsolute(config.paths.output),
    "Workspace and output must be absolute paths",
  );
  check(
    ["local", "ssh"].includes(config.execution.target),
    "Invalid execution target",
  );
  for (const value of Object.values(config.execution)) text(value);
  if (config.execution.target === "ssh") {
    check(
      /^[a-zA-Z0-9][a-zA-Z0-9_.@-]*$/.test(config.execution.sshHost),
      "Use an OpenSSH host alias",
    );
    check(
      config.execution.remoteWorkspace.startsWith("/") &&
        config.execution.remotePython,
      "Set an absolute remote checkout and Python executable",
    );
  }
  text(config.pi.cli);
  text(config.pi.solExtension);
  for (const key of [
    "enabled",
    "actionFusion",
    "observationPack",
    "evidencePreservingReducer",
    "onlineContextCompact",
  ])
    check(typeof config.sol[key] === "boolean", "Invalid SoL switch");
  check(
    Number.isFinite(config.sol.cacheWriteReadRatio) &&
      config.sol.cacheWriteReadRatio > 0,
    "Invalid compaction cache ratio",
  );
  text(config.sol.reducerProvider);
  text(config.sol.reducerModel);
  if (config.sol.evidencePreservingReducer)
    check(
      config.providers.some(
        (p) =>
          p.id === config.sol.reducerProvider &&
          p.models.some((m) => m.id === config.sol.reducerModel),
      ),
      "Register a reducer model before enabling remote log reduction",
    );
  return config;
}
export async function atomicJson(path, value) {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const temporary = `${path}.${randomUUID()}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2) + "\n", {
    mode: 0o600,
    flag: "wx",
  });
  await rename(temporary, path);
  await chmod(path, 0o600);
}
export async function readJson(path, fallback) {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    if (error.code === "ENOENT" && fallback !== undefined) return fallback;
    throw error;
  }
}
export class ConfigStore {
  constructor(root = privateRoot) {
    this.root = root;
    this.knownSecrets = new Set();
  }
  async load() {
    await mkdir(this.root, { recursive: true, mode: 0o700 });
    await chmod(this.root, 0o700);
    this.config = validateConfig(
      await readJson(resolve(this.root, "config.json"), defaults()),
    );
    this.secrets = await readJson(resolve(this.root, "credentials.json"), {});
    return this;
  }
  view() {
    return {
      config: this.config,
      credentials: Object.keys(this.secrets).filter((k) =>
        Boolean(this.secrets[k]),
      ),
    };
  }
  // Saving through one queue keeps concurrent browser tabs from losing key updates.
  save(input) {
    const action = async () => {
      const config = validateConfig(input.config);
      const secrets = { ...this.secrets };
      for (const [name, value] of Object.entries(input.secrets || {})) {
        check(
          /^[A-Z_][A-Z0-9_]*$/.test(name) &&
            typeof value === "string" &&
            value.length <= 16384,
          "Invalid credential",
        );
        if (value === "") delete secrets[name];
        else secrets[name] = value;
      }
      await atomicJson(resolve(this.root, "credentials.json"), secrets);
      await atomicJson(resolve(this.root, "config.json"), config);
      this.secretValues();
      this.config = config;
      this.secrets = secrets;
      return this.view();
    };
    const saved = (this.saving || Promise.resolve()).then(action);
    this.saving = saved.catch(() => {});
    return saved;
  }
  secret(name) {
    const value = this.secrets[name] || process.env[name] || "";
    if (value) this.knownSecrets.add(value);
    return value;
  }
  secretValues(config = this.config) {
    for (const value of [
      ...Object.values(this.secrets),
      ...config.providers.map((p) => process.env[p.apiKeyEnv]),
      ...Object.values(config.image).map((p) => process.env[p.apiKeyEnv]),
    ].filter(Boolean))
      this.knownSecrets.add(value);
    return [...this.knownSecrets];
  }
  redact(value) {
    const secrets = this.secretValues();
    return JSON.parse(
      JSON.stringify(value, (_key, item) =>
        typeof item === "string" ? redactText(item, secrets) : item,
      ),
    );
  }
}
function redactText(text, secrets) {
  for (const secret of secrets) text = text.split(secret).join("[redacted]");
  return text;
}
// Preserve only a possible key prefix so a key split across pipe chunks stays private.
export function streamRedactor(secrets) {
  let pending = "";
  return (chunk = "", final = false) => {
    pending = redactText(pending + chunk, secrets);
    let keep = 0;
    if (!final)
      for (const secret of secrets)
        for (
          let n = Math.min(secret.length - 1, pending.length);
          n > keep;
          n--
        ) {
          if (secret.startsWith(pending.slice(-n))) {
            keep = n;
            break;
          }
        }
    const text = keep ? pending.slice(0, -keep) : pending;
    pending = keep ? pending.slice(-keep) : "";
    return text;
  };
}
