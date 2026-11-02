import {
  RpcClient,
  buildSessionContext,
  parseSessionEntries,
} from "@earendil-works/pi-coding-agent";
import { randomUUID, randomBytes } from "node:crypto";
import { mkdir, readdir, access, readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  check,
  identifier,
  atomicJson,
  readJson,
  appRoot,
  thinkingLevels,
} from "./config.mjs";

export class Sessions {
  constructor(store, emit) {
    this.store = store;
    this.emit = emit;
    this.running = new Map();
    this.tokens = new Map();
    this.pending = new Map();
  }
  root(id) {
    return resolve(this.store.root, "sessions", identifier(id));
  }
  async list() {
    const root = resolve(this.store.root, "sessions");
    await mkdir(root, { recursive: true, mode: 0o700 });
    const records = await Promise.all(
      (await readdir(root, { withFileTypes: true }))
        .filter((x) => x.isDirectory())
        .map(async (x) => {
          const saved = await readJson(
            resolve(root, x.name, "record.json"),
            null,
          );
          return saved
            ? {
                ...saved,
                state: this.running.get(saved.id)?.state || "stopped",
              }
            : null;
        }),
    );
    return records
      .filter(Boolean)
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }
  async create({ parentId = null, name = "", task = "" } = {}) {
    let config = structuredClone(this.store.config);
    if (parentId) {
      const parent = await readJson(
        resolve(this.root(parentId), "record.json"),
      );
      check(!parent.parentId, "Child sessions cannot delegate again");
      config = await readJson(resolve(this.root(parentId), "config.json"));
    }
    const id = randomUUID();
    const root = this.root(id);
    await mkdir(root, { recursive: true, mode: 0o700 });
    const record = {
      id,
      parentId,
      name: String(
        name || (parentId ? "Child task" : "New conversation"),
      ).slice(0, 100),
      createdAt: new Date().toISOString(),
      profile: { ...config.agents[parentId ? "child" : "main"] },
      state: "stopped",
    };
    check(
      record.profile.provider && record.profile.model,
      "Configure an agent provider and model first",
    );
    await atomicJson(resolve(root, "config.json"), config);
    await atomicJson(resolve(root, "record.json"), record);
    this.emit({ type: "session_created", session: record });
    try {
      const runtime = await this.open(id);
      if (task) await this.command(id, { command: "prompt", message: task });
      return { ...record, state: runtime.state };
    } catch (error) {
      this.emit({ type: "session_error", sessionId: id, error: error.message });
      throw error;
    }
  }
  async open(id) {
    const runtime = this.running.get(id);
    if (runtime) {
      try {
        await runtime.client.getState();
        return runtime;
      } catch (error) {
        await this.stop(id);
        throw new Error(
          `Pi session unavailable; inspect native effects before reopening: ${error.message}`,
        );
      }
    }
    if (this.pending.has(id)) return this.pending.get(id);
    const opening = this.start(id);
    this.pending.set(id, opening);
    try {
      return await opening;
    } finally {
      this.pending.delete(id);
    }
  }
  async start(id) {
    const root = this.root(id);
    const record = await readJson(resolve(root, "record.json"));
    const config = await readJson(resolve(root, "config.json"));
    const agentDir = resolve(root, "agent");
    await mkdir(agentDir, { recursive: true, mode: 0o700 });
    const providers = {};
    const env = {
      PI_CODING_AGENT_DIR: agentDir,
      PI_TELEMETRY: "0",
      DATAEVOLVER_PYTHON: config.paths.python,
    };
    for (const [index, p] of config.providers.entries()) {
      const name = `DATAEVOLVER_PROVIDER_${index}_KEY`;
      const key = p.apiKeyEnv ? this.store.secret(p.apiKeyEnv) : "not-required";
      if (
        p.id === record.profile.provider ||
        (config.sol.evidencePreservingReducer &&
          p.id === config.sol.reducerProvider)
      )
        check(key, `Credential not configured: ${p.apiKeyEnv}`);
      env[name] = key;
      providers[p.id] = {
        api: p.api,
        baseUrl: p.baseUrl,
        apiKey: "${" + name + "}",
        models: p.models,
      };
    }
    await atomicJson(resolve(agentDir, "models.json"), { providers });
    await atomicJson(resolve(agentDir, "settings.json"), {
      packages: [],
      extensions: [],
      skills: [],
      promptTemplates: [],
      themes: [],
    });
    const { enabled, reducerProvider, reducerModel, ...features } = config.sol;
    const solConfig = { version: 1, ...features };
    if (reducerProvider && reducerModel)
      Object.assign(solConfig, {
        evidencePreservingReducerProvider: reducerProvider,
        evidencePreservingReducerModel: reducerModel,
      });
    await atomicJson(resolve(agentDir, "sol-pi.json"), solConfig);
    const token = randomBytes(32).toString("hex");
    env.DATAEVOLVER_WORKBENCH_BRIDGE = this.bridgeUrl;
    env.DATAEVOLVER_WORKBENCH_CAPABILITY = token;
    const args = [
      "--offline",
      "--no-approve",
      "--no-extensions",
      "--no-mcp",
      "--no-skills",
      "--no-prompt-templates",
      "--no-themes",
      "--no-context-files",
      "--session",
      resolve(root, "pi.jsonl"),
      "--thinking",
      record.profile.thinking,
    ];
    if (enabled) {
      await access(config.pi.solExtension);
      args.push("-e", config.pi.solExtension);
    }
    args.push("-e", resolve(appRoot, "extensions/dataevolver.ts"));
    args.push(
      "--append-system-prompt",
      `You are the DataEvolver outer agent. Workspace: ${config.paths.workspace}. Read AGENT_SETUP.md and selected workflow contracts. Use dataevolver_harness to compile-plan or worker-compile, then create and bounded step. The existing Controller owns campaign state, budgets, dispatch and recovery. SoL update_plan is your conversation plan, not the frozen Harness goal. Inspect receipts after interruptions; never resend an unknown external effect. Campaign resume immediately runs work; conversation resume only reopens this Pi session. Deployment target and configured tools: ${JSON.stringify({ paths: config.paths, execution: config.execution })}. Author code with the native tools. Use configured image and delegation tools. Child tasks are independent sessions; send explicit context and inspect their result. Do not claim execution, quality or recovery without evidence.`,
    );
    const cliPath =
      config.pi.cli ||
      resolve(
        dirname(
          fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent")),
        ),
        "bundle/cli.js",
      );
    const client = new RpcClient({
      cliPath,
      cwd: config.paths.workspace,
      env,
      provider: record.profile.provider,
      model: record.profile.model,
      args,
    });
    const runtime = { client, record, config, state: "starting", token };
    client.onEvent((event) => {
      if (event.type === "agent_start") runtime.state = "running";
      if (event.type === "agent_settled")
        runtime.state = event.aborted ? "aborted" : "idle";
      this.emit({
        type: "pi_event",
        sessionId: id,
        state: runtime.state,
        event,
      });
    });
    try {
      this.tokens.set(token, id);
      await client.start();
      await client.setAutoRetry(false);
      await client.getState();
      runtime.state = "idle";
      this.running.set(id, runtime);
      this.emit({
        type: "session_opened",
        sessionId: id,
        state: runtime.state,
      });
      return runtime;
    } catch (error) {
      this.tokens.delete(token);
      await client.stop();
      throw error;
    }
  }
  async details(id) {
    const runtime = this.running.get(id);
    const record = await readJson(resolve(this.root(id), "record.json"));
    if (runtime) {
      try {
        return {
          session: record,
          state: runtime.state,
          messages: await runtime.client.getMessages(),
          pi: await runtime.client.getState(),
        };
      } catch (error) {
        runtime.state = "unknown";
        await this.stop(id);
        this.emit({
          type: "session_error",
          sessionId: id,
          error: error.message,
        });
      }
    }
    const text = await readFile(
      resolve(this.root(id), "pi.jsonl"),
      "utf8",
    ).catch((error) => {
      if (error.code === "ENOENT") return "";
      throw error;
    });
    return {
      session: record,
      state: runtime?.state || "stopped",
      messages: buildSessionContext(
        parseSessionEntries(text).filter((x) => x.type !== "session"),
      ).messages,
      archived: true,
    };
  }
  async command(id, input) {
    const { command, message } = input;
    const allowed = [
      "prompt",
      "steer",
      "follow_up",
      "abort",
      "resume",
      "set_model",
      "stop",
    ];
    check(allowed.includes(command), "Unknown session command");
    if (command === "stop") {
      await this.stop(id);
      return { stopped: true };
    }
    if (command === "abort") {
      const runtime = this.running.get(id) || (await this.pending.get(id));
      if (runtime) {
        await runtime.client.clearQueue();
        await runtime.client.abort();
        runtime.state = "aborted";
      }
      for (const child of (await this.list()).filter((x) => x.parentId === id))
        await this.stop(child.id);
      return { aborted: true, campaignPaused: false };
    }
    const runtime = await this.open(id);
    const { client } = runtime;
    if (command === "resume")
      return { state: await client.getState(), executionStarted: false };
    if (command === "set_model") {
      check(runtime.state !== "running", "Switch models between agent runs");
      check(
        runtime.config.providers.some(
          (p) =>
            p.id === input.provider &&
            p.models.some((m) => m.id === input.model),
        ),
        "Model is not in this session snapshot; create a new session after adding providers",
      );
      check(thinkingLevels.includes(input.thinking), "Invalid thinking level");
      await client.setModel(input.provider, input.model);
      await client.setThinkingLevel(input.thinking);
      runtime.record.profile = {
        provider: input.provider,
        model: input.model,
        thinking: input.thinking,
      };
      await atomicJson(resolve(this.root(id), "record.json"), runtime.record);
      return runtime.record;
    }
    check(
      typeof message === "string" && message.trim() && message.length <= 100000,
      "A bounded nonempty message is required",
    );
    let disposition;
    if (command === "steer") disposition = await client.steer(message);
    else if (command === "follow_up")
      disposition = await client.followUp(message);
    else disposition = await client.prompt(message);
    return { disposition };
  }
  async stop(id) {
    if (this.pending.has(id)) await this.pending.get(id).catch(() => {});
    const runtime = this.running.get(id);
    if (!runtime) return;
    this.tokens.delete(runtime.token);
    this.running.delete(id);
    await runtime.client.stop();
    this.emit({ type: "session_stopped", sessionId: id });
  }
  async close() {
    await Promise.all(
      [...new Set([...this.running.keys(), ...this.pending.keys()])].map((id) =>
        this.stop(id),
      ),
    );
  }
}
