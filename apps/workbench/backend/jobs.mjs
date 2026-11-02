import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync } from "node:fs";
import { mkdir, readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  atomicJson,
  readJson,
  check,
  identifier,
  streamRedactor,
} from "./config.mjs";

const fields = {
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
export function harnessArgs(command, values = {}) {
  check(Object.hasOwn(fields, command), "Unsupported Harness command");
  check(
    Object.keys(values).every((key) => fields[command].includes(key)),
    "Unknown Harness argument",
  );
  const args = ["-m", "dataevolver", command];
  for (const key of fields[command]) {
    const value = values[key];
    if (key === "maxActions")
      check(
        Number.isSafeInteger(value) && value > 0 && value <= 100,
        "maxActions must be between 1 and 100",
      );
    else
      check(
        typeof value === "string" &&
          value.length > 0 &&
          value.length <= 8192 &&
          !value.includes("\0"),
        `Missing or invalid ${key}`,
      );
    args.push(
      "--" + key.replace(/[A-Z]/g, (letter) => "-" + letter.toLowerCase()),
      String(value),
    );
  }
  return args;
}
export function shellQuote(value) {
  return "'" + String(value).replace(/'/g, "'\\''") + "'";
}
export function launchCommand(config, args) {
  const target = config.execution;
  const keys = ["blender", "ue5", "ueProject", "assets", "datasets", "output"];
  const names = [
    "BLENDER_BIN",
    "UE5_BIN",
    "UE_PROJECT",
    "ASSET_ROOT",
    "DATASET_ROOT",
    "OUTPUT_ROOT",
  ];
  const bindings = Object.fromEntries(
    keys.map((key, i) => [
      `DATAEVOLVER_${names[i]}`,
      target.target === "ssh"
        ? target["remote" + key[0].toUpperCase() + key.slice(1)] || ""
        : config.paths[key],
    ]),
  );
  if (target.target === "local")
    return {
      executable: config.paths.python,
      args,
      cwd: config.paths.workspace,
      env: { ...process.env, ...bindings },
    };
  check(
    /^[a-zA-Z0-9][a-zA-Z0-9_.@-]*$/.test(target.sshHost),
    "Invalid OpenSSH alias",
  );
  const script = `cd ${shellQuote(target.remoteWorkspace)} && exec env ${Object.entries(
    bindings,
  )
    .map(([key, value]) => shellQuote(`${key}=${value}`))
    .join(" ")} ${[target.remotePython, ...args].map(shellQuote).join(" ")}`;
  return {
    executable: "ssh",
    args: [
      "-T",
      "-o",
      "BatchMode=yes",
      "-o",
      "StrictHostKeyChecking=yes",
      "-o",
      "ForwardAgent=no",
      "--",
      target.sshHost,
      script,
    ],
    cwd: config.paths.workspace,
    env: process.env,
  };
}
export class Jobs {
  constructor(store, emit) {
    this.store = store;
    this.emit = emit;
    this.active = new Map();
  }
  root(id) {
    return resolve(this.store.root, "jobs", identifier(id));
  }
  async list() {
    const root = resolve(this.store.root, "jobs");
    await mkdir(root, { recursive: true, mode: 0o700 });
    const rows = await Promise.all(
      (await readdir(root, { withFileTypes: true }))
        .filter((x) => x.isDirectory())
        .map(async (x) => {
          const record = await readJson(
            resolve(root, x.name, "record.json"),
            null,
          );
          if (record?.status === "running" && !this.active.has(record.id))
            record.status = "unknown";
          return record;
        }),
    );
    return rows
      .filter(Boolean)
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }
  async details(id) {
    const record = (await this.list()).find((x) => x.id === id);
    check(record, "Job not found");
    const output = await readFile(resolve(this.root(id), "output.log"), "utf8");
    return {
      ...record,
      output: output.slice(-200000),
      truncated: output.length > 200000,
    };
  }
  async start(
    { command, values },
    sessionId = null,
    frozenConfig = this.store.config,
  ) {
    const args = harnessArgs(command, values);
    const config = structuredClone(frozenConfig);
    const processSpec = launchCommand(config, args);
    const id = randomUUID();
    const root = this.root(id);
    await mkdir(root, { recursive: true, mode: 0o700 });
    const record = {
      id,
      sessionId,
      command,
      values,
      target: config.execution.target,
      host: config.execution.sshHost,
      status: "running",
      createdAt: new Date().toISOString(),
    };
    await atomicJson(resolve(root, "record.json"), record);
    appendFileSync(resolve(root, "output.log"), "", { mode: 0o600 });
    let child;
    let outputFailed = false;
    const writeOutput = (text) => {
      if (!text || outputFailed) return;
      try {
        appendFileSync(resolve(root, "output.log"), text);
      } catch (error) {
        outputFailed = true;
        throw error;
      }
      this.emit({ type: "job_output", jobId: id, text });
    };
    const done = new Promise((resolveDone) => {
      let completed = false;
      const finish = async (code, signal, error) => {
        if (completed) return;
        completed = true;
        record.exitCode = code;
        record.signal = signal;
        record.finishedAt = new Date().toISOString();
        record.status =
          error || signal || (record.target === "ssh" && code === 255)
            ? "unknown"
            : code === 0
              ? "succeeded"
              : "failed";
        if (error) record.error = error.message;
        try {
          await atomicJson(resolve(root, "record.json"), record);
        } catch (failure) {
          record.status = "unknown";
          record.error = failure.message;
        }
        this.active.delete(id);
        this.emit({ type: "job_finished", job: record });
        resolveDone(record);
      };
      try {
        child = spawn(processSpec.executable, processSpec.args, {
          cwd: processSpec.cwd,
          env: processSpec.env,
          stdio: ["ignore", "pipe", "pipe"],
        });
        for (const stream of [child.stdout, child.stderr]) {
          const redact = streamRedactor(this.store.secretValues(config));
          stream.setEncoding("utf8");
          stream.on("data", (chunk) => {
            try {
              writeOutput(redact(chunk));
            } catch (error) {
              child.kill("SIGTERM");
              void finish(null, null, error);
            }
          });
          stream.once("end", () => {
            try {
              writeOutput(redact("", true));
            } catch (error) {
              void finish(null, null, error);
            }
          });
        }
        child.once("error", (error) => void finish(null, null, error));
        child.once("close", (code, signal) => void finish(code, signal));
      } catch (error) {
        void finish(null, null, error);
      }
    });
    this.active.set(id, { child, done });
    this.emit({ type: "job_started", job: record });
    return { record, done };
  }
  async close() {
    // Ending the local SSH transport cannot prove that a remote worker stopped.
    for (const { child } of this.active.values()) child?.kill("SIGTERM");
  }
}
