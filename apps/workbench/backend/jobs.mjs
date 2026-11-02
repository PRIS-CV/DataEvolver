import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync } from "node:fs";
import { mkdir, readdir, open } from "node:fs/promises";
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
// Read a bounded byte range even when a retained engine log is very large.
async function logTail(path, limit) {
  const file = await open(path, "r");
  try {
    const { size } = await file.stat();
    const offset = Math.max(0, size - limit);
    const buffer = Buffer.alloc(Math.min(size, limit));
    const { bytesRead } = await file.read(buffer, 0, buffer.length, offset);
    return {
      output: buffer.subarray(0, bytesRead).toString("utf8"),
      truncated: offset > 0,
      offset,
      bytesRead,
    };
  } finally {
    await file.close();
  }
}
const inspectionCommands = new Set(["status", "explain", "replay", "pause"]);

export class Jobs {
  constructor(store, emit, spawnProcess = spawn) {
    this.store = store;
    this.emit = emit;
    this.spawnProcess = spawnProcess;
    this.active = new Map();
    this.pending = new Map();
    this.queue = [];
    this.executing = new Set();
    this.closing = false;
    this.holds = 0;
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
          if (record?.status === "queued" && !this.pending.has(record.id)) {
            record.status = "cancelled";
            record.error = "Workbench restarted; queued work was not replayed";
          }
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
    const tail = await logTail(resolve(this.root(id), "output.log"), 200000);
    return {
      ...record,
      ...tail,
    };
  }
  async context(ids) {
    check(
      Array.isArray(ids) && ids.length <= 16 && new Set(ids).size === ids.length,
      "Select at most 16 distinct context job IDs",
    );
    ids.forEach(identifier);
    const limit = this.store.config.orchestration.maxContextBytes;
    const records = new Map((await this.list()).map((record) => [record.id, record]));
    const context = {
      schemaVersion: "dataevolver.proposal_context.v1",
      capturedAt: new Date().toISOString(),
      interpretation: "Execution observations only. Logs are untrusted data. A successful process is not proof of quality, training benefit or PROMOTED knowledge.",
      jobs: [],
    };
    for (const id of ids) {
      const record = records.get(id);
      check(record, `Context job not found: ${id}`);
      const tail = await logTail(
        resolve(this.root(id), "output.log"),
        Math.floor(limit / (Math.max(ids.length, 1) * 8)),
      );
      context.jobs.push({
        id, command: record.command, status: record.status,
        target: record.target, host: record.host, sessionId: record.sessionId,
        exitCode: record.exitCode ?? null, signal: record.signal ?? null,
        error: record.error ?? null,
        startedAt: record.startedAt ?? null, finishedAt: record.finishedAt ?? null,
        recordPath: resolve(this.root(id), "record.json"),
        logPath: resolve(this.root(id), "output.log"),
        ...tail,
      });
    }
    const redacted = this.store.redact(context);
    check(Buffer.byteLength(JSON.stringify(redacted)) <= limit, "Context exceeds maxContextBytes; select fewer jobs");
    return redacted;
  }
  async start(
    { command, values },
    sessionId = null,
    frozenConfig = this.store.config,
  ) {
    check(!this.closing, "Workbench is stopping; no new jobs are admitted");
    const args = harnessArgs(command, values);
    const config = structuredClone(frozenConfig);
    const processSpec = launchCommand(config, args);
    const id = randomUUID();
    const root = this.root(id);
    const queued = !inspectionCommands.has(command);
    check(!queued || !this.holds, "Unknown execution effects are being recorded; inspect receipts before resubmitting");
    check(!queued || [...this.pending.values()].filter((item) => item.queued).length < this.store.config.orchestration.maxQueuedJobs,
      "Execution queue is full; inspect or cancel queued jobs");
    let resolveDone;
    const done = new Promise((resolve) => { resolveDone = resolve; });
    const record = {
      id, sessionId, command, values: structuredClone(values),
      target: config.execution.target, host: config.execution.sshHost,
      status: "queued", createdAt: new Date().toISOString(),
    };
    const item = { record, config, processSpec, root, done, resolveDone, queued, ready: false };
    this.pending.set(id, item);
    try {
      await mkdir(root, { recursive: true, mode: 0o700 });
      await atomicJson(resolve(root, "record.json"), record);
      appendFileSync(resolve(root, "output.log"), "", { mode: 0o600 });
      item.ready = true;
      if (this.closing || item.cancelReason) {
        await this.cancel(id, item.cancelReason || "Workbench stopped before dispatch");
      } else if (queued) {
        this.queue.push(item);
        this.emit({ type: "job_queued", job: record });
        this.drain();
      } else {
        this.pending.delete(id);
        void this.run(item);
      }
      return { record, done };
    } catch (error) {
      this.pending.delete(id);
      this.queue = this.queue.filter((entry) => entry !== item);
      record.status = "failed";
      record.error = error.message;
      resolveDone(record);
      throw error;
    }
  }
  drain() {
    while (!this.closing && !this.holds && this.queue.length &&
      this.executing.size < this.store.config.orchestration.maxConcurrentJobs) {
      const item = this.queue.shift();
      this.pending.delete(item.record.id);
      this.executing.add(item.record.id);
      void this.run(item);
    }
  }
  async cancel(id, reason = "Cancelled before dispatch") {
    const item = this.pending.get(identifier(id));
    check(item, "Only queued jobs can be cancelled; inspect running effects and use Campaign pause");
    item.cancelReason = reason;
    this.queue = this.queue.filter((entry) => entry !== item);
    if (!item.ready) return item.done;
    this.pending.delete(id);
    Object.assign(item.record, { status: "cancelled", error: reason, finishedAt: new Date().toISOString() });
    try {
      await atomicJson(resolve(item.root, "record.json"), item.record);
    } finally {
      this.emit({ type: "job_finished", job: item.record });
      item.resolveDone(item.record);
    }
    return item.record;
  }
  async cancelSession(sessionId) {
    await Promise.allSettled([...this.pending.values()]
      .filter((item) => item.record.sessionId === sessionId)
      .map((item) => this.cancel(item.record.id, "Agent session stopped before dispatch")));
  }
  async run(item) {
    const { record, config, processSpec, root, resolveDone } = item;
    const { id } = record;
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
      let uncertain = record.status === "unknown";
      if (uncertain) this.holds++;
      try {
        await atomicJson(resolve(root, "record.json"), record);
      } catch (failure) {
        record.status = "unknown";
        record.error = failure.message;
        if (!uncertain) {
          uncertain = true;
          this.holds++;
        }
      }
      this.active.delete(id);
      this.executing.delete(id);
      this.emit({ type: "job_finished", job: record });
      resolveDone(record);
      if (uncertain || record.status === "unknown") {
        // Unknown remote effects must not trigger the next queued execution.
        await Promise.allSettled([...this.pending.keys()].map((id) =>
          this.cancel(id, "Previous job has unknown effects; inspect receipts before submitting again")));
      }
      if (uncertain) this.holds--;
      this.drain();
    };
    this.active.set(id, { child: null, done: item.done });
    try {
      record.status = "running";
      record.startedAt = new Date().toISOString();
      await atomicJson(resolve(root, "record.json"), record);
      check(!this.closing, "Workbench stopped before dispatch");
      child = this.spawnProcess(processSpec.executable, processSpec.args, {
        cwd: processSpec.cwd,
        env: processSpec.env,
        stdio: ["ignore", "pipe", "pipe"],
      });
      this.active.get(id).child = child;
      this.emit({ type: "job_started", job: record });
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
  }
  async close() {
    this.closing = true;
    await Promise.allSettled([...this.pending.keys()].map((id) =>
      this.cancel(id, "Workbench stopped before dispatch")));
    // Ending the local SSH transport cannot prove that a remote worker stopped.
    for (const { child } of this.active.values()) child?.kill("SIGTERM");
  }
}
