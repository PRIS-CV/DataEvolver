import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { createHash } from "node:crypto";
import { mkdtemp, rm, mkdir, writeFile, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { ConfigStore, defaults, validateConfig, atomicJson, readJson } from "./config.mjs";
import { Jobs, harnessArgs, launchCommand } from "./jobs.mjs";
import { Sessions, harnessSessionConfig, sessionRoleInstructions } from "./sessions.mjs";

const step = (campaign) => ({
  command: "step",
  values: { campaign, campaignRoot: "runs", maxActions: 1 },
});

async function fixture(t) {
  const root = await mkdtemp(resolve(tmpdir(), "dataevolver-orchestration-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const store = await new ConfigStore(root).load();
  const processes = [];
  const events = new EventEmitter();
  const seen = [];
  const jobs = new Jobs(store, (event) => {
    seen.push(event);
    events.emit("event", event);
  }, (executable, args, options) => {
    const child = new EventEmitter();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = () => true;
    child.finish = (code, signal = null) => {
      child.stdout.end();
      child.stderr.end();
      child.emit("close", code, signal);
    };
    processes.push({ child, executable, args, options });
    return child;
  });
  const started = async (id) => {
    if (seen.some((event) => event.type === "job_started" && event.job.id === id)) return;
    await new Promise((resolve) => {
      const listener = (event) => {
        if (event.type === "job_started" && event.job.id === id) {
          events.off("event", listener);
          resolve();
        }
      };
      events.on("event", listener);
    });
  };
  return { root, store, jobs, processes, started };
}

test("version-1 configurations upgrade without mutating their input", () => {
  const old = defaults();
  delete old.orchestration;
  assert.deepEqual(validateConfig(old).orchestration, defaults().orchestration);
  assert.equal(Object.hasOwn(old, "orchestration"), false);
  for (const value of [0, 1.5, 33, "2"]) {
    const invalid = defaults();
    invalid.orchestration.maxConcurrentJobs = value;
    assert.throws(() => validateConfig(invalid), /orchestration limit/);
  }
});

test("FIFO execution keeps its config snapshot and permits inspection", { timeout: 5000 }, async (t) => {
  const { store, jobs, processes, started } = await fixture(t);
  const first = await jobs.start(step("first"));
  await started(first.record.id);
  const second = await jobs.start(step("second"));
  assert.equal(second.record.status, "queued");
  const python = store.config.paths.python;
  store.config.paths.python = "/another/configured/python";
  const inspection = await jobs.start({
    command: "status", values: { campaign: "first", campaignRoot: "runs" },
  });
  await started(inspection.record.id);
  assert.equal(processes.length, 2);
  processes[0].child.finish(0);
  await first.done;
  await started(second.record.id);
  assert.equal(processes[2].executable, python);
  processes[1].child.finish(0);
  processes[2].child.finish(0);
  await Promise.all([second.done, inspection.done]);
});

test("overlapping starts reserve admission order before setup completes", { timeout: 5000 }, async (t) => {
  const { jobs, processes, started } = await fixture(t);
  const firstStart = jobs.start(step("first"));
  const secondStart = jobs.start(step("second"));
  assert.deepEqual(jobs.queue.map((item) => item.record.values.campaign), ["first", "second"]);
  // A ready follower must not overtake a head whose receipt is still being written.
  jobs.queue[1].ready = true;
  jobs.drain();
  assert.equal(jobs.active.size, 0);
  jobs.queue[1].ready = false;
  const [first, second] = await Promise.all([firstStart, secondStart]);
  await started(first.record.id);
  assert.equal(processes.length, 1);
  assert.ok(processes[0].args.includes("first"));
  processes[0].child.finish(0);
  await started(second.record.id);
  assert.ok(processes[1].args.includes("second"));
  processes[1].child.finish(0);
  await Promise.all([first.done, second.done]);
});

test("queue capacity and session abort cancel only work not dispatched", { timeout: 5000 }, async (t) => {
  const { store, jobs, processes, started } = await fixture(t);
  store.config.orchestration.maxQueuedJobs = 1;
  const first = await jobs.start(step("running"), "parent");
  await started(first.record.id);
  const waiting = await jobs.start(step("waiting"), "parent");
  await assert.rejects(jobs.start(step("overflow")), /queue is full/);
  await jobs.cancelSession("parent");
  assert.equal((await waiting.done).status, "cancelled");
  assert.equal(processes.length, 1);
  await assert.rejects(jobs.cancel(first.record.id), /Only queued/);
  processes[0].child.finish(0);
  await first.done;
});

test("an unknown SSH effect cancels waiting executions", { timeout: 5000 }, async (t) => {
  const { store, jobs, processes, started } = await fixture(t);
  Object.assign(store.config.execution, {
    target: "ssh", sshHost: "example-workstation", remoteWorkspace: "/workspace",
  });
  const first = await jobs.start(step("first"));
  await started(first.record.id);
  const waiting = await jobs.start(step("waiting"));
  const inspectionPromise = jobs.start({
    command: "status", values: { campaign: "first", campaignRoot: "runs" },
  });
  processes[0].child.finish(255);
  assert.equal((await first.done).status, "unknown");
  assert.equal((await waiting.done).status, "cancelled");
  const inspection = await inspectionPromise;
  await started(inspection.record.id);
  assert.equal(processes.length, 2);
  processes[1].child.finish(0);
  assert.equal((await inspection.done).status, "succeeded");
});

test("a live dispatch is not mistaken for a queued restart leftover", async (t) => {
  const { jobs } = await fixture(t);
  const id = "dispatching";
  await mkdir(jobs.root(id), { recursive: true });
  await atomicJson(resolve(jobs.root(id), "record.json"), {
    id, status: "queued", createdAt: "2026-01-01T00:00:00Z",
  });
  jobs.active.set(id, { child: null, done: Promise.resolve() });
  const record = (await jobs.list()).find((item) => item.id === id);
  assert.equal(record.status, "queued");
  assert.equal(record.error, undefined);
  jobs.active.delete(id);
});

test("an unknown read-only inspection leaves waiting execution intact", { timeout: 5000 }, async (t) => {
  const { store, jobs, processes, started } = await fixture(t);
  Object.assign(store.config.execution, {
    target: "ssh", sshHost: "example-workstation", remoteWorkspace: "/workspace",
  });
  const first = await jobs.start(step("first"));
  await started(first.record.id);
  const waiting = await jobs.start(step("waiting"));
  const inspection = await jobs.start({
    command: "status", values: { campaign: "first", campaignRoot: "runs" },
  });
  await started(inspection.record.id);
  processes[1].child.finish(255);
  assert.equal((await inspection.done).status, "unknown");
  assert.equal(jobs.pending.get(waiting.record.id)?.record.status, "queued");
  processes[0].child.finish(0);
  await started(waiting.record.id);
  processes[2].child.finish(0);
  await Promise.all([first.done, waiting.done]);
});

test("an uncertain pause still cancels waiting execution", { timeout: 5000 }, async (t) => {
  const { store, jobs, processes, started } = await fixture(t);
  Object.assign(store.config.execution, {
    target: "ssh", sshHost: "example-workstation", remoteWorkspace: "/workspace",
  });
  const first = await jobs.start(step("first"));
  await started(first.record.id);
  const waiting = await jobs.start(step("waiting"));
  const pause = await jobs.start({
    command: "pause", values: { campaign: "first", campaignRoot: "runs" },
  });
  await started(pause.record.id);
  processes[1].child.finish(255);
  assert.equal((await pause.done).status, "unknown");
  assert.equal((await waiting.done).status, "cancelled");
  processes[0].child.finish(0);
  await first.done;
});

test("restart inspection does not replay persisted running or queued work", async (t) => {
  const { jobs, processes } = await fixture(t);
  for (const status of ["running", "queued"]) {
    await mkdir(jobs.root(status), { recursive: true });
    await atomicJson(resolve(jobs.root(status), "record.json"), {
      id: status, status, createdAt: "2026-01-01T00:00:00Z",
    });
  }
  const records = new Map((await jobs.list()).map((record) => [record.id, record]));
  assert.equal(records.get("running").status, "unknown");
  assert.equal(records.get("queued").status, "cancelled");
  assert.equal(processes.length, 0);
});

test("explicit context is bounded, redacted and observation-only", async (t) => {
  const { store, jobs } = await fixture(t);
  store.config.orchestration.maxContextBytes = 2048;
  store.secrets.CONTEXT_KEY = "private-context-key";
  const root = jobs.root("evidence");
  await mkdir(root, { recursive: true });
  await atomicJson(resolve(root, "record.json"), {
    id: "evidence", command: "step", status: "succeeded", target: "local",
    createdAt: "2026-01-01T00:00:00Z", exitCode: 0,
  });
  await writeFile(resolve(root, "output.log"), "x".repeat(100000) + store.secrets.CONTEXT_KEY);
  const context = await jobs.context(["evidence"]);
  assert.ok(Buffer.byteLength(JSON.stringify(context)) <= 2048);
  assert.equal(JSON.stringify(context).includes(store.secrets.CONTEXT_KEY), false);
  assert.equal(context.jobs[0].truncated, true);
  assert.equal(context.jobs[0].status, "succeeded");
  assert.match(context.interpretation, /not proof/);
  await assert.rejects(jobs.context(["evidence", "evidence"]), /distinct/);
});

test("proposal children get distinct empty workspaces; task children retain shared paths", async (t) => {
  const { store, jobs } = await fixture(t);
  store.config.agents.main = store.config.agents.child = {
    provider: "fixture", model: "fixture", thinking: "off",
  };
  class InertSessions extends Sessions {
    async open() { return { state: "idle" }; }
    async command() { throw new Error("This fixture must not prompt an agent"); }
  }
  const sessions = new InertSessions(store, () => {}, jobs);
  const parent = await sessions.create();
  const first = await sessions.create({ parentId: parent.id, mode: "proposal" });
  const second = await sessions.create({ parentId: parent.id, mode: "proposal" });
  const task = await sessions.create({ parentId: parent.id });
  assert.notEqual(first.workspace, parent.workspace);
  assert.notEqual(first.workspace, second.workspace);
  assert.equal(task.workspace, parent.workspace);
  assert.equal(first.sourceWorkspace, parent.workspace);
  assert.equal(createHash("sha256").update(await readFile(first.contextPath)).digest("hex"), first.contextSha256);
  const config = JSON.parse(await readFile(resolve(sessions.root(first.id), "config.json")));
  assert.notEqual(config.paths.output, store.config.paths.output);
  const execution = harnessSessionConfig({ record: first, config });
  for (const command of ["status", "explain", "replay"]) {
    const args = harnessArgs(command, {
      campaign: "parent-campaign", campaignRoot: ".dataevolver/campaigns",
    });
    assert.equal(launchCommand(execution, args).cwd, parent.workspace);
  }
  assert.equal(config.paths.workspace, first.workspace);
  assert.equal(execution.execution, config.execution);
  const taskConfig = JSON.parse(await readFile(resolve(sessions.root(task.id), "config.json")));
  assert.equal(harnessSessionConfig({ record: task, config: taskConfig }), taskConfig);
  assert.equal(harnessSessionConfig({ record: {}, config: taskConfig }), taskConfig);
  await assert.rejects(sessions.create({ parentId: first.id }), /cannot delegate again/);
});

test("capacity rejection creates no child directory or history", async (t) => {
  const { store, jobs } = await fixture(t);
  store.config.agents.main = store.config.agents.child = {
    provider: "fixture", model: "fixture", thinking: "off",
  };
  store.config.orchestration.maxOpenChildren = 1;
  class InertSessions extends Sessions {
    async open() { return { state: "idle" }; }
  }
  const sessions = new InertSessions(store, () => {}, jobs);
  const parent = await sessions.create();
  await sessions.create({ parentId: parent.id });
  const directory = resolve(store.root, "sessions");
  const before = (await readdir(directory)).sort();
  for (const mode of ["task", "proposal"])
    await assert.rejects(sessions.create({ parentId: parent.id, mode }), /capacity reached/);
  assert.deepEqual((await readdir(directory)).sort(), before);
  assert.equal((await sessions.list()).length, 2);
  assert.equal(sessions.childSlots.size, 1);
  sessions.childSlots.clear();
  sessions.open = async () => { throw new Error("fixture startup failure"); };
  await assert.rejects(sessions.create({ parentId: parent.id }), /fixture startup failure/);
  assert.equal(sessions.childSlots.size, 0);
});

test("delegation guidance is limited to main sessions", () => {
  const config = defaults();
  assert.match(sessionRoleInstructions({}, config), /delegate with mode=proposal/);
  const child = sessionRoleInstructions({ parentId: "parent", mode: "task" }, config);
  assert.match(child, /cannot delegate/);
  assert.doesNotMatch(child, /delegate with mode=proposal/);
  const proposal = sessionRoleInstructions({ parentId: "parent", mode: "proposal" }, config);
  assert.match(proposal, /only Harness status\/explain\/replay/);
  assert.doesNotMatch(proposal, /delegate with mode=proposal/);
});

test("review freezes a redacted original question without copying author history", async (t) => {
  const { store, jobs } = await fixture(t);
  store.config.agents.main = store.config.agents.child = {
    provider: "fixture", model: "fixture", thinking: "off",
  };
  store.secrets.REVIEW_KEY = "private-review-key";
  const prompts = [];
  class InertSessions extends Sessions {
    async open() { return { state: "idle" }; }
    async command(id, input) { prompts.push({ id, ...input }); }
  }
  const sessions = new InertSessions(store, () => {}, jobs);
  const parent = await sessions.create();
  await writeFile(resolve(sessions.root(parent.id), "pi.jsonl"), "AUTHOR_CONCLUSION");
  const task = `Inspect src/dataevolver/harness/controller.py statically. ${store.secrets.REVIEW_KEY}`;
  const child = await sessions.create({ parentId: parent.id, mode: "review", task });
  const bytes = await readFile(child.contextPath);
  assert.ok(bytes.length <= store.config.orchestration.maxContextBytes);
  const context = JSON.parse(bytes);
  assert.equal(context.schemaVersion, "dataevolver.review_context.v1");
  assert.equal(context.reviewRequest.sourceWorkspace, parent.workspace);
  assert.match(context.reviewRequest.question, /controller.py/);
  assert.equal(bytes.includes(Buffer.from(store.secrets.REVIEW_KEY)), false);
  assert.equal(bytes.includes(Buffer.from("AUTHOR_CONCLUSION")), false);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), child.contextSha256);
  assert.deepEqual(await readdir(child.workspace), []);
  assert.equal(prompts.length, 1);
  assert.equal(prompts[0].message.includes(task), false);
  assert.ok(prompts[0].message.includes(child.contextPath));
  const config = await readJson(resolve(sessions.root(child.id), "config.json"));
  assert.equal(harnessSessionConfig({ record: child, config }).paths.workspace, parent.workspace);
  const instructions = sessionRoleInstructions(child, config);
  assert.match(instructions, /not independent reproduction/);
  assert.match(instructions, /only Harness status\/explain\/replay/);
  assert.match(instructions, /not a security sandbox/);
  assert.doesNotMatch(instructions, /delegate with mode=proposal/);
  await writeFile(child.contextPath, "{}");
  await assert.rejects(sessions.start(child.id), /context changed/);
});

test("review input and profile overrides are checked before creating a child", async (t) => {
  const { store, jobs } = await fixture(t);
  store.config.providers = [{
    id: "fixture", api: "openai-completions", baseUrl: "https://models.example/v1",
    apiKeyEnv: "", models: ["author", "reviewer"].map((id) => ({
      id, reasoning: true, input: ["text"], contextWindow: 4096, maxTokens: 1024,
    })),
  }];
  store.config.agents.main = store.config.agents.child = {
    provider: "fixture", model: "author", thinking: "off",
  };
  class InertSessions extends Sessions {
    async open() { return { state: "idle" }; }
    async command() {}
  }
  const sessions = new InertSessions(store, () => {}, jobs);
  const parent = await sessions.create();
  const before = await readdir(resolve(store.root, "sessions"));
  const review = { parentId: parent.id, mode: "review", task: "Inspect source/file.py." };
  await assert.rejects(sessions.create({ ...review, task: " " }), /original question/);
  await assert.rejects(sessions.create({ ...review, task: "x".repeat(40000) }), /maxContextBytes/);
  for (const profile of [
    { provider: "fixture", model: "missing", thinking: "off" },
    { provider: "fixture", model: "reviewer", thinking: "invalid" },
    { provider: "fixture", model: "reviewer", thinking: "off", endpoint: "https://other.example" },
    { provider: "", model: "", thinking: "off" },
  ]) await assert.rejects(sessions.create({ ...review, profile }));
  // A new global provider is unavailable to an already-created parent.
  store.config.providers.push({ ...store.config.providers[0], id: "later" });
  await assert.rejects(sessions.create({ ...review, profile: { provider: "later", model: "reviewer", thinking: "off" } }), /not registered/);
  assert.deepEqual(await readdir(resolve(store.root, "sessions")), before);
  assert.equal(sessions.childSlots.size, 0);
  const profile = { provider: "fixture", model: "reviewer", thinking: "high" };
  const child = await sessions.create({ ...review, profile });
  assert.deepEqual(child.profile, profile);
  const bytes = await readFile(child.contextPath);
  const oneByteTooLarge = review.task + "x".repeat(store.config.orchestration.maxContextBytes - bytes.length + 1);
  const directories = await readdir(resolve(store.root, "sessions"));
  await assert.rejects(sessions.create({ ...review, task: oneByteTooLarge, profile }), /maxContextBytes/);
  assert.deepEqual(await readdir(resolve(store.root, "sessions")), directories);
  const saved = await readJson(resolve(sessions.root(child.id), "config.json"));
  assert.deepEqual(saved.agents.child, profile);
  assert.equal(saved.providers.length, 1);
  assert.equal(store.config.agents.child.model, "author");
  const normal = await sessions.create({ parentId: parent.id });
  assert.equal(normal.profile.model, "author");
  assert.equal(normal.workspace, parent.workspace);
});
