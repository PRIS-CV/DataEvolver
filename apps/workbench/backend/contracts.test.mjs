import test from "node:test";
import assert from "node:assert/strict";
import {
  defaults,
  validateConfig,
  endpoint,
  streamRedactor,
} from "./config.mjs";
import { harnessArgs, launchCommand, shellQuote } from "./jobs.mjs";

test("deployment endpoints reject credentials and query secrets", () => {
  for (const url of [
    "file:///etc/passwd",
    "https://key@host/v1",
    "https://host/v1?key=secret",
  ])
    assert.throws(() => endpoint(url));
  assert.equal(
    endpoint("https://models.example/v1/"),
    "https://models.example/v1",
  );
});
test("agent model bindings must refer to declared provider models", () => {
  const config = defaults();
  config.agents.main = {
    provider: "missing",
    model: "missing",
    thinking: "high",
  };
  assert.throws(() => validateConfig(config), /not registered/);
});
test("bounded step is explicit and arbitrary CLI fields cannot be injected", () => {
  assert.deepEqual(
    harnessArgs("step", {
      campaign: "example",
      campaignRoot: "runs",
      maxActions: 2,
    }),
    [
      "-m",
      "dataevolver",
      "step",
      "--campaign",
      "example",
      "--campaign-root",
      "runs",
      "--max-actions",
      "2",
    ],
  );
  assert.throws(() =>
    harnessArgs("step", {
      campaign: "example",
      campaignRoot: "runs",
      maxActions: 0,
    }),
  );
  assert.throws(() =>
    harnessArgs("status", {
      campaign: "example",
      campaignRoot: "runs",
      arbitrary: "--help",
    }),
  );
  assert.throws(() => harnessArgs("evolve", {}));
});
test("SSH keeps metacharacters in arguments and uses strict host checking", () => {
  assert.equal(shellQuote("a'b; $(value)"), "'a'\\''b; $(value)'");
  const config = defaults();
  Object.assign(config.execution, {
    target: "ssh",
    sshHost: "workstation",
    remoteWorkspace: "/srv/my checkout",
  });
  const spec = launchCommand(config, [
    "-m",
    "dataevolver",
    "status",
    "--campaign",
    "sample;echo injected",
  ]);
  assert.equal(spec.executable, "ssh");
  assert.ok(spec.args.includes("StrictHostKeyChecking=yes"));
  assert.ok(spec.args.at(-1).includes("'sample;echo injected'"));
  config.execution.sshHost = "-oProxyCommand=bad";
  assert.throws(() => launchCommand(config, []));
});

test("chunked output never emits a configured key split across chunks", () => {
  const redact = streamRedactor(["private-key"]);
  assert.equal(redact("before pri"), "before ");
  assert.equal(redact("vate-key after"), "[redacted] after");
  assert.equal(redact("", true), "");
});
test("configuration cannot round-trip a credential hidden in an unknown property", () => {
  const config = defaults();
  config.apiKey = "example";
  assert.throws(() => validateConfig(config), /configuration field/);
});
