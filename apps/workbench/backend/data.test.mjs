import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readDataPackage } from "./data.mjs";
import { harnessArgs } from "./jobs.mjs";

// Explicit Python is supplied by the developer's environment, never a model/engine.
test("package viewer contains paths, pages complete UTF-8 records and checks image hashes", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "dataevolver-viewer-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "export/images"), { recursive: true });
  const image = Buffer.from("fixture-image-bytes");
  const imageHash = createHash("sha256").update(image).digest("hex");
  const manifest = Array.from(
    { length: 21 },
    (_, n) => JSON.stringify({ id: `样本-${n}` }) + "\n",
  ).join("");
  await writeFile(join(root, "export/manifest.jsonl"), manifest);
  await writeFile(join(root, "export/images/a.png"), image);
  await writeFile(
    join(root, "export/dataset.json"),
    JSON.stringify({
      schema_version: "dataevolver.curated_images.v2",
      count: 21,
      members: {
        "manifest.jsonl": "not-validated-by-viewer",
        "images/a.png": imageHash,
      },
    }),
  );
  const config = {
    execution: { target: "local" },
    paths: {
      python: process.env.DATAEVOLVER_VIEWER_TEST_PYTHON || "python3",
      workspace: root,
      datasets: root,
    },
  };
  const read = (input) =>
    readDataPackage(config, {
      ...input,
      scope: JSON.stringify([config.execution, config.paths]),
    });
  await assert.rejects(
    readDataPackage(config, { package: "export", scope: "stale-host" }),
    /Reload Workbench/,
  );
  const page = await read({
    package: "export",
    table: "manifest.jsonl",
  });
  assert.equal(page.rows.length, 20);
  assert.equal(page.rows[19].id, "样本-19");
  const tail = await read({
    package: "export",
    table: "manifest.jsonl",
    cursor: page.nextCursor,
  });
  assert.equal(tail.rows[0].id, "样本-20");
  assert.equal(tail.nextCursor, null);
  assert.match(
    (
      await read({
        package: "export",
        image: "images/a.png",
      })
    ).dataUrl,
    /^data:image\/png;base64,/,
  );
  await writeFile(join(root, "export/images/a.png"), "changed");
  await assert.rejects(
    read({ package: "export", image: "images/a.png" }),
    /hash mismatch/,
  );
  await assert.rejects(read({ package: "../outside" }), /relative/);
  await assert.rejects(
    read({
      package: "export",
      table: "manifest.jsonl",
      cursor: 1,
    }),
    /record boundary/,
  );
});

test("plan bridge accepts only the existing explicit Goal argument", () => {
  assert.deepEqual(harnessArgs("plan", { goal: "goals/collect.json" }), [
    "-m",
    "dataevolver",
    "plan",
    "--goal",
    "goals/collect.json",
  ]);
  assert.throws(
    () => harnessArgs("plan", { goal: "goal.json", command: "step" }),
    /Unknown Harness argument/,
  );
});

test("fixture creation forwards only an explicit boolean dry-run flag", () => {
  const values = {
    goal: "fixture.json",
    campaignRoot: "campaigns",
    idempotencyKey: "fixture",
  };
  assert.equal(
    harnessArgs("create", { ...values, dryRun: true }).at(-1),
    "--dry-run",
  );
  assert.equal(
    harnessArgs("create", { ...values, dryRun: false }).includes("--dry-run"),
    false,
  );
  assert.throws(
    () => harnessArgs("create", { ...values, dryRun: "true" }),
    /boolean/,
  );
});
