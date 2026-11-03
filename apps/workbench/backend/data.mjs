import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { check } from "./config.mjs";
import { launchCommand } from "./jobs.mjs";

const reader = await readFile(
  new URL("./data_reader.py", import.meta.url),
  "utf8",
);

// Use the same host/Python selection as Harness jobs; this reader imports no project code.
export async function readDataPackage(config, input) {
  check(
    input && typeof input === "object" && !Array.isArray(input),
    "Invalid package request",
  );
  check(
    Object.keys(input).every((key) =>
      ["package", "table", "cursor", "image", "scope"].includes(key),
    ),
    "Unknown package argument",
  );
  check(
    typeof input.package === "string" && input.package.length <= 8192,
    "Invalid package path",
  );
  check(
    !input.table ||
      ["manifest.jsonl", "quarantine.jsonl", "reviews.jsonl"].includes(
        input.table,
      ),
    "Unsupported record table",
  );
  check(
    input.cursor === undefined ||
      (Number.isSafeInteger(input.cursor) && input.cursor >= 0),
    "Invalid cursor",
  );
  check(
    input.image === undefined ||
      (typeof input.image === "string" &&
        input.image.length > 0 &&
        input.image.length <= 8192),
    "Invalid image path",
  );
  check(
    input.scope === JSON.stringify([config.execution, config.paths]),
    "Execution host or paths changed. Reload Workbench before reading the package",
  );
  const { scope: _scope, ...request } = input;
  const root =
    config.execution.target === "ssh"
      ? config.execution.remoteDatasets
      : config.paths.datasets;
  check(
    root,
    "Configure the dataset directory in Settings for the selected host",
  );
  const command = launchCommand(config, [
    "-c",
    reader,
    JSON.stringify({ ...request, root }),
  ]);
  return new Promise((resolve, reject) => {
    const child = spawn(command.executable, command.args, {
      cwd: command.cwd,
      env: command.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const chunks = [];
    let length = 0;
    let errorLength = 0;
    const errors = [];
    let failed = false;
    const fail = (message) => {
      if (failed) return;
      failed = true;
      child.kill("SIGTERM");
      reject(new Error(message));
    };
    const timer = setTimeout(
      () => fail("Dataset reader timed out; inspect the host connection"),
      30000,
    );
    child.stdout.on("data", (chunk) => {
      length += chunk.length;
      if (length > 16 * 1024 * 1024)
        return fail("Dataset response exceeds 16 MiB");
      chunks.push(chunk);
    });
    child.stderr.on("data", (chunk) => {
      errorLength += chunk.length;
      if (errorLength <= 8192) errors.push(chunk);
    });
    child.once("error", (error) => {
      clearTimeout(timer);
      fail(error.message);
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (failed) return;
      try {
        const output = Buffer.concat(chunks).toString("utf8");
        if (code !== 0) {
          let message =
            Buffer.concat(errors).toString("utf8") ||
            `Dataset reader exited ${code}`;
          try {
            message = JSON.parse(output).error || message;
          } catch {
            /* SSH can fail before Python starts. */
          }
          throw new Error(message);
        }
        resolve(JSON.parse(output));
      } catch (error) {
        reject(error);
      }
    });
  });
}
