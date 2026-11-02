import { readFile, mkdir, realpath, writeFile, stat } from "node:fs/promises";
import { createHash, randomUUID } from "node:crypto";
import { resolve, relative, extname, sep } from "node:path";
import { check, endpoint, atomicJson } from "./config.mjs";

export async function artifactPath(config, path) {
  const file = await realpath(resolve(config.paths.workspace, path));
  const roots = [
    config.paths.output,
    config.paths.assets,
    config.paths.datasets,
    config.paths.workspace,
  ].filter(Boolean);
  for (const root of roots) {
    const known = await realpath(root).catch(() => null);
    if (!known) continue;
    const rel = relative(known, file);
    if (rel && rel !== ".." && !rel.startsWith(".." + sep)) return file;
  }
  throw new Error("Artifact is outside configured roots");
}
export async function imageRequest(store, input, config = store.config) {
  check(
    ["generation", "edit", "promptEnhancer"].includes(input.kind),
    "Unknown image operation",
  );
  check(
    typeof input.prompt === "string" &&
      input.prompt.trim() &&
      input.prompt.length <= 50000,
    "Supply a bounded prompt",
  );
  const route = config.image[input.kind];
  check(route.model && route.baseUrl, "Configure this image route first");
  const key = route.apiKeyEnv ? store.secret(route.apiKeyEnv) : "";
  check(!route.apiKeyEnv || key, "Image API credential is missing");
  const headers = key ? { Authorization: `Bearer ${key}` } : {};
  let body;
  let suffix;
  if (input.kind === "promptEnhancer") {
    suffix = "/chat/completions";
    headers["Content-Type"] = "application/json";
    const content = [{ type: "text", text: input.prompt }];
    if (input.image) {
      const file = await artifactPath(config, input.image);
      check(
        (await stat(file)).size <= 20 * 1024 * 1024,
        "Image exceeds 20 MiB",
      );
      const bytes = await readFile(file);
      const mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
      }[extname(file).toLowerCase()];
      check(mime, "Use PNG, JPEG or WebP");
      content.push({
        type: "image_url",
        image_url: { url: `data:${mime};base64,${bytes.toString("base64")}` },
      });
    }
    body = JSON.stringify({
      model: route.model,
      messages: [{ role: "user", content }],
      stream: false,
    });
  } else if (input.kind === "generation") {
    suffix = "/images/generations";
    headers["Content-Type"] = "application/json";
    body = JSON.stringify({
      model: route.model,
      prompt: input.prompt,
      n: 1,
      response_format: "b64_json",
    });
  } else {
    check(
      typeof input.image === "string" && input.image,
      "Editing requires an input image path on the workbench host",
    );
    const file = await artifactPath(config, input.image);
    check((await stat(file)).size <= 20 * 1024 * 1024, "Image exceeds 20 MiB");
    const bytes = await readFile(file);
    const mime = {
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".webp": "image/webp",
    }[extname(file).toLowerCase()];
    check(mime, "Use PNG, JPEG or WebP");
    suffix = "/images/edits";
    body = new FormData();
    body.set("model", route.model);
    body.set("prompt", input.prompt);
    body.set("response_format", "b64_json");
    body.set(
      "image",
      new Blob([bytes], { type: mime }),
      "input" + extname(file),
    );
  }
  const url = endpoint(route.baseUrl, false) + suffix;
  const root = resolve(config.paths.output, "workbench-images", randomUUID());
  await mkdir(root, { recursive: true, mode: 0o700 });
  const intent = {
    kind: input.kind,
    model: route.model,
    prompt: input.prompt,
    createdAt: new Date().toISOString(),
    status: "pending",
  };
  await atomicJson(resolve(root, "receipt.json"), store.redact(intent));
  try {
    const response = await fetch(url, {
      method: "POST",
      headers,
      body,
      redirect: "error",
      signal: AbortSignal.timeout(600000),
    });
    check(
      response.ok,
      `Image endpoint returned HTTP ${response.status}; inspect the retained intent before resending`,
    );
    let length = 0;
    const chunks = [];
    for await (const chunk of response.body) {
      length += chunk.length;
      check(length <= 45 * 1024 * 1024, "Image response exceeds 45 MiB");
      chunks.push(chunk);
    }
    const data = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    let result;
    if (input.kind === "promptEnhancer") {
      const text = data.choices?.[0]?.message?.content;
      check(
        typeof text === "string" && text,
        "Prompt enhancer returned no text",
      );
      result = { text };
    } else {
      const encoded = data.data?.[0]?.b64_json;
      check(
        typeof encoded === "string" &&
          encoded.length <= 100000000 &&
          /^[A-Za-z0-9+/\s]*={0,2}$/.test(encoded),
        "Endpoint must return a bounded b64_json image",
      );
      const bytes = Buffer.from(encoded, "base64");
      check(bytes.length <= 30 * 1024 * 1024, "Returned image exceeds 30 MiB");
      const png = bytes
        .subarray(0, 8)
        .equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
      const jpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
      const webp =
        bytes.toString("ascii", 0, 4) === "RIFF" &&
        bytes.toString("ascii", 8, 12) === "WEBP";
      check(png || jpeg || webp, "Unsupported returned image format");
      const path = resolve(
        root,
        `image.${png ? "png" : jpeg ? "jpg" : "webp"}`,
      );
      await writeFile(path, bytes, { mode: 0o600, flag: "wx" });
      result = {
        path,
        sha256: createHash("sha256").update(bytes).digest("hex"),
        host: "workbench",
      };
    }
    const receipt = {
      ...intent,
      status: "completed",
      result,
      usage: data.usage || null,
    };
    await atomicJson(resolve(root, "receipt.json"), store.redact(receipt));
    return { ...result, receipt: resolve(root, "receipt.json") };
  } catch (error) {
    await atomicJson(
      resolve(root, "receipt.json"),
      store.redact({ ...intent, status: "unknown", error: error.message }),
    );
    throw Object.assign(error, { receipt: resolve(root, "receipt.json") });
  }
}
