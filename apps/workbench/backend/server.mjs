import { createServer } from "node:http";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { access, readFile, realpath, stat } from "node:fs/promises";
import { resolve, relative, extname, sep } from "node:path";
import { homedir } from "node:os";
import { spawn } from "node:child_process";
import { ConfigStore, appRoot, repoRoot, check } from "./config.mjs";
import { Sessions } from "./sessions.mjs";
import { Jobs } from "./jobs.mjs";
import { artifactPath, imageRequest } from "./images.mjs";

const port = Number(process.env.DATAEVOLVER_WORKBENCH_PORT || 3080);
check(
  Number.isInteger(port) && port > 0 && port <= 65535,
  "Invalid workbench port",
);
const address = `http://127.0.0.1:${port}`;
const browserOrigin = new URL(
  process.env.DATAEVOLVER_WORKBENCH_BROWSER_ORIGIN || address,
).origin;
const hosts = new Set([new URL(address).host, new URL(browserOrigin).host]);
let startupToken = randomBytes(32).toString("hex");
const cookieToken = randomBytes(32).toString("hex");
const cookieName = `dataevolver_${port}`;
const subscribers = new Set();
const store = await new ConfigStore().load();
const emit = (event) => {
  const encoded = `data: ${JSON.stringify(store.redact(event))}\n\n`;
  for (const res of subscribers) {
    if (!res.write(encoded)) {
      subscribers.delete(res);
      res.end();
    }
  }
};
const sessions = new Sessions(store, emit);
sessions.bridgeUrl = address + "/internal/action";
const jobs = new Jobs(store, emit);
const equal = (a, b) =>
  typeof a === "string" &&
  Buffer.byteLength(a) === Buffer.byteLength(b) &&
  timingSafeEqual(Buffer.from(a), Buffer.from(b));
const mime = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
};
async function body(req) {
  let length = 0;
  const buffers = [];
  for await (const chunk of req) {
    length += chunk.length;
    check(length <= 1024 * 1024, "Request exceeds 1 MiB");
    buffers.push(chunk);
  }
  check(
    req.headers["content-type"]?.startsWith("application/json"),
    "Use application/json",
  );
  return JSON.parse(Buffer.concat(buffers).toString("utf8"));
}
async function inspect() {
  let sshHosts = [];
  try {
    const source = await readFile(resolve(homedir(), ".ssh/config"), "utf8");
    sshHosts = [...source.matchAll(/^\s*Host\s+([^#\r\n]+)/gim)]
      .flatMap((match) => match[1].trim().split(/\s+/))
      .filter((x) => /^[a-zA-Z0-9][a-zA-Z0-9_.@-]*$/.test(x));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  const paths = {};
  for (const [key, value] of Object.entries(store.config.paths))
    paths[key] = value
      ? await access(value).then(
          () => "exists",
          () => "not-found",
        )
      : "not-configured";
  return {
    paths,
    sshHosts: [...new Set(sshHosts)],
    enginesStarted: false,
    sshConnected: false,
  };
}
async function internal(id, input) {
  const runtime = sessions.running.get(id);
  check(runtime, "Session is not active");
  if (input.op === "harness") {
    const job = await jobs.start(input.data, id, runtime.config);
    await job.done;
    return jobs.details(job.record.id);
  }
  if (input.op === "image")
    return imageRequest(store, input.data, runtime.config);
  if (input.op === "delegate") {
    check(!runtime.record.parentId, "Only a main session can delegate");
    check(
      typeof input.data.task === "string" && input.data.task.trim(),
      "Child task is required",
    );
    return sessions.create({
      parentId: id,
      name: input.data.name,
      task: input.data.task,
    });
  }
  if (input.op === "child") {
    const child = (await sessions.list()).find(
      (x) => x.id === input.data.id && x.parentId === id,
    );
    check(child, "Child does not belong to this session");
    if (input.data.action === "stop") await sessions.stop(child.id);
    else if (input.data.action === "wait") {
      const running = sessions.running.get(child.id);
      if (running?.state === "running")
        await running.client.waitForIdle(600000);
    } else check(input.data.action === "inspect", "Unknown child action");
    return sessions.details(child.id);
  }
  throw new Error("Unknown internal operation");
}
const server = createServer(async (req, res) => {
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("X-Frame-Options", "DENY");
  res.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
  );
  const send = (status, value, type = "application/json; charset=utf-8") => {
    res.writeHead(status, { "Content-Type": type });
    res.end(
      type.startsWith("application/json")
        ? JSON.stringify(store.redact(value))
        : value,
    );
  };
  try {
    if (!hosts.has(req.headers.host))
      return send(403, { error: "Host is not allowed" });
    if (
      req.headers.origin &&
      req.headers.origin !== address &&
      req.headers.origin !== browserOrigin
    )
      return send(403, { error: "Origin is not allowed" });
    if (req.headers["sec-fetch-site"] === "cross-site")
      return send(403, { error: "Cross-site access is not allowed" });
    const url = new URL(req.url, address);
    const path = url.pathname;
    if (path === "/internal/action") {
      const auth = req.headers.authorization;
      const id = auth?.startsWith("Bearer ")
        ? sessions.tokens.get(auth.slice(7))
        : null;
      if (!id || req.method !== "POST")
        return send(403, { error: "Invalid session capability" });
      return send(200, await internal(id, await body(req)));
    }
    if (
      req.method === "GET" &&
      path === "/" &&
      startupToken &&
      equal(url.searchParams.get("token"), startupToken)
    ) {
      startupToken = null;
      res.setHeader(
        "Set-Cookie",
        `${cookieName}=${cookieToken}; HttpOnly; SameSite=Strict; Path=/${browserOrigin.startsWith("https:") ? "; Secure" : ""}`,
      );
      res.writeHead(303, { Location: "/" });
      return res.end();
    }
    const cookie = req.headers.cookie
      ?.split(";")
      .map((x) => x.trim())
      .find((x) => x.startsWith(cookieName + "="))
      ?.slice(cookieName.length + 1);
    if (!equal(cookie, cookieToken))
      return send(
        401,
        "Use the startup link printed by launch.sh. Restart the workbench to obtain a new link.",
        "text/plain; charset=utf-8",
      );
    if (req.method === "GET" && path === "/api/events") {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      });
      res.write('data: {"type":"connected"}\n\n');
      subscribers.add(res);
      const heartbeat = setInterval(() => res.write(": heartbeat\n\n"), 25000);
      req.once("close", () => {
        clearInterval(heartbeat);
        subscribers.delete(res);
      });
      return;
    }
    if (req.method === "GET" && path === "/api/bootstrap")
      return send(200, {
        ...store.view(),
        sessions: await sessions.list(),
        jobs: await jobs.list(),
        inspection: await inspect(),
      });
    if (req.method === "POST" && path === "/api/config")
      return send(200, await store.save(await body(req)));
    if (req.method === "POST" && path === "/api/sessions")
      return send(201, await sessions.create(await body(req)));
    const session = path.match(
      /^\/api\/sessions\/([a-zA-Z0-9_.-]+)(\/commands)?$/,
    );
    if (session && req.method === "GET" && !session[2])
      return send(200, await sessions.details(session[1]));
    if (session && req.method === "POST" && session[2])
      return send(200, await sessions.command(session[1], await body(req)));
    if (req.method === "POST" && path === "/api/harness") {
      const job = await jobs.start(await body(req));
      return send(202, job.record);
    }
    const job = path.match(/^\/api\/jobs\/([a-zA-Z0-9_.-]+)$/);
    if (req.method === "GET" && job)
      return send(200, await jobs.details(job[1]));
    if (req.method === "POST" && path === "/api/image")
      return send(200, await imageRequest(store, await body(req)));
    if (req.method === "GET" && path === "/api/artifact") {
      const file = await artifactPath(
        store.config,
        url.searchParams.get("path") || "",
      );
      const type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
      }[extname(file).toLowerCase()];
      check(
        type && (await stat(file)).size <= 30 * 1024 * 1024,
        "Use an image of at most 30 MiB",
      );
      return send(200, await readFile(file), type);
    }
    if (path.startsWith("/api/"))
      return send(404, { error: "Unknown API route" });
    if (req.method !== "GET") return send(405, { error: "Method not allowed" });
    if (path === "/logo.png")
      return send(
        200,
        await readFile(
          resolve(repoRoot, "apps/project-page/frontend/logo.png"),
        ),
        "image/png",
      );
    const root = await realpath(resolve(appRoot, "dist"));
    const file = await realpath(
      resolve(
        root,
        path === "/" ? "index.html" : "." + decodeURIComponent(path),
      ),
    );
    const rel = relative(root, file);
    check(
      rel && rel !== ".." && !rel.startsWith(".." + sep) && mime[extname(file)],
      "Invalid asset",
    );
    return send(200, await readFile(file), mime[extname(file)]);
  } catch (error) {
    if (!res.headersSent)
      send(error.code === "ENOENT" ? 404 : 400, {
        error: error.message,
        receipt: error.receipt,
      });
    else res.end();
  }
});
server.requestTimeout = 0;
server.headersTimeout = 15000;
server.listen(port, "127.0.0.1", () => {
  const url = `${browserOrigin}/?token=${startupToken}`;
  console.log(
    `DataEvolver Workbench: ${url}\nPrivate configuration: ${store.root}\nSingle-user agent tools run with this host account's permissions.\nFor a remote host: ssh -N -L ${port}:127.0.0.1:${port} <your-ssh-alias>`,
  );
  if (!process.argv.includes("--no-open") && !process.env.SSH_CONNECTION) {
    const command = process.platform === "darwin" ? "open" : "xdg-open";
    const child = spawn(command, [url], { stdio: "ignore" });
    child.on("error", () =>
      console.log("Open the printed URL in your browser."),
    );
  }
});
server.on("error", (error) => {
  console.error(error.message);
  process.exitCode = 1;
});
for (const signal of ["SIGINT", "SIGTERM"])
  process.once(signal, async () => {
    for (const res of subscribers) res.end();
    await jobs.close();
    await sessions.close();
    server.close();
    server.closeAllConnections();
  });
