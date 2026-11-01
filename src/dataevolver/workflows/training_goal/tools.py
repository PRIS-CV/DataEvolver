"""Two read-only MCP tools plus experience query; no model or shell execution."""
from __future__ import annotations

import argparse
import hashlib
import html
import ipaddress
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def public_url(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("source URL must be public HTTPS without credentials")
    if parts.port not in (None, 443):
        raise ValueError("nonstandard source port")
    host = parts.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".internal")):
        raise ValueError("private source endpoint")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and not ip.is_global:
        raise ValueError("private source address")
    return url


class ReadOnlyTools:
    def __init__(self, run_dir):
        self.root = Path(run_dir)
        self.catalog = json.loads((self.root / "catalog.json").read_text())
        self.snapshot = json.loads((self.root / "experience-snapshot.json").read_text())

    def _get(self, url):
        request = urllib.request.Request(public_url(url), headers={"User-Agent": "DataEvolver-Research/1.0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            public_url(response.url)
            body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("source exceeds 2 MiB; select a smaller primary-source page")
            return body, response.url, dict(response.headers)

    def read_source(self, url):
        original, resolved, version = public_url(url), url, None
        # Resolve mutable GitHub source links to a commit before reading code.
        match = re.match(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url)
        if match:
            owner, repo, ref, relative = match.groups()
            if re.fullmatch(r"[0-9a-f]{40}", ref):
                version = ref
            else:
                body, _, _ = self._get(f"https://api.github.com/repos/{owner}/{repo}/commits/{ref}")
                version = json.loads(body)["sha"]
            resolved = f"https://raw.githubusercontent.com/{owner}/{repo}/{version}/{relative}"
        body, fetched, headers = self._get(resolved)
        content_hash = hashlib.sha256(body).hexdigest()
        version = version or headers.get("ETag") or "sha256:" + content_hash
        text = body.decode("utf-8", errors="replace")
        if "<html" in text.lower() or "<!doctype html" in text.lower():
            text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", text, flags=re.S | re.I)
            text = html.unescape(re.sub(r"<[^>]+>", " ", text))
            text = re.sub(r"[ \t]+", " ", text)
        identity = "source_" + hashlib.sha256((fetched + content_hash).encode()).hexdigest()[:16]
        (self.root / "sources").mkdir(exist_ok=True)
        (self.root / "sources" / (identity + ".txt")).write_text(text)
        evidence = {"evidence_id": identity, "ok": True, "url": original, "resolved_url": fetched,
                    "version": version, "sha256": content_hash, "fetched_at": time.time(),
                    "content_file": f"sources/{identity}.txt", "bytes": len(body)}
        append(self.root / "source-evidence.jsonl", evidence)
        return {**evidence, "content": text[:50000], "truncated": len(text) > 50000}

    def call(self, name, arguments):
        start = time.time()
        try:
            if name == "read_source":
                value = self.read_source(arguments["url"])
            elif name == "query_operator_catalog":
                query = arguments.get("query", "").lower()
                rows = [r for r in self.catalog["operators"] if any(
                    word in (r["operator"] + " " + r["description"]).lower() for word in query.split()
                )] if query else self.catalog["operators"]
                value = {**self.catalog, "operators": rows or self.catalog["operators"]}
            elif name == "query_experiences":
                value = self.snapshot
            else:
                raise ValueError("unregistered read-only tool")
            append(self.root / "tool-trace.jsonl", {"tool": name, "arguments": arguments,
                   "ok": True, "started_at": start, "finished_at": time.time(),
                   "result": {k: v for k, v in value.items() if k != "content"}})
            return value
        except Exception as exc:
            error = {"tool": name, "arguments": arguments, "ok": False,
                     "started_at": start, "finished_at": time.time(),
                     "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            append(self.root / "tool-trace.jsonl", error)
            raise


TOOLS = [
    {"name": "read_source", "description": "Actually fetch a primary source; returns evidence ID and exact version for citations and external code candidates.",
     "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"], "additionalProperties": False}},
    {"name": "query_operator_catalog", "description": "Query actual local implementations, reusable candidates, I/O and readiness. Designed operations are not executable.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "query_experiences", "description": "Read compatible PROMOTED experiences from the immutable per-plan snapshot; empty is valid.",
     "inputSchema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
]

# These tools read public sources or immutable snapshots. Explicit MCP hints
# prevent a missing annotation from being interpreted as a destructive action.
for descriptor in TOOLS:
    descriptor["annotations"] = {"readOnlyHint": True, "destructiveHint": False,
                                 "idempotentHint": True,
                                 "openWorldHint": descriptor["name"] == "read_source"}


def serve(run_dir):
    service = ReadOnlyTools(run_dir)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if "id" not in request:
                continue
            method = request.get("method")
            if method == "initialize":
                result = {"protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                          "capabilities": {"tools": {}}, "serverInfo": {"name": "dataevolver-planning", "version": "1"}}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request["params"]
                try:
                    value = service.call(params["name"], params.get("arguments", {}))
                    result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}
                except Exception as exc:
                    result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
            elif method == "ping":
                result = {}
            else:
                raise ValueError("unsupported MCP method")
            answer = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as exc:
            answer = {"jsonrpc": "2.0", "id": locals().get("request", {}).get("id"),
                      "error": {"code": -32603, "message": str(exc)}}
        print(json.dumps(answer, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    serve(parser.parse_args().run_dir)
