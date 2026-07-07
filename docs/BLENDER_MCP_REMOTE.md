# Blender MCP Remote Rendering

DataEvolver can expose a remote Blender GUI session to Codex through
`ahujasid/blender-mcp`. This is an optional operator tool. It does not change the
default Stage 4 pipeline or any model execution path.

## Local Install

Start with the dry-run onboarding profile:

```bash
bash src/dataevolver/cli/bootstrap_dataevolver_default.sh \
  --profile blender_mcp \
  --dry-run \
  --write-local-config
```

This writes:

- `.dataevolver/local/env.config.json`
- `.dataevolver/local/ENVIRONMENT.md`
- `.dataevolver/local/env.sh.example`
- `.dataevolver/local/blender_mcp.codex.toml`

The upstream repository is kept as a local checkout:

```bash
git clone https://github.com/ahujasid/blender-mcp.git external/blender-mcp
```

The local checkout is kept for auditing the exact addon/server code and for
copying `addon.py` to the remote machine. For Codex Desktop, the preferred MCP
server mode is SSH stdio: Codex starts `uvx blender-mcp` on the remote host, and
that remote MCP process connects to the remote Blender addon over
`127.0.0.1:9876`.

## Remote Blender Session

Blender MCP needs a non-background Blender process. On a headless server, start
Blender under Xvfb instead of using `blender -b`.

Set the target for your own remote Blender host. These are placeholders; keep
site-specific aliases and filesystem paths in your shell or in
`.dataevolver/local/env.sh.example`, not in committed files.

```bash
export DATAEVOLVER_SSH_BIN=ssh.exe
export DATAEVOLVER_SCP_BIN=scp.exe
export DATAEVOLVER_BLENDER_MCP_REMOTE=my-blender-server
export DATAEVOLVER_REMOTE_ROOT=/path/to/DataEvolver
export DATAEVOLVER_REMOTE_BLENDER_BIN=/path/to/blender
```

Start the remote addon:

```bash
src/dataevolver/workflows/stages/setup_blender_mcp_remote.sh start
src/dataevolver/workflows/stages/setup_blender_mcp_remote.sh status
```

Optional tunnel mode for manual socket debugging:

```bash
src/dataevolver/workflows/stages/setup_blender_mcp_remote.sh tunnel
```

The tunnel forwards local `127.0.0.1:9876` to the remote Blender addon. This is
not the default Codex setup because the SSH stdio mode avoids a long-lived local
tunnel.

```text
BLENDER_HOST=127.0.0.1
BLENDER_PORT=9876
```

Stop the remote Blender session:

```bash
src/dataevolver/workflows/stages/setup_blender_mcp_remote.sh stop
```

## Codex MCP Config

The generated `.dataevolver/local/blender_mcp.codex.toml` starts the MCP server on the remote host:

```toml
[mcp_servers.blender]
command = "ssh.exe"
args = ["my-blender-server", "env", "BLENDER_HOST=127.0.0.1", "BLENDER_PORT=9876", "UV_PYTHON_PREFERENCE=only-managed", "uvx", "--python", "3.11", "blender-mcp"]
startup_timeout_sec = 30
tool_timeout_sec = 240
```

Restart Codex after editing the config so it reloads MCP servers.

Do not let the bootstrap script edit global Codex config directly. Review and copy the generated snippet into the user's own Codex config.

## Notes

- Keep only one Blender MCP client connected to a Blender addon at a time.
- Do not store API keys for Sketchfab, Hyper3D, Hunyuan3D, or other providers in
  repository files. Pass them through the remote shell environment only when
  needed.
- Do not commit lab hostnames, private mount points, local drive paths, or
  user-specific Blender binary paths. Override the placeholder values locally.
