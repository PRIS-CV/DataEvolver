#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi

SSH_BIN="${DATAEVOLVER_SSH_BIN:-ssh.exe}"
SCP_BIN="${DATAEVOLVER_SCP_BIN:-scp.exe}"
REMOTE="${DATAEVOLVER_BLENDER_MCP_REMOTE:-my-blender-server}"
REMOTE_ROOT="${DATAEVOLVER_BLENDER_MCP_REMOTE_ROOT:-${DATAEVOLVER_REMOTE_ROOT:-/path/to/DataEvolver}}"
REMOTE_MCP_DIR="${DATAEVOLVER_BLENDER_MCP_REMOTE_ADDON_DIR:-${DATAEVOLVER_BLENDER_MCP_REMOTE_DIR:-$REMOTE_ROOT/.dataevolver/blender-mcp}}"
REMOTE_BLENDER_BIN="${DATAEVOLVER_REMOTE_BLENDER_BIN:-/path/to/blender}"
REMOTE_PORT="${BLENDER_MCP_REMOTE_PORT:-9876}"
REMOTE_UVX="${BLENDER_MCP_REMOTE_UVX:-uvx}"
LOCAL_PORT="${BLENDER_MCP_LOCAL_PORT:-9876}"
TMUX_SESSION="${BLENDER_MCP_TMUX_SESSION:-dataevolver-blender-mcp}"
REMOTE_LOG="${BLENDER_MCP_REMOTE_LOG:-$REMOTE_MCP_DIR/blender-mcp.log}"

ADDON_SRC="$PROJECT_ROOT/external/blender-mcp/addon.py"
BOOTSTRAP_SRC="$PROJECT_ROOT/src/dataevolver/workflows/stages/blender_mcp_bootstrap.py"

SCP_OPTS=()
if [[ -n "${DATAEVOLVER_SCP_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  SCP_OPTS=(${DATAEVOLVER_SCP_OPTS})
elif [[ "$SCP_BIN" == *.exe ]]; then
  SCP_OPTS=(-O)
fi

usage() {
  cat <<'EOF'
Usage: src/dataevolver/workflows/stages/setup_blender_mcp_remote.sh <install|start|tunnel|status|stop|restart>

Environment overrides:
  DATAEVOLVER_SSH_BIN                 ssh.exe by default, useful from WSL
  DATAEVOLVER_SCP_BIN                 scp.exe by default, useful from WSL
  DATAEVOLVER_BLENDER_MCP_REMOTE      my-blender-server by default
  DATAEVOLVER_BLENDER_MCP_REMOTE_ROOT remote DataEvolver checkout
  DATAEVOLVER_BLENDER_MCP_REMOTE_ADDON_DIR remote addon/bootstrap directory
  DATAEVOLVER_REMOTE_BLENDER_BIN      remote Blender executable
  BLENDER_MCP_REMOTE_PORT             remote addon port, default 9876
  BLENDER_MCP_REMOTE_UVX              remote uvx executable, default uvx
  BLENDER_MCP_LOCAL_PORT              local forwarded port, default 9876
  BLENDER_MCP_TMUX_SESSION            remote tmux session name
  BLENDER_MCP_REMOTE_LOG              remote Blender stdout/stderr log

Typical flow:
  src/dataevolver/workflows/stages/setup_blender_mcp_remote.sh install
  src/dataevolver/workflows/stages/setup_blender_mcp_remote.sh start
  src/dataevolver/workflows/stages/setup_blender_mcp_remote.sh tunnel
EOF
}

remote_sh() {
  "$SSH_BIN" "$REMOTE" "$@"
}

local_scp_path() {
  if [[ "$SCP_BIN" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
    wslpath -w "$1"
  else
    printf '%s\n' "$1"
  fi
}

install_remote() {
  if [[ ! -f "$ADDON_SRC" ]]; then
    echo "Missing $ADDON_SRC. Clone https://github.com/ahujasid/blender-mcp.git into external/blender-mcp first." >&2
    exit 1
  fi
  remote_sh "mkdir -p '$REMOTE_MCP_DIR'"
  "$SCP_BIN" "${SCP_OPTS[@]}" "$(local_scp_path "$ADDON_SRC")" "$(local_scp_path "$BOOTSTRAP_SRC")" "$REMOTE:$REMOTE_MCP_DIR/"
  remote_sh "test -x '$REMOTE_BLENDER_BIN' && (case '$REMOTE_UVX' in */*) test -x '$REMOTE_UVX' ;; *) command -v '$REMOTE_UVX' >/dev/null ;; esac) && command -v xvfb-run >/dev/null && command -v tmux >/dev/null"
  echo "Installed Blender MCP addon files to $REMOTE:$REMOTE_MCP_DIR"
}

start_remote() {
  install_remote
  remote_sh "tmux has-session -t '$TMUX_SESSION' 2>/dev/null && tmux kill-session -t '$TMUX_SESSION' || true"
  remote_sh "cd '$REMOTE_ROOT' && : > '$REMOTE_LOG' && tmux new-session -d -s '$TMUX_SESSION' 'mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root && XDG_RUNTIME_DIR=/tmp/runtime-root GDK_BACKEND=x11 WAYLAND_DISPLAY= BLENDER_MCP_ADDON=\"$REMOTE_MCP_DIR/addon.py\" BLENDER_MCP_PORT=\"$REMOTE_PORT\" xvfb-run -a \"$REMOTE_BLENDER_BIN\" --python \"$REMOTE_MCP_DIR/blender_mcp_bootstrap.py\" >> \"$REMOTE_LOG\" 2>&1'"
  echo "Started remote Blender MCP tmux session: $TMUX_SESSION"
}

tunnel_local() {
  echo "Forwarding 127.0.0.1:$LOCAL_PORT -> $REMOTE:127.0.0.1:$REMOTE_PORT"
  exec "$SSH_BIN" -N -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$REMOTE"
}

status_remote() {
  remote_sh "printf 'tmux: '; tmux has-session -t '$TMUX_SESSION' 2>/dev/null && printf 'running\n' || printf 'stopped\n'; printf 'port: '; (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null || true) | grep ':$REMOTE_PORT ' || true; pgrep -af 'blender.*blender_mcp_bootstrap|Blender MCP' || true; printf 'log: $REMOTE_LOG\n'; tail -40 '$REMOTE_LOG' 2>/dev/null || true"
}

stop_remote() {
  remote_sh "tmux has-session -t '$TMUX_SESSION' 2>/dev/null && tmux kill-session -t '$TMUX_SESSION' || true"
  echo "Stopped remote Blender MCP tmux session if it existed: $TMUX_SESSION"
}

case "${1:-}" in
  install) install_remote ;;
  start) start_remote ;;
  tunnel) tunnel_local ;;
  status) status_remote ;;
  stop) stop_remote ;;
  restart) stop_remote; start_remote ;;
  *) usage; exit 2 ;;
esac
