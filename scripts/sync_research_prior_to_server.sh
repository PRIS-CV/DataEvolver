#!/usr/bin/env bash
# Sync a local research_prior.json to the Linux data-build server.
#
# This script intentionally has no private defaults. Pass host and remote run
# root explicitly, or provide them through your local environment.

set -euo pipefail

PRIOR_PATH=""
REMOTE="${DATAEVOLVER_REMOTE:-}"
REMOTE_DIR="${DATAEVOLVER_REMOTE_RUN_ROOT:-}"
DEST_NAME="research_prior.json"
METHOD="${DATAEVOLVER_SYNC_METHOD:-scp}"
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage:
  scripts/sync_research_prior_to_server.sh \
    --prior-path /local/path/research_prior.json \
    --remote user-or-ssh-alias \
    --remote-dir /remote/run/root \
    [--dest-name research_prior.json] [--method scp|rsync] [--dry-run]

Output:
  Prints the remote Stage1 argument:
    --research-prior-path /remote/run/root/research_prior.json

No server host, username, or private path is hardcoded in this script.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prior-path) PRIOR_PATH="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --dest-name) DEST_NAME="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$PRIOR_PATH" || -z "$REMOTE" || -z "$REMOTE_DIR" ]]; then
    echo "[ERROR] --prior-path, --remote, and --remote-dir are required" >&2
    usage
    exit 1
fi

if [[ ! -f "$PRIOR_PATH" ]]; then
    echo "[ERROR] prior file not found: $PRIOR_PATH" >&2
    exit 1
fi

if [[ "$DEST_NAME" == */* ]]; then
    echo "[ERROR] --dest-name must be a file name, not a path" >&2
    exit 1
fi

REMOTE_PATH="${REMOTE_DIR%/}/${DEST_NAME}"

mkdir_cmd=(ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'")
if [[ "$METHOD" == "scp" ]]; then
    copy_cmd=(scp "$PRIOR_PATH" "${REMOTE}:${REMOTE_PATH}")
elif [[ "$METHOD" == "rsync" ]]; then
    copy_cmd=(rsync -av "$PRIOR_PATH" "${REMOTE}:${REMOTE_PATH}")
else
    echo "[ERROR] --method must be scp or rsync" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] ${mkdir_cmd[*]}"
    echo "[dry-run] ${copy_cmd[*]}"
else
    "${mkdir_cmd[@]}"
    "${copy_cmd[@]}"
fi

echo "--research-prior-path ${REMOTE_PATH}"
