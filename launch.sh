#!/usr/bin/env bash
set -euo pipefail

# Keep the original entry point while the launcher belongs to the optional app.
checkout_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
exec "$checkout_dir/apps/workbench/launch.sh" "$@"
