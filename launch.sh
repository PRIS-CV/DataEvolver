#!/usr/bin/env bash
set -euo pipefail

checkout_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
workbench_dir="$checkout_dir/apps/workbench"
venv_dir="${DATAEVOLVER_VENV:-$checkout_dir/.venv}"
python_bin="${DATAEVOLVER_BOOTSTRAP_PYTHON:-python3}"
sol_dir="${DATAEVOLVER_SOL_ROOT:-$checkout_dir/.dataevolver/workbench-deps/sol-pi}"
sol_revision="${DATAEVOLVER_SOL_REVISION:-219568034fa54ed121ede80961655e33dfb9d0e3}"
install=1
open_browser=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-install) install=0 ;;
    --no-open) open_browser=0 ;;
    --port)
      [ "$#" -ge 2 ] || { echo '--port needs a value' >&2; exit 2; }
      export DATAEVOLVER_WORKBENCH_PORT="$2"; shift ;;
    --help|-h)
      echo 'Usage: ./launch.sh [--no-install] [--no-open] [--port PORT]'
      echo 'Installs the core Python/Node environment and opens the private workbench.'
      echo 'Native engines, weights, GPU software and remote hosts remain host-managed.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

command -v node >/dev/null || { echo 'Install Node.js 22.19+ first.' >&2; exit 1; }
command -v npm >/dev/null || { echo 'Install npm first.' >&2; exit 1; }
command -v "$python_bin" >/dev/null || { echo 'Install Python 3.10+ or set DATAEVOLVER_BOOTSTRAP_PYTHON.' >&2; exit 1; }
case "$venv_dir:$sol_dir" in
  /*:/*) ;;
  *) echo 'DATAEVOLVER_VENV and DATAEVOLVER_SOL_ROOT must use absolute directories.' >&2; exit 1 ;;
esac
node -e 'const [major,minor]=process.versions.node.split(".").map(Number); if(major<22 || major===22 && minor<19) { console.error("Node.js 22.19+ is required"); process.exit(1); }'
"$python_bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is required")'

if [ "$install" -eq 1 ]; then
  [ -x "$venv_dir/bin/python" ] || "$python_bin" -m venv "$venv_dir"
  "$venv_dir/bin/python" -m pip install -e "$checkout_dir[harness]"
  (cd "$workbench_dir" && npm ci --ignore-scripts)
  command -v git >/dev/null || { echo 'Install Git to obtain SoL-Pi.' >&2; exit 1; }
  if [ ! -d "$sol_dir/.git" ]; then
    mkdir -p "$(dirname -- "$sol_dir")"
    git clone https://github.com/NVlabs/SoL-Pi.git "$sol_dir"
  fi
  if [ "$(git -C "$sol_dir" rev-parse HEAD)" != "$sol_revision" ]; then
    if [ -n "$(git -C "$sol_dir" status --porcelain)" ]; then
      echo "SoL-Pi has local changes. Preserve them before updating $sol_dir." >&2; exit 1
    fi
    git -C "$sol_dir" fetch origin "$sol_revision"
    git -C "$sol_dir" checkout --detach FETCH_HEAD
  fi
  (cd "$sol_dir" && npm ci --ignore-scripts)
  (cd "$workbench_dir" && npm run build)
fi

[ -x "$venv_dir/bin/python" ] || { echo 'Python environment missing; rerun without --no-install.' >&2; exit 1; }
[ -f "$workbench_dir/dist/index.html" ] || { echo 'Frontend build missing; rerun without --no-install.' >&2; exit 1; }
[ -f "$workbench_dir/node_modules/@earendil-works/pi-coding-agent/package.json" ] || { echo 'Pi dependency missing; rerun without --no-install.' >&2; exit 1; }
[ -f "$sol_dir/src/sol-pi/index.ts" ] || { echo 'SoL-Pi is missing; rerun without --no-install or configure DATAEVOLVER_SOL_ROOT.' >&2; exit 1; }
node -e 'const fs=require("node:fs"); const p=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); const expected=JSON.parse(fs.readFileSync(process.argv[2],"utf8")).dependencies[p.name]; if(p.version!==expected) { console.error("Pi version does not match the workbench dependency; reinstall dependencies."); process.exit(1); } console.log("Pi:",p.version);' "$workbench_dir/node_modules/@earendil-works/pi-coding-agent/package.json" "$workbench_dir/package.json"
for native_tool in blender UnrealEditor ssh; do
  if command -v "$native_tool" >/dev/null; then
    echo "$native_tool: available on PATH"
  else
    echo "$native_tool: configure its executable or OpenSSH on the intended host in Settings"
  fi
done
export DATAEVOLVER_WORKBENCH_PYTHON="$venv_dir/bin/python"
export DATAEVOLVER_SOL_EXTENSION="$sol_dir/src/sol-pi/index.ts"
if [ "$open_browser" -eq 0 ]; then
  exec node "$workbench_dir/backend/server.mjs" --no-open
fi
exec node "$workbench_dir/backend/server.mjs"
