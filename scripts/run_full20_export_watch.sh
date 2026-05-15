#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${1:-$(pwd)}"
SOURCE_ROOT="${2:-pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404}"
EXPORT_DIR="${3:-pipeline/data/export_scene_v7_full20_rotation8_best_multiview_20260405_horizontal}"
QUEUE_PATH="${QUEUE_PATH:-$CODE_ROOT/pipeline/data/regeneration_queue.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
SCENE_TEMPLATE_PATH="${SCENE_TEMPLATE_PATH:-configs/scene_template.json}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
IDLE_MB="${IDLE_MB:-10000}"
AZIMUTHS_CSV="${AZIMUTHS_CSV:-0,45,90,135,180,225,270,315}"
ELEVATIONS_CSV="${ELEVATIONS_CSV:-0}"
MAX_SCENE_ROUNDS="${MAX_SCENE_ROUNDS:-15}"
TUNER_DONE_MARKER="${TUNER_DONE_MARKER:-$CODE_ROOT/runtime/full20_scene_20260404/render_tuner/render_tuner_done.json}"

cd "$CODE_ROOT"

queue_has_active_jobs() {
  python3 -c "import json, pathlib, sys; p=pathlib.Path(r'$QUEUE_PATH'); data=json.load(open(p,'r',encoding='utf-8')) if p.exists() else []; active=[j for j in data if j.get('status') in {'queued','running'}]; print(len(active)); sys.exit(0 if active else 1)"
}

all_gpus_idle() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v threshold="$IDLE_MB" '$1>=threshold{exit 1} END{exit 0}'
}

scene_workers_active() {
  pgrep -af "run_scene_agent_monitor.py --worker-mode" >/dev/null 2>&1
}

render_tuner_active() {
  pgrep -af "run_render_feedback_tuner.py" >/dev/null 2>&1
}

scene_pairs_pending() {
  "$PYTHON_BIN" - <<PY
import json
import pathlib
import re
import sys

root = pathlib.Path(r"$SOURCE_ROOT")
registry_path = root.parent / "asset_registry.json"
blocked = {"deprecated_pending_replacement", "manual_reprompt_required", "deprecated", "replacement_failed"}
terminal = {"kept", "max_rounds_reached", "deprecated"}
registry = {}
if registry_path.exists():
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

def latest_status(pair_dir: pathlib.Path):
    decision_files = sorted((pair_dir / "decisions").glob("round*_decision.json"))
    if decision_files:
        data = json.loads(decision_files[-1].read_text(encoding="utf-8"))
        return data.get("status")
    best_round = -1
    for agent_path in pair_dir.glob("agent_round*.json"):
        m = re.fullmatch(r"agent_round(\d+)\.json", agent_path.name)
        if m:
            best_round = max(best_round, int(m.group(1)))
    if best_round >= int(r"$MAX_SCENE_ROUNDS"):
        return "max_rounds_reached"
    return None

pending = 0
for pair_dir in root.rglob("obj_*_yaw*"):
    if not pair_dir.is_dir():
        continue
    m = re.fullmatch(r"(obj_\d+)_yaw(\d+)", pair_dir.name)
    if not m:
        continue
    obj_id = m.group(1)
    if (registry.get(obj_id) or {}).get("status") in blocked:
        continue
    status = latest_status(pair_dir)
    if status not in terminal:
        pending += 1

print(pending)
sys.exit(0 if pending else 1)
PY
}

DONE_MARKER="$EXPORT_DIR/_export_done.json"

while true; do
  echo "[export-watch] $(date '+%F %T') tick"

  if scene_workers_active; then
    echo "[export-watch] scene-workers-active"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if scene_pairs_pending; then
    echo "[export-watch] scene-pairs-pending"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if queue_has_active_jobs; then
    echo "[export-watch] queue-not-empty"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if render_tuner_active && [[ ! -f "$TUNER_DONE_MARKER" ]]; then
    echo "[export-watch] render-tuner-active"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if ! all_gpus_idle; then
    echo "[export-watch] gpus-busy"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if [[ -f "$DONE_MARKER" ]]; then
    echo "[export-watch] already-done"
    exit 0
  fi

  mkdir -p "$EXPORT_DIR"
  "$PYTHON_BIN" scripts/export_scene_multiview_from_pair_evolution.py \
    --source-root "$SOURCE_ROOT" \
    --output-dir "$EXPORT_DIR" \
    --gpus 0,1,2 \
    --blender "$BLENDER_BIN" \
    --meshes-dir pipeline/data/meshes \
    --scene-template "$SCENE_TEMPLATE_PATH" \
    --azimuths "$AZIMUTHS_CSV" \
    --elevations "$ELEVATIONS_CSV"

  python3 - <<PY
import json
import pathlib
import time
p = pathlib.Path(r"$DONE_MARKER")
p.write_text(json.dumps({"done_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  echo "[export-watch] export-finished"
  exit 0
done
