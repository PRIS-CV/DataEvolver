#!/usr/bin/env python3
"""Fix patch 3b: add _update_obj_state after result dump in run_evolution_loop.py"""
import os

path = "/aaaidata/zhangqisong/data_build/run_evolution_loop.py"
with open(path) as f:
    content = f.read()

old = '''    result_path = os.path.join(obj_dir, "evolution_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    _fs = best_score if best_score is not None else 0.0'''

new = '''    result_path = os.path.join(obj_dir, "evolution_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    # v5: Update PIPELINE_STATE.json
    if "_pstate" in dir():
        _update_obj_state(_pstate, obj_id, result)
    _fs = best_score if best_score is not None else 0.0'''

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("OK: Added _update_obj_state after result dump")
else:
    print("ERROR: pattern not found")
    # Show nearby content for debugging
    idx = content.find("evolution_result.json")
    if idx >= 0:
        print("Context:", content[idx-50:idx+200])
