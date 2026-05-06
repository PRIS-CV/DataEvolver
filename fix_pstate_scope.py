#!/usr/bin/env python3
"""Fix _pstate scope: move _update_obj_state from evolve_object to run_evolution"""
path = "/aaaidata/zhangqisong/data_build/run_evolution_loop.py"
with open(path) as f:
    content = f.read()

# 1. Remove the ineffective call inside evolve_object
old_inside = '''    # v5: Update PIPELINE_STATE.json
    if "_pstate" in dir():
        _update_obj_state(_pstate, obj_id, result)
    _fs = best_score if best_score is not None else 0.0'''
new_inside = '''    _fs = best_score if best_score is not None else 0.0'''
content = content.replace(old_inside, new_inside, 1)

# 2. Add _update call in run_evolution after all_results[obj_id] = result
old_run = '''            all_results[obj_id] = result
        except Exception as e:'''
new_run = '''            all_results[obj_id] = result
            _update_obj_state(_pstate, obj_id, result)
        except Exception as e:'''
content = content.replace(old_run, new_run, 1)

# 3. Also handle the error case
old_err = '''            all_results[obj_id] = {"obj_id": obj_id, "error": str(e)}'''
new_err = '''            all_results[obj_id] = {"obj_id": obj_id, "error": str(e)}
            _update_obj_state(_pstate, obj_id, {"final_hybrid": 0, "exit_reason": "error", "accepted": False, "probes_run": 0})'''
content = content.replace(old_err, new_err, 1)

with open(path, "w") as f:
    f.write(content)
print("OK: Moved _update_obj_state to run_evolution scope")
