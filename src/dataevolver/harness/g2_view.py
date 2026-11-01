"""Build an answer-me-with-html draft directly from G2 execution records.

No model calls or hand-entered results. Render the fact-reviewed draft using
the installed answer-me-with-html CLI; media copies belong to reports/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactStore, file_sha256
from dataevolver.dataset.g2_pipeline import load_pair_manifest


def build_draft(manifest, roots, output, media_dir):
    manifest, output, media_dir = Path(manifest).resolve(), Path(output).resolve(), Path(media_dir).resolve()
    store = ArtifactStore(roots)
    records = json.loads(manifest.read_text())
    loaded = {r["task_id"]: r for r in load_pair_manifest(manifest, store, include_rejected=True)}
    media_dir.mkdir(parents=True, exist_ok=True)
    summary, usage = records["summary"], records["usage"]
    lines = ["---", "template: doc", "theme: paper", "lang: zh", "title: G2 任务查看", "---", "## A 执行记录", "",
             "本页直接读取图片对清单。自动检查结果仍需抽样复核。", "",
             "| 项目 | 记录 |", "| --- | --- |",
             f"| 任务总数 | {summary['tasks']} |", f"| 产出图片对 | {summary['produced_pairs']} |",
             f"| 自动检查通过 | {summary['qualified_pairs']} |", f"| 失败任务 | {summary['failed_tasks']} |",
             f"| 模型调用（含失败） | {usage['model_calls']} |", f"| 已观测耗时（秒） | {usage['observed_wall_seconds']:.3f} |",
             f"| 货币成本 | {usage['monetary_cost'] if usage['monetary_cost'] is not None else '未测量'} |",
             f"| 评价标记 | {records['calibration_status']} |", "",
             f"清单 SHA-256：`{file_sha256(manifest)}`。", ""]
    for index, row in enumerate(records["pairs"]):
        lines += [f"## {chr(66+index)} {row['task_id']}", "", f"状态：`{row['status']}`。用途：`{row['purpose']}`。", "",
                  f"原始任务：{row['task']['original_instruction']}", ""]
        if row["task_id"] in loaded:
            pair = loaded[row["task_id"]]
            for field, label in (("source_path", "编辑前"), ("output_path", "编辑后")):
                src = Path(pair[field]); dst = media_dir/f"{row['task_id']}-{field}{src.suffix}"
                shutil.copy2(src, dst)
                lines += [f"![{label}：{row['task_id']}]({dst})", ""]
        else:
            lines += ["此次任务没有可独立加载的编辑结果。", ""]
        lines += ["| 配方字段 | 实际值 |", "| --- | --- |"]
        lines += [f"| {key} | {str(value).replace('|', '/')} |" for key, value in row["recipe"].items()]
        for model in row.get("attempts", row["model_requests"]):
            request = model["request"]
            if request["role"] == "visual.edit":
                parameters = json.loads(request["messages"][0]["content"])
                lines += ["", f"实际请求：`{request['request_id']}`。", "", "```json title=实际编辑参数",
                          json.dumps(parameters, ensure_ascii=False, indent=2), "```", ""]
        if row["preservation"]:
            pixels = row["preservation"]
            lines += [f"目标区域 MAE：{pixels['target_mae']:.6f}；其余区域 MAE：{pixels['outside_mae']:.6f}。", ""]
        if row["review"]:
            lines += ["| 原始约束 | 判断 | 观测证据 |", "| --- | --- | --- |"]
            lines += [f"| {name} | {value['status']} | {value['evidence'].replace('|', '/')} |" for name, value in row["review"]["checks"].items()]
        if row["error"]:
            lines += ["", "```json title=失败记录", json.dumps(row["error"], ensure_ascii=False, indent=2), "```"]
        attempts = [a for a in usage["attempts"] if f"-{row['task_id']}-" in a["request_id"]]
        lines += ["", "| 调用 | 状态 | 耗时（秒） |", "| --- | --- | --- |"]
        lines += [f"| {a['request_id']} | {a['state']} | {a['wall_seconds'] if a['wall_seconds'] is not None else '未知'} |" for a in attempts]
        lines += [""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return {"draft": str(output), "source_manifest": str(manifest), "source_sha256": file_sha256(manifest), "panels": 1+len(records["pairs"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stores-json", required=True, help="JSON object mapping store IDs to absolute roots")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--media-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_draft(args.manifest, json.loads(Path(args.stores_json).read_text()), args.output_md, args.media_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
