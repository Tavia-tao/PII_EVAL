import json
import os
from pathlib import Path

from src.io import write_json, write_jsonl
from src.stats import paired_compare, paired_rate_change


def save_report(report, output_dir, samples=None, predictions=None, data_root=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", report)
    write_jsonl(
        output_dir / "failures.jsonl",
        [r for r in report["rows"] if r["fp"] or r["fn"] or r["execution_error"]],
    )
    # 独立HTML自包含；内容用textContent显示，禁止将原始文本当HTML执行。
    data = {"report": report, "samples": {}}
    for s in samples or []:
        source = (Path(data_root) / s.input.file_path).resolve() if s.input.file_path else None
        preview = source.with_suffix(source.suffix + ".preview.png") if source else None
        pred = predictions.get(s.sample_id) if predictions else None
        data["samples"][s.sample_id] = {
            "text": s.gold.text,
            "source": os.path.relpath(source, output_dir).replace("\\", "/") if source else None,
            "preview": os.path.relpath(preview, output_dir).replace("\\", "/")
            if preview and preview.exists()
            else None,
            "export": os.path.relpath(pred.stages["S6_export"].artifact, output_dir).replace("\\", "/")
            if pred and pred.stages.get("S6_export") and pred.stages["S6_export"].artifact
            else None,
        }
    payload = (
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    (output_dir / "report.html").write_text(template.replace("__REPORT_DATA__", payload), encoding="utf-8")
    lines = [
        "# PII 评测报告",
        "",
        f"运行：{report['meta']['run_tag']}",
        "",
        "数据范围："
        + (
            "合成工程验证，不能外推真实模型效果。"
            if report["meta"]["synthetic_only"]
            else "以当前样本及标注规则为限。"
        ),
        "",
        "| 轨道 | 样本数 | Exact micro F1 | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for track, metrics in report["tracks"].items():
        ci = report["ci"][track]
        lines.append(f"| {track} | {metrics['n']} | {metrics['f1']} | [{ci['lo']:.4f}, {ci['hi']:.4f}] |")
    lines.extend(
        [
            "",
            "详见 report.html：类别/版式/难度筛选、原文字符区间、六阶段产物与最小复现命令。",
            "",
            f"执行失败：{report['operational']['execution_errors']}；缓存命中：{report['meta']['cache_hits']}。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_reports(a, b, n_boot=1000):
    for key in ["dataset_hash", "rule_version", "metric_version", "sample_set_hash"]:
        if a["meta"].get(key) != b["meta"].get(key):
            raise ValueError(f"不可比: {key}不一致")
    tracks = {}
    for track in a["tracks"].keys() & b["tracks"].keys():
        aa, bb = [r for r in a["rows"] if r["track"] == track], [r for r in b["rows"] if r["track"] == track]
        tracks[track] = paired_compare(aa, bb, n_boot)
    candidates = []
    for label, report in [("A", a), ("B", b)]:
        op = report["operational"]
        if (
            (op.get("high_risk_min") or 0) >= 0.98
            and op.get("false_delete") is not None
            and op["false_delete"] <= 0.01
        ):
            candidates.append(
                {
                    "model": label,
                    "latency_p95": op["latency_p95"],
                    "token_cost": op["token_cost"] if op["cost_measured"] else None,
                }
            )
    date_a = sum(r["hard_negative_fp"].get("contract_date", 0) for r in a["rows"])
    date_b = sum(r["hard_negative_fp"].get("contract_date", 0) for r in b["rows"])
    date_ci = paired_rate_change(
        [
            {
                "sample_id": r["sample_id"],
                "layout": r["layout"],
                "count": r["hard_negative_fp"].get("contract_date", 0),
            }
            for r in a["rows"]
        ],
        [
            {
                "sample_id": r["sample_id"],
                "layout": r["layout"],
                "count": r["hard_negative_fp"].get("contract_date", 0),
            }
            for r in b["rows"]
        ],
        n_boot,
    )
    return {
        "models": {"A": a["meta"]["model_version"], "B": b["meta"]["model_version"]},
        "tracks": tracks,
        "date_fp": {
            "A": date_a,
            "B": date_b,
            "relative_reduction_B_vs_A": (date_a - date_b) / date_a if date_a else None,
            "note": "不得套用简历86%；提升结论需足够失败样本和配对区间",
            "paired_ci": date_ci,
        },
        "eligible_candidates": candidates,
        "recommendation": "无候选同时满足高风险召回及误删约束，需修复/分层复核"
        if not candidates
        else "候选已通过质量约束；结合实测时延、可用成本证据及路由子集比较选择。",
        "cost_evidence_complete": all(r["operational"]["cost_measured"] for r in [a, b]),
        "synthetic_only": a["meta"]["synthetic_only"] or b["meta"]["synthetic_only"],
    }
