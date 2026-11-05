import argparse
import math

from src.config import load_config
from src.io import digest, read_json, write_json
from src.schema import HIGH_RISK


def _value(report, name):
    if name == "overall_f1":
        return report.get("tracks", {}).get("standard_text", {}).get("f1")
    if name == "complex_f1":
        return report.get("tracks", {}).get("complex_file", {}).get("f1")
    return report.get("operational", {}).get(name)


def judge_gate(report, baseline, cfg=None, level="L2"):
    if level not in {"L1", "L2", "L3"}:
        raise ValueError("未知门禁级别")
    config = cfg or load_config()
    gate = config["gate"]
    fails, warns = [], []

    def fail(name, detail, **kwargs):
        fails.append({"metric": name, "reason": detail, **kwargs})

    for name in ["dataset_hash", "rule_version", "metric_version", "sample_set_hash"]:
        if not report.get("meta", {}).get(name) or report["meta"][name] != baseline.get("meta", {}).get(name):
            fail("comparability", f"{name}缺失或与基线不一致")
    meta, op = report.get("meta", {}), report.get("operational", {})
    for run in [report, baseline]:
        policy_hash = run.get("meta", {}).get("gate_policy_hash")
        if policy_hash is not None and policy_hash != digest(gate):
            fail("gate_policy", "运行/基线门禁策略与当前策略不一致，需独立审批并重建基线")
    if meta.get("incremental") or not meta.get("complete_gate"):
        fail("coverage", "正式门禁必须覆盖完整gate，增量结果只能作为诊断")
    if op.get("execution_errors", 1) != 0:
        fail("execution_errors", "存在失败或缺失阶段")
    if op.get("negative_coverage", 0) < 1:
        fail("negative_coverage", "部分文档缺少非PII候选分母")
    for t in config.get("rules", {}).get("high_risk_types", HIGH_RISK):
        v = op.get("by_type", {}).get(t, {})
        recall = v.get("recall")
        if v.get("n", 0) < gate["min_per_high_risk_type"]:
            fail("high_risk_coverage", f"{t} 样本不足", entity_type=t, n=v.get("n", 0))
        elif (
            recall is None
            or not math.isfinite(recall)
            or recall < gate["thresholds"]["high_risk_recall"]["hard"]
        ):
            fail(
                "high_risk_per_type",
                f"{t} 召回低于硬阈值",
                entity_type=t,
                value=recall,
                threshold=gate["thresholds"]["high_risk_recall"]["hard"],
            )
    for name, spec in gate["thresholds"].items():
        value = _value(report, name)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            fail(name, "缺失/非有限指标，不允许默认通过")
            continue
        hard, soft, direction = spec["hard"], spec["soft"], spec["dir"]
        if direction.startswith("rel_"):
            base = _value(baseline, name)
            if base is None or not math.isfinite(base):
                fail(name, "基线指标缺失")
                continue
            if direction == "rel_min":
                hard, soft, direction = base + hard, base + soft, "min"
            elif direction == "rel_max":
                hard, soft, direction = base * hard, base * soft, "max"
            else:
                raise ValueError("未知相对阈值方向")
        if direction not in {"min", "max"}:
            raise ValueError("阈值方向错误")
        broken = value < hard if direction == "min" else value > hard
        warning = value < soft if direction == "min" else value > soft
        record = {
            "metric": name,
            "value": value,
            "threshold": hard if broken else soft,
            "reason": "硬阈值未通过" if broken else "软阈值未通过",
        }
        if broken:
            fails.append(record)
        elif warning:
            warns.append(record)
    human_kappa = meta.get("human_kappa")
    if human_kappa is not None:
        if human_kappa < 0.8:
            fail("annotation_kappa", "人工标注Kappa<0.80", value=human_kappa)
        elif human_kappa < 0.85:
            warns.append({"metric": "annotation_kappa", "value": human_kappa, "reason": "Kappa<0.85"})
    if gate.get("require_real_quality") and (meta.get("synthetic_only") or human_kappa is None):
        fail("data_qualification", "正式发布要求真实/人工核验的数据与一致性证据")
    if level == "L3" and not meta.get("signoffs", {}).get("compliance"):
        fail("compliance_signoff", "L3需要合规签字，工具不得自动代签")
    verdict = "BLOCK" if fails or (level == "L1" and warns) else "WARN" if warns else "PASS"
    failed_types = {f.get("entity_type") for f in fails if f.get("entity_type")}
    failure_samples = []
    issues = fails + warns

    def affected(row, issue):
        metric = issue["metric"]
        if metric == "latency_p95":
            return row.get("latency_ms", 0) / 1000 > issue.get("threshold", float("inf"))
        if metric == "file_openable":
            return not row.get("file_openable", True)
        if metric == "execution_errors":
            return row.get("execution_error", False)
        if metric == "false_delete":
            return row.get("false_delete_count", 0) > 0
        if metric == "risk_score":
            return row.get("risk_num", 0) > 0
        if metric in {"high_risk_recall", "high_risk_per_type"}:
            types = [issue["entity_type"]] if issue.get("entity_type") else failed_types or HIGH_RISK
            return any(row.get("by_type", {}).get(t, {}).get("fn", 0) > 0 for t in types)
        if metric in {"overall_f1", "complex_f1"}:
            track = "standard_text" if metric == "overall_f1" else "complex_file"
            return row.get("track") == track and (row.get("fp", 0) > 0 or row.get("fn", 0) > 0)
        return False

    for issue in issues:
        issue["sample_ids"] = [r["sample_id"] for r in report.get("rows", []) if affected(r, issue)]
    for row in report.get("rows", []):
        metrics = [issue["metric"] for issue in issues if affected(row, issue)]
        if metrics:
            failure_samples.append(
                {
                    "sample_id": row["sample_id"],
                    "metrics": sorted(set(metrics)),
                    "stages": row.get("error_stages", []),
                    "trace_id": row.get("trace_id"),
                    "reproduce": f"pii-eval run --sample {row['sample_id']} --model "
                    f"{meta.get('model_alias', 'demo-student')} --no-cache",
                }
            )
    return {
        "verdict": verdict,
        "level": level,
        "fails": fails,
        "warns": warns,
        "failure_samples": failure_samples,
        "qualification": "synthetic_engineering_only"
        if meta.get("synthetic_only")
        else "human_validation_required",
        "release_ready": bool(
            verdict == "PASS"
            and not meta.get("synthetic_only")
            and human_kappa is not None
            and human_kappa >= 0.85
            and meta.get("signoffs", {}).get("compliance")
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--config")
    p.add_argument("--level", default="L2", choices=["L1", "L2", "L3"])
    p.add_argument("--out", default="reports/gate.json")
    args = p.parse_args()
    result = judge_gate(
        read_json(args.report), read_json(args.baseline), load_config(args.config), args.level
    )
    write_json(args.out, result)
    print(f"[{result['verdict']}] {len(result['fails'])} hard failures; {len(result['warns'])} warnings")
    raise SystemExit(2 if result["verdict"] == "BLOCK" else 0)


if __name__ == "__main__":
    main()
