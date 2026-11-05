"""无服务命令行入口；只有显式配置并执行时才连接外部依赖。"""

import argparse
import asyncio
import json
from pathlib import Path

from src.config import load_config
from src.io import read_json, read_jsonl, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pii-eval", description="PII 双轨评测 / 六阶段归因 / 门禁 / 反馈")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="生成完全虚构的测试数据及实际文件")
    g.add_argument("--out", default="data/generated/final")
    for option, default in [("standard", 14000), ("complex", 2000), ("gate", 1500), ("seed", 42)]:
        g.add_argument("--" + option, type=int, default=default)
    v = sub.add_parser("validate", help="校验schema、文档/父文档/文本隔离及文件摘要")
    v.add_argument("--data", default="data/generated/final")
    v.add_argument("--register-version", help="为准备好的两个JSONL建立新版本manifest，不覆盖旧manifest")
    for name in ["run", "attribute"]:
        p = sub.add_parser(name, help="评测" if name == "run" else "S2/S3金标替代实验")
        p.add_argument("--config")
        p.add_argument("--data")
        p.add_argument("--model")
        p.add_argument("--split", default="gate")
        p.add_argument("--out", default=f"reports/work/{name}")
        p.add_argument("--allow-holdout", action="store_true")
        p.add_argument("--bootstrap", type=int)
        if name == "run":
            p.add_argument("--sample")
            p.add_argument("--layout", action="append")
            p.add_argument("--entity-type", action="append")
            p.add_argument("--concurrency", type=int)
            p.add_argument("--no-cache", action="store_true")
            p.add_argument("--signoffs")
    gate = sub.add_parser("gate", help="门禁，BLOCK返回退出码2")
    gate.add_argument("--report", required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--config")
    gate.add_argument("--level", choices=["L1", "L2", "L3"], default="L2")
    gate.add_argument("--out", default="reports/gate.json")
    compare = sub.add_parser("compare", help="同样本配对bootstrap与约束选型")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--bootstrap", type=int, default=1000)
    compare.add_argument("--out", default="reports/comparison.json")
    verify = sub.add_parser("verify", help="三次复现、五类劣化和trace完整性验收")
    verify.add_argument("--data", default="data/generated/final")
    verify.add_argument("--config")
    verify.add_argument("--out", default="reports/work/verification")
    q = sub.add_parser("quality", help="双人候选标注一致性及质检抽样")
    q.add_argument("--left")
    q.add_argument("--right")
    q.add_argument("--data")
    q.add_argument("--out", default="reports/annotation-quality.json")
    queue = sub.add_parser("review-queue", help="报告中的错误/灰区样本入待复核池")
    queue.add_argument("--report", required=True)
    queue.add_argument("--db", default="runtime/feedback.sqlite")
    f = sub.add_parser("feedback", help="录入采纳状态、双人确认、导出回流清单")
    f.add_argument("action", choices=["list", "record", "promote", "export"])
    f.add_argument("--db", default="runtime/feedback.sqlite")
    f.add_argument("--case")
    f.add_argument("--reviewer")
    f.add_argument("--corrected", help="含spans数组的JSON文件")
    f.add_argument("--reject", action="store_true")
    f.add_argument("--target")
    f.add_argument("--out", default="reports/approved-cases.jsonl")
    drift = sub.add_parser("drift", help="读取监控快照，生成可审计告警，不启动监控服务")
    drift.add_argument("--points", required=True)
    drift.add_argument("--baseline", required=True)
    drift.add_argument("--out", default="reports/drift-alerts.json")
    health = sub.add_parser("gate-health", help="识别连续满分、建议轮换的类别")
    health.add_argument("reports", nargs="+")
    health.add_argument("--out", default="reports/gate-health.json")
    apply = sub.add_parser("apply-feedback", help="双人复核清单生成新的门禁数据版本")
    apply.add_argument("--data", required=True)
    apply.add_argument("--approved", required=True)
    apply.add_argument("--out", required=True)
    apply.add_argument("--version", required=True)
    approval = sub.add_parser("approve-change", help="记录阈值/规则/数据修正的独立审批证据")
    approval.add_argument("--kind", choices=["threshold", "rule", "data"], required=True)
    approval.add_argument("--before", required=True)
    approval.add_argument("--after", required=True)
    approval.add_argument("--approval", required=True)
    approval.add_argument("--out", required=True)
    calibration = sub.add_parser("judge-calibrate", help="用人工裁决校准judge，不足0.8自动禁用")
    calibration.add_argument("--reviews", required=True)
    calibration.add_argument("--out", default="runtime/judge-calibration.json")
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, KeyError, OSError) as e:
        parser.exit(1, f"错误: {e}\n")
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


def dispatch(a):
    if a.command == "generate":
        from src.generate import generate_dataset

        manifest = generate_dataset(a.out, a.standard, a.complex, a.gate, a.seed)
        return {k: manifest[k] for k in ["version", "n", "counts", "warning"]} | {
            "files": len(manifest["artifacts"]),
            "manifest": str(Path(a.out).resolve() / "manifest.json"),
        }
    if a.command == "validate":
        from src.dataset import dataset_health, load_dataset, register_dataset

        if a.register_version:
            register_dataset(a.data, a.register_version)
        samples, manifest = load_dataset(a.data)
        return {"version": manifest["version"], **dataset_health(samples)}
    if a.command in {"run", "attribute"}:
        cfg = load_config(a.config)
        if a.data:
            cfg["dataset"]["root"] = a.data
        if a.model:
            if a.model not in cfg["models"]:
                raise ValueError("未知模型别名，请在配置models中注册")
            cfg["model"] = a.model
        if a.bootstrap:
            cfg["execution"]["bootstrap"] = a.bootstrap
        if a.command == "attribute":
            from src.attribution import run_attribution

            return asyncio.run(run_attribution(cfg, a.out, a.split.split(","), a.allow_holdout))
        from src.runner import run_evaluation

        if a.concurrency is not None:
            if a.concurrency < 1:
                raise ValueError("concurrency必须为正")
            cfg["execution"]["concurrency"] = a.concurrency
        if a.no_cache:
            cfg["execution"].update(cache_enabled=False, resume=False)
        r = asyncio.run(
            run_evaluation(
                cfg,
                a.out,
                a.split.split(","),
                a.sample,
                a.layout,
                a.entity_type,
                a.allow_holdout,
                signoffs=read_json(a.signoffs) if a.signoffs else None,
            )
        )
        return {
            "run_id": r["meta"]["run_id"],
            "n": r["meta"]["n"],
            "tracks": {k: {"f1": v["f1"], "n": v["n"]} for k, v in r["tracks"].items()},
            "execution_errors": r["operational"]["execution_errors"],
            "report": str(Path(a.out).resolve() / "report.html"),
        }
    if a.command == "gate":
        from src.gate import judge_gate

        result = judge_gate(read_json(a.report), read_json(a.baseline), load_config(a.config), a.level)
        write_json(a.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        raise SystemExit(2 if result["verdict"] == "BLOCK" else 0)
    if a.command == "compare":
        from src.report import compare_reports

        result = compare_reports(read_json(a.a), read_json(a.b), a.bootstrap)
        write_json(a.out, result)
        return result
    if a.command == "verify":
        from src.verification import acceptance_run

        cfg = load_config(a.config)
        cfg["dataset"]["root"] = a.data
        result = asyncio.run(acceptance_run(cfg, a.out))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if a.command == "quality":
        from src.dataset import load_dataset
        from src.quality import annotation_agreement, quality_sample_plan

        if a.data:
            samples, _ = load_dataset(a.data)
            result = quality_sample_plan(samples)
        elif a.left and a.right:
            left, right = read_json(a.left), read_json(a.right)
            if left["candidate_ids"] != right["candidate_ids"]:
                raise ValueError("两份标注的候选ID/顺序不一致")
            result = annotation_agreement(left["labels"], right["labels"])
        else:
            raise ValueError("请指定--data生成质检计划，或--left和--right计算一致性")
        write_json(a.out, result)
        return result
    if a.command in {"review-queue", "feedback"}:
        from src.feedback import FeedbackStore

        store = FeedbackStore(a.db)
        if a.command == "review-queue":
            from src.schema import HIGH_RISK

            report = read_json(a.report)
            cases = []
            for row in report["rows"]:
                if row["split"] == "holdout":
                    continue
                if row["fp"] or row["fn"] or row["needs_review"]:
                    cases.append(
                        store.enqueue(
                            row["sample_id"],
                            row["pred_spans"],
                            "boundary"
                            if row["partial"]
                            else "false_negative"
                            if row["fn"]
                            else "false_positive",
                            row["track"],
                            row["layout"],
                            any(s["type"] in HIGH_RISK for s in row["fn_spans"]),
                            row["split"],
                        )
                    )
            return {"n": len(cases), "case_ids": cases}
        if a.action == "list":
            return store.bad_cases()
        if a.action == "record":
            if not (a.case and a.reviewer and a.corrected):
                raise ValueError("record需要--case、--reviewer和--corrected")
            return store.record(a.case, a.reviewer, read_json(a.corrected)["spans"], accepted=not a.reject)
        if not a.target:
            raise ValueError("需要--target指定新数据版本")
        return (
            store.promote(a.case, a.target)
            if a.action == "promote"
            else store.export_approved(a.out, a.target)
        )
    if a.command == "drift":
        from src.monitor import detect_drift

        points = list(read_jsonl(a.points)) if Path(a.points).suffix == ".jsonl" else read_json(a.points)
        baseline = read_json(a.baseline)
        result = detect_drift(points, baseline.get("operational", baseline))
        write_json(a.out, result)
        return result
    if a.command == "gate-health":
        from src.monitor import gate_health

        result = gate_health([read_json(p) for p in a.reports])
        write_json(a.out, result)
        return result
    if a.command == "apply-feedback":
        from src.dataset import apply_feedback

        m = apply_feedback(a.data, a.approved, a.out, a.version)
        return {"version": m["version"], "n": m["n"], "content_hash": m["content_hash"]}
    if a.command == "approve-change":
        from src.governance import approve_change

        result = approve_change(a.kind, read_json(a.before), read_json(a.after), read_json(a.approval))
        write_json(a.out, result)
        return result
    if a.command == "judge-calibrate":
        from src.metrics.date_disambig import DateResolver

        rows = list(read_jsonl(a.reviews))
        result = DateResolver().calibrate([r["judge_label"] == r["human_label"] for r in rows])
        write_json(a.out, result)
        return result


if __name__ == "__main__":
    main()
