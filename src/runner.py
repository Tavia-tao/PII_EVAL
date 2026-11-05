"""异步编排：完整性校验、限流/重试、断点续跑、四元组缓存、双轨聚合与报告。"""

import argparse
import asyncio
import copy
import json
import os
import re
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from src.adapter.factory import build_adapter
from src.cache import PredictionCache
from src.config import load_config
from src.dataset import dataset_health, load_dataset
from src.evaluation import aggregate, evaluate_sample
from src.io import digest, read_json, write_json, write_jsonl
from src.metrics.date_disambig import DateResolver, build_date_judge
from src.observability import Observer
from src.schema import STAGES, Prediction, Stage
from src.stats import bootstrap_ci


def code_version():
    root = Path(__file__).resolve().parents[1]
    files = sorted(p for p in (root / "src").rglob("*") if p.suffix in {".py", ".html"} and p.is_file())
    source_hash = digest({str(p.relative_to(root)): digest(p.read_bytes()) for p in files})
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "uncommitted",
        "source_hash": source_hash,
    }


async def run_evaluation(
    cfg,
    output_dir="reports/work/latest",
    splits=None,
    sample_id=None,
    layouts=None,
    entity_types=None,
    allow_holdout=False,
    adapter=None,
    overrides=None,
    signoffs=None,
):
    from src.report import save_report

    cfg = copy.deepcopy(cfg)
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples, manifest = load_dataset(cfg["dataset"]["root"])
    splits = splits or ["gate"]
    if "holdout" in splits and not allow_holdout:
        raise ValueError("holdout需要显式 --allow-holdout，仅用于最终验收，不得调参")
    if sample_id:
        selected = [s for s in all_samples if s.sample_id == sample_id]
        if selected and selected[0].split == "holdout" and not allow_holdout:
            raise ValueError("单样本复现holdout也需要显式授权开关")
    else:
        selected = [s for s in all_samples if s.split in splits]
    if layouts:
        selected = [s for s in selected if s.layout in layouts]
    if entity_types:
        selected = [s for s in selected if any(e.type in entity_types for e in s.gold.spans)]
    if not selected:
        raise ValueError("没有命中样本")
    if any(s.meta.rule_version != cfg["rules"]["version"] for s in selected):
        raise ValueError("样本标注规则与评测配置不一致，需复核后创建新数据版本")
    versions = code_version()
    model_cfg = cfg["models"][cfg["model"]]
    sample_hash = digest(sorted(s.sample_id for s in selected))
    config_hash = digest(cfg)
    identity = {
        "dataset_hash": manifest["content_hash"],
        "rule_version": cfg["rules"]["version"],
        "metric_version": digest(cfg["rules"]),
        "sample_set_hash": sample_hash,
        "model_alias": cfg["model"],
        "model_version": model_cfg["version"],
        "prompt_version": model_cfg["prompt_version"],
        "config_hash": config_hash,
        "code_version": versions,
        "gate_policy_hash": digest(cfg["gate"]),
    }
    # 输出目录的断点必须绑定全部版本，不能续接来自旧版本的预测。
    run_state = output_dir / "run_state.json"
    previous = read_json(run_state) if run_state.exists() and cfg["execution"]["resume"] else None
    identity_hash = digest(identity)
    if previous and previous["identity_hash"] != identity_hash:
        raise ValueError("输出目录已有不同版本的运行，请换目录；禁止混合断点")
    run_id = (
        previous["run_id"]
        if previous
        else datetime.now(UTC).strftime("run_%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:8]
    )
    run_tag = f"{model_cfg['version']}__{cfg['rules']['version']}__{manifest['version']}__{run_id}"
    write_json(run_state, {"run_id": run_id, "identity_hash": identity_hash, "status": "running"})
    if any(s.split == "holdout" for s in selected):
        write_json(
            output_dir / "holdout_access.json",
            {
                "at": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "purpose": "explicit_final_evaluation",
                "samples": sample_hash,
                "actor": os.getenv("USERNAME") or os.getenv("USER", "unknown"),
            },
        )
    observer = Observer(cfg["langfuse"], run_id, identity)
    sut = adapter or build_adapter(cfg, output_dir / "artifacts", observer)
    cache = PredictionCache(cfg["execution"]["cache_dir"], cfg["execution"]["cache_ttl_days"])
    semaphore = asyncio.Semaphore(cfg["execution"]["concurrency"])
    rate_lock = asyncio.Lock()
    next_request = [0.0]
    results, traces, predictions = {}, {}, {}
    date_judge = None
    if cfg["judge"]["enabled"]:
        if not cfg["judge"]["model"]:
            raise ValueError("启用judge必须配置具体模型与凭据")
        metric = build_date_judge(cfg["judge"]["model"])

        async def date_judge(context, value):
            from deepeval.test_case import LLMTestCase

            score = await metric.a_measure(
                LLMTestCase(input=context, actual_output=f"Date {value} is personal PII")
            )
            return {
                "label": "pii" if score >= 0.8 else "not_pii" if score <= 0.2 else "uncertain",
                "confidence": max(score, 1 - score),
            }

    resolver = DateResolver(date_judge, len(selected), cfg["judge"]["max_ratio"], cfg["judge"]["threshold"])
    calibration_path = cfg["judge"].get("calibration_path")
    if calibration_path and Path(calibration_path).exists():
        calibration = read_json(calibration_path)
        if calibration.get("agreement") is None or calibration["agreement"] < 0.8:
            resolver.enabled = False
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    async def process(sample):
        async with semaphore:
            source = Path(cfg["dataset"]["root"]) / sample.input.file_path if sample.input.file_path else None
            file_hash = digest(source.read_bytes()) if source and source.is_file() else None
            parts = {
                "sample_hash": digest(sample.input.model_dump()),
                "file_hash": file_hash,
                "model_version": digest(model_cfg),
                "rule_version": digest(cfg["rules"]),
                "code_version": versions["source_hash"],
                "dataset_version": manifest["version"],
                "config_hash": config_hash,
                "overrides": digest(overrides) if overrides else None,
            }
            key = cache.key(parts)
            pred = cache.get(key) if cfg["execution"]["cache_enabled"] else None
            checkpoint = checkpoints / (sample.sample_id + ".json")
            if pred is None and previous and checkpoint.exists():
                try:
                    record = read_json(checkpoint)
                    if record["key"] == key:
                        candidate = Prediction.model_validate(record["prediction"])
                        if not candidate.error and all(
                            not s.artifact or Path(s.artifact).is_file() for s in candidate.stages.values()
                        ):
                            pred = candidate
                            pred.cached = True
                except (ValueError, KeyError):
                    pass
            with observer.sample_trace(sample) as trace_id:
                if pred is None:
                    for attempt in range(cfg["execution"]["retry_max"] + 1):
                        async with rate_lock:
                            delay = next_request[0] - time.monotonic()
                            if delay > 0:
                                await asyncio.sleep(delay)
                            next_request[0] = time.monotonic() + cfg["execution"]["request_interval_seconds"]
                        try:
                            pred = await asyncio.wait_for(
                                sut.predict(sample, overrides), timeout=cfg["execution"]["timeout_seconds"]
                            )
                        except Exception as e:
                            pred = Prediction(
                                sample_id=sample.sample_id,
                                model_version=model_cfg["version"],
                                error=type(e).__name__,
                                stages={s: Stage(ok=False, error="call_failed") for s in STAGES},
                            )
                        if not pred.error or attempt == cfg["execution"]["retry_max"]:
                            break
                        await asyncio.sleep(cfg["execution"]["retry_backoff_seconds"] * 2**attempt)
                    if cfg["execution"]["cache_enabled"]:
                        cache.put(key, pred)
                pred.trace_id = trace_id
                date_decisions = []
                for line in pred.coordinate_text.split("\n"):
                    for match in re.finditer(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", line):
                        decision = await resolver.resolve(line, match.group(0))
                        decision["candidate"] = match.group(0)
                        decision["audit_selected"] = (
                            decision["source"] == "judge"
                            and int(digest([sample.sample_id, match.start()])[:8], 16) % 20 == 0
                        )
                        date_decisions.append(decision)
                pred.needs_review = pred.needs_review or any(d["needs_human_review"] for d in date_decisions)
                write_json(checkpoint, {"key": key, "prediction": pred.model_dump()})
                row = evaluate_sample(
                    sample,
                    pred,
                    high_risk=cfg["rules"]["high_risk_types"],
                    weights=cfg["rules"]["weights"],
                    iou_threshold=cfg["rules"]["iou_threshold"],
                )
                row["date_decisions"] = date_decisions
                # DeepEval原生Metric同步/异步接口与报告复用同一匹配算法。
                os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
                from deepeval.test_case import LLMTestCase

                from src.metrics import BoundaryMetric, HighRiskRecallMetric, PIIF1Metric

                tc = LLMTestCase(
                    input=sample.sample_id,
                    actual_output=json.dumps({"spans": row["pred_spans"]}),
                    expected_output=json.dumps({"spans": row["gold_spans"]}),
                )
                metrics = [
                    PIIF1Metric(),
                    HighRiskRecallMetric(cfg["rules"]["high_risk_types"]),
                    BoundaryMetric(),
                ]
                row["deepeval"] = {
                    m.__name__: {"score": await m.a_measure(tc), "passed": m.is_successful()} for m in metrics
                }
                observer.report_scores(trace_id, row)
                results[sample.sample_id], predictions[sample.sample_id] = row, pred
                traces[sample.sample_id] = {
                    "trace_id": trace_id,
                    "run_id": run_id,
                    "metadata": {
                        **identity,
                        "sample_id": sample.sample_id,
                        "layout": sample.layout,
                        "difficulty": sample.difficulty,
                        "doc_type": sample.doc_type,
                        "route": pred.route,
                        "route_reason": pred.route_reason,
                        "cached": pred.cached,
                    },
                    "stages": row["stages"],
                    "scores": {
                        name: row.get(name)
                        for name in [
                            "f1",
                            "high_risk_recall",
                            "false_delete",
                            "false_negative",
                            "boundary_acc",
                            "risk_score",
                        ]
                    },
                }
                return {
                    "sample_id": sample.sample_id,
                    "trace_id": trace_id,
                    "f1": row["f1"],
                    "risk_score": row["risk_score"],
                    "execution_error": row["execution_error"],
                }

    try:
        if observer.client:
            items = await asyncio.to_thread(observer.sync_samples, selected, manifest)
            by_id = {s.sample_id: s for s in selected}

            async def task(*, item, **kwargs):
                return await process(by_id[item.input["sample_id"]])

            experiment = await asyncio.to_thread(
                observer.client.run_experiment,
                name=run_tag,
                run_name=run_tag,
                data=items,
                task=task,
                max_concurrency=cfg["execution"]["concurrency"],
                metadata={"version_snapshot": digest(identity)},
            )
            experiment_url = experiment.experiment_url
            experiment_id = experiment.experiment_id
        else:
            experiment_url = None
            experiment_id = None
            shard_size = cfg["execution"]["shard_size"]
            for offset in range(0, len(selected), shard_size):
                await asyncio.gather(*(process(s) for s in selected[offset : offset + shard_size]))
        if len(results) != len(selected):
            raise RuntimeError("评测样本缺失，不能生成通过报告")
        rows = [results[s.sample_id] for s in selected]
        groups = defaultdict(list)
        for row in rows:
            groups[row["track"]].append(row)
        tracks = {
            key: aggregate(value, high_risk=cfg["rules"]["high_risk_types"]) for key, value in groups.items()
        }
        operational = aggregate(rows, high_risk=cfg["rules"]["high_risk_types"])
        # 不将混合双轨F1作为报告/选型数字。
        for key in ["f1", "precision", "recall", "macro_f1", "weighted_f1"]:
            operational.pop(key, None)
        by_layout, by_difficulty, by_language, by_stage = {}, {}, {}, {}
        for field, target in [("layout", by_layout), ("difficulty", by_difficulty), ("lang", by_language)]:
            for value in sorted({r[field] for r in rows}):
                target[value] = aggregate([r for r in rows if r[field] == value])
        for name in STAGES:
            values = [r["stages"].get(name, {"ok": False, "elapsed_ms": 0}) for r in rows]
            by_stage[name] = {
                "ok_rate": sum(v["ok"] for v in values) / len(values),
                "mean_elapsed_ms": sum(v["elapsed_ms"] for v in values) / len(values),
            }
            metric_names = {k for v in values for k in v.get("metrics", {})}
            by_stage[name]["metrics"] = {
                k: sum(float(v["metrics"][k]) for v in values if v.get("metrics", {}).get(k) is not None)
                / sum(v.get("metrics", {}).get(k) is not None for v in values)
                for k in metric_names
                if any(v.get("metrics", {}).get(k) is not None for v in values)
            }
        gate_ids = {s.sample_id for s in all_samples if s.split == "gate"}
        kappa = [
            s.meta.kappa for s in selected if s.meta.quality_source == "human" and s.meta.kappa is not None
        ]
        meta = {
            **identity,
            "run_id": run_id,
            "run_tag": run_tag,
            "dataset_version": manifest["version"],
            "created_at": datetime.now(UTC).isoformat(),
            "n": len(rows),
            "splits": splits,
            "synthetic_only": all(s.source == "synthetic" for s in selected),
            "incremental": bool(sample_id or layouts or entity_types),
            "complete_gate": gate_ids <= {s.sample_id for s in selected},
            "cache_hits": sum(r["cached"] for r in rows),
            "wall_seconds": time.perf_counter() - started,
            "human_kappa": min(kappa) if len(kappa) == len(selected) else None,
            "signoffs": signoffs or {},
            "experiment_url": experiment_url,
            "judge_calls": resolver.calls,
            "judge_ratio": resolver.calls / len(selected),
            "latency_note": "真实计时；缓存命中保留原始推理耗时；合成规则基线不代表真实模型时延",
        }
        report = {
            "schema_version": 1,
            "meta": meta,
            "tracks": tracks,
            "operational": operational,
            "by_layout": by_layout,
            "by_difficulty": by_difficulty,
            "by_language": by_language,
            "by_stage": by_stage,
            "health": dataset_health(selected),
            "ci": {
                track: bootstrap_ci(
                    value, n_boot=cfg["execution"]["bootstrap"], seed=cfg["execution"]["seed"]
                )
                for track, value in groups.items()
            },
            "rows": rows,
        }
        observer.report_run_scores(report, experiment_id)
        write_jsonl(output_dir / "traces.jsonl", [traces[s.sample_id] for s in selected])
        write_jsonl(
            output_dir / "predictions.jsonl", [predictions[s.sample_id].model_dump() for s in selected]
        )
        save_report(report, output_dir, selected, predictions, Path(cfg["dataset"]["root"]))
        if cfg["storage"]["enabled"]:
            from src.storage import ResultStore

            await asyncio.to_thread(
                ResultStore(os.environ[cfg["storage"]["database_url_env"]]).save_run, run_id, report
            )
        if cfg["storage"]["clickhouse_enabled"]:
            from src.storage import ClickHouseStore

            await asyncio.to_thread(ClickHouseStore().save_run, run_id, report)
        write_json(run_state, {"run_id": run_id, "identity_hash": identity_hash, "status": "completed"})
        return report
    finally:
        await asyncio.to_thread(observer.close)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--model", default=None)
    p.add_argument("--split", default="gate")
    p.add_argument("--out", default="reports/work/latest")
    p.add_argument("--sample")
    p.add_argument("--layout", action="append")
    p.add_argument("--entity-type", action="append")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--allow-holdout", action="store_true")
    p.add_argument("--signoffs")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.concurrency:
        cfg["execution"]["concurrency"] = args.concurrency
    if args.no_cache:
        cfg["execution"]["cache_enabled"] = False
        cfg["execution"]["resume"] = False
    report = asyncio.run(
        run_evaluation(
            cfg,
            args.out,
            args.split.split(","),
            args.sample,
            args.layout,
            args.entity_type,
            args.allow_holdout,
            signoffs=read_json(args.signoffs) if args.signoffs else None,
        )
    )
    print(
        json.dumps(
            {
                "n": report["meta"]["n"],
                "tracks": {k: v["f1"] for k, v in report["tracks"].items()},
                "report": str(Path(args.out).resolve() / "report.html"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
