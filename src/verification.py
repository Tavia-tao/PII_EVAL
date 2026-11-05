"""工程验收：保留对照组，验证特定劣化确实触发对应门禁。"""

import copy
from pathlib import Path

from src.adapter.factory import build_adapter
from src.faults import FaultSUT
from src.gate import judge_gate
from src.io import read_jsonl, write_json
from src.runner import run_evaluation
from src.schema import STAGES


async def acceptance_run(cfg, output_dir):
    output_dir = Path(output_dir)
    cfg = copy.deepcopy(cfg)
    cfg["execution"]["cache_enabled"] = False
    cfg["execution"]["resume"] = False
    reports = []
    for i in range(3):
        reports.append(await run_evaluation(cfg, output_dir / f"repeat_{i + 1}", splits=["gate"]))
    ranges = {
        track: max(r["tracks"][track]["f1"] for r in reports) - min(r["tracks"][track]["f1"] for r in reports)
        for track in reports[0]["tracks"]
    }
    traces = list(read_jsonl(output_dir / "repeat_3" / "traces.jsonl"))
    complete = all(
        set(t["stages"]) == set(STAGES)
        and all(t["metadata"].get(k) for k in ["rule_version", "model_version", "sample_id", "layout"])
        for t in traces
    )
    baseline = reports[0]
    control = judge_gate(baseline, baseline, cfg)
    specs = ["high_risk_recall", "false_delete", "risk_score", "latency_p95", "file_openable"]
    degradations = []
    for metric in specs:
        fault_cfg = copy.deepcopy(cfg)
        fault_cfg["execution"]["retry_max"] = 0
        fault_cfg["models"][cfg["model"]]["version"] += "-fault-" + metric
        folder = output_dir / ("fault_" + metric)
        sut = FaultSUT(
            build_adapter(fault_cfg, folder / "artifacts"),
            metric,
            delay=max(0.05, baseline["operational"]["latency_p95"] * 2),
        )
        degraded = await run_evaluation(fault_cfg, folder, splits=["gate"], adapter=sut)
        gate = judge_gate(degraded, baseline, cfg)
        result = {
            "fault": metric,
            "fault_mode": "sut",
            "value": degraded["operational"][metric],
            "verdict": gate["verdict"],
            "blocked_for_target_metric": gate["verdict"] == "BLOCK"
            and any(f["metric"] == metric for f in gate["fails"]),
        }
        degradations.append(result)
        write_json(
            output_dir / "degraded" / (metric + ".json"),
            {
                "injected_fault": metric,
                "note": "实际SUT输出/等待/导出故障注入；不是训练得到的真实模型版本",
                "gate": gate,
            },
        )
    result = {
        "passed": all(v <= 0.003 for v in ranges.values())
        and complete
        and all(d["blocked_for_target_metric"] for d in degradations),
        "reproducible": all(v <= 0.003 for v in ranges.values()),
        "f1_ranges": ranges,
        "trace_complete": complete,
        "trace_n": len(traces),
        "degradations": degradations,
        "control_verdict": control["verdict"],
        "control_fails": control["fails"],
        "synthetic_only": baseline["meta"]["synthetic_only"],
        "scope": "验证工程逻辑与确定性。五个SUT故障版本实际运行完整gate；不代表真实模型性能。",
        "real_model_verified": False,
        "external_services_verified": False,
    }
    write_json(output_dir / "acceptance.json", result)
    return result
