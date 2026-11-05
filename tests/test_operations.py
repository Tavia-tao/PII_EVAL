import asyncio
import copy
import math

import pytest


def gate_fixture():
    metrics = {
        "f1": 0.96,
        "high_risk_recall": 0.995,
        "high_risk_min": 0.995,
        "false_delete": 0.005,
        "risk_score": 0.01,
        "latency_p95": 2.0,
        "file_openable": 1.0,
        "execution_errors": 0,
        "negative_coverage": 1.0,
        "by_type": {
            t: {"n": 100, "recall": 0.995}
            for t in ["ID_CARD", "BANK_CARD", "PHONE", "PASSPORT", "MEDICAL_ID"]
        },
    }
    meta = {
        "dataset_hash": "d",
        "rule_version": "r",
        "metric_version": "strict-v1",
        "sample_set_hash": "s",
        "synthetic_only": True,
        "model_version": "demo",
        "complete_gate": True,
        "incremental": False,
    }
    return {
        "meta": meta,
        "tracks": {"standard_text": copy.deepcopy(metrics), "complex_file": copy.deepcopy(metrics)},
        "operational": metrics,
        "rows": [],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("high_risk_recall", 0.97),
        ("false_delete", 0.02),
        ("risk_score", 0.011),
        ("latency_p95", 3),
        ("file_openable", 0.999),
    ],
)
def test_gate_blocks_five_degradations(field, value):
    from src.gate import judge_gate

    baseline = gate_fixture()
    report = copy.deepcopy(baseline)
    report["operational"][field] = value
    assert judge_gate(report, baseline)["verdict"] == "BLOCK"


def test_gate_never_allows_missing_nan_incremental_or_mismatched_evidence():
    from src.gate import judge_gate

    baseline = gate_fixture()
    for change in [
        lambda r: r["operational"].pop("false_delete"),
        lambda r: r["operational"].update(high_risk_recall=math.nan),
        lambda r: r["meta"].update(incremental=True),
        lambda r: r["meta"].update(dataset_hash="changed"),
        lambda r: r["meta"].update(complete_gate=False),
        lambda r: r["operational"]["by_type"].pop("PASSPORT"),
    ]:
        r = copy.deepcopy(baseline)
        change(r)
        assert judge_gate(r, baseline)["verdict"] == "BLOCK"


def test_soft_warning_l1_binary_and_release_not_claimed_for_synthetic():
    from src.gate import judge_gate

    baseline = gate_fixture()
    assert judge_gate(baseline, baseline)["verdict"] == "PASS"
    assert judge_gate(baseline, baseline)["release_ready"] is False
    r = copy.deepcopy(baseline)
    r["operational"]["latency_p95"] = 2.3
    assert judge_gate(r, baseline, level="L2")["verdict"] == "WARN"
    assert judge_gate(r, baseline, level="L1")["verdict"] == "BLOCK"


def test_weak_high_risk_category_not_hidden_by_average():
    from src.gate import judge_gate

    r = gate_fixture()
    baseline = copy.deepcopy(r)
    r["operational"]["by_type"]["PASSPORT"]["recall"] = 0.9
    assert judge_gate(r, baseline)["verdict"] == "BLOCK"


def test_feedback_status_requires_two_independent_reviewers_before_promotion(tmp_path):
    from src.feedback import FeedbackStore

    store = FeedbackStore(tmp_path / "feedback.sqlite")
    original = [{"start": 0, "end": 2, "type": "NAME"}]
    corrected = [{"start": 0, "end": 3, "type": "NAME"}]
    case = store.enqueue("s1", original, "boundary", track="complex_file", layout="scan", high_risk=True)
    assert store.record(case, "a", original, accepted=True)["status"] == "accepted_as_is"
    assert store.record(case, "a", corrected, accepted=True)["status"] == "accepted_with_edit"
    with pytest.raises(ValueError, match="复核|独立"):
        store.promote(case, "gate-v2")
    store.record(case, "b", corrected, accepted=True)
    assert store.promote(case, "gate-v2")["target"] == "gate-v2"
    assert store.bad_cases()[0]["sample_id"] == "s1"
    c = store.enqueue("s2", [], "false_positive")
    assert store.record(c, "c", [], accepted=False)["status"] == "rejected"


def test_drift_sustained_latency_and_three_day_recall():
    from src.monitor import detect_drift

    baseline = {"high_risk_recall": 0.99, "false_delete": 0.01, "latency_p95": 1.0, "token_cost": 0.01}
    daily = [
        {
            "timestamp": f"2026-11-{i + 1:02d}T00:00:00+08:00",
            "high_risk_recall": 0.99 - 0.03 * i,
            "false_delete": 0.02,
            "latency_p95": 1.5,
            "token_cost": 0.02,
            "teacher_rate": 0.8,
            "parse_failure_rate": 0.01,
            "audited_n": 200,
        }
        for i in range(4)
    ]
    alerts = detect_drift(daily, baseline, teacher_max_ratio=0.3)
    assert {a["metric"] for a in alerts} >= {
        "high_risk_recall",
        "false_delete",
        "latency_p95",
        "token_cost",
        "teacher_rate",
        "parse_failure_rate",
    }
    short = [daily[-1]]
    assert "latency_p95" not in {a["metric"] for a in detect_drift(short, baseline)}


def test_cache_identity_contains_file_model_rule_code_and_config(tmp_path):
    from src.cache import PredictionCache

    cache = PredictionCache(tmp_path / "cache")
    parts = {
        "sample_hash": "a",
        "model_version": "b",
        "rule_version": "c",
        "code_version": "d",
        "dataset_version": "e",
        "config_hash": "f",
    }
    a = cache.key(parts)
    for k in parts:
        assert cache.key({**parts, k: "changed"}) != a


def test_runner_resume_error_retry_and_trace(tmp_path):
    from src.config import load_config
    from src.generate import generate_dataset
    from src.io import read_jsonl
    from src.runner import run_evaluation

    generate_dataset(tmp_path / "data", standard_count=50, complex_count=16, gate_count=20)
    cfg = load_config()
    cfg["dataset"]["root"] = str(tmp_path / "data")
    cfg["execution"]["cache_dir"] = str(tmp_path / "cache")
    cfg["execution"]["bootstrap"] = 30
    first = asyncio.run(run_evaluation(cfg, output_dir=tmp_path / "one", splits=["gate"]))
    second = asyncio.run(run_evaluation(cfg, output_dir=tmp_path / "two", splits=["gate"]))
    assert first["meta"]["sample_set_hash"] == second["meta"]["sample_set_hash"]
    assert first["tracks"]["standard_text"]["f1"] == second["tracks"]["standard_text"]["f1"]
    assert second["meta"]["cache_hits"] == first["meta"]["n"]
    traces = list(read_jsonl(tmp_path / "two" / "traces.jsonl"))
    assert len(traces) == 20
    assert all(len(t["stages"]) == 6 and t["metadata"]["rule_version"] for t in traces)
    assert (tmp_path / "two" / "report.html").is_file()


def test_store_sqlite_roundtrip_and_health(tmp_path):
    from src.storage import ResultStore

    store = ResultStore("sqlite+pysqlite:///" + (tmp_path / "runs.db").as_posix())
    store.save_run("r1", {"meta": {"model_version": "demo"}, "tracks": {"standard_text": {"f1": 0.9}}})
    assert store.get_run("r1")["tracks"]["standard_text"]["f1"] == 0.9
    assert len(store.list_runs()) == 1


def test_latency_gate_failure_lists_slow_documents_not_only_detection_errors():
    from src.gate import judge_gate

    report = gate_fixture()
    baseline = copy.deepcopy(report)
    report["operational"]["latency_p95"] = 3
    report["rows"] = [
        {
            "sample_id": "slow",
            "latency_ms": 3000,
            "fp": 0,
            "fn": 0,
            "execution_error": False,
            "file_openable": True,
        }
    ]
    result = judge_gate(report, baseline)
    assert [r["sample_id"] for r in result["failure_samples"]] == ["slow"]
