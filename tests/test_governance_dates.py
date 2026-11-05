import asyncio
import copy

import pytest


def test_policy_change_needs_independent_approval():
    from src.governance import approve_change

    old = {"threshold": 0.98}
    new = {"threshold": 0.90}
    with pytest.raises(ValueError, match="独立"):
        approve_change(
            "threshold", old, new, {"author": "a", "reviewer": "a", "reason": "test", "version": "v2"}
        )
    r = approve_change(
        "threshold",
        old,
        new,
        {"author": "a", "reviewer": "b", "reason": "reviewed evidence", "version": "v2"},
    )
    assert r["before_hash"] != r["after_hash"]


def test_date_false_positive_reduction_has_paired_ci():
    from src.stats import paired_rate_change

    a = [{"sample_id": str(i), "layout": "table", "count": 1} for i in range(20)]
    b = [{"sample_id": str(i), "layout": "table", "count": 0} for i in range(20)]
    r = paired_rate_change(a, b, n_boot=30)
    assert r["relative_reduction"] == 1
    assert r["reduction_ci"] == [1, 1]
    assert r["significant"]


def test_date_gray_rule_runs_in_actual_evaluation(tmp_path):
    from test_contracts_metrics import sample_dict

    from src.config import load_config
    from src.dataset import register_dataset
    from src.io import write_jsonl
    from src.runner import run_evaluation

    d = sample_dict()
    d["input"]["text"] = d["gold"]["text"] = "姓名：张三，日期：2024-03-05"
    # 这个日期无业务/个人锚点，规则不能擅自判定。
    d["gold"]["non_pii_candidates"][0]["reason"] = "adjudicated_not_pii"
    root = tmp_path / "data"
    write_jsonl(root / "standard_text.jsonl", [d])
    write_jsonl(root / "complex_file.jsonl", [])
    register_dataset(root, "date-gray-v1")
    cfg = load_config()
    cfg["dataset"]["root"] = str(root)
    cfg["execution"].update(cache_enabled=False, bootstrap=20)
    r = asyncio.run(run_evaluation(cfg, tmp_path / "run"))
    assert r["rows"][0]["needs_review"]
    assert r["rows"][0]["date_decisions"][0]["label"] == "uncertain"
    assert r["meta"]["judge_calls"] == 0


def test_threshold_hash_mismatch_blocks_gate():
    from test_operations import gate_fixture

    from src.config import load_config
    from src.gate import judge_gate
    from src.io import digest

    cfg = load_config()
    report = gate_fixture()
    report["meta"]["gate_policy_hash"] = digest(cfg["gate"])
    baseline = copy.deepcopy(report)
    cfg["gate"]["thresholds"]["high_risk_recall"]["hard"] = 0.90
    result = judge_gate(report, baseline, cfg)
    assert result["verdict"] == "BLOCK"
    assert any(f["metric"] == "gate_policy" for f in result["fails"])
