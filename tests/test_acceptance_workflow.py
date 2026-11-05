import asyncio


def test_acceptance_checks_controls_and_degradations(tmp_path):
    from src.config import load_config
    from src.generate import generate_dataset
    from src.verification import acceptance_run

    generate_dataset(tmp_path / "data", 500, 64, 180)
    cfg = load_config()
    cfg["dataset"]["root"] = str(tmp_path / "data")
    cfg["execution"].update(bootstrap=20, retry_backoff_seconds=0)
    # 这个缩小集只验证工程闭环，不伪装成满足正式50/类别样本量。
    cfg["gate"]["min_per_high_risk_type"] = 1
    result = asyncio.run(acceptance_run(cfg, tmp_path / "verification"))
    assert result["reproducible"]
    assert len(result["degradations"]) == 5
    assert all(r["blocked_for_target_metric"] for r in result["degradations"])
    assert all(r["fault_mode"] == "sut" for r in result["degradations"])
    assert (tmp_path / "verification" / "fault_high_risk_recall" / "report.json").exists()
    assert result["trace_complete"]
    assert result["synthetic_only"]


def test_annotation_quality_plan_does_not_fabricate_human_scores():
    from test_contracts_metrics import sample_dict

    from src.quality import quality_sample_plan
    from src.schema import Sample

    result = quality_sample_plan([Sample.model_validate(sample_dict())])
    assert len(result["assignments"]["double_blind_20pct"]) == 1
    assert "kappa" not in result
