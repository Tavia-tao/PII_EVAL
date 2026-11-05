import json

import pytest


def sample_dict():
    return {
        "sample_id": "s1",
        "doc_id": "d1",
        "parent_doc_id": "d1",
        "split": "gate",
        "track": "standard_text",
        "source": "synthetic",
        "layout": "pure_text",
        "input": {"text": "姓名：张三，日期：2024-03-05"},
        "gold": {
            "text": "姓名：张三，日期：2024-03-05",
            "spans": [{"start": 3, "end": 5, "type": "NAME", "text": "张三", "risk": "normal"}],
            "non_pii_candidates": [
                {
                    "start": 9,
                    "end": 19,
                    "type": "DATE_NON_PII",
                    "text": "2024-03-05",
                    "reason": "contract_date",
                }
            ],
        },
        "meta": {"rule_version": "rule-v2.1"},
    }


def test_schema_checks_gold_text_not_just_json_shape():
    from src.schema import Sample

    d = sample_dict()
    assert Sample.model_validate(d).gold.spans[0].text == "张三"
    d["gold"]["spans"][0]["end"] = 6
    with pytest.raises(ValueError, match="text|文本|边界"):
        Sample.model_validate(d)


def test_schema_rejects_crossing_and_duplicate_gold():
    from src.schema import Sample

    d = sample_dict()
    d["gold"]["spans"] = [{"start": 0, "end": 4, "type": "ADDRESS"}, {"start": 3, "end": 5, "type": "NAME"}]
    with pytest.raises(ValueError, match="重叠"):
        Sample.model_validate(d)
    d = sample_dict()
    d["gold"]["spans"] *= 2
    with pytest.raises(ValueError, match="重复"):
        Sample.model_validate(d)


def test_parent_and_content_leakage_detected():
    from src.dataset import assert_no_leakage
    from src.schema import Sample

    a = Sample.model_validate(sample_dict())
    b = a.model_copy(deep=True)
    b.sample_id, b.split, b.doc_id = "s2", "holdout", "d2"
    with pytest.raises(ValueError, match="泄漏|重复"):
        assert_no_leakage([a, b])


def test_same_split_duplicate_content_is_rejected():
    from src.dataset import assert_no_leakage
    from src.schema import Sample

    a = Sample.model_validate(sample_dict())
    b = a.model_copy(deep=True)
    b.sample_id, b.doc_id, b.parent_doc_id = "s2", "d2", "d2"
    with pytest.raises(ValueError, match="重复"):
        assert_no_leakage([a, b])


def test_non_pii_denominator_and_strict_partial():
    from src.evaluation import evaluate_sample
    from src.schema import Prediction, Sample

    s = Sample.model_validate(sample_dict())
    p = Prediction(
        sample_id="s1",
        spans=[{"start": 3, "end": 6, "type": "NAME"}, {"start": 9, "end": 19, "type": "DATE"}],
        coordinate_text=s.gold.text,
    )
    r = evaluate_sample(s, p)
    assert (r["tp"], r["fp"], r["fn"], r["partial"]) == (0, 2, 1, 1)
    assert r["false_delete"] == 1.0  # 分母是显式非PII候选，不把边界错误重复算为业务误删
    assert r["false_negative"] == 1.0
    assert r["boundary_acc"] == 0.0
    assert r["hard_negative_fp"]["contract_date"] == 1


def test_missing_negative_universe_is_unknown():
    from src.evaluation import evaluate_sample
    from src.schema import Prediction, Sample

    d = sample_dict()
    d["gold"]["non_pii_candidates"] = []
    r = evaluate_sample(Sample.model_validate(d), Prediction(sample_id="s1"))
    assert r["false_delete"] is None


def test_wrong_type_is_charged_to_both_categories():
    from src.evaluation import evaluate_sample
    from src.schema import Prediction, Sample

    p = Prediction(sample_id="s1", spans=[{"start": 3, "end": 5, "type": "PHONE"}])
    r = evaluate_sample(Sample.model_validate(sample_dict()), p)
    assert r["by_type"]["PHONE"]["fp"] == 1
    assert r["by_type"]["NAME"]["fn"] == 1


def test_kappa_requires_true_negative_universe():
    from src.quality import annotation_agreement, cohen_kappa

    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    r = annotation_agreement(
        ["NAME", "PHONE", "NOT_PII", "NOT_PII"], ["NAME", "NOT_PII", "NOT_PII", "NOT_PII"]
    )
    assert r["n"] == 4
    assert not r["passed"]


def test_bootstrap_paired_delta_has_correct_direction():
    from src.stats import paired_compare

    a = [{"sample_id": str(i), "layout": "scan", "tp": 1, "fp": 0, "fn": 0} for i in range(20)]
    b = [{"sample_id": str(i), "layout": "scan", "tp": 0, "fp": 0, "fn": 1} for i in range(20)]
    r = paired_compare(a, b, n_boot=100)
    assert r["lo"] == r["hi"] == r["point"] == 1
    assert r["significant"] and r["direction"] == "A>B"
    with pytest.raises(ValueError):
        paired_compare(a, b[1:], n_boot=100)


def test_deepeval_sync_async_share_strict_formula():
    import asyncio

    from deepeval.test_case import LLMTestCase

    from src.metrics import BoundaryMetric, HighRiskRecallMetric, PIIF1Metric

    tc = LLMTestCase(
        input="abc",
        expected_output=json.dumps({"spans": [{"start": 0, "end": 2, "type": "PHONE"}]}),
        actual_output=json.dumps({"spans": [{"start": 0, "end": 3, "type": "PHONE"}]}),
    )
    for metric in [PIIF1Metric(), HighRiskRecallMetric(["PHONE"]), BoundaryMetric()]:
        assert metric.measure(tc) == 0
        assert asyncio.run(metric.a_measure(tc)) == 0
        assert metric.is_successful() is False


def test_ocr_cer_uses_actual_intermediate_and_gold_in_evaluator(tmp_path):
    from src.evaluation import evaluate_sample
    from src.io import write_json
    from src.schema import Prediction, Sample, Stage

    s = Sample.model_validate(sample_dict())
    artifact = tmp_path / "stages.json"
    write_json(
        artifact,
        {
            "S1_parse": {"text": s.gold.text},
            "S2_ocr": {"text": s.gold.text.replace("张", "弓")},
            "S3_layout": {"block_order": []},
        },
    )
    pred = Prediction(sample_id="s1", stages={"S2_ocr": Stage(artifact=str(artifact))})
    row = evaluate_sample(s, pred)
    assert row["stages"]["S2_ocr"]["metrics"]["cer"] == pytest.approx(1 / 19)


def test_all_negative_document_does_not_generate_nan_report():
    from src.evaluation import aggregate, evaluate_sample
    from src.schema import Prediction, Sample

    d = sample_dict()
    d["gold"]["spans"] = []
    r = aggregate([evaluate_sample(Sample.model_validate(d), Prediction(sample_id="s1"))])
    assert r["f1"] is None
    assert r["macro_f1"] is None
