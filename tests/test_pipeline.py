from pathlib import Path

import httpx
import pytest
from test_contracts_metrics import sample_dict


def test_generate_real_files_reproducibly_without_split_leakage(tmp_path):
    from src.dataset import assert_no_leakage, load_dataset
    from src.generate import generate_dataset

    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(a, standard_count=100, complex_count=24, gate_count=24, seed=42)
    generate_dataset(b, standard_count=100, complex_count=24, gate_count=24, seed=42)
    sa, ma = load_dataset(a)
    sb, mb = load_dataset(b)
    assert len(sa) == 124
    assert sum(s.split == "gate" for s in sa) == 24
    assert sum(s.track == "complex_file" for s in sa) == 24
    assert ma["content_hash"] == mb["content_hash"]
    assert [s.model_dump() for s in sa] == [s.model_dump() for s in sb]
    assert_no_leakage(sa)
    paths = [a / s.input.file_path for s in sa if s.input.file_path]
    assert all(p.is_file() and p.stat().st_size > 100 for p in paths)
    assert {p.suffix for p in paths} >= {".pdf", ".docx", ".png"}
    assert {s.lang for s in sa} >= {"zh", "en", "zh_en_mixed"}
    assert any(s.meta.boundary == "nested" for s in sa)
    from docx import Document

    table_sample = next(s for s in sa if s.layout == "two_column_table")
    assert len(Document(a / table_sample.input.file_path).tables[0].columns) == 2


@pytest.mark.asyncio
async def test_local_prediction_does_not_read_gold_and_emits_six_stages(tmp_path):
    from src.adapter.local_pipeline import LocalPipeline
    from src.schema import STAGES, Sample

    a = Sample.model_validate(sample_dict())
    b = a.model_copy(deep=True)
    b.gold.spans = []
    b.gold.text = "完全不同的金标"
    pipeline = LocalPipeline(output_dir=tmp_path)
    pa, pb = await pipeline.predict(a), await pipeline.predict(b)
    assert [(s.start, s.end, s.type) for s in pa.spans] == [(3, 5, "NAME")]
    assert pa.spans == pb.spans
    assert list(pa.stages) == list(STAGES)
    assert all(s.ok for s in pa.stages.values())
    assert pa.stages["S6_export"].metrics["openable"] is True
    assert Path(pa.stages["S2_ocr"].artifact).is_file()
    assert "张三" not in Path(pa.stages["S6_export"].artifact).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_http_detector_sends_no_gold_and_rejects_out_of_bounds(tmp_path):
    from src.adapter.models import HTTPDetector

    captured = []

    def handler(request):
        import json

        data = json.loads(request.content)
        captured.append(data)
        return httpx.Response(200, json={"spans": [{"start": 0, "end": 500, "type": "NAME"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        d = HTTPDetector("https://model.example/predict", "model-v1", client=client)
        with pytest.raises(ValueError, match="边界|长度"):
            await d.detect("张三")
    assert captured == [{"text": "张三", "model": "model-v1", "prompt_version": "pii-span-v1"}]


@pytest.mark.asyncio
async def test_routing_catches_raw_high_risk_when_student_misses():
    from src.adapter.models import RegexDetector, RoutedDetector

    class MissAll(RegexDetector):
        async def detect(self, text):
            return [], {"input": 0, "output": 0, "cost_usd": 0, "measured": False}

    d = RoutedDetector(MissAll(), RegexDetector(teacher=True))
    spans, _ = await d.detect("身份证：110101199005121234")
    assert spans[0].type == "ID_CARD"
    assert d.last_route == "teacher"
    assert "high_risk" in d.last_reason


def test_length_preserving_mask_unions_nested_spans():
    from src.adapter.documents import mask_text
    from src.schema import Entity

    text = "地址：北京13800138000号"
    spans = [Entity(start=3, end=len(text), type="ADDRESS"), Entity(start=5, end=16, type="PHONE")]
    result = mask_text(text, spans)
    assert len(result) == len(text)
    assert result[:3] == "地址："
    assert set(result[3:]) == {"█"}


def test_docx_redaction_preserves_run_style_and_table(tmp_path):
    from docx import Document

    from src.adapter.documents import export_document, parse_document
    from src.schema import Entity

    p = tmp_path / "a.docx"
    doc = Document()
    r = doc.add_paragraph().add_run("姓名：张")
    r.bold = True
    doc.paragraphs[0].add_run("三")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "备注：测试"
    doc.save(p)
    blocks = parse_document(p)
    out = tmp_path / "masked.docx"
    export_document(p, blocks, [Entity(start=3, end=5, type="NAME")], out)
    got = Document(out)
    assert got.paragraphs[0].text == "姓名：██"
    assert got.paragraphs[0].runs[0].bold is True
    assert got.tables[0].cell(0, 0).text == "备注：测试"


@pytest.mark.asyncio
async def test_gray_date_only_judged_with_strict_budget_and_no_gold_mutation():
    from src.metrics.date_disambig import DateResolver

    calls = []

    async def judge(context, value):
        calls.append(value)
        return {"label": "pii", "confidence": 0.9}

    resolver = DateResolver(judge=judge, total_samples=100, max_ratio=0.04)
    assert (await resolver.resolve("出生日期：2000-01-01", "2000-01-01"))["label"] == "pii"
    assert (await resolver.resolve("合同签订日：2000-01-01", "2000-01-01"))["label"] == "not_pii"
    for _ in range(10):
        r = await resolver.resolve("2000-01-01", "2000-01-01")
        assert r["needs_human_review"]
    assert len(calls) == 4
    resolver.calibrate([True, False, False, True])
    assert resolver.enabled is False


def test_generated_manifest_tamper_is_not_accepted(tmp_path):
    from src.dataset import load_dataset
    from src.generate import generate_dataset

    generate_dataset(tmp_path, standard_count=30, complex_count=6, gate_count=12)
    file = tmp_path / "standard_text.jsonl"
    file.write_text(file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="摘要"):
        load_dataset(tmp_path)
