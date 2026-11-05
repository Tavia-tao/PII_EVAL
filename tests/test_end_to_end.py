import asyncio
import json
import subprocess
import sys

import pytest


def test_cli_entry_generate_validate_run_and_gate(tmp_path):
    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "src.cli", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )

    generated = run(
        "generate", "--out", str(tmp_path / "data"), "--standard", "50", "--complex", "16", "--gate", "20"
    )
    assert generated.returncode == 0, generated.stderr
    validated = run("validate", "--data", str(tmp_path / "data"))
    assert validated.returncode == 0, validated.stderr
    evaluated = run(
        "run", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "report"), "--bootstrap", "30"
    )
    assert evaluated.returncode == 0, evaluated.stderr
    report = tmp_path / "report" / "report.json"
    gated = run(
        "gate", "--report", str(report), "--baseline", str(report), "--out", str(tmp_path / "gate.json")
    )
    assert gated.returncode == 2  # 20份样本不足以证明五类高风险覆盖，不能放行。
    assert json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))["verdict"] == "BLOCK"


def test_source_file_tamper_changes_dataset_integrity(tmp_path):
    from src.dataset import load_dataset
    from src.generate import generate_dataset

    generate_dataset(tmp_path, 30, 8, 12)
    file = next((tmp_path / "files").glob("*.png"))
    file.write_bytes(file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="摘要|artifact|文件"):
        load_dataset(tmp_path)


def test_pdf_export_removes_underlying_pii(tmp_path):
    import pymupdf as fitz
    from reportlab.pdfgen import canvas

    from src.adapter.documents import export_document, parse_document
    from src.schema import Entity

    original = tmp_path / "original.pdf"
    c = canvas.Canvas(str(original))
    c.drawString(30, 700, "Phone: 19900000001")
    c.save()
    blocks = parse_document(original)
    output = tmp_path / "masked.pdf"
    export_document(original, blocks, [Entity(start=7, end=18, type="PHONE")], output)
    with fitz.open(output) as doc:
        assert "19900000001" not in "".join(p.get_text() for p in doc)
    with fitz.open(original) as doc:
        assert "19900000001" in doc[0].get_text()


def test_attribution_reexecutes_downstream_and_keeps_model_gap(tmp_path):
    from src.attribution import run_attribution
    from src.config import load_config
    from src.generate import generate_dataset

    generate_dataset(tmp_path / "data", 30, 16, 24)
    cfg = load_config()
    cfg["dataset"]["root"] = str(tmp_path / "data")
    cfg["execution"]["bootstrap"] = 20
    result = asyncio.run(run_attribution(cfg, tmp_path / "attr"))
    assert set(result["scores"]) == {"baseline", "ocr", "layout", "ocr_layout"}
    assert result["pipeline_total_gap"] == pytest.approx(
        result["scores"]["ocr_layout"] - result["scores"]["baseline"]
    )
    assert (tmp_path / "attr" / "ocr" / "traces.jsonl").is_file()


def test_retry_error_cannot_be_hidden_and_partial_resume_is_versioned(tmp_path):
    from src.config import load_config
    from src.generate import generate_dataset
    from src.runner import run_evaluation
    from src.schema import STAGES, Prediction, Stage

    generate_dataset(tmp_path / "data", 30, 8, 12)

    class AlwaysFail:
        calls = 0

        async def predict(self, sample, overrides=None):
            self.calls += 1
            return Prediction(
                sample_id=sample.sample_id,
                error="service_unavailable",
                stages={s: Stage(ok=False) for s in STAGES},
            )

    cfg = load_config()
    cfg["dataset"]["root"] = str(tmp_path / "data")
    cfg["execution"].update(bootstrap=20, retry_backoff_seconds=0, cache_enabled=False)
    adapter = AlwaysFail()
    report = asyncio.run(run_evaluation(cfg, tmp_path / "run", adapter=adapter))
    assert report["operational"]["execution_errors"] == 12
    assert adapter.calls == 48
    cfg["rules"]["weights"]["NAME"] = 3
    with pytest.raises(ValueError, match="不同版本"):
        asyncio.run(run_evaluation(cfg, tmp_path / "run"))


def test_comparison_rejects_different_cohort_and_reports_no_significance_for_identical_runs():
    import copy

    from test_operations import gate_fixture

    from src.report import compare_reports

    a = gate_fixture()
    a["rows"] = [
        {
            "sample_id": "s",
            "track": "standard_text",
            "layout": "pure_text",
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "hard_negative_fp": {},
        },
        {
            "sample_id": "cx",
            "track": "complex_file",
            "layout": "scan",
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "hard_negative_fp": {},
        },
    ]
    a["meta"]["synthetic_only"] = True
    a["operational"].update(cost_measured=False, token_cost=0)
    result = compare_reports(a, a, n_boot=30)
    assert result["tracks"]["standard_text"]["direction"] == "uncertain"
    b = copy.deepcopy(a)
    b["meta"]["sample_set_hash"] = "another"
    with pytest.raises(ValueError, match="不可比"):
        compare_reports(a, b)


def test_langfuse_sdk_trace_and_scores_without_network():
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from test_contracts_metrics import sample_dict

    from src.observability import Observer
    from src.schema import STAGES, Sample, Stage

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key="pk-offline-test",
        secret_key="sk-offline-test",
        tracer_provider=provider,
        span_exporter=exporter,
        flush_at=512,
        flush_interval=600,
    )
    observer = Observer({"enabled": False}, "r", {"rule_version": "r1", "model_version": "m1"})
    observer.client = client
    with observer.sample_trace(Sample.model_validate(sample_dict())) as trace_id:
        for stage in STAGES:
            obs = observer.start_stage(stage)
            observer.end_stage(obs, Stage(metrics={"ok": True}))
    assert len(trace_id) == 32
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert set(STAGES) <= names
    assert all(s.parent is not None for s in spans if s.name in STAGES)
    client.shutdown()
    provider.shutdown()


def test_langfuse_run_scores_keep_track_dimensions():
    from src.observability import Observer

    captured = []

    class ScoresClient:
        def create_score(self, **kwargs):
            captured.append(kwargs)

    observer = Observer({"enabled": False}, "run-test", {"rule_version": "r1", "model_version": "m1"})
    observer.client = ScoresClient()
    observer.report_run_scores(
        {
            "tracks": {
                "standard_text": {"f1": 0.9, "latency_p95": 2.0, "token_cost": 0.01, "cost_measured": True}
            },
            "meta": {},
        },
        "experiment-1",
    )
    assert {r["name"] for r in captured} >= {"overall_f1", "latency_p95", "token_cost"}
    assert all(r["metadata"]["track"] == "standard_text" for r in captured)
    assert all(r["dataset_run_id"] == "experiment-1" for r in captured)
