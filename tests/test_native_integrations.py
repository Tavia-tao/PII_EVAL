import asyncio
import json
import threading


def test_native_deepeval_evaluate_runs_custom_metrics_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONFIDENT_API_KEY", "")
    from deepeval import evaluate
    from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
    from deepeval.test_case import LLMTestCase

    from src.metrics import BoundaryMetric, HighRiskRecallMetric, PIIF1Metric

    spans = json.dumps({"spans": [{"start": 0, "end": 2, "type": "PHONE"}]})
    result = evaluate(
        test_cases=[LLMTestCase(input="test", actual_output=spans, expected_output=spans)],
        metrics=[PIIF1Metric(), HighRiskRecallMetric(["PHONE"]), BoundaryMetric()],
        async_config=AsyncConfig(run_async=True, max_concurrent=2),
        display_config=DisplayConfig(show_indicator=False, print_results=False, inspect_after_run=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
    assert len(result.test_results) == 1
    assert result.test_results[0].success
    assert all(metric.score == 1 for metric in result.test_results[0].metrics_data)


def test_pdf_calls_stay_on_one_thread_under_async_concurrency(tmp_path, monkeypatch):
    from reportlab.pdfgen import canvas
    from test_contracts_metrics import sample_dict

    import src.adapter.local_pipeline as module
    from src.schema import Sample

    file = tmp_path / "a.pdf"
    pdf = canvas.Canvas(str(file))
    pdf.drawString(30, 700, "Phone: 19900000001")
    pdf.save()
    threads = []
    parse, export = module.parse_document, module.export_document

    def parse_spy(*args):
        threads.append(threading.get_ident())
        return parse(*args)

    def export_spy(*args):
        threads.append(threading.get_ident())
        return export(*args)

    monkeypatch.setattr(module, "parse_document", parse_spy)
    monkeypatch.setattr(module, "export_document", export_spy)
    pipeline = module.LocalPipeline(data_root=tmp_path, output_dir=tmp_path / "out")

    async def run():
        samples = []
        for i in range(12):
            d = sample_dict()
            d.update(sample_id=f"s{i}", track="complex_file", doc_type="pdf")
            d["input"].update(file_path="a.pdf")
            samples.append(Sample.model_validate(d))
        return await asyncio.gather(*(pipeline.predict(s) for s in samples))

    predictions = asyncio.run(run())
    assert all(p.stages["S6_export"].metrics["openable"] for p in predictions)
    assert len(threads) == 24
    assert set(threads) == {threading.get_ident()}
