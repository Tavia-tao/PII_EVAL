"""仅用于工程验收的显式故障注入。改变实际预测/等待/导出，不改最终聚合分数。"""

import asyncio
import re
import time
from pathlib import Path

from src.adapter.base import SUTAdapter
from src.schema import HIGH_RISK, Entity


class FaultDetector:
    def __init__(self, inner, fault):
        self.inner, self.fault = inner, fault

    async def detect(self, text):
        spans, usage = await self.inner.detect(text)
        if self.fault == "high_risk_recall":
            spans = [s for s in spans if s.type not in HIGH_RISK]
        elif self.fault == "risk_score":
            spans = [s for s in spans if s.type != "ID_CARD"]
        elif self.fault == "false_delete":
            for m in re.finditer(r"合同签订日[：: ]+(\d{4}-\d{2}-\d{2})", text):
                spans.append(Entity(start=m.start(1), end=m.end(1), type="DATE"))
        return spans, usage


class FaultSUT(SUTAdapter):
    def __init__(self, inner, fault, delay=0):
        self.inner, self.fault, self.delay = inner, fault, delay
        if fault in {"high_risk_recall", "risk_score", "false_delete"}:
            if not hasattr(inner, "detector"):
                raise ValueError("输出级故障注入需要本地stage适配器，不能伪改远端测量结果")
            inner.detector = FaultDetector(inner.detector, fault)

    async def predict(self, sample, overrides=None):
        start = time.perf_counter()
        pred = await self.inner.predict(sample, overrides)
        if self.fault == "latency_p95":
            await asyncio.sleep(self.delay)
            pred.stages["S4_detect"].elapsed_ms += self.delay * 1000
            pred.latency_ms = (time.perf_counter() - start) * 1000
        if self.fault == "file_openable":
            stage = pred.stages["S6_export"]
            if stage.artifact:
                Path(stage.artifact).write_bytes(b"INTENTIONAL_INVALID_EXPORT")
            stage.ok = False
            stage.metrics["openable"] = False
            stage.error = "injected_export_corruption"
            pred.error = "injected_export_corruption"
        return pred
