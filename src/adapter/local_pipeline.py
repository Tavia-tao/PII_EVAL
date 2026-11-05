import asyncio
import os
import subprocess
import time
from pathlib import Path

import httpx

from src.adapter.base import SUTAdapter
from src.adapter.documents import export_document, join_blocks, mask_text, parse_document
from src.adapter.models import RegexDetector, RoutedDetector
from src.io import write_json
from src.schema import STAGES, Block, Prediction, Stage, Usage


class LocalPipeline(SUTAdapter):
    def __init__(
        self,
        detector=None,
        model_version="demo-student-v1",
        data_root=".",
        output_dir="runtime/artifacts",
        ocr_endpoint=None,
        ocr_timeout=60,
        observer=None,
    ):
        self.detector = detector or RegexDetector()
        self.model_version, self.data_root = model_version, Path(data_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.ocr_endpoint, self.ocr_timeout = ocr_endpoint, ocr_timeout
        self.observer = observer

    async def _ocr(self, sample, source, blocks):
        mode = sample.input.ocr_mode
        if mode == "replay":
            if sample.input.ocr_text is None:
                raise ValueError("replay 模式需要显式 OCR 产物")
            lines = sample.input.ocr_text.split("\n")
            # 回放文本单独提供，不能从 gold 获取。合成图片对应固定行区域。
            return [
                Block(block_id=f"b{i}", text=line, order=i, bbox=(25, 15 + 40 * i, 990, 55 + 40 * i))
                for i, line in enumerate(lines)
            ]
        if mode == "tesseract":
            result = await asyncio.to_thread(
                subprocess.run,
                ["tesseract", str(source), "stdout", "-l", os.getenv("OCR_LANGUAGE", "chi_sim+eng"), "tsv"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.ocr_timeout,
                check=True,
            )
            import csv
            import io

            groups = {}
            for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
                if not row["text"].strip():
                    continue
                key = (row["page_num"], row["block_num"], row["par_num"], row["line_num"])
                groups.setdefault(key, []).append(row)
            out = []
            for key, words in groups.items():
                x1, y1 = min(int(w["left"]) for w in words), min(int(w["top"]) for w in words)
                x2 = max(int(w["left"]) + int(w["width"]) for w in words)
                y2 = max(int(w["top"]) + int(w["height"]) for w in words)
                out.append(
                    Block(
                        block_id=f"b{len(out)}",
                        order=len(out),
                        text=" ".join(w["text"] for w in words),
                        page=int(key[0]),
                        bbox=(x1, y1, x2, y2),
                    )
                )
            return out
        if mode == "http":
            if not self.ocr_endpoint:
                raise ValueError("未配置 OCR endpoint")
            async with httpx.AsyncClient(timeout=self.ocr_timeout) as client:
                with source.open("rb") as file:
                    response = await client.post(
                        self.ocr_endpoint,
                        files={"file": (source.name, file)},
                        headers={"Authorization": "Bearer " + os.environ.get("OCR_API_KEY", "")},
                    )
                response.raise_for_status()
                return [Block.model_validate(b) for b in response.json()["blocks"]]
        if not any(b.text for b in blocks):
            raise ValueError("文件无文本层，请显式配置 replay/tesseract/http OCR")
        return blocks

    async def predict(self, sample, overrides=None):
        p = Prediction(sample_id=sample.sample_id, model_version=self.model_version)
        source = None
        current_stage = STAGES[0]
        observation = self.observer.start_stage(current_stage) if self.observer else None
        begin = time.perf_counter()
        intermediates = {}
        stage_file = (
            self.output_dir / "stages" / self.model_version.replace("/", "_") / (sample.sample_id + ".json")
        )
        try:
            t = time.perf_counter()
            if sample.input.file_path:
                source = (self.data_root / sample.input.file_path).resolve()
                if not source.is_relative_to(self.data_root):
                    raise ValueError("源文件路径越界")
                # PyMuPDF不支持多线程：PDF始终在当前评测线程执行。
                blocks = (
                    parse_document(source)
                    if source.suffix.lower() == ".pdf"
                    else await asyncio.to_thread(parse_document, source)
                )
            else:
                blocks = [
                    Block(block_id=f"b{i}", text=line, order=i)
                    for i, line in enumerate(sample.input.text.split("\n"))
                ]
            p.stages[current_stage] = Stage(
                elapsed_ms=(time.perf_counter() - t) * 1000,
                metrics={"blocks": len(blocks)},
                artifact=str(stage_file),
            )
            intermediates[current_stage] = {
                "text": join_blocks(blocks),
                "blocks": [b.model_dump() for b in blocks],
            }
            if self.observer:
                self.observer.end_stage(observation, p.stages[current_stage])
            current_stage = STAGES[1]
            observation = self.observer.start_stage(current_stage) if self.observer else None
            t = time.perf_counter()
            blocks = await self._ocr(sample, source, blocks)
            if overrides and overrides.ocr_text is not None:
                lines = overrides.ocr_text.split("\n")
                blocks = [
                    b.model_copy(update={"text": lines[i]}) if i < len(lines) else b
                    for i, b in enumerate(blocks)
                ]
                if len(lines) != len(blocks):
                    blocks = [Block(block_id=f"b{i}", order=i, text=line) for i, line in enumerate(lines)]
            p.stages[current_stage] = Stage(
                elapsed_ms=(time.perf_counter() - t) * 1000,
                metrics={
                    "replay": sample.input.ocr_mode == "replay",
                    "oracle_override": bool(overrides and overrides.ocr_text is not None),
                },
                artifact=str(stage_file),
            )
            intermediates[current_stage] = {
                "text": join_blocks(blocks),
                "blocks": [b.model_dump() for b in blocks],
            }
            if self.observer:
                self.observer.end_stage(observation, p.stages[current_stage])
            current_stage = STAGES[2]
            observation = self.observer.start_stage(current_stage) if self.observer else None
            t = time.perf_counter()
            if overrides and overrides.layout_blocks:
                # 布局替换只改变顺序/坐标，不覆盖OCR文本，避免混淆S2/S3贡献。
                order = {b.block_id: i for i, b in enumerate(overrides.layout_blocks)}
                blocks = sorted(blocks, key=lambda b: order.get(b.block_id, len(order)))
            else:
                blocks = sorted(blocks, key=lambda b: (b.page, b.order))
            text = join_blocks(blocks)
            p.coordinate_text = text
            p.stages[current_stage] = Stage(
                elapsed_ms=(time.perf_counter() - t) * 1000,
                metrics={
                    "blocks": len(blocks),
                    "oracle_override": bool(overrides and overrides.layout_blocks),
                },
                artifact=str(stage_file),
            )
            intermediates[current_stage] = {"text": text, "block_order": [b.block_id for b in blocks]}
            if self.observer:
                self.observer.end_stage(observation, p.stages[current_stage])
            current_stage = STAGES[3]
            observation = self.observer.start_stage(current_stage) if self.observer else None
            t = time.perf_counter()
            if isinstance(self.detector, RoutedDetector):
                spans, usage = await self.detector.detect(text, sample.layout)
                p.route, p.route_reason = self.detector.last_route, self.detector.last_reason
            else:
                spans, usage = await self.detector.detect(text)
                p.route = "teacher" if "teacher" in self.model_version else "student"
            if any(s.end > len(text) for s in spans):
                raise ValueError("模型输出越界")
            p.spans, p.usage = spans, Usage.model_validate(usage)
            p.needs_review = any(s.conf < 0.85 for s in spans)
            p.stages[current_stage] = Stage(
                elapsed_ms=(time.perf_counter() - t) * 1000,
                metrics={"span_count": len(spans)},
                artifact=str(stage_file),
            )
            intermediates[current_stage] = {"spans": [s.model_dump() for s in spans]}
            if self.observer:
                self.observer.end_stage(observation, p.stages[current_stage], p.usage)
            current_stage = STAGES[4]
            observation = self.observer.start_stage(current_stage) if self.observer else None
            t = time.perf_counter()
            masked = mask_text(text, spans)
            p.stages[current_stage] = Stage(
                elapsed_ms=(time.perf_counter() - t) * 1000,
                metrics={"coverage": 1.0, "length_preserved": len(masked) == len(text), "reversible": False},
                artifact=str(stage_file),
            )
            intermediates[current_stage] = {"text": masked}
            if self.observer:
                self.observer.end_stage(observation, p.stages[current_stage])
            current_stage = STAGES[5]
            observation = self.observer.start_stage(current_stage) if self.observer else None
            t = time.perf_counter()
            suffix = source.suffix.lower() if source else ".txt"
            if suffix in {".jpeg", ".tiff", ".bmp"}:
                suffix = ".png"
            out = self.output_dir / self.model_version.replace("/", "_") / (sample.sample_id + suffix)
            artifact = (
                export_document(source, blocks, spans, out)
                if suffix == ".pdf"
                else await asyncio.to_thread(export_document, source, blocks, spans, out)
            )
            p.stages[current_stage] = Stage(
                elapsed_ms=(time.perf_counter() - t) * 1000,
                metrics={
                    "openable": True,
                    "format_preserved": suffix not in {".png", ".jpg"},
                    "layout_verified": False,
                },
                artifact=artifact,
            )
            if self.observer:
                self.observer.end_stage(observation, p.stages[current_stage])
        except Exception as e:
            p.error = f"{current_stage}: {type(e).__name__}: {str(e)[:200]}"
            p.stages[current_stage] = Stage(ok=False, error=p.error)
            if self.observer:
                self.observer.end_stage(observation, p.stages[current_stage])
            for name in STAGES:
                if name not in p.stages:
                    p.stages[name] = Stage(ok=False, error="upstream_failed")
        write_json(stage_file, intermediates)
        p.latency_ms = (time.perf_counter() - begin) * 1000
        return p
