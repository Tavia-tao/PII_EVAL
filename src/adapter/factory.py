import os

import httpx

from src.adapter.base import SUTAdapter
from src.adapter.local_pipeline import LocalPipeline
from src.adapter.models import ChatDetector, HTTPDetector, RegexDetector, RoutedDetector
from src.schema import Prediction


def build_detector(alias, cfg):
    model = cfg["models"][alias]
    kind = model["detector"]
    if kind in {"regex", "regex_teacher"}:
        return RegexDetector(teacher=kind == "regex_teacher")
    if kind == "routed":
        return RoutedDetector(
            build_detector(model["student"], cfg),
            build_detector(model["teacher"], cfg),
            confidence=model["confidence"],
            teacher_max_ratio=model["teacher_max_ratio"],
        )
    if kind in {"http", "chat"}:
        cls = HTTPDetector if kind == "http" else ChatDetector
        return cls(
            endpoint=model.get("endpoint") or os.environ[model["endpoint_env"]],
            model_version=model["version"],
            api_key_env=model["api_key_env"],
            prompt_version=model["prompt_version"],
            timeout=cfg["execution"]["timeout_seconds"],
        )
    raise ValueError(f"未知detector: {kind}")


class HTTPPipeline(SUTAdapter):
    """被测系统自带六阶段时，以统一契约直接评测，不用本地解析器替代。"""

    def __init__(self, model, timeout):
        self.model, self.timeout = model, timeout

    async def predict(self, sample, overrides=None):
        payload = {
            "sample_id": sample.sample_id,
            "input": sample.input.model_dump(),
            "model_version": self.model["version"],
        }
        if overrides:
            payload["evaluation_overrides"] = {
                "ocr_text": overrides.ocr_text,
                "layout_blocks": [b.model_dump() for b in overrides.layout_blocks]
                if overrides.layout_blocks
                else None,
            }
        endpoint = self.model.get("endpoint") or os.environ[self.model["endpoint_env"]]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                endpoint,
                json=payload,
                headers={"Authorization": "Bearer " + os.environ.get(self.model["api_key_env"], "")},
            )
            resp.raise_for_status()
            result = Prediction.model_validate(resp.json())
            if result.sample_id != sample.sample_id or result.model_version != self.model["version"]:
                raise ValueError("被测系统响应的样本或模型版本不一致")
            return result


def build_adapter(cfg, output_dir, observer=None):
    model = cfg["models"][cfg["model"]]
    if model["adapter"] == "http_pipeline":
        return HTTPPipeline(model, cfg["execution"]["timeout_seconds"])
    return LocalPipeline(
        build_detector(cfg["model"], cfg),
        model_version=model["version"],
        data_root=cfg["dataset"]["root"],
        output_dir=output_dir,
        ocr_endpoint=os.getenv(cfg["ocr"]["endpoint_env"]),
        ocr_timeout=cfg["ocr"]["timeout_seconds"],
        observer=observer,
    )
