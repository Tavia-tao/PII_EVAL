"""替代实验：真实重跑 S2、S3 单独及联合替换，差值不强行分摊为可加贡献。"""

import copy
from pathlib import Path

from src.adapter.base import Overrides, SUTAdapter
from src.adapter.factory import build_adapter
from src.dataset import load_dataset
from src.io import write_json
from src.runner import run_evaluation
from src.stats import paired_compare


class InterventionAdapter(SUTAdapter):
    def __init__(self, inner, variant):
        self.inner, self.variant = inner, variant

    async def predict(self, sample, overrides=None):
        override = Overrides(
            ocr_text=sample.gold.ocr_text if "ocr" in self.variant else None,
            layout_blocks=sample.gold.layout_blocks if "layout" in self.variant else None,
        )
        return await self.inner.predict(sample, override)


async def run_attribution(cfg, output_dir, splits=None, allow_holdout=False):
    output_dir = Path(output_dir)
    cfg = copy.deepcopy(cfg)
    cfg["execution"]["cache_enabled"] = False
    samples, _ = load_dataset(cfg["dataset"]["root"])
    selected = [s for s in samples if s.split in (splits or ["gate"]) and s.track == "complex_file"]
    if not selected or any(s.gold.ocr_text is None or not s.gold.layout_blocks for s in selected):
        raise ValueError("归因需要复杂文件金标OCR和布局产物")
    reports = {}
    for name in ["baseline", "ocr", "layout", "ocr_layout"]:
        folder = output_dir / name
        adapter = build_adapter(cfg, folder / "artifacts")
        if name != "baseline":
            adapter = InterventionAdapter(adapter, name)
        reports[name] = await run_evaluation(
            cfg,
            folder,
            splits=splits or ["gate"],
            layouts=sorted({s.layout for s in selected}),
            adapter=adapter,
            allow_holdout=allow_holdout,
        )
    scores = {name: r["tracks"]["complex_file"]["f1"] for name, r in reports.items()}
    if any(score is None for score in scores.values()):
        raise ValueError("归因实验没有有效F1")
    result = {
        "scores": scores,
        "ocr_contribution": scores["ocr"] - scores["baseline"],
        "layout_contribution": scores["layout"] - scores["baseline"],
        "pipeline_total_gap": scores["ocr_layout"] - scores["baseline"],
        "model_gap": 1 - scores["ocr_layout"],
        "note": "单独贡献存在交互，不能相加；负值如实保留。只针对本次样本和输入产物。",
        "paired_ci": {
            name: paired_compare(
                reports[name]["rows"], reports["baseline"]["rows"], n_boot=cfg["execution"]["bootstrap"]
            )
            for name in ["ocr", "layout", "ocr_layout"]
        },
    }
    write_json(output_dir / "attribution.json", result)
    return result
