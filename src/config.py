import copy
from pathlib import Path

import yaml

from src.schema import ENTITY_TYPES

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "conf" / "eval.yaml"


def load_config(path=None):
    default = (
        DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else Path(__file__).with_name("defaults") / "eval.yaml"
    )
    with Path(path or default).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if cfg["schema_version"] != 1:
        raise ValueError("不支持的配置版本")
    if not cfg["rules"]["exact_match_only"]:
        raise ValueError("主口径必须使用 exact")
    if not 0 < cfg["rules"]["iou_threshold"] <= 1:
        raise ValueError("IoU阈值错误")
    if set(cfg["rules"]["weights"]) != ENTITY_TYPES:
        raise ValueError("权重必须覆盖全部实体类别")
    if any(v <= 0 for v in cfg["rules"]["weights"].values()):
        raise ValueError("权重必须为正数")
    if not set(cfg["rules"]["high_risk_types"]) <= ENTITY_TYPES:
        raise ValueError("未知高风险实体")
    for name in ["concurrency", "shard_size", "timeout_seconds", "bootstrap"]:
        if cfg["execution"][name] <= 0:
            raise ValueError(f"{name}必须为正")
    if not 0 <= cfg["judge"]["max_ratio"] < 0.05:
        raise ValueError("judge预算必须小于5%")
    return cfg
