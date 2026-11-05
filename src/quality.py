import math
import random
from collections import Counter


def cohen_kappa(a, b):
    if not a or len(a) != len(b):
        raise ValueError("双人标注必须基于同一组候选且非空")
    n = len(a)
    pa, pb = Counter(a), Counter(b)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum(pa[t] * pb[t] for t in set(a) | set(b)) / n**2
    return (observed - expected) / (1 - expected) if expected < 1 else None


def annotation_agreement(a, b, threshold=0.85):
    k = cohen_kappa(a, b)
    return {
        "n": len(a),
        "kappa": k,
        "passed": k is not None and k >= threshold,
        "disagreements": [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y],
        "scope": "预先枚举候选的类别一致性（包括 NOT_PII），非开放实体集合伪Kappa",
    }


def quality_sample_plan(samples, seed=42):
    rng = random.Random(seed)
    groups = {}
    for s in samples:
        groups.setdefault((s.track, s.layout), []).append(s.sample_id)
    plans = {"double_blind_20pct": [], "expert_5pct": [], "teacher_10pct": []}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        for name, rate in [("double_blind_20pct", 0.2), ("expert_5pct", 0.05), ("teacher_10pct", 0.1)]:
            plans[name].extend(ids[: math.ceil(len(ids) * rate)])
    return {
        "seed": seed,
        "n": len(samples),
        "assignments": plans,
        "thresholds": {"kappa": 0.85, "expert_error_max": 0.01, "teacher_disagreement_resolved": 1.0},
        "note": "这是待执行分配计划，不是已完成的人工质检证明。",
    }
