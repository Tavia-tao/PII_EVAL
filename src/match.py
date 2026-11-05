"""唯一 span 匹配口径。Python Unicode 码点，下标左闭右开；partial 也是 FP+FN。"""

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    type: str

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start or not self.type:
            raise ValueError("span 必须满足 0 <= start < end 且类型非空")


def iou(a: Span, b: Span) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    return inter / (a.end - a.start + b.end - b.start - inter) if inter else 0.0


def match_one(pred: Span, gold: Span, iou_thr: float = 0.5) -> str:
    if pred.type != gold.type:
        return "wrong_type" if iou(pred, gold) > 0 else "none"
    if pred == gold:
        return "exact"
    return "partial" if iou(pred, gold) >= iou_thr else "none"


@dataclass
class MatchResult:
    exact: list[tuple[Span, Span]] = field(default_factory=list)
    partial: list[tuple[Span, Span]] = field(default_factory=list)
    false_positives: list[Span] = field(default_factory=list)
    false_negatives: list[Span] = field(default_factory=list)
    wrong_type: list[Span] = field(default_factory=list)

    @property
    def tp(self):
        return len(self.exact)

    @property
    def fp(self):
        return len(self.false_positives)

    @property
    def fn(self):
        return len(self.false_negatives)

    @property
    def counts(self):
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn}


def evaluate_spans(preds: list[Span], golds: list[Span], iou_thr: float = 0.5) -> MatchResult:
    if not 0 < iou_thr <= 1:
        raise ValueError("IoU 阈值必须在 (0,1]")

    def key(s):
        return s.end - s.start, s.start, s.end, s.type

    ps, gs = sorted(preds, key=key), sorted(golds, key=key)
    index = defaultdict(deque)
    for j, g in enumerate(gs):
        index[g].append(j)
    result = MatchResult()
    used_p, used_g = set(), set()
    # 必须先为所有 exact 预留配对，避免 partial 或 wrong_type 抢占。
    for i, p in enumerate(ps):
        if index[p]:
            j = index[p].popleft()
            result.exact.append((p, gs[j]))
            used_p.add(i)
            used_g.add(j)
    pp = [p for i, p in enumerate(ps) if i not in used_p]
    gg = [g for j, g in enumerate(gs) if j not in used_g]
    result.false_positives, result.false_negatives = pp, gg
    # 剩余配对只用于诊断。最大配对数优先，其次最大 IoU；不影响严格 TP/FP/FN。
    if pp and gg:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        scores = np.zeros((len(pp), len(gg)))
        bonus = min(len(pp), len(gg)) + 1
        for i, p in enumerate(pp):
            for j, g in enumerate(gg):
                overlap = iou(p, g)
                if p.type == g.type and overlap >= iou_thr:
                    scores[i, j] = bonus + overlap
        rows, cols = linear_sum_assignment(scores, maximize=True)
        result.partial = [(pp[i], gg[j]) for i, j in zip(rows, cols, strict=True) if scores[i, j] > 0]
    result.wrong_type = [p for p in pp if any(p.type != g.type and iou(p, g) > 0 for g in gs)]
    return result


def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }
