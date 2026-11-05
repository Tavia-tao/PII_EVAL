"""DeepEval 契约，所有数字复用 match.py，不调用 LLM。"""

import json

from deepeval.metrics import BaseMetric

from src.match import Span, compute_metrics, evaluate_spans
from src.schema import HIGH_RISK


def load_spans(payload):
    raw = json.loads(payload) if isinstance(payload, str) else payload
    values = raw["spans"] if isinstance(raw, dict) else raw
    return [Span(v["start"], v["end"], v["type"]) for v in values]


class PIIF1Metric(BaseMetric):
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.score, self.success, self.reason, self.details = None, False, "", {}
        self.async_mode = True
        self.strict_mode = False
        self.evaluation_cost = 0
        self.evaluation_model = "deterministic-span"
        self.verbose_mode = False
        self.error = None

    def measure(self, test_case, *args, **kwargs):
        r = evaluate_spans(load_spans(test_case.actual_output), load_spans(test_case.expected_output))
        self.details = {**r.counts, **compute_metrics(**r.counts), "partial": len(r.partial)}
        # 空实体用例无F1；作为“无错误”断言返回1，报表仍为 null，不混入门禁。
        self.score = self.details["f1"] if self.details["f1"] is not None else 1.0
        self.success = self.score >= self.threshold
        self.reason = json.dumps(self.details)
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return bool(self.success)

    @property
    def __name__(self):
        return "PII Span F1"


class HighRiskRecallMetric(PIIF1Metric):
    def __init__(self, high_risk_types=HIGH_RISK, threshold=0.98):
        super().__init__(threshold)
        self.types = tuple(high_risk_types)

    def measure(self, test_case, *args, **kwargs):
        ps, gs = load_spans(test_case.actual_output), load_spans(test_case.expected_output)
        recalls = {}
        for t in self.types:
            gold = [g for g in gs if g.type == t]
            if gold:
                r = evaluate_spans([p for p in ps if p.type == t], gold)
                recalls[t] = r.tp / len(gold)
        self.score = sum(recalls.values()) / len(recalls) if recalls else 1.0
        self.success = min(recalls.values(), default=1.0) >= self.threshold
        self.details = {"recall_by_type": recalls, "not_present": [t for t in self.types if t not in recalls]}
        self.reason = json.dumps(self.details)
        return self.score

    @property
    def __name__(self):
        return "High-Risk Recall"


class BoundaryMetric(PIIF1Metric):
    def measure(self, test_case, *args, **kwargs):
        r = evaluate_spans(load_spans(test_case.actual_output), load_spans(test_case.expected_output))
        self.score = r.tp / (r.tp + len(r.partial)) if r.tp + len(r.partial) else 0.0
        self.success = self.score >= self.threshold
        self.details = {"exact": r.tp, "partial": len(r.partial)}
        self.reason = json.dumps(self.details)
        return self.score

    @property
    def __name__(self):
        return "PII Boundary Accuracy"
