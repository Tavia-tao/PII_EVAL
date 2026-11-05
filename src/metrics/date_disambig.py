"""日期裁决只产出待人工复核建议，不覆盖已冻结金标。"""

import math
import re


class DateResolver:
    def __init__(self, judge=None, total_samples=0, max_ratio=0.04, threshold=0.8):
        if not 0 <= max_ratio < 0.05:
            raise ValueError("judge 调用比例必须小于5%")
        self.judge, self.threshold = judge, threshold
        self.budget = math.floor(total_samples * max_ratio)
        self.calls = 0
        self.enabled = judge is not None
        self.audit = []

    async def resolve(self, context, value):
        if re.search(r"出生|生日|入职|证件有效期|birth|employment", context, re.I):
            return {"label": "pii", "source": "rule", "needs_human_review": False}
        if re.search(r"合同|报表|发布|日志|季度|contract|report|release|log", context, re.I):
            return {"label": "not_pii", "source": "rule", "needs_human_review": False}
        if not self.enabled or self.calls >= self.budget:
            return {"label": "uncertain", "source": "budget_or_disabled", "needs_human_review": True}
        self.calls += 1  # 在await前预留，异步并发不能超预算。
        try:
            result = await self.judge(context, value)
            if result["label"] not in {"pii", "not_pii", "uncertain"}:
                raise ValueError("未知judge标签")
            if result.get("confidence", 0) < self.threshold:
                result = {**result, "label": "uncertain"}
            output = {**result, "source": "judge", "needs_human_review": True}
        except Exception as e:
            output = {
                "label": "uncertain",
                "source": "judge_error",
                "error": type(e).__name__,
                "needs_human_review": True,
            }
        self.audit.append({"value": value, "decision": output})
        return output

    def calibrate(self, matches):
        agreement = sum(matches) / len(matches) if matches else None
        if agreement is None or agreement < 0.8:
            self.enabled = False
        return {"agreement": agreement, "enabled": self.enabled, "n": len(matches)}


def build_date_judge(model):
    """DeepEval GEval 作为语义参考，调用方必须另行抽检。"""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name="Date PII Disambiguation",
        criteria="Only dates tied to a natural person are PII; business dates are not. "
        "If attribution is unknown, do not assume a person. Judge the proposed date classification.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=model,
    )
