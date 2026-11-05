from collections import defaultdict
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from src.io import read_json
from src.match import Span, compute_metrics, evaluate_spans, iou
from src.schema import HIGH_RISK, STAGES, WEIGHTS


def align_spans(spans, observed: str, reference: str) -> list[Span]:
    if not observed or observed == reference:
        return [Span(s.start, s.end, s.type) for s in spans]
    mapping = {}
    for tag, a, b, c, d in SequenceMatcher(None, observed, reference, autojunk=False).get_opcodes():
        if tag in {"equal", "replace"}:
            for i in range(a, b):
                mapping[i] = c + min(i - a, max(d - c - 1, 0)) if d > c else None
    result = []
    for s in spans:
        positions = [mapping.get(i) for i in range(s.start, s.end)]
        positions = [p for p in positions if p is not None]
        # 不能映射的预测保留为超出参考长度的 FP，不可静默丢弃。
        result.append(
            Span(min(positions), max(positions) + 1, s.type)
            if positions
            else Span(len(reference) + s.start, len(reference) + s.end, s.type)
        )
    return result


def _covered(spans, n):
    out = set()
    for s in spans:
        out.update(range(s.start, min(s.end, n)))
    return out


def edit_distance(a, b):
    if a == b:
        return 0
    # 先去除公共前后缀；常见单字符OCR损伤不需要全矩阵。
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    a, b = a[prefix:], b[prefix:]
    suffix = 0
    while suffix < min(len(a), len(b)) and a[-suffix - 1] == b[-suffix - 1]:
        suffix += 1
    if suffix:
        a, b = a[:-suffix], b[:-suffix]
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def stage_diagnostics(sample, stages):
    ref = sample.gold.ocr_text or sample.gold.text
    artifact = stages.get("S2_ocr", {}).get("artifact")
    if not artifact or not Path(artifact).is_file():
        return
    raw = read_json(artifact)
    observed = raw.get("S2_ocr", {}).get("text")
    if observed is not None and ref:
        cer = edit_distance(observed, ref) / len(ref)
        stages["S2_ocr"]["metrics"].update(cer=cer, reference_chars=len(ref))
    parsed = raw.get("S1_parse", {}).get("text")
    if parsed and ref and "S1_parse" in stages:
        equal = sum(m.size for m in SequenceMatcher(None, parsed, ref, autojunk=False).get_matching_blocks())
        stages["S1_parse"]["metrics"]["content_recall"] = equal / len(ref)
    reference_order = [b.block_id for b in sample.gold.layout_blocks]
    observed_order = raw.get("S3_layout", {}).get("block_order", [])
    common = [bid for bid in observed_order if bid in reference_order]
    if common and reference_order and "S3_layout" in stages:
        rank = {v: i for i, v in enumerate(reference_order)}
        total = len(common) * (len(common) - 1) / 2
        correct = sum(
            rank[common[i]] < rank[common[j]] for i in range(len(common)) for j in range(i + 1, len(common))
        )
        stages["S3_layout"]["metrics"].update(
            reading_order_acc=correct / total if total else 1.0,
            block_coverage=len(common) / len(reference_order),
        )


def evaluate_sample(sample, pred, *, high_risk=HIGH_RISK, weights=None, iou_threshold=0.5):
    weights = weights or WEIGHTS
    if pred.sample_id != sample.sample_id:
        raise ValueError("预测与样本ID不一致")
    if pred.coordinate_text and any(s.end > len(pred.coordinate_text) for s in pred.spans):
        raise ValueError("预测 span 超出实际输入长度")
    ps = align_spans(pred.spans, pred.coordinate_text, sample.gold.text)
    gs = [Span(s.start, s.end, s.type) for s in sample.gold.spans]
    matched = evaluate_spans(ps, gs, iou_threshold)
    per_type = {}
    for t in sorted({s.type for s in ps + gs}):
        r = evaluate_spans([p for p in ps if p.type == t], [g for g in gs if g.type == t], iou_threshold)
        per_type[t] = {**r.counts, "n": r.tp + r.fn, **compute_metrics(**r.counts)}
    negatives = sample.gold.non_pii_candidates or sample.gold.non_pii_hard_negatives
    touched = [n for n in negatives if any(max(n.start, p.start) < min(n.end, p.end) for p in ps)]
    hard_neg = defaultdict(int)
    for n in touched:
        hard_neg[n.reason] += 1
    pred_chars, gold_chars = _covered(ps, len(sample.gold.text)), _covered(gs, len(sample.gold.text))
    total_risk = sum(weights[g.type] for g in gs)
    missed_risk = sum(weights[g.type] for g in matched.false_negatives)
    hr = {t: v for t, v in per_type.items() if t in high_risk and v["n"]}
    stages = {name: stage.model_dump() for name, stage in pred.stages.items()}
    stage_diagnostics(sample, stages)
    errors = [name for name in STAGES if name not in stages or not stages[name]["ok"]]
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "track": sample.track,
        "layout": sample.layout,
        "difficulty": sample.difficulty,
        "lang": sample.lang,
        "doc_type": sample.doc_type,
        "page_count": sample.page_count,
        **matched.counts,
        **compute_metrics(**matched.counts),
        "by_type": per_type,
        "partial": len(matched.partial),
        "wrong_type": len(matched.wrong_type),
        "boundary_acc": matched.tp / (matched.tp + len(matched.partial))
        if matched.tp + len(matched.partial)
        else None,
        "ious": [1.0] * matched.tp + [iou(p, g) for p, g in matched.partial],
        "false_delete_count": len(touched),
        "non_pii_count": len(negatives),
        "false_delete": len(touched) / len(negatives) if negatives else None,
        "false_negative": matched.fn / len(gs) if gs else None,
        "critical_false_delete": sum(n.critical for n in touched),
        "char_false_delete_count": len(pred_chars - gold_chars),
        "char_false_negative_count": len(gold_chars - pred_chars),
        "pii_char_count": len(gold_chars),
        "non_pii_char_count": len(sample.gold.text) - len(gold_chars),
        "risk_num": missed_risk,
        "risk_den": total_risk,
        "risk_score": missed_risk / total_risk if total_risk else None,
        "high_risk_recall": sum(v["recall"] for v in hr.values()) / len(hr) if hr else None,
        "latency_ms": pred.latency_ms,
        "token_cost": pred.usage.cost_usd / sample.page_count,
        "cost_usd": pred.usage.cost_usd,
        "cost_measured": pred.usage.measured,
        "route": pred.route,
        "route_reason": pred.route_reason,
        "needs_review": pred.needs_review,
        "file_openable": bool(
            pred.stages.get("S6_export") and pred.stages["S6_export"].metrics.get("openable")
        ),
        "stages": stages,
        "execution_error": bool(pred.error or errors),
        "error_stages": errors,
        "error": pred.error,
        "cached": pred.cached,
        "trace_id": pred.trace_id,
        "hard_negative_fp": dict(hard_neg),
        "fp_spans": [asdict(s) for s in matched.false_positives],
        "fn_spans": [asdict(s) for s in matched.false_negatives],
        "pred_spans": [asdict(s) for s in ps],
        "gold_spans": [asdict(s) for s in gs],
    }


def aggregate(rows, *, high_risk=HIGH_RISK):
    if not rows:
        return {"n": 0, "f1": None}
    counts = {k: sum(r[k] for r in rows) for k in ["tp", "fp", "fn"]}
    types = sorted({t for r in rows for t in r["by_type"]})
    by_type = {}
    for t in types:
        c = {k: sum(r["by_type"].get(t, {}).get(k, 0) for r in rows) for k in ["tp", "fp", "fn", "n"]}
        by_type[t] = {**c, **compute_metrics(c["tp"], c["fp"], c["fn"])}
    negatives = sum(r["non_pii_count"] for r in rows)
    risk_den = sum(r["risk_den"] for r in rows)
    hrs = [by_type[t] for t in high_risk if t in by_type and by_type[t]["n"]]
    hr_gold = sum(v["n"] for v in hrs)
    partial = sum(r["partial"] for r in rows)
    total_n = sum(v["n"] for v in by_type.values())
    n_pages = sum(r["page_count"] for r in rows)
    ious = [i for r in rows for i in r["ious"]]
    return {
        "n": len(rows),
        **counts,
        **compute_metrics(**counts),
        "by_type": by_type,
        "macro_f1": float(np.mean([v["f1"] for v in by_type.values() if v["f1"] is not None]))
        if by_type
        else None,
        "weighted_f1": sum((v["f1"] or 0) * v["n"] for v in by_type.values()) / total_n if total_n else None,
        "high_risk_recall": sum(v["recall"] for v in hrs) / len(hrs) if hrs else None,
        "high_risk_micro": sum(v["tp"] for v in hrs) / hr_gold if hr_gold else None,
        "high_risk_min": min((v["recall"] for v in hrs), default=None),
        "false_delete": sum(r["false_delete_count"] for r in rows) / negatives if negatives else None,
        "false_delete_denominator": negatives,
        "negative_coverage": sum(r["non_pii_count"] > 0 for r in rows) / len(rows),
        "char_false_delete": sum(r["char_false_delete_count"] for r in rows)
        / max(sum(r["non_pii_char_count"] for r in rows), 1),
        "char_false_negative": sum(r["char_false_negative_count"] for r in rows)
        / max(sum(r["pii_char_count"] for r in rows), 1),
        "false_negative": counts["fn"] / total_n if total_n else None,
        "risk_score": sum(r["risk_num"] for r in rows) / risk_den if risk_den else None,
        "boundary_acc": counts["tp"] / (counts["tp"] + partial) if counts["tp"] + partial else None,
        "partial": partial,
        "wrong_type": sum(r["wrong_type"] for r in rows),
        "iou_quantiles": [float(np.quantile(ious, q)) for q in [0, 0.25, 0.5, 0.75, 1]] if ious else [],
        "latency_p95": float(np.percentile([r["latency_ms"] / 1000 for r in rows], 95)),
        "latency_p99": float(np.percentile([r["latency_ms"] / 1000 for r in rows], 99)),
        "latency_per_page": sum(r["latency_ms"] for r in rows) / 1000 / n_pages,
        "token_cost": sum(r["cost_usd"] for r in rows) / n_pages,
        "cost_measured": all(r["cost_measured"] for r in rows),
        "review_rate": sum(r["needs_review"] for r in rows) / len(rows),
        "teacher_rate": sum(r["route"] in {"teacher", "escalate"} for r in rows) / len(rows),
        "file_openable": sum(r["file_openable"] for r in rows) / len(rows),
        "execution_errors": sum(r["execution_error"] for r in rows),
    }
