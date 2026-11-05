from datetime import datetime, timedelta


def detect_drift(points, baseline, teacher_max_ratio=0.3, config=None):
    from src.config import load_config

    cfg = config or load_config()["monitor"]
    points = sorted(points, key=lambda p: p["timestamp"])
    if not points:
        return []
    alerts = []

    def alert(metric, reason, value=None):
        alerts.append(
            {"metric": metric, "reason": reason, "value": value, "timestamp": points[-1]["timestamp"]}
        )

    latest = points[-1]
    # 只接受连续日且每天足额人工抽检的数据，不能用无金标线上预测推算召回。
    valid = [
        p
        for p in points
        if p.get("audited_n", 0) >= cfg["min_daily_audit"] and p.get("high_risk_recall") is not None
    ]
    days = cfg["recall_days"]
    daily = {datetime.fromisoformat(p["timestamp"]).date(): p for p in valid}
    if len(daily) >= days + 1:
        dates = sorted(daily)[-days - 1 :]
        if all(
            dates[i] - dates[i - 1] == timedelta(days=1)
            and daily[dates[i - 1]]["high_risk_recall"] - daily[dates[i]]["high_risk_recall"]
            > cfg["recall_daily_drop"]
            for i in range(1, len(dates))
        ):
            alert("high_risk_recall", "连续三日人工抽检召回下降超过阈值", latest.get("high_risk_recall"))
    for metric, multiplier in [
        ("false_delete", cfg["false_delete_multiplier"]),
        ("token_cost", cfg["cost_multiplier"]),
    ]:
        value, base = latest.get(metric), baseline.get(metric)
        if value is not None and base is not None and value > base * multiplier:
            alert(metric, "超过基线/预算倍数", value)
    # 以连续观测窗口判断持续1小时，出现正常点即打断。
    high = []
    base_latency = baseline.get("latency_p95")
    if base_latency is not None:
        for p in reversed(points):
            if p.get("latency_p95", 0) <= base_latency * cfg["latency_multiplier"]:
                break
            high.append(datetime.fromisoformat(p["timestamp"]))
        if len(high) >= 2 and (high[0] - high[-1]).total_seconds() >= cfg["latency_duration_minutes"] * 60:
            alert("latency_p95", "P95超过基线且持续至少1小时", latest.get("latency_p95"))
    if latest.get("teacher_rate", 0) > teacher_max_ratio:
        alert("teacher_rate", "Teacher路由比例超过预算", latest["teacher_rate"])
    if latest.get("parse_failure_rate", 0) > cfg["parse_failure_max"]:
        alert("parse_failure_rate", "解析/导出失败率超过阈值", latest["parse_failure_rate"])
    return alerts


def gate_health(reports):
    """季度健康检查：连续满分类别失去区分度；报告轮换建议，不自动篡改数据。"""
    counts = {}
    for r in reports:
        for track in r["tracks"].values():
            for t, v in track.get("by_type", {}).items():
                counts.setdefault(t, []).append(v.get("f1"))
    return {
        "rotate_candidates": [
            t for t, values in counts.items() if len(values) >= 3 and all(v == 1 for v in values)
        ],
        "recommendation": "独立复核新难例，创建新数据版本后重新计算基线",
    }
