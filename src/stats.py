from collections import defaultdict

import numpy as np


def _f1(values):
    counts = np.sum(values, axis=0)
    den = 2 * counts[0] + counts[1] + counts[2]
    return float(2 * counts[0] / den) if den else 0.0


def bootstrap_ci(rows, n_boot=1000, alpha=0.05, seed=42):
    if not rows or n_boot < 10 or not 0 < alpha < 1:
        raise ValueError("bootstrap 需要非空样本，至少10次重采样，alpha在(0,1)")
    groups = defaultdict(list)
    for row in rows:
        groups[row["layout"]].append([row[k] for k in ["tp", "fp", "fn"]])
    rng = np.random.default_rng(seed)
    draws = np.zeros((n_boot, 3))
    for key in sorted(groups):
        a = np.asarray(groups[key], dtype=float)
        # 逐次采样防止 16k×1000×多指标的矩阵占用过大。
        for b in range(n_boot):
            draws[b] += a[rng.integers(0, len(a), len(a))].sum(axis=0)
    den = 2 * draws[:, 0] + draws[:, 1] + draws[:, 2]
    values = np.divide(2 * draws[:, 0], den, out=np.zeros(n_boot), where=den != 0)
    return {
        "point": _f1([[r[k] for k in ["tp", "fp", "fn"]] for r in rows]),
        "lo": float(np.quantile(values, alpha / 2)),
        "hi": float(np.quantile(values, 1 - alpha / 2)),
        "n": len(rows),
        "n_boot": n_boot,
        "seed": seed,
        "stratify": "layout",
    }


def paired_compare(a, b, n_boot=1000, seed=42):
    ia, ib = {r["sample_id"]: r for r in a}, {r["sample_id"]: r for r in b}
    if len(ia) != len(a) or len(ib) != len(b) or ia.keys() != ib.keys() or not ia:
        raise ValueError("配对比较要求相同且唯一的样本ID")
    strata = defaultdict(list)
    for sid in sorted(ia):
        if ia[sid]["layout"] != ib[sid]["layout"]:
            raise ValueError("配对样本版式不一致")
        strata[ia[sid]["layout"]].append(sid)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        ids = [g[i] for key in sorted(strata) for g in [strata[key]] for i in rng.integers(0, len(g), len(g))]
        values.append(
            _f1([[ia[s][k] for k in ["tp", "fp", "fn"]] for s in ids])
            - _f1([[ib[s][k] for k in ["tp", "fp", "fn"]] for s in ids])
        )
    point = _f1([[r[k] for k in ["tp", "fp", "fn"]] for r in a]) - _f1(
        [[r[k] for k in ["tp", "fp", "fn"]] for r in b]
    )
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n": len(a),
        "n_boot": n_boot,
        "significant": bool(lo > 0 or hi < 0),
        "direction": "A>B" if lo > 0 else "B>A" if hi < 0 else "uncertain",
    }


def paired_rate_change(a, b, n_boot=1000, seed=42):
    """同文档误报计数的配对分层重采样，同时报告相对降幅区间。"""
    ia, ib = {r["sample_id"]: r for r in a}, {r["sample_id"]: r for r in b}
    if not ia or ia.keys() != ib.keys() or len(ia) != len(a) or len(ib) != len(b):
        raise ValueError("误报比较需要唯一、完整配对样本")
    strata = defaultdict(list)
    for sid in sorted(ia):
        strata[ia[sid]["layout"]].append(sid)
    rng = np.random.default_rng(seed)
    deltas, reductions = [], []
    for _ in range(n_boot):
        ids = [g[i] for key in sorted(strata) for g in [strata[key]] for i in rng.integers(0, len(g), len(g))]
        aa, bb = sum(ia[s]["count"] for s in ids), sum(ib[s]["count"] for s in ids)
        deltas.append((aa - bb) / len(ids))
        if aa:
            reductions.append((aa - bb) / aa)
    aa, bb = sum(r["count"] for r in a), sum(r["count"] for r in b)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "n": len(a),
        "A_count": aa,
        "B_count": bb,
        "mean_delta": (aa - bb) / len(a),
        "delta_ci": [float(lo), float(hi)],
        "relative_reduction": (aa - bb) / aa if aa else None,
        "reduction_ci": [float(v) for v in np.quantile(reductions, [0.025, 0.975])] if reductions else None,
        "significant": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }
