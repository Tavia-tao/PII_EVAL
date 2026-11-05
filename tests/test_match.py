"""人工核算的匹配契约：错误边界必须同时计入 FP/FN，exact 不被早到的 partial 抢占。"""

import pytest

CASES = [
    ([], [], (0, 0, 0, 0, 0)),
    ([(0, 3, "NAME")], [], (0, 1, 0, 0, 0)),
    ([], [(0, 3, "NAME")], (0, 0, 1, 0, 0)),
    ([(0, 3, "NAME")], [(0, 3, "NAME")], (1, 0, 0, 0, 0)),
    ([(0, 3, "PHONE")], [(0, 3, "NAME")], (0, 1, 1, 0, 1)),
    ([(0, 2, "NAME")], [(0, 3, "NAME")], (0, 1, 1, 1, 0)),
    ([(0, 4, "NAME")], [(0, 3, "NAME")], (0, 1, 1, 1, 0)),
    ([(1, 3, "NAME")], [(0, 3, "NAME")], (0, 1, 1, 1, 0)),
    ([(0, 1, "NAME")], [(0, 4, "NAME")], (0, 1, 1, 0, 0)),
    ([(0, 2, "NAME")], [(2, 4, "NAME")], (0, 1, 1, 0, 0)),
    ([(0, 2, "NAME")], [(1, 3, "NAME")], (0, 1, 1, 0, 0)),
    ([(0, 3, "NAME")], [(1, 4, "NAME")], (0, 1, 1, 1, 0)),
    ([(0, 2, "NAME"), (2, 4, "NAME")], [(0, 4, "NAME")], (0, 2, 1, 1, 0)),
    ([(0, 4, "NAME")], [(0, 2, "NAME"), (2, 4, "NAME")], (0, 1, 2, 1, 0)),
    ([(0, 3, "NAME"), (0, 3, "NAME")], [(0, 3, "NAME")], (1, 1, 0, 0, 0)),
    ([(0, 3, "NAME")], [(0, 3, "NAME"), (0, 3, "NAME")], (1, 0, 1, 0, 0)),
    ([(0, 4, "NAME"), (0, 3, "NAME")], [(0, 3, "NAME")], (1, 1, 0, 0, 0)),
    ([(0, 3, "PHONE"), (0, 3, "NAME")], [(0, 3, "NAME")], (1, 1, 0, 0, 1)),
    ([(2, 4, "PHONE"), (0, 6, "ADDRESS")], [(0, 6, "ADDRESS"), (2, 4, "PHONE")], (2, 0, 0, 0, 0)),
    ([(0, 6, "ADDRESS")], [(0, 6, "ADDRESS"), (2, 4, "PHONE")], (1, 0, 1, 0, 0)),
    ([(4, 6, "NAME"), (0, 2, "NAME")], [(0, 2, "NAME"), (4, 6, "NAME")], (2, 0, 0, 0, 0)),
    ([(0, 5, "NAME")], [(1, 5, "PHONE")], (0, 1, 1, 0, 1)),
    ([(0, 5, "NAME")], [(6, 8, "PHONE")], (0, 1, 1, 0, 0)),
    ([(0, 1, "NAME")], [(0, 2, "NAME")], (0, 1, 1, 1, 0)),
    ([(0, 1000, "ADDRESS")], [(0, 1000, "ADDRESS")], (1, 0, 0, 0, 0)),
    ([(1000, 1005, "NAME")], [(1000, 1005, "NAME")], (1, 0, 0, 0, 0)),
    ([(0, 6, "NAME"), (3, 7, "NAME")], [(0, 4, "NAME"), (4, 8, "NAME")], (0, 2, 2, 2, 0)),
    ([(0, 4, "NAME"), (0, 2, "NAME")], [(0, 2, "NAME"), (0, 4, "NAME")], (2, 0, 0, 0, 0)),
    ([(0, 2, "NAME")], [(0, 2, "PHONE"), (0, 2, "NAME")], (1, 0, 1, 0, 0)),
    ([(0, 2, "NAME"), (4, 6, "PHONE")], [(0, 2, "NAME"), (4, 6, "NAME")], (1, 1, 1, 0, 1)),
]


@pytest.mark.parametrize("pred,gold,want", CASES)
def test_strict_matching(pred, gold, want):
    from src.match import Span, evaluate_spans

    r = evaluate_spans([Span(*s) for s in pred], [Span(*s) for s in gold])
    assert (r.tp, r.fp, r.fn, len(r.partial), len(r.wrong_type)) == want
    assert r.tp + r.fp == len(pred)
    assert r.tp + r.fn == len(gold)


def test_permutation_cannot_change_matching():
    from src.match import Span, evaluate_spans

    p = [Span(0, 4, "NAME"), Span(0, 3, "NAME")]
    g = [Span(0, 3, "NAME")]
    assert evaluate_spans(p, g).counts == evaluate_spans(p[::-1], g).counts


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 2), (3, 2)])
def test_invalid_span_rejected(start, end):
    from src.match import Span

    with pytest.raises(ValueError):
        Span(start, end, "NAME")
