import numpy as np, pandas as pd
from common import (guard_threshold, marginal_lb_offset, conformal_rank, split_units,
                    snapshot, alert_metrics, load_fleet)


def test_conformal_rank():
    assert conformal_rank(9, 0.1) == 9
    assert conformal_rank(8, 0.1) is None       # fewer than 9 units -> no 90% guarantee
    assert conformal_rank(99, 0.1) == 90


def test_guard_miss_rate_is_bounded():
    # exchangeable at-risk scores: realised miss rate must average <= alpha
    rng = np.random.default_rng(0)
    for n in (9, 20, 50):
        misses = []
        for _ in range(4000):
            s = rng.gamma(2, 10, size=n + 1)
            t = guard_threshold(s[:n], 0.1)
            misses.append(s[n] > t)
        assert np.mean(misses) <= 0.1 + 0.015, (n, np.mean(misses))
        assert np.mean(misses) >= 0.1 - 1 / (n + 1) - 0.015   # and not wastefully conservative


def test_guard_too_few_units_alerts_always():
    assert guard_threshold([1, 2, 3], 0.1) == np.inf


def test_marginal_lb_coverage():
    rng = np.random.default_rng(1)
    cov = []
    for _ in range(3000):
        y = rng.uniform(0, 100, 31); p = y + rng.normal(0, 10, 31)
        q = marginal_lb_offset(p[:30], y[:30], 0.1)
        cov.append(y[30] >= p[30] - q)
    assert abs(np.mean(cov) - 0.9) < 0.03


def test_split_units_disjoint_and_complete():
    u = np.arange(1, 101)
    a, b, c = split_units(u, [0.5, 0.25, 0.25], np.random.default_rng(0))
    assert len(set(a) & set(b)) == len(set(a) & set(c)) == len(set(b) & set(c)) == 0
    assert sorted(np.concatenate([a, b, c])) == list(u)


def test_snapshot_one_row_per_unit_and_at_risk():
    df = pd.DataFrame({"unit": np.repeat([1, 2, 3], 50), "RUL": np.tile(np.arange(49, -1, -1), 3)})
    s = snapshot(df, np.random.default_rng(0), at_risk_w=10)
    assert sorted(s.unit) == [1, 2, 3] and (s.RUL <= 10).all()


def test_alert_metrics():
    m = alert_metrics([5, 10, 40, 80], [False, True, True, False], 30)
    assert m["miss"] == 0.5 and m["false_alarm"] == 0.5


def test_cmapss_rul_labels():
    tr, te = load_fleet("FD001")
    assert tr.groupby("unit").RUL.min().eq(0).all()
    last = te.groupby("unit").tail(1).RUL.values
    ref = pd.read_csv("data/cmapss/RUL_FD001.txt", header=None)[0].values
    assert (last == ref).all()
