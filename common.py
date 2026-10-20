"""Shared code: C-MAPSS loading, features, snapshot sampling and alert rules.

An *alert rule* turns a model score into "service now" / "keep running".
A unit is *at risk* at a snapshot when its true RUL <= W (it would fail
before the next scheduled inspection). A *miss* is an at-risk unit with no alert.
"""
import math
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

CMAPSS = "data/cmapss"
CAP = 125          # piecewise-linear RUL target cap (standard for C-MAPSS)
COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
# sensors that carry degradation signal in all four sub-datasets
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
N_COND = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def _read(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def load_fleet(fd):
    """Return (train, test) for one sub-dataset; both carry a true 'RUL' column."""
    tr, te = _read(f"{CMAPSS}/train_{fd}.txt"), _read(f"{CMAPSS}/test_{fd}.txt")
    tr["RUL"] = tr.groupby("unit")["cycle"].transform("max") - tr["cycle"]
    final = pd.read_csv(f"{CMAPSS}/RUL_{fd}.txt", header=None)[0].values
    last = te.groupby("unit")["cycle"].transform("max")
    te["RUL"] = last - te["cycle"] + final[te["unit"].values - 1]
    return normalise(tr, te, N_COND[fd])


def normalise(tr, te, k):
    """Z-score sensors per operating condition, with stats from the fleet's own
    data. Uses no failure labels, so it is available for a newly deployed fleet."""
    both = pd.concat([tr, te], keys=["tr", "te"])
    if k > 1:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(both[["os1", "os2", "os3"]].round(1))
        both["cond"] = km.labels_
    else:
        both["cond"] = 0
    g = both.groupby("cond")[SENSORS]
    mu, sd = g.transform("mean"), g.transform("std").replace(0, 1)
    both[SENSORS] = ((both[SENSORS] - mu) / sd).fillna(0)
    return both.loc["tr"].reset_index(drop=True), both.loc["te"].reset_index(drop=True)


def features(df, w=10):
    """Current reading, rolling mean and rolling slope per sensor, plus cycle."""
    g = df.groupby("unit")[SENSORS]
    mean = g.transform(lambda s: s.rolling(w, min_periods=1).mean()).add_suffix("_m")
    t = df["cycle"].astype(float)
    # slope of the last 2w cycles via rolling covariance / variance
    def slope(s):
        x = t.loc[s.index]
        cov = s.rolling(2 * w, min_periods=3).cov(x)
        var = x.rolling(2 * w, min_periods=3).var()
        return (cov / var).fillna(0)
    sl = g.transform(slope).add_suffix("_d")
    X = pd.concat([df[SENSORS], mean, sl], axis=1)
    X["cycle"] = df["cycle"]
    return X


def split_units(units, fracs, rng):
    u = np.array(sorted(units)); rng.shuffle(u)
    cuts = np.cumsum([int(round(f * len(u))) for f in fracs[:-1]])
    return np.split(u, cuts)


def snapshot(df, rng, at_risk_w=None):
    """One row per unit: uniform over the unit's life, or (if at_risk_w) uniform
    over the cycles where RUL <= at_risk_w. One row per unit keeps units exchangeable."""
    d = df if at_risk_w is None else df[df["RUL"] <= at_risk_w]
    idx = d.groupby("unit").apply(lambda g: rng.choice(g.index.values), include_groups=False)
    return df.loc[idx.values]


# ---------------- alert rules ----------------
# Every rule returns a threshold t: alert iff score <= t (score = predicted RUL).

def conformal_rank(n, alpha):
    """Rank k (1-based) of the finite-sample conformal quantile; inf if n too small."""
    k = math.ceil((n + 1) * (1 - alpha))
    return k if k <= n else None


def guard_threshold(pred_at_risk, alpha):
    """Label-conditional conformal threshold: with exchangeable at-risk units,
    P(pred > t | at risk) <= alpha. Needs n >= 1/alpha - 1 at-risk units."""
    s = np.sort(np.asarray(pred_at_risk))
    k = conformal_rank(len(s), alpha)
    return np.inf if k is None else s[k - 1]


def marginal_lb_offset(pred, y, alpha):
    """Split-conformal one-sided lower bound: y >= pred - q with prob >= 1-alpha
    marginally over all snapshots (not conditionally on being at risk)."""
    r = np.sort(np.asarray(pred) - np.asarray(y))
    k = conformal_rank(len(r), alpha)
    return np.inf if k is None else r[k - 1]


def alert_metrics(y, alert, W):
    y, alert = np.asarray(y), np.asarray(alert, bool)
    risk = y <= W
    return dict(n=len(y), n_risk=int(risk.sum()),
                miss=float((~alert & risk).sum() / max(risk.sum(), 1)),
                false_alarm=float((alert & ~risk).sum() / max((~risk).sum(), 1)),
                alerts=int(alert.sum()))
