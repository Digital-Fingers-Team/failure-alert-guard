"""Backblaze hard-drive alert experiments (real fleet).

Score = classifier probability that the drive fails within W days; alert iff score >= threshold.
dev : P1 only. Drives split fit / cal / eval (50/25/25). Manufacturer shift: one maker is
      held out of fit/cal; half its drives evaluate, the other half is the recalibration pool.
test: fit on P1, calibrate on P2, evaluate on P3 (3 months later). Manufacturer shift: maker
      held out of P1/P2 fitting and calibration, evaluated on its P3 drives, recalibrated
      with k of its P2 failures.
usage: python experiments_bb.py dev|test REPS OUT.csv
"""
import sys, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from common import guard_threshold

ALPHAS = [0.05, 0.10, 0.20]
KS = [10, 20, 40, 80]
BUDGETS = [0.01, 0.05]
SMART_RULE = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]
XGB = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, n_jobs=2, tree_method="hist", eval_metric="logloss")


def maker(m):
    m = m.upper()
    if m.startswith("ST") or m.startswith("SEAGATE"): return "Seagate"
    if m.startswith(("HGST", "HUH", "HMS", "HDS")): return "HGST"
    if m.startswith(("TOSHIBA", "MG", "DT01")): return "Toshiba"
    if m.startswith(("WDC", "WUH", "WD")): return "WDC"
    return "other"


def load():
    s = pd.read_parquet("data/bb/snapshots.parquet")
    s["maker"] = s["model"].map(maker)
    s["cap_tb"] = s["capacity_bytes"].where(s["capacity_bytes"] > 0) / 1e12
    feats = [c for c in s.columns if c.startswith("smart_")] + ["cap_tb"]
    return s, feats


def train(fit, feats, W, seed):
    d = fit[(fit.W == W) & ((fit.kind == "uniform") | (fit.at_risk == 1))]
    return XGBClassifier(random_state=seed, **XGB).fit(d[feats], d.at_risk)


def guard_t(model, risk, feats, a):
    """alert iff score >= t; label-conditional conformal on at-risk calibration drives."""
    if len(risk) == 0:
        return -np.inf
    return -guard_threshold(-model.predict_proba(risk[feats])[:, 1], a)


def thresholds(model, cal, feats, W):
    cu = cal[(cal.W == W) & (cal.kind == "uniform")]
    cr = cal[(cal.W == W) & (cal.kind == "risk") & (cal.at_risk == 1)]
    p, y = model.predict_proba(cu[feats])[:, 1], cu.at_risk.values
    best, bt = -1, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 400))):
        tp = ((p >= t) & (y == 1)).sum(); f1 = 2 * tp / max((p >= t).sum() + y.sum(), 1)
        if f1 > best: best, bt = f1, t
    T = {"default_0.5": 0.5, "f1_opt": bt}
    for a in ALPHAS:
        T[f"guard_{a}"] = guard_t(model, cr, feats, a)
    # budget mode: threshold set by the share of healthy drives the operator can inspect;
    # certificate = conformal bound (m + 1) / (n + 1) on the miss rate from calibration failures
    sr = model.predict_proba(cr[feats])[:, 1]
    for b in BUDGETS:
        t = float(np.quantile(p[y == 0], 1 - b))
        T[f"budget_{b}"] = t
        CERT[f"budget_{b}"] = ((sr < t).sum() + 1) / (len(sr) + 1)
    return T


def evaluate(model, T, ev, feats, W, extra):
    er = ev[(ev.W == W) & (ev.kind == "risk") & (ev.at_risk == 1)]
    eu = ev[(ev.W == W) & (ev.kind == "uniform") & (ev.at_risk == 0)]
    pr, pu = model.predict_proba(er[feats])[:, 1], model.predict_proba(eu[feats])[:, 1]
    out = []
    rule_r = (er[SMART_RULE].fillna(0) > 0).any(axis=1).values
    rule_u = (eu[SMART_RULE].fillna(0) > 0).any(axis=1).values
    alerts = {"smart_rule": (rule_r, rule_u)}
    for name, t in T.items():
        alerts[name] = (pr >= t, pu >= t)
    for name, (ar, au) in alerts.items():
        out.append(dict(extra, W=W, method=name, n_risk=len(er), misses=int((~ar).sum()),
                        n_safe=len(eu), false_alarms=int(au.sum()), cert=CERT.pop(name, np.nan)))
        if extra["mode"] == "test" and W == 30 and not name.endswith(("0.05", "0.2")):
            UNITS.append(pd.DataFrame(dict(extra, method=name, serial=er.serial.values, alert=ar.astype(int))))
    return out


UNITS = []   # per-drive outcomes for at-risk test drives, primary W = 30
CERT = {}    # budget-mode miss-rate certificates from the latest thresholds() call


def split_serials(df, fracs, rng):
    u = np.array(sorted(df.serial.unique())); rng.shuffle(u)
    cuts = np.cumsum([int(round(f * len(u))) for f in fracs[:-1]])
    return [set(x) for x in np.split(u, cuts)]


def recal_rows(model, pool, ev, feats, W, rng, extra):
    rows = []
    pr = pool[(pool.W == W) & (pool.kind == "risk") & (pool.at_risk == 1)]
    for k in KS:
        if len(pr) < k:
            continue
        s = pr.sample(k, random_state=int(rng.integers(1 << 31)))
        T = {f"recal{k}_{a}": guard_t(model, s, feats, a) for a in ALPHAS}
        rows += [r for r in evaluate(model, T, ev, feats, W, extra) if r["method"] != "smart_rule"]
    return rows


def run(mode, reps, out):
    s, feats = load()
    makers = [m for m in ["Seagate", "HGST", "Toshiba", "WDC"]]
    rows, t0 = [], time.time()
    for rep in range(reps):
        rng = np.random.default_rng(20_000 * (mode == "test") + rep)
        for W in (7, 30):
            if mode == "dev":
                p1 = s[s.period == "P1"]
                f, c, e = split_serials(p1, [0.5, 0.25, 0.25], rng)
                fit, cal, ev = p1[p1.serial.isin(f)], p1[p1.serial.isin(c)], p1[p1.serial.isin(e)]
            else:
                fit, cal, ev = s[s.period == "P1"], s[s.period == "P2"], s[s.period == "P3"]
            m = train(fit, feats, W, rep)
            rows += evaluate(m, thresholds(m, cal, feats, W), ev, feats, W,
                             dict(mode=mode, rep=rep, target="all"))
            for mk in makers:
                if mode == "dev":
                    a, b = split_serials(ev[ev.maker == mk], [0.5, 0.5], rng)
                    tev, pool = ev[ev.serial.isin(a)], ev[ev.serial.isin(b)]
                else:
                    tev, pool = ev[ev.maker == mk], cal[cal.maker == mk]
                # in-distribution model on this maker's drives (maker was seen in training)
                rows += evaluate(m, thresholds(m, cal, feats, W), tev, feats, W,
                                 dict(mode=mode, rep=rep, target=mk, held_out=False))
                # maker held out of fitting and calibration
                mh = train(fit[fit.maker != mk], feats, W, rep)
                ex = dict(mode=mode, rep=rep, target=mk, held_out=True)
                rows += evaluate(mh, thresholds(mh, cal[cal.maker != mk], feats, W), tev, feats, W, ex)
                rows += recal_rows(mh, pool, tev, feats, W, rng, ex)
            print(f"rep {rep} W={W} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        if UNITS:
            pd.concat(UNITS).to_parquet(out.replace(".csv", "_units.parquet"), index=False)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
