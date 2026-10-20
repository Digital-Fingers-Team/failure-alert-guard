"""C-MAPSS alert experiments.

dev : uses only the official *training* files. Units are split fit / cal / eval;
      eval units give one at-risk snapshot (miss rate) and one uniform snapshot
      (false alarms). Shift targets are evaluated on their own training units.
test: model fitted on 70% of the source training units, calibrated on 30%;
      evaluated once on the official test files of every target fleet.
In both modes, "recal-k" re-calibrates the guard on k run-to-failure units
of the target fleet (drawn from the target training file, never from eval units).

usage: python experiments_cmapss.py dev|test REPS OUT.csv
"""
import sys, time
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from common import (load_fleet, features, split_units, snapshot, guard_threshold,
                    marginal_lb_offset, alert_metrics, CAP)

FLEETS = ["FD001", "FD002", "FD003", "FD004"]
WINDOWS = [20, 30, 50]
ALPHAS = [0.05, 0.10, 0.20]
KS = [5, 10, 20, 40]
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, n_jobs=2, tree_method="hist")


def fit_models(X, y, seed, alpha_lo):
    point = XGBRegressor(random_state=seed, **XGB).fit(X, y)
    lo = {a: XGBRegressor(random_state=seed, objective="reg:quantileerror",
                          quantile_alpha=a, **XGB).fit(X, y) for a in alpha_lo}
    return point, lo


def rules(point, lo, cal_all, cal_risk, W, a):
    """Return {method: function(X) -> alert bool array}, all calibrated on source cal."""
    p_all = point.predict(cal_all[0]); y_all = cal_all[1]
    rmse = float(np.sqrt(np.mean((p_all - np.minimum(y_all, CAP)) ** 2)))
    q = marginal_lb_offset(p_all, y_all, a)
    e = marginal_lb_offset(lo[a].predict(cal_all[0]), y_all, a)
    t_guard = guard_threshold(point.predict(cal_risk[0]), a)
    return {
        "point": lambda X: point.predict(X) <= W,
        "margin": lambda X: point.predict(X) <= W + rmse,
        "conformal_lb": lambda X: point.predict(X) - q <= W,
        "cqr_lb": lambda X: lo[a].predict(X) - e <= W,
        "guard": lambda X: point.predict(X) <= t_guard,
    }


def run(mode, reps, out):
    data = {}
    for fd in FLEETS:
        tr, te = load_fleet(fd)
        data[fd] = (tr, features(tr), te, features(te))
        print("loaded", fd, len(tr), len(te), flush=True)
    rows, t0 = [], time.time()
    for rep in range(reps):
        for src in FLEETS:
            rng = np.random.default_rng(10_000 * (mode == "test") + 100 * rep + FLEETS.index(src))
            tr, Xtr, _, _ = data[src]
            if mode == "dev":
                fit_u, cal_u, ev_u = split_units(tr.unit.unique(), [0.5, 0.25, 0.25], rng)
            else:
                fit_u, cal_u = split_units(tr.unit.unique(), [0.7, 0.3], rng); ev_u = None
            m = tr.unit.isin(fit_u)
            point, lo = fit_models(Xtr[m], tr.RUL[m].clip(upper=CAP), rep, ALPHAS)
            cal = tr[tr.unit.isin(cal_u)]
            ca = snapshot(cal, rng); cr = {W: snapshot(cal, rng, W) for W in WINDOWS}
            cal_all = (Xtr.loc[ca.index], ca.RUL.values)
            for tgt in FLEETS:
                ttr, tXtr, tte, tXte = data[tgt]
                # evaluation snapshots and the pool of target units usable for recalibration
                if mode == "dev":
                    if tgt == src:
                        ev, pool = tr[tr.unit.isin(ev_u)], None
                    else:
                        r_u, e_u = split_units(ttr.unit.unique(), [0.5, 0.5], rng)
                        ev, pool = ttr[ttr.unit.isin(e_u)], ttr[ttr.unit.isin(r_u)]
                    evX = tXtr if tgt != src else Xtr
                    ev_sets = {W: snapshot(ev, rng, W) for W in WINDOWS}
                    ev_uni = snapshot(ev, rng)
                else:
                    pool = ttr if tgt != src else None
                    evX = tXte
                    last = tte.groupby("unit").tail(1)
                    ev_sets = {W: last for W in WINDOWS}; ev_uni = last
                for W in WINDOWS:
                    cal_risk = (Xtr.loc[cr[W].index], cr[W].RUL.values)
                    # dev: at-risk snapshots give misses, uniform ones false alarms;
                    # test: the official last cycles serve both
                    risk_X, risk_y = evX.loc[ev_sets[W].index], ev_sets[W].RUL.values
                    uni_X, uni_y = evX.loc[ev_uni.index], ev_uni.RUL.values
                    for a in ALPHAS:
                        R = rules(point, lo, cal_all, cal_risk, W, a)
                        if pool is not None:
                            for k in KS:
                                pu = rng.choice(pool.unit.unique(), size=k, replace=False)
                                s = snapshot(pool[pool.unit.isin(pu)], rng, W)
                                t = guard_threshold(point.predict(tXtr.loc[s.index]), a)
                                R[f"recal{k}"] = (lambda t: lambda X: point.predict(X) <= t)(t)
                        for name, f in R.items():
                            mr = alert_metrics(risk_y, f(risk_X), W)
                            mu = alert_metrics(uni_y, f(uni_X), W)
                            rows.append(dict(mode=mode, rep=rep, src=src, tgt=tgt, W=W, alpha=a,
                                            method=name, n_risk=mr["n_risk"],
                                            misses=round(mr["miss"] * mr["n_risk"]),
                                            n_safe=mu["n"] - mu["n_risk"],
                                            false_alarms=round(mu["false_alarm"] * (mu["n"] - mu["n_risk"]))))
            print(f"rep {rep} {src} done {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
