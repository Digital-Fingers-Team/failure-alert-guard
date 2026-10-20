"""Development only: which feature set gives the cheapest guard on Backblaze P1 (W = 30)?
Reports the false-alarm rate needed for 80% / 90% catch rate and the catch rate at 1% / 5% alarms."""
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from experiments_bb import load, XGB, split_serials
from common import guard_threshold

s, base = load()
p1 = s[(s.period == "P1") & (s.W == 30)].copy()
p1["model_cat"] = p1["model"].astype("category")
p1["age_days"] = p1["smart_9_raw"] / 24
lvl = [c for c in base if not any(c.endswith(x) for x in ("_d7", "_d30"))]
variants = {"current+deltas": base, "current only": lvl, "+model id": base + ["model_cat"]}
for rep in range(2):
    rng = np.random.default_rng(rep)
    f, c, e = split_serials(p1, [0.5, 0.25, 0.25], rng)
    for name, F in variants.items():
        fit = p1[p1.serial.isin(f) & ((p1.kind == "uniform") | (p1.at_risk == 1))]
        m = XGBClassifier(random_state=rep, enable_categorical=True, **XGB).fit(fit[F], fit.at_risk)
        cal = p1[p1.serial.isin(c) & (p1.kind == "risk") & (p1.at_risk == 1)]
        er = p1[p1.serial.isin(e) & (p1.kind == "risk") & (p1.at_risk == 1)]
        eu = p1[p1.serial.isin(e) & (p1.kind == "uniform") & (p1.at_risk == 0)]
        sc, pr, pu = (m.predict_proba(x[F])[:, 1] for x in (cal, er, eu))
        out = []
        for a in (0.2, 0.1):
            t = -guard_threshold(-sc, a)
            out.append(f"a={a}: miss {np.mean(pr < t):.3f} FA {np.mean(pu >= t):.3f}")
        for budget in (0.01, 0.05):
            t = np.quantile(pu, 1 - budget)
            out.append(f"catch@{budget:.0%} {np.mean(pr >= t):.3f}")
        print(rep, f"{name:16s}", " | ".join(out), flush=True)
