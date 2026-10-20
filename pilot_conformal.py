"""Pilot: does a point RUL model make safe alerts, and does a conformal lower bound fix it?

Alert rule: raise "inspect now" when predicted RUL (or its lower bound) <= W cycles.
Miss = engine with true RUL <= W that got no alert. Also tests shift: FD001-trained
model applied to FD003 (same condition, extra fault mode).
"""
import sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

D, SEED, CAP, W, ALPHA = sys.argv[1], 42, 125, 30, 0.10
rng = np.random.default_rng(SEED)
cols = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
rd = lambda f: pd.read_csv(f"{D}/{f}", sep=r"\s+", header=None, names=cols)

def with_rul(df):
    df = df.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df

def roll(df, feats, w=10):
    # per-engine rolling mean: a cheap denoising feature
    r = df.groupby("unit")[feats].transform(lambda s: s.rolling(w, min_periods=1).mean())
    return pd.concat([df, r.add_suffix("_m")], axis=1)

train = with_rul(rd("train_FD001.txt"))
feats = [c for c in cols[2:] if train[c].std() > 1e-4]
train = roll(train, feats); F = feats + [f + "_m" for f in feats]

units = train["unit"].unique(); rng.shuffle(units)
cal_u, fit_u = units[:30], units[30:]
fit, cal = train[train.unit.isin(fit_u)], train[train.unit.isin(cal_u)]
m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
m.fit(fit[F], fit["RUL"].clip(upper=CAP))

# calibration: one-sided residual (pred - true) on rows near the decision region
cal = cal[cal.RUL <= CAP]
res = m.predict(cal[F]) - cal["RUL"].values
q = np.quantile(res, 1 - ALPHA)  # lower bound = pred - q

def evaluate(name, test_file, rul_file):
    te = roll(rd(test_file), feats)
    last = te.groupby("unit").tail(1).sort_values("unit")
    y = pd.read_csv(f"{D}/{rul_file}", header=None)[0].values
    p = np.clip(m.predict(last[F]), 0, None); lb = p - q
    need = y <= W
    for label, s in (("point", p), ("conformal LB", lb)):
        alert = s <= W
        print(f"{name:6s} {label:13s} MAE={np.abs(p - y).mean():5.2f}  "
              f"coverage(y>=LB)={(y >= lb).mean():.2f}  "
              f"need={need.sum():2d} missed={(need & ~alert).sum():2d} "
              f"false_alarms={(alert & ~need).sum():2d}")

print(f"q={q:.1f} cycles (alpha={ALPHA}, window={W})")
evaluate("FD001", "test_FD001.txt", "RUL_FD001.txt")
evaluate("FD003", "test_FD003.txt", "RUL_FD003.txt")
