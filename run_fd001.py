"""Reproduce the report's Section 10 plan on NASA C-MAPSS FD001.

Train LR / RF / XGBoost on train_FD001, tune nothing on test, evaluate once on
the last cycle of each of the 100 official test engines against RUL_FD001.
Runs two target variants: RUL capped at 125 (common practice) and uncapped.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

D = sys.argv[1]
SEED = 42
cols = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
rd = lambda f: pd.read_csv(f"{D}/{f}", sep=r"\s+", header=None, names=cols)
train, test = rd("train_FD001.txt"), rd("test_FD001.txt")
rul_test = pd.read_csv(f"{D}/RUL_FD001.txt", header=None)[0].values

train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
# Drop near-constant columns (FD001 has a single operating condition)
feats = [c for c in cols[2:] if train[c].std() > 1e-4]
dropped = [c for c in cols[2:] if c not in feats]
test_last = test.groupby("unit").tail(1).sort_values("unit")

def models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=SEED, n_jobs=2),
    }

def score(y, p):
    return dict(MAE=round(mean_absolute_error(y, p), 2),
                RMSE=round(float(np.sqrt(mean_squared_error(y, p))), 2),
                R2=round(r2_score(y, p), 3),
                late=int((p > y).sum()), early=int((p < y).sum()))

out = {"features": feats, "dropped": dropped, "seed": SEED}
for cap in (125, None):
    y = train["RUL"].clip(upper=cap) if cap else train["RUL"]
    # engine-level validation split (no row leakage between engines)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, va = next(gss.split(train, y, groups=train["unit"]))
    yt = np.minimum(rul_test, cap) if cap else rul_test
    res = {}
    for name, m in models().items():
        m.fit(train.iloc[tr][feats], y.iloc[tr])
        val = score(y.iloc[va].values, m.predict(train.iloc[va][feats]))
        m.fit(train[feats], y)  # refit on all training engines
        p = np.clip(m.predict(test_last[feats]), 0, None)
        res[name] = {"val": val, "test": score(yt, p), "test_vs_true_uncapped": score(rul_test, p)}
        pd.DataFrame({"unit": test_last["unit"].values, "true": rul_test, "pred": p}).to_csv(
            f"preds_{name.replace(' ', '_')}_cap{cap}.csv", index=False)
    out[f"cap={cap}"] = res
json.dump(out, open("results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
