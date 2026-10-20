"""Quick pooled summary of an experiments CSV (development use; not the frozen analysis)."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
keys = [c for c in ["src", "tgt", "target", "held_out", "W", "alpha", "method"] if c in df.columns]
q = " and ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
if q:
    df = df.query(q)
g = df.groupby(keys, dropna=False)[["n_risk", "misses", "n_safe", "false_alarms"]].sum()
g["miss"] = (g.misses / g.n_risk).round(3)
g["fa"] = (g.false_alarms / g.n_safe).round(4)
pd.set_option("display.width", 200, "display.max_rows", 500)
print(g[["n_risk", "miss", "fa"]])
