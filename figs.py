"""Figures and tables for the paper (presentation only; reads results/*, never changes them).
usage: python figs.py test|dev"""
import json, math, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta

MODE = sys.argv[1] if len(sys.argv) > 1 else "test"
OUT = "figs"
W, A = 30, 0.10
INK, INK2, RULE = "#0b0b0b", "#52514e", "#d9d8d3"
C = {"point": "#a12a1f", "margin": "#c98a1b", "conformal_lb": "#7b5ea7", "cqr_lb": "#2a8a8a",
     "guard": "#1f5fbf", "smart_rule": "#a12a1f", "default_0.5": "#c98a1b", "f1_opt": "#7b5ea7"}
LBL = {"point": "Point", "margin": "Margin", "conformal_lb": "Conformal LB", "cqr_lb": "CQR LB", "guard": "Guard"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})


def rates(df, keys):
    g = df.groupby(keys)[["n_risk", "misses", "n_safe", "false_alarms"]].sum()
    g["miss"] = g.misses / g.n_risk
    g["fa"] = g.false_alarms / g.n_safe
    return g.reset_index()


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)


cm = pd.read_csv(f"results/{MODE}_cmapss.csv")
same, cross = cm[cm.src == cm.tgt], cm[cm.src != cm.tgt]
tables = {}

# ---- Fig 2: same-fleet miss vs false alarms --------------------------------------------
r = rates(same[(same.W == W) & (same.alpha == A)], ["tgt", "method"])
p = rates(same[(same.W == W) & (same.alpha == A)], ["method"])
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.axhline(A, color=INK2, ls="--", lw=0.8)
ax.text(0.003, A + 0.006, f"target miss rate α = {A:.0%}", color=INK2, fontsize=7)
for m in LBL:
    d = r[r.method == m]
    ax.scatter(d.fa, d.miss, s=14, color=C[m], alpha=0.45, lw=0)
    q = p[p.method == m].iloc[0]
    ax.scatter(q.fa, q.miss, s=55, color=C[m], edgecolor="white", lw=0.8, zorder=3, label=LBL[m])
ax.set_xlabel("False-alarm rate (healthy engines alerted)")
ax.set_ylabel("Miss rate (failing engines not alerted)")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
save(fig, "fig2_same_fleet")
tables["same_fleet"] = r.assign(n_risk=r.n_risk).to_dict("records")
tables["same_fleet_pooled"] = p.to_dict("records")

# ---- Fig 3: cross-fleet transfers --------------------------------------------------------
cr = rates(cross[(cross.W == W) & (cross.alpha == A)], ["src", "tgt", "method"])
pairs = cr[["src", "tgt"]].drop_duplicates().values.tolist()
show = [("point", "Point", C["point"]), ("conformal_lb", "Conformal LB", C["conformal_lb"]),
        ("guard", "Guard (source-calibrated)", "#8fb1e3"), ("recal20", "Guard recalibrated, 20 target failures", C["guard"])]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
x = np.arange(len(pairs)); w = 0.2
for i, (m, l, c) in enumerate(show):
    v = [cr[(cr.src == s) & (cr.tgt == t) & (cr.method == m)].miss.iloc[0] for s, t in pairs]
    ax.bar(x + (i - 1.5) * w, v, w, color=c, label=l)
ax.axhline(A, color=INK2, ls="--", lw=0.8)
ax.set_xticks(x, [f"{s[-1]}→{t[-1]}" for s, t in pairs])
ax.set_xlabel("Transfer (source fleet FD00x → target fleet FD00y)")
ax.set_ylabel("Miss rate")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "fig3_cross_fleet")
tables["cross_pooled"] = rates(cross[(cross.W == W) & (cross.alpha == A)], ["method"]).to_dict("records")

# ---- Fig 4: recalibration size vs miss / false alarms, with Beta prediction ----------------
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
ks = [5, 10, 20, 40]
for a, c in [(0.05, "#8fb1e3"), (0.10, C["guard"]), (0.20, "#0d2f66")]:
    rr = rates(cross[(cross.W == W) & (cross.alpha == a)], ["method"]).set_index("method")
    miss = [rr.loc[f"recal{k}", "miss"] for k in ks]
    fa = [rr.loc[f"recal{k}", "fa"] for k in ks]
    lo, hi = [], []
    for k in ks:
        kk = math.ceil((k + 1) * (1 - a))
        if kk > k:
            lo.append(0); hi.append(0)
        else:
            lo.append(beta.ppf(0.1, k + 1 - kk, kk)); hi.append(beta.ppf(0.9, k + 1 - kk, kk))
    axs[0].fill_between(ks, lo, hi, color=c, alpha=0.08, lw=0)
    axs[0].plot(ks, miss, "o-", color=c, ms=3.5, lw=1.2, label=f"α = {a:.0%}")
    axs[0].axhline(a, color=c, ls=":", lw=0.8)
    axs[1].plot(ks, fa, "o-", color=c, ms=3.5, lw=1.2)
for ax, yl in zip(axs, ["Miss rate (pooled over 12 transfers)", "False-alarm rate"]):
    ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("Target-fleet failures used to recalibrate")
    ax.set_ylabel(yl); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
axs[0].legend(frameon=False, fontsize=7, loc="upper right")
axs[1].set_yscale("log"); axs[1].set_ylim(0.01, 1.3)
axs[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
axs[1].text(5.3, 0.75, "k < 1/α − 1: no threshold exists,\nso every engine is alerted", fontsize=6.5, color=INK2, va="top")
fig.subplots_adjust(wspace=0.38)
save(fig, "fig4_recal")

# ---- Backblaze ---------------------------------------------------------------------------
bb = pd.read_csv(f"results/{MODE}_bb.csv")
allb = rates(bb[(bb.target == "all") & (bb.W == W)], ["method"])
allb = allb.merge(bb[(bb.target == "all") & (bb.W == W)].groupby("method").cert.mean().reset_index(), on="method")
tables["bb_all"] = allb.to_dict("records")
tables["bb_all_w7"] = rates(bb[(bb.target == "all") & (bb.W == 7)], ["method"]).to_dict("records")
mk = bb[(bb.target != "all") & (bb.W == W)]
tables["bb_maker"] = rates(mk, ["target", "held_out", "method"]).merge(
    mk.groupby(["target", "held_out", "method"]).cert.mean().reset_index(),
    on=["target", "held_out", "method"]).to_dict("records")

# Fig 5: catch-rate / false-alarm frontier on the future period, rebuilt with the frozen code
if MODE == "test":
    from experiments_bb import load, train
    s, feats = load()
    P1, P3 = s[s.period == "P1"], s[s.period == "P3"]
    m = train(P1, feats, W, 0)
    er = P3[(P3.W == W) & (P3.kind == "risk") & (P3.at_risk == 1)]
    eu = P3[(P3.W == W) & (P3.kind == "uniform") & (P3.at_risk == 0)]
    pr, pu = m.predict_proba(er[feats])[:, 1], m.predict_proba(eu[feats])[:, 1]
    ts = np.unique(np.quantile(pu, np.linspace(0, 1, 2001)))
    fa = np.array([(pu >= t).mean() for t in ts]); catch = np.array([(pr >= t).mean() for t in ts])
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.plot(fa, catch, color=INK2, lw=1.2, label="Model, every possible threshold (P3)")
    pts = [("smart_rule", "SMART rule", C["smart_rule"], "s"), ("default_0.5", "p ≥ 0.5", C["default_0.5"], "s"),
           ("f1_opt", "F1-optimal", C["f1_opt"], "s"), ("budget_0.01", "Budget 1%", "#2a8a8a", "D"),
           ("budget_0.05", "Budget 5%", "#2a8a8a", "D"), ("guard_0.2", "Guard α=20%", C["guard"], "o"),
           ("guard_0.1", "Guard α=10%", C["guard"], "o"), ("guard_0.05", "Guard α=5%", C["guard"], "o")]
    ab = allb.set_index("method")
    for key, l, c, mk_ in pts:
        ax.scatter(ab.loc[key, "fa"], 1 - ab.loc[key, "miss"], color=c, marker=mk_, s=30, zorder=3,
                   edgecolor="white", lw=0.6)
        off = {"budget_0.05": (-48, 6), "smart_rule": (6, -10), "guard_0.05": (-20, 7)}.get(key, (5, -9))
        ax.annotate(l, (ab.loc[key, "fa"], 1 - ab.loc[key, "miss"]), xytext=off, textcoords="offset points",
                    fontsize=6.5, color=c)
    ax.set_xscale("log"); ax.set_xlim(3e-4, 1)
    ax.set_xlabel("False-alarm rate (healthy drives alerted, log scale)")
    ax.set_ylabel("Catch rate (failing drives alerted)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 1))
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig5_drive_frontier")

json.dump(tables, open(f"{OUT}/tables_{MODE}.json", "w"), indent=1, default=float)
print("figures written")
