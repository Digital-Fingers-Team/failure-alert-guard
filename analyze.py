"""Pre-registered confirmatory analysis. DO NOT EDIT after prereg.json is written.

Primary configuration: W = 30 (cycles / days), alpha = 0.10, non-inferiority margin 0.05.
Unit of resampling: the engine (C-MAPSS) or the drive (Backblaze). Each unit's outcome is
its alert / miss fraction averaged over repetitions; 95% CIs come from a 10,000-draw
bootstrap over units, stratified by fleet (C-MAPSS) or by target (Backblaze).
One-sided bootstrap p-values are Holm-corrected across H1-H8.
"""
import json
import numpy as np, pandas as pd

W, A, MARGIN, B = 30, 0.10, 0.05, 10_000
rng = np.random.default_rng(2026)


def boot(groups, stat):
    """groups: list of 1-D arrays (strata); stat: f(list of resampled arrays) -> float."""
    est = stat(groups)
    draws = np.empty(B)
    for b in range(B):
        draws[b] = stat([g[rng.integers(0, len(g), len(g))] for g in groups])
    return est, np.quantile(draws, [0.025, 0.975]), draws


pooled = lambda gs: float(np.concatenate(gs).mean())


def cm_units():
    u = pd.read_parquet("results/test_cmapss_units.parquet")
    u["miss"] = ((u.rul <= W) & (u.alert == 0)).astype(float)
    u["fa"] = ((u.rul > W) & (u.alert == 1)).astype(float)
    return u


def cm_strata(u, method, shift, col, risk):
    d = u[(u.method == method) & ((u.src != u.tgt) if shift else (u.src == u.tgt))]
    d = d[(d.rul <= W) if risk else (d.rul > W)]
    # one value per (target fleet, unit): mean over reps and over source fleets
    per = d.groupby(["tgt", "unit"])[col].mean().reset_index()
    return [g[col].values for _, g in per.groupby("tgt")]


def bb_strata(method, held_out, target=None):
    u = pd.read_parquet("results/test_bb_units.parquet")
    u["miss"] = 1.0 - u.alert
    d = u[u.method == method]
    if target == "all":
        d = d[d.target == "all"]
    else:
        d = d[(d.target != "all") & (d.held_out == held_out)]
    per = d.groupby(["target", "serial"]).miss.mean().reset_index()
    return [g.miss.values for _, g in per.groupby("target")]


def verdict(name, claim, est, ci, draws, kind, thr):
    # kind "gt": supported if CI lower bound > thr; "le": supported if CI upper bound <= thr;
    # "lt0": supported if CI upper bound < 0.
    if kind == "gt":
        ok, p = ci[0] > thr, float((draws <= thr).mean())
    elif kind == "le":
        ok, p = ci[1] <= thr, float((draws > thr).mean())
    else:
        ok, p = ci[1] < 0, float((draws >= 0).mean())
    return dict(id=name, claim=claim, estimate=round(est, 4), ci95=[round(ci[0], 4), round(ci[1], 4)],
                p_one_sided=p, supported_ci=bool(ok))


def main():
    out, u = [], cm_units()
    e, ci, d = boot(cm_strata(u, "point", False, "miss", True), pooled)
    out.append(verdict("H1", "C-MAPSS in-fleet: point-alert miss rate > alpha", e, ci, d, "gt", A))
    e, ci, d = boot(cm_strata(u, "guard", False, "miss", True), pooled)
    out.append(verdict("H2", "C-MAPSS in-fleet: guard miss rate <= alpha + margin", e, ci, d, "le", A + MARGIN))
    for i, other in enumerate(["conformal_lb", "cqr_lb"]):
        p = u[(u.src == u.tgt) & (u.rul > W) & u.method.isin(["guard", other])]
        p = p.pivot_table(index=["tgt", "unit"], columns="method", values="fa", aggfunc="mean")
        diff = (p["guard"] - p[other]).reset_index(name="d")
        e, ci, d = boot([g.d.values for _, g in diff.groupby("tgt")], pooled)
        out.append(verdict(f"H3{'ab'[i]}", f"C-MAPSS in-fleet: guard false-alarm rate < {other}", e, ci, d, "lt0", 0))
    e, ci, d = boot(cm_strata(u, "guard", True, "miss", True), pooled)
    out.append(verdict("H4", "C-MAPSS cross-fleet: source-calibrated guard miss > alpha", e, ci, d, "gt", A))
    e, ci, d = boot(cm_strata(u, "recal20", True, "miss", True), pooled)
    out.append(verdict("H5", "C-MAPSS cross-fleet: guard recalibrated on 20 target failures, miss <= alpha + margin",
                       e, ci, d, "le", A + MARGIN))
    e, ci, d = boot(bb_strata("smart_rule", None, "all"), pooled)
    out.append(verdict("H6", "Backblaze, 3 months later: SMART-rule miss rate > alpha", e, ci, d, "gt", A))
    e, ci, d = boot(bb_strata("guard_0.1", None, "all"), pooled)
    out.append(verdict("H7", "Backblaze, 3 months later: guard miss rate <= alpha + margin", e, ci, d, "le", A + MARGIN))
    e, ci, d = boot(bb_strata("recal20_0.1", True), pooled)
    out.append(verdict("H8", "Backblaze, unseen manufacturer: guard recalibrated on 20 failures, miss <= alpha + margin",
                       e, ci, d, "le", A + MARGIN))
    # Holm correction across the family
    order = np.argsort([h["p_one_sided"] for h in out]); m = len(out); running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * out[i]["p_one_sided"]))
        out[i]["p_holm"] = round(running, 4)
        out[i]["supported"] = bool(out[i]["supported_ci"] and running < 0.05)
    json.dump(out, open("results/confirmatory.json", "w"), indent=1)
    for h in out:
        print(f"{h['id']:4s} {'SUPPORTED' if h['supported'] else 'not supported':14s} "
              f"est={h['estimate']:.3f} CI={h['ci95']} p_holm={h['p_holm']}  {h['claim']}")


if __name__ == "__main__":
    main()
