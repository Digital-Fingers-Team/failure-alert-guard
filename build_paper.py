"""Assemble paper.html from its parts, fill tables from result files, print to PDF."""
import json, re, subprocess
import pandas as pd

pct = lambda x: f"{100 * x:.1f}%"


def rates(df, keys):
    g = df.groupby(keys)[["n_risk", "misses", "n_safe", "false_alarms"]].sum()
    g["miss"] = g.misses / g.n_risk
    g["fa"] = g.false_alarms / g.n_safe
    return g


cm = pd.read_csv("results/test_cmapss.csv")
same, cross = cm[cm.src == cm.tgt], cm[cm.src != cm.tgt]
bb = pd.read_csv("results/test_bb.csv")
hyp = json.load(open("results/confirmatory.json"))

# Table 1: fleets
info = {"FD001": (1, 1, 100, 100), "FD002": (6, 1, 260, 259), "FD003": (1, 2, 100, 100), "FD004": (6, 2, 249, 248)}
risk = same[(same.rep == 0) & (same.method == "point") & (same.W == 30) & (same.alpha == 0.1)].set_index("tgt").n_risk
t_fleets = "".join(f"<tr><td>{f}</td><td>{c}</td><td>{m}</td><td class='n'>{a}</td><td class='n'>{b}</td>"
                   f"<td class='n'>{risk[f]}</td></tr>" for f, (c, m, a, b) in info.items())

# Table 4: hypotheses
def res(h):
    return "<span class='ok'>Supported</span>" if h["supported"] else "<span class='no-ok'>Not supported</span>"
pval = lambda x: "&lt;0.001" if x < 0.001 else f"{x:.3f}"
t_hyp = "".join(f"<tr><td>{h['id']}</td><td>{h['claim']}</td><td class='n'>{h['estimate']:.3f}</td>"
                f"<td class='n'>[{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}]</td>"
                f"<td class='n'>{pval(h['p_holm'])}</td><td>{res(h)}</td></tr>" for h in hyp)

# Table 5: same fleet per fleet
r = rates(same[(same.W == 30) & (same.alpha == 0.1)], ["tgt", "method"])
p = rates(same[(same.W == 30) & (same.alpha == 0.1)], ["method"])
ms = ["point", "margin", "conformal_lb", "cqr_lb", "guard"]
cell = lambda x: f"{pct(x.miss)} / {pct(x.fa)}"
t_same = "".join(f"<tr><td>{f}</td><td class='n'>{risk[f]}</td>" + "".join(
    f"<td class='n'>{cell(r.loc[(f, m)])}</td>" for m in ms) + "</tr>" for f in info)
t_same += "<tr class='hl'><td>Pooled</td><td class='n'>" + str(int(risk.sum())) + "</td>" + "".join(
    f"<td class='n'>{cell(p.loc[m])}</td>" for m in ms) + "</tr>"

# Table 6: alpha tracking
a = rates(same, ["W", "alpha", "method"])
t_alpha = "".join(f"<tr><td>{w}</td>" + "".join(f"<td class='n'>{pct(a.loc[(w, al, 'guard')].miss)}</td>" for al in (0.05, 0.1, 0.2))
                  + f"<td class='n'>{pct(a.loc[(w, 0.1, 'conformal_lb')].miss)}</td><td class='n'>{pct(a.loc[(w, 0.1, 'point')].miss)}</td></tr>"
                  for w in (20, 30, 50))

# Table 7: Backblaze
b30 = rates(bb[(bb.target == "all") & (bb.W == 30)], ["method"])
b7 = rates(bb[(bb.target == "all") & (bb.W == 7)], ["method"])
cert = bb[(bb.target == "all") & (bb.W == 30)].groupby("method").cert.mean()
names = [("smart_rule", "SMART rule (industry)"), ("default_0.5", "Model, p ≥ 0.5"), ("f1_opt", "Model, F1-optimal threshold"),
         ("budget_0.01", "Budget mode, 1% inspected"), ("budget_0.05", "Budget mode, 5% inspected"),
         ("guard_0.2", "Guard, α = 20%"), ("guard_0.1", "Guard, α = 10%"), ("guard_0.05", "Guard, α = 5%")]
t_bb = "".join(f"<tr{' class=hl' if k == 'guard_0.1' else ''}><td>{n}</td><td class='n'>{pct(b30.loc[k].miss)}</td>"
               f"<td class='n'>{pct(b30.loc[k].fa)}</td><td class='n'>{'' if pd.isna(cert.get(k)) else '≤ ' + pct(cert[k])}</td>"
               f"<td class='n'>{pct(b7.loc[k].miss)}</td><td class='n'>{pct(b7.loc[k].fa)}</td></tr>" for k, n in names)


def fig(n, src, cap):
    return f"<figure><img src='figs/{src}.png'><figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>"


figs = {
    "FIG2": fig(2, "fig2_same_fleet", "C-MAPSS, same fleet, official test engines (<i>W</i> = 30, α = 10%). Large dots: pooled over the four fleets; small dots: each fleet. Only the guard sits at the requested miss rate; the interval methods buy unrequested extra safety with 3–4× more false alarms."),
    "FIG3": fig(3, "fig3_cross_fleet", "Miss rate when a model trained and calibrated on one fleet is used on another (official test engines, α = 10%). Recalibrating the guard's threshold with 20 failures from the target fleet (dark blue) brings the pooled miss rate back to 10.0%; individual transfers range from 4% to 18%, the spread expected from only 20 calibration failures."),
    "FIG4": fig(4, "fig4_recal", "Recalibration with <i>k</i> target-fleet failures, pooled over the 12 transfers. Left: miss rate, with the 10–90% range predicted by the Beta(<i>n</i>+1−<i>k</i>, <i>k</i>) distribution for a single calibration set (shaded). Right: false alarms. Below ⌈1/α⌉ − 1 failures no threshold exists and every engine is alerted."),
    "FIG5": fig(5, "fig5_drive_frontier", "Backblaze, October–December 2024 (model fitted on January–June, thresholds set on July–September). The curve shows every threshold of one model; markers show the rules of Table 7. The steep upper part is the price of catching failures that SMART does not reveal."),
}

part1 = open("paper_part1.html").read()
blocks = dict(re.findall(r"<!--@(\w+)-->\n(.*?)(?=\n<!--@|\Z)", open("paper_results.html").read(), flags=re.S))
html = (part1.replace("<!--ABSTRACT-->", blocks["ABSTRACT"]).replace("<!--TABLE_FLEETS-->", t_fleets)
        .replace("<!--FIG_PIPELINE-->", open("fig_pipeline.svg").read())
        + blocks["RESULTS"] + open("paper_end.html").read())
for k, v in {"TABLE_HYP": t_hyp, "TABLE_SAME": t_same, "TABLE_ALPHA": t_alpha, "TABLE_BB": t_bb, **figs}.items():
    html = html.replace(f"<!--{k}-->", v)
assert not re.search(r"<!--[A-Z_0-9]+-->", html), "unfilled placeholder"
open("paper.html", "w").write(html)
subprocess.run(["chromium-browser", "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                "--virtual-time-budget=15000", "--print-to-pdf=Failure_Alerts_Research_Paper.pdf", "paper.html"],
               check=True, capture_output=True)
print("ok")
