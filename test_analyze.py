"""Run analyze.py end to end on synthetic result files with known answers."""
import json, os
import numpy as np, pandas as pd
import analyze


def _fake(tmp, miss):
    rng = np.random.default_rng(0)
    os.makedirs(tmp / "results")
    rows = []
    for src in ["FD001", "FD002"]:
        for tgt in ["FD001", "FD002"]:
            for rep in range(3):
                for unit in range(1, 201):
                    rul = unit % 100            # 31% of units at risk
                    for m in ["point", "guard", "conformal_lb", "cqr_lb", "recal20"]:
                        p = miss[m] if rul <= 30 else {"guard": 0.02}.get(m, 0.10)
                        a = rng.random() >= p if rul <= 30 else rng.random() < p
                        rows.append((rep, src, tgt, m, unit, rul, int(a)))
    pd.DataFrame(rows, columns=["rep", "src", "tgt", "method", "unit", "rul", "alert"]).to_parquet(
        tmp / "results/test_cmapss_units.parquet")
    b = []
    for t, ho in [("all", None), ("WDC", True), ("HGST", True)]:
        for m in ["smart_rule", "guard_0.1", "recal20_0.1"]:
            for s in range(400):
                b.append(dict(mode="test", rep=0, target=t, held_out=ho, method=m, serial=f"{t}{s}",
                              alert=int(rng.random() >= miss[m])))
    pd.DataFrame(b).to_parquet(tmp / "results/test_bb_units.parquet")


def test_known_answers(tmp_path, monkeypatch):
    miss = {"point": 0.30, "guard": 0.08, "conformal_lb": 0.01, "cqr_lb": 0.02, "recal20": 0.09,
            "smart_rule": 0.40, "guard_0.1": 0.09, "recal20_0.1": 0.30}
    _fake(tmp_path, miss)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analyze, "B", 500)
    analyze.main()
    h = {x["id"]: x for x in json.load(open("results/confirmatory.json"))}
    assert h["H1"]["supported"] and h["H2"]["supported"]
    assert h["H3a"]["supported"] and h["H3b"]["supported"]
    assert not h["H4"]["supported"]          # guard (8%) is not above alpha
    assert h["H5"]["supported"] and h["H6"]["supported"] and h["H7"]["supported"]
    assert not h["H8"]["supported"]          # 30% miss is not within alpha + margin
    assert abs(h["H1"]["estimate"] - 0.30) < 0.04
