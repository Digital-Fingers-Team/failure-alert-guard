"""Write prereg.json: hypotheses, primary settings and SHA-256 of all code and data
used by the confirmatory runs. Run once, before any test-mode experiment."""
import hashlib, json, glob, datetime, os

if os.path.exists("prereg.json"):
    raise SystemExit("prereg.json already exists - the analysis is frozen")

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
code = ["common.py", "experiments_cmapss.py", "experiments_bb.py", "analyze.py",
        "fetch_backblaze.py", "prep_backblaze.py"]
data = sorted(glob.glob("data/cmapss/*_FD00*.txt")) + ["data/bb/snapshots.parquet"]

prereg = {
    "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "primary": {"W": 30, "alpha": 0.10, "non_inferiority_margin": 0.05,
                "bootstrap": "10,000 draws over engines/drives, stratified by fleet/target",
                "multiplicity": "Holm across H1-H8, one-sided bootstrap p-values",
                "supported_if": "95% CI criterion met AND Holm p < 0.05"},
    "runs": ["python experiments_cmapss.py test 50 results/test_cmapss.csv",
             "python experiments_bb.py test 5 results/test_bb.csv",
             "python analyze.py"],
    "hypotheses": {
        "H1": "C-MAPSS, same fleet, official test units: point alerts (predicted RUL <= W) miss more than alpha of at-risk engines (CI lower > 0.10).",
        "H2": "C-MAPSS, same fleet: the guard misses at most alpha + margin (CI upper <= 0.15).",
        "H3a": "C-MAPSS, same fleet: guard false-alarm rate is lower than the split-conformal lower bound's (CI upper of difference < 0).",
        "H3b": "C-MAPSS, same fleet: guard false-alarm rate is lower than conformalized quantile regression's (CI upper of difference < 0).",
        "H4": "C-MAPSS, 12 cross-fleet transfers: the source-calibrated guard misses more than alpha (CI lower > 0.10).",
        "H5": "C-MAPSS, cross-fleet: the guard re-calibrated on 20 target-fleet failures misses at most alpha + margin (CI upper <= 0.15).",
        "H6": "Backblaze, P3 (3 months after calibration): the SMART-rule (any of SMART 5/187/188/197/198 raw > 0) misses more than alpha (CI lower > 0.10).",
        "H7": "Backblaze, P3: the guard (fit P1, calibrated P2) misses at most alpha + margin (CI upper <= 0.15).",
        "H8": "Backblaze, manufacturer held out of fitting and calibration: the guard re-calibrated on 20 of its P2 failures misses at most alpha + margin on its P3 drives (CI upper <= 0.15).",
    },
    "exploratory": [
        "all W in {20,30,50} / {7,30} and alpha in {0.05,0.1,0.2}; recalibration with k in {5,10,20,40} / {10,20,40,80}",
        "false-alarm cost of every method; Backblaze catch rate at 1% / 5% inspection budgets and whether the calibration certificate (m+1)/(n+1) holds in P3",
        "Beta(n+1-k, k) prediction of the spread of realised miss rates after recalibration",
    ],
    "disclosures": [
        "Before freezing, the official FD001 test set was used to replicate the public benchmark (run_fd001.py), and FD001/FD003 test sets were used once in pilot_conformal.py (Random Forest, different features). All method and parameter choices were then made on training-file splits (dev mode) only.",
        "Backblaze development used P1 (Jan-Jun 2024) only; P2/P3 outcomes were not examined before freezing.",
    ],
    "sha256": {p: sha(p) for p in code + data},
}
json.dump(prereg, open("prereg.json", "w"), indent=1)
print("frozen", prereg["frozen_utc"])
