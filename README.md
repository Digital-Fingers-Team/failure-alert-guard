# Low Error Is Not a Safe Alarm

Certified miss-rate guarantees for machine-failure alerts, tested on NASA C-MAPSS (simulated jet engines) and on 343,944 real Backblaze hard drives. The paper is `Failure_Alerts_Research_Paper.pdf`.

The **guard** turns any RUL or risk model into an alarm. Given *n* recorded failures, it takes the k-th smallest calibration score, with k = ⌈(n+1)(1−α)⌉, as the threshold. If new failing units are exchangeable with the calibration failures, at most a fraction α of failing units are missed.

## Pre-registered results (W = 30, α = 0.10)

| | Claim | Estimate [95% CI] | |
|---|---|---|---|
| H1 | C-MAPSS same fleet: point alarm misses > α | 12.9% [8.5, 17.7] | not supported |
| H2 | C-MAPSS same fleet: guard misses ≤ α + 0.05 | 7.9% [4.9, 11.3] | supported |
| H3a/b | Guard has fewer false alarms than the conformal / CQR bounds | −10.0 / −5.6 pp | supported |
| H4 | Cross-fleet: source-calibrated guard misses > α | 27.1% [22.6, 31.6] | supported |
| H5 | Cross-fleet: guard recalibrated on 20 failures, misses ≤ α + 0.05 | 10.0% [7.1, 13.3] | supported |
| H6 | Drives, 3 months later: SMART rule misses > α | 25.6% [23.0, 28.3] | supported |
| H7 | Drives, 3 months later: guard misses ≤ α + 0.05 | 7.4% [6.0, 9.0] | supported |
| H8 | Drives, unseen manufacturer, recalibrated on 20 failures: misses ≤ α + 0.05 | 8.3% [7.1, 9.7] | supported |

## Reproduce

```sh
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python pandas scikit-learn xgboost pyarrow matplotlib scipy duckdb pytest
# data: C-MAPSS zip from NASA (phm-datasets S3) into data/cmapss/; Backblaze quarters:
python fetch_backblaze.py data_Q1_2024 data_Q2_2024 data_Q3_2024 data_Q4_2024 data_Q1_2025
python prep_backblaze.py
python -m pytest -q
python experiments_cmapss.py test 50 results/test_cmapss.csv
python experiments_bb.py test 5 results/test_bb.csv
python analyze.py            # frozen: its SHA-256 is in prereg.json
python figs.py test && python build_paper.py
```

| File | Role |
|---|---|
| `run_fd001.py` | reproduction of the public FD001 benchmark |
| `pilot_conformal.py` | first pilot (touched FD001/FD003 test sets; disclosed) |
| `common.py` | loading, features, snapshot sampling, guard / conformal thresholds |
| `experiments_cmapss.py`, `experiments_bb.py` | dev (training data / P1 only) and test runs |
| `analyze.py`, `freeze.py`, `prereg.json` | pre-registered confirmatory analysis (`git tag freeze`) |
| `figs.py`, `build_paper.py`, `paper_*.html` | figures and paper |

Data: NASA C-MAPSS (NASA Open Data Portal); Backblaze Drive Stats (free to use with attribution to Backblaze).
