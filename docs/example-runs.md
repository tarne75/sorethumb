# Example runs

Validation sweep of sorethumb across nine real-world datasets (`scripts/run_validation.py`). Every case fits on a 70/30 train/held-out split and scores the held-out split via `score_forward` -- never trains and evaluates on the same rows. Every detector in a combo targets the same fixed 5% review budget (`scoring.contamination = 0.05`, `combination = "intersection"`); OneClassSVM's `nu` is derived automatically from that budget, not searched for. ROC-AUC/AP are reported only for the one dataset with a genuine anomaly ground truth (kddcup99_sa) -- every other dataset shown here has no anomaly label, so only operational metrics (rows flagged, elapsed time) are reported for it; inventing an accuracy number without ground truth would be dishonest.

Generated 2026-10-19 from 144 cases at commit `b8ce922` (numpy 2.5.2, polars 1.44.1, pydantic 2.13.5, scikit-learn 1.9.0, scipy 1.18.1, sorethumb-ml 0.1.0). Regenerate with `uv run python scripts/run_validation.py` then `uv run python scripts/gen_example_runs.py`.

---

## kddcup99_sa

**Source:** KDD Cup 1999 (UCI ML Repository) — SA subset, 10 % sample

**Dimensions:** 100,655 rows x 42 columns

**Description:** Network intrusion detection. 41 connection features (numeric + categorical). `target` is a real anomaly ground truth -- the one dataset here with held-out ROC-AUC/AP.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time | ROC-AUC | AP |
|---|---|---|---|---|---|
| if + km + oc | 183 / 30,196 | 0.61% | 1.0m | 0.9294 | 0.2140 |
| if + km + oc + ecod | 183 / 30,196 | 0.61% | 1.0m | 0.9433 | 0.2407 |
| if + km + oc + lof | 26 / 30,196 | 0.09% | 1.6m | 0.9049 | 0.2733 |
| if + km + oc + hbos | 158 / 30,196 | 0.52% | 1.0m | 0.9581 | 0.3024 |
| if + km + oc + ecod + lof | 26 / 30,196 | 0.09% | 1.6m | 0.9315 | 0.2391 |
| if + km + oc + ecod + hbos | 158 / 30,196 | 0.52% | 1.1m | 0.9461 | 0.2788 |
| if + km + oc + lof + hbos | 14 / 30,196 | 0.05% | 1.7m | 0.9547 | 0.3282 |
| if + km + oc + ecod + lof + hbos | 14 / 30,196 | 0.05% | 1.7m | 0.9384 | 0.2257 |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time | ROC-AUC | AP |
|---|---|---|---|---|---|
| if + km + oc | 208 / 30,196 | 0.69% | 1.1m | 0.9318 | 0.2001 |
| if + km + oc + ecod | 149 / 30,196 | 0.49% | 1.1m | 0.8596 | 0.1134 |
| if + km + oc + lof | 34 / 30,196 | 0.11% | 1.0m | 0.8655 | 0.1421 |
| if + km + oc + hbos | 168 / 30,196 | 0.56% | 59.1s | 0.8617 | 0.1134 |
| if + km + oc + ecod + lof | 29 / 30,196 | 0.10% | 1.2m | 0.8400 | 0.1209 |
| if + km + oc + ecod + hbos | 145 / 30,196 | 0.48% | 1.1m | 0.7819 | 0.1800 |
| if + km + oc + lof + hbos | 30 / 30,196 | 0.10% | 1.2m | 0.8441 | 0.1207 |
| if + km + oc + ecod + lof + hbos | 28 / 30,196 | 0.09% | 1.2m | 0.8021 | 0.0978 |


**Most flagged:** `if + km + oc` (PCA=on) — 208 / 30,196 (0.69%)


---

## electricity

**Source:** Harries (1999) via OpenML — Electricity dataset

**Dimensions:** 45,312 rows x 9 columns

**Description:** Half-hourly Australian electricity demand 1996-1998. Price and demand for NSW and Victoria plus transfer. `class` (UP/DOWN) excluded.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | 447 / 13,594 | 3.29% | 42.1s |
| if + km + oc + ecod | 358 / 13,594 | 2.63% | 44.5s |
| if + km + oc + lof | 61 / 13,594 | 0.45% | 55.5s |
| if + km + oc + hbos | 379 / 13,594 | 2.79% | 49.5s |
| if + km + oc + ecod + lof | 45 / 13,594 | 0.33% | 59.4s |
| if + km + oc + ecod + hbos | 300 / 13,594 | 2.21% | 51.5s |
| if + km + oc + lof + hbos | 55 / 13,594 | 0.40% | 54.2s |
| if + km + oc + ecod + lof + hbos | 40 / 13,594 | 0.29% | 54.1s |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | 333 / 13,594 | 2.45% | 41.2s |
| if + km + oc + ecod | 248 / 13,594 | 1.82% | 42.0s |
| if + km + oc + lof | 63 / 13,594 | 0.46% | 48.0s |
| if + km + oc + hbos | 284 / 13,594 | 2.09% | 40.8s |
| if + km + oc + ecod + lof | 62 / 13,594 | 0.46% | 43.3s |
| if + km + oc + ecod + hbos | 226 / 13,594 | 1.66% | 38.6s |
| if + km + oc + lof + hbos | 59 / 13,594 | 0.43% | 41.1s |
| if + km + oc + ecod + lof + hbos | 58 / 13,594 | 0.43% | 39.8s |


**Most flagged:** `if + km + oc` (PCA=off) — 447 / 13,594 (3.29%)


---

## weather_australia

**Source:** Australian Bureau of Meteorology via UCI ML Repository

**Dimensions:** 690 rows x 15 columns

**Description:** Daily weather observations (anonymous columns A1-A14). A15 is the binary RainTomorrow label, excluded from features.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | 3 / 207 | 1.45% | 1.3s |
| if + km + oc + ecod | 2 / 207 | 0.97% | 1.1s |
| if + km + oc + lof | 2 / 207 | 0.97% | 1.3s |
| if + km + oc + hbos | 2 / 207 | 0.97% | 1.1s |
| if + km + oc + ecod + lof | 1 / 207 | 0.48% | 1.2s |
| if + km + oc + ecod + hbos | 2 / 207 | 0.97% | 1.2s |
| if + km + oc + lof + hbos | 1 / 207 | 0.48% | 1.3s |
| if + km + oc + ecod + lof + hbos | 1 / 207 | 0.48% | 1.3s |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | 2 / 207 | 0.97% | 1.1s |
| if + km + oc + ecod | 2 / 207 | 0.97% | 1.1s |
| if + km + oc + lof | 1 / 207 | 0.48% | 1.1s |
| if + km + oc + hbos | 2 / 207 | 0.97% | 1.3s |
| if + km + oc + ecod + lof | 1 / 207 | 0.48% | 1.1s |
| if + km + oc + ecod + hbos | 2 / 207 | 0.97% | 1.4s |
| if + km + oc + lof + hbos | 1 / 207 | 0.48% | 1.2s |
| if + km + oc + ecod + lof + hbos | 1 / 207 | 0.48% | 1.3s |


**Most flagged:** `if + km + oc` (PCA=off) — 3 / 207 (1.45%)


---

## macro_us_quarterly

**Source:** statsmodels macrodata — US Federal Reserve

**Dimensions:** 203 rows x 14 columns

**Description:** Quarterly US macroeconomic indicators 1959-2009 (GDP, inflation, unemployment, interest rates). Year and quarter excluded as indices.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=142/holdout=61 < scoring.min_records) | — | — |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=142/holdout=61 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=142/holdout=61 < scoring.min_records) | — | — |


---

## elnino_sst

**Source:** statsmodels elnino — NOAA/TOGA-TAO buoy array

**Dimensions:** 61 rows x 13 columns

**Description:** Annual mean sea-surface temperatures across 12 Pacific buoy locations 1950-2010. YEAR excluded as index.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=43/holdout=18 < scoring.min_records) | — | — |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=43/holdout=18 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=43/holdout=18 < scoring.min_records) | — | — |


---

## sunspots_annual

**Source:** statsmodels sunspots — Royal Observatory of Belgium

**Dimensions:** 309 rows x 2 columns

**Description:** Annual Wolf sunspot number 1700-2008. Single numeric feature after excluding YEAR.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=216/holdout=93 < scoring.min_records) | — | — |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=216/holdout=93 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=216/holdout=93 < scoring.min_records) | — | — |


---

## longley_multicollinear

**Source:** statsmodels longley — Longley (1967)

**Dimensions:** 16 rows x 7 columns

**Description:** Annual US macro data 1947-1962 (7 highly collinear features). 16 rows only — extreme edge case. Included for completeness.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=11/holdout=5 < scoring.min_records) | — | — |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=11/holdout=5 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=11/holdout=5 < scoring.min_records) | — | — |


---

## natops_mts

**Source:** UEA Time Series Classification Archive — NATOPS dataset

**Dimensions:** 360 rows x 1225 columns

**Description:** 24-channel aircraft hand-signal motion capture (51 timepoints), stored wide (1,224 numeric columns). `label` excluded.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | 1 / 108 | 0.93% | 11.1s |
| if + km + oc + ecod | 1 / 108 | 0.93% | 11.4s |
| if + km + oc + lof | 1 / 108 | 0.93% | 11.2s |
| if + km + oc + hbos | 1 / 108 | 0.93% | 11.3s |
| if + km + oc + ecod + lof | 1 / 108 | 0.93% | 11.7s |
| if + km + oc + ecod + hbos | 1 / 108 | 0.93% | 11.4s |
| if + km + oc + lof + hbos | 1 / 108 | 0.93% | 11.5s |
| if + km + oc + ecod + lof + hbos | 1 / 108 | 0.93% | 11.6s |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | 1 / 108 | 0.93% | 11.4s |
| if + km + oc + ecod | 1 / 108 | 0.93% | 11.4s |
| if + km + oc + lof | 1 / 108 | 0.93% | 11.2s |
| if + km + oc + hbos | 1 / 108 | 0.93% | 11.1s |
| if + km + oc + ecod + lof | 1 / 108 | 0.93% | 11.4s |
| if + km + oc + ecod + hbos | 1 / 108 | 0.93% | 11.2s |
| if + km + oc + lof + hbos | 1 / 108 | 0.93% | 11.3s |
| if + km + oc + ecod + lof + hbos | 1 / 108 | 0.93% | 11.5s |


**Most flagged:** `if + km + oc` (PCA=off) — 1 / 108 (0.93%)


---

## basic_motions_mts

**Source:** UEA Time Series Classification Archive — BasicMotions dataset

**Dimensions:** 80 rows x 601 columns

**Description:** 6-axis IMU data for 4 activities (100 timepoints x 6 channels = 600 cols). `label` excluded.


### PCA disabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=56/holdout=24 < scoring.min_records) | — | — |


### PCA enabled

| Detectors | Flagged / Holdout | Rate | Time |
|---|---|---|---|
| if + km + oc | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + ecod | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + lof | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + hbos | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + ecod + hbos | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + lof + hbos | skipped (train=56/holdout=24 < scoring.min_records) | — | — |
| if + km + oc + ecod + lof + hbos | skipped (train=56/holdout=24 < scoring.min_records) | — | — |


---

## Notes

- `ERROR` means the run raised an unhandled exception; see `validation/results.json` for details.
- "skipped" means the train or holdout split fell below `scoring.min_records` (default 100) -- too few rows for either half to fit/score meaningfully, not a failure.
- `combination = "intersection"` means a row is only flagged when **all** detectors in the combo agree. Adding more detectors generally reduces the flagged count.
- ROC-AUC/AP columns only appear for datasets with a real anomaly label (currently kddcup99_sa); `n/a` means the metric was undefined for that split (e.g. the holdout sample was single-class).
