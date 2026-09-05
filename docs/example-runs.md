# Example runs

Systematic validation of sorethumb across nine real-world datasets.
Every run uses `combination = "intersection"` and `contamination = "auto"`.
The baseline detector set is `isolation_forest + kmeans_distance + one_class_svm`;
additional detectors are added one at a time, then in pairs, then all together.
Each dataset is tested with PCA disabled and enabled.
OC-SVM `nu` is tuned per dataset (smallest value that yields at least one anomaly).
All other detector parameters are at their defaults.

---

## kddcup99_sa

**Source:** KDD Cup 1999 (UCI ML Repository) — SA subset, 10 % sample

**Dimensions:** 100,655 rows × 42 columns

**Description:** Network intrusion detection. 41 connection features (numeric + categorical). Ground-truth `target` label excluded. Industry-standard anomaly benchmark.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 2,487 / 100,655 | 2.47% | 1.6m | 0.1 |
| if + km + oc + ecod | 1,508 / 100,655 | 1.50% | 1.6m | 0.1 |
| if + km + oc + lof | 364 / 100,655 | 0.36% | 1.8m | 0.1 |
| if + km + oc + hbos | 924 / 100,655 | 0.92% | 1.6m | 0.1 |
| if + km + oc + ecod + lof | 260 / 100,655 | 0.26% | 1.8m | 0.1 |
| if + km + oc + ecod + hbos | 782 / 100,655 | 0.78% | 1.7m | 0.1 |
| if + km + oc + lof + hbos | 154 / 100,655 | 0.15% | 1.7m | 0.1 |
| if + km + oc + ecod + lof + hbos | 141 / 100,655 | 0.14% | 1.7m | 0.1 |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 1,290 / 100,655 | 1.28% | 1.4m | 0.1 |
| if + km + oc + ecod | 901 / 100,655 | 0.90% | 1.5m | 0.1 |
| if + km + oc + lof | 222 / 100,655 | 0.22% | 1.5m | 0.1 |
| if + km + oc + hbos | 1,189 / 100,655 | 1.18% | 1.4m | 0.1 |
| if + km + oc + ecod + lof | 190 / 100,655 | 0.19% | 1.5m | 0.1 |
| if + km + oc + ecod + hbos | 881 / 100,655 | 0.88% | 1.5m | 0.1 |
| if + km + oc + lof + hbos | 211 / 100,655 | 0.21% | 1.5m | 0.1 |
| if + km + oc + ecod + lof + hbos | 186 / 100,655 | 0.18% | 1.5m | 0.1 |


**Best result:** `if + km + oc` (PCA=off) — 2,487 anomalies (2.47%)


---

## electricity

**Source:** Harries (1999) via OpenML — Electricity dataset

**Dimensions:** 45,312 rows × 9 columns

**Description:** Half-hourly Australian electricity demand 1996–1998. Price and demand for NSW and Victoria plus transfer. `class` (UP/DOWN) excluded.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 393 / 45,312 | 0.87% | 54.0s | 0.1 |
| if + km + oc + ecod | 212 / 45,312 | 0.47% | 53.9s | 0.1 |
| if + km + oc + lof | 38 / 45,312 | 0.08% | 54.6s | 0.1 |
| if + km + oc + hbos | 199 / 45,312 | 0.44% | 53.4s | 0.1 |
| if + km + oc + ecod + lof | 30 / 45,312 | 0.07% | 54.6s | 0.1 |
| if + km + oc + ecod + hbos | 168 / 45,312 | 0.37% | 53.3s | 0.1 |
| if + km + oc + lof + hbos | 32 / 45,312 | 0.07% | 54.5s | 0.1 |
| if + km + oc + ecod + lof + hbos | 28 / 45,312 | 0.06% | 54.5s | 0.1 |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 377 / 45,312 | 0.83% | 54.4s | 0.1 |
| if + km + oc + ecod | 200 / 45,312 | 0.44% | 54.4s | 0.1 |
| if + km + oc + lof | 32 / 45,312 | 0.07% | 54.8s | 0.1 |
| if + km + oc + hbos | 292 / 45,312 | 0.64% | 54.5s | 0.1 |
| if + km + oc + ecod + lof | 28 / 45,312 | 0.06% | 55.0s | 0.1 |
| if + km + oc + ecod + hbos | 197 / 45,312 | 0.43% | 54.5s | 0.1 |
| if + km + oc + lof + hbos | 31 / 45,312 | 0.07% | 54.8s | 0.1 |
| if + km + oc + ecod + lof + hbos | 28 / 45,312 | 0.06% | 55.0s | 0.1 |


**Best result:** `if + km + oc` (PCA=off) — 393 anomalies (0.87%)


---

## weather_australia

**Source:** Australian Bureau of Meteorology via UCI ML Repository

**Dimensions:** 690 rows × 15 columns

**Description:** Daily weather observations (anonymous columns A1–A14). A15 is the binary RainTomorrow label, excluded from features.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 6 / 690 | 0.87% | 1.3s | 0.1 |
| if + km + oc + ecod | 5 / 690 | 0.72% | 1.3s | 0.1 |
| if + km + oc + lof | 1 / 690 | 0.14% | 1.4s | 0.1 |
| if + km + oc + hbos | 4 / 690 | 0.58% | 1.3s | 0.1 |
| if + km + oc + ecod + lof | 1 / 690 | 0.14% | 1.3s | 0.1 |
| if + km + oc + ecod + hbos | 3 / 690 | 0.43% | 1.3s | 0.1 |
| if + km + oc + lof + hbos | 0 / 690 | 0.00% | 0.3s | 0.1 |
| if + km + oc + ecod + lof + hbos | 0 / 690 | 0.00% | 0.3s | 0.1 |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 6 / 690 | 0.87% | 1.0s | 0.1 |
| if + km + oc + ecod | 6 / 690 | 0.87% | 1.0s | 0.1 |
| if + km + oc + lof | 1 / 690 | 0.14% | 1.0s | 0.1 |
| if + km + oc + hbos | 6 / 690 | 0.87% | 1.0s | 0.1 |
| if + km + oc + ecod + lof | 1 / 690 | 0.14% | 1.0s | 0.1 |
| if + km + oc + ecod + hbos | 6 / 690 | 0.87% | 1.0s | 0.1 |
| if + km + oc + lof + hbos | 1 / 690 | 0.14% | 1.1s | 0.1 |
| if + km + oc + ecod + lof + hbos | 1 / 690 | 0.14% | 1.1s | 0.1 |


**Best result:** `if + km + oc` (PCA=off) — 6 anomalies (0.87%)


---

## macro_us_quarterly

**Source:** statsmodels macrodata — US Federal Reserve

**Dimensions:** 203 rows × 14 columns

**Description:** Quarterly US macroeconomic indicators 1959–2009 (GDP, inflation, unemployment, interest rates). Year and quarter excluded as indices.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 10 / 203 | 4.93% | 0.6s | 0.1 |
| if + km + oc + ecod | 7 / 203 | 3.45% | 0.7s | 0.1 |
| if + km + oc + lof | 8 / 203 | 3.94% | 0.7s | 0.1 |
| if + km + oc + hbos | 6 / 203 | 2.96% | 0.7s | 0.1 |
| if + km + oc + ecod + lof | 6 / 203 | 2.96% | 0.7s | 0.1 |
| if + km + oc + ecod + hbos | 5 / 203 | 2.46% | 0.7s | 0.1 |
| if + km + oc + lof + hbos | 6 / 203 | 2.96% | 0.7s | 0.1 |
| if + km + oc + ecod + lof + hbos | 5 / 203 | 2.46% | 0.7s | 0.1 |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 9 / 203 | 4.43% | 0.6s | 0.1 |
| if + km + oc + ecod | 6 / 203 | 2.96% | 0.6s | 0.1 |
| if + km + oc + lof | 8 / 203 | 3.94% | 0.6s | 0.1 |
| if + km + oc + hbos | 7 / 203 | 3.45% | 0.6s | 0.1 |
| if + km + oc + ecod + lof | 6 / 203 | 2.96% | 0.7s | 0.1 |
| if + km + oc + ecod + hbos | 6 / 203 | 2.96% | 0.8s | 0.1 |
| if + km + oc + lof + hbos | 7 / 203 | 3.45% | 0.7s | 0.1 |
| if + km + oc + ecod + lof + hbos | 6 / 203 | 2.96% | 0.7s | 0.1 |


**Best result:** `if + km + oc` (PCA=off) — 10 anomalies (4.93%)


---

## elnino_sst

**Source:** statsmodels elnino — NOAA/TOGA-TAO buoy array

**Dimensions:** 61 rows × 13 columns

**Description:** Annual mean sea-surface temperatures across 12 Pacific buoy locations 1950–2010. YEAR excluded as index.

> **Note:** 61 rows — all detector combinations produce zero anomalies. The sea-surface temperature data is too small and homogeneous for the intersection of three independent detectors to reach unanimous agreement. A union strategy would surface anomalies but is outside the scope of this validation.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + lof | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + hbos | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + lof | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + hbos | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + lof + hbos | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + lof + hbos | 0 / 61 | 0.00% | 0.0s | 0.1 |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + lof | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + hbos | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + lof | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + hbos | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + lof + hbos | 0 / 61 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + lof + hbos | 0 / 61 | 0.00% | 0.0s | 0.1 |


---

## sunspots_annual

**Source:** statsmodels sunspots — Royal Observatory of Belgium

**Dimensions:** 309 rows × 2 columns

**Description:** Annual Wolf sunspot number 1700–2008. Single numeric feature after excluding YEAR.

> **Note:** Only one effective feature (SUNACTIVITY) after excluding YEAR. Isolation Forest triggers a TreeSHAP fallback on 1-D data (non-fatal; heuristic attribution is used instead). Results are valid.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 5 / 309 | 1.62% | 1.4s | 0.1 ⚠ |
| if + km + oc + ecod | 5 / 309 | 1.62% | 1.4s | 0.1 ⚠ |
| if + km + oc + lof | 2 / 309 | 0.65% | 1.4s | 0.1 ⚠ |
| if + km + oc + hbos | 5 / 309 | 1.62% | 1.4s | 0.1 ⚠ |
| if + km + oc + ecod + lof | 2 / 309 | 0.65% | 1.5s | 0.1 ⚠ |
| if + km + oc + ecod + hbos | 5 / 309 | 1.62% | 1.4s | 0.1 ⚠ |
| if + km + oc + lof + hbos | 2 / 309 | 0.65% | 1.4s | 0.1 ⚠ |
| if + km + oc + ecod + lof + hbos | 2 / 309 | 0.65% | 1.4s | 0.1 ⚠ |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 5 / 309 | 1.62% | 1.4s | 0.1 ⚠ |
| if + km + oc + ecod | 5 / 309 | 1.62% | 1.4s | 0.1 ⚠ |
| if + km + oc + lof | 2 / 309 | 0.65% | 1.4s | 0.1 ⚠ |
| if + km + oc + hbos | 5 / 309 | 1.62% | 1.4s | 0.1 ⚠ |
| if + km + oc + ecod + lof | 2 / 309 | 0.65% | 1.4s | 0.1 ⚠ |
| if + km + oc + ecod + hbos | 5 / 309 | 1.62% | 1.4s | 0.1 ⚠ |
| if + km + oc + lof + hbos | 2 / 309 | 0.65% | 1.4s | 0.1 ⚠ |
| if + km + oc + ecod + lof + hbos | 2 / 309 | 0.65% | 1.4s | 0.1 ⚠ |


**Best result:** `if + km + oc` (PCA=off) — 5 anomalies (1.62%)


---

## longley_multicollinear

**Source:** statsmodels longley — Longley (1967)

**Dimensions:** 16 rows × 7 columns

**Description:** Annual US macro data 1947–1962 (7 highly collinear features). 16 rows only — extreme edge case. Included for completeness.

> **Note:** 16 rows — all detector combinations produce zero anomalies regardless of PCA setting or nu tuning. Intersection of three detectors on 16 points requires unanimous agreement that is never achieved on this dataset. Included for completeness as an extreme edge case.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + lof | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + hbos | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + lof | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + hbos | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + lof + hbos | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + lof + hbos | 0 / 16 | 0.00% | 0.0s | 0.1 |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + lof | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + hbos | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + lof | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + hbos | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + lof + hbos | 0 / 16 | 0.00% | 0.0s | 0.1 |
| if + km + oc + ecod + lof + hbos | 0 / 16 | 0.00% | 0.0s | 0.1 |


---

## natops_mts

**Source:** UEA Time Series Classification Archive — NATOPS dataset

**Dimensions:** 360 rows × 1225 columns

**Description:** 24-channel aircraft hand-signal motion capture (51 timepoints), stored wide (1,224 numeric columns). `label` excluded.

> **Note:** 1,224 numeric columns (24 channels x 51 timepoints). PCA=off: the raw feature space produces 3 consistent anomalies across all 8 combos — a robust, stable signal. PCA=on: dimensionality reduction collapses the discriminating structure and produces zero anomalies across all combos. PCA is counterproductive here.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 3 / 360 | 0.83% | 3.9s | 0.1 |
| if + km + oc + ecod | 3 / 360 | 0.83% | 4.2s | 0.1 |
| if + km + oc + lof | 3 / 360 | 0.83% | 4.1s | 0.1 |
| if + km + oc + hbos | 3 / 360 | 0.83% | 4.1s | 0.1 |
| if + km + oc + ecod + lof | 3 / 360 | 0.83% | 4.4s | 0.1 |
| if + km + oc + ecod + hbos | 3 / 360 | 0.83% | 4.5s | 0.1 |
| if + km + oc + lof + hbos | 3 / 360 | 0.83% | 4.3s | 0.1 |
| if + km + oc + ecod + lof + hbos | 3 / 360 | 0.83% | 4.7s | 0.1 |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 0 / 360 | 0.00% | 3.3s | 0.1 |
| if + km + oc + ecod | 0 / 360 | 0.00% | 3.3s | 0.1 |
| if + km + oc + lof | 0 / 360 | 0.00% | 3.3s | 0.1 |
| if + km + oc + hbos | 0 / 360 | 0.00% | 3.3s | 0.1 |
| if + km + oc + ecod + lof | 0 / 360 | 0.00% | 3.4s | 0.1 |
| if + km + oc + ecod + hbos | 0 / 360 | 0.00% | 3.4s | 0.1 |
| if + km + oc + lof + hbos | 0 / 360 | 0.00% | 3.4s | 0.1 |
| if + km + oc + ecod + lof + hbos | 0 / 360 | 0.00% | 3.4s | 0.1 |


**Best result:** `if + km + oc` (PCA=off) — 3 anomalies (0.83%)


---

## basic_motions_mts

**Source:** UEA Time Series Classification Archive — BasicMotions dataset

**Dimensions:** 80 rows × 601 columns

**Description:** 6-axis IMU data for 4 activities (100 timepoints x 6 channels = 600 cols). `label` excluded.

> **Note:** 80 rows, 600 numeric columns (6 channels x 100 timepoints). Zero anomalies across all combinations and both PCA settings. The dataset is too small for three independent detectors to reach intersection consensus; all activity classes contribute similar feature distributions.

**Baseline detector params:** `isolation_forest` n_estimators=200 · `kmeans_distance` k=auto · `one_class_svm` nu=0.1 (PCA=off), nu=0.1 (PCA=on) · All other params at defaults.


### PCA disabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + ecod | 0 / 80 | 0.00% | 0.7s | 0.1 |
| if + km + oc + lof | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + hbos | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + ecod + lof | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + ecod + hbos | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + lof + hbos | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + ecod + lof + hbos | 0 / 80 | 0.00% | 0.6s | 0.1 |


### PCA enabled

| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |
|---|---|---|---|---|
| if + km + oc | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + ecod | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + lof | 0 / 80 | 0.00% | 0.7s | 0.1 |
| if + km + oc + hbos | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + ecod + lof | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + ecod + hbos | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + lof + hbos | 0 / 80 | 0.00% | 0.6s | 0.1 |
| if + km + oc + ecod + lof + hbos | 0 / 80 | 0.00% | 0.6s | 0.1 |


---

## Notes

- ⚠ indicates the run completed but emitted sorethumb warnings (e.g. FeatureWidthWarning, SlowStageWarning). Results are still valid.
- `ERROR` means the run raised an unhandled exception; see `validation/results.json` for details.
- `combination = "intersection"` means a row is only flagged when **all** detectors agree. Adding more detectors generally reduces the anomaly count.
- `contamination = "auto"` uses each detector's natural learned threshold. The OC-SVM `nu` parameter is the only explicitly tuned value.
