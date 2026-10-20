# Getting started

A walk through the command line, from a single file with no configuration to a
configured workspace, and the file formats sorethumb reads. The
[README](../README.md) has the two-minute version; the
[CLI reference](cli_reference.md) has every command and option.

## The command line: a local CSV, watching live output

### Zero-config: point at a file and go

No config file needed. Pass the data file directly and sorethumb runs with its
defaults (the three default detectors, the unanimous `intersection` policy, and
`scoring.contamination = "auto"`), writing every artefact under one dedicated
`./sorethumb-workspace/` directory — relative to the directory you run the
command from, not beside your data file, and not scattered loose in the current
directory:

```bash
cd /path/to/my-analysis
sorethumb run --log-level INFO /path/to/data.csv
```

The defaults do not choose how many rows get flagged: each detector uses its own
boundary, and a row is flagged only when all of them flag it. The run says so, with
the count it came to, and prints the top-ranked rows whether or not they were
flagged. Then read the results from the same directory; no config file is needed
for that either:

```bash
sorethumb anomalies --top 10      # the flagged rows, with reasons
sorethumb anomalies --ranked      # the top of the ranking, flagged or not
sorethumb runs
sorethumb report
```

To review a fixed number of rows instead, set a review budget; see
[Recommended setup if you have a review budget](ensemble-policy.md#recommended-setup-if-you-have-a-review-budget).

On an interactive terminal, the first run offers to save a `sorethumb.toml` for
future runs. Under cron, CI or a pipe it doesn't ask (it says how to save one on
stderr); `--save-config` / `--no-save-config` decide without asking. Re-running the same command always uses the file you pass — overriding whatever
`uri` is in the config — so iterating across datasets is frictionless. The same
`./sorethumb-workspace/` default applies to a config file that does not set
`run.workdir`; `--workdir` and `run.workdir` override it (see
[Where the workspace lives](cli_reference.md#where-the-workspace-lives)).

> **The workspace is executable, not just data.** Its `models/` directory holds
> fitted estimators as `joblib`/pickle files, which run arbitrary code when
> loaded. Only open a workspace you created or fully trust — see
> [Limitations](limitations.md) and `SECURITY.md`.

### Config-based workflow (full control)

**1. Create a workspace next to your data**

```bash
sorethumb init /path/to/my-analysis
cd /path/to/my-analysis
```

`init` writes a fully-commented `sorethumb.toml` and creates a
`sorethumb-workspace/` directory next to it (it does nothing if a
`sorethumb.toml` is already there). Open the file and set the one
field it leaves for you — `workdir` is already filled in, pointing at the
workspace `init` just created:

```toml
[source]
uri = "/absolute/path/to/your/data.csv"    # also accepts .parquet, .json, and http(s):// URLs

[run]
workdir = "/path/to/my-analysis/sorethumb-workspace"   # where models, results and the SQLite ledger are stored
```

**2. Check how your columns will be treated (no models trained)**

```bash
sorethumb inspect
```

This profiles every column and prints the feature plan (abridged):

```
                                 Feature plan
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Column     ┃ Class           ┃ Treatment     ┃ Reason                                 ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ amount     │ numeric         │ impute_median │ numeric dtype (Float64)                │
│ country    │ categorical     │ one_hot       │ String with cardinality_ratio=0.0004   │
│ session_id │ identifier_like │ drop          │ UUID pattern detected in sample (n=1000)│
│ created_at │ temporal        │ derive_time   │ temporal dtype (Datetime(...))         │
│ region     │ categorical     │ one_hot       │ String with cardinality_ratio=0.0002   │
└────────────┴─────────────────┴───────────────┴────────────────────────────────────────┘

Projected feature width: 38 columns
```

Edit `sorethumb.toml` to override any column treatment before running:

```toml
[columns]
id_column = "session_id"      # excluded from features, used as row label
```

**3. Run detection — watch progress live**

```bash
sorethumb run --log-level INFO
```

Each stage logs as it runs, and the run ends with a summary (abridged; the
numbers are illustrative):

```
… sorethumb_ml.detectors.one_class_svm INFO OneClassSVM: fitting nu=0.100 kernel=rbf on 25000 rows x 38 features.
… sorethumb_ml.scoring.combine INFO ScoreEnsemble: weighting=equal combination=intersection contamination=auto [heuristic] per-detector rates {'isolation_forest': 1.8, 'kmeans_distance': 0.7, 'one_class_svm': 2.1} → intersection: 421/84231 (0.5%).
… sorethumb_ml.store.results INFO Results written: run=run_5c72… group=4413… rows=421 path=…/anomalies.parquet

Run run_5c720f703718c39e3c5b55b171083f30  succeeded=1  skipped=0  failed=0
  Ensemble policy: intersection@1
  Flagged for review: 421 (0.50% of 84,231 rows) — a review shortlist, not an
estimate of true prevalence

╭───────────────────────── Shortlist size not chosen ──────────────────────────╮
│ Nobody chose how many rows to flag. With scoring.contamination = "auto" and  │
│ the intersection@1 policy, each detector flags the rows past its own         │
│ boundary, and only rows every detector flags are kept: 421 of 84,231 rows    │
│ (0.5%) here. …                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯

  Top-ranked rows (highest combined score, flagged or not):
    rank   score  flagged  row     amount  country  created_at
       1  0.9998  yes      51200   9841.5  NZ       2025-03-02 03:14:07
       2  0.9996  yes      7731    8120.0  US       2025-03-02 03:15:51
       …
    rank = position in its group's ranking (1 = most unusual). `sorethumb
anomalies --ranked` lists the top 20 per group.

  Realised detector flag rates (own natural boundary):
    isolation_forest=1.83%, kmeans_distance=0.74%, one_class_svm=2.10%

  Group timings (slowest first):
    __all__                           51.0s  flagged=421

Report: ./sorethumb-workspace/reports/run_5c720f703718c39e3c5b55b171083f30/index.html
```

The three per-detector rates are heuristic boundaries, not measurements, and
they disagree. Under the default (`"auto"`, `intersection`), a row is flagged only
when all three boundaries flag it, so the flagged count is what they agree on. It is
not a budget you set. Treat the count as *"records worth a look"*, never *"this is
how many anomalies the data has"*.

**4. Print the flagged rows with their reasons**

```bash
sorethumb anomalies           # latest run, top reasons for every flagged row
sorethumb anomalies --top 20  # just the 20 most anomalous rows
sorethumb anomalies --reasons 5 --top 50   # five reason columns
sorethumb anomalies --detail --top 5       # the record behind each reason
```

```
                      Flagged rows — run run_5c720f70
┏━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃  # ┃ row   ┃  score ┃ kind      ┃ reason 1             ┃ reason 2           ┃
┡━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│  1 │ 51200 │ 0.9821 │ heuristic │ amount=14500.0       │ country=NG         │
├────┼───────┼────────┼───────────┼──────────────────────┼────────────────────┤
│  2 │ 7731  │ 0.9714 │ heuristic │ hour_of_day=3        │ failed_attempts=12 │
├────┼───────┼────────┼───────────┼──────────────────────┼────────────────────┤
│  … │       │        │           │                      │                    │
└────┴───────┴────────┴───────────┴──────────────────────┴────────────────────┘
  421 flagged row(s)   run=run_5c720f703718c39e3c5b55b171083f30   workspace=sorethumb-workspace
```

`kind=exact` means ECOD/HBOS's own additive score decomposition (summing it
recovers the score with zero error, by construction); `kind=model_specific`
means TreeSHAP (Isolation Forest), which uses the fitted trees' actual
structure but decomposes path length, of which the score is a nonlinear
transform; `kind=heuristic` means centroid distance (KMeans), a
finite-difference gradient or KernelSHAP (One-Class SVM, LOF), or a
reference-scaled deviation where a score is flat. With several detectors, a row
gets the weakest kind among them, so the default ensemble's rows are usually
`heuristic`. See
[docs/explanations.md](explanations.md).

For a machine-readable result pipe `--json`:

```bash
sorethumb anomalies --top 100 --json | jq '.[] | {rank, score: .composite_score, r1: .reason_1}'
```

## Supported file formats

| Format | Extensions | Notes |
| --- | --- | --- |
| Parquet | `.parquet` | Recommended for large datasets; column-oriented, fast |
| CSV | `.csv`, `.csv.gz` | Auto-detects delimiter; override via `read_options.separator` |
| TSV | `.tsv`, `.tsv.gz` | Tab-separated; `\t` separator injected automatically |
| JSON | `.json`, `.json.gz` | Full document loaded eagerly (not streamed) |
| JSONL / NDJSON | `.jsonl`, `.ndjson`, `.jsonl.gz`, `.ndjson.gz` | Streamed line-by-line |
| TSF | `.tsf` | [Monash Time Series Forecasting](https://github.com/rakshitha123/TSForecasting/tree/master/utils) format — each series becomes one row; `@attribute` columns preserved, series observations expand to `value_0`, `value_1`, … |

**Text encoding.** CSV, TSV and JSON sources must be UTF-8 (a byte-order mark
is fine). Excel's plain "CSV" format on Windows writes cp1252 instead, which
fails with a message pointing here. Re-save it as "CSV UTF-8 (Comma delimited)",
or convert it once, in PowerShell 7 with
`Get-Content data.csv -Encoding windows-1252 | Set-Content data-utf8.csv -Encoding utf8`
or with Python:
`python -c "open('data-utf8.csv', 'w', encoding='utf-8').write(open('data.csv', encoding='cp1252').read())"`.
As a last resort, `read_options = { encoding = "utf8-lossy" }` reads the file
with undecodable characters replaced.

Format is auto-detected from the file extension. Set `source.format` explicitly when the extension is ambiguous:

```toml
[source]
uri = "/data/my_file.dat"
format = "csv"
```
