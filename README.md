<div align="center">
  <img src="https://raw.githubusercontent.com/tarne75/sorethumb/main/docs/banner.svg" alt="sorethumb — unsupervised anomaly detection for tabular data" width="860"/>
</div>

<div align="center">

[![CI](https://github.com/tarne75/sorethumb/actions/workflows/ci.yml/badge.svg)](https://github.com/tarne75/sorethumb/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/tarne75/sorethumb/graph/badge.svg?token=UJLJS8GHDV)](https://codecov.io/github/tarne75/sorethumb)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?color=123A66)](https://github.com/tarne75/sorethumb)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?color=2F80ED)](https://github.com/tarne75/sorethumb/blob/main/LICENSE)

</div>

`sorethumb` takes a dataset it knows nothing about, works out how to treat every
column, fits an ensemble of detectors, ranks the records that stand out, and
explains *why* each one stands out in terms of the original columns — so the
result is actionable without a background in machine learning.

**It ranks; it does not count.** The result is an ordering of records by statistical
oddity, plus a *flag decision* on top of it: a shortlist for a person to review. The
flagged count is never a measurement of how many anomalies your data holds, and by
default (`scoring.contamination = "auto"`) nobody chose its size: each detector flags
by its own boundary, and a row is flagged only if all of them flag it. So **read the
ranking first.** Every run ends with its top-ranked rows, flagged or not, and a first
run says all this prominently.

It runs on one machine (Linux, macOS or Windows) and is measured up to 1,000,000 rows
([scale guide](https://github.com/tarne75/sorethumb/blob/main/docs/scale.md)).

> **Status: pre-release (0.1.0).** The API and on-disk formats may still change
> between minor versions, and the package is not yet published to PyPI — install
> from source (see below). See [Limitations](https://github.com/tarne75/sorethumb/blob/main/docs/limitations.md) for what
> it does not do.

---

## Installation

Not yet on PyPI. Install from a clone:

```bash
git clone https://github.com/tarne75/sorethumb
cd sorethumb
pip install .
```

The core install is everything the quickstart below, the `sorethumb` command, and
the CSV/HTML reports need. Optional extras add to it:

- `pip install ".[explain]"` adds the SHAP package for TreeSHAP and KernelSHAP
  attributions. Without it those explanations fall back to a finite-difference
  method and say so with a warning.
- `pip install ".[benchmark]"` adds what the benchmark harness needs (pandas,
  pyarrow and Hugging Face `datasets`).

---

## 60-second quickstart

No network access and nothing beyond sorethumb's own core dependencies — the
dataset is a small synthetic table generated in-process, with 30 of its 2,000
rows nudged far outside the normal range so there's something for the
detectors to actually find:

```python
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from sorethumb_ml import Config, SourceConfig, run_detection

# A fresh directory every run -- sorethumb's runs are idempotent/resumable
# (it skips a group it's already completed for a given config+data), so a
# fixed path would make a second copy-paste of this snippet report 0 newly
# succeeded groups instead of re-running the demo.
tmp_dir = Path(tempfile.mkdtemp(prefix="sorethumb_quickstart_"))

rng = np.random.default_rng(0)
n_rows, n_anomalies = 2_000, 30
amount = rng.normal(loc=50.0, scale=10.0, size=n_rows)
latency_ms = rng.normal(loc=200.0, scale=30.0, size=n_rows)
region = rng.choice(["us", "eu", "apac"], size=n_rows)

# Push a random slice of rows far outside the normal range.
anomaly_idx = rng.choice(n_rows, size=n_anomalies, replace=False)
amount[anomaly_idx] = rng.uniform(500.0, 1000.0, size=n_anomalies)
latency_ms[anomaly_idx] = rng.uniform(2000.0, 5000.0, size=n_anomalies)

df = pl.DataFrame({"amount": amount, "latency_ms": latency_ms, "region": region})
df.write_csv(tmp_dir / "quickstart.csv")

config = Config(
    source=SourceConfig(uri=str(tmp_dir / "quickstart.csv")),
    run={"workdir": str(tmp_dir / "workspace")},
)
result = run_detection(config, no_report=True)
print(f"Flagged {result.n_anomalies} rows for review across {result.n_succeeded} group(s)")

# The ranking is the result; the flagged rows are a shortlist drawn from it.
for row in result.groups[0].top_ranked[:5]:
    flag = "flagged" if row["flagged"] else "not flagged"
    print(row["score_rank"], f"{row['composite_score']:.4f}", flag, row["values"])
```

It flags more rows than the 30 it planted (52 when this was written): at the default
`scoring.contamination = "auto"` nobody chose that count. The planted rows are what
lead the ranking, which is why the ranking, not the count, is the result to read.

---

## What it does not do

The short version; every point is expanded, with where it is measured, in
[Limitations](https://github.com/tarne75/sorethumb/blob/main/docs/limitations.md).

- **It ranks; it does not count.** The flagged number is a review shortlist, not an
  estimate of how many anomalies your data holds. By default its size is set by the
  detectors' own boundaries, not by you.
- **It is unsupervised.** Detectors fit on the same data they score, with no
  held-out normal reference, and the slower ones fit on a row sample.
- **Scores are rankings, not probabilities,** and two independently fitted runs
  are not on a common scale. Reuse a fitted run (`sorethumb score --from-run`), or
  set an anchor run for `sorethumb history`, to compare on one scale.
- **Explanations are exact only for ECOD and HBOS.** The rest are model-specific
  or heuristic, labelled as such, and a missing `explain` extra downgrades some.
- **A workspace is executable, not just data.** Persisted models are pickles; open
  only a workspace you created or fully trust.
- **One machine, local storage.** It runs on Linux, macOS and Windows, keeps its
  workspace on a local drive (not a network share or a synced folder), and is
  measured up to 1,000,000 rows; larger datasets are untested.

---

## Is it the right tool?

Use it where a ranked shortlist for a person to review is the goal: on a
single machine (Linux, macOS or Windows, see [Supported platforms](#supported-platforms)),
over data that fits [the scale guide](https://github.com/tarne75/sorethumb/blob/main/docs/scale.md), with someone able to judge
whether a flagged record is actually interesting.

- **Do not use it as the only control where a missed anomaly is unacceptable.**
  It is unsupervised and ranks statistical oddity, so it can miss exactly the
  record you care about. Where missing one has serious consequences, use a
  labelled, supervised or domain-specific control and treat this as an extra
  signal at most.
- **Validate before you rely on error rates.** If a decision depends on how
  many flagged records are false positives, or how many true anomalies are
  missed, measure that on labelled data and have a domain reviewer check the
  flagged records. Scores and flagged counts are not calibrated risk
  probabilities and are not a prevalence estimate, and should not be presented
  as either.
- **Compare over time on one scale.** For an absolute comparison across periods,
  fit one accepted reference run and score later data against it with
  `sorethumb score --from-run RUN_ID`, or set it as `history.anchor_run` so
  `sorethumb history` measures every period against it. Without one, treat
  `backfill` and independent-run history as relative diagnostics only.
- **Only reuse models from a workspace you made or fully trust.** A workspace is
  executable, not just data: do not accept a downloaded, shared or externally
  supplied workspace. See [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md#trust-boundary-workspaces-are-executable-not-just-data).

### Which setup?

| You want to | Do this |
| --- | --- |
| Look at a new dataset | Run it with the defaults and read the top-ranked rows; the flagged rows are where the detectors agree. |
| Review a fixed number of rows | Set a review budget: `scoring.contamination` to that share, with the `composite-agreement` policy. See [Recommended setup if you have a review budget](https://github.com/tarne75/sorethumb/blob/main/docs/ensemble-policy.md#recommended-setup-if-you-have-a-review-budget). |
| Know which policy works on your data | Run `sorethumb backtest` against labelled rows of your own data. |
| Compare periods on one scale | Fit one accepted run, then `sorethumb score --from-run RUN_ID`, or set it as `history.anchor_run`. See [trends](https://github.com/tarne75/sorethumb/blob/main/docs/trends.md). |
| Change how a column is treated | `sorethumb inspect` first, then the settings in [Adapting to your data](https://github.com/tarne75/sorethumb/blob/main/docs/adapting-to-your-data.md). |

---

## From the command line

No config file is needed. Pass a data file and sorethumb runs with its defaults,
writing everything under one `./sorethumb-workspace/` directory in the directory you
run it from:

```bash
cd /path/to/my-analysis
sorethumb run /path/to/data.csv
sorethumb anomalies --top 10      # the flagged rows, with reasons
sorethumb anomalies --ranked      # the top of the ranking, flagged or not
sorethumb report                  # the HTML report
```

For full control, `sorethumb init` writes a commented `sorethumb.toml`; set its
source and check the column plan before running:

```toml
[source]
uri = "/absolute/path/to/your/data.csv"    # also .parquet, .json, and http(s):// URLs
```

```bash
sorethumb inspect                 # how each column will be treated; no models trained
sorethumb run
```

> **The workspace is executable, not just data.** Its `models/` directory holds
> fitted estimators as `joblib`/pickle files, which run arbitrary code when loaded.
> Only open a workspace you created or fully trust. See
> [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md).

[Getting started](https://github.com/tarne75/sorethumb/blob/main/docs/getting-started.md) walks through both, with the output you will
see and every file format it reads.

---

## Supported platforms

| OS | Python | Tested in CI on every change |
| --- | --- | --- |
| Linux (x86-64) | 3.11–3.14 | unit, contract, integration, property, packaging (`ubuntu-latest`) |
| macOS (Apple silicon) | 3.11–3.14 | unit, contract, integration (`macos-latest`) |
| Windows 10/11 (x64) | 3.11–3.14 | unit, contract, integration, and installing the built wheel and sdist and running them from PowerShell (`windows-latest`) |

Unit and contract tests run on every listed Python version; the integration suite runs
on 3.12 in the release gate and on 3.11 and 3.14 daily.
[Supported platforms](https://github.com/tarne75/sorethumb/blob/main/docs/platforms.md) has the detail, and what is not
supported (a workspace on a network share or synced folder, among others).

---

## Documentation

The [documentation index](https://github.com/tarne75/sorethumb/blob/main/docs/index.md) is the map. The pages a new user needs first:

- [Getting started](https://github.com/tarne75/sorethumb/blob/main/docs/getting-started.md): the command line, step by step.
- [Adapting to your data](https://github.com/tarne75/sorethumb/blob/main/docs/adapting-to-your-data.md): what to change when the defaults do not fit.
- [Limitations](https://github.com/tarne75/sorethumb/blob/main/docs/limitations.md): the full list.
- [Ensemble policy](https://github.com/tarne75/sorethumb/blob/main/docs/ensemble-policy.md): what decides which rows are flagged, and the evidence.
- [Explanations](https://github.com/tarne75/sorethumb/blob/main/docs/explanations.md): what each reason means.
- [Benchmark results](https://github.com/tarne75/sorethumb/blob/main/docs/benchmarks.md) and the [scale guide](https://github.com/tarne75/sorethumb/blob/main/docs/scale.md): how well it works, and what it costs.
- [CLI reference](https://github.com/tarne75/sorethumb/blob/main/docs/cli_reference.md) and [configuration reference](https://github.com/tarne75/sorethumb/blob/main/docs/configuration.md).
- [Security policy](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md) and [contributing](https://github.com/tarne75/sorethumb/blob/main/CONTRIBUTING.md).
