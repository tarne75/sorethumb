# CLI reference

Complete reference for every `sorethumb` command, argument, and option.

---

## Invocation

```bash
sorethumb [OPTIONS] COMMAND [ARGS...]

# via uv (recommended during development)
uv run sorethumb [OPTIONS] COMMAND [ARGS...]
```

**Global options** (before any command):

| Flag | Short | Description |
|------|-------|-------------|
| `--version` | `-V` | Print version and exit. |
| `--help` | | Show top-level help. |

---

## Common options

These options appear on most commands and behave identically everywhere:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config PATH` | `-c` | `sorethumb.toml` | Path to the config file. Env: `SORETHUMB_CONFIG`. |
| `--workdir PATH` | `-w` | from config | Workspace root; overrides `run.workdir` in the config. |
| `--log-level STR` | | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `--seed INT` | | from config | Random seed; overrides `run.seed`. |
| `--strict / --no-strict` | | off | Promote all library warnings to errors. |
| `--json` | | off | Emit machine-readable JSON to stdout instead of a rich table. |
| `--dry-run` | | off | Plan work and print what would happen without writing anything. |

---

## Environment variables

| Variable | Equivalent flag | Notes |
|----------|-----------------|-------|
| `SORETHUMB_CONFIG` | `--config` | Path to `sorethumb.toml`. |

---

## Log files

All commands write logs to both the console and a rotating file:

```
{workdir}/logs/sorethumb.log   (10 MB limit, 5 backups)
```

The file handler is created as soon as the config is loaded. When running
without a config file (`sorethumb run <data_file>`), the log is written to
`./logs/sorethumb.log` since workdir defaults to the current directory.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | One or more groups failed (returned by `run`, `score`, `backfill`). |
| `2` | Configuration or argument error; no work was attempted. |

---

## `sorethumb init [path]`

Create a workspace directory and write a fully-commented `sorethumb.toml`
starter file. This is the recommended onboarding path when you want full
control over every setting.

```bash
sorethumb init                        # initialise current directory
sorethumb init /path/to/my-workspace  # create and initialise a new directory
```

After `init`, open `sorethumb.toml` and set `source.uri` to your data file,
then run `sorethumb inspect` to verify column classification before training.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `path` | `.` | Directory to create and initialise. |

---

## `sorethumb inspect`

Profile the dataset and print the feature plan — every column's classification
and the reason it was classified that way — without training any models.

```bash
sorethumb inspect
sorethumb inspect --log-level DEBUG   # verbose profiling trace
```

Use this before `sorethumb run` to confirm that high-cardinality string columns
will be dropped, identifiers excluded, and categorical columns encoded as
expected. Edit `sorethumb.toml` and re-run `inspect` until the plan looks right.

**Options:** `--config`, `--workdir`, `--log-level`, `--seed`

---

## `sorethumb run [data_file]`

Run full anomaly detection: load data, build features, train detectors, score,
explain (SHAP), and write a report.

```bash
# Zero-config — runs with all defaults, workdir = current directory
sorethumb run /path/to/data.parquet

# Config-based
sorethumb run
sorethumb run --log-level DEBUG --force

# Override just the data file, keep everything else from the config
sorethumb run /path/to/new_data.csv

# Subset to specific groups
sorethumb run --only-group store_42 --only-group store_99
sorethumb run --group-filter "^store_(1|2|3)$"
```

Groups that are already marked complete in the ledger are skipped unless
`--force` is passed. This makes repeated invocations cheap.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `data_file` | Optional path to a data file. Overrides `source.uri` in the config. When no `sorethumb.toml` exists, all settings default and workdir defaults to `.`; you are prompted to save a config for future runs. |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level`, `--seed`, `--strict` | — | See [Common options](#common-options). |
| `--force` | off | Re-run groups that are already complete in the ledger. |
| `--no-report` | off | Skip HTML/CSV/JSON report generation. |
| `--only-group STR` | — | Run only these group label(s). Repeatable. |
| `--group-filter REGEX` | — | Run only groups whose label matches this regex. |
| `--period YYYY-MM-DD` | — | Force a specific period label (for time-series datasets). |
| `--limit-groups INT` | — | Cap the number of groups processed (reserved for future use). |
| `--json` | off | Machine-readable JSON summary on stdout. |
| `--dry-run` | off | Print planned work without writing anything. |

---

## `sorethumb score`

Score new data using a previous run's persisted models and calibrators.
The feature plan from the source run is applied without re-fitting, so scores
are comparable across time. Schema drift is detected per group.

```bash
sorethumb score --from-run abc12345
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--from-run STR` | **required** | Run ID whose models to reuse. |
| `--config`, `--workdir`, `--log-level`, `--seed`, `--strict` | — | See [Common options](#common-options). |
| `--no-report` | off | Skip report generation. |
| `--json` | off | Machine-readable JSON summary. |

---

## `sorethumb anomalies [run_id]`

Print the flagged rows from a completed run, ordered by rank (1 = most
anomalous), with SHAP-derived reason columns.

```bash
sorethumb anomalies                      # latest run, top 3 reasons, all rows
sorethumb anomalies abc12345             # specific run
sorethumb anomalies --top 20            # limit to 20 rows
sorethumb anomalies --reasons 5 --top 50
sorethumb anomalies --json | jq '.[] | {rank, score: .composite_score}'
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `run_id` | most recent | Run ID to inspect. |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level` | — | See [Common options](#common-options). |
| `--top INT` | 0 (all) | Show only the top-N anomalies. |
| `--reasons INT` | 3 | Number of reason columns to display. |
| `--json` | off | Machine-readable JSON on stdout. |

---

## `sorethumb runs`

List recent runs with their status, dataset, group counts, and wall-clock
duration. Useful for monitoring a scheduled pipeline.

```bash
sorethumb runs
sorethumb runs --limit 50
sorethumb runs --json | jq '.[] | select(.status == "failed")'
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level` | — | See [Common options](#common-options). |
| `--limit INT` | 20 | Maximum number of runs to display. |
| `--json` | off | Machine-readable JSON. |

---

## `sorethumb show <run_id>`

Show full detail for one run: status, config hash, group summary, and
feature plan overview.

```bash
sorethumb show abc12345
sorethumb show abc12345 --group store_42
sorethumb show abc12345 --json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `run_id` | **required** | Run ID to inspect. |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level` | — | See [Common options](#common-options). |
| `--group STR` | — | Show detail for a specific group key only. |
| `--json` | off | Machine-readable JSON. |

---

## `sorethumb report [run_id]`

Re-render HTML/CSV/JSON reports from already-persisted results without
recomputing inference. Useful after changing `[report]` configuration.

```bash
sorethumb report              # re-render the latest run
sorethumb report abc12345
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `run_id` | most recent | Run ID to re-render. |

**Options:** `--config`, `--workdir`, `--log-level`

---

## `sorethumb explain-plan [run_id]`

Print the full `FeaturePlan` for a completed run: every column decision —
what was dropped, encoded, derived, and why — along with any columns that
were excluded by correlation reduction.

```bash
sorethumb explain-plan
sorethumb explain-plan abc12345 --json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `run_id` | most recent | Run ID whose plan to display. |

**Options:** `--config`, `--workdir`, `--log-level`, `--json`

---

## `sorethumb history`

Show rolling-window anomaly-rate trends over time for the configured dataset.
Requires a `time_column` in the config and at least `[history].bootstrap_periods`
of completed runs.

```bash
sorethumb history
sorethumb history --window 7 --window 28   # two rolling windows
sorethumb history --group store_42
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level` | — | See [Common options](#common-options). |
| `--window INT` | from config | Rolling window size(s) in periods. Repeatable. |
| `--group STR` | — | Limit trend display to a specific group. |

---

## `sorethumb backfill`

Fill missing historical periods using the most recent run's persisted models.
Skipped automatically when no `time_column` is configured.

```bash
sorethumb backfill
sorethumb backfill --dry-run                    # print periods to be filled
sorethumb backfill --force-period 2026-08-01    # recompute a specific period
sorethumb backfill --max-periods 14             # cap depth
```

By default, reuses the most recent run's models (equivalent to `--reuse-models`)
so trend scores are comparable across backfilled periods. Emits a warning and
falls back to self-calibration when no source run exists.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level`, `--seed`, `--strict` | — | See [Common options](#common-options). |
| `--force-period STR` | — | Recompute this period label even if already complete. Repeatable. |
| `--max-periods INT` | from config | Override `history.max_backfill_periods`. |
| `--dry-run` | off | Print what would be backfilled without running it. |

---

## `sorethumb detectors`

List all registered anomaly detectors — built-in and any third-party extensions
installed in the current environment.

```bash
sorethumb detectors
sorethumb detectors --json
```

**Options:** `--json`

---

## `sorethumb benchmark`

Run the evaluation harness against labelled benchmark datasets. Requires the
`[benchmark]` optional dependency group (`uv sync --extra benchmark`).

```bash
sorethumb benchmark --log-level DEBUG
```

**Options:** `--log-level`

---

## `sorethumb config` sub-commands

### `sorethumb config check`

Validate a config file and report every error at once (not just the first).
Exit code `0` means the config is valid.

```bash
sorethumb config check
sorethumb config check --config /other/path/sorethumb.toml
```

**Options:** `--config`, `--workdir`

---

### `sorethumb config schema`

Emit the full JSON schema for `sorethumb.toml` — useful for editor
autocompletion or for validating configs programmatically.

```bash
sorethumb config schema                   # print to stdout
sorethumb config schema --output schema.json
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output PATH` | `-o` | Write schema to a file instead of stdout. |

---

## `sorethumb workspace` sub-commands

### `sorethumb workspace ls`

List runs, datasets, and artefact counts in the workspace.

```bash
sorethumb workspace ls
sorethumb workspace ls --json
```

**Options:** `--config`, `--workdir`, `--log-level`, `--json`

---

### `sorethumb workspace du`

Show disk usage broken down by regenerable artefacts (models, feature matrices)
versus non-regenerable artefacts (results, ledger). Helps decide whether
`prune` is worth running.

```bash
sorethumb workspace du
```

**Options:** `--config`, `--workdir`, `--log-level`

---

### `sorethumb workspace prune`

Remove regenerable artefacts and failed runs older than `--days`. Always removes
files and database rows together — never one without the other.

```bash
sorethumb workspace prune --dry-run     # preview what would be removed
sorethumb workspace prune               # remove artefacts older than 90 days
sorethumb workspace prune --days 30
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level` | — | See [Common options](#common-options). |
| `--days INT` | 90 | Retention window; artefacts older than this are removed. |
| `--dry-run` | off | Preview without removing anything. |

---

### `sorethumb workspace vacuum`

Run SQLite `VACUUM` on the workspace database and reconcile any orphan files
that have no corresponding database row.

```bash
sorethumb workspace vacuum
```

**Options:** `--config`, `--workdir`, `--log-level`

---

### `sorethumb workspace migrate`

Apply pending schema migrations to the workspace database. Run this after
upgrading sorethumb to a new minor version.

```bash
sorethumb workspace migrate
```

**Options:** `--config`, `--workdir`, `--log-level`

---

### `sorethumb workspace reset`

**Destructively delete all workspace data** — runs, models, results, ledger,
logs. Requires interactive confirmation of the workspace path, or `--yes` for
unattended use.

```bash
sorethumb workspace reset
sorethumb workspace reset --yes   # skip confirmation (CI / scripted teardown)
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level` | — | See [Common options](#common-options). |
| `--yes` | off | Skip interactive confirmation. |

---

## Typical workflows

### First run on a new dataset

```bash
# Zero-config path — sorethumb prompts to save the config
sorethumb run --log-level INFO /data/transactions.parquet

# Config-based path — full control from the start
sorethumb init ~/analysis/transactions
cd ~/analysis/transactions
# edit sorethumb.toml: set source.uri
sorethumb inspect                    # verify column treatment
sorethumb run --log-level INFO
sorethumb anomalies --top 50
```

### Iterate on column config without retraining

```bash
# edit sorethumb.toml: adjust [columns] or [profiling]
sorethumb inspect                    # re-check column plan
sorethumb run --force                # re-run (force because ledger has a result)
```

### Score new data against existing models

```bash
# Get the run ID to reuse
sorethumb runs --limit 5

# Score new data with that run's models
sorethumb score --from-run abc12345
```

### Maintain a scheduled pipeline

```bash
# Daily cron
sorethumb run
sorethumb backfill --dry-run         # check for gaps
sorethumb backfill                   # fill any gaps
sorethumb workspace prune --days 30  # keep workspace lean
```

### Machine-readable output for downstream tools

```bash
sorethumb run --json | jq '.groups[] | select(.n_anomalies > 0)'
sorethumb anomalies --top 100 --json | jq '.[] | {rank, score: .composite_score, r1: .reason_1}'
sorethumb runs --json | jq '.[] | select(.status == "failed") | .run_id'
```
