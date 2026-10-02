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
| `--workdir PATH` | `-w` | `run.workdir`, else `./sorethumb-workspace/` | Workspace root; overrides `run.workdir` in the config. See [Where the workspace lives](#where-the-workspace-lives). |
| `--log-level STR` | | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `--seed INT` | | from config | Random seed; overrides `run.seed`. |
| `--strict / --no-strict` | | off | Promote all library warnings to errors. |
| `--json` | | off | Emit machine-readable JSON to stdout instead of a rich table. |
| `--dry-run` | | off | Print the planned work. For `backfill` / `workspace` this writes nothing; for `run` see its own row below. |

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
`./sorethumb-workspace/logs/sorethumb.log`, since workdir defaults to a
dedicated `./sorethumb-workspace/` directory rather than the current
directory itself (see [Where the workspace lives](#where-the-workspace-lives)).

---

## Where the workspace lives

Every artefact sorethumb writes — the SQLite ledger, fitted models, results,
reports, logs and caches — goes under one **workspace** directory. Its location
is resolved the same way for every command, with or without a config file:

1. `--workdir PATH`, if given;
2. otherwise `run.workdir` from the config file, if it sets one;
3. otherwise the default, `./sorethumb-workspace/`.

The default is **relative to the current directory** you run the command from —
not to the config file's directory and not to the data file's. A relative
`run.workdir` in a config file is resolved the same way (against the current
directory), so run commands from the directory you used when you created the
workspace, or set an absolute path. (As a library, `run.workdir` is a required
field of `Config`; only the CLI supplies the default.)

*With no config file* (`sorethumb run data.csv`) nothing is written beside your
data. The workspace is created at the default location. On an interactive
terminal you are asked whether to save a config (to `--config`, default
`sorethumb.toml` in the current directory) for future runs; when stdin is not a
terminal (cron, CI, a pipe) the answer is "no" and a note on stderr says how to
save one. `--save-config` / `--no-save-config` decide without asking. *With a
config file that omits `run.workdir`*, the same default applies.

Commands that only read or maintain a workspace — `anomalies`, `runs`, `show`,
`report`, `history`, `explain-plan RUN_ID` and `workspace ls/du/prune/vacuum/migrate/reset`
— don't need a config file either. With no `sorethumb.toml` and no `--config`,
they use `--workdir`, or the default `./sorethumb-workspace/`, so the follow-up
commands after a zero-config `run` find its workspace. `history` then reports on
the dataset and configuration of the workspace's latest run. Naming a config
with `--config` (or `SORETHUMB_CONFIG`) that doesn't exist is still an error
(exit code 2), as is any command that reads the data source (`run` without a
data file, `inspect`, `score`, `backfill`, `explain-plan` without a run ID).

`sorethumb init [path]` is the explicit route: it writes `path/sorethumb.toml`
and creates the workspace at `path/sorethumb-workspace/`, with `run.workdir` in the
new file already set to that absolute path.

If a `sorethumb.db` exists in the current directory (a workspace made under the
older default of `.`) and nothing names a workspace, sorethumb refuses to guess:
pass `--workdir .` to keep using it, or move its contents into
`./sorethumb-workspace/`.

### What is in a workspace — and why it is executable

```
sorethumb.db            SQLite ledger: runs, groups, history, artefact index
models/<run>/…          fitted feature plan, and per group and detector the
                        estimator and calibrator as joblib/pickle files, plus a manifest
results/<run>/<group>/  per-group results (Parquet)
reports/<run>/          rendered HTML / CSV / JSON reports
cache/  tmp/  logs/     downloaded-dataset and feature caches, scratch, rotating logs
```

The `models/` files make a workspace **executable, not just data**: unpickling
them is arbitrary code execution. `sorethumb score --from-run` and
`run.reuse_models` load them with no sandboxing, and the SHA-256 digests only
detect accidental corruption, not a crafted file. Treat a workspace you did not
create yourself like a script you downloaded — see
[SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md).

---

## Exit codes

Every command uses the same five codes. They are part of the CLI's stable
interface (pre-1.0 policy: a change to any of them is called out in the
CHANGELOG).

| Code | `kind` / `outcome` | Meaning |
|------|--------------------|---------|
| `0` | `ok` | Success. |
| `1` | `runtime` | Work was attempted and failed: a store or database error, a detector failure, every group failing, an error while rendering a requested report on its own, or a declined/aborted destructive command. |
| `2` | `preflight` | Rejected before any work: bad arguments, a missing or invalid config file, a source or schema that cannot be read (including a source file that does not exist), a plan that cannot be built, or a group selector (`--only-group`/`--group-filter`) that matches no group. |
| `3` | `not_found` | A run, workspace or persisted file that the command was asked to use does not exist (unknown run ID, no workspace at the configured path, a run with no persisted plan, `score --from-run` against a missing run). |
| `4` | `partial` | Results were produced but something is missing: some groups failed while at least one succeeded, `backfill` periods failed while others completed, or every group succeeded but the requested report could not be rendered. The completed work is valid and kept. |

Classification is by the type of the raised project error (each
`SorethumbError` subclass carries a `failure_kind`; a third-party subclass that
does not set one is `runtime`). `typer`'s own argument-parsing errors also exit `2`.

### Machine-readable failures

Every command with a `--json` flag emits **exactly one JSON document on stdout**
for every outcome, and never Rich text on stderr in that mode. A failure with no
result to report has the same three-key shape for all of them and for every
failure class:

```json
{"error": "Run not found: run_abc", "kind": "not_found", "exit_code": 3}
```

`kind` is `preflight`, `not_found` or `runtime`, and `exit_code` always equals the
process exit code. `run --json` and `score --json` that did complete (possibly
partially) print their normal result document instead, which carries `exit_code`
and `outcome` (`ok`, `partial`, `runtime` or `preflight`) alongside the usual
fields, so a caller can branch on one field whether the run finished cleanly or
not.

The run document counts rows **flagged for review** as `n_flagged`, both at the top
level (all groups) and per group. It is the size of the review shortlist at the
chosen budget, not an estimate of how many records are truly anomalous, which is
why the field is not called `n_anomalies`. There is no alias under the old name.

---

## `sorethumb init [path]`

Write a fully-commented `sorethumb.toml` starter file into `path` and create the
workspace at `path/sorethumb-workspace/` (the file's `run.workdir` is filled in
with that absolute path). This is the recommended onboarding path when you want
full control over every setting. If `sorethumb.toml` already exists in `path`,
`init` does nothing.

If the workspace cannot be created (for example a file is in the way, or the
location is not writable), `init` prints an error, **no success banner**, and
exits `1`. The config file is written before the workspace, so in that case it
already exists; the error message says explicitly that `sorethumb.toml` was
written. Remove it and re-run `init` once the problem is fixed, because `init`
never touches an existing `sorethumb.toml`. If the target directory itself cannot
be created or the file cannot be written, nothing is written at all.

```bash
sorethumb init                        # sorethumb.toml and sorethumb-workspace/ in the current directory
sorethumb init /path/to/my-analysis   # the same, in a new directory
```

After `init`, open `sorethumb.toml` and set `source.uri` to your data file,
then run `sorethumb inspect` to verify column classification before training.
The workspace holds pickled models, so only share or open one you trust — see
[Where the workspace lives](#where-the-workspace-lives).

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `path` | `.` | Directory that receives `sorethumb.toml` and the `sorethumb-workspace/` workspace (created if missing). |

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
# Zero-config — runs with all defaults, workdir = ./sorethumb-workspace/
sorethumb run /path/to/data.parquet

# Config-based
sorethumb run
sorethumb run --log-level DEBUG --force

# Override just the data file, keep everything else from the config
sorethumb run /path/to/new_data.csv

# Subset to specific groups
sorethumb run --only-group store_42 --only-group store_99
sorethumb run --group-filter "^store_(1|2|3)$"

# Override detector selection on the fly (updates sorethumb.toml automatically)
sorethumb run --detectors if,km,oc,lof
sorethumb run -d if,ecod,hbos
```

Groups that are already marked complete in the ledger are skipped unless
`--force` is passed. This makes repeated invocations cheap.

If `--only-group`/`--group-filter` matches none of the groups actually
discovered in the data, the run processes zero groups and fails outright
(exit code 2; `--json` output's `"group_selection_error"` names the reason, and its `"outcome"` is `preflight`)
rather than silently completing a no-op — no group/period history marker is
written either, so a later `sorethumb backfill` still sees the gap. This is
distinct from a genuinely empty source period (a period with no matching
rows at all), which is not treated as an error.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `data_file` | Optional path to a data file. Overrides `source.uri` in the config. When no `sorethumb.toml` exists, all settings default and workdir defaults to `./sorethumb-workspace/` in the current directory; on an interactive terminal you are asked whether to save a config for future runs (never when stdin is not a terminal, or with `--json`). |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `--workdir`, `--log-level`, `--seed`, `--strict` | — | See [Common options](#common-options). |
| `--force` | off | Re-run groups that are already complete in the ledger. |
| `--no-report` | off | Skip HTML/CSV/JSON report generation. |
| `--only-group STR` | — | Run only these group label(s). Repeatable. |
| `--group-filter REGEX` | — | Run only groups whose label matches this regex. |
| `--period YYYY-MM-DD` | — | Force a specific period label (for time-series datasets). |
| `--limit-groups INT` | — | Cap the number of groups processed, applied after `--only-group`/`--group-filter`. Groups are sorted by label first, so the same limit always keeps the same groups. Must be >= 1 when given (exit code 2 otherwise). |
| `--detectors STR`, `-d` | — | Comma-separated detector aliases, replacing the config list for this invocation only. Aliases: `if`=isolation_forest · `km`=kmeans_distance · `oc`=one_class_svm · `ecod` · `lof` · `hbos`. Full names also accepted. Never modifies `sorethumb.toml`. |
| `--save-config / --no-save-config` | ask on a terminal, else no | With `data_file` and no `sorethumb.toml`: save (or don't save) this run's settings to `sorethumb.toml` without asking. No effect when the config file already exists. |
| `--json` | off | Machine-readable JSON summary on stdout. |
| `--dry-run` | off | Resolve the plan and register the run, but fit no models. Still writes the workspace + schema migrations, the `dataset` / `dataset_snapshot` rows, and the `run` row (left in status `running`). Skips the feature plan, detector models, per-group results, history rows and the report. |

---

## `sorethumb score`

Score new data using a previous run's persisted FeaturePlan, per-detector
models and calibrators — nothing is re-fitted. The source run's calibrators map
the new scores onto its reference distribution, so results are comparable
across runs. Schema drift and library-version drift are detected per group
(`--strict` promotes both to errors). A new, distinct run is written
(`score_…` id) that records the source run.

Only `scoring`, `explain`, `report` and `run` settings can legally differ
from the source run — they really are recomputed fresh on every
`score --from-run` call. `columns`, `profiling`, `features`, and any enabled
detector's `params`/`train_row_cap` are baked into the source run's fitted
plan and persisted models; a config that claims different values there is
rejected outright (regardless of `--strict`), since score-forward has no way
to actually apply them — re-run `sorethumb run` instead if you need
different fit-time settings. `run_id`, `--json` output, and the rendered
report all show `source_run_id` for a score-forward run.

Loading the source run unpickles its persisted estimator and calibrator
files (`joblib`), which is code execution with no sandboxing — not safe
data loading. The SHA-256 file digests checked on load catch corruption or
a swapped file, not a deliberately malicious one. Only run this against a
workspace you created yourself or fully trust. See [SECURITY.md](../SECURITY.md).

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

Rebuild a run's `reports/<run_id>/index.html` (and its per-group CSV siblings)
from the persisted config, FeaturePlan and per-group results Parquet — no
inference is re-run. Use it after changing `[report]` configuration, or to
restore a report that was deleted. Re-rendering the same run reproduces the
same file.

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

Fill missing historical periods by running detection once per pending period.
Skipped automatically when no `time_column` is configured.

```bash
sorethumb backfill
sorethumb backfill --dry-run                    # print periods to be filled
sorethumb backfill --force-period 2026-08-01    # recompute a specific period
sorethumb backfill --max-periods 14             # cap depth
```

Each pending period is fitted and scored independently (a full `sorethumb run`
for that period's window) and self-calibrated. The resulting `sorethumb history`
trend therefore shows *relative* period-to-period movement, not an absolute
anomaly level on a shared scale — for that, score every period against one fixed
run with `sorethumb score --from-run RUN_ID`.

A failing period never stops the others. Periods whose run finished with failed
groups, and periods whose run raised a project error (for example an unreadable
source), are collected and summarised separately once every pending period has
been attempted, and the command then exits `1`. Neither kind is recorded as
complete, so the next `sorethumb backfill` retries them.

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

## `sorethumb config` sub-commands

### `sorethumb config check`

Validate a config file and report every error at once (not just the first).
Exit code `0` means the config is valid. `--json` prints the fully-resolved
config (defaults, env vars, and CLI overrides all applied) instead, with
credentials redacted — a source URI's embedded userinfo or signed-download
token is masked the same way it is before being persisted to the database;
an `auth_env_var` value is never included in `Config` in the first place, so
there is nothing to redact there.

```bash
sorethumb config check
sorethumb config check --config /other/path/sorethumb.toml
sorethumb config check --json
```

**Options:** `--config`, `--workdir`, `--json`

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

### `sorethumb config show <run_id>`

Display the exact config used for a past run — useful for reproducing or
adapting a previous experiment.

```bash
sorethumb config show abc12345678           # human-readable summary (default)
sorethumb config show abc12345678 --json    # raw JSON config
sorethumb config show abc12345678 --output repro.toml   # write as reusable sorethumb.toml
```

The default output shows a Rich table of detectors plus key settings
(source URI, workdir, seed, scoring). `--json` returns the full stored config.
`--output` reconstructs a minimal `sorethumb.toml` you can edit and re-run.

Exit code `3` if the run ID is not found in the workspace.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `run_id` | Run ID to inspect. Find run IDs with `sorethumb runs`. |

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--json` | | Print the raw stored config as indented JSON. |
| `--output PATH` | `-o` | Write a minimal `sorethumb.toml` to this path. |
| `--config`, `--workdir` | | See [Common options](#common-options). |

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

Only what sorethumb creates is deleted: `sorethumb.db` (with any
`-wal`/`-shm`/`-journal` files beside it) and the `cache/`, `logs/`,
`models/`, `reports/`, `results/` and `tmp/` directories. Anything else in the
directory is kept — so a `--workdir` pointed at a directory that already held
your own files leaves them in place — and is listed. The directory itself is
removed only when nothing else remains in it (exit 0 either way). A symlink
or Windows directory junction among those entries is removed, never followed,
and read-only files are deleted too.

The confirmation prompt shows the exact path to type. Any spelling of the same
directory is accepted: surrounding quotes (as Windows Explorer's "Copy as path"
adds), a trailing separator, `~`, and on Windows a different letter case or
forward slashes.

Refuses outright, regardless of `--yes`, unless the target actually opens
as a real sorethumb workspace (has a `sorethumb.db` marker) — and always
refuses a filesystem/drive root, your home directory, the current working
directory, a git repository root, or a suspiciously shallow path, even if
one of those happened to contain a workspace marker. A deletion failure
(e.g. a permissions error partway through) is reported and exits non-zero,
never silently ignored.

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
# Zero-config path — sorethumb offers to save the config (on a terminal)
sorethumb run --log-level INFO /data/transactions.parquet
sorethumb anomalies --top 50         # no config needed to read the results

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
sorethumb run --json | jq '.groups[] | select(.n_flagged > 0)'
sorethumb anomalies --top 100 --json | jq '.[] | {rank, score: .composite_score, r1: .reason_1}'
sorethumb runs --json | jq '.[] | select(.status == "failed") | .run_id'
```
