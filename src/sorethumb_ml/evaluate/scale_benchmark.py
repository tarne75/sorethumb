"""End-to-end scale benchmark: what a default run costs at practical row counts.

The accuracy suites fit small synthetic frames, and the memory-budget tests isolate the
feature-matrix path with explanations off. Neither says what a user's
``sorethumb run data.csv`` costs on 100,000 or 1,000,000 rows. This harness does:

- **The user's run.** The config is built by the CLI's own ``_load_config`` from a bare
  data path, in an empty directory, exactly as a zero-config ``sorethumb run data.csv``
  builds it: default detectors, explanations on, HTML and CSV reports.
- **Realistic data.** A mixed transaction-like table: skewed, correlated and
  zero-inflated numerics, low- and high-cardinality categoricals, a timestamp and short
  free text, with 0.5% planted anomalies whose labels stay outside the data.
- **One fresh process per measurement** (``multiprocessing`` spawn), so the OS-tracked
  peak RSS belongs to that run alone. Data is generated beforehand in the parent.
- **Where the time goes.** The pipeline's own stage boundaries are wrapped inside the
  child process: loading, feature fitting, feature application, each detector's
  training and, separately, its scoring of every row, explanations, result writes and
  the report.
  Each stage records *exclusive* time (a nested stage's time is subtracted from its
  parent), so the stages sum to the measured total plus a small unattributed remainder,
  which is reported rather than hidden.
- **What a user sees afterwards.** Feature width, the rows each detector actually
  trained on (its sample cap), rows explained, output sizes on disk, and as a sanity
  check the flags' precision and recall against the planted anomalies.
- **Scoring new data.** A second, same-sized file is scored with ``score_forward``
  against the fitted run, in its own process: the cost of a recurring production run.
"""

from __future__ import annotations

import json
import multiprocessing
import platform
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import polars as pl

SCALE_SIZES: tuple[int, ...] = (10_000, 100_000, 1_000_000)
ANOMALY_FRACTION = 0.005

# The pipeline functions wrapped for stage timing: (module attribute, stage name).
# tests/unit/evaluation/test_scale_benchmark.py pins that each still exists, so a
# rename fails a unit test rather than silently dropping a stage from the benchmark.
# Detector training and scoring are timed per detector instead (``fit:<name>`` and
# ``score:<name>``), by wrapping ``_fit_detector`` and ``score_and_flag`` in ``_instrument``.
TIMED_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("apply_feature_plan", "feature-apply"),
    ("score_with_existing", "score-forward-setup"),
    ("_explain_group", "explain"),
    ("write_results", "write-results"),
    ("write_explanations", "write-results"),
    ("write_attributions", "write-results"),
    ("_render_report_and_get_status", "report"),
)
# The pipeline's own ``_timed_stage`` contexts kept as stages (others count as "other").
TIMED_STAGE_NAMES: frozenset[str] = frozenset({"load-dataset", "feature-fit"})

_MERCHANTS = 400
_TEXT_TEMPLATES = (
    "card payment",
    "online order",
    "subscription renewal",
    "refund issued",
    "in-store purchase",
    "transfer to savings",
)


# ── Data ─────────────────────────────────────────────────────────────────────


def make_scale_frame(n_rows: int, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    """Return ``(frame, labels)``: a mixed transaction-like table with planted anomalies.

    15 columns: an ``id``, a timestamp, seven numerics (lognormal amounts, a count, a
    zero-inflated discount, correlated sensor-like readings), four categoricals (2, 3, 6
    and 400 levels, the last Zipf-distributed) and a short free-text column. Anomalies
    (``ANOMALY_FRACTION`` of rows) are a mix of extreme amounts, an amount that breaks its
    usual relation to quantity and price, a merchant never seen in normal rows at 3 a.m., and a joint
    shift in the correlated readings. ``labels`` is 1 for a planted anomaly, by row.
    """
    rng = np.random.default_rng(seed)
    n = n_rows
    start = datetime(2025, 1, 1, tzinfo=UTC)
    # Days across a year; hours concentrated on the working day.
    days = rng.integers(0, 365, n)
    hours = np.clip(rng.normal(13.0, 3.0, n), 0, 23.99)
    seconds = days * 86400 + hours * 3600
    ts = [start + timedelta(seconds=float(x)) for x in seconds]

    quantity = rng.poisson(3, n) + 1
    unit_price = rng.lognormal(3.0, 0.6, n)
    amount = quantity * unit_price * rng.normal(1.0, 0.03, n)
    discount = np.where(rng.random(n) < 0.85, 0.0, rng.uniform(1, 30, n))
    base = rng.normal(0, 1, n)
    temp = 20 + 2 * base + rng.normal(0, 0.5, n)
    pressure = 1013 + 5 * base + rng.normal(0, 1.0, n)
    latency_ms = rng.gamma(2.0, 40.0, n)

    zipf = 1.0 / np.arange(1, _MERCHANTS + 1) ** 1.1
    merchant = rng.choice(np.array([f"m{r:03d}" for r in range(_MERCHANTS)]), n, p=zipf / zipf.sum())
    region = rng.choice(
        ["north", "south", "east", "west", "central", "overseas"], n, p=[0.25, 0.25, 0.2, 0.2, 0.08, 0.02]
    )
    channel = rng.choice(["web", "app", "store"], n, p=[0.5, 0.35, 0.15])
    is_member = rng.random(n) < 0.4
    note = np.array(_TEXT_TEMPLATES)[rng.integers(0, len(_TEXT_TEMPLATES), n)]

    labels = np.zeros(n, dtype=np.int8)
    n_anom = max(4, round(n * ANOMALY_FRACTION))
    idx = rng.choice(n, n_anom, replace=False)
    kinds = np.array_split(idx, 4)
    amount[kinds[0]] *= rng.uniform(40, 80, len(kinds[0]))
    amount[kinds[1]] = quantity[kinds[1]] * unit_price[kinds[1]] * rng.uniform(5, 9, len(kinds[1]))
    # A merchant never seen in normal rows, at 3 a.m.
    merchant = merchant.astype(object)
    merchant[kinds[2]] = "m_unseen"
    for i in kinds[2]:
        ts[i] = ts[i].replace(hour=3)
    temp[kinds[3]] += 9
    pressure[kinds[3]] -= 25
    labels[idx] = 1

    frame = pl.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "event_time": ts,
            "amount": amount,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "temp": temp,
            "pressure": pressure,
            "latency_ms": latency_ms,
            "merchant": merchant,
            "region": region,
            "channel": channel,
            "is_member": is_member,
            "note": note,
        }
    )
    return frame, labels


# ── Stage timing ─────────────────────────────────────────────────────────────


@dataclass
class StageClock:
    """Exclusive wall time and peak-memory growth per stage, for nested stages.

    ``enter``/``exit`` keep a stack: when a stage ends, its elapsed time is added to
    its own total and subtracted from the enclosing stage's, so each stage holds only
    the time spent in it and not in stages it called. ``peak_growth_mb`` records how much
    the process's peak resident memory rose during each stage (exclusive in the same way),
    which shows which stage sets the run's peak.
    """

    clock: Callable[[], float] = time.perf_counter
    memory: Callable[[], float | None] = lambda: None
    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    peak_growth_mb: dict[str, float] = field(default_factory=dict)
    _stack: list[list[Any]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block as stage *name*."""
        frame: list[Any] = [name, self.clock(), 0.0, self.memory(), 0.0]
        self._stack.append(frame)
        try:
            yield
        finally:
            self._stack.pop()
            elapsed = self.clock() - frame[1]
            mem_end = self.memory()
            growth = (mem_end - frame[3]) if mem_end is not None and frame[3] is not None else 0.0
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed - frame[2]
            self.peak_growth_mb[name] = self.peak_growth_mb.get(name, 0.0) + growth - frame[4]
            self.calls[name] = self.calls.get(name, 0) + 1
            if self._stack:
                self._stack[-1][2] += elapsed
                self._stack[-1][4] += growth

    def wrap(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        """Return *fn* timed as stage *name* on every call."""

        def timed(*args: Any, **kwargs: Any) -> Any:
            with self.stage(name):
                return fn(*args, **kwargs)

        timed.__wrapped__ = fn  # type: ignore[attr-defined]
        return timed


def _instrument(clock: StageClock, observed: dict[str, Any]) -> None:
    """Wrap the pipeline's stage functions (in this process only) to feed *clock* and *observed*."""
    import sorethumb_ml._pipeline as pipeline  # noqa: PLC0415
    from sorethumb_ml.store import models  # noqa: PLC0415

    for attr, stage in TIMED_FUNCTIONS:
        setattr(pipeline, attr, clock.wrap(getattr(pipeline, attr), stage))

    fit_detector = pipeline._fit_detector  # noqa: SLF001

    def observed_fit(det_name: str, X: np.ndarray, config: Any) -> Any:
        with clock.stage(f"fit:{det_name}"):
            out = fit_detector(det_name, X, config)
        observed.setdefault("detectors", {})[det_name] = {"train_rows": int(out[4]), "scored_rows": len(X)}
        observed["feature_width"] = int(X.shape[1])
        return out

    pipeline._fit_detector = observed_fit  # noqa: SLF001

    # Scoring every row, per detector: inside a run (via _fit_detector) and when
    # scoring forward (via store.models.score_with_existing).
    score_and_flag = getattr(pipeline, "score_and_flag")  # noqa: B009 -- not re-exported

    def timed_score(detector: Any, X: np.ndarray) -> Any:
        with clock.stage(f"score:{getattr(detector, 'name', type(detector).__name__)}"):
            return score_and_flag(detector, X)

    setattr(pipeline, "score_and_flag", timed_score)  # noqa: B010
    setattr(models, "score_and_flag", timed_score)  # noqa: B010

    original_stage = pipeline._timed_stage  # noqa: SLF001

    @contextmanager
    def timed_stage(name: str, threshold_seconds: float | None) -> Iterator[None]:
        with original_stage(name, threshold_seconds):
            if name in TIMED_STAGE_NAMES:
                with clock.stage(name):
                    yield
            else:
                yield

    pipeline._timed_stage = timed_stage  # noqa: SLF001


# ── One measured run, in its own process ─────────────────────────────────────


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def _output_sizes(workspace: Path) -> dict[str, float]:
    sizes = {
        child.name: round(_dir_size_mb(child), 2) for child in sorted(workspace.iterdir()) if child.is_dir()
    }
    sizes.update(
        {
            child.name: round(child.stat().st_size / 1e6, 2)
            for child in sorted(workspace.iterdir())
            if child.is_file()
        }
    )
    sizes["total"] = round(_dir_size_mb(workspace), 2)
    return sizes


def _measure_child(
    kind: str, data_path: str, workdir: str, cwd: str, source_run_id: str | None, queue: Any
) -> None:
    """Run (``kind="run"``) or score forward (``kind="score"``) once, instrumented; put a dict on *queue*."""
    import os  # noqa: PLC0415

    os.chdir(cwd)  # empty directory: no sorethumb.toml can leak in
    try:
        from sorethumb_ml import cli  # noqa: PLC0415
        from sorethumb_ml._pipeline import run_detection, score_forward  # noqa: PLC0415
        from sorethumb_ml.evaluate.pipeline_benchmark import _peak_memory_mb  # noqa: PLC0415

        baseline_mb = _peak_memory_mb()
        clock = StageClock(memory=_peak_memory_mb)
        observed: dict[str, Any] = {}
        _instrument(clock, observed)
        cfg = cli._load_config(  # noqa: SLF001 -- the CLI's own zero-config construction
            None,
            workdir=Path(workdir),
            seed=None,
            strict=None,
            log_level="WARNING",
            uri_override=data_path,
            detectors_override=None,
            json_output=True,
        )
        t0 = time.perf_counter()
        with clock.stage("other"):
            if kind == "run":
                result = run_detection(cfg)
            else:
                assert source_run_id is not None
                result = score_forward(cfg, source_run_id)
        total = time.perf_counter() - t0
        (group,) = result.groups
        queue.put(
            {
                "ok": group.status == "success",
                "error": group.error,
                "run_id": result.run_id,
                "total_seconds": total,
                "stage_seconds": dict(clock.seconds),
                "stage_peak_growth_mb": dict(clock.peak_growth_mb),
                "baseline_peak_mb": baseline_mb,
                "peak_mb": _peak_memory_mb(),
                "feature_width": observed.get("feature_width"),
                "detectors": observed.get("detectors", {}),
                "n_flagged": group.n_anomalies,
                "explain_status": group.explain_status,
                "n_explained": group.n_explained,
                "n_explain_unavailable": group.n_explain_unavailable,
                "results_path": str(group.results_path) if group.results_path else None,
                "report_status": result.report_status,
                "config": {
                    "detectors": [d.name for d in cfg.detectors if d.enabled],
                    "contamination": cfg.scoring.contamination,
                    "combination": cfg.scoring.combination,
                    "explain_max_rows": cfg.explain.max_rows,
                    "report_formats": list(cfg.report.formats),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        # _load_config attaches the workspace log file handler, as a CLI command does.
        from sorethumb_ml import cli  # noqa: PLC0415

        cli._detach_file_handlers()  # noqa: SLF001


def _run_in_child(
    kind: str, data_path: Path, workdir: Path, cwd: Path, source_run_id: str | None
) -> dict[str, Any]:
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_measure_child, args=(kind, str(data_path), str(workdir), str(cwd), source_run_id, queue)
    )
    proc.start()
    # Poll rather than block: a child that dies without reporting (killed for memory,
    # or unable to start) must fail the benchmark, not hang it.
    while True:
        try:
            out: dict[str, Any] = queue.get(timeout=5)
            break
        except Empty:
            if not proc.is_alive():
                raise RuntimeError(
                    f"{kind} process exited with code {proc.exitcode} without a result"
                ) from None
    proc.join()
    if not out.get("ok"):
        raise RuntimeError(f"{kind} failed: {out.get('error')}")
    return out


def _flag_quality(results_path: str, labels: np.ndarray) -> dict[str, float | None]:
    """Precision and recall of the flagged rows against the planted anomalies.

    Only flagged rows are persisted, so this is what a reviewer sees: of the rows put in
    front of them, how many were planted, and how many planted rows they were shown.
    """
    ids = pl.read_parquet(results_path, columns=["row_id"])["row_id"].to_numpy()
    hits = int(labels[ids].sum())
    return {
        "flag_precision": hits / len(ids) if len(ids) else None,
        "flag_recall": hits / int(labels.sum()),
    }


def run_scale_case(n_rows: int, scratch: Path, seed: int = 0, *, keep: bool = False) -> dict[str, Any]:
    """Generate data, run and score forward at *n_rows*, and return every measurement.

    Everything is written under *scratch*, removed afterwards unless *keep*.
    """
    case_dir = scratch / f"n{n_rows}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "cwd").mkdir(parents=True)
    workspace = case_dir / "workspace"
    try:
        fit_frame, fit_labels = make_scale_frame(n_rows, seed)
        new_frame, new_labels = make_scale_frame(n_rows, seed + 1)
        n_columns = fit_frame.width
        fit_csv, new_csv = case_dir / "data.csv", case_dir / "new.csv"
        fit_frame.write_csv(fit_csv)
        new_frame.write_csv(new_csv)
        del fit_frame, new_frame

        run = _run_in_child("run", fit_csv, workspace, case_dir / "cwd", None)
        run["flag_quality"] = _flag_quality(run.pop("results_path"), fit_labels)
        run_outputs = _output_sizes(workspace)

        score = _run_in_child("score", new_csv, workspace, case_dir / "cwd", run["run_id"])
        score["flag_quality"] = _flag_quality(score.pop("results_path"), new_labels)

        return {
            "n_rows": n_rows,
            "n_input_columns": n_columns,
            "input_csv_mb": round(fit_csv.stat().st_size / 1e6, 2),
            "n_planted_anomalies": int(fit_labels.sum()),
            "run": run,
            "run_outputs_mb": run_outputs,
            "score_forward": score,
            "workspace_after_score_mb": round(_dir_size_mb(workspace), 2),
        }
    finally:
        if not keep:
            shutil.rmtree(case_dir, ignore_errors=True)


def environment() -> dict[str, Any]:
    """Provenance for a published result: machine, Python, library versions, commit."""
    import importlib.metadata as md  # noqa: PLC0415
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    def version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "not installed"

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        commit += "+dirty" if dirty.strip() else ""
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "packages": {
            name: version(name)
            for name in ("sorethumb-ml", "numpy", "scipy", "scikit-learn", "polars", "shap", "numba")
        },
    }


def run_scale_benchmark(sizes: tuple[int, ...], scratch: Path, seed: int = 0) -> dict[str, Any]:
    """Run every size in *sizes* (smallest first) and return the full result document."""
    cases = []
    for n in sorted(sizes):
        print(f"scale benchmark: {n:,} rows", flush=True)
        cases.append(run_scale_case(n, scratch, seed))
    return {"schema_version": 1, "seed": seed, "environment": environment(), "cases": cases}


# ── Rendering ────────────────────────────────────────────────────────────────


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def render_markdown(doc: dict[str, Any]) -> str:
    """Render the result document as the tables published in docs/benchmarks.md."""
    env = doc["environment"]
    cases = doc["cases"]
    pk = env["packages"]
    lines = [
        f"Generated {env['generated_at']} at commit `{env['commit']}` on {env['platform']} "
        f"({env['cpu_count']} CPUs), Python {env['python']}, sorethumb {pk['sorethumb-ml']}, "
        f"numpy {pk['numpy']}, scikit-learn {pk['scikit-learn']}, polars {pk['polars']}, shap {pk['shap']}. "
        f"Seed {doc['seed']}; one run per size. Full results: "
        "[`benchmark-results/scale.json`](benchmark-results/scale.json).",
        "",
    ]
    cols = [f"{c['n_rows']:,} rows" for c in cases]
    header = "| | " + " | ".join(cols) + " |"
    rule = "| --- |" + " --- |" * len(cases)

    def row(label: str, values: list[str]) -> str:
        return f"| {label} | " + " | ".join(values) + " |"

    run = [c["run"] for c in cases]
    score = [c["score_forward"] for c in cases]
    det_names = list(run[0]["detectors"])
    lines += ["**`sorethumb run` (fit, score, explain, report)**", "", header, rule]
    lines.append(row("Input CSV", [f"{c['input_csv_mb']:.1f} MB" for c in cases]))
    lines.append(row("Elapsed", [_fmt_s(r["total_seconds"]) for r in run]))
    lines.append(row("Peak memory (RSS)", [f"{r['peak_mb']:,.0f} MB" for r in run]))
    lines.append(
        row("of which: Python with sorethumb imported", [f"{r['baseline_peak_mb']:,.0f} MB" for r in run])
    )
    lines.append(row("Feature width", [str(r["feature_width"]) for r in run]))
    for name in det_names:
        lines.append(
            row(
                f"`{name}` trained on",
                [
                    f"{r['detectors'][name]['train_rows']:,} of {r['detectors'][name]['scored_rows']:,}"
                    for r in run
                ],
            )
        )
    lines.append(row("Rows flagged", [f"{r['n_flagged']:,}" for r in run]))
    lines.append(row("Rows explained", [f"{r['n_explained']:,}" for r in run]))
    lines.append(row("Workspace on disk", [f"{c['run_outputs_mb']['total']:,.1f} MB" for c in cases]))
    lines.append(row("Flags vs planted anomalies (precision / recall)", [_quality(r) for r in run]))
    lines += [
        "",
        "**Where the run's time and memory go** (exclusive seconds per stage, and how much the stage "
        "raised the process's peak memory)",
        "",
        header,
        rule,
    ]
    for s in _ordered_stages({s for r in run for s in r["stage_seconds"]}):
        lines.append(
            row(
                _stage_label(s),
                [
                    f"{r['stage_seconds'].get(s, 0.0):.1f} s · +{r['stage_peak_growth_mb'].get(s, 0.0):,.0f} MB"
                    for r in run
                ],
            )
        )
    lines += ["", "**`sorethumb score --from-run` on a new file of the same size**", "", header, rule]
    lines.append(row("Elapsed", [_fmt_s(r["total_seconds"]) for r in score]))
    lines.append(row("Peak memory (RSS)", [f"{r['peak_mb']:,.0f} MB" for r in score]))
    lines.append(row("Explanations", [_fmt_s(r["stage_seconds"].get("explain", 0.0)) for r in score]))
    lines.append(row("Flags vs planted anomalies (precision / recall)", [_quality(r) for r in score]))
    return "\n".join(lines) + "\n"


def _quality(r: dict[str, Any]) -> str:
    q = r["flag_quality"]
    precision = "n/a" if q["flag_precision"] is None else f"{q['flag_precision']:.2f}"
    return f"{precision} / {q['flag_recall']:.2f}"


_STAGE_ORDER = [
    "load-dataset",
    "feature-fit",
    "feature-apply",
    "fit:",
    "score-forward-setup",
    "score:",
    "explain",
    "write-results",
    "report",
    "other",
]
_STAGE_LABELS = {
    "load-dataset": "Load and parse",
    "feature-fit": "Profile and fit features",
    "feature-apply": "Build the feature matrix",
    "score-forward-setup": "Load models, check the plan",
    "explain": "Explanations",
    "write-results": "Write results",
    "report": "Report",
    "other": "Other (bookkeeping, unattributed)",
}


def _stage_key(stage: str) -> tuple[int, str]:
    prefix = stage.split(":", 1)[0] + ":" if ":" in stage else stage
    return (_STAGE_ORDER.index(prefix) if prefix in _STAGE_ORDER else len(_STAGE_ORDER), stage)


def _ordered_stages(stages: set[str]) -> list[str]:
    return sorted(stages, key=_stage_key)


def _stage_label(stage: str) -> str:
    if stage.startswith("fit:"):
        return f"Train `{stage[4:]}`"
    if stage.startswith("score:"):
        return f"Score every row: `{stage[6:]}`"
    return _STAGE_LABELS.get(stage, stage)


def write_results(doc: dict[str, Any], json_path: Path, markdown_path: Path | None) -> None:
    """Write the JSON document, and splice the rendered tables between markers in *markdown_path*."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if markdown_path is None:
        return
    start, end = "<!-- scale-results-start -->", "<!-- scale-results-end -->"
    text = markdown_path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise ValueError(f"{markdown_path} has no {start} / {end} markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    note = (
        "<!-- AUTO-GENERATED — do not edit manually; run "
        "`uv run python scripts/run_scale_benchmark.py` to regenerate. -->\n\n"
    )
    markdown_path.write_text(head + start + "\n" + note + render_markdown(doc) + end + tail, encoding="utf-8")
