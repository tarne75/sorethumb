"""Benchmark harness: dataset registry × detector configurations.

Produces a Markdown and CSV comparison table of ROC-AUC, average precision,
precision@k, recall@k, F1, and wall-clock fit/score times per detector per dataset.

Requires the ``[benchmark]`` extra (``pip install sorethumb[benchmark]``).
Run via ``pytest -m benchmark`` or ``sorethumb benchmark``.

Design principles
-----------------
- Each dataset entry records its licence and provenance; no dataset is used
  silently.
- Network fetches are cached under ``cache_dir`` so repeated runs are cheap.
- Synthetic generators with known injected anomalies are always available and
  exercise the full code path without network access.
- Real datasets (KDDCup99, Covtype, ADBench) are fetched on first use and
  cached; they never re-download on a warm cache.
- Metrics are computed via ``evaluate_scores`` — the same function used in
  production so the numbers are directly comparable.
- precision@k, recall@k, and F1 are evaluated at a fixed review budget
  (``_REVIEW_BUDGET``), never at the dataset's true contamination. Deriving k
  from the labels being scored (``k = round(n_total * y.mean())``) makes
  ``k == n_positives``, which forces precision@k, recall@k, and F1 to be the
  same number (see ``evaluate_scores``) — a leak of the answer into the
  operating point, not three independent metrics.
- Each (dataset, detector) pair is run over several seeds and reported as
  mean ± standard deviation, so a single lucky/unlucky draw doesn't read as
  a stable result.
- Every run records ``BenchmarkMetadata`` (timestamp, platform, Python,
  sorethumb, numpy, scipy, scikit-learn versions) alongside the table, so a
  published number can be tied to the environment that produced it.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _fmt_or_na(value: float) -> str:
    """Format a float as fixed-point, or "n/a" when it's NaN (undefined)."""
    return "n/a" if math.isnan(value) else f"{value:.4f}"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass
class DatasetEntry:
    """A registered benchmark dataset."""

    name: str
    description: str
    licence: str
    provenance: str
    contamination: float  # true anomaly fraction

    def load(self, cache_dir: Path, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) where y=1 means anomaly.

        ``seed`` lets repeated-seed benchmark runs draw a fresh sample where
        the dataset supports it (see ``SyntheticEntry``); fixed real datasets
        ignore it and always return the same array.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


@dataclass
class SyntheticEntry(DatasetEntry):
    """Gaussian cluster with injected point anomalies — always available, no network."""

    n_normal: int = 1000
    n_anomaly: int = 50
    n_features: int = 8
    seed: int = 0

    def load(self, cache_dir: Path, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:  # noqa: ARG002
        """Generate Gaussian cluster with injected uniform-noise anomalies."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        X_normal = rng.multivariate_normal(
            mean=np.zeros(self.n_features),
            cov=np.eye(self.n_features),
            size=self.n_normal,
        )
        X_anom = rng.uniform(-6, -4, size=(self.n_anomaly, self.n_features))
        X = np.vstack([X_normal, X_anom]).astype(np.float32)
        y = np.concatenate([np.zeros(self.n_normal), np.ones(self.n_anomaly)]).astype(int)
        perm = rng.permutation(len(X))
        return X[perm], y[perm]


@dataclass
class SklearnEntry(DatasetEntry):
    """Dataset fetched from sklearn's built-in toy datasets."""

    sklearn_name: str = ""  # "kddcup99" | "covtype"
    subset: str | None = None  # e.g. "SA" for KDDCup99

    def load(self, cache_dir: Path, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:  # noqa: ARG002
        """Fetch dataset from sklearn and return (X, y) with y=1 for anomalies.

        ``seed`` is accepted for interface parity with ``DatasetEntry.load``
        but ignored: this is a fixed, cached real dataset, not a generator.
        """
        from sklearn import datasets  # noqa: PLC0415

        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.sklearn_name == "kddcup99":
            bunch = datasets.fetch_kddcup99(
                subset=self.subset,
                shuffle=False,
                random_state=0,
                data_home=str(cache_dir),
                as_frame=False,
                percent10=True,
            )
            X_raw = bunch.data
            # Drop non-numeric columns (protocol type, service, flag)
            numeric_cols = [
                i for i in range(X_raw.shape[1]) if X_raw.dtype == np.float64 or _is_numeric_col(X_raw, i)
            ]
            X = X_raw[:, numeric_cols].astype(np.float32) if numeric_cols else X_raw.astype(np.float32)
            # "normal." records are negative; everything else is an anomaly
            y = np.where(bunch.target == b"normal.", 0, 1).astype(int)
        elif self.sklearn_name == "covtype":
            bunch = datasets.fetch_covtype(
                shuffle=False,
                random_state=0,
                data_home=str(cache_dir),
                as_frame=False,
            )
            X = bunch.data.astype(np.float32)
            # Class 2 is the majority (normal); class 4 is rare (treat as anomaly).
            # This is the standard ODDS formulation.
            y = np.where(bunch.target == 4, 1, 0).astype(int)
        else:
            raise ValueError(f"Unknown sklearn dataset: {self.sklearn_name!r}")

        return X, y


def _is_numeric_col(arr: np.ndarray, col: int) -> bool:
    try:
        arr[:, col].astype(np.float32)
        return True
    except (ValueError, TypeError):
        return False


# The canonical dataset registry.
DATASETS: list[DatasetEntry] = [
    SyntheticEntry(
        name="synthetic_gaussian",
        description="Gaussian cluster (n=1000) with 50 uniform-noise anomalies in 8 features.",
        licence="Public domain (generated)",
        provenance="sorethumb.evaluate.benchmark.SyntheticEntry (no network access)",
        contamination=50 / 1050,
        n_normal=1000,
        n_anomaly=50,
        n_features=8,
    ),
    SyntheticEntry(
        name="synthetic_highd",
        description="High-dimensional Gaussian cluster (n=2000, d=32) with 100 anomalies.",
        licence="Public domain (generated)",
        provenance="sorethumb.evaluate.benchmark.SyntheticEntry (no network access)",
        contamination=100 / 2100,
        n_normal=2000,
        n_anomaly=100,
        n_features=32,
    ),
    SklearnEntry(
        name="kddcup99_sa",
        description="KDDCup 1999 network intrusion (SA subset, 10 % sample). Anomalies = non-normal traffic.",
        licence="KDD Cup 1999 data is freely available for research (UCI ML Repository).",
        provenance="sklearn.datasets.fetch_kddcup99(subset='SA', percent10=True)",
        contamination=0.05,  # approximate; re-derived at load time
        sklearn_name="kddcup99",
        subset="SA",
    ),
    SklearnEntry(
        name="covtype",
        description="UCI Covertype. Class 4 (cottonwood/willow, ~0.5 %) treated as anomaly per ODDS.",
        licence="Public domain (Blackard, Jock A. and Dean J. Colorado State University).",
        provenance="sklearn.datasets.fetch_covtype()",
        contamination=0.005,
        sklearn_name="covtype",
    ),
]


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetadata:
    """Environment the benchmark ran in, recorded alongside its results.

    A number like "AP=0.94" is unfalsifiable without knowing when it was
    produced and against which library versions — those can silently change
    a detector's behaviour (e.g. a scikit-learn IsolationForest release).
    """

    generated_at: str  # UTC ISO 8601, e.g. "2026-09-14T12:00:00+00:00"
    platform: str
    python_version: str
    sorethumb_version: str
    numpy_version: str
    scipy_version: str
    scikit_learn_version: str

    def as_markdown(self) -> str:
        """Render as a one-line Markdown summary preceding the results table."""
        return (
            f"Generated {self.generated_at} on {self.platform} — "
            f"Python {self.python_version}, sorethumb {self.sorethumb_version}, "
            f"numpy {self.numpy_version}, scipy {self.scipy_version}, "
            f"scikit-learn {self.scikit_learn_version}."
        )


def generate_metadata() -> BenchmarkMetadata:
    """Capture the environment a benchmark run is about to execute in."""
    import platform as _platform  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    import scipy  # noqa: PLC0415
    import sklearn  # noqa: PLC0415

    from sorethumb import __version__ as sorethumb_version  # noqa: PLC0415

    return BenchmarkMetadata(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        platform=_platform.platform(),
        python_version=_platform.python_version(),
        sorethumb_version=sorethumb_version,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        scikit_learn_version=sklearn.__version__,
    )


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRow:
    """One row in the benchmark comparison table.

    ``roc_auc`` through ``f1_at_contamination`` are means over ``n_seeds``
    repeated runs; the matching ``*_std`` field is the sample standard
    deviation across those seeds (0.0 when ``n_seeds == 1``). ``roc_auc``
    and ``average_precision`` (and their ``*_std``) are NaN if every seed's
    sample was single-class, where those metrics are mathematically
    undefined (see ``evaluate_scores``).
    """

    dataset: str
    detector: str
    n_rows: int
    n_features: int
    contamination: float
    n_seeds: int
    roc_auc: float
    roc_auc_std: float
    average_precision: float
    average_precision_std: float
    precision_at_k: float
    precision_at_k_std: float
    recall_at_k: float
    recall_at_k_std: float
    f1_at_contamination: float
    f1_at_contamination_std: float
    fit_seconds: float
    score_seconds: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a string-valued dict of every field, for CSV serialisation.

        Mean and standard deviation are kept as separate columns here so
        downstream tooling can parse them without splitting a "mean ± std"
        string; ``as_display_dict`` is the human-facing, merged version used
        for the Markdown table. ROC-AUC/AP are undefined (NaN) when a seed's
        sample was single-class — rendered as "n/a", never "nan", so it
        can't be mistaken for a numeric value in a spreadsheet.
        """
        return {
            "dataset": self.dataset,
            "detector": self.detector,
            "n_rows": str(self.n_rows),
            "n_features": str(self.n_features),
            "contamination": f"{self.contamination:.4f}",
            "n_seeds": str(self.n_seeds),
            "roc_auc": _fmt_or_na(self.roc_auc),
            "roc_auc_std": _fmt_or_na(self.roc_auc_std),
            "average_precision": _fmt_or_na(self.average_precision),
            "average_precision_std": _fmt_or_na(self.average_precision_std),
            "precision_at_k": f"{self.precision_at_k:.4f}",
            "precision_at_k_std": f"{self.precision_at_k_std:.4f}",
            "recall_at_k": f"{self.recall_at_k:.4f}",
            "recall_at_k_std": f"{self.recall_at_k_std:.4f}",
            "f1_at_contamination": f"{self.f1_at_contamination:.4f}",
            "f1_at_contamination_std": f"{self.f1_at_contamination_std:.4f}",
            "fit_seconds": f"{self.fit_seconds:.3f}",
            "score_seconds": f"{self.score_seconds:.3f}",
            "error": self.error or "",
        }

    def _fmt(self, mean: float, std: float) -> str:
        """Format a metric as "mean" alone, or "mean ± std" over >1 seed."""
        if math.isnan(mean):
            return "n/a"
        if self.n_seeds <= 1:
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"

    def as_display_dict(self) -> dict[str, str]:
        """Return the Markdown-table view: metrics merged as "mean ± std"."""
        d = self.as_dict()
        d["roc_auc"] = self._fmt(self.roc_auc, self.roc_auc_std)
        d["average_precision"] = self._fmt(self.average_precision, self.average_precision_std)
        d["precision_at_k"] = self._fmt(self.precision_at_k, self.precision_at_k_std)
        d["recall_at_k"] = self._fmt(self.recall_at_k, self.recall_at_k_std)
        d["f1_at_contamination"] = self._fmt(self.f1_at_contamination, self.f1_at_contamination_std)
        return d


# ---------------------------------------------------------------------------
# Config for benchmark runs
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    dataset_names: list[str] = field(default_factory=list)  # empty = all
    detector_names: list[str] = field(default_factory=list)  # empty = all built-ins
    seed: int = 0
    cache_dir: Path = field(default_factory=lambda: Path(".sorethumb_benchmark_cache"))
    output_dir: Path = field(default_factory=Path)
    # Cap rows per dataset to keep CI fast (0 = no cap)
    max_rows: int = 0
    # Repeat each (dataset, detector) pair at seeds [seed, seed+1, ...,
    # seed+n_seeds-1] and report mean ± std, so one lucky/unlucky draw
    # doesn't read as a stable result. 1 = the old single-run behaviour.
    n_seeds: int = 1


# Fixed review-budget operating point for precision@k / recall@k / F1@k.
#
# This must NOT be derived from the labels being scored (e.g. ``y.mean()``)
# or from a dataset's registered ``contamination`` when that was set to the
# true injected rate (as the synthetic entries above are): either makes
# k == n_positives, which forces precision@k, recall@k, and F1 to be the same
# number (see the warning in ``evaluate_scores``). Using one fixed budget
# across every dataset — as a real reviewer with no oracle would — keeps the
# three metrics independent.
_REVIEW_BUDGET = 0.05


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_METRIC_KEYS = ("roc_auc", "average_precision", "precision_at_k", "recall_at_k", "f1_at_contamination")


def _run_pair_over_seeds(
    det_cls: type[Any],
    seeds: list[int],
    seed_data: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], float, float, str | None]:
    """Fit/score/evaluate one detector at every seed; return raw per-seed results.

    Returns ``(metrics_kwargs, fit_secs_mean, score_secs_mean, error)``.
    ``metrics_kwargs`` has both the mean and ``*_std`` key for every entry in
    ``_METRIC_KEYS``. On error (or zero completed seeds) everything is 0.0.
    ``roc_auc``/``average_precision`` are averaged with ``nanmean``/``nanstd``
    so a single-class seed (mathematically undefined for those two metrics,
    see ``evaluate_scores``) doesn't poison the mean over the other seeds;
    if every seed was single-class the result is NaN, not silently 0.0.
    """
    from sorethumb.evaluate.metrics import evaluate_scores  # noqa: PLC0415
    from sorethumb.scoring.calibrate import Calibrator  # noqa: PLC0415

    seed_metrics: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
    fit_secs_list: list[float] = []
    score_secs_list: list[float] = []
    error: str | None = None

    for seed in seeds:
        X_seed, y_seed = seed_data[seed]
        det = det_cls()

        try:
            t0 = time.perf_counter()
            det.fit(X_seed, seed=seed)
            fit_secs = time.perf_counter() - t0

            t1 = time.perf_counter()
            raw_scores = det.score_samples(X_seed)
            score_secs = time.perf_counter() - t1

            # Calibrate: higher = more anomalous
            cal = Calibrator()
            cal.fit(raw_scores)
            calibrated = cal.transform(raw_scores)

            metrics = evaluate_scores(calibrated, y_seed, contamination=_REVIEW_BUDGET)
        except Exception as exc:  # noqa: BLE001
            logger.warning("    Error (seed=%d): %s", seed, exc)
            error = str(exc)[:200]
            break

        for key in _METRIC_KEYS:
            seed_metrics[key].append(getattr(metrics, key))
        fit_secs_list.append(fit_secs)
        score_secs_list.append(score_secs)

    metrics_kwargs: dict[str, float] = {}
    if error is not None or not seed_metrics["roc_auc"]:
        for key in _METRIC_KEYS:
            metrics_kwargs[key] = 0.0
            metrics_kwargs[f"{key}_std"] = 0.0
        return metrics_kwargs, 0.0, 0.0, error

    for key in _METRIC_KEYS:
        arr = np.asarray(seed_metrics[key], dtype=float)
        if np.all(np.isnan(arr)):
            metrics_kwargs[key] = float("nan")
            metrics_kwargs[f"{key}_std"] = float("nan")
        else:
            metrics_kwargs[key] = float(np.nanmean(arr))
            metrics_kwargs[f"{key}_std"] = float(np.nanstd(arr))
    return (
        metrics_kwargs,
        float(np.mean(fit_secs_list)),
        float(np.mean(score_secs_list)),
        None,
    )


def run_benchmark(cfg: BenchmarkConfig | None = None) -> list[BenchmarkRow]:
    """Run the full benchmark matrix and return the results.

    Parameters
    ----------
    cfg:
        Optional configuration. Defaults are used when *None*.

    Returns
    -------
    List of BenchmarkRow, one per (dataset, detector) pair. Each row's
    metrics are the mean over ``cfg.n_seeds`` seeds; the matching ``*_std``
    fields on ``BenchmarkRow`` hold the standard deviation.

    """
    from sorethumb.detectors import registry  # noqa: PLC0415

    if cfg is None:
        cfg = BenchmarkConfig()

    selected_datasets = [d for d in DATASETS if not cfg.dataset_names or d.name in cfg.dataset_names]
    selected_detectors = [
        (name, cls) for name, cls in registry.items() if not cfg.detector_names or name in cfg.detector_names
    ]
    seeds = [cfg.seed + i for i in range(max(1, cfg.n_seeds))]

    rows: list[BenchmarkRow] = []

    for ds in selected_datasets:
        logger.info("Loading dataset: %s", ds.name)

        # Load (and, for datasets that support it, regenerate) per seed up
        # front, then reuse across every detector below — the data does not
        # depend on which detector is being scored.
        seed_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for seed in seeds:
                X_seed, y_seed = ds.load(cfg.cache_dir, seed=seed)
                if cfg.max_rows and len(X_seed) > cfg.max_rows:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(len(X_seed), cfg.max_rows, replace=False)
                    X_seed, y_seed = X_seed[idx], y_seed[idx]
                seed_data[seed] = (X_seed, y_seed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s — skipping.", ds.name, exc)
            continue

        n_rows, n_features = seed_data[seeds[0]][0].shape

        for det_name, det_cls in selected_detectors:
            logger.info("  Detector: %s", det_name)

            metrics_kwargs, fit_secs_mean, score_secs_mean, error = _run_pair_over_seeds(
                det_cls, seeds, seed_data
            )
            n_seeds_run = 0 if error is not None else len(seeds)

            rows.append(
                BenchmarkRow(
                    dataset=ds.name,
                    detector=det_name,
                    n_rows=n_rows,
                    n_features=n_features,
                    contamination=_REVIEW_BUDGET,
                    n_seeds=n_seeds_run,
                    fit_seconds=fit_secs_mean,
                    score_seconds=score_secs_mean,
                    error=error,
                    **metrics_kwargs,
                )
            )

    return rows


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    "dataset",
    "detector",
    "n_rows",
    "n_features",
    "n_seeds",
    "roc_auc",
    "average_precision",
    "precision_at_k",
    "recall_at_k",
    "f1_at_contamination",
    "fit_seconds",
    "score_seconds",
]


def to_markdown(rows: list[BenchmarkRow], metadata: BenchmarkMetadata | None = None) -> str:
    """Render benchmark rows as a GitHub-flavoured Markdown table.

    The metric columns show "mean ± std" across the row's seeds (bare mean
    when only one seed ran); see ``BenchmarkRow.as_display_dict``. When
    *metadata* is given, its summary line is prepended so the table carries
    its own provenance (generation time, platform, library versions).
    """
    if not rows:
        table = "_No benchmark results._\n"
    else:
        header = "| " + " | ".join(_TABLE_COLS) + " |"
        sep = "| " + " | ".join("---" for _ in _TABLE_COLS) + " |"
        body_lines = []
        for row in rows:
            d = row.as_display_dict()
            body_lines.append("| " + " | ".join(d.get(c, "") for c in _TABLE_COLS) + " |")
        table = "\n".join([header, sep, *body_lines]) + "\n"

    if metadata is None:
        return table
    return f"{metadata.as_markdown()}\n\n{table}"


def to_csv(rows: list[BenchmarkRow]) -> str:
    """Render benchmark rows as CSV text."""
    buf = io.StringIO()
    all_cols = list(BenchmarkRow.__dataclass_fields__.keys())
    writer = csv.DictWriter(buf, fieldnames=all_cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_dict())
    return buf.getvalue()


def write_outputs(
    rows: list[BenchmarkRow],
    output_dir: Path,
    metadata: BenchmarkMetadata | None = None,
) -> tuple[Path, Path]:
    """Write Markdown and CSV files (and, if *metadata* is given, a JSON
    sidecar recording the run's environment) to *output_dir*.

    Returns ``(md_path, csv_path)``.
    """
    import json  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "benchmark_results.md"
    csv_path = output_dir / "benchmark_results.csv"
    md_path.write_text(to_markdown(rows, metadata), encoding="utf-8")
    csv_path.write_text(to_csv(rows), encoding="utf-8")
    if metadata is not None:
        meta_path = output_dir / "benchmark_metadata.json"
        meta_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    logger.info("Benchmark results written to %s and %s", md_path, csv_path)
    return md_path, csv_path


# ---------------------------------------------------------------------------
# README injection
# ---------------------------------------------------------------------------

_RESULTS_MARKER_START = "<!-- benchmark-results-start -->"
_RESULTS_MARKER_END = "<!-- benchmark-results-end -->"


def inject_into_readme(
    rows: list[BenchmarkRow],
    readme_path: Path,
    metadata: BenchmarkMetadata | None = None,
) -> bool:
    """Inject the Markdown table into README.md between marker comments.

    Returns True if the file was modified, False if it was unchanged.
    Idempotent: repeated injection produces identical output.
    """
    if not readme_path.exists():
        logger.warning("README not found at %s; skipping injection.", readme_path)
        return False

    original = readme_path.read_text(encoding="utf-8")
    if _RESULTS_MARKER_START not in original or _RESULTS_MARKER_END not in original:
        logger.warning(
            "README at %s does not contain benchmark markers; skipping injection.\n"
            "Add these markers where you want the table:\n%s\n%s",
            readme_path,
            _RESULTS_MARKER_START,
            _RESULTS_MARKER_END,
        )
        return False

    auto_gen = "<!-- AUTO-GENERATED — do not edit manually; run `sorethumb benchmark` to regenerate. -->"
    table_md = f"{_RESULTS_MARKER_START}\n{auto_gen}\n\n{to_markdown(rows, metadata)}{_RESULTS_MARKER_END}"

    before = original[: original.index(_RESULTS_MARKER_START)]
    after = original[original.index(_RESULTS_MARKER_END) + len(_RESULTS_MARKER_END) :]
    updated = before + table_md + after

    if updated == original:
        return False

    readme_path.write_text(updated, encoding="utf-8")
    return True
