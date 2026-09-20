"""The validation matrix's dataset registry and detector combos."""

from __future__ import annotations

import polars as pl

from scripts.validation.schema import ComboSpec, DatasetSpec


def _kddcup_is_anomaly(target: pl.Series) -> pl.Series:
    """Return True for every KDDCup99 label except 'normal.'.

    A module-level function, not a lambda, so DatasetSpec stays picklable
    for ProcessPoolExecutor-based --workers parallelism.
    """
    return target != "normal."


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="kddcup99_sa",
        file="kddcup99_sa.parquet",
        ignore=["target"],
        source="KDD Cup 1999 (UCI ML Repository) — SA subset, 10 % sample",
        description="Network intrusion detection. 41 connection features (numeric + categorical). "
        "`target` is a real anomaly ground truth (normal. vs every attack type) -- the one dataset "
        "here that supports held-out ROC-AUC/AP, not just an operational smoke test.",
        rows=100_655,
        cols=42,
        label_column="target",
        label_is_anomaly=_kddcup_is_anomaly,
    ),
    DatasetSpec(
        name="electricity",
        file="electricity.parquet",
        ignore=["class"],
        source="Harries (1999) via OpenML — Electricity dataset",
        description="Half-hourly Australian electricity demand 1996–1998. "
        "Price and demand for NSW and Victoria plus transfer. `class` (UP/DOWN) excluded -- it is a "
        "price-direction label, not an anomaly ground truth, so only operational metrics apply.",
        rows=45_312,
        cols=9,
    ),
    DatasetSpec(
        name="weather_australia",
        file="weather_australia.parquet",
        ignore=["A15"],
        source="Australian Bureau of Meteorology via UCI ML Repository",
        description="Daily weather observations (anonymous columns A1–A14). "
        "A15 is the binary RainTomorrow label, excluded from features -- not an anomaly label.",
        rows=690,
        cols=15,
    ),
    DatasetSpec(
        name="macro_us_quarterly",
        file="macro_us_quarterly.parquet",
        ignore=["year", "quarter"],
        source="statsmodels macrodata — US Federal Reserve",
        description="Quarterly US macroeconomic indicators 1959–2009 "
        "(GDP, inflation, unemployment, interest rates). Year and quarter excluded as indices.",
        rows=203,
        cols=14,
    ),
    DatasetSpec(
        name="elnino_sst",
        file="elnino_sst.parquet",
        ignore=["YEAR"],
        source="statsmodels elnino — NOAA/TOGA-TAO buoy array",
        description="Annual mean sea-surface temperatures across 12 Pacific buoy locations "
        "1950–2010. YEAR excluded as index.",
        rows=61,
        cols=13,
    ),
    DatasetSpec(
        name="sunspots_annual",
        file="sunspots_annual.parquet",
        ignore=["YEAR"],
        source="statsmodels sunspots — Royal Observatory of Belgium",
        description="Annual Wolf sunspot number 1700–2008. Single numeric feature after "
        "excluding YEAR. Very small — edge-case/sanity dataset.",
        rows=309,
        cols=2,
    ),
    DatasetSpec(
        name="longley_multicollinear",
        file="longley_multicollinear.parquet",
        ignore=[],
        source="statsmodels longley — Longley (1967)",
        description="Annual US macro data 1947–1962 (7 highly collinear features). "
        "16 rows only — extreme edge case. Included for completeness.",
        rows=16,
        cols=7,
    ),
    DatasetSpec(
        name="natops_mts",
        file="natops_mts.parquet",
        ignore=["label"],
        source="UEA Time Series Classification Archive — NATOPS dataset",
        description="24-channel aircraft hand-signal motion capture (51 timepoints), "
        "stored wide (1224 numeric columns). `label` excluded. PCA recommended.",
        rows=360,
        cols=1_225,
    ),
    DatasetSpec(
        name="basic_motions_mts",
        file="basic_motions_mts.parquet",
        ignore=["label"],
        source="UEA Time Series Classification Archive — BasicMotions dataset",
        description="6-axis IMU data for 4 activities (100 timepoints × 6 channels = 600 cols). "
        "`label` excluded. PCA recommended.",
        rows=80,
        cols=601,
    ),
]

_BASE_DETECTORS = ("isolation_forest", "kmeans_distance", "one_class_svm")

COMBOS: list[ComboSpec] = [
    ComboSpec("baseline", _BASE_DETECTORS),
    ComboSpec("baseline+ecod", (*_BASE_DETECTORS, "ecod")),
    ComboSpec("baseline+lof", (*_BASE_DETECTORS, "lof")),
    ComboSpec("baseline+hbos", (*_BASE_DETECTORS, "hbos")),
    ComboSpec("baseline+ecod+lof", (*_BASE_DETECTORS, "ecod", "lof")),
    ComboSpec("baseline+ecod+hbos", (*_BASE_DETECTORS, "ecod", "hbos")),
    ComboSpec("baseline+lof+hbos", (*_BASE_DETECTORS, "lof", "hbos")),
    ComboSpec("all6", (*_BASE_DETECTORS, "ecod", "lof", "hbos")),
]


def dataset_by_name(name: str) -> DatasetSpec | None:
    """Look up a DatasetSpec by name, or None if unregistered."""
    return next((d for d in DATASETS if d.name == name), None)


def combo_by_name(name: str) -> ComboSpec | None:
    """Look up a ComboSpec by name, or None if unregistered."""
    return next((c for c in COMBOS if c.name == name), None)
