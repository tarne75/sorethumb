# Contributing to sorethumb

## From zero to a passing test suite

```bash
# 1. Clone and enter the repo
git clone https://github.com/tarne75/sorethumb.git
cd sorethumb

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create the virtualenv and install every extra (--frozen: use the
#    committed uv.lock as-is, matching what CI installs)
uv sync --all-extras --frozen

# 4. Install pre-commit hooks
uv run pre-commit install

# 5. Verify everything passes (mirrors CI's required PR lanes)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run python docs/generate_config_docs.py --check
uv run pytest -m "unit or contract"
uv run pytest -m integration
uv run pytest -m property
uv run pytest -m repo_check
```

If you changed `pyproject.toml`'s dependencies, run `uv lock` to update
`uv.lock` and commit both together — CI runs `uv lock --check` and will fail
a PR where they've drifted apart.

## Optional extras

The mandatory install (`pip install sorethumb` / `uv sync`) is deliberately
minimal. Everything below is opt-in, matched to the feature it enables:

| Extra | Adds | Enables |
|---|---|---|
| `explain` | shap, numba | TreeSHAP / KernelSHAP explanations. Without it, explanations fall back to the pure-numpy gradient method with a warning — the run itself never fails. |
| `report` | matplotlib | Trend charts in the HTML report. |
| `benchmark` | datasets, pandas | `sorethumb benchmark` (both the real-dataset and full-pipeline-scenario suites). |
| `dev` | pytest, ruff, mypy, pre-commit, hypothesis, ... | Everything needed to run the test suite and quality checks in this repo. |

`uv sync --all-extras --frozen` installs all four, which is what you want for
contributing. A production install that only ever calls `run_detection`
without SHAP explanations can skip straight to core: `pip install sorethumb`.

## Running specific test groups

Every test carries exactly one marker (see `[tool.pytest.ini_options]` in
`pyproject.toml`); bare `uv run pytest` runs only `unit`/`contract` by
default (its `addopts` excludes the rest). These mirror CI's own job-by-job
`-m` selection (`.github/workflows/ci.yml`) — nothing here should drift from
what a job actually runs.

```bash
# Default: unit + contract only — fast, deterministic, no filesystem/
# network/subprocess/model-serialisation I/O. Matches CI's fast-tests job.
uv run pytest

# Integration: real workspace, SQLite, CLI process, full pipeline, report
# rendering. May touch the local filesystem and subprocesses, not the network.
uv run pytest -m integration

# Hypothesis-driven property tests.
uv run pytest -m property

# Repository/docs consistency checks (generated docs, README snippets).
uv run pytest -m repo_check

# Benchmark suite (opt-in, measures accuracy on real or synthetic datasets).
uv run pytest -m benchmark

# Every required-PR-lane test in one invocation (matches CI's coverage job).
uv run pytest -m "unit or contract or integration or property or repo_check"

# With coverage
uv run pytest --cov=sorethumb --cov-report=term-missing
```

## Adding a detector

`sorethumb` discovers detectors via the `sorethumb.detectors` entry-point group. You can
add a detector in a separate package without modifying this repository.

### 1. Implement the `Detector` protocol

```python
# my_package/my_detector.py
from typing import ClassVar, Any
import numpy as np

class MyDetector:
    name: ClassVar[str] = "my_detector"
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 100_000

    def fit(self, X: np.ndarray, *, seed: int) -> None:
        # fit your model; store it on self
        ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # MUST return higher values for MORE NORMAL records
        # (matches sklearn's score_samples convention)
        ...

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        # return a boolean array: True = anomaly, using the model's own boundary
        ...

    def get_params(self) -> dict[str, Any]:
        return {}
```

### 2. Register it via entry points in your package's `pyproject.toml`

```toml
[project.entry-points."sorethumb.detectors"]
my_detector = "my_package.my_detector:MyDetector"
```

### 3. Verify it appears

```bash
pip install -e .
sorethumb detectors
```

Your detector should appear in the list alongside the built-ins.

## Code standards

- ruff for linting and formatting (`uv run ruff check --fix src/ && uv run ruff format src/`)
- mypy strict on `src/` (`uv run mypy src/`)
- No `print` in `src/sorethumb/` — use `logging.getLogger(__name__)`
- No literal thresholds in modules other than `config.py`
- Every new degradation point gets a named `SorethumbWarning` subclass in `errors.py`
