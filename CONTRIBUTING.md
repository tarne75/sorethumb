# Contributing to sorethumb

## From zero to a passing test suite

```bash
# 1. Clone and enter the repo
git clone https://github.com/tarne75/sorethumb.git
cd sorethumb

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create the virtualenv and install all dependencies
uv sync --all-extras

# 4. Install shap separately (Python 3.12 workaround — uv pulls an incompatible numba)
.venv/bin/pip install "shap>=0.45"

# 5. Install pre-commit hooks
uv run pre-commit install

# 6. Verify everything passes
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest -m "not integration and not benchmark"
```

## Running specific test groups

```bash
# Default (unit + property tests only, no network)
uv run pytest

# Include integration tests (may hit the network)
uv run pytest -m integration

# Benchmark suite (opt-in, measures accuracy on real datasets)
uv run pytest -m benchmark

# With coverage
uv run pytest --cov=sorethumb --cov-report=term-missing
```

## SHAP + Python 3.12

`uv add shap` fails because uv resolves `numba==0.53.1` which does not support
Python 3.12. The workaround is to install shap after `uv sync`:

```bash
.venv/bin/pip install "shap>=0.45"
```

This resolves `numba>=0.67.0` and `llvmlite>=0.49.0` which both support Python 3.12.
The `pyproject.toml` pins `numba>=0.67` to prevent uv from pulling an incompatible version.

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
