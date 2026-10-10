"""Detector protocol: the structural interface every detector must satisfy.

Every detector returns ``score_samples`` values where **higher = more normal**,
matching sklearn's IsolationForest convention. The sign flip to "higher = more
anomalous" happens once, in ``scoring/combine.py``, so no shared code branches
on model type. Applying the flip inside a detector would break composite scoring.

``natural_flag`` uses the model's own learned boundary rather than a fixed
contamination quota. This is the value used by ``contamination: auto`` to estimate
the expected anomaly rate from the data, not from a user guess.

Output contract
---------------
For an input matrix ``X`` of shape ``(n_rows, n_features)`` (float64, finite):

* ``score_samples(X)`` returns a **1-D real-numeric ``numpy.ndarray`` of shape
  ``(n_rows,)``**, one score per input row in input order (row ``i`` of the
  output scores row ``i`` of ``X``), **higher = more normal**, every value
  finite (no NaN, no +/-inf). Integer or float dtypes are accepted and are
  converted to float64; bool, object, complex, string, lists and 2-D arrays
  (including ``(n_rows, 1)``) are rejected, never reshaped or coerced.
* ``natural_flag(scores)`` receives the array ``score_samples`` just returned
  and returns a **1-D ``numpy.ndarray`` with dtype exactly ``bool``** and the
  same length, ``True`` = the model's own boundary calls the row anomalous.
  Integer 0/1, float, object arrays and ``None`` are rejected.
* Both methods are valid only on a **fitted** detector: after ``fit()``, or
  after the detector has been loaded back from a saved model (so every
  attribute they read must survive pickling). Before that they must raise, not
  return placeholder values. They must not mutate the model, so scoring the same
  rows twice, or via a reloaded copy, gives the same output.
* ``score_samples`` is called on matrices of any row count, including the
  2-row batches used by gradient attribution; ``natural_flag`` must accept scores
  from rows it was not fitted on.

The pipeline checks all of this with :func:`validate_scores` /
:func:`validate_flags` (via :func:`score_and_flag`) immediately after every call
-- freshly fitted, ``run.reuse_models`` and ``score --from-run`` models alike --
and raises :class:`~sorethumb_ml.errors.DetectorError` naming the detector, the
failing condition, the expected contract and the actual type/dtype/shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from sorethumb_ml.errors import DetectorError
from sorethumb_ml.scoring._finite import nonfinite_summary

if TYPE_CHECKING:
    from collections.abc import Callable

_REQUIRED_CLASS_ATTRS: frozenset[str] = frozenset({"name", "supports_tree_shap", "default_train_row_cap"})
_REQUIRED_METHODS: frozenset[str] = frozenset({"fit", "score_samples", "natural_flag", "get_params"})


@runtime_checkable
class Detector(Protocol):
    """Structural interface for anomaly detectors."""

    name: ClassVar[str]
    supports_tree_shap: ClassVar[bool]
    default_train_row_cap: ClassVar[int]

    def fit(self, X: np.ndarray, *, seed: int) -> None:
        """Fit the detector on *X*. Called once per (group, run)."""
        ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return per-row scores as a finite 1-D numeric ndarray of shape ``(len(X),)``.

        Higher = more normal. Row ``i`` of the result scores row ``i`` of *X*.
        Valid only on a fitted (or loaded) detector; raise otherwise.
        """
        ...

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Return a 1-D ``bool`` ndarray, same length as *scores*, True = anomalous.

        *scores* is the array :meth:`score_samples` returned. The result must have
        dtype ``bool`` exactly (not int 0/1, not object). Valid only on a fitted
        (or loaded) detector.

        Uses the model's own boundary (not a fixed quota) so that
        ``contamination: auto`` has a principled rate to work from.
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters for manifest storage."""
        ...


def check_protocol(cls: type) -> None:
    """Raise DetectorError if *cls* does not satisfy the Detector protocol.

    Checks both the required ClassVar attributes and the required methods.
    Called at registration time so a bad detector fails loudly on import,
    not silently during a long run.
    """
    missing: list[str] = []
    for attr in _REQUIRED_CLASS_ATTRS:
        if not hasattr(cls, attr):
            missing.append(attr)
    for method in _REQUIRED_METHODS:
        if not callable(getattr(cls, method, None)):
            missing.append(method)
    if missing:
        raise DetectorError(
            f"Detector class {cls.__qualname__!r} does not satisfy the Detector protocol. "
            f"Missing or non-callable: {sorted(missing)}"
        )


def _describe(value: object) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray {value.dtype} shape {value.shape}"
    return type(value).__name__


def _fail(detector: str, method: str, condition: str, expected: str, actual: object) -> DetectorError:
    return DetectorError(
        f"Detector {detector!r} {method}() output invalid: {condition}. "
        f"Expected {expected}; got {_describe(actual)}."
    )


def validate_scores(detector: str, scores: object, n_rows: int) -> np.ndarray:
    """Check ``score_samples`` output against the contract; return it as float64.

    Raises :class:`~sorethumb_ml.errors.DetectorError` (never a numpy error) naming
    *detector*, the failing condition, the expected contract and the actual
    type/dtype/shape.
    """
    expected = f"a finite 1-D real-numeric ndarray of shape ({n_rows},) (higher = more normal)"
    method = "score_samples"
    if not isinstance(scores, np.ndarray):
        raise _fail(detector, method, "not a numpy ndarray", expected, scores)
    if scores.dtype.kind not in "iuf":
        raise _fail(detector, method, f"dtype {scores.dtype} is not integer or float", expected, scores)
    if scores.ndim != 1:
        raise _fail(detector, method, f"{scores.ndim}-dimensional, not 1-D", expected, scores)
    if scores.shape[0] != n_rows:
        raise _fail(
            detector, method, f"wrong length {scores.shape[0]} for {n_rows} input rows", expected, scores
        )
    out = scores.astype(np.float64, copy=False)
    bad = nonfinite_summary(out)
    if bad is not None:
        raise _fail(detector, method, f"non-finite values: {bad}", expected, scores)
    return out


def validate_flags(detector: str, flags: object, n_rows: int) -> np.ndarray:
    """Check ``natural_flag`` output against the contract; return it unchanged.

    Requires a 1-D ndarray of dtype ``bool`` and length *n_rows*. Integer 0/1,
    float and object arrays are rejected rather than coerced.
    """
    expected = f"a 1-D bool ndarray of shape ({n_rows},) (True = anomalous)"
    method = "natural_flag"
    if not isinstance(flags, np.ndarray):
        raise _fail(detector, method, "not a numpy ndarray", expected, flags)
    if flags.dtype != np.bool_:
        raise _fail(detector, method, f"dtype {flags.dtype} is not bool", expected, flags)
    if flags.ndim != 1:
        raise _fail(detector, method, f"{flags.ndim}-dimensional, not 1-D", expected, flags)
    if flags.shape[0] != n_rows:
        raise _fail(
            detector, method, f"wrong length {flags.shape[0]} for {n_rows} input rows", expected, flags
        )
    return flags


def _call(detector: str, method: str, fn: Callable[[Any], Any], arg: Any) -> Any:
    try:
        return fn(arg)
    except DetectorError:
        raise
    except Exception as exc:
        msg = f"Detector {detector!r} {method}() raised {type(exc).__name__}: {exc}"
        raise DetectorError(msg) from exc


def score_samples_checked(detector: Any, X: np.ndarray) -> np.ndarray:
    """Call ``detector.score_samples(X)`` and validate the result (one row per row of X)."""
    name = str(getattr(detector, "name", type(detector).__name__))
    return validate_scores(name, _call(name, "score_samples", detector.score_samples, X), len(X))


def score_and_flag(detector: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score *X* and take the natural flags, validating both immediately.

    The one place the pipeline calls ``score_samples`` + ``natural_flag`` on a
    full matrix, so fresh-fit, reused and score-forward detectors get identical
    checks before anything downstream (calibration, persistence, ensembling)
    sees the values. Returns ``(float64 scores, bool flags)``.
    """
    name = str(getattr(detector, "name", type(detector).__name__))
    raw = score_samples_checked(detector, X)
    flags = validate_flags(name, _call(name, "natural_flag", detector.natural_flag, raw), len(X))
    return raw, flags
