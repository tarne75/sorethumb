"""Validated pass-through of arbitrary estimator hyper-parameters.

The sklearn-backed detector wrappers (``isolation_forest``, ``lof``,
``kmeans_distance``, ``one_class_svm``) each expose a curated set of constructor
arguments. ``extra_params`` is the escape hatch: a dict handed straight to the
underlying sklearn estimator's constructor, so a power user can set anything the
wrapper does not surface (``n_jobs``, ``max_features``, KMeans ``tol`` /
``max_iter`` / ``algorithm``, LOF ``leaf_size`` / ``metric`` / ``p``, …).

The pipeline fills it from ``[[detectors]] params.extra_params`` in the TOML
config, e.g.::

    [[detectors]]
    name = "isolation_forest"
    params = { n_estimators = 300, extra_params = { n_jobs = 4, max_features = 0.8 } }

Keys are validated the moment the detector is constructed:

* Keys sorethumb manages itself are rejected with a pointer to the right knob
  (``random_state`` → ``run.seed``; ``contamination`` → ``scoring.contamination``;
  ``novelty`` for LOF; ``n_clusters`` for KMeans).
* Keys already exposed as a curated wrapper argument are rejected so there is one
  unambiguous way to set them.
* Any remaining key that the target estimator does not accept is rejected, with
  the estimator's real parameter list in the message — a typo fails here, not
  deep inside sklearn at fit time.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any

from sorethumb.errors import ConfigError

# Hyper-parameters the pipeline sets itself. A pass-through value would break an
# invariant the rest of the system relies on. Value = what to use instead.
_MANAGED_GLOBALLY: dict[str, str] = {
    "random_state": "determinism is controlled by run.seed",
    "seed": "determinism is controlled by run.seed",
    "contamination": "thresholding is controlled by scoring.contamination",
    "novelty": "LOF is always fitted with novelty=True",
    "n_clusters": "set the cluster count via the detector's `k` argument",
}


def _estimator_param_names(estimator_cls: type) -> list[str]:
    """Return the constructor parameter names accepted by an sklearn-style estimator."""
    getter = getattr(estimator_cls, "_get_param_names", None)
    if callable(getter):
        # sklearn raises RuntimeError if __init__ uses *args; fall back to signature.
        with contextlib.suppress(RuntimeError, TypeError):
            return list(getter())
    return list(inspect.signature(estimator_cls).parameters)


def estimator_extra_param_defaults(
    estimator_cls: type,
    curated: frozenset[str],
) -> dict[str, Any]:
    """Return ``{name: sklearn-default}`` for every key a wrapper accepts via ``extra_params``.

    That is the estimator's constructor parameters minus the globally-managed
    keys (:data:`_MANAGED_GLOBALLY`) and the wrapper's own ``curated`` arguments.
    Used to document the escape hatch in ``sorethumb init`` and the config
    reference, so the list can never drift from the installed scikit-learn.
    """
    defaults: dict[str, Any]
    try:
        defaults = dict(estimator_cls().get_params(deep=False))
    except Exception:  # noqa: BLE001 — not a get_params estimator; read signature defaults
        params = inspect.signature(estimator_cls).parameters
        defaults = {
            name: (p.default if p.default is not inspect.Parameter.empty else None)
            for name, p in params.items()
        }
    excluded = set(_MANAGED_GLOBALLY) | set(curated)
    return {k: defaults[k] for k in sorted(defaults) if k not in excluded}


def validate_extra_params(
    detector_name: str,
    estimator_cls: type | None,
    extra: dict[str, Any] | None,
    *,
    curated: frozenset[str],
) -> dict[str, Any]:
    """Return a cleaned, key-sorted copy of *extra*, or raise :class:`ConfigError`.

    Parameters
    ----------
    detector_name:
        Used only in error messages.
    estimator_cls:
        The underlying estimator class whose constructor will receive the kwargs,
        or ``None`` for a detector that has no underlying estimator (ecod, hbos) —
        any non-empty *extra* is then an error.
    extra:
        The user-supplied mapping, or ``None``.
    curated:
        Names already exposed as first-class wrapper arguments; passing them
        through ``extra_params`` is rejected as ambiguous.

    """
    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise ConfigError(
            f"{detector_name}: extra_params must be a table/mapping, got {type(extra).__name__}."
        )
    if not extra:
        return {}
    if not all(isinstance(k, str) for k in extra):
        raise ConfigError(f"{detector_name}: every extra_params key must be a string.")

    if estimator_cls is None:
        raise ConfigError(
            f"{detector_name} has no underlying estimator and takes no extra_params (got {sorted(extra)})."
        )

    managed = sorted(k for k in extra if k in _MANAGED_GLOBALLY)
    if managed:
        detail = "; ".join(f"{k!r} ({_MANAGED_GLOBALLY[k]})" for k in managed)
        raise ConfigError(f"{detector_name}: {detail}. Remove these from extra_params.")

    collide = sorted(k for k in extra if k in curated)
    if collide:
        raise ConfigError(
            f"{detector_name}: {collide} are already wrapper arguments — set them there, "
            "not in extra_params, so there is one unambiguous source."
        )

    accepted = set(_estimator_param_names(estimator_cls))
    unknown = sorted(k for k in extra if k not in accepted)
    if unknown:
        raise ConfigError(
            f"{detector_name}: {estimator_cls.__name__} has no hyper-parameter(s) {unknown}. "
            f"Accepted: {sorted(accepted)}."
        )

    return dict(sorted(extra.items()))
