"""Evaluation metrics for anomaly detection.

Headline metric is average precision (AP), not ROC-AUC: AP accounts for the
class imbalance that is fundamental to anomaly detection — a model that flags
every row achieves 100% recall but near-zero precision, and AP penalises it
appropriately. ROC-AUC is still included for compatibility with the literature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Evaluation metrics for one scored population."""

    roc_auc: float
    average_precision: float
    precision_at_k: float
    recall_at_k: float
    f1_at_contamination: float
    contamination_used: float
    n_positives: int
    n_total: int
    k_used: int

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"ROC-AUC={self.roc_auc:.4f}  AP={self.average_precision:.4f}  "
            f"P@{self.k_used}={self.precision_at_k:.4f}  "
            f"R@{self.k_used}={self.recall_at_k:.4f}  "
            f"F1@c={self.contamination_used:.3f}={self.f1_at_contamination:.4f}"
        )


def _validate_evaluate_inputs(scores_arr: np.ndarray, labels_raw: np.ndarray, contamination: float) -> None:
    """Validate evaluate_scores input.

    Fails closed rather than letting a malformed input silently corrupt a
    metric or crash deep inside sklearn with a confusing error.
    """
    if scores_arr.ndim != 1:
        msg = f"scores must be 1-D; got shape {scores_arr.shape}"
        raise ValueError(msg)
    if labels_raw.ndim != 1:
        msg = f"labels must be 1-D; got shape {labels_raw.shape}"
        raise ValueError(msg)
    if len(scores_arr) == 0:
        msg = "scores/labels must be non-empty"
        raise ValueError(msg)
    if len(scores_arr) != len(labels_raw):
        msg = f"scores and labels must have equal length; got {len(scores_arr)} and {len(labels_raw)}"
        raise ValueError(msg)
    if not np.all(np.isfinite(scores_arr)):
        msg = "scores must be finite (no NaN/±Inf)"
        raise ValueError(msg)
    bad_labels = sorted(set(labels_raw.tolist()) - {0.0, 1.0})
    if bad_labels:
        msg = f"labels must be binary (0/1); found other value(s): {bad_labels}"
        raise ValueError(msg)
    if not (0.0 < contamination < 1.0):
        msg = f"contamination must be in (0, 1); got {contamination}"
        raise ValueError(msg)


def evaluate_scores(
    scores: Any,
    labels: Any,
    contamination: float = 0.05,
) -> Metrics:
    """Compute anomaly detection metrics against ground-truth binary labels.

    Parameters
    ----------
    scores:
        1-D array-like of finite anomaly scores (higher = more anomalous).
    labels:
        1-D array-like of binary (0/1) ground-truth labels (1 = anomaly).
    contamination:
        Fraction of rows to flag as anomalous for threshold-dependent metrics
        (a review-budget operating point), in (0, 1). Choose this
        independently of the labels' true positive rate. Setting it to
        ``y.mean()`` makes ``k = round(n_total * contamination)`` equal
        ``n_positives`` exactly, which forces ``precision_at_k ==
        recall_at_k == f1_at_contamination`` (all three reduce to
        ``n_true_at_k / n_positives``) — three names for one number, not
        three signals.

    Returns
    -------
    Metrics dataclass with ROC-AUC, AP, P@k, R@k, and F1 at contamination.

    Raises
    ------
    ValueError
        scores/labels are not 1-D, not the same length, empty, contain a
        non-finite score, contain a label other than 0/1, or contamination
        is not in (0, 1).

    """
    from sklearn.metrics import (  # noqa: PLC0415
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    scores_arr = np.asarray(scores, dtype=float)
    labels_raw = np.asarray(labels, dtype=float)
    _validate_evaluate_inputs(scores_arr, labels_raw, contamination)
    labels_arr = labels_raw.astype(int)

    n_total = len(scores_arr)
    n_positives = int(labels_arr.sum())

    if n_positives in {0, n_total}:
        logger.warning(
            "evaluate_scores: all labels are identical (%d positives / %d total); "
            "ROC-AUC and AP are undefined — returning NaN.",
            n_positives,
            n_total,
        )
        return Metrics(
            roc_auc=float("nan"),
            average_precision=float("nan"),
            precision_at_k=0.0,
            recall_at_k=0.0,
            f1_at_contamination=0.0,
            contamination_used=contamination,
            n_positives=n_positives,
            n_total=n_total,
            k_used=0,
        )

    roc_auc = float(roc_auc_score(labels_arr, scores_arr))
    ap = float(average_precision_score(labels_arr, scores_arr))

    k = max(1, round(n_total * contamination))
    if k == n_positives:
        logger.warning(
            "evaluate_scores: k=%d equals n_positives=%d -- precision_at_k, "
            "recall_at_k, and f1_at_contamination will be the same number by "
            "construction. contamination=%.4f was likely derived from this "
            "same label set; pick it independently (e.g. a fixed review "
            "budget) to get three distinct signals.",
            k,
            n_positives,
            contamination,
        )
    top_k_idx = np.argsort(scores_arr)[::-1][:k]
    predicted_at_k = np.zeros(n_total, dtype=int)
    predicted_at_k[top_k_idx] = 1

    n_true_at_k = int((predicted_at_k & labels_arr).sum())
    prec_k = n_true_at_k / k if k > 0 else 0.0
    rec_k = n_true_at_k / n_positives if n_positives > 0 else 0.0

    # F1 at the contamination threshold
    _, _, f1_arr, _ = precision_recall_fscore_support(
        labels_arr, predicted_at_k, average=None, zero_division=0
    )
    f1_positive = float(f1_arr[1]) if len(f1_arr) > 1 else 0.0

    return Metrics(
        roc_auc=roc_auc,
        average_precision=ap,
        precision_at_k=prec_k,
        recall_at_k=rec_k,
        f1_at_contamination=f1_positive,
        contamination_used=contamination,
        n_positives=n_positives,
        n_total=n_total,
        k_used=k,
    )
