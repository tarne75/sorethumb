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


def evaluate_scores(
    scores: Any,
    labels: Any,
    contamination: float = 0.05,
) -> Metrics:
    """Compute anomaly detection metrics against ground-truth binary labels.

    Parameters
    ----------
    scores:
        1-D array-like of anomaly scores (higher = more anomalous).
    labels:
        1-D array-like of ground-truth labels (1 = anomaly, 0 = normal).
    contamination:
        Fraction of rows to flag as anomalous for threshold-dependent metrics.
        Also used as the assumed positive rate for F1.

    Returns
    -------
    Metrics dataclass with ROC-AUC, AP, P@k, R@k, and F1 at contamination.

    """
    from sklearn.metrics import (  # noqa: PLC0415
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)

    n_total = len(scores_arr)
    n_positives = int(labels_arr.sum())

    if n_positives in {0, n_total}:
        logger.warning(
            "evaluate_scores: all labels are identical (%d positives / %d total); "
            "ROC-AUC and AP are undefined — returning 0.0.",
            n_positives,
            n_total,
        )
        return Metrics(
            roc_auc=0.0,
            average_precision=0.0,
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
