from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    pr_auc: float
    roc_auc: float
    brier: float
    precision: float
    recall: float
    f1: float
    threshold: float


def evaluate_binary_classifier(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> ClassificationMetrics:
    y_pred = (y_score >= threshold).astype(int)
    return ClassificationMetrics(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, y_score)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
    )


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 19)
    f1_vals = [f1_score(y_true, (y_score >= thr).astype(int), zero_division=0) for thr in candidates]
    return float(candidates[int(np.argmax(f1_vals))])
