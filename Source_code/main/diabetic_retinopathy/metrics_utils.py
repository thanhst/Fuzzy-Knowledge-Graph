"""Single source of truth for the classification metrics reported in the
diabetic retinopathy experiments.

Every model family (deep baselines, native FKG, FKGS) must report numbers that
come from this module so that a reviewer can compare rows of the same table
without wondering whether "Recall" means positive-class recall in one row and a
macro average in the next.

The module intentionally depends on nothing but the standard library and numpy:
the deep baseline virtual environment ships torch without scikit-learn, while
the FIS/FKG environment ships scikit-learn without torch, and both need to call
the same code.

Conventions
-----------
* The task is binary: ``positive_label`` is the diabetic-retinopathy class.
* ``sensitivity`` / ``precision`` / ``f1`` are positive-class values. They are
  the numbers a clinician cares about on a 6.6% prevalence problem.
* Macro averages are also returned, suffixed ``macro_``, so older tables that
  quoted macro numbers stay reproducible instead of silently changing meaning.
* ``auc_roc`` and ``auc_pr`` need a continuous ``y_score`` for the positive
  class. When it is missing they fall back to the hard predictions, which makes
  the AUC a two-point curve; ``score_kind`` in the output records which case
  applies so a degenerate AUC can never be mistaken for a ranked one.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np


# Order used whenever metrics are flattened into a CSV header or a report table.
METRIC_NAMES: List[str] = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "auc_roc",
    "auc_pr",
    "mcc",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]

COUNT_NAMES: List[str] = ["tp", "tn", "fp", "fn", "n", "n_pos", "n_neg", "prevalence"]


def _as_int_array(values: Iterable) -> np.ndarray:
    return np.asarray([int(float(value)) for value in values], dtype=np.int64)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def roc_auc(y_true_binary: Sequence[int], y_score: Sequence[float]) -> float:
    """Rank-based AUC-ROC (Mann-Whitney U) with mid-ranks for tied scores.

    Returns NaN when one of the two classes is absent, because AUC is undefined
    there and a 0.0 would quietly drag a fold mean down.
    """
    truth = np.asarray(y_true_binary, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    positive_count = int(truth.sum())
    negative_count = int(truth.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return math.nan

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    cursor = 0
    while cursor < sorted_scores.size:
        end = cursor + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end

    positive_rank_sum = float(ranks[truth == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def average_precision(y_true_binary: Sequence[int], y_score: Sequence[float]) -> float:
    """Area under the precision-recall curve, computed as average precision.

    AP = sum_n (R_n - R_{n-1}) * P_n over the thresholds given by the distinct
    scores, which is the same estimator scikit-learn's
    ``average_precision_score`` uses. Unlike a trapezoidal interpolation it does
    not reward a model for a curve it never actually achieves.

    The interesting property for this dataset: a random model scores the
    positive prevalence (~0.066), not 0.5, so AUC-PR separates the models that
    genuinely rank the 1070 positive eyes from the ones that do not.
    """
    truth = np.asarray(y_true_binary, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    positive_count = int(truth.sum())
    if positive_count == 0 or truth.size == 0:
        return math.nan

    order = np.argsort(-scores, kind="mergesort")
    truth_sorted = truth[order]
    scores_sorted = scores[order]

    cumulative_tp = np.cumsum(truth_sorted)
    cumulative_predicted = np.arange(1, truth.size + 1, dtype=np.float64)

    # Only thresholds at the end of a run of equal scores are real operating
    # points; keeping the intermediate ones would invent precision values that
    # no threshold can produce.
    distinct = np.ones(truth.size, dtype=bool)
    distinct[:-1] = scores_sorted[:-1] != scores_sorted[1:]

    precision = cumulative_tp[distinct] / cumulative_predicted[distinct]
    recall = cumulative_tp[distinct] / positive_count
    recall_delta = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(recall_delta * precision))


def macro_prf(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]):
    """Macro precision / recall / F1, matching sklearn's ``average='macro'``."""
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    truth = _as_int_array(y_true)
    predicted = _as_int_array(y_pred)
    for label in labels:
        tp = int(np.sum((truth == label) & (predicted == label)))
        fp = int(np.sum((truth != label) & (predicted == label)))
        fn = int(np.sum((truth == label) & (predicted != label)))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(_safe_divide(2.0 * precision * recall, precision + recall))
    count = max(1, len(labels))
    return sum(precisions) / count, sum(recalls) / count, sum(f1s) / count


def binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
    positive_label: int = 1,
    labels: Sequence[int] | None = None,
) -> Dict[str, float]:
    """Full metric set for one evaluation (one fold, one turn, one model).

    ``y_score`` is the continuous score for ``positive_label``. Pass it whenever
    the model can produce one; AUC-ROC and AUC-PR are only meaningful then.
    """
    truth = _as_int_array(y_true)
    predicted = _as_int_array(y_pred)
    if truth.size != predicted.size:
        raise ValueError(
            f"y_true and y_pred length mismatch: {truth.size} vs {predicted.size}"
        )
    if truth.size == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")

    if labels is None:
        labels = sorted(set(truth.tolist()) | set(predicted.tolist()))
    labels = [int(label) for label in labels]

    positive_label = int(positive_label)
    truth_binary = (truth == positive_label).astype(np.int64)
    predicted_binary = (predicted == positive_label).astype(np.int64)

    tp = int(np.sum((truth_binary == 1) & (predicted_binary == 1)))
    tn = int(np.sum((truth_binary == 0) & (predicted_binary == 0)))
    fp = int(np.sum((truth_binary == 0) & (predicted_binary == 1)))
    fn = int(np.sum((truth_binary == 1) & (predicted_binary == 0)))

    precision = _safe_divide(tp, tp + fp)
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2.0 * precision * sensitivity, precision + sensitivity)

    if y_score is None:
        scores = predicted_binary.astype(np.float64)
        score_kind = "hard_prediction"
    else:
        scores = np.asarray([float(value) for value in y_score], dtype=np.float64)
        if scores.size != truth.size:
            raise ValueError(
                f"y_score length mismatch: {scores.size} vs {truth.size}"
            )
        score_kind = "positive_class_score"

    mcc_denominator = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = (
        (float(tp) * float(tn) - float(fp) * float(fn)) / mcc_denominator
        if mcc_denominator
        else 0.0
    )

    macro_precision, macro_recall, macro_f1 = macro_prf(truth, predicted, labels)

    return {
        "accuracy": _safe_divide(tp + tn, truth.size),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc_roc": roc_auc(truth_binary, scores),
        "auc_pr": average_precision(truth_binary, scores),
        "mcc": mcc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "n": float(truth.size),
        "n_pos": float(tp + fn),
        "n_neg": float(tn + fp),
        "prevalence": _safe_divide(tp + fn, truth.size),
        "positive_label": float(positive_label),
        "score_kind": score_kind,
    }


def positive_scores_from_confidence(
    y_pred: Sequence[int],
    confidences: Sequence[float] | None,
    positive_label: int,
) -> List[float]:
    """Recover the positive-class score from FIS/FKG ``(label, confidence)``.

    FISA returns ``max(DARR) / sum(DARR)`` for the winning class. For a binary
    problem the two class scores sum to 1, so the positive-class score is the
    confidence itself when the positive class won and its complement otherwise.
    This reconstruction is exact for two classes only -- do not reuse it for a
    multi-class head, where the losing mass is split across several classes.
    """
    scores: List[float] = []
    if confidences is None:
        confidences = [None] * len(list(y_pred))
    for predicted, confidence in zip(y_pred, confidences):
        if confidence is None:
            scores.append(1.0 if int(predicted) == int(positive_label) else 0.0)
            continue
        value = float(confidence)
        if not math.isfinite(value):
            scores.append(1.0 if int(predicted) == int(positive_label) else 0.0)
            continue
        value = min(1.0, max(0.0, value))
        scores.append(value if int(predicted) == int(positive_label) else 1.0 - value)
    return scores


def aggregate_metrics(
    per_fold: Sequence[Dict[str, float]],
    metric_names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Mean and sample standard deviation across folds.

    Two deliberate choices, both of which were wrong in the previous reports:

    * the spread is taken **across folds**, never across classes, so
      ``x_std`` answers "how stable is this model between patients splits";
    * ``ddof=1`` is used, which is the standard deviation a paper means by
      "5-fold mean +/- std". NaN folds (an AUC on a fold with one class) are
      dropped rather than poisoning the mean.
    """
    if metric_names is None:
        metric_names = METRIC_NAMES
    summary: Dict[str, float] = {"folds": float(len(per_fold))}
    for name in metric_names:
        values = [
            float(fold[name])
            for fold in per_fold
            if name in fold and fold[name] is not None and not _is_nan(fold[name])
        ]
        if not values:
            summary[f"{name}_mean"] = math.nan
            summary[f"{name}_std"] = math.nan
            summary[f"{name}_n"] = 0.0
            continue
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[f"{name}_n"] = float(len(values))
    return summary


def _is_nan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def format_mean_std(
    summary: Dict[str, float],
    name: str,
    decimals: int = 1,
    as_percent: bool = True,
) -> str:
    """``87.5 +/- 2.1`` style cell for the comparison tables."""
    mean = summary.get(f"{name}_mean")
    std = summary.get(f"{name}_std")
    if mean is None or _is_nan(mean):
        return "n/a"
    scale = 100.0 if as_percent else 1.0
    if std is None or _is_nan(std):
        return f"{mean * scale:.{decimals}f}"
    return f"{mean * scale:.{decimals}f} +/- {std * scale:.{decimals}f}"
