import math

import numpy as np


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _binary_auc(y_true, y_score):
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
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
    rank_sum = float(ranks[truth == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(y_true, y_score):
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    n_pos = int(truth.sum())
    if n_pos == 0:
        return math.nan

    order = np.argsort(-scores, kind="mergesort")
    truth = truth[order]
    scores = scores[order]
    cumulative_tp = np.cumsum(truth)
    cumulative_n = np.arange(1, truth.size + 1, dtype=np.float64)
    distinct = np.ones(truth.size, dtype=bool)
    distinct[:-1] = scores[:-1] != scores[1:]
    precision = cumulative_tp[distinct] / cumulative_n[distinct]
    recall = cumulative_tp[distinct] / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def classification_metrics(y_true, y_pred, y_score, class_tokens):
    if not y_true:
        raise ValueError("Cannot evaluate an empty sample collection.")
    positive = class_tokens[-1]
    truth = np.asarray([1 if value == positive else 0 for value in y_true], dtype=np.int64)
    predicted = np.asarray([1 if value == positive else 0 for value in y_pred], dtype=np.int64)

    tp = int(np.sum((truth == 1) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    precision = _safe_divide(tp, tp + fp)
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2.0 * precision * sensitivity, precision + sensitivity)

    class_f1 = []
    for class_token in class_tokens:
        class_tp = sum(a == class_token and b == class_token for a, b in zip(y_true, y_pred))
        class_fp = sum(a != class_token and b == class_token for a, b in zip(y_true, y_pred))
        class_fn = sum(a == class_token and b != class_token for a, b in zip(y_true, y_pred))
        class_precision = _safe_divide(class_tp, class_tp + class_fp)
        class_recall = _safe_divide(class_tp, class_tp + class_fn)
        class_f1.append(_safe_divide(2.0 * class_precision * class_recall,
                                     class_precision + class_recall))

    return {
        "accuracy": _safe_divide(tp + tn, len(y_true)),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "f1_macro": float(np.mean(class_f1)),
        "auc_roc": _binary_auc(truth, y_score),
        "auc_pr": _average_precision(truth, y_score),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_class": positive,
    }


def fidelity_metrics(reference_probabilities, candidate_probabilities, class_tokens):
    if len(reference_probabilities) != len(candidate_probabilities):
        raise ValueError("Reference and candidate prediction counts differ.")
    if not reference_probabilities:
        return {"agreement": math.nan, "mean_kl_divergence": math.nan}

    agreements = []
    divergences = []
    for reference, candidate in zip(reference_probabilities, candidate_probabilities):
        p = np.asarray([reference.get(c, 0.0) for c in class_tokens], dtype=np.float64)
        q = np.asarray([candidate.get(c, 0.0) for c in class_tokens], dtype=np.float64)
        p = np.clip(p / max(p.sum(), 1e-12), 1e-12, 1.0)
        q = np.clip(q / max(q.sum(), 1e-12), 1e-12, 1.0)
        agreements.append(int(np.argmax(p) == np.argmax(q)))
        divergences.append(float(np.sum(p * np.log(p / q))))
    return {
        "agreement": float(np.mean(agreements)),
        "mean_kl_divergence": float(np.mean(divergences)),
    }
