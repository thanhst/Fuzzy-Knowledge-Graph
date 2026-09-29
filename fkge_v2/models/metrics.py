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
    reference_labels = []
    candidate_labels = []
    bound_holds = []
    for reference, candidate in zip(reference_probabilities, candidate_probabilities):
        p = np.asarray([reference.get(c, 0.0) for c in class_tokens], dtype=np.float64)
        q = np.asarray([candidate.get(c, 0.0) for c in class_tokens], dtype=np.float64)
        p = np.clip(p / max(p.sum(), 1e-12), 1e-12, 1.0)
        q = np.clip(q / max(q.sum(), 1e-12), 1e-12, 1.0)
        reference_label = int(np.argmax(p))
        candidate_label = int(np.argmax(q))
        divergence = float(np.sum(p * np.log(p / q)))
        margin = float(np.sort(p)[-1] - np.sort(p)[-2]) if len(p) > 1 else 1.0
        reference_labels.append(reference_label)
        candidate_labels.append(candidate_label)
        agreements.append(int(reference_label == candidate_label))
        divergences.append(divergence)
        bound_holds.append(divergence < margin * margin / 2.0)
    reference_labels = np.asarray(reference_labels)
    candidate_labels = np.asarray(candidate_labels)
    recalls = [float(np.mean(candidate_labels[reference_labels == index] == index))
               for index in range(len(class_tokens)) if np.any(reference_labels == index)]
    reference_freq = np.bincount(reference_labels, minlength=len(class_tokens)) / len(agreements)
    candidate_freq = np.bincount(candidate_labels, minlength=len(class_tokens)) / len(agreements)
    expected_agreement = float(np.dot(reference_freq, candidate_freq))
    observed_agreement = float(np.mean(agreements))
    return {
        "agreement": observed_agreement,
        "fidelity_balanced_accuracy": float(np.mean(recalls)) if len(recalls) == len(class_tokens) else math.nan,
        "cohen_kappa": ((observed_agreement - expected_agreement) / (1 - expected_agreement)
                        if expected_agreement < 1 else math.nan),
        "mean_kl_divergence": float(np.mean(divergences)),
        "fidelity_bound_coverage": float(np.mean(bound_holds)),
        "fidelity_bound_violations": int(sum(hold and not agree
                                              for hold, agree in zip(bound_holds, agreements))),
    }
