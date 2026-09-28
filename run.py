"""Reproduce the patient-aware five-fold FKG-MM result."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from fkg_mm import FKGMM


METRICS = ("accuracy", "precision", "sensitivity", "specificity", "f1", "auc_roc")


def read_rules(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    return data.astype(np.int64)


def read_patient_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["patient_id"] for row in csv.DictReader(handle)}


def rank_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(truth.sum())
    negative_count = int(len(truth) - positive_count)
    if not positive_count or not negative_count:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    rank_sum = float(ranks[truth == 1].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def metrics(y_true, y_pred, scores, positive_label: int) -> dict[str, float]:
    truth = y_true == positive_label
    predicted = y_pred == positive_label
    tp = int(np.sum(truth & predicted))
    tn = int(np.sum(~truth & ~predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))

    divide = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    precision = divide(tp, tp + fp)
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        "accuracy": divide(tp + tn, len(y_true)),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": divide(2 * precision * sensitivity, precision + sensitivity),
        "auc_roc": rank_auc(truth.astype(np.int64), scores),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([int(float(row["predicted_label"])) for row in rows]),
        np.asarray([float(row["confidence"]) for row in rows]),
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="fail unless labels match the archived native FKG predictions",
    )
    args = parser.parse_args()

    fold_rows: list[dict] = []
    all_reference_match = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold_dir in sorted(args.data_dir.glob("fold_*")):
        train_patients = read_patient_ids(fold_dir / "train_ids.csv")
        test_patients = read_patient_ids(fold_dir / "test_ids.csv")
        overlap = train_patients & test_patients
        if overlap:
            raise RuntimeError(f"patient leakage in {fold_dir.name}: {sorted(overlap)[:5]}")

        train = read_rules(fold_dir / "train_rules.csv")
        test = read_rules(fold_dir / "test_rules.csv")
        model = FKGMM()

        started = time.perf_counter()
        model.fit(train)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted, confidence = model.predict_batch_with_confidence(test[:, :-1])
        test_seconds = time.perf_counter() - started

        positive_label = int(test[:, -1].max())
        positive_scores = np.where(
            predicted == positive_label, confidence, 1.0 - confidence
        )
        row = {
            "fold": fold_dir.name,
            "train_rule_rows": len(train),
            "validation_rows": len(test),
            "train_patients": len(train_patients),
            "validation_patients": len(test_patients),
            "patient_overlap": len(overlap),
            "feature_count": train.shape[1] - 1,
            **metrics(test[:, -1], predicted, positive_scores, positive_label),
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
        }

        reference_path = Path("results/reference_predictions") / f"{fold_dir.name}.csv"
        reference_labels, reference_confidence = read_reference(reference_path)
        labels_match = bool(np.array_equal(predicted, reference_labels))
        confidence_delta = float(np.max(np.abs(confidence - reference_confidence)))
        row["reference_labels_match"] = labels_match
        row["max_confidence_delta"] = confidence_delta
        all_reference_match &= labels_match
        fold_rows.append(row)

        prediction_rows = [
            {
                "true_label": int(true),
                "predicted_label": int(pred),
                "confidence": float(conf),
                "positive_score": float(score),
            }
            for true, pred, conf, score in zip(
                test[:, -1], predicted, confidence, positive_scores
            )
        ]
        write_rows(args.output_dir / "predictions" / f"{fold_dir.name}.csv", prediction_rows)

    if not fold_rows:
        raise RuntimeError(f"no fold_* directories found below {args.data_dir}")

    summary = {"folds": len(fold_rows), "reference_labels_match": all_reference_match}
    for name in METRICS:
        values = np.asarray([float(row[name]) for row in fold_rows])
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=1))

    write_rows(args.output_dir / "per_fold_metrics.csv", fold_rows)
    write_rows(args.output_dir / "summary.csv", [summary])
    (args.output_dir / "run.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nFKG-MM patient-aware 5-fold validation")
    print("metric        mean +/- std (%)")
    for name in ("accuracy", "f1", "auc_roc", "specificity", "sensitivity"):
        print(
            f"{name:<13} {100 * summary[name + '_mean']:5.1f} +/- "
            f"{100 * summary[name + '_std']:4.1f}"
        )
    print(f"reference labels match: {all_reference_match}")
    print(f"outputs: {args.output_dir.resolve()}")

    if args.check_reference and not all_reference_match:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
