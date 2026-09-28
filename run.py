"""Run the complete patient-aware FKG-MM pipeline on all five folds."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fis import FIS
from fkg_mm import FKGMM
from preprocess import load_source, prepare_fold


METRICS = ("accuracy", "precision", "sensitivity", "specificity", "f1", "auc_roc")


def rank_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    if not positives or not negatives:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[truth == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def score(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> dict:
    positive = int(y_true.max())
    truth, predicted = y_true == positive, y_pred == positive
    tp = int(np.sum(truth & predicted))
    tn = int(np.sum(~truth & ~predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    divide = lambda a, b: a / b if b else 0.0
    precision = divide(tp, tp + fp)
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    positive_scores = np.where(predicted, confidence, 1.0 - confidence)
    return {
        "accuracy": divide(tp + tn, len(y_true)),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": divide(2 * precision * sensitivity, precision + sensitivity),
        "auc_roc": rank_auc(truth.astype(int), positive_scores),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def read_rules(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=np.int64)


def write_predictions(path: Path, truth, predicted, confidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = int(np.max(truth))
    pd.DataFrame({
        "true_label": truth,
        "predicted_label": predicted,
        "confidence": confidence,
        "positive_score": np.where(predicted == positive, confidence, 1.0 - confidence),
    }).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reference-dir", type=Path, default=Path("reference"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, labels, ids = load_source(args.data_dir)
    outer_ids = set(pd.read_csv(args.data_dir / "splits" / "train.csv", dtype=str)["image_id"])
    mask = ids["image_id"].isin(outer_ids)
    features, labels, ids = (
        features.loc[mask].reset_index(drop=True),
        labels.loc[mask].reset_index(drop=True),
        ids.loc[mask].reset_index(drop=True),
    )
    if len(ids) != 1208:
        raise RuntimeError(f"expected 1,208 outer-train images, got {len(ids)}")

    rows = []
    all_rules_match = True
    all_predictions_match = True
    for fold in args.folds:
        fold_start = time.perf_counter()
        prepared = prepare_fold(features, labels, ids, args.data_dir / "splits", fold, args.seed)
        fold_name = f"fold_{fold:02d}"
        fold_output = args.output_dir / fold_name
        fold_output.mkdir(parents=True, exist_ok=True)
        prepared["train"].to_csv(fold_output / "train_data.csv", index=False)
        prepared["test"].to_csv(fold_output / "test_data.csv", index=False)
        prepared["train_ids"].to_csv(fold_output / "train_ids.csv", index=False)
        prepared["test_ids"].to_csv(fold_output / "test_ids.csv", index=False)
        prepared["selected_features"].to_csv(fold_output / "selected_features.csv", index=False)

        train = prepared["train"].to_numpy(dtype=float)
        test = prepared["test"].to_numpy(dtype=float)
        fis_start = time.perf_counter()
        fis = FIS().fit(train[:, :-1])
        train_rules = fis.make_rules(train[:, :-1], train[:, -1])
        test_rules = fis.make_rules(test[:, :-1], test[:, -1])
        fis_seconds = time.perf_counter() - fis_start
        pd.DataFrame(train_rules).to_csv(fold_output / "train_rules.csv", index=False)
        pd.DataFrame(test_rules).to_csv(fold_output / "test_rules.csv", index=False)

        reference_fold = args.reference_dir / fold_name
        train_rules_match = np.array_equal(train_rules, read_rules(reference_fold / "train_rules.csv"))
        test_rules_match = np.array_equal(test_rules, read_rules(reference_fold / "test_rules.csv"))
        all_rules_match &= train_rules_match and test_rules_match

        model_start = time.perf_counter()
        model = FKGMM().fit(train_rules)
        train_seconds = time.perf_counter() - model_start
        test_start = time.perf_counter()
        predicted, confidence = model.predict_batch_with_confidence(test_rules[:, :-1])
        test_seconds = time.perf_counter() - test_start
        write_predictions(fold_output / "predictions.csv", test_rules[:, -1], predicted, confidence)

        reference_predictions = pd.read_csv(Path("results/reference_predictions") / f"{fold_name}.csv")
        prediction_match = np.array_equal(
            predicted, reference_predictions["predicted_label"].to_numpy(dtype=int)
        )
        all_predictions_match &= prediction_match
        row = {
            "fold": fold,
            "train_source_rows": len(prepared["train_ids"]),
            "train_rows_after_smote": len(train),
            "validation_rows": len(test),
            "train_patients": prepared["train_patients"],
            "validation_patients": prepared["test_patients"],
            "feature_count": train.shape[1] - 1,
            **score(test_rules[:, -1], predicted, confidence),
            "reference_train_rules_match": train_rules_match,
            "reference_test_rules_match": test_rules_match,
            "reference_predictions_match": prediction_match,
            "fis_seconds": fis_seconds,
            "fkg_train_seconds": train_seconds,
            "fkg_test_seconds": test_seconds,
            "total_seconds": time.perf_counter() - fold_start,
        }
        rows.append(row)
        print(
            f"[{fold_name}] acc={row['accuracy']:.4f} f1={row['f1']:.4f} "
            f"auc={row['auc_roc']:.4f} rules={train_rules_match and test_rules_match} "
            f"predictions={prediction_match}"
        )

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    summary = {
        "folds": len(rows),
        "reference_rules_match": all_rules_match,
        "reference_predictions_match": all_predictions_match,
    }
    for metric in METRICS:
        values = per_fold[metric].to_numpy(dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFKG-MM patient-aware validation (mean +/- sample std)")
    for metric in ("accuracy", "f1", "auc_roc", "specificity", "sensitivity"):
        print(f"{metric:<13} {100 * summary[metric + '_mean']:5.1f} +/- {100 * summary[metric + '_std']:4.1f} %")
    print(f"reference rules match: {all_rules_match}")
    print(f"reference predictions match: {all_predictions_match}")
    print(f"outputs: {args.output_dir.resolve()}")
    if args.check_reference and not (all_rules_match and all_predictions_match):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
