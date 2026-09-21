"""Rebuild the full metric table from per-sample predictions already on disk.

The deep baselines and the native FKG runs both write one CSV per fold holding
``(true label, predicted label, positive-class score)`` for every test sample.
That is everything the added metrics need, so Sensitivity / Specificity / F1 /
AUC-ROC / AUC-PR can be produced for those two families without paying for
another multi-hour rerun, and -- more importantly for a reviewer -- every row
of the output table is derived by the same code in
:mod:`metrics_utils` from raw predictions rather than copied from whatever each
runner happened to print.

FKGS (FKG-S) is deliberately absent here: the sampling-based runner never wrote
its predictions, so its extra metrics require a rerun.

Usage
-----
    python Source_code/main/diabetic_retinopathy/recompute_full_metrics.py \
        --deep-predictions ROOT_DATA/train_test_selection/deep_baselines/<run>/predictions \
        --fkg-output-root "Source_code/data/FIS/output" \
        --output-stem result/diabetic_retinopathy_full_metrics
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from metrics_utils import (  # noqa: E402  (path setup must run first)
    METRIC_NAMES,
    aggregate_metrics,
    binary_classification_metrics,
    format_mean_std,
)

DEEP_PREDICTION_PATTERN = re.compile(
    r"^(?P<model>.+)_fold_(?P<fold>\d+)_(?P<split>[a-z]+)_predictions\.csv$"
)

DEEP_MODEL_LABELS = {
    "mlp": ("MLP", "Tabular"),
    "resnet": ("ResNet", "Image"),
    "early_fusion": ("Early Fusion (MLP)", "Multimodal"),
    "late_fusion": ("Late Fusion (Ensemble)", "Multimodal"),
}

# Display-name -> (row label, data type). These are the folder names the KFold
# runner creates under the FIS output root.
FKG_MODALITIES = {
    "Diabetic Retinopathy Image Feature FT Selection KFold": ("FKG-UM (Anh)", "Image"),
    "Diabetic Retinopathy Metadata Feature FT Selection KFold": ("FKG-UM (Bang)", "Tabular"),
    "Diabetic Retinopathy Fusion Feature FT Selection KFold": ("FKG-MM (de xuat)", "Multimodal"),
    "Diabetic Retinopathy Fusion Feature Filter KFold": ("FKG-MM (Filter)", "Multimodal"),
    "Diabetic Retinopathy Fusion Feature Hadamard KFold": ("FKG-MM (Hadamard)", "Multimodal"),
    "Diabetic Retinopathy Fusion Feature Tensor KFold": ("FKG-MM (Tensor)", "Multimodal"),
    "Diabetic Retinopathy Fusion Feature Wrapper KFold": ("FKG-MM (Wrapper)", "Multimodal"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Accuracy/Precision/Sensitivity/Specificity/F1/AUC-ROC/AUC-PR "
            "for every fold from the saved per-sample predictions."
        )
    )
    parser.add_argument(
        "--deep-predictions",
        type=Path,
        nargs="*",
        default=[],
        help="One or more deep baseline predictions/ directories.",
    )
    parser.add_argument(
        "--deep-split",
        default="val",
        help="Which deep baseline eval split to score (val or test).",
    )
    parser.add_argument(
        "--fkg-output-root",
        type=Path,
        default=None,
        help="FIS output root that holds <display name>/fold_XX/Predictions_FKG.csv.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        required=True,
        help="Output path without extension; writes _by_fold.csv, _summary.csv and .md.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Decimal places in the markdown table.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def score_deep_predictions(
    predictions_dir: Path,
    split: str,
) -> Dict[str, List[Dict[str, float]]]:
    """Return {model_key: [fold metrics, ...]} for one deep baseline run."""
    by_model: Dict[str, List[Dict[str, float]]] = {}
    for path in sorted(predictions_dir.glob("*_predictions.csv")):
        match = DEEP_PREDICTION_PATTERN.match(path.name)
        if match is None or match.group("split") != split:
            continue
        rows = read_rows(path)
        if not rows:
            continue
        metrics = binary_classification_metrics(
            y_true=[row["true_label"] for row in rows],
            y_pred=[row["pred_label"] for row in rows],
            y_score=[float(row["score_positive"]) for row in rows],
            positive_label=1,
            labels=[0, 1],
        )
        metrics["fold"] = float(match.group("fold"))
        metrics["source_path"] = str(path)
        by_model.setdefault(match.group("model"), []).append(metrics)
    for folds in by_model.values():
        folds.sort(key=lambda item: item["fold"])
    return by_model


def score_fkg_predictions(fkg_root: Path) -> Dict[str, List[Dict[str, float]]]:
    """Return {display_name: [fold metrics, ...]} for the native FKG runs.

    The FKG rule files are 1-based, so the diabetic-retinopathy class is the
    larger of the two labels. It is read per fold from the labels actually
    present instead of being hard-coded, and a fold whose test split somehow
    holds a single class is reported rather than silently scored.
    """
    by_modality: Dict[str, List[Dict[str, float]]] = {}
    for display_name in sorted(FKG_MODALITIES):
        modality_dir = fkg_root / display_name
        if not modality_dir.is_dir():
            continue
        for fold_dir in sorted(modality_dir.glob("fold_*")):
            path = fold_dir / "Predictions_FKG.csv"
            if not path.exists():
                continue
            rows = read_rows(path)
            if not rows:
                continue
            true_labels = [int(float(row["true_label"])) for row in rows]
            pred_labels = [int(float(row["predicted_label"])) for row in rows]
            present = sorted(set(true_labels))
            if len(present) != 2:
                print(
                    f"[WARN] {path} test fold holds labels {present}; skipping.",
                    file=sys.stderr,
                )
                continue
            positive_label = max(present)
            scores = [float(row["positive_score"]) for row in rows]
            metrics = binary_classification_metrics(
                y_true=true_labels,
                y_pred=pred_labels,
                y_score=scores,
                positive_label=positive_label,
                labels=sorted(set(true_labels) | set(pred_labels)),
            )
            metrics["fold"] = float(fold_dir.name.split("_")[-1])
            metrics["source_path"] = str(path)
            by_modality.setdefault(display_name, []).append(metrics)
    for folds in by_modality.values():
        folds.sort(key=lambda item: item["fold"])
    return by_modality


def write_by_fold(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    header = ["model", "data_type", "source_family", "fold"] + METRIC_NAMES + [
        "tp",
        "tn",
        "fp",
        "fn",
        "n",
        "n_pos",
        "n_neg",
        "prevalence",
        "positive_label",
        "score_kind",
        "source_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    header = ["model", "data_type", "source_family", "folds"]
    for name in METRIC_NAMES:
        header += [f"{name}_mean", f"{name}_std", f"{name}_n"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


MARKDOWN_COLUMNS = [
    ("accuracy", "Accuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("precision", "Precision"),
    ("f1", "F1"),
    ("auc_roc", "AUC-ROC"),
    ("auc_pr", "AUC-PR"),
]


def write_markdown(
    rows: Sequence[Dict[str, object]],
    path: Path,
    decimals: int,
    prevalence: float | None,
) -> None:
    lines = [
        "# Diabetic retinopathy KFold - full metric set",
        "",
        "All values are mean +/- standard deviation across the 5 patient-level folds",
        "(std over folds, ddof=1). Sensitivity, Specificity, Precision and F1 are",
        "positive-class (diabetic retinopathy) values, not macro averages, so every",
        "row of this table means the same thing.",
        "",
    ]
    if prevalence is not None:
        lines += [
            f"AUC-PR baseline for a random model on this test prevalence: "
            f"{prevalence * 100:.1f}%.",
            "",
        ]
    header = ["Model", "Data"] + [title for _, title in MARKDOWN_COLUMNS] + ["Folds"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        cells = [str(row["model"]), str(row["data_type"])]
        for name, _title in MARKDOWN_COLUMNS:
            cells.append(format_mean_std(row, name, decimals=decimals))
        cells.append(f"{int(float(row['folds']))}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    by_fold_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    prevalences: List[float] = []

    for predictions_dir in args.deep_predictions:
        predictions_dir = resolve(predictions_dir)
        if not predictions_dir.is_dir():
            raise FileNotFoundError(f"Deep predictions directory not found: {predictions_dir}")
        for model_key, folds in score_deep_predictions(predictions_dir, args.deep_split).items():
            label, data_type = DEEP_MODEL_LABELS.get(model_key, (model_key, "unknown"))
            for metrics in folds:
                by_fold_rows.append(
                    {
                        "model": label,
                        "data_type": data_type,
                        "source_family": "deep_baseline_KFold",
                        **metrics,
                    }
                )
                prevalences.append(float(metrics["prevalence"]))
            summary_rows.append(
                {
                    "model": label,
                    "data_type": data_type,
                    "source_family": "deep_baseline_KFold",
                    **aggregate_metrics(folds),
                }
            )

    if args.fkg_output_root is not None:
        fkg_root = resolve(args.fkg_output_root)
        if not fkg_root.is_dir():
            raise FileNotFoundError(f"FKG output root not found: {fkg_root}")
        for display_name, folds in score_fkg_predictions(fkg_root).items():
            label, data_type = FKG_MODALITIES[display_name]
            for metrics in folds:
                by_fold_rows.append(
                    {
                        "model": label,
                        "data_type": data_type,
                        "source_family": "FKG_KFold",
                        **metrics,
                    }
                )
                prevalences.append(float(metrics["prevalence"]))
            summary_rows.append(
                {
                    "model": label,
                    "data_type": data_type,
                    "source_family": "FKG_KFold",
                    **aggregate_metrics(folds),
                }
            )

    if not summary_rows:
        raise SystemExit("Nothing to score: pass --deep-predictions and/or --fkg-output-root.")

    output_stem = resolve(args.output_stem)
    by_fold_path = output_stem.with_name(output_stem.name + "_by_fold.csv")
    summary_path = output_stem.with_name(output_stem.name + "_summary.csv")
    markdown_path = output_stem.with_suffix(".md")

    write_by_fold(by_fold_rows, by_fold_path)
    write_summary(summary_rows, summary_path)
    write_markdown(
        summary_rows,
        markdown_path,
        args.decimals,
        sum(prevalences) / len(prevalences) if prevalences else None,
    )

    print(f"[OK] per-fold metrics: {by_fold_path}")
    print(f"[OK] summary metrics : {summary_path}")
    print(f"[OK] markdown table  : {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
