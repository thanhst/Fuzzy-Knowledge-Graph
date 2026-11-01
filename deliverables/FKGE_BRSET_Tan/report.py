"""CSV/PNG reports: outer folds are the only units in mean and standard deviation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

TARGET = "diabetic_retinopathy"


def binary_metrics(y, prediction, score):
    precision, recall, _ = precision_recall_curve(y, score)
    return {"accuracy": accuracy_score(y, prediction), "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0), "f1": f1_score(y, prediction, zero_division=0),
            "f1_macro": f1_score(y, prediction, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y, score), "average_precision": average_precision_score(y, score),
            "pr_auc_trapezoidal": auc(recall, precision)}


def table_image(frame, path, title):
    fig, ax = plt.subplots(figsize=(max(10, len(frame.columns) * 1.7), 1.8 + len(frame) * 0.5))
    ax.axis("off")
    ax.set_title(title, pad=18)
    shown = frame.map(lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
    table = ax.table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for column in range(len(frame.columns)):
        table[(0, column)].set_height(table[(0, column)].get_height() * 1.65)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_report(output, scope):
    output = Path(output)
    report = output / "reports"
    report.mkdir(exist_ok=True)
    populations, metrics, predictions, selections = [], [], [], []
    for directory in sorted(output.glob("fold_[0-9][0-9]")):
        populations.append(json.loads((directory / "preparation.json").read_text(encoding="utf-8")))
        if not (directory / "results.json").exists():
            continue
        result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        for model in ("FISA", "FKG-E"):
            metrics.append({"fold": result["fold"], "model": model, **result[model]})
        selections.append({"fold": result["fold"], **result["selected_parameters"],
                           "validation_log_loss": result["validation_log_loss"], "epochs": result["selected_epochs"]})
        prediction = pd.read_csv(directory / "predictions.csv", dtype={"image_id": str, "patient_id": str})
        prediction["fold"] = result["fold"]
        predictions.append(prediction)
    counts = pd.DataFrame(populations)
    columns = ["fold", "outer_train_images", "outer_train_patients", "inner_train_patients", "validation_patients",
               "test_images", "test_patients", "rules_including_smote", "smote_rows", "teacher_units", "largest_unit_rows"]
    counts[columns].to_csv(report / "fold_population.csv", index=False)
    names = ["Fold", "Ảnh\nouter train", "Bệnh nhân\nouter train", "Bệnh nhân\ninner train", "Bệnh nhân\nvalidation",
             "Ảnh\ntest", "Bệnh nhân\ntest", "Luật\n(gồm SMOTE)", "Mẫu\nSMOTE", "Nhóm\nteacher", "Nhóm lớn nhất\n(số dòng)"]
    shown_counts = counts[columns].rename(columns=dict(zip(columns, names)))
    table_image(shown_counts, report / "fold_population.png", f"{scope}: số lượng theo fold (nhóm teacher dùng cho tính toán)")
    if not metrics:
        return
    frame = pd.DataFrame(metrics)
    frame.to_csv(report / "metrics_by_fold.csv", index=False)
    pd.DataFrame(selections).to_csv(report / "selected_hyperparameters.csv", index=False)
    score_columns = [column for column in frame if column not in {"fold", "model"}]
    stats = frame.groupby("model")[score_columns].agg(["mean", "std"])
    stats.columns = [f"{metric}_{statistic}" for metric, statistic in stats.columns]
    stats.reset_index().to_csv(report / "metrics_mean_std.csv", index=False)
    shown_metrics = frame[["fold", "model", "roc_auc", "average_precision", "pr_auc_trapezoidal", "f1", "balanced_accuracy"]].rename(
        columns={"fold": "Fold", "model": "Mô hình", "roc_auc": "ROC-AUC", "average_precision": "AP",
                 "pr_auc_trapezoidal": "PR-AUC\n(hình thang)", "f1": "F1", "balanced_accuracy": "Balanced\nAccuracy"})
    table_image(shown_metrics,
                report / "metrics_by_fold.png", f"{scope}: held-out fold metrics")
    table_image(stats.reset_index()[["model", "roc_auc_mean", "roc_auc_std", "average_precision_mean", "average_precision_std"]],
                report / "metrics_mean_std.png", f"{scope}: mean and sample SD across {len(predictions)} outer folds")
    oof = pd.concat(predictions, ignore_index=True)
    if oof.image_id.duplicated().any() or (oof.groupby("patient_id").fold.nunique() != 1).any():
        raise AssertionError("OOF predictions duplicate images or cross patient folds")
    oof.to_csv(report / "predictions_oof.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pooled = []
    for model, prefix in (("FISA", "fisa"), ("FKG-E", "fkge")):
        y, score = oof[TARGET], oof[f"{prefix}_score"]
        values = binary_metrics(y, oof[f"{prefix}_prediction"], score)
        pooled.append({"model": model, **values})
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, label=f"{model}: AUC={values['roc_auc']:.4f}")
        axes[1].plot(recall, precision, label=f"{model}: AP={values['average_precision']:.4f}")
        cm = confusion_matrix(y, oof[f"{prefix}_prediction"], labels=[0, 1])
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(report / f"confusion_{prefix}.csv")
    pd.DataFrame(pooled).to_csv(report / "metrics_pooled_oof.csv", index=False)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].axhline(oof[TARGET].mean(), color="gray", linestyle="--", label="DR prevalence")
    for ax, xlabel, ylabel in ((axes[0], "False positive rate", "True positive rate"), (axes[1], "Recall", "Precision")):
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    fig.suptitle(f"{scope}: pooled held-out predictions ({len(predictions)} folds)")
    fig.tight_layout()
    fig.savefig(report / "roc_pr.png", dpi=170)
    plt.close(fig)
    (report / "README.txt").write_text(
        f"Scope: {scope}\nCompleted outer folds: {len(predictions)}\n"
        "SD uses ddof=1 across outer folds. Teacher partitions, SMOTE rows and memory blocks are not replicates.\n"
        "AP = average_precision_score; PR-AUC trapezoidal is reported separately. Pooled OOF differs from fold mean.\n"
        "A synthetic smoke run is not a BRSET experiment.\n", encoding="utf-8")
