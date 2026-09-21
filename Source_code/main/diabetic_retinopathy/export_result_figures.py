"""Render the result figures for the diabetic retinopathy comparison.

Reads the outputs of :mod:`recompute_full_metrics` (per-fold metrics and the
5-fold summary) plus the raw per-sample predictions, and writes publication
figures as PNG.

Figures
-------
``fig1_metrics_by_model``   small multiples, one panel per metric, bars = models
``fig2_fkg_mm_vs_um``       the multimodal-vs-unimodal FKG comparison
``fig3_fusion_strategies``  FKG-MM under each fusion strategy
``fig4_roc_curves``         ROC curves, predictions pooled over the 5 folds
``fig5_pr_curves``          precision-recall curves, with the prevalence baseline
``fig6_comparison_table``   the summary table as an image

Design notes: colour encodes the model *family* (three fixed hues, never
cycled), never the rank of a bar, so a figure stays readable if a model is
added or dropped. Every bar carries a visible value label, which is also what
lets the lighter hues be used on a light surface. One measure per axis; no
dual-axis panels.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from recompute_full_metrics import (  # noqa: E402
    DEEP_MODEL_LABELS,
    DEEP_PREDICTION_PATTERN,
    FKG_MODALITIES,
    read_rows,
)

# Categorical slots 1-3 of the validated palette, assigned to families in a
# fixed order. Three slots is the cap that clears the all-pairs colour-vision
# floors, which is why family (not model) carries the hue.
FAMILY_COLORS = {
    "deep_baseline_KFold": "#2a78d6",
    "FKG_KFold": "#eb6834",
    "FKGS_KFold": "#1baf7a",
}
FAMILY_LABELS = {
    "deep_baseline_KFold": "Deep baseline",
    "FKG_KFold": "FKG (native)",
    "FKGS_KFold": "FKG-S (sampling)",
}
# Line charts use the adjacent pairlist, where the full eight-hue order is
# validated; curves are also direct-labelled at their right edge.
LINE_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dcdcd8"
SURFACE = "#fcfcfb"

PANEL_METRICS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("sensitivity", "Sensitivity (Recall+)"),
    ("specificity", "Specificity"),
    ("f1", "F1 (positive class)"),
    ("auc_roc", "AUC-ROC"),
    ("auc_pr", "AUC-PR"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render result figures as PNG.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--by-fold", type=Path, required=True)
    parser.add_argument("--deep-predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--deep-split", default="val")
    parser.add_argument("--fkg-output-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--title-suffix",
        default="",
        help="Appended to every figure title, e.g. a run id.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def to_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def load_summary(path: Path) -> List[Dict[str, str]]:
    rows = read_rows(path)
    order = {family: index for index, family in enumerate(FAMILY_COLORS)}
    rows.sort(
        key=lambda row: (
            order.get(row["source_family"], 99),
            -to_float(row.get("auc_pr_mean")) if not math.isnan(to_float(row.get("auc_pr_mean"))) else 0.0,
        )
    )
    return rows


def figure_metrics_by_model(
    summary: Sequence[Dict[str, str]],
    output_path: Path,
    dpi: int,
    title_suffix: str,
) -> None:
    """Small multiples: one panel per metric so no panel needs a second scale."""
    labels = [row["model"] for row in summary]
    colors = [FAMILY_COLORS.get(row["source_family"], "#888888") for row in summary]
    y = np.arange(len(labels))[::-1]

    ncols = 3
    nrows = int(math.ceil(len(PANEL_METRICS) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.1 * ncols, 0.42 * len(labels) * nrows + 1.9 * nrows),
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes).ravel()

    for index, (metric, title) in enumerate(PANEL_METRICS):
        ax = axes[index]
        means = np.array([to_float(row.get(f"{metric}_mean")) * 100 for row in summary])
        stds = np.array([to_float(row.get(f"{metric}_std")) * 100 for row in summary])
        stds = np.nan_to_num(stds, nan=0.0)
        ax.barh(
            y,
            np.nan_to_num(means, nan=0.0),
            height=0.62,
            color=colors,
            xerr=stds,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.0, "capsize": 2.5},
        )
        # Place the label clear of the error-bar cap, not on top of it.
        for y_pos, mean, std in zip(y, means, stds):
            if math.isnan(mean):
                continue
            ax.text(
                mean + std + 2.5,
                y_pos,
                f"{mean:.1f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color=TEXT_PRIMARY,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels if index % ncols == 0 else [""] * len(labels), fontsize=8)
        if index % ncols != 0:
            ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, pad=8)
        style_axes(ax)

    for ax in axes[len(PANEL_METRICS):]:
        ax.set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for family, color in FAMILY_COLORS.items()
        if any(row["source_family"] == family for row in summary)
    ]
    names = [
        FAMILY_LABELS[family]
        for family in FAMILY_COLORS
        if any(row["source_family"] == family for row in summary)
    ]
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=len(names),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        f"Patient-level 5-fold results, all metrics on the positive class (%){title_suffix}",
        fontsize=12,
        color=TEXT_PRIMARY,
        y=0.995,
    )
    fig.text(
        0.5,
        0.028,
        "Error bars: standard deviation across the 5 patient-level folds (ddof=1).",
        ha="center",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.975))
    fig.savefig(output_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def figure_grouped_models(
    summary: Sequence[Dict[str, str]],
    wanted: Sequence[str],
    output_path: Path,
    title: str,
    dpi: int,
    note: str = "",
) -> None:
    """Grouped bars: a few named models across every metric."""
    rows = [row for name in wanted for row in summary if row["model"] == name]
    if not rows:
        print(f"[WARN] No rows matched {list(wanted)}; skipping {output_path.name}")
        return

    # A model named in the title but absent from the data must be called out on
    # the figure; a chart titled "A vs B" that silently plots only A misleads.
    found = {row["model"] for row in rows}
    missing = [name for name in wanted if name not in found]
    if missing:
        warning = "KHONG CO DU LIEU: " + ", ".join(missing)
        note = f"{note}  |  {warning}" if note else warning
        print(f"[WARN] {output_path.name}: missing {missing}")

    x = np.arange(len(PANEL_METRICS))
    width = min(0.8 / len(rows), 0.28)
    fig, ax = plt.subplots(figsize=(11, 5.0), facecolor=SURFACE)

    for index, row in enumerate(rows):
        offset = (index - (len(rows) - 1) / 2) * width
        means = [to_float(row.get(f"{metric}_mean")) * 100 for metric, _ in PANEL_METRICS]
        stds = [to_float(row.get(f"{metric}_std")) * 100 for metric, _ in PANEL_METRICS]
        bars = ax.bar(
            x + offset,
            np.nan_to_num(means, nan=0.0),
            width * 0.92,
            label=row["model"],
            color=LINE_COLORS[index % len(LINE_COLORS)],
            yerr=np.nan_to_num(stds, nan=0.0),
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.0, "capsize": 2.5},
        )
        for bar, mean in zip(bars, means):
            if math.isnan(mean):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 1.6,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=TEXT_PRIMARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([title for _, title in PANEL_METRICS], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("%", fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, pad=10)
    ax.legend(frameon=False, fontsize=9, ncol=min(len(rows), 4))
    style_axes(ax)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(output_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def collect_pooled_predictions(
    deep_dirs: Sequence[Path],
    deep_split: str,
    fkg_root: Path | None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Pool every fold's predictions per model into one (y_true, y_score) pair."""
    pooled: Dict[str, Tuple[List[int], List[float]]] = {}

    for predictions_dir in deep_dirs:
        for path in sorted(predictions_dir.glob("*_predictions.csv")):
            match = DEEP_PREDICTION_PATTERN.match(path.name)
            if match is None or match.group("split") != deep_split:
                continue
            label = DEEP_MODEL_LABELS.get(match.group("model"), (match.group("model"), ""))[0]
            truth, score = pooled.setdefault(label, ([], []))
            for row in read_rows(path):
                truth.append(1 if int(float(row["true_label"])) == 1 else 0)
                score.append(float(row["score_positive"]))

    if fkg_root is not None:
        for display_name, (label, _data_type) in FKG_MODALITIES.items():
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
                positive_label = max(set(true_labels))
                truth, score = pooled.setdefault(label, ([], []))
                for row, value in zip(rows, true_labels):
                    truth.append(1 if value == positive_label else 0)
                    score.append(float(row["positive_score"]))

    return {
        label: (np.asarray(truth), np.asarray(score))
        for label, (truth, score) in pooled.items()
        if truth and len(set(truth)) == 2
    }


def roc_points(truth: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    truth_sorted = truth[order]
    tps = np.cumsum(truth_sorted)
    fps = np.cumsum(1 - truth_sorted)
    tpr = np.concatenate(([0.0], tps / max(1, tps[-1])))
    fpr = np.concatenate(([0.0], fps / max(1, fps[-1])))
    return fpr, tpr


def pr_points(truth: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    truth_sorted = truth[order]
    tps = np.cumsum(truth_sorted)
    predicted = np.arange(1, truth.size + 1)
    recall = tps / max(1, int(truth.sum()))
    precision = tps / predicted
    return recall, precision


def figure_curves(
    pooled: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    kind: str,
    dpi: int,
    title_suffix: str,
) -> None:
    if not pooled:
        print(f"[WARN] No pooled predictions; skipping {output_path.name}")
        return

    fig, ax = plt.subplots(figsize=(7.4, 6.2), facecolor=SURFACE)
    labels = sorted(pooled)
    for index, label in enumerate(labels):
        truth, score = pooled[label]
        if kind == "roc":
            x, y = roc_points(truth, score)
        else:
            x, y = pr_points(truth, score)
        ax.plot(
            x,
            y,
            linewidth=2.0,
            color=LINE_COLORS[index % len(LINE_COLORS)],
            label=label,
        )

    if kind == "roc":
        ax.plot([0, 1], [0, 1], linewidth=1.2, linestyle="--", color=TEXT_SECONDARY, alpha=0.6)
        ax.set_xlabel("False positive rate (1 - Specificity)", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel("True positive rate (Sensitivity)", fontsize=9, color=TEXT_SECONDARY)
        ax.set_title(f"ROC, predictions pooled over 5 folds{title_suffix}", fontsize=12, color=TEXT_PRIMARY)
    else:
        any_truth = next(iter(pooled.values()))[0]
        prevalence = float(any_truth.mean())
        ax.axhline(prevalence, linewidth=1.2, linestyle="--", color=TEXT_SECONDARY, alpha=0.6)
        ax.text(
            0.99,
            prevalence + 0.012,
            f"random = prevalence {prevalence * 100:.1f}%",
            ha="right",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
        ax.set_xlabel("Recall (Sensitivity)", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel("Precision", fontsize=9, color=TEXT_SECONDARY)
        ax.set_title(
            f"Precision-Recall, predictions pooled over 5 folds{title_suffix}",
            fontsize=12,
            color=TEXT_PRIMARY,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="upper right" if kind == "roc" else "upper right")
    style_axes(ax)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def figure_table(
    summary: Sequence[Dict[str, str]],
    output_path: Path,
    dpi: int,
    title_suffix: str,
) -> None:
    header = ["Model", "Data"] + [title.split(" (")[0] for _, title in PANEL_METRICS]
    body = []
    for row in summary:
        cells = [row["model"], row["data_type"]]
        for metric, _title in PANEL_METRICS:
            mean = to_float(row.get(f"{metric}_mean")) * 100
            std = to_float(row.get(f"{metric}_std")) * 100
            cells.append("n/a" if math.isnan(mean) else f"{mean:.1f} ± {0.0 if math.isnan(std) else std:.1f}")
        body.append(cells)

    fig, ax = plt.subplots(
        figsize=(1.55 * len(header), 0.42 * (len(body) + 2)),
        facecolor=SURFACE,
    )
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=header, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#eef1f4")
            cell.set_text_props(color=TEXT_PRIMARY, fontweight="bold")
        else:
            family = summary[row_index - 1]["source_family"]
            cell.set_facecolor("#ffffff" if row_index % 2 else "#f7f7f5")
            if col_index == 0:
                cell.set_text_props(color=FAMILY_COLORS.get(family, TEXT_PRIMARY))
    ax.set_title(
        f"Patient-level 5-fold summary, mean ± std (%){title_suffix}",
        fontsize=12,
        color=TEXT_PRIMARY,
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = load_summary(resolve(args.summary))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" - {args.title_suffix}" if args.title_suffix else ""

    figure_metrics_by_model(summary, output_dir / "fig1_metrics_by_model.png", args.dpi, suffix)

    figure_grouped_models(
        summary,
        ["FKG-UM (Anh)", "FKG-UM (Bang)", "FKG-MM (de xuat)"],
        output_dir / "fig2_fkg_mm_vs_um.png",
        "FKG-MM vs FKG-UM" + suffix,
        args.dpi,
        note="Positive class = diabetic retinopathy. Error bars: std across 5 patient-level folds.",
    )

    fusion_models = [
        row["model"]
        for row in summary
        if row["model"].startswith("FKG-MM") and row["source_family"] == "FKG_KFold"
    ]
    figure_grouped_models(
        summary,
        fusion_models,
        output_dir / "fig3_fusion_strategies.png",
        "FKG-MM: fusion strategy comparison" + suffix,
        args.dpi,
        note="Each bar is the same FKG model over a different multimodal feature-fusion strategy.",
    )

    pooled = collect_pooled_predictions(
        [resolve(path) for path in args.deep_predictions],
        args.deep_split,
        resolve(args.fkg_output_root) if args.fkg_output_root else None,
    )
    figure_curves(pooled, output_dir / "fig4_roc_curves.png", "roc", args.dpi, suffix)
    figure_curves(pooled, output_dir / "fig5_pr_curves.png", "pr", args.dpi, suffix)

    figure_table(summary, output_dir / "fig6_comparison_table.png", args.dpi, suffix)

    for path in sorted(output_dir.glob("*.png")):
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
