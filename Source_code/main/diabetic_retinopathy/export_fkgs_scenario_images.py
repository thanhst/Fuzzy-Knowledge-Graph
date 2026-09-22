"""Export reviewer-facing FKGS scenario tables as high-resolution PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_LABELS = {
    "table": "FKG-UM (Bảng)",
    "image": "FKG-UM (Ảnh)",
    "fusion": "FKG-MM (Đề xuất)",
    "fusion_filter": "FKG-MM (Filter)",
    "fusion_hadamard": "FKG-MM (Hadamard)",
    "fusion_tensor": "FKG-MM (Tensor-style)",
    "fusion_wrapper": "FKG-MM (Wrapper)",
}

METRICS = [
    ("accuracy_pct", "accuracy_std_pct", "Acc (%)"),
    ("sensitivity_pct", "sensitivity_std_pct", "Sensitivity (%)"),
    ("specificity_pct", "specificity_std_pct", "Specificity (%)"),
    ("f1_pct", "f1_std_pct", "F1 (%)"),
    ("auc_roc_pct", "auc_roc_std_pct", "AUC-ROC (%)"),
    ("auc_pr_pct", "auc_pr_std_pct", "AUC-PR (%)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_mean_std(row: pd.Series, mean_column: str, std_column: str) -> str:
    return f"{float(row[mean_column]):.2f} ± {float(row[std_column]):.2f}"


def scenario_rows(frame: pd.DataFrame, modalities: list[str]) -> tuple[list[str], list[list[str]], list[float]]:
    ordered = frame[frame["modality_key"].isin(modalities)].copy()
    modality_order = {name: index for index, name in enumerate(modalities)}
    ordered["_order"] = ordered["modality_key"].map(modality_order)
    ordered = ordered.sort_values(["ran", "epsilon", "_order"])

    headers = [
        "Cấu hình",
        "Mô hình",
        *[label for _, _, label in METRICS],
        "Train (s)",
        "Test (s)",
        "Total (s)",
    ]
    rows: list[list[str]] = []
    sensitivities: list[float] = []
    for _, row in ordered.iterrows():
        rows.append(
            [
                f"ran={int(row['ran'])}%, ε={float(row['epsilon']):.1f}",
                MODEL_LABELS[str(row["modality_key"])],
                *[
                    format_mean_std(row, mean_column, std_column)
                    for mean_column, std_column, _ in METRICS
                ],
                f"{float(row['train_time_s']):.2f}",
                f"{float(row['test_time_s']):.2f}",
                f"{float(row['end_to_end_time_s']):.2f}",
            ]
        )
        sensitivities.append(float(row["sensitivity_pct"]))
    return headers, rows, sensitivities


def render_table(
    frame: pd.DataFrame,
    modalities: list[str],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    headers, rows, sensitivities = scenario_rows(frame, modalities)
    figure_height = max(6.0, 1.9 + len(rows) * 0.48)
    fig, ax = plt.subplots(figsize=(19.0, figure_height), facecolor="#f8f9fb")
    ax.axis("off")

    column_widths = [0.095, 0.14, 0.095, 0.108, 0.108, 0.095, 0.105, 0.105, 0.067, 0.067, 0.067]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=column_widths,
        bbox=[0.01, 0.09, 0.98, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)

    header_color = "#243447"
    header_text = "#ffffff"
    group_colors = ["#eef4fb", "#f5f7fa"]
    model_colors = {"FKG-UM": "#dcecff", "FKG-MM": "#ddf3e8"}
    low_sensitivity_color = "#fde5e2"

    for column in range(len(headers)):
        cell = table[(0, column)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text, weight="bold")
        cell.set_edgecolor("#ffffff")
        cell.set_linewidth(0.8)

    rows_per_config = len(modalities)
    for row_index, row in enumerate(rows, start=1):
        group_index = (row_index - 1) // rows_per_config
        for column in range(len(headers)):
            cell = table[(row_index, column)]
            cell.set_facecolor(group_colors[group_index % len(group_colors)])
            cell.set_edgecolor("#cbd3dc")
            cell.set_linewidth(0.45)
            cell.set_text_props(color="#17212b")
        table[(row_index, 1)].set_facecolor(
            model_colors["FKG-UM"] if row[1].startswith("FKG-UM") else model_colors["FKG-MM"]
        )
        if sensitivities[row_index - 1] < 10.0:
            table[(row_index, 3)].set_facecolor(low_sensitivity_color)
            table[(row_index, 3)].set_text_props(color="#9b1c1c", weight="bold")

    for row_index in range(1, len(rows) + 1):
        table[(row_index, 0)].set_text_props(weight="bold")
        table[(row_index, 1)].set_text_props(ha="left")

    fig.suptitle(title, fontsize=16, fontweight="bold", color="#17212b", y=0.965)
    fig.text(
        0.5,
        0.925,
        "Patient-grouped stratified 5-fold CV | Trung bình ± độ lệch chuẩn mẫu",
        ha="center",
        fontsize=10.5,
        color="#4a5562",
    )
    fig.text(
        0.012,
        0.035,
        "Lớp dương: diabetic retinopathy. Số bệnh nhân trùng giữa train/validation = 0. "
        "Ô đỏ đánh dấu Sensitivity < 10%.",
        ha="left",
        fontsize=9.5,
        color="#4a5562",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    output_dir = resolve(args.output_dir)
    frame = pd.read_csv(input_path)

    required = {
        "modality_key",
        "ran",
        "epsilon",
        "train_time_s",
        "test_time_s",
        "end_to_end_time_s",
    }
    required.update(column for pair in METRICS for column in pair[:2])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    render_table(
        frame,
        ["table", "image", "fusion"],
        "Kịch bản 1: FKG-MM so với FKG-UM ảnh và bảng",
        output_dir / "scenario1_fkg_mm_vs_um_full_metrics.png",
        args.dpi,
    )
    render_table(
        frame,
        ["fusion", "fusion_filter", "fusion_hadamard", "fusion_tensor", "fusion_wrapper"],
        "Kịch bản 2: So sánh các chiến lược fusion FKG-MM",
        output_dir / "scenario2_fusion_strategies_full_metrics.png",
        args.dpi,
    )
    print(f"[DONE] {output_dir / 'scenario1_fkg_mm_vs_um_full_metrics.png'}")
    print(f"[DONE] {output_dir / 'scenario2_fusion_strategies_full_metrics.png'}")


if __name__ == "__main__":
    main()
