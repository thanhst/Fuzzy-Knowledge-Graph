import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULT_DIR = os.path.join(ROOT, "results", "fusion_smoke")


def main(result_dir=DEFAULT_RESULT_DIR):
    json_path = os.path.join(result_dir, "results.json")
    output_path = os.path.join(result_dir, "result_summary.png")
    with open(json_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)

    methods = [
        ("FISA tuần tự", payload["methods"]["FISA sequential"]),
        ("FISA bảng tra", payload["methods"]["FISA lookup"]),
        ("FKG-E không nhãn", payload["methods"]["FKG-E unsupervised"]),
        ("FKG-E đầy đủ", payload["methods"]["FKG-E full"]),
    ]
    columns = [
        "Phương pháp", "AUC-ROC", "AUC-PR", "Accuracy", "BalAcc", "F1",
        "Sensitivity", "Specificity", "ms/mẫu",
    ]
    rows = [
        [
            name,
            f"{metrics['auc_roc']:.4f}",
            f"{metrics['auc_pr']:.4f}",
            f"{metrics['accuracy']:.4f}",
            f"{metrics['balanced_accuracy']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['sensitivity']:.4f}",
            f"{metrics['specificity']:.4f}",
            f"{metrics['avg_time_per_query_ms']:.4f}",
        ]
        for name, metrics in methods
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axis = plt.subplots(figsize=(15, 5.2), facecolor="white")
    axis.axis("off")
    axis.text(0.0, 1.04, "Bảng kết quả BRSET fusion - FKG-E",
              transform=axis.transAxes, fontsize=18, fontweight="bold", va="bottom")
    axis.text(
        0.0, 0.965,
        "Smoke run: fold 1, seed 42, 1 epoch; 1.778 mẫu train, 242 mẫu validation; patient overlap = 0",
        transform=axis.transAxes, fontsize=11, va="bottom",
    )

    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.18, 0.095, 0.095, 0.095, 0.085, 0.075, 0.105, 0.105, 0.09],
        bbox=[0.0, 0.28, 1.0, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#E9ECEF")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
        if column == 0:
            cell.set_text_props(ha="left")

    data = payload["data"]
    graph = payload["graph"]
    full = payload["methods"]["FKG-E full"]
    fisa = payload["methods"]["FISA lookup"]
    note = (
        f"FKG-MM đầu vào: {graph['rule_count']} luật, {graph['token_count']} token, "
        f"{graph['intra_modal_edge_count']} cạnh nội mô thức, "
        f"{graph['cross_modal_edge_count']} cạnh liên mô thức. "
        f"FKG-E đầy đủ tăng AUC-ROC {full['auc_roc'] - fisa['auc_roc']:+.4f} so với FISA."
    )
    axis.text(0.0, 0.19, note, transform=axis.transAxes, fontsize=10.5, va="top")
    axis.text(
        0.0, 0.10,
        f"Baseline lớp đa số: Accuracy {data['majority_accuracy']:.4f}. "
        "Kết quả này chỉ dùng kiểm chứng pipeline, chưa phải kết quả luận án chính thức.",
        transform=axis.transAxes, fontsize=10.5, va="top",
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output_path)
    return output_path


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULT_DIR
    main(directory)
