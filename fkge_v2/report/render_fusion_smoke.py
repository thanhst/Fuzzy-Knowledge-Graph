import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULT_DIR = os.path.join(ROOT, "results", "fusion_smoke")


def main(result_dir=DEFAULT_RESULT_DIR):
    json_path = os.path.join(result_dir, "results.json")
    output_path = os.path.join(result_dir, "result_summary.png")
    with open(json_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)

    methods = list(payload["methods"])
    short_names = ["FISA tuần tự", "FISA bảng tra", "FKG-E không nhãn", "FKG-E đầy đủ"]
    colors = ["#315A7D", "#4A9D8F", "#D49A32", "#C9514A"]
    metric_specs = [
        ("auc_roc", "AUC-ROC"),
        ("auc_pr", "AUC-PR"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "figure.facecolor": "#F7F8FA",
        "axes.facecolor": "#FFFFFF",
    })
    fig = plt.figure(figsize=(15, 9), facecolor="#F7F8FA")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.8, 3.2, 2.2], hspace=0.42, wspace=0.28)

    title_axis = fig.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(0.0, 0.78, "KẾT QUẢ KIỂM CHỨNG FKG-E TRÊN BRSET FUSION",
                    fontsize=22, fontweight="bold", color="#1F2933", va="center")
    title_axis.text(
        0.0, 0.25,
        "Smoke run: 1 fold • 1 seed • 1 epoch • Không phải kết quả luận án chính thức",
        fontsize=12, color="#B13C35", va="center",
    )
    protocol = payload["protocol"]
    data = payload["data"]
    graph = payload["graph"]
    title_axis.text(
        1.0, 0.55,
        f"Fold {protocol['fold']}  |  Seed {protocol['seed']}  |  "
        f"Train {data['train_rows']}  |  Validation {data['validation_rows']}",
        fontsize=11, color="#425466", ha="right", va="center",
    )

    quality_axis = fig.add_subplot(grid[1, 0])
    x = np.arange(len(methods))
    width = 0.24
    offsets = [-width, 0, width]
    for offset, (field, label) in zip(offsets, metric_specs):
        values = [payload["methods"][method][field] for method in methods]
        bars = quality_axis.bar(x + offset, values, width, label=label)
        for bar, value in zip(bars, values):
            quality_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                              f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    quality_axis.set_ylim(0, 1.05)
    quality_axis.set_ylabel("Điểm số")
    quality_axis.set_title("Chất lượng dự đoán")
    quality_axis.set_xticks(x)
    quality_axis.set_xticklabels(short_names, rotation=13, ha="right")
    quality_axis.grid(axis="y", alpha=0.22)
    quality_axis.legend(loc="lower left", fontsize=9, frameon=False)

    speed_axis = fig.add_subplot(grid[1, 1])
    times = [payload["methods"][method]["avg_time_per_query_ms"] for method in methods]
    bars = speed_axis.barh(short_names, times, color=colors)
    speed_axis.set_xscale("log")
    speed_axis.set_xlabel("Thời gian suy diễn trung bình (ms/mẫu, thang log)")
    speed_axis.set_title("Chi phí suy diễn")
    speed_axis.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, times):
        speed_axis.text(value * 1.08, bar.get_y() + bar.get_height() / 2,
                        f"{value:.4f} ms", va="center", fontsize=9)
    speed_axis.invert_yaxis()

    graph_axis = fig.add_subplot(grid[2, 0])
    edge_values = [graph["intra_modal_edge_count"], graph["cross_modal_edge_count"]]
    edge_labels = ["Cạnh nội mô thức", "Cạnh liên mô thức"]
    edge_colors = ["#4A9D8F", "#315A7D"]
    edge_bars = graph_axis.bar(edge_labels, edge_values, color=edge_colors, width=0.55)
    graph_axis.set_title("Cấu trúc FKG-MM đầu vào")
    graph_axis.set_ylabel("Số cạnh")
    graph_axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(edge_bars, edge_values):
        graph_axis.text(bar.get_x() + bar.get_width() / 2, value + 45,
                        f"{value:,}".replace(",", "."), ha="center", fontweight="bold")
    graph_axis.text(
        0.02, 0.95,
        f"{graph['rule_count']:,} luật  •  {graph['token_count']} token  •  "
        f"patient overlap = {data['patient_overlap_count']}",
        transform=graph_axis.transAxes, va="top", color="#425466",
    )

    conclusion_axis = fig.add_subplot(grid[2, 1])
    conclusion_axis.axis("off")
    full = payload["methods"]["FKG-E full"]
    fisa = payload["methods"]["FISA lookup"]
    unsupervised = payload["methods"]["FKG-E unsupervised"]
    speedup = payload["methods"]["FISA sequential"]["avg_time_per_query_ms"] / max(
        fisa["avg_time_per_query_ms"], 1e-12)
    conclusion_axis.text(0.0, 0.96, "Kết luận của lượt kiểm chứng", fontsize=15,
                         fontweight="bold", color="#1F2933", va="top")
    lines = [
        ("FKG-E đầy đủ", f"AUC-ROC {full['auc_roc']:.4f}; AUC-PR {full['auc_pr']:.4f}"),
        ("So với FISA", f"ΔAUC = {full['auc_roc'] - fisa['auc_roc']:+.4f}"),
        ("FISA bảng tra", f"nhanh hơn tuần tự khoảng {speedup:.1f} lần"),
        ("Không nhãn", f"BalAcc {unsupervised['balanced_accuracy']:.4f}; sụp về lớp đa số"),
        ("Giới hạn", "cần 5 seed × 5 fold, nested validation và root test"),
    ]
    y = 0.76
    for heading, detail in lines:
        conclusion_axis.text(0.0, y, heading, fontsize=11, fontweight="bold",
                             color="#315A7D", va="top")
        conclusion_axis.text(0.32, y, detail, fontsize=11, color="#333333", va="top")
        y -= 0.15

    fig.subplots_adjust(left=0.065, right=0.97, top=0.95, bottom=0.09)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output_path)
    return output_path


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULT_DIR
    main(directory)
