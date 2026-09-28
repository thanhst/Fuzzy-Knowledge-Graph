"""Render a plain table image from one BRSET FKG-E quick run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _method(result, name):
    return next(row for row in result["methods"] if row["method"] == name)


def _table(axis, title, columns, rows, widths, height):
    axis.axis("off")
    axis.text(0, 1.04, title, transform=axis.transAxes,
              fontsize=13, fontweight="bold", va="bottom")
    table = axis.table(cellText=rows, colLabels=columns,
                       cellLoc="center", colLoc="center",
                       colWidths=widths, bbox=[0, 0, 1, height])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.7)
        cell.set_facecolor("#E9ECEF" if row == 0 else "white")
        if row == 0:
            cell.set_text_props(fontweight="bold")
        if column == 0:
            cell.set_text_props(ha="left")


def main(run_dir, output_path=None):
    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path else run_dir / "result_summary.png"
    kb1 = _load(run_dir / "kb1_results.json")
    kb2 = _load(run_dir / "kb2_results.json")
    manifest = _load(run_dir / "run_manifest.json")

    if len(kb1) != 1 or kb1[0]["is_synthetic"] or kb1[0]["dataset"] != "BRSET fusion":
        raise ValueError("Expected one real BRSET fusion KB1 dataset.")
    if manifest["status"] != "completed" or not manifest["quick"]:
        raise ValueError("Expected a completed quick run.")
    if kb1[0]["patient_overlap_count"] != 0:
        raise ValueError("Patient overlap must be zero.")
    if (len(kb2["rows"]) != 2
            or [row["kb"] for row in kb2["rows"]] != ["KB2-full", "KB2-sampled"]
            or any(row["is_synthetic"] or row["patient_overlap_count"] != 0
                   for row in kb2["rows"])):
        raise ValueError("Expected real, non-overlapping KB2 full and sampled results.")

    labels = [
        ("FISA tuần tự", "FISA sequential"),
        ("FISA bảng tra", "FISA lookup"),
        ("FKG-E (λP=0)", "FKG-E unsupervised"),
        ("FKG-E đầy đủ", "FKG-E full"),
    ]
    kb1_rows = []
    for label, key in labels:
        row = _method(kb1[0], key)
        kb1_rows.append([
            label, f"{row['auc_roc_mean']:.4f}",
            f"{row['balanced_accuracy_mean']:.4f}",
            f"{row['f1_mean']:.4f}",
            f"{row['avg_time_per_query_ms_mean']:.4f}",
        ])

    kb2_rows = []
    for label, result in zip(("FKG đầy đủ", "FKGS lấy mẫu 30%"), kb2["rows"]):
        row = _method(result, "FKG-E full")
        rule_count = f"{result['n_rules_mean']:,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")
        kb2_rows.append([
            label, rule_count,
            f"{row['auc_roc_mean']:.4f}",
            f"{row['balanced_accuracy_mean']:.4f}",
            f"{row['avg_time_per_query_ms_mean']:.4f}",
        ])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    fig.text(0.05, 0.95, "Kết quả BRSET fusion — FKG-E",
             fontsize=19, fontweight="bold", va="top")
    fig.text(0.05, 0.895,
             f"KB1/KB2 quick: {kb1[0]['k_fold']} fold × {manifest['configuration']['n_seeds']} seed "
             f"× {manifest['configuration']['epochs']} epoch; bệnh nhân giao nhau = 0",
             fontsize=11, va="top")

    ax1 = fig.add_axes([0.05, 0.48, 0.9, 0.31])
    _table(ax1, "KB1 — FKG-E so với FISA trên cùng FKG",
           ["Phương pháp", "AUC-ROC", "Balanced Accuracy", "F1", "ms/mẫu"],
           kb1_rows, [0.28, 0.16, 0.25, 0.13, 0.18], 0.88)

    ax2 = fig.add_axes([0.05, 0.18, 0.9, 0.19])
    _table(ax2, "KB2 — FKG-E với tập luật đầy đủ và lấy mẫu",
           ["Tập luật", "Số luật TB", "AUC-ROC", "Balanced Accuracy", "ms/mẫu"],
           kb2_rows, [0.28, 0.16, 0.16, 0.25, 0.15], 0.82)

    fig.text(0.05, 0.10,
             "FKGS 30% là phép lấy mẫu luật mô phỏng. Các KB3–KB6, ablation và baseline "
             "nằm trong báo cáo cùng lượt chạy.", fontsize=10.5)
    fig.text(0.05, 0.06,
             "Chỉ dùng kiểm chứng pipeline; chưa phải kết quả luận án 5 seed × 5 fold.",
             fontsize=10.5, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.run_dir, args.output)
