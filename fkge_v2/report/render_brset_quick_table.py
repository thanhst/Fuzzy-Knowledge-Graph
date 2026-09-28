"""Render six FKG-E scenario tables from one BRSET quick run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _method(result):
    return next(row for row in result["methods"] if row["method"] == "FKG-E full")


def _number(value, decimals=4):
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _table(axis, title, columns, rows, widths):
    axis.axis("off")
    axis.text(0, 0.91, title, transform=axis.transAxes,
              fontsize=13, fontweight="bold", va="bottom")
    table = axis.table(cellText=rows, colLabels=columns,
                       cellLoc="center", colLoc="center",
                       colWidths=widths, bbox=[0, 0, 1, 0.84])
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
    kb3 = _load(run_dir / "kb3_results.json")
    kb4 = _load(run_dir / "kb4_results.json")
    kb5 = _load(run_dir / "kb5_results.json")
    kb6 = _load(run_dir / "kb6_results.json")
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
    if not all(result["rows"] for result in (kb3, kb4, kb5, kb6)):
        raise ValueError("Expected nonempty KB3 through KB6 results.")

    full = _method(kb1[0])
    kb1_rows = [["FKG-E đầy đủ", _number(full["auc_roc_mean"]),
                 _number(full["balanced_accuracy_mean"]), _number(full["f1_mean"]),
                 _number(full["avg_time_per_query_ms_mean"])]]
    kb2_rows = []
    for label, result in zip(("FKG đầy đủ", "FKGS lấy mẫu 30%"), kb2["rows"]):
        row = _method(result)
        kb2_rows.append([label, _number(result["n_rules_mean"], 1),
                         _number(row["auc_roc_mean"]),
                         _number(row["balanced_accuracy_mean"]),
                         _number(row["avg_time_per_query_ms_mean"])])
    kb3_rows = [[str(row["d"]), _number(row["auc_roc_mean"]),
                 _number(row["auc_pr_mean"]), _number(row["n_parameters_mean"], 0),
                 _number(row["avg_time_per_query_ms_mean"])]
                for row in kb3["rows"]]
    weights = {name: {} for name in kb4["supported_weights"]}
    for row in kb4["rows"]:
        weights[row["weight"]][row["multiplier"]] = row["auc_roc_mean"]
    kb4_rows = []
    for name, values in weights.items():
        zero, default = values[0.0], values[1.0]
        kb4_rows.append([name.replace("lambda_", "λ"), _number(zero),
                         _number(default), "+" + _number(default - zero)])
    kb5_rows = [[str(row["w"]), str(row["K"]), _number(row["auc_roc_mean"]),
                 _number(row["auc_pr_mean"])] for row in kb5["rows"]]
    kb6_rows = [[f"{row['ratio']:.0%}", _number(row["n_rules_mean"], 0),
                 _number(row["fkge_avg_time_per_query_ms_mean"])]
                for row in kb6["rows"]]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(14, 15.5), facecolor="white")
    fig.text(0.05, 0.97, "Kết quả BRSET fusion — FKG-E (KB1–KB6)",
             fontsize=19, fontweight="bold", va="top")
    fig.text(0.05, 0.94,
             f"Quick, 1 seed × {manifest['configuration']['epochs']} epoch; "
             "KB1–KB2: 5 fold; KB3–KB6: 1 fold; bệnh nhân giao nhau = 0",
             fontsize=10.5, va="top")

    grid = fig.add_gridspec(6, 1, left=0.05, right=0.95, top=0.90,
                            bottom=0.09, hspace=0.55,
                            height_ratios=[2, 3, 3, 6, 5, 3])
    tables = [
        ("KB1 — FKG-E trên FKG đầy đủ",
         ["Cấu hình", "AUC-ROC", "Balanced Accuracy", "F1", "ms/mẫu"],
         kb1_rows, [0.30, 0.16, 0.25, 0.12, 0.17]),
        ("KB2 — Tập luật đầy đủ và lấy mẫu",
         ["Tập luật", "Số luật TB", "AUC-ROC", "Balanced Accuracy", "ms/mẫu"],
         kb2_rows, [0.30, 0.16, 0.16, 0.23, 0.15]),
        ("KB3 — Chiều nhúng d",
         ["d", "AUC-ROC", "AUC-PR", "Số tham số", "ms/mẫu"],
         kb3_rows, [0.15, 0.20, 0.20, 0.25, 0.20]),
        ("KB4 — Ảnh hưởng từng trọng số loss",
         ["Trọng số", "AUC khi λ=0", "AUC mặc định", "Δ AUC"],
         kb4_rows, [0.25, 0.25, 0.25, 0.25]),
        ("KB5 — Cửa sổ ngữ cảnh w và số mẫu âm K",
         ["w", "K", "AUC-ROC", "AUC-PR"],
         kb5_rows, [0.20, 0.20, 0.30, 0.30]),
        ("KB6 — Thời gian suy diễn FKG-E theo số luật",
         ["Tỉ lệ luật", "Số luật TB", "FKG-E ms/mẫu"],
         kb6_rows, [0.33, 0.33, 0.34]),
    ]
    for index, (title, columns, rows, widths) in enumerate(tables):
        _table(fig.add_subplot(grid[index]), title, columns, rows, widths)

    fig.text(0.05, 0.055,
             "FKGS 30% là lấy mẫu luật mô phỏng; các quét KB3–KB6 chỉ dùng lưới quick.",
             fontsize=10.5)
    fig.text(0.05, 0.035,
             "Chỉ kiểm chứng pipeline; chưa phải kết quả luận án 5 seed × 5 fold và outer test.",
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
