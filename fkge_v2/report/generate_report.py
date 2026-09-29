import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

OUT = C.PATHS.OUTPUT_DIR
FIG_DIR = os.path.join(OUT, "figures")


def _load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_csv(name, rows, fields):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _table(rows, fields, headers=None):
    headers = headers or fields
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = (f"{value:.2e}" if 0 < abs(value) < 0.00005
                         else f"{value:.4f}")
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _flatten_kb1(data):
    rows = []
    for dataset in data:
        for method in dataset["methods"]:
            rows.append({
                "dataset": dataset["dataset"],
                "source": dataset["source"],
                "synthetic": dataset["is_synthetic"],
                "n_rules": dataset["n_rules_mean"],
                **method,
            })
    return rows


def report_kb1():
    data = _load("kb1_results.json")
    if not data:
        return ""
    rows = _flatten_kb1(data)
    fields = ["dataset", "method", "n_rules", "auc_roc_mean",
              "auc_roc_std", "auc_roc_ci95_low", "auc_roc_ci95_high",
              "f1_mean", "f1_std", "accuracy_mean", "balanced_accuracy_mean",
              "balanced_accuracy_std",
              "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high",
              "agreement_mean", "agreement_std", "agreement_ci95_low",
              "agreement_ci95_high", "fidelity_balanced_accuracy_mean",
              "fidelity_balanced_accuracy_std", "cohen_kappa_mean",
              "fidelity_bound_coverage_mean", "mean_kl_divergence_mean",
              "avg_time_per_query_ms_mean",
              "train_time_s_mean", "synthetic"]
    _write_csv("table_KB1.csv", rows, fields)

    datasets = [item["dataset"] for item in data]
    methods = ["FISA sequential", "FISA lookup", "FKG-E unsupervised", "FKG-E full"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(datasets))
    width = 0.18
    for index, method in enumerate(methods):
        selected = [next(row for row in rows
                         if row["dataset"] == dataset and row["method"] == method)
                    for dataset in datasets]
        axes[0].bar(x + (index - 1.5) * width,
                    [row["auc_roc_mean"] for row in selected], width, label=method)
        axes[1].bar(x + (index - 1.5) * width,
                    [row["avg_time_per_query_ms_mean"] for row in selected], width,
                    label=method)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(datasets, rotation=12, ha="right")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("KB1: chất lượng")
    axes[1].set_ylabel("ms/mẫu")
    axes[1].set_yscale("log")
    axes[1].set_title("KB1: thời gian suy diễn")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "kb1_comparison.png"), dpi=150)
    plt.close(fig)
    return (_table(rows, fields)
            + "\n\n`*_std` là độ lệch chuẩn mẫu của 25 lượt fold × seed "
              "(FISA: 5 fold). Khoảng tin cậy 95%: bootstrap theo 5 fold, "
              "lấy trung bình seed "
              "trong từng fold (10.000 lượt lấy mẫu). Objective hiện chỉ gồm "
              "L_SGNS, L_node, L_inf, L_pred và L2; chưa có L_edge, L_A, "
              "L_B, L_rule. FISA và ngưỡng quyết định chưa được hiệu chỉnh "
              "trên tập xác thực lồng theo bệnh nhân.")


def report_kb2():
    data = _load("kb2_results.json")
    if not data:
        return ""
    rows = _flatten_kb1(data["rows"])
    fields = ["dataset", "method", "n_rules", "auc_roc_mean",
              "auc_roc_std", "auc_roc_ci95_low", "auc_roc_ci95_high",
              "balanced_accuracy_mean", "balanced_accuracy_std",
              "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high",
              "agreement_mean", "agreement_std", "agreement_ci95_low",
              "agreement_ci95_high", "fidelity_balanced_accuracy_mean",
              "fidelity_balanced_accuracy_std",
              "avg_time_per_query_ms_mean"]
    _write_csv("table_KB2.csv", rows, fields)
    verdict = data["verdict"]
    return (_table(rows, fields) + "\n\n"
            + "`*_std` là độ lệch chuẩn mẫu của 25 lượt fold × seed "
              "(FISA: 5 fold). "
            + f"Trạng thái đánh giá: `{verdict['interpretation']}`; "
            + f"delta AUC={verdict['delta_auc_sampled_minus_full']:.4f}, "
            + f"delta BalAcc={verdict['delta_balanced_accuracy_sampled_minus_full']:.4f}, "
            + f"tỉ số thời gian={verdict['inference_speed_ratio_full_over_sampled']:.4f}. "
            + "Tập con 30% dùng hạn ngạch tối thiểu 40% luật cho mỗi lớp; "
              "đây là mô phỏng lấy mẫu, không phải FKGS chính thức.")


def report_kb3():
    data = _load("kb3_results.json")
    if not data:
        return ""
    rows = data["rows"]
    fields = ["d", "auc_roc_mean", "auc_roc_std", "auc_pr_mean",
              "auc_pr_std", "n_parameters_mean",
              "train_time_s_mean", "avg_time_per_query_ms_mean",
              "embedding_memory_bytes_mean"]
    _write_csv("table_KB3.csv", rows, fields)
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot([row["d"] for row in rows], [row["auc_roc_mean"] for row in rows], marker="o")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Chiều nhúng d")
    axis.set_ylabel("AUC-ROC validation")
    axis.set_title(f"KB3: d*={data['selected_d']}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "kb3_dim_sensitivity.png"), dpi=150)
    plt.close(fig)
    return (_table(rows, fields)
            + f"\n\nChọn `d*={data['selected_d']}`. `*_std` là SD mẫu "
              "trên 5 fold × 5 seed.")


def report_kb4():
    data = _load("kb4_results.json")
    if not data:
        return ""
    rows = data["rows"]
    fields = ["weight", "multiplier", "effective_value", "auc_roc_mean",
              "auc_roc_std", "agreement_mean", "agreement_std",
              "mean_kl_divergence_mean", "mean_kl_divergence_std"]
    _write_csv("table_KB4.csv", rows, fields)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for weight in data["supported_weights"]:
        subset = [row for row in rows if row["weight"] == weight]
        axis.plot([row["multiplier"] for row in subset],
                  [row["auc_roc_mean"] for row in subset], marker="o", label=weight)
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set_xlabel("Hệ số nhân so với mặc định")
    axis.set_ylabel("AUC-ROC validation")
    axis.set_title("KB4: quét từng trọng số đã triển khai")
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "kb4_weight_sensitivity.png"), dpi=150)
    plt.close(fig)
    missing = ", ".join(data["unimplemented_weights"])
    return (_table(rows, fields)
            + f"\n\nChưa triển khai trong model hiện tại: `{missing}`. "
              "`*_std` là SD mẫu trên 5 fold × 5 seed.")


def report_kb5():
    data = _load("kb5_results.json")
    if not data:
        return ""
    rows = data["rows"]
    fields = ["cooccurrence", "K", "auc_roc_mean", "auc_roc_std",
              "auc_pr_mean", "auc_pr_std", "agreement_mean", "agreement_std"]
    _write_csv("table_KB5.csv", rows, fields)
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    for window in sorted({row["w"] for row in rows}, key=lambda value: (-1 if value is None else value)):
        subset = [row for row in rows if row["w"] == window]
        axis.plot([row["K"] for row in subset],
                  [row["auc_roc_mean"] for row in subset], marker="o",
                  label="toàn luật" if window is None else f"w={window}")
    axis.set_xlabel("Số mẫu âm K")
    axis.set_ylabel("AUC-ROC validation")
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "kb5_wK_sensitivity.png"), dpi=150)
    plt.close(fig)
    return (_table(rows, fields) + "\n\n"
            + f"Biên độ AUC={data['auc_range']:.4f}. Đây là thống kê mô tả; "
            "chưa kết luận H-E5 khi đóng góp của SGNS chưa được xác nhận. "
            "`*_std` là SD mẫu trên 5 fold × 5 seed.")


def report_kb6():
    data = _load("kb6_results.json")
    if not data:
        return ""
    rows = data["rows"]
    fields = ["ratio", "n_rules_mean", "fisa_sequential_avg_time_per_query_ms_mean",
              "fisa_sequential_avg_time_per_query_ms_std",
              "fisa_lookup_avg_time_per_query_ms_mean",
              "fisa_lookup_avg_time_per_query_ms_std",
              "fkge_avg_time_per_query_ms_mean",
              "fkge_avg_time_per_query_ms_std"]
    _write_csv("table_KB6.csv", rows, fields)
    fig, axis = plt.subplots(figsize=(6.7, 4.4))
    for label, field in {
        "FISA tuần tự": "fisa_sequential_avg_time_per_query_ms_mean",
        "FISA bảng tra": "fisa_lookup_avg_time_per_query_ms_mean",
        "FKG-E": "fkge_avg_time_per_query_ms_mean",
    }.items():
        axis.plot([row["n_rules_mean"] for row in rows],
                  [row[field] for row in rows], marker="o", label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Số luật |R|")
    axis.set_ylabel("ms/mẫu")
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "kb6_scalability.png"), dpi=150)
    plt.close(fig)
    slopes = data["log_log_slopes"]
    return (_table(rows, fields) + "\n\nHệ số góc log-log: "
            + ", ".join(f"{key}={value:.4f}" for key, value in slopes.items())
            + ". `*_std` là SD mẫu trên 5 fold (FISA) hoặc 25 lượt "
              "fold × seed (FKG-E).")


def report_ablation():
    data = _load("ablation_results.json")
    if not data:
        return ""
    rows = data["rows"]
    fields = ["variant", "auc_roc_mean", "auc_roc_std", "f1_mean",
              "f1_std", "balanced_accuracy_mean", "balanced_accuracy_std",
              "agreement_mean", "agreement_std", "mean_kl_divergence_mean",
              "mean_kl_divergence_std", "delta_auc_vs_full"]
    _write_csv("table_ablation.csv", rows, fields)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(rows))
    axis.bar(x, [row["auc_roc_mean"] for row in rows])
    axis.set_xticks(x)
    axis.set_xticklabels([row["variant"] for row in rows], rotation=25, ha="right")
    axis.set_ylabel("AUC-ROC validation")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation_comparison.png"), dpi=150)
    plt.close(fig)
    missing = ", ".join(data["unimplemented_components"])
    return (_table(rows, fields)
            + f"\n\nChưa triển khai: `{missing}`. `*_std` là SD mẫu "
              "trên 5 fold × 5 seed.")


def report_baseline():
    data = _load("baseline_comparison.json")
    if not data:
        return ""
    rows = data["rows"]
    fields = ["method", "status", "official_baseline", "auc_roc_mean",
              "auc_roc_std", "auc_roc_ci95_low", "auc_roc_ci95_high",
              "f1_mean", "f1_std", "balanced_accuracy_mean",
              "balanced_accuracy_std", "balanced_accuracy_ci95_low",
              "balanced_accuracy_ci95_high", "accuracy_mean", "agreement_mean",
              "fidelity_balanced_accuracy_mean", "cohen_kappa_mean", "train_time_s_mean",
              "avg_time_per_query_ms_mean", "n_parameters_mean"]
    _write_csv("table_baseline.csv", rows, fields)
    completed = [row for row in rows if row["status"] == "completed"]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(completed))
    axis.bar(x, [row["auc_roc_mean"] for row in completed])
    axis.set_xticks(x)
    axis.set_xticklabels([row["method"] for row in completed], rotation=25, ha="right")
    axis.set_ylabel("AUC-ROC validation")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "baseline_comparison.png"), dpi=150)
    plt.close(fig)
    return (_table(rows, fields) + "\n\n`*_std` là độ lệch chuẩn mẫu của "
            "25 lượt fold × seed (FISA: 5 fold). Khoảng tin cậy 95% lấy mẫu lại theo "
            "5 fold (trung bình seed trong từng fold). Các baseline hậu tố "
            "-lite chỉ dùng kiểm tra luồng, không là đối chứng chuẩn.")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    manifest = _load("run_manifest.json") or {}
    sections = [
        "# Báo cáo thực nghiệm FKG-E",
        f"Run ID: `{manifest.get('run_id', 'unknown')}`",
        f"Trạng thái: `{manifest.get('status', 'unknown')}`",
        f"Chế độ quick: `{manifest.get('quick', 'unknown')}`",
    ]
    builders = [
        ("KB1", report_kb1), ("KB2", report_kb2), ("KB3", report_kb3),
        ("KB4", report_kb4), ("KB5", report_kb5), ("KB6", report_kb6),
        ("Ablation", report_ablation), ("Baseline", report_baseline),
    ]
    for title, builder in builders:
        body = builder()
        if body:
            sections.append(f"## {title}\n\n{body}")
    path = os.path.join(OUT, "report_tong_hop.md")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n\n".join(sections) + "\n")
    print(f">>> Báo cáo: {path}")
    return path


if __name__ == "__main__":
    main()
