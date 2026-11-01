"""Rebuild the current FKG-E numerical report and explain patient counts.

No training is performed. Fold-level seed means are the descriptive units;
within-fold seed SDs are not pooled as independent observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from render import Archive, stats

METRICS = [
    ("auc", "ROC-AUC"), ("pr_auc", "PR-AUC (AP)"),
    ("balanced_accuracy", "Balanced Accuracy"), ("sensitivity", "Độ nhạy"),
    ("specificity", "Độ đặc hiệu"), ("f1_pos", "F1 lớp dương"),
    ("accuracy", "Accuracy"),
]
OLD = [
    ("KB1", "kb1_results.json", ["dataset"]),
    ("KB2", "kb2_results.json", ["kb"]),
    ("KB3", "kb3_results.json", ["d"]),
    ("KB4", "kb4_results.json", ["lambda", "beta"]),
    ("KB5", "kb5_results.json", ["w", "K"]),
    ("KB6", "kb6_results.json", ["ratio"]),
    ("ablation", "ablation_results.json", ["variant"]),
    ("baseline", "baseline_comparison.json", ["method"]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def ms(values, digits=4):
    result = stats(values)
    return f"{result['mean']:.{digits}f} ± {result['sd']:.{digits}f}"


def table(columns, rows):
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *["| " + " | ".join(number(v).replace("|", "\\|") for v in row) + " |" for row in rows],
    ])


def record_text(archive, suffix):
    matches = [name for name in archive.zip.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one source document: {suffix}")
    data = archive.zip.read(matches[0])
    archive.hashes[matches[0]] = hashlib.sha256(data).hexdigest()


def collect_old(archive):
    sections, raw = {}, []
    for name, filename, keys in OLD:
        groups = defaultdict(list)
        for fold in range(1, 6):
            obj = archive.read(f"/fold_{fold:02}/{filename}")
            rows = obj["rows"] if isinstance(obj, dict) else obj
            seen = set()
            for row in rows:
                key = tuple(row[key] for key in keys)
                if key in seen:
                    raise ValueError(f"Duplicate {name} configuration in fold {fold}")
                seen.add(key)
                groups[key].append(row)
                raw.append({"section": name, "fold": f"fold_{fold:02}", **row})
        aggregated = []
        for key, rows in groups.items():
            if len(rows) != 5:
                raise ValueError(f"Incomplete {name} configuration: {key}")
            metric_keys = [key for key in rows[0] if key not in keys and not key.endswith("_std")
                           and all(isinstance(row.get(key), (int, float))
                                   and not isinstance(row[key], bool) and math.isfinite(row[key]) for row in rows)]
            aggregated.append({"configuration": dict(zip(keys, key)),
                               "metrics": {key: stats([row[key] for row in rows]) for key in metric_keys}})
        sections[name] = aggregated
    # These documents define the reporting layout; values above come from raw folds.
    record_text(archive, "/tong_hop_5fold/report_tong_hop_5fold.md")
    return sections, raw


def old_csv_rows(section):
    rows = []
    for item in section:
        row = {**item["configuration"], "n_folds": 5}
        for name, values in item["metrics"].items():
            row[name + "_fold_mean"] = values["mean"]
            row[name + "_fold_sd"] = values["sd"]
        rows.append(row)
    return rows


def collect_new(archive, replay_directory):
    archived = []
    for fold in range(1, 6):
        name = f"fold_{fold:02}"
        obj = archive.read(f"/expected/fkge_tuned/{name}/tuned_results.json")
        if obj["criterion"] != "val_pred_loss_mean" or len(obj["rows"]) != 108:
            raise ValueError("Unexpected tuned selection protocol")
        if obj["selected"] != min(obj["rows"], key=lambda row: row[obj["criterion"]]):
            raise ValueError("Selected configuration is not the validation argmin")
        archived.append({"fold": name, **obj})
    record_text(archive, "/fkge_branch/report/report_nested.py")
    record_text(archive, "/expected/fkge_tuned/report_nested.md")
    replay_path = replay_directory / "replay_results.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if replay["manifest"]["status"] != "completed":
        raise ValueError("The selected replay is not complete")
    results = sorted(replay["folds"], key=lambda row: row["fold"])
    if [r["fold"] for r in results] != [r["fold"] for r in archived]:
        raise ValueError("Expected five distinct, matching folds")
    differences = []
    for actual, expected in zip(results, archived):
        for metric, _ in METRICS:
            for model, reference in [("fisa", expected["fisa"][metric]),
                                     ("fkge", expected["selected"][metric + "_mean"])]:
                differences.append({"fold": actual["fold"], "model": model,
                                    "metric": metric, "replayed": actual[model][metric],
                                    "expected": reference, "delta": actual[model][metric] - reference})
    return archived, results, differences, replay["manifest"]


def summarize_new(archived, replay):
    main, inference = [], []
    n_train = float(np.mean([row["n_train"] for row in archived]))
    n_test = float(np.mean([row["n_test"] for row in archived]))
    for key, label in METRICS:
        a = np.array([fold["fisa"][key] for fold in replay])
        b = np.array([fold["fkge"][key] for fold in replay])
        d = b - a
        main.append({"metric": key, "label": label, "n_folds": 5,
                     "fisa_mean": float(a.mean()), "fisa_sd": float(a.std(ddof=1)),
                     "fkge_mean": float(b.mean()), "fkge_sd": float(b.std(ddof=1)),
                     "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
                     "fkge_better_folds": int((d > 0).sum())})
        half_width = scipy_stats.t.ppf(0.975, 4) * b.std(ddof=1) / np.sqrt(5)
        t_value = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 5 + n_test / n_train))
        inference.append({"metric": key, "fkge_ci95_low_original_formula": float(b.mean() - half_width),
                          "fkge_ci95_high_original_formula": float(b.mean() + half_width),
                          "p_nb_original_formula": float(2 * scipy_stats.t.sf(abs(t_value), 4)),
                          "n_test_over_n_train_after_smote": n_test / n_train,
                          "reference_only": True})
    selected, split, auxiliary = [], [], []
    for actual, expected in zip(replay, archived):
        m, f, info = actual["fkge"], actual["fisa"], actual["info"]
        selected.append({"fold": actual["fold"], **actual["selected_parameters"],
                         "val_pred_loss": m["val_pred_loss"], "val_auc": m["val_auc"],
                         "val_pr_auc": m["val_pr_auc"], "test_auc": m["auc"], "test_ap": m["pr_auc"],
                         "fisa_test_auc": f["auc"], "fisa_test_ap": f["pr_auc"], "epochs": m["epochs"]})
        split.append({"fold": actual["fold"], **{key: value for key, value in info.items() if key != "fold"},
                      "n_train_before_smote": info["n_train"] - info["n_synthetic"]})
        auxiliary.append({"fold": actual["fold"], "n_rules": m["n_rules"],
                          "agreement_fisa": m["agreement_fisa"], "dev_rule": m["dev_rule"],
                          "fisa_threshold": f["threshold"], "fkge_threshold": m["threshold"],
                          "teacher_T": m["teacher_T"], "epochs": m["epochs"], "stop": m["stop"],
                          "final_grad_norm": m["final_grad_norm"], "n_proj": m["n_proj"], "n_ls_fail": m["n_ls_fail"],
                          "fisa_table_ms": f["ms_per_query_table"], "fisa_sequential_ms": f["ms_per_query_sequential"],
                          "fkge_ms": m["ms_per_query"], "train_time_s_replay": m["train_time_s"],
                          "train_time_s_archived": expected["selected"]["train_time_s_mean"],
                          "whole_fold_replay_s": actual["seconds"]})
    return main, inference, selected, split, auxiliary


def draw_table(ax, title, columns, rows, widths=None, fontsize=10):
    ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=14)
    item = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colWidths=widths,
                    bbox=[0, 0, 1, 0.95])
    item.auto_set_font_size(False)
    item.set_fontsize(fontsize)
    for (row, _), cell in item.get_celld().items():
        cell.set_edgecolor("#d1d8e0")
        cell.set_facecolor("#e6eef8" if row == 0 else ("#f7f9fc" if row % 2 else "white"))
        if row == 0:
            cell.set_text_props(fontweight="bold")


def new_figure(main, selected, output):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [8, 6]})
    rows = [[r["label"], f"{r['fisa_mean']:.4f} ± {r['fisa_sd']:.4f}",
             f"{r['fkge_mean']:.4f} ± {r['fkge_sd']:.4f}", f"{r['delta_mean']:+.4f}",
             f"{r['fkge_better_folds']}/5"] for r in main]
    draw_table(axes[0], "Kết quả chính — trung bình ± SD qua 5 fold (ddof=1)",
               ["Độ đo", "FISA", "FKG-E tuned", "Δ (E − FISA)", "Fold E hơn"], rows,
               [0.24, 0.25, 0.25, 0.15, 0.11], 11)
    rows = [[r["fold"], f"{r['alpha']:g}", r["d_e"], f"{r['lam_I']:g}", f"{r['lam_P']:g}",
             f"{r['val_pred_loss']:.4f}", f"{r['test_auc']:.4f}", f"{r['test_ap']:.4f}", r["epochs"]]
            for r in selected]
    draw_table(axes[1], "Cấu hình FKG-E theo từng fold",
               ["Fold", "α", "d", "λ_I", "λ_P", "Loss val", "ROC-AUC", "AP", "Vòng"], rows,
               [0.14, 0.07, 0.07, 0.07, 0.07, 0.16, 0.14, 0.14, 0.14])
    fig.suptitle("Kết quả FKG-E trên BRSET", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.025, "Chạy lại 5 cấu hình đã chọn; seed 42; subset 1.208 ảnh / 729 bệnh nhân.\n"
             "Chưa chạy lại lưới chọn tham số; validation theo gói nguồn còn hạn chế.", ha="center", fontsize=10, color="#59636f")
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.92), h_pad=3)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def old_cell(row, key, digits=4):
    v = row["metrics"].get(key)
    return f"{v['mean']:.{digits}f} ± {v['sd']:.{digits}f}" if v else "Không có"


def old_figure(old, output):
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), gridspec_kw={"height_ratios": [2, 3, 5, 5, 3, 5]})
    rows = []
    for label, prefix in [("FISA", "fisa"), ("FKG-E", "fkge")]:
        row = old["KB1"][0]
        fields = ["fisa_auc", "fisa_bacc", "fisa_f1", "fisa_avg_ms"] if prefix == "fisa" else \
                 ["fkge_auc_mean", "fkge_bacc_mean", "fkge_f1_mean", "fkge_avg_ms_mean"]
        rows.append([label] + [old_cell(row, key) for key in fields])
    draw_table(axes[0], "KB1 — FISA và FKG-E trên cơ sở luật đầy đủ",
               ["Mô hình", "ROC-AUC", "BalAcc", "F1 macro", "ms / truy vấn"], rows)
    rows = [[r["configuration"]["kb"], old_cell(r, "n_rules", 1), old_cell(r, "fkge_auc_mean"),
             old_cell(r, "fkge_bacc_mean"), old_cell(r, "fkge_avg_ms_mean")] for r in old["KB2"]]
    draw_table(axes[1], "KB2 — Cơ sở luật đầy đủ và rút gọn",
               ["Tập luật", "Số luật", "FKG-E ROC-AUC", "FKG-E BalAcc", "FKG-E ms"], rows)
    rows = [[r["configuration"]["d"]] + [old_cell(r, key) for key in
             ["accuracy_mean", "avg_ms_mean", "train_time_s_mean"]] for r in old["KB3"]]
    draw_table(axes[2], "KB3 — Chiều nhúng d", ["d", "Accuracy", "ms / truy vấn", "Huấn luyện (s)"], rows)
    values = sorted({r["configuration"]["beta"] for r in old["KB4"]})
    rows = []
    for lam in sorted({r["configuration"]["lambda"] for r in old["KB4"]}):
        rows.append([f"{lam:g}"] + [old_cell(next(r for r in old["KB4"] if r["configuration"] == {"lambda": lam, "beta": beta}),
                                                   "accuracy_mean") for beta in values])
    draw_table(axes[3], "KB4 — Accuracy theo cặp (λ_I, λ_P)", ["λ_I / λ_P"] + [f"{v:g}" for v in values], rows, fontsize=9)
    rows = [[r["configuration"]["w"], r["configuration"]["K"], old_cell(r, "accuracy_mean")] for r in old["KB5"]]
    draw_table(axes[4], "KB5 — Số mẫu âm K", ["Đồng xuất hiện", "K", "Accuracy"], rows)
    rows = [[f"{r['configuration']['ratio']:.0%}", old_cell(r, "n_rules", 1)] + [old_cell(r, key) for key in
             ["fisa_seq_avg_ms", "fisa_avg_ms", "fkge_avg_ms_mean"]] for r in old["KB6"]]
    draw_table(axes[5], "KB6 — Thời gian suy diễn theo số luật",
               ["Tỉ lệ luật", "Số luật", "FISA tuần tự ms", "FISA bảng tra ms", "FKG-E ms"], rows)
    fig.suptitle("FKG-E trên BRSET — bảng KB1–KB6 từ zip cũ", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, "Số liệu đã lưu; mỗi ô = trung bình ± SD của 5 giá trị fold (đã gộp seed trong fold).\n"
             "Giao thức cũ chia 60/40 theo ảnh và có trùng bệnh nhân; không dùng làm kết quả zip mới.",
             ha="center", fontsize=10, color="#59636f")
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.96), h_pad=2.5)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def load_population(path, split):
    population = json.loads(path.read_text(encoding="utf-8"))
    train = population["root_train"]
    root_test = population["root_test"]
    exported = population["image_export"]
    for key in ("images", "patients"):
        if train[key] + root_test[key] != exported[key]:
            raise ValueError(f"Root population does not add up: {key}")
    evidence = {f"fold_{row['fold']:02}": row for row in population["folds"]}
    for row in split:
        source = evidence[row["fold"]]
        actual = (row["n_train_before_smote"], row["n_test"], row["n_patients"])
        if actual != (source["train_images"], source["test_images"], source["train_patients"]):
            raise ValueError(f"Population evidence differs from replay: {row['fold']}")
        if row["n_train_before_smote"] + row["n_test"] != train["images"]:
            raise ValueError("Fold image counts differ from the input population")
        if row["n_patients"] + source["test_patients"] != train["patients"]:
            raise ValueError("Fold patient counts differ from the input population")
        row["n_train_patients"] = row["n_patients"]
        row["n_test_patients"] = source["test_patients"]
    return population


def population_rows(population):
    return [{"stage": label, "images": population[key]["images"], "patients": population[key]["patients"]}
            for key, label in [("full_labels", "Bảng nhãn BRSET đầy đủ"),
                               ("image_export", "Ảnh khớp nhãn trong lần xuất dữ liệu"),
                               ("root_test", "Tập test giữ riêng từ đầu"),
                               ("root_train", "Phần dữ liệu được chia 5 fold")]]


def patient_figure(population, split, output):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [5, 6]})
    draw_table(axes[0], "Phạm vi đầu vào của thực nghiệm",
               ["Phạm vi", "Số ảnh", "Số bệnh nhân"],
               [[r["stage"], f"{r['images']:,}", f"{r['patients']:,}"] for r in population_rows(population)],
               [0.60, 0.20, 0.20], 11)
    draw_table(axes[1], "5 fold trên 1.208 ảnh của 729 bệnh nhân",
               ["Fold", "Ảnh train gốc", "Bệnh nhân train", "Ảnh giữ lại", "Bệnh nhân giữ lại"],
               [[r["fold"], r["n_train_before_smote"], r["n_train_patients"], r["n_test"], r["n_test_patients"]]
                for r in split], [0.14, 0.20, 0.23, 0.20, 0.23], 11)
    fig.suptitle("Vì sao mỗi fold có 583–584 bệnh nhân train?", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.025, "Fold 1: 583 train + 146 giữ lại = 729 bệnh nhân.\n"
             "966 ảnh train + 812 mẫu SMOTE = 1.778 dòng/luật; số bệnh nhân train vẫn là 583.",
             ha="center", fontsize=10, color="#59636f")
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.92), h_pad=3)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def report(main, inference, selected, split, auxiliary, population, differences, output):
    lines = ["# Kết quả FKG-E trên BRSET", "",
             "Báo cáo sử dụng bản FKG-E hiện tại. Kết quả được tái lập từ 5 cấu hình đã chọn; "
             "không chạy lại quá trình chọn 108 cấu hình/fold. Seed 42, tối đa 200 vòng, early stopping theo nguồn.", "",
             "## 1. Vì sao mỗi fold chỉ có 583–584 bệnh nhân train?", "",
             "**Cột bệnh nhân là số bệnh nhân trong tập train của từng fold. Toàn bộ phần dữ liệu được chia fold "
             "ở thực nghiệm này gồm 1.208 ảnh của 729 bệnh nhân.**", "",
             table(["Phạm vi", "Số ảnh", "Số bệnh nhân"],
                   [[r["stage"], r["images"], r["patients"]] for r in population_rows(population)]), "",
             "Code xuất dữ liệu quét các file ảnh trong thư mục đầu vào rồi ghép với bảng nhãn. "
             "Ở lần xuất này, thư mục có 1.529 ảnh khớp nhãn của 897 bệnh nhân. "
             "Mặc dù bảng nhãn chứa 16.266 ảnh của 8.524 bệnh nhân, số ảnh đầu vào thực sự chỉ là một subset. "
             "Sự thu hẹp phạm vi đã xảy ra ở bước lấy ảnh đầu vào, trước khi chia 5 fold.", "",
             "Trong 897 bệnh nhân đó, 168 bệnh nhân với 321 ảnh được giữ riêng làm root test. "
             "Phần còn lại có **729 bệnh nhân với 1.208 ảnh** và mới được chia thành 5 fold. "
             "Tập root test giữ riêng không được đánh giá trong bảng kết quả 5 fold này.", "",
             table(["Fold", "Ảnh train trước SMOTE", "Bệnh nhân train", "Ảnh giữ lại", "Bệnh nhân giữ lại"],
                   [[r["fold"], r["n_train_before_smote"], r["n_train_patients"], r["n_test"], r["n_test_patients"]]
                    for r in split]), "",
             "Ví dụ fold 1: **583 + 146 = 729 bệnh nhân**, tương ứng **966 + 242 = 1.208 ảnh**. "
             "Mỗi lượt dùng bốn phần để train, nên số bệnh nhân train xấp xỉ `729 × 4/5 = 583,2`; "
             "bốn fold có 583 bệnh nhân train và một fold có 584. Một bệnh nhân có thể có nhiều ảnh, "
             "nên số ảnh không bằng số bệnh nhân. Không cộng số bệnh nhân train của năm lượt: "
             "mỗi bệnh nhân được dùng để train ở bốn lượt.", "",
             "Sau SMOTE, fold 1 có **966 + 812 = 1.778 dòng huấn luyện/luật**. "
             "812 dòng bổ sung là mẫu tổng hợp; số bệnh nhân train thực vẫn là **583**.", "",
             "Nếu chia 5 fold trực tiếp trên toàn bộ 8.524 bệnh nhân thì mỗi lượt train sẽ khoảng 6.800 bệnh nhân. "
             "Kết quả hiện tại cần được ghi là thực nghiệm trên subset BRSET. "
             "Để báo cáo trên toàn bộ BRSET, cần đưa đủ ảnh vào pipeline, tạo lại manifest/fold, "
             "đặc trưng, mờ hóa, cân bằng, luật và chạy lại mô hình. "
             "Nạp lại các file fold/đặc trưng đã xuất sẵn trên Colab vẫn giữ nguyên quy mô subset.", "",
             "![Số bệnh nhân theo fold](new_zip/patient_counts.png)", "",
             "[CSV phạm vi dữ liệu](new_zip/table_population.csv) · [CSV chi tiết từng fold](new_zip/table_split.csv) · "
             "[Bằng chứng số lượng](../../population.json)", "",
             "## 2. Kết quả chính FISA và FKG-E tuned", "",
             "Mỗi ô là trung bình ± SD mẫu qua **5 fold** (`ddof=1`), với một seed. "
             "Không coi 50 lần đo thời gian là 50 mẫu độc lập. Các fold dùng chung một phần dữ liệu huấn luyện, "
             "nên SD này chỉ mô tả độ biến thiên, không khẳng định ý nghĩa thống kê.", "",
             table(["Độ đo", "FISA", "FKG-E", "Δ (FKG-E − FISA)", "Fold FKG-E hơn"],
                   [[r["label"], f"{r['fisa_mean']:.4f} ± {r['fisa_sd']:.4f}",
                     f"{r['fkge_mean']:.4f} ± {r['fkge_sd']:.4f}", f"{r['delta_mean']:+.4f}",
                     f"{r['fkge_better_folds']}/5"] for r in main]), "",
             "PR-AUC là Average Precision theo code nguồn. F1 ở bảng này là **F1 lớp dương**. "
             "Các chỉ số dùng ngưỡng của mô hình nguồn; AUC/AP dùng điểm liên tục. "
             "AUC/AP tốt hơn không đồng nghĩa mọi chỉ số phân loại tốt hơn.", "",
             "![Bảng kết quả FKG-E](new_zip/result_summary.png)", "",
             "[CSV kết quả chính](new_zip/table_main.csv) · [Kiểm tra tái lập 70 chỉ số](new_zip/table_replay_checks.csv)", "",
             f"Chênh lệch lớn nhất khi tái lập 7 chỉ số × 2 mô hình × 5 fold so với kết quả nguồn: "
             f"`{max(abs(r['delta']) for r in differences):.3g}`. "
             "Các con số phản ánh subset **1.208 ảnh / 729 bệnh nhân**. Inner validation vẫn dùng dữ liệu đã tiền xử lý/SMOTE "
             "và đồ thị nguồn; bảng này không chứng nhận đã khắc phục yêu cầu validation của phản biện.", "",
             "## 3. Cấu hình và từng fold", "",
             table(["Fold", "α", "d", "λ_I", "λ_P", "Loss val", "AUC val", "AP val", "AUC test", "AP test", "Vòng"],
                   [[r[k] for k in ["fold", "alpha", "d_e", "lam_I", "lam_P", "val_pred_loss", "val_auc", "val_pr_auc",
                                    "test_auc", "test_ap", "epochs"]] for r in selected]), "",
             "[CSV cấu hình](new_zip/table_selected_folds.csv) · [Chỉ số từng fold và mô hình](new_zip/table_fold_metrics.csv) · "
             "[Toàn bộ lưới đã lưu](new_zip/table_archived_grid.csv)", "",
             "## 4. Chia dữ liệu và thông số chạy", "",
             table(["Fold", "Train trước SMOTE", "Train sau SMOTE / luật", "Bệnh nhân train", "SMOTE", "Học / val", "Test (dương)"],
                   [[r["fold"], r["n_train_before_smote"], r["n_train"], r["n_patients"], r["n_synthetic"],
                     f"{r['n_fkge_train']} / {r['n_fkge_val']}", f"{r['n_test']} ({r['test_pos']})"] for r in split]), "",
             table(["Chỉ số phụ của 5 cấu hình đã chọn", "Trung bình ± SD"],
                   [[label, ms([r[key] for r in auxiliary])] for key, label in [
                       ("agreement_fisa", "Đồng thuận nhãn FISA"), ("dev_rule", "Dev căn cứ luật"),
                       ("fisa_table_ms", "FISA bảng tra (ms/truy vấn, replay)"),
                       ("fisa_sequential_ms", "FISA tuần tự (ms/truy vấn, replay)"),
                       ("fkge_ms", "FKG-E (ms/truy vấn, replay)"),
                       ("train_time_s_replay", "Huấn luyện FKG-E (s, replay)"),
                       ("train_time_s_archived", "Huấn luyện FKG-E (s, tác giả lưu)")]]), "",
             "Thời gian phụ thuộc máy và tải khi chạy hai worker; không gộp thời gian tác giả với replay để tính speedup. "
             "Số luật là số dòng sau SMOTE, không phải số bệnh nhân. "
             "[CSV chia dữ liệu](new_zip/table_split.csv) · [Ngưỡng, T, gradient, dừng và thời gian](new_zip/table_auxiliary.csv)", "",
             "## 5. CI/p-value theo công thức trong code nguồn (tham khảo)", "",
             "Mục này chỉ tái hiện công thức trong `fkge_branch/report/report_nested.py`: CI t có 4 bậc tự do; "
             "p NB dùng tỉ lệ test/train **sau SMOTE**. Tỉ lệ đó khác số ảnh train gốc "
             "(test/train trước SMOTE = 0,25). CI có thể vượt [0;1]. Không dùng các cột này để khẳng định ý nghĩa thống kê "
             "hay xem 5 fold là các thí nghiệm độc lập.", "",
             table(["Độ đo", "CI95 FKG-E theo code gốc", "p NB theo code gốc"],
                   [[r["metric"], f"[{r['fkge_ci95_low_original_formula']:.4f}; {r['fkge_ci95_high_original_formula']:.4f}]",
                     f"{r['p_nb_original_formula']:.3f}"] for r in inference]), "",
             "[CSV công thức nguồn](new_zip/table_reference_inference.csv)", "",
             "## 6. Phạm vi kết quả", "",
             "Báo cáo này trình bày kết quả tuned của năm cấu hình đã chọn trên validation. "
             "Các thí nghiệm KB2–KB6, ablation và baseline khác cần được chạy với pipeline hiện tại "
             "trước khi có thể báo cáo như kết quả của phiên bản này. "
             "Không chọn cấu hình theo chỉ số test cao nhất. "
             "Xem `statistics_manifest.json` để đối chiếu SHA256 nguồn, script và các file được xuất.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-zip", type=Path, help="Optional archived scenario export, outside the main report.")
    parser.add_argument("--package-zip", "--new-zip", dest="package_zip", type=Path, required=True)
    parser.add_argument("--population", type=Path, default=Path(__file__).with_name("population.json"))
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    new_archive = Archive(args.package_zip)
    archives = [new_archive]
    archived, replay, differences, replay_manifest = collect_new(new_archive, args.replay)
    main_rows, inference, selected, split, auxiliary = summarize_new(archived, replay)
    population = load_population(args.population, split)
    new_dir, old_dir = args.output / "new_zip", args.output / "old_zip"
    new_dir.mkdir(exist_ok=True)
    for name, rows in [("main", main_rows), ("reference_inference", inference), ("selected_folds", selected),
                       ("split", split), ("auxiliary", auxiliary), ("replay_checks", differences)]:
        write_csv(new_dir / f"table_{name}.csv", rows)
    write_csv(new_dir / "table_fold_metrics.csv", [
        {"fold": fold["fold"], "model": model, **{key: fold[model][key] for key, _ in METRICS}}
        for fold in replay for model in ["fisa", "fkge"]])
    write_csv(new_dir / "table_archived_grid.csv", [
        {"fold": fold["fold"], **row} for fold in archived for row in fold["rows"]])
    write_csv(new_dir / "table_population.csv", population_rows(population))
    if args.old_zip:
        old_archive = Archive(args.old_zip)
        old, _ = collect_old(old_archive)
        archives.append(old_archive)
        for name, section in old.items():
            write_csv(old_dir / f"table_{name}.csv", old_csv_rows(section))
        old_figure(old, old_dir / "result_summary.png")
    new_figure(main_rows, selected, new_dir / "result_summary.png")
    patient_figure(population, split, new_dir / "patient_counts.png")
    report(main_rows, inference, selected, split, auxiliary, population, differences, args.output)
    metadata = {
        "archives": [archive.provenance() for archive in archives],
        "script_sha256": sha(Path(__file__)), "shared_renderer_sha256": sha(Path(__file__).with_name("render.py")),
        "replay_results_sha256": sha(args.replay / "replay_results.json"),
        "environment": replay_manifest["environment"], "seed": replay_manifest["seed"],
        "population_sha256": sha(args.population),
        "n_descriptive_fold_units": 5,
        "archived_grid_rows": sum(len(f["rows"]) for f in archived),
        "replay_metric_checks": len(differences),
        "maximum_absolute_replay_metric_delta": max(abs(r["delta"]) for r in differences),
        "within_seed_sd_columns_pooled": False,
        "inference_formulas_are_reference_only": True,
        "files_sha256": {p.relative_to(args.output).as_posix(): sha(p)
                         for p in sorted([*new_dir.iterdir(), args.output / "report.md"]) if p.is_file()},
    }
    (args.output / "statistics_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(metadata['files_sha256'])} report/CSV/PNG files to {args.output}")


if __name__ == "__main__":
    main()
