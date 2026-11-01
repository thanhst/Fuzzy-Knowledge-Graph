"""Export traceable FKG-E figures from the two supplied zip archives and a replay.

Only aggregate metrics and images are written publicly. Patient scores remain
in the private replay directory. Archived and newly replayed results are labeled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

BLUE, ORANGE, GREY = "#2367ad", "#da6b35", "#79838e"
OLD_FOOTER = "ZIP cũ — số liệu đã lưu; trung bình ± SD qua 5 fold.\nKB1–KB6 có phép chia 60/40 theo ảnh; kết quả lịch sử để tham khảo."
NEW_FOOTER = "ZIP mới — số liệu expected/; trung bình ± SD qua 5 fold, 1 seed.\nValidation trong gói nguồn còn hạn chế; subset 1.208 ảnh."
REPLAY_FOOTER = "ZIP mới — chạy lại 5 cấu hình đã chọn, seed 42; không chạy lại lưới.\nValidation theo gói nguồn còn hạn chế; subset 1.208 ảnh, không phải toàn BRSET."


class Archive:
    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(path)
        self.hashes = {}

    def read(self, suffix):
        names = [name for name in self.zip.namelist() if name.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"Expected one {suffix} in {self.path.name}, found {len(names)}")
        payload = self.zip.read(names[0])
        self.hashes[names[0]] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload.decode("utf-8-sig"))

    def provenance(self):
        return {"archive": self.path.name, "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "read_members_sha256": self.hashes}


def stats(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("Expected five finite fold-level metrics.")
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "per_fold": values.tolist()}


def grouped(archive, filename, keys, metrics):
    groups = defaultdict(list)
    for fold in range(1, 6):
        obj = archive.read(f"/ket_qua_geomean/fkge/fold_{fold:02}/{filename}")
        rows = obj["rows"] if isinstance(obj, dict) else obj
        seen = set()
        for row in rows:
            key = tuple(row[name] for name in keys)
            if key in seen:
                raise ValueError(f"Duplicate configuration in {filename}, fold {fold}")
            seen.add(key)
            groups[key].append(row)
    return [{**dict(zip(keys, key)), **{metric: stats([row[metric] for row in rows]) for metric in metrics}}
            for key, rows in groups.items()]


def collect_old(archive):
    result = {}
    result["kb1"] = grouped(archive, "kb1_results.json", ["dataset"],
                            ["fisa_accuracy", "fkge_accuracy_mean", "fisa_bacc", "fkge_bacc_mean",
                             "fisa_auc", "fkge_auc_mean"])
    result["kb2"] = grouped(archive, "kb2_results.json", ["kb"],
                            ["fisa_accuracy", "fkge_accuracy_mean", "fkge_auc_mean", "n_rules"])
    result["kb3"] = sorted(grouped(archive, "kb3_results.json", ["d"],
                                   ["accuracy_mean", "avg_ms_mean", "train_time_s_mean"]), key=lambda row: row["d"])
    result["kb4"] = grouped(archive, "kb4_results.json", ["lambda", "beta"], ["accuracy_mean"])
    result["kb5"] = sorted(grouped(archive, "kb5_results.json", ["w", "K"], ["accuracy_mean"]),
                            key=lambda row: row["K"])
    result["kb6"] = sorted(grouped(archive, "kb6_results.json", ["ratio"],
                                   ["n_rules", "fisa_avg_ms", "fisa_seq_avg_ms", "fkge_avg_ms_mean"]),
                            key=lambda row: row["ratio"])
    result["main_kfold"] = archive.read("/ket_qua_geomean/fkge/kfold/brset_kfold.json")
    return result


def collect_new(archive):
    result = []
    for fold in range(1, 6):
        obj = archive.read(f"/expected/fkge_tuned/fold_{fold:02}/tuned_results.json")
        candidates = obj["rows"]
        chosen = min(candidates, key=lambda row: row[obj["criterion"]])
        if obj["selected"] != chosen or len(candidates) != 108:
            raise ValueError("Selected configuration does not match the 108-row validation argmin.")
        result.append({"fold": f"fold_{fold:02}", "selected": obj["selected"], "fisa": obj["fisa"],
                       "grid": obj["grid"], "n_configs": obj["n_configs"], "candidates": candidates})
    manifest = archive.read("/expected/fkge_tuned/run_manifest.json")
    # Author-specific absolute paths are unnecessary in the public aggregate.
    manifest["config"] = {key: value for key, value in manifest["config"].items() if key not in ("root", "out")}
    return {"manifest": manifest, "folds": result}


def finish(fig, path, title, footer):
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985, color="#172b44")
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=8, color="#59636f")
    fig.tight_layout(rect=(0.015, 0.10, 0.985, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def errorbar(ax, x, rows, metric, label=None, color=BLUE):
    ax.errorbar(x, [row[metric]["mean"] for row in rows],
                yerr=[row[metric]["sd"] for row in rows], marker="o", capsize=4,
                color=color, linewidth=1.8, label=label)
    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)


def old_panel(ax, number, old):
    if number == 1:
        row = old["kb1"][0]
        for offset, names, label, color in [(-0.18, ["fisa_accuracy", "fisa_bacc", "fisa_auc"], "FISA", ORANGE),
                                            (0.18, ["fkge_accuracy_mean", "fkge_bacc_mean", "fkge_auc_mean"], "FKG-E", BLUE)]:
            ax.bar(np.arange(3) + offset, [row[key]["mean"] for key in names], width=0.34,
                   yerr=[row[key]["sd"] for key in names], capsize=4, color=color, label=label)
        ax.set_xticks(range(3), ["Accuracy", "Balanced accuracy", "ROC-AUC"])
        ax.set_ylim(0, 1.04)
        ax.legend()
        ax.set_title("KB1 — FISA và FKG-E")
    elif number == 2:
        rows = old["kb2"]
        for offset, metric, label, color in [(-0.18, "fisa_accuracy", "FISA", ORANGE),
                                            (0.18, "fkge_accuracy_mean", "FKG-E", BLUE)]:
            ax.bar(np.arange(len(rows)) + offset, [row[metric]["mean"] for row in rows], width=0.34,
                   yerr=[row[metric]["sd"] for row in rows], color=color, capsize=4, label=label)
        ax.set_xticks(range(len(rows)), ["Luật đầy đủ" if row["kb"] == "KB2-full" else "S-FKGS 30%" for row in rows])
        ax.set(ylabel="Accuracy", ylim=(0, 1.04))
        ax.legend()
        ax.set_title("KB2 — Tập luật đầy đủ và rút gọn")
    elif number == 3:
        rows = old["kb3"]
        errorbar(ax, range(len(rows)), rows, "accuracy_mean")
        ax.set_xticks(range(len(rows)), [str(row["d"]) for row in rows])
        ax.set(xlabel="Chiều nhúng d", ylabel="Accuracy", ylim=(0.60, 1.0))
        ax.set_title("KB3 — Độ nhạy theo chiều nhúng")
    elif number == 4:
        rows = old["kb4"]
        ls = sorted({row["lambda"] for row in rows})
        ps = sorted({row["beta"] for row in rows})
        values = np.array([[next(row["accuracy_mean"]["mean"] for row in rows
                                 if row["lambda"] == l and row["beta"] == p) for p in ps] for l in ls])
        image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0.78, vmax=0.9)
        for i in range(len(ls)):
            for j in range(len(ps)):
                ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if values[i,j] > 0.865 else "#172b44")
        ax.set_xticks(range(len(ps)), [str(v) for v in ps])
        ax.set_yticks(range(len(ls)), [str(v) for v in ls])
        ax.set(xlabel="λ_P", ylabel="λ_I")
        ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Accuracy")
        ax.set_title("KB4 — Trọng số nhất quán / dự đoán")
    elif number == 5:
        rows = old["kb5"]
        errorbar(ax, range(len(rows)), rows, "accuracy_mean")
        ax.set_xticks(range(len(rows)), [str(row["K"]) for row in rows])
        ax.set(xlabel="Số mẫu âm K (đồng xuất hiện toàn luật)", ylabel="Accuracy", ylim=(0.65, 1.0))
        ax.set_title("KB5 — Độ nhạy theo K")
    elif number == 6:
        rows = old["kb6"]
        x = [row["n_rules"]["mean"] for row in rows]
        for metric, name, color in [("fisa_seq_avg_ms", "FISA tuần tự", GREY),
                                     ("fisa_avg_ms", "FISA bảng tra", ORANGE),
                                     ("fkge_avg_ms_mean", "FKG-E", BLUE)]:
            errorbar(ax, x, rows, metric, name, color)
        ax.set(xlabel="Số luật trung bình", ylabel="ms / truy vấn", yscale="log")
        ax.legend(fontsize=8)
        ax.set_title("KB6 — Thời gian suy diễn theo số luật")
    if number != 4:
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)


def render_old(old, directory):
    for number in range(1, 7):
        fig, ax = plt.subplots(figsize=(9, 5.7))
        old_panel(ax, number, old)
        finish(fig, directory / f"kb{number}.png", f"FKG-E — KB{number}, bản zip cũ", OLD_FOOTER)
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    for number, ax in enumerate(axes.flat, 1):
        old_panel(ax, number, old)
    finish(fig, directory / "kb1_kb6_summary.png", "FKG-E — tổng hợp KB1–KB6 từ zip cũ", OLD_FOOTER)


def summary_panel(ax, values, names, title):
    x = np.arange(len(values))
    means = [float(np.mean(value)) for value in values]
    sd = [float(np.std(value, ddof=1)) for value in values]
    bars = ax.bar(x, means, yerr=sd, capsize=5, color=[ORANGE, BLUE][:len(values)], width=0.58)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.08), title=title)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    for bar, mean, dev in zip(bars, means, sd):
        ax.text(bar.get_x() + bar.get_width()/2, mean + dev + 0.025,
                f"{mean:.4f} ± {dev:.4f}", ha="center", fontsize=10)


def render_new(new, directory):
    folds = new["folds"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, key, title in zip(axes, ["auc", "pr_auc"], ["ROC-AUC", "Average Precision (PR-AUC)"]):
        summary_panel(ax, [[row["fisa"][key] for row in folds],
                           [row["selected"][key + "_mean"] for row in folds]], ["FISA", "FKG-E tuned"], title)
    finish(fig, directory / "tuned_summary_expected.png", "Zip mới — kết quả tuned được tác giả cung cấp", NEW_FOOTER)
    columns = ["Fold", "α", "d", "λ_I", "λ_P", "Loss validation", "ROC-AUC", "AP"]
    cells = [[row["fold"].replace("fold_0", ""), f"{row['selected']['alpha']:g}",
              str(row["selected"]["d_e"]), f"{row['selected']['lam_I']:g}", f"{row['selected']['lam_P']:g}",
              f"{row['selected']['val_pred_loss_mean']:.4f}", f"{row['selected']['auc_mean']:.4f}",
              f"{row['selected']['pr_auc_mean']:.4f}"] for row in folds]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)
    for (i, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dde5")
        cell.set_facecolor("#e6eef8" if i == 0 else ("#f7f9fc" if i % 2 else "white"))
    finish(fig, directory / "selected_configurations.png", "Zip mới — cấu hình có validation loss nhỏ nhất", NEW_FOOTER)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, fold in zip(axes.flat, folds):
        rows = fold["candidates"]
        ax.scatter([r["val_pred_loss_mean"] for r in rows], [r["pr_auc_mean"] for r in rows],
                   c=[r["alpha"] for r in rows], cmap="viridis", s=20, alpha=0.65, vmin=0.3, vmax=0.9)
        chosen = fold["selected"]
        ax.scatter([chosen["val_pred_loss_mean"]], [chosen["pr_auc_mean"]], marker="*", s=180,
                   color=ORANGE, edgecolor="#172b44", linewidth=0.7, label="Được chọn theo loss")
        ax.set(title=fold["fold"], xlabel="Loss validation", ylabel="AP trên held-out")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.7, "108 cấu hình / fold\nMàu: α = 0.3 → 0.9\n★ Cấu hình được chọn bằng loss validation\n\nĐiểm AP không tham gia hàm chọn.",
                       fontsize=11, va="top")
    finish(fig, directory / "validation_selection_grid.png", "Zip mới — toàn bộ lưới cấu hình đã lưu", NEW_FOOTER)


def read_predictions(directory, fold):
    with (directory / fold / "predictions_PRIVATE.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["image_id"] for row in rows}) != len(rows):
        raise AssertionError("Duplicate replay image scores.")
    return (np.array([int(row["y_true"]) for row in rows]),
            np.array([float(row["score_fisa"]) for row in rows]),
            np.array([float(row["score_fkge"]) for row in rows]))


def render_replay(directory, destination):
    obj = json.loads((directory / "replay_results.json").read_text(encoding="utf-8"))
    if obj["manifest"]["status"] != "completed" or len(obj["folds"]) != 5:
        raise ValueError("Replay must contain five completed folds.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    average_precision = {"FISA": [], "FKG-E": []}
    roc = {"FISA": [], "FKG-E": []}
    grid = np.linspace(0, 1, 201)
    precision_grid = defaultdict(list)
    prevalence = []
    score_hashes = {}
    for fold in obj["folds"]:
        y, sf, se = read_predictions(directory, fold["fold"])
        score_path = directory / fold["fold"] / "predictions_PRIVATE.csv"
        score_hashes[fold["fold"]] = hashlib.sha256(score_path.read_bytes()).hexdigest()
        prevalence.append(float(y.mean()))
        for name, score, color, recorded in [("FISA", sf, ORANGE, fold["fisa"]), ("FKG-E", se, BLUE, fold["fkge"])]:
            ap = float(average_precision_score(y, score))
            auc = float(roc_auc_score(y, score))
            if abs(ap - recorded["pr_auc"]) > 1e-10 or abs(auc - recorded["auc"]) > 1e-10:
                raise AssertionError("Replay prediction scores do not match recorded metrics.")
            average_precision[name].append(ap)
            roc[name].append(auc)
            p, r, _ = precision_recall_curve(y, score)
            axes[0].step(r, p, where="post", color=color, alpha=0.24, linewidth=0.9)
            precision_grid[name].append([p[np.where(r >= target)[0][-1]] for target in grid])
    for name, color in [("FISA", ORANGE), ("FKG-E", BLUE)]:
        values = average_precision[name]
        axes[0].plot(grid, np.mean(precision_grid[name], axis=0), color=color, linewidth=2.4,
                     label=f"{name}: AP {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")
        axes[1].plot(range(1, 6), values, marker="o", color=color, linewidth=1.5, label=name)
    axes[0].axhline(np.mean(prevalence), color=GREY, linestyle="--", label=f"Tỷ lệ dương {np.mean(prevalence):.4f}")
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Đường PR: mảnh từng fold, đậm trung bình")
    axes[1].set(xticks=range(1, 6), xlabel="Fold", ylabel="Average Precision", ylim=(0, 1), title="AP từng fold")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    finish(fig, destination / "pr_replayed.png", "Zip mới — đường PR từ điểm dự đoán chạy lại", REPLAY_FOOTER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    summary_panel(axes[0], [roc["FISA"], roc["FKG-E"]], ["FISA", "FKG-E"], "ROC-AUC")
    summary_panel(axes[1], [average_precision["FISA"], average_precision["FKG-E"]], ["FISA", "FKG-E"], "Average Precision")
    finish(fig, destination / "tuned_summary_replayed.png", "Zip mới — kết quả chạy lại 5 cấu hình được chọn", REPLAY_FOOTER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for fold in obj["folds"]:
        for ax, metric in zip(axes, ["loss", "val_pred"]):
            values = fold["history"][metric]
            ax.plot(range(1, len(values) + 1), values, label=fold["fold"])
    axes[0].set(title="Mục tiêu huấn luyện", xlabel="Vòng lặp", ylabel="Loss")
    axes[1].set(title="Cross-entropy validation", xlabel="Vòng lặp", ylabel="Loss validation")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    finish(fig, destination / "learning_curves_replayed.png", "Zip mới — đường học của lần chạy lại", REPLAY_FOOTER)
    # Publish aggregate metrics/history only, never the predictions or IDs.
    return {"manifest": obj["manifest"], "folds": obj["folds"],
            "recomputed_from_scores": {"ap": average_precision, "roc_auc": roc,
                                       "prevalence": prevalence, "private_score_sha256": score_hashes}}


def comparison(old, new, path):
    old_metrics = old["main_kfold"]["summary"]
    original_roc = old_metrics["FKGE"]["auc"]
    new_roc = stats([row["selected"]["auc_mean"] for row in new["folds"]])
    new_ap = stats([row["selected"]["pr_auc_mean"] for row in new["folds"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    axes[0].bar([0, 1], [original_roc["mean"], new_roc["mean"]],
                yerr=[original_roc["sd"], new_roc["sd"]], capsize=5, color=[GREY, BLUE])
    axes[0].set(xticks=[0, 1], xticklabels=["ZIP cũ: kfold-only", "ZIP mới: tuned"], ylim=(0, 1.04), title="FKG-E ROC-AUC được cung cấp")
    axes[1].bar([1], [new_ap["mean"]], yerr=[new_ap["sd"]], capsize=5, color=BLUE)
    axes[1].text(0, 0.25, "Không có AP\nFKG-E trong zip cũ", ha="center", color=GREY)
    axes[1].set(xticks=[0, 1], xticklabels=["ZIP cũ", "ZIP mới: tuned"], xlim=(-0.6, 1.6), ylim=(0, 1.04), title="FKG-E Average Precision được cung cấp")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    finish(fig, path, "Đối chiếu hai zip — khác giao thức đánh giá",
           "Số liệu đã lưu trong hai gói; không phải một phép so sánh mô hình cùng giao thức.\nKhông suy ra ưu thế hoặc ý nghĩa thống kê từ biểu đồ này.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-zip", type=Path, required=True)
    parser.add_argument("--new-zip", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    old_archive, new_archive = Archive(args.old_zip), Archive(args.new_zip)
    old, new = collect_old(old_archive), collect_new(new_archive)
    render_old(old, args.output / "old_zip")
    render_new(new, args.output / "new_zip")
    comparison(old, new, args.output / "archive_comparison.png")
    replay = render_replay(args.replay, args.output / "new_zip") if args.replay else None
    aggregate = {"old": old, "new": new, "replay": replay}
    (args.output / "aggregate_results.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {"purpose": "Figure export, with archived and selected-configuration replay clearly separated.",
                  "archives": [old_archive.provenance(), new_archive.provenance()],
                  "python": platform.python_version(),
                  "packages": {name: importlib.metadata.version(name) for name in ["numpy", "matplotlib", "scikit-learn"]},
                  "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "replay_provenance": {
                      "runner_sha256": hashlib.sha256(Path(__file__).with_name("replay_selected.py").read_bytes()).hexdigest(),
                      "private_results_sha256": hashlib.sha256((args.replay / "replay_results.json").read_bytes()).hexdigest(),
                  } if args.replay else None,
                  "images": {path.relative_to(args.output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted(args.output.rglob("*.png"))}}
    (args.output / "figure_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(provenance['images'])} PNG figures to {args.output}")


if __name__ == "__main__":
    main()
