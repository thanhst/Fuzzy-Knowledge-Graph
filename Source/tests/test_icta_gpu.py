#!/usr/bin/env python3
"""
Run FKG test on ICTA dataset with backend report (CPU/GPU), fallback, and timing.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_windows_dll_dirs() -> None:
    if os.name != "nt":
        return

    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")

    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass


def resolve_module_dir(module_dir: str) -> Path:
    root = project_root()
    candidates = []
    if module_dir in ("gpu", "auto"):
        candidates.append(root / "GPU" / "Source")
    if module_dir in ("source", "auto"):
        candidates.append(root / "Source")

    for path in candidates:
        if any(path.glob("fisa_module*.pyd")):
            return path
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not resolve module dir for fisa_module.")


def discretize_column(series, bins: int):
    import pandas as pd

    unique_count = int(series.nunique(dropna=False))
    if unique_count <= 1:
        return pd.Series([1] * len(series), index=series.index, dtype="int64")

    q = min(max(2, bins), unique_count)
    ranked = series.rank(method="first")
    try:
        return (pd.qcut(ranked, q=q, labels=False, duplicates="drop") + 1).astype("int64")
    except ValueError:
        min_v = float(series.min())
        max_v = float(series.max())
        if max_v <= min_v:
            return pd.Series([1] * len(series), index=series.index, dtype="int64")
        scaled = ((series - min_v) / (max_v - min_v) * (q - 1)).round().astype("int64") + 1
        return scaled.clip(1, q)


def load_icta_records(csv_path: Path, bins: int):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("ICTA CSV must contain feature columns and one label column.")

    features = df.iloc[:, :-1].copy()
    labels = df.iloc[:, -1]

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    binned_cols = []
    for col in features.columns:
        binned_cols.append(discretize_column(features[col], bins))
    features_binned = pd.concat(binned_cols, axis=1)

    labels_num = pd.to_numeric(labels, errors="coerce").fillna(0).astype("int64")
    unique_labels = sorted(labels_num.unique().tolist())
    label_to_index = {label: i + 1 for i, label in enumerate(unique_labels)}
    indexed_labels = labels_num.map(label_to_index).astype("int64")

    records = []
    for feat, lbl in zip(features_binned.values.tolist(), indexed_labels.tolist()):
        records.append([int(v) for v in feat] + [int(lbl)])

    return records, label_to_index


def split_train_test(records, test_ratio: float, seed: int):
    if len(records) < 4:
        raise ValueError("ICTA dataset is too small for train/test split.")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    split = int(len(records) * (1.0 - test_ratio))
    split = max(2, min(split, len(records) - 2))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = [records[i] for i in train_idx]
    test = [records[i] for i in test_idx]
    return train, test


def build_confusion_matrix(actual, predicted, labels):
    label_to_pos = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for a, p in zip(actual, predicted):
        if a in label_to_pos and p in label_to_pos:
            matrix[label_to_pos[a]][label_to_pos[p]] += 1
    return matrix


def print_confusion_matrix(matrix, labels) -> None:
    label_text = [str(x) for x in labels]
    max_cell = max([len(x) for x in label_text] + [len(str(v)) for row in matrix for v in row] + [4])
    col_w = max_cell + 2

    print("[INFO] confusion matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(col_w) + "".join(lbl.rjust(col_w) for lbl in label_text)
    print(header)
    for i, row in enumerate(matrix):
        row_text = label_text[i].ljust(col_w) + "".join(str(v).rjust(col_w) for v in row)
        print(row_text)


def save_confusion_matrix_csv(matrix, labels, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + [str(x) for x in labels])
        for i, row in enumerate(matrix):
            writer.writerow([str(labels[i])] + row)


def save_numeric_matrix_csv(matrix, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row)


def save_confusion_matrix_image(matrix, labels, output_png: Path, title: str) -> tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return False, f"Cannot import matplotlib: {exc}"

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = [str(x) for x in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    vmax = max(1, max(max(r) for r in matrix))
    threshold = vmax / 2.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return True, ""


def save_scores_image(precision_pct, recall_pct, labels, output_png: Path, title: str) -> tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        return False, f"Cannot import matplotlib/numpy: {exc}"

    output_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    rect1 = ax.bar(x - width / 2, precision_pct, width, label="Precision")
    rect2 = ax.bar(x + width / 2, recall_pct, width, label="Recall")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.set_xlabel("Class")
    ax.set_xticks(x)
    ax.set_xticklabels([str(x) for x in labels])
    ax.set_title(title)
    ax.legend()

    for rect in list(rect1) + list(rect2):
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return True, ""


def compute_class_metrics(actual, predicted, labels):
    metrics = {}
    for label in labels:
        tp = sum(1 for a, p in zip(actual, predicted) if a == label and p == label)
        fp = sum(1 for a, p in zip(actual, predicted) if a != label and p == label)
        fn = sum(1 for a, p in zip(actual, predicted) if a == label and p != label)
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def calculate_abcm_python_fallback(train_records, n_classes):
    import numpy as np

    n_rows = len(train_records)
    n_cols = len(train_records[0])
    n_features = n_cols - 1

    features = [row[:-1] for row in train_records]
    labels = [int(row[-1]) for row in train_records]

    comb4 = list(itertools.combinations(range(n_features), 4))
    comb3 = list(itertools.combinations(range(n_features), 3))

    A = np.zeros((n_rows, len(comb4)), dtype=float)
    for idx, cols in enumerate(comb4):
        counts = {}
        keys = []
        for row in features:
            key = (row[cols[0]], row[cols[1]], row[cols[2]], row[cols[3]])
            keys.append(key)
            counts[key] = counts.get(key, 0) + 1
        for r, key in enumerate(keys):
            A[r, idx] = counts[key] / n_rows

    M = np.zeros((n_rows, n_features), dtype=float)
    for feat_idx in range(n_features):
        counts = {}
        keys = []
        for row, label in zip(features, labels):
            key = (row[feat_idx], label)
            keys.append(key)
            counts[key] = counts.get(key, 0) + 1
        for r, key in enumerate(keys):
            M[r, feat_idx] = counts[key] / n_rows

    B = np.zeros((n_rows, len(comb3)), dtype=float)
    sum_a = A.sum(axis=1)
    for idx, cols in enumerate(comb3):
        mins = np.minimum(np.minimum(M[:, cols[0]], M[:, cols[1]]), M[:, cols[2]])
        B[:, idx] = sum_a * mins

    C = np.zeros((n_rows, len(comb3) * n_classes), dtype=float)
    class_values = list(range(1, n_classes + 1))
    for comb_idx, cols in enumerate(comb3):
        agg = {}
        for r, row in enumerate(features):
            key = (row[cols[0]], row[cols[1]], row[cols[2]], labels[r])
            agg[key] = agg.get(key, 0.0) + float(B[r, comb_idx])

        for r, row in enumerate(features):
            for class_pos, class_val in enumerate(class_values):
                c_idx = class_pos * len(comb3) + comb_idx
                key = (row[cols[0]], row[cols[1]], row[cols[2]], class_val)
                C[r, c_idx] = agg.get(key, 0.0)

    return A, B, C, M


def collect_abcm_from_fkg(fkg, fkg_class, train_records, n_classes):
    # Prefer matrices already computed in native C++ train() path.
    has_getters = hasattr(fkg, "get_A") and hasattr(fkg, "get_B") and hasattr(fkg, "get_C")
    if has_getters:
        A = fkg.get_A()
        B = fkg.get_B()
        C = fkg.get_C()
        if hasattr(fkg, "get_M"):
            M = fkg.get_M()
            return A, B, C, M, "native_getters"
        if hasattr(fkg_class, "calculateM"):
            M = fkg_class.calculateM(train_records)
            return A, B, C, M, "native_getters+calculateM"

    # Secondary path: use C++ static methods.
    can_use_static = all(
        hasattr(fkg_class, fn) for fn in ["calculateA", "calculateM", "calculateB", "calculateC"]
    )
    if can_use_static:
        A = fkg_class.calculateA(train_records)
        M = fkg_class.calculateM(train_records)
        B = fkg_class.calculateB(train_records, A, M)
        C = fkg_class.calculateC(train_records, B, n_classes)
        return A, B, C, M, "native_static"

    # Last-resort fallback to Python implementation.
    A, B, C, M = calculate_abcm_python_fallback(train_records, n_classes)
    return A, B, C, M, "python_fallback"


def pick_backend(fkg, requested: str):
    import fisa_module  # noqa: PLC0415

    if hasattr(fisa_module, "GPU_COMPILED"):
        gpu_compiled = bool(fisa_module.GPU_COMPILED)
    elif hasattr(fisa_module, "is_gpu_compiled"):
        gpu_compiled = bool(fisa_module.is_gpu_compiled())
    elif hasattr(fkg, "isGPUCompiled"):
        gpu_compiled = bool(fkg.isGPUCompiled())
    else:
        gpu_compiled = False

    if hasattr(fisa_module, "is_gpu_available"):
        gpu_available = bool(fisa_module.is_gpu_available())
    elif hasattr(fkg, "isGPUAvailable"):
        gpu_available = bool(fkg.isGPUAvailable())
    else:
        gpu_available = False

    if requested == "gpu":
        if not (gpu_compiled and gpu_available):
            print("[WARN] GPU requested but unavailable. Falling back to CPU.")
            return "cpu", gpu_compiled, gpu_available
        return "gpu", gpu_compiled, gpu_available

    if requested == "cpu":
        return "cpu", gpu_compiled, gpu_available

    return ("gpu" if (gpu_compiled and gpu_available) else "cpu"), gpu_compiled, gpu_available


def set_backend(fkg, backend: str) -> None:
    use_gpu = backend == "gpu"
    if hasattr(fkg, "set_use_gpu"):
        fkg.set_use_gpu(use_gpu)
        return
    if hasattr(fkg, "setBackend"):
        fkg.setBackend(backend)
        return
    if hasattr(fkg, "setUseGPU"):
        fkg.setUseGPU(use_gpu)


def main() -> int:
    parser = argparse.ArgumentParser(description="ICTA backend/timing test for fisa_module")
    parser.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--module-dir", choices=["auto", "source", "gpu"], default="auto")
    parser.add_argument(
        "--csv",
        default=str(project_root() / "Source_code" / "data" / "ICTA" / "ICTA.csv"),
        help="Path to ICTA CSV dataset",
    )
    parser.add_argument("--bins", type=int, default=6, help="Discretization bins per feature")
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=str(project_root() / "result" / "ICTA"),
        help="Output directory for confusion matrix files",
    )
    parser.add_argument(
        "--save-cm-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save confusion matrix image (.png)",
    )
    parser.add_argument(
        "--save-fkg-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save FKG-style outputs (A/B/C/M, Results_FKG.csv, scores/confusion images)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"[ERROR] Dataset not found: {csv_path}")
        return 2

    module_dir = resolve_module_dir(args.module_dir)
    add_windows_dll_dirs()
    sys.path.insert(0, str(module_dir))

    t_import = time.perf_counter()
    import fisa_module  # noqa: PLC0415

    import_ms = (time.perf_counter() - t_import) * 1000.0
    fkg = fisa_module.fkg.FKG()

    backend_used, gpu_compiled, gpu_available = pick_backend(fkg, args.backend)
    set_backend(fkg, backend_used)

    records, label_to_index = load_icta_records(csv_path, args.bins)
    train, test = split_train_test(records, args.test_ratio, args.seed)

    n_classes = len(label_to_index)
    t0 = time.perf_counter()
    fkg.train(train, n_classes)
    train_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    predicted = []
    confidences = []
    for row in test:
        pred, conf = fkg.predict(row[:-1])
        predicted.append(int(pred))
        confidences.append(float(conf))
    infer_ms = (time.perf_counter() - t1) * 1000.0

    actual = [int(row[-1]) for row in test]
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    accuracy = (correct / len(actual)) if actual else 0.0
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

    inv_label = {v: k for k, v in label_to_index.items()}
    class_labels = [inv_label[i] for i in sorted(inv_label.keys())]
    pred_orig_full = [inv_label.get(x, x) for x in predicted]
    actual_orig_full = [inv_label.get(x, x) for x in actual]
    cm = build_confusion_matrix(actual_orig_full, pred_orig_full, class_labels)
    pred_orig = [inv_label.get(x, x) for x in predicted[:10]]
    actual_orig = [inv_label.get(x, x) for x in actual[:10]]

    print("ICTA FKG Backend Test")
    print(f"[INFO] dataset           : {csv_path}")
    print(f"[INFO] module path       : {module_dir}")
    print(f"[INFO] backend request   : {args.backend}")
    print(f"[INFO] backend used      : {backend_used}")
    print(f"[INFO] gpu compiled      : {gpu_compiled}")
    print(f"[INFO] gpu available     : {gpu_available}")
    print(f"[INFO] import time       : {import_ms:.3f} ms")
    print(f"[INFO] samples train/test: {len(train)}/{len(test)}")
    print(f"[INFO] classes           : {n_classes}")
    print(f"[INFO] train time        : {train_ms:.3f} ms")
    print(f"[INFO] infer time total  : {infer_ms:.3f} ms")
    print(f"[INFO] infer time/sample : {infer_ms / max(1, len(test)):.3f} ms")
    print(f"[INFO] accuracy          : {accuracy * 100.0:.2f}%")
    print(f"[INFO] avg confidence    : {avg_conf:.6f}")
    print(f"[INFO] first10 pred(orig): {pred_orig}")
    print(f"[INFO] first10 true(orig): {actual_orig}")
    print_confusion_matrix(cm, class_labels)

    out_dir = Path(args.out_dir).resolve()
    cm_csv_path = out_dir / f"icta_confusion_matrix_{backend_used}.csv"
    save_confusion_matrix_csv(cm, class_labels, cm_csv_path)
    print(f"[INFO] confusion matrix csv: {cm_csv_path}")

    if args.save_cm_image:
        cm_png_path = out_dir / f"icta_confusion_matrix_{backend_used}.png"
        ok, err = save_confusion_matrix_image(
            cm,
            class_labels,
            cm_png_path,
            title=f"ICTA Confusion Matrix ({backend_used.upper()})",
        )
        if ok:
            print(f"[INFO] confusion matrix png: {cm_png_path}")
        else:
            print(f"[WARN] Failed to save confusion matrix image: {err}")

    # Save FKG-style artifacts for compatibility with old pipeline outputs.
    if args.save_fkg_style:
        try:
            fkg_dir = out_dir

            metric_by_class = compute_class_metrics(actual_orig_full, pred_orig_full, class_labels)
            precision_pct = [metric_by_class[c]["precision"] * 100.0 for c in class_labels]
            recall_pct = [metric_by_class[c]["recall"] * 100.0 for c in class_labels]
            f1_pct = [metric_by_class[c]["f1"] * 100.0 for c in class_labels]

            scores_png = fkg_dir / "scores.png"
            ok, err = save_scores_image(
                precision_pct,
                recall_pct,
                class_labels,
                scores_png,
                title=f"Precision and Recall per Class ({backend_used.upper()})",
            )
            if ok:
                print(f"[INFO] scores png          : {scores_png}")
            else:
                print(f"[WARN] Failed to save scores image: {err}")

            conf_png = fkg_dir / "conf_matrix.png"
            ok, err = save_confusion_matrix_image(
                cm,
                class_labels,
                conf_png,
                title=f"Confusion Matrix ({backend_used.upper()})",
            )
            if ok:
                print(f"[INFO] conf matrix png     : {conf_png}")
            else:
                print(f"[WARN] Failed to save FKG conf image: {err}")

            conf_csv = fkg_dir / "conf_matrix.csv"
            save_confusion_matrix_csv(cm, class_labels, conf_csv)
            print(f"[INFO] conf matrix csv     : {conf_csv}")

            predictions_csv = fkg_dir / "Predictions_FKG.csv"
            predictions_csv.parent.mkdir(parents=True, exist_ok=True)
            with predictions_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "true_label", "pred_label", "confidence"])
                for i, (a, p, c) in enumerate(zip(actual_orig_full, pred_orig_full, confidences)):
                    writer.writerow([i, a, p, f"{c:.8f}"])
            print(f"[INFO] predictions csv     : {predictions_csv}")

            abcm_start = time.perf_counter()
            A, B, C, M, matrix_source = collect_abcm_from_fkg(
                fkg,
                fisa_module.fkg.FKG,
                train,
                n_classes,
            )
            abcm_ms = (time.perf_counter() - abcm_start) * 1000.0

            a_csv = fkg_dir / "A.csv"
            b_csv = fkg_dir / "B.csv"
            c_csv = fkg_dir / "C.csv"
            m_csv = fkg_dir / "M.csv"
            save_numeric_matrix_csv(A, a_csv)
            save_numeric_matrix_csv(B, b_csv)
            save_numeric_matrix_csv(C, c_csv)
            save_numeric_matrix_csv(M, m_csv)
            print(f"[INFO] A/B/C/M csv         : {a_csv}, {b_csv}, {c_csv}, {m_csv}")
            print(f"[INFO] matrix source       : {matrix_source}")

            avg_precision = sum(precision_pct) / len(precision_pct) if precision_pct else 0.0
            avg_recall = sum(recall_pct) / len(recall_pct) if recall_pct else 0.0
            rank_counts = {}
            for conf in confidences:
                key = f"{conf:.4f}"
                rank_counts[key] = rank_counts.get(key, 0) + 1

            results_csv = fkg_dir / "Results_FKG.csv"
            results_csv.parent.mkdir(parents=True, exist_ok=True)
            with results_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Train Time",
                        "Test Time",
                        "Total Time",
                        "Import Time (ms)",
                        "ABCM Time (ms)",
                        "Matrix Source",
                        "Test Accuracy",
                        "Test Precision",
                        "Test Recall",
                        "Test F1",
                        "Count List Rank",
                        "List Rank Length",
                        "Label",
                        "Backend Request",
                        "Backend Used",
                        "GPU Compiled",
                        "GPU Available",
                    ]
                )
                writer.writerow(
                    [
                        f"{train_ms / 1000.0:.6f}",
                        f"{infer_ms / 1000.0:.6f}",
                        f"{(train_ms + infer_ms) / 1000.0:.6f}",
                        f"{import_ms:.3f}",
                        f"{abcm_ms:.3f}",
                        matrix_source,
                        f"{accuracy * 100.0:.4f}",
                        f"{avg_precision:.4f}",
                        f"{avg_recall:.4f}",
                        f"{(sum(f1_pct) / len(f1_pct) if f1_pct else 0.0):.4f}",
                        json.dumps(rank_counts, ensure_ascii=False),
                        str(len(confidences)),
                        json.dumps(pred_orig_full, ensure_ascii=False),
                        args.backend,
                        backend_used,
                        str(gpu_compiled),
                        str(gpu_available),
                    ]
                )
            print(f"[INFO] results csv         : {results_csv}")
        except Exception as exc:
            print(f"[WARN] Failed to save full FKG-style artifacts: {exc}")

    print("[OK] ICTA test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
