#!/usr/bin/env python3
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.fkg_python.fkg_runtime import try_import_fisa_module as shared_try_import_fisa_module


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def try_import_fisa_module() -> Tuple[Optional[object], Optional[Path], Optional[str]]:
    result = shared_try_import_fisa_module(preferred="source", clear_existing=True)
    return result.module, result.module_dir, result.error or None


def discretize_column(series: pd.Series, bins: int) -> pd.Series:
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


def build_records(df: pd.DataFrame, bins: int) -> Tuple[List[List[int]], Dict[int, int], Dict[int, int], List[str]]:
    feature_cols = [c for c in df.columns if c not in {"patient_id", "Outcome"}]
    if not feature_cols:
        raise ValueError("Fused dataset has no feature columns.")

    features = df[feature_cols].copy()
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    binned_cols = [discretize_column(features[col], bins=bins) for col in feature_cols]
    features_binned = pd.concat(binned_cols, axis=1)

    labels = pd.to_numeric(df["Outcome"], errors="coerce").fillna(0).astype(int)
    unique_labels = sorted(labels.unique().tolist())
    label_to_idx = {label: i + 1 for i, label in enumerate(unique_labels)}
    idx_to_label = {i + 1: label for i, label in enumerate(unique_labels)}
    indexed_labels = labels.map(label_to_idx).astype(int)

    records: List[List[int]] = []
    for feat, lbl in zip(features_binned.values.tolist(), indexed_labels.tolist()):
        records.append([int(v) for v in feat] + [int(lbl)])
    return records, label_to_idx, idx_to_label, feature_cols


def split_train_test(records: List[List[int]], test_ratio: float, seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    if len(records) < 4:
        raise ValueError("Dataset is too small for train/test split.")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    split = int(len(records) * (1.0 - test_ratio))
    split = max(2, min(split, len(records) - 2))

    train = [records[i] for i in indices[:split]]
    test = [records[i] for i in indices[split:]]
    return train, test


def calculate_abcm_python(train_records: List[List[int]], n_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(train_records)
    n_cols = len(train_records[0])
    n_features = n_cols - 1

    features = [row[:-1] for row in train_records]
    labels = [int(row[-1]) for row in train_records]

    comb4 = list(itertools.combinations(range(n_features), 4))
    comb3 = list(itertools.combinations(range(n_features), 3))

    A = np.zeros((n_rows, len(comb4)), dtype=float)
    for idx, cols in enumerate(comb4):
        counts: Dict[Tuple[int, int, int, int], int] = {}
        keys = []
        for row in features:
            key = (row[cols[0]], row[cols[1]], row[cols[2]], row[cols[3]])
            keys.append(key)
            counts[key] = counts.get(key, 0) + 1
        for r, key in enumerate(keys):
            A[r, idx] = counts[key] / n_rows

    M = np.zeros((n_rows, n_features), dtype=float)
    for feat_idx in range(n_features):
        counts: Dict[Tuple[int, int], int] = {}
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
        agg: Dict[Tuple[int, int, int, int], float] = {}
        for r, row in enumerate(features):
            key = (row[cols[0]], row[cols[1]], row[cols[2]], labels[r])
            agg[key] = agg.get(key, 0.0) + float(B[r, comb_idx])

        for r, row in enumerate(features):
            for class_pos, class_val in enumerate(class_values):
                c_idx = class_pos * len(comb3) + comb_idx
                key = (row[cols[0]], row[cols[1]], row[cols[2]], class_val)
                C[r, c_idx] = agg.get(key, 0.0)

    return A, B, C, M


def predict_one_python(train_records: List[List[int]], C: np.ndarray, sample: List[int], n_classes: int) -> Tuple[int, float]:
    n_features = len(sample)
    comb3 = list(itertools.combinations(range(n_features), 3))
    n_comb = len(comb3)
    scores = np.zeros(n_classes, dtype=float)

    majority_label = int(max(train_records, key=lambda r: sum(1 for x in train_records if x[-1] == r[-1]))[-1])
    for class_idx in range(1, n_classes + 1):
        c_values = np.zeros(n_comb, dtype=float)
        for comb_idx, cols in enumerate(comb3):
            best_val = 0.0
            for r_idx, row in enumerate(train_records):
                if row[-1] != class_idx:
                    continue
                if row[cols[0]] == sample[cols[0]] and row[cols[1]] == sample[cols[1]] and row[cols[2]] == sample[cols[2]]:
                    c_idx = (class_idx - 1) * n_comb + comb_idx
                    best_val = max(best_val, float(C[r_idx, c_idx]))
            c_values[comb_idx] = best_val
        scores[class_idx - 1] = float(np.max(c_values) + np.min(c_values))

    if float(np.sum(scores)) <= 1e-12:
        return majority_label, 0.0

    pred = int(np.argmax(scores) + 1)
    confidence = float(scores[pred - 1] / np.sum(scores))
    return pred, confidence


def compute_class_metrics(actual: List[int], predicted: List[int], labels: List[int]) -> Dict[int, Dict[str, float]]:
    metrics: Dict[int, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for a, p in zip(actual, predicted) if a == label and p == label)
        fp = sum(1 for a, p in zip(actual, predicted) if a != label and p == label)
        fn = sum(1 for a, p in zip(actual, predicted) if a == label and p != label)
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def build_confusion_matrix(actual: List[int], predicted: List[int], labels: List[int]) -> List[List[int]]:
    label_to_pos = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for a, p in zip(actual, predicted):
        if a in label_to_pos and p in label_to_pos:
            matrix[label_to_pos[a]][label_to_pos[p]] += 1
    return matrix


def save_confusion_matrix_image(cm: List[List[int]], labels: List[int], output_png: Path, title: str) -> Tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        return False, f"Cannot import matplotlib: {exc}"

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = [str(x) for x in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    vmax = max(1, max(max(r) for r in cm)) if cm else 1
    threshold = vmax / 2.0
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return True, ""


def save_scores_image(
    precision_pct: List[float],
    recall_pct: List[float],
    labels: List[int],
    output_png: Path,
    title: str,
) -> Tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
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


def save_numeric_matrix_csv(matrix: Iterable[Iterable[float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row)


def save_confusion_csv(cm: List[List[int]], labels: List[int], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + row)


def pick_backend(fisa_module: object, fkg_instance: object, requested: str) -> Tuple[str, bool, bool]:
    gpu_compiled = False
    gpu_available = False

    if hasattr(fisa_module, "GPU_COMPILED"):
        gpu_compiled = bool(getattr(fisa_module, "GPU_COMPILED"))
    elif hasattr(fisa_module, "is_gpu_compiled"):
        gpu_compiled = bool(fisa_module.is_gpu_compiled())
    elif hasattr(fkg_instance, "isGPUCompiled"):
        gpu_compiled = bool(fkg_instance.isGPUCompiled())

    if hasattr(fisa_module, "is_gpu_available"):
        gpu_available = bool(fisa_module.is_gpu_available())
    elif hasattr(fkg_instance, "isGPUAvailable"):
        gpu_available = bool(fkg_instance.isGPUAvailable())

    if requested == "gpu":
        if gpu_compiled and gpu_available:
            return "gpu", gpu_compiled, gpu_available
        return "cpu", gpu_compiled, gpu_available
    if requested == "cpu":
        return "cpu", gpu_compiled, gpu_available
    return ("gpu" if (gpu_compiled and gpu_available) else "cpu"), gpu_compiled, gpu_available


def set_backend(fkg_instance: object, backend: str) -> None:
    use_gpu = backend == "gpu"
    if hasattr(fkg_instance, "set_use_gpu"):
        fkg_instance.set_use_gpu(use_gpu)
    elif hasattr(fkg_instance, "setUseGPU"):
        fkg_instance.setUseGPU(use_gpu)
    elif hasattr(fkg_instance, "setBackend"):
        fkg_instance.setBackend(backend)


def run_with_native_fkg(
    fisa_module: object,
    train_records: List[List[int]],
    test_records: List[List[int]],
    backend_request: str,
) -> Dict[str, object]:
    n_classes = len(sorted({row[-1] for row in train_records + test_records}))
    fkg = fisa_module.fkg.FKG()
    backend_used, gpu_compiled, gpu_available = pick_backend(fisa_module, fkg, backend_request)
    set_backend(fkg, backend_used)

    t_train = time.perf_counter()
    fkg.train(train_records, n_classes)
    train_time = time.perf_counter() - t_train

    test_inputs = [row[:-1] for row in test_records]
    actual = [int(row[-1]) for row in test_records]

    t_test = time.perf_counter()
    predicted: List[int] = []
    confidences: List[float] = []
    if hasattr(fkg, "predict_batch_with_confidence"):
        batch = fkg.predict_batch_with_confidence(test_inputs)
        for pred, conf in batch:
            predicted.append(int(pred))
            confidences.append(float(conf))
    else:
        for row in test_inputs:
            pred, conf = fkg.predict(row)
            predicted.append(int(pred))
            confidences.append(float(conf))
    test_time = time.perf_counter() - t_test

    if hasattr(fkg, "get_A") and hasattr(fkg, "get_B") and hasattr(fkg, "get_C"):
        A = np.array(fkg.get_A(), dtype=float)
        B = np.array(fkg.get_B(), dtype=float)
        C = np.array(fkg.get_C(), dtype=float)
        if hasattr(fkg, "get_M"):
            M = np.array(fkg.get_M(), dtype=float)
            matrix_source = "native_getters"
        else:
            _A, _B, _C, M = calculate_abcm_python(train_records, n_classes)
            matrix_source = "native_getters+python_M"
    else:
        A, B, C, M = calculate_abcm_python(train_records, n_classes)
        matrix_source = "python_fallback_abcm"

    return {
        "engine": "native_fisa_module",
        "backend_used": backend_used,
        "gpu_compiled": gpu_compiled,
        "gpu_available": gpu_available,
        "train_time": train_time,
        "test_time": test_time,
        "predicted": predicted,
        "actual": actual,
        "confidences": confidences,
        "A": A,
        "B": B,
        "C": C,
        "M": M,
        "matrix_source": matrix_source,
    }


def run_with_python_fkg(train_records: List[List[int]], test_records: List[List[int]]) -> Dict[str, object]:
    n_classes = len(sorted({row[-1] for row in train_records + test_records}))

    t_train = time.perf_counter()
    A, B, C, M = calculate_abcm_python(train_records, n_classes)
    train_time = time.perf_counter() - t_train

    t_test = time.perf_counter()
    predicted = []
    confidences = []
    actual = [int(row[-1]) for row in test_records]
    for row in test_records:
        pred, conf = predict_one_python(train_records, C, row[:-1], n_classes=n_classes)
        predicted.append(int(pred))
        confidences.append(float(conf))
    test_time = time.perf_counter() - t_test

    return {
        "engine": "python_fallback",
        "backend_used": "cpu",
        "gpu_compiled": False,
        "gpu_available": False,
        "train_time": train_time,
        "test_time": test_time,
        "predicted": predicted,
        "actual": actual,
        "confidences": confidences,
        "A": A,
        "B": B,
        "C": C,
        "M": M,
        "matrix_source": "python_fallback_abcm",
    }


def run_fkg_flow(fusion_csv: Path, output_dir: Path, bins: int, test_ratio: float, seed: int, backend: str) -> Dict[str, object]:
    df = pd.read_csv(fusion_csv)
    if "Outcome" not in df.columns:
        raise ValueError("Fusion CSV must contain Outcome column.")
    if "patient_id" not in df.columns:
        raise ValueError("Fusion CSV must contain patient_id column.")

    records, label_to_idx, idx_to_label, feature_cols = build_records(df, bins=bins)
    train_records, test_records = split_train_test(records, test_ratio=test_ratio, seed=seed)

    import_error = None
    fisa_module, module_dir, import_error = try_import_fisa_module()
    if fisa_module is not None:
        try:
            result = run_with_native_fkg(
                fisa_module=fisa_module,
                train_records=train_records,
                test_records=test_records,
                backend_request=backend,
            )
            result["module_dir"] = str(module_dir) if module_dir else ""
            result["import_error"] = ""
        except Exception as exc:  # pragma: no cover
            result = run_with_python_fkg(train_records, test_records)
            result["module_dir"] = str(module_dir) if module_dir else ""
            result["import_error"] = f"native_runtime_error: {exc}"
    else:
        result = run_with_python_fkg(train_records, test_records)
        result["module_dir"] = ""
        result["import_error"] = import_error or "cannot_import_fisa_module"

    predicted_idx = result["predicted"]
    actual_idx = result["actual"]
    confidences = result["confidences"]

    pred_orig = [idx_to_label.get(i, i) for i in predicted_idx]
    actual_orig = [idx_to_label.get(i, i) for i in actual_idx]
    class_labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]

    correct = sum(1 for p, a in zip(pred_orig, actual_orig) if p == a)
    accuracy = (correct / len(actual_orig)) if actual_orig else 0.0
    metrics = compute_class_metrics(actual_orig, pred_orig, class_labels)
    precision_avg = float(np.mean([m["precision"] for m in metrics.values()])) if metrics else 0.0
    recall_avg = float(np.mean([m["recall"] for m in metrics.values()])) if metrics else 0.0
    f1_avg = float(np.mean([m["f1"] for m in metrics.values()])) if metrics else 0.0
    conf_matrix = build_confusion_matrix(actual_orig, pred_orig, class_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_numeric_matrix_csv(result["A"], output_dir / "A.csv")
    save_numeric_matrix_csv(result["B"], output_dir / "B.csv")
    save_numeric_matrix_csv(result["C"], output_dir / "C.csv")
    save_numeric_matrix_csv(result["M"], output_dir / "M.csv")
    save_confusion_csv(conf_matrix, class_labels, output_dir / "conf_matrix.csv")

    precision_pct = [metrics[c]["precision"] * 100.0 for c in class_labels]
    recall_pct = [metrics[c]["recall"] * 100.0 for c in class_labels]
    conf_png = output_dir / "conf_matrix.png"
    scores_png = output_dir / "scores.png"
    conf_ok, conf_err = save_confusion_matrix_image(
        conf_matrix,
        class_labels,
        conf_png,
        title=f"ICTA Confusion Matrix ({str(result['backend_used']).upper()})",
    )
    score_ok, score_err = save_scores_image(
        precision_pct,
        recall_pct,
        class_labels,
        scores_png,
        title=f"Precision and Recall per Class ({str(result['backend_used']).upper()})",
    )

    pred_csv = output_dir / "Predictions_FKG.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_label", "confidence"])
        for i, (a, p, c) in enumerate(zip(actual_orig, pred_orig, confidences)):
            writer.writerow([i, a, p, f"{c:.8f}"])

    rank_counts: Dict[str, int] = {}
    for c in confidences:
        key = f"{float(c):.4f}"
        rank_counts[key] = rank_counts.get(key, 0) + 1

    results_csv = output_dir / "Results_FKG.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Train Time",
                "Test Time",
                "Total Time",
                "Test Accuracy",
                "Test Precision",
                "Test Recall",
                "Count List Rank",
                "List Rank Length",
                "Label",
                "Test F1",
                "Engine",
                "Backend Request",
                "Backend Used",
                "GPU Compiled",
                "GPU Available",
                "Matrix Source",
                "Feature Count",
                "Train Samples",
                "Test Samples",
                "Import Error",
                "Module Dir",
            ]
        )
        writer.writerow(
            [
                f"{result['train_time']:.6f}",
                f"{result['test_time']:.6f}",
                f"{(result['train_time'] + result['test_time']):.6f}",
                f"[{accuracy * 100.0:.2f}]",
                f"{precision_avg * 100.0:.3f}",
                f"{recall_avg * 100.0:.3f}",
                json.dumps(rank_counts, ensure_ascii=False),
                str(len(confidences)),
                json.dumps(pred_orig, ensure_ascii=False),
                f"{f1_avg * 100.0:.3f}",
                result["engine"],
                backend,
                result["backend_used"],
                str(result["gpu_compiled"]),
                str(result["gpu_available"]),
                result["matrix_source"],
                str(len(feature_cols)),
                str(len(train_records)),
                str(len(test_records)),
                result.get("import_error", ""),
                result.get("module_dir", ""),
            ]
        )

    summary = {
        "results_csv": results_csv,
        "predictions_csv": pred_csv,
        "conf_matrix_csv": output_dir / "conf_matrix.csv",
        "conf_matrix_png": conf_png,
        "scores_png": scores_png,
        "A_csv": output_dir / "A.csv",
        "B_csv": output_dir / "B.csv",
        "C_csv": output_dir / "C.csv",
        "M_csv": output_dir / "M.csv",
        "engine": result["engine"],
        "backend_used": result["backend_used"],
        "accuracy_pct": accuracy * 100.0,
        "precision_pct": precision_avg * 100.0,
        "recall_pct": recall_avg * 100.0,
        "f1_pct": f1_avg * 100.0,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "conf_png_ok": conf_ok,
        "scores_png_ok": score_ok,
        "conf_png_error": conf_err,
        "scores_png_error": score_err,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FKG flow on fused ICTA-style data (GPU-first, Python fallback).")
    parser.add_argument(
        "--fusion-csv",
        type=Path,
        default=source_root() / "Data" / "processing" / "fusion" / "fusion_selected.csv",
        help="Fusion CSV input path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=source_root() / "Data" / "result" / "ICTA",
        help="Output result directory.",
    )
    parser.add_argument("--bins", type=int, default=6, help="Discretization bins per feature.")
    parser.add_argument("--test-ratio", type=float, default=0.30, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto", help="Requested backend.")
    args = parser.parse_args()

    if not args.fusion_csv.exists():
        raise FileNotFoundError(f"Fusion CSV not found: {args.fusion_csv}")

    summary = run_fkg_flow(
        fusion_csv=args.fusion_csv,
        output_dir=args.out_dir,
        bins=max(2, int(args.bins)),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        backend=str(args.backend),
    )
    print(f"[OK] Engine: {summary['engine']}")
    print(f"[OK] Backend used: {summary['backend_used']}")
    print(f"[OK] Accuracy: {summary['accuracy_pct']:.2f}%")
    print(f"[OK] Results: {summary['results_csv']}")
    if summary.get("conf_png_ok"):
        print(f"[OK] Confusion image: {summary['conf_matrix_png']}")
    else:
        print(f"[WARN] Failed to save confusion image: {summary.get('conf_png_error', '')}")
    if summary.get("scores_png_ok"):
        print(f"[OK] Scores image: {summary['scores_png']}")
    else:
        print(f"[WARN] Failed to save scores image: {summary.get('scores_png_error', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
