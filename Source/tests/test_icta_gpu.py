#!/usr/bin/env python3
"""
Run FKG test on ICTA dataset with backend report (CPU/GPU), fallback, and timing.
"""

from __future__ import annotations

import argparse
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
    print("[OK] ICTA test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
