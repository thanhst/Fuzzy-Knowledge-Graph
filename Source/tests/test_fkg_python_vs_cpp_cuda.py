#!/usr/bin/env python3
"""
Benchmark and compare:
1) Pure Python FKG reference
2) C++ FKG CPU
3) C++ FKG CUDA (when available)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configure_windows_dll() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates = []
    cuda = os.environ.get("CUDA_PATH")
    if cuda:
        candidates.append(Path(cuda) / "bin")
    candidates.append(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"))
    candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")

    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass


def _configure_paths(module_dir: str) -> None:
    root = _repo_root()
    source_dir = root / "Source"
    gpu_dir = root / "GPU" / "Source"

    if module_dir == "source":
        candidates = [source_dir]
    elif module_dir == "gpu":
        candidates = [gpu_dir, source_dir]
    else:
        candidates = [gpu_dir, source_dir]

    for path in reversed(candidates):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _combination(k: int, n: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    result = 1
    for i in range(1, k + 1):
        result = (result * (n - k + i)) // i
    return result


def _minmax_normalize(c_matrix: np.ndarray) -> np.ndarray:
    mins = c_matrix.min(axis=0)
    maxs = c_matrix.max(axis=0)
    ranges = maxs - mins
    out = np.zeros_like(c_matrix)
    nz = ranges > 0
    out[:, nz] = (c_matrix[:, nz] - mins[nz]) / ranges[nz]
    return out


def _python_reference_abcm(base: List[List[float]], n_classes: int):
    n_rows = len(base)
    n_cols = len(base[0])
    n_features = n_cols - 1

    feats = [row[:-1] for row in base]
    labels = [int(row[-1]) for row in base]

    comb4 = list(itertools.combinations(range(n_features), 4))
    comb3 = list(itertools.combinations(range(n_features), 3))

    a_matrix = np.zeros((n_rows, len(comb4)), dtype=np.float64)
    for idx, cols in enumerate(comb4):
        counts: Dict[Tuple[float, float, float, float], int] = {}
        keys = []
        for row in feats:
            key = (row[cols[0]], row[cols[1]], row[cols[2]], row[cols[3]])
            keys.append(key)
            counts[key] = counts.get(key, 0) + 1
        for r, key in enumerate(keys):
            a_matrix[r, idx] = counts[key] / n_rows

    m_matrix = np.zeros((n_rows, n_features), dtype=np.float64)
    for feat_idx in range(n_features):
        counts: Dict[Tuple[float, int], int] = {}
        keys = []
        for row, label in zip(feats, labels):
            key = (row[feat_idx], label)
            keys.append(key)
            counts[key] = counts.get(key, 0) + 1
        for r, key in enumerate(keys):
            m_matrix[r, feat_idx] = counts[key] / n_rows

    b_matrix = np.zeros((n_rows, len(comb3)), dtype=np.float64)
    sum_a = a_matrix.sum(axis=1)
    for idx, cols in enumerate(comb3):
        mins = np.minimum(np.minimum(m_matrix[:, cols[0]], m_matrix[:, cols[1]]), m_matrix[:, cols[2]])
        b_matrix[:, idx] = sum_a * mins

    c_cols = 6 * len(comb3)
    c_matrix = np.zeros((n_rows, c_cols), dtype=np.float64)
    class_limit = min(n_classes, 6)
    for comb_idx, cols in enumerate(comb3):
        agg: Dict[Tuple[float, float, float, int], float] = {}
        for r in range(n_rows):
            key = (feats[r][cols[0]], feats[r][cols[1]], feats[r][cols[2]], labels[r])
            agg[key] = agg.get(key, 0.0) + b_matrix[r, comb_idx]
        for r in range(n_rows):
            key_head = (feats[r][cols[0]], feats[r][cols[1]], feats[r][cols[2]])
            for cls in range(1, class_limit + 1):
                c_index = (cls - 1) * len(comb3) + comb_idx
                c_matrix[r, c_index] = agg.get((key_head[0], key_head[1], key_head[2], cls), 0.0)

    c_norm = _minmax_normalize(c_matrix)
    return a_matrix, m_matrix, b_matrix, c_norm, comb3


def _python_reference_fisa(
    base: List[List[float]],
    c_norm: np.ndarray,
    sample: List[float],
    n_classes: int,
    comb3: List[Tuple[int, int, int]],
) -> Tuple[int, float]:
    n_rows = len(base)
    n_features = len(sample)
    cols = _combination(3, n_features)
    c_dict = {i: [0.0] * cols for i in range(1, n_classes + 1)}

    for comb_idx, (a, b, c) in enumerate(comb3):
        for r in range(n_rows - 1):  # keep legacy behavior
            if base[r][a] == sample[a] and base[r][b] == sample[b] and base[r][c] == sample[c]:
                label = int(base[r][-1])
                if 1 <= label <= n_classes:
                    c_col = (label - 1) * cols + comb_idx
                    if c_col < c_norm.shape[1]:
                        c_dict[label][comb_idx] = float(c_norm[r, c_col])

    d_dict = {}
    for label in range(1, n_classes + 1):
        vec = c_dict[label]
        d_dict[label] = max(vec) + min(vec)

    best = max(d_dict, key=d_dict.get)
    max_d = d_dict[best]
    sum_d = sum(d_dict.values())
    conf = (max_d / sum_d) if sum_d > 0 else 0.0
    return int(best), float(conf)


def _accuracy(pred: List[int], true: List[int]) -> float:
    if not true:
        return 0.0
    correct = sum(1 for p, t in zip(pred, true) if p == t)
    return correct / len(true) * 100.0


def _make_data(n_samples: int, n_features: int, n_classes: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        feat = [float(rng.randint(1, 8)) for _ in range(n_features)]
        score = int(sum(feat) + feat[0] * 2 + feat[-1])
        label = float((score % n_classes) + 1)
        rows.append(feat + [label])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--module-dir", default="source", choices=["source", "gpu", "auto"])
    parser.add_argument("--samples", type=int, default=420)
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    _configure_windows_dll()
    _configure_paths(args.module_dir)
    import fisa_module  # noqa: PLC0415

    data = _make_data(args.samples, args.features, args.classes, args.seed)
    split = int(len(data) * (1.0 - args.test_ratio))
    split = max(2, min(split, len(data) - 2))
    train_data = data[:split]
    test_data = data[split:]
    y_true = [int(r[-1]) for r in test_data]
    test_inputs = [row[:-1] for row in test_data]

    gpu_compiled = bool(getattr(fisa_module, "GPU_COMPILED", False))
    gpu_available = bool(fisa_module.is_gpu_available() if hasattr(fisa_module, "is_gpu_available") else False)
    run_gpu = gpu_compiled and gpu_available and args.backend != "cpu"
    if args.backend == "gpu":
        run_gpu = gpu_compiled and gpu_available

    py_t0 = time.perf_counter()
    py_A, py_M, py_B, py_C, py_comb3 = _python_reference_abcm(train_data, args.classes)
    py_train_ms = (time.perf_counter() - py_t0) * 1000.0

    py_t1 = time.perf_counter()
    py_pred = []
    for row in test_data:
        cls, _conf = _python_reference_fisa(train_data, py_C, row[:-1], args.classes, py_comb3)
        py_pred.append(int(cls))
    py_infer_ms = (time.perf_counter() - py_t1) * 1000.0
    py_acc = _accuracy(py_pred, y_true)

    fkg_cpu = fisa_module.fkg.FKG()
    fkg_cpu.set_use_gpu(False)
    cpu_t0 = time.perf_counter()
    fkg_cpu.train(train_data, args.classes)
    cpu_train_ms = (time.perf_counter() - cpu_t0) * 1000.0

    cpu_t1 = time.perf_counter()
    if hasattr(fkg_cpu, "predict_batch"):
        cpu_pred = [int(v) for v in fkg_cpu.predict_batch(test_inputs)]
    else:
        cpu_pred = [int(fkg_cpu.predict(row[:-1])[0]) for row in test_data]
    cpu_infer_ms = (time.perf_counter() - cpu_t1) * 1000.0
    cpu_acc = _accuracy(cpu_pred, y_true)

    cpu_A = np.asarray(fkg_cpu.get_A(), dtype=np.float64)
    cpu_M = np.asarray(fkg_cpu.get_M(), dtype=np.float64)
    cpu_B = np.asarray(fkg_cpu.get_B(), dtype=np.float64)
    cpu_C = np.asarray(fkg_cpu.get_C(), dtype=np.float64)

    matrix_diff_cpu = {
        "A_max_abs_diff": float(np.max(np.abs(py_A - cpu_A))) if py_A.size else 0.0,
        "M_max_abs_diff": float(np.max(np.abs(py_M - cpu_M))) if py_M.size else 0.0,
        "B_max_abs_diff": float(np.max(np.abs(py_B - cpu_B))) if py_B.size else 0.0,
        "C_max_abs_diff": float(np.max(np.abs(py_C - cpu_C))) if py_C.size else 0.0,
    }

    gpu_report = None
    if run_gpu:
        fkg_gpu = fisa_module.fkg.FKG()
        fkg_gpu.set_use_gpu(True)

        gpu_t0 = time.perf_counter()
        fkg_gpu.train(train_data, args.classes)
        gpu_train_ms = (time.perf_counter() - gpu_t0) * 1000.0

        gpu_t1 = time.perf_counter()
        if hasattr(fkg_gpu, "predict_batch"):
            gpu_pred = [int(v) for v in fkg_gpu.predict_batch(test_inputs)]
        else:
            gpu_pred = [int(fkg_gpu.predict(row[:-1])[0]) for row in test_data]
        gpu_infer_ms = (time.perf_counter() - gpu_t1) * 1000.0
        gpu_acc = _accuracy(gpu_pred, y_true)

        gpu_A = np.asarray(fkg_gpu.get_A(), dtype=np.float64)
        gpu_M = np.asarray(fkg_gpu.get_M(), dtype=np.float64)
        gpu_B = np.asarray(fkg_gpu.get_B(), dtype=np.float64)
        gpu_C = np.asarray(fkg_gpu.get_C(), dtype=np.float64)

        gpu_report = {
            "train_ms": gpu_train_ms,
            "infer_ms": gpu_infer_ms,
            "accuracy_pct": gpu_acc,
            "speedup_vs_cpp_cpu_train": (cpu_train_ms / gpu_train_ms) if gpu_train_ms > 0 else 0.0,
            "speedup_vs_python_train": (py_train_ms / gpu_train_ms) if gpu_train_ms > 0 else 0.0,
            "pred_match_vs_cpp_cpu_pct": _accuracy(gpu_pred, cpu_pred),
            "matrix_diff_vs_cpp_cpu": {
                "A_max_abs_diff": float(np.max(np.abs(gpu_A - cpu_A))) if gpu_A.size else 0.0,
                "M_max_abs_diff": float(np.max(np.abs(gpu_M - cpu_M))) if gpu_M.size else 0.0,
                "B_max_abs_diff": float(np.max(np.abs(gpu_B - cpu_B))) if gpu_B.size else 0.0,
                "C_max_abs_diff": float(np.max(np.abs(gpu_C - cpu_C))) if gpu_C.size else 0.0,
            },
        }

    report = {
        "gpu_compiled": gpu_compiled,
        "gpu_available": gpu_available,
        "samples": args.samples,
        "features": args.features,
        "classes": args.classes,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "python_ref": {
            "train_ms": py_train_ms,
            "infer_ms": py_infer_ms,
            "accuracy_pct": py_acc,
        },
        "cpp_cpu": {
            "train_ms": cpu_train_ms,
            "infer_ms": cpu_infer_ms,
            "accuracy_pct": cpu_acc,
            "speedup_vs_python_train": (py_train_ms / cpu_train_ms) if cpu_train_ms > 0 else 0.0,
            "matrix_diff_vs_python": matrix_diff_cpu,
        },
        "cpp_gpu": gpu_report,
    }

    print("=" * 80)
    print("FKG Python vs C++ CPU vs CUDA")
    print("=" * 80)
    print(json.dumps(report, indent=2))
    print("=" * 80)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[INFO] Saved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
