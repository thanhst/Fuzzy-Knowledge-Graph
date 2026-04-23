#!/usr/bin/env python
"""
Small FKG smoke test with backend/timing report.

Usage:
    python test_fkg_small.py --backend auto
    python test_fkg_small.py --backend gpu --module-dir gpu
"""

import argparse
import os
import random
import sys
import time
from typing import List, Tuple


def configure_paths(module_dir: str) -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(repo_root, "Source")
    gpu_source_dir = os.path.join(repo_root, "GPU", "Source")

    if module_dir == "source":
        candidates = [source_dir]
    elif module_dir == "gpu":
        candidates = [gpu_source_dir, source_dir]
    else:
        candidates = [gpu_source_dir, source_dir]

    for path in reversed(candidates):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    for path in candidates:
        if os.path.isdir(path):
            return path
    return source_dir


def configure_windows_dll_dirs() -> List[object]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    dirs = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dirs.append(os.path.join(cuda_path, "bin"))
    dirs.append(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    dirs.append(os.path.dirname(sys.executable))
    system_root = os.environ.get("SystemRoot")
    if system_root:
        dirs.append(os.path.join(system_root, "System32"))

    handles = []
    for path in dirs:
        if os.path.isdir(path):
            try:
                handles.append(os.add_dll_directory(path))
            except OSError:
                pass
    return handles


def pick_backend(requested: str, gpu_compiled: bool, gpu_available: bool) -> bool:
    if requested == "cpu":
        return False
    if requested == "gpu":
        return gpu_compiled and gpu_available
    return gpu_compiled and gpu_available


def make_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    seed: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        features = [float(rng.randint(1, 4)) for _ in range(n_features)]
        # Keep label mapping simple so smoke test has stable/meaningful accuracy.
        score = int(features[0] + 2 * features[1] + features[-1])
        label = float((score % n_classes) + 1)
        rows.append(features + [label])

    split = int(n_samples * 0.8)
    return rows[:split], rows[split:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--module-dir", default="auto", choices=["auto", "source", "gpu"])
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    module_search_root = configure_paths(args.module_dir)
    _dll_handles = configure_windows_dll_dirs()

    print("============================================================")
    print("FKG Small Test")
    print("============================================================")
    print(f"[INFO] module search root : {module_search_root}")
    print(f"[INFO] backend request    : {args.backend}")
    print(f"[INFO] samples/features   : {args.samples}/{args.features}")
    print(f"[INFO] classes            : {args.classes}")

    import_start = time.perf_counter()
    try:
        import fisa_module
    except Exception as exc:
        print("[ERROR] Khong import duoc fisa_module.")
        print(f"[ERROR] {exc}")
        return 1
    import_ms = (time.perf_counter() - import_start) * 1000.0

    gpu_compiled = bool(getattr(fisa_module, "GPU_COMPILED", False))
    gpu_available = bool(
        fisa_module.is_gpu_available() if hasattr(fisa_module, "is_gpu_available") else False
    )
    use_gpu = pick_backend(args.backend, gpu_compiled, gpu_available)
    effective_backend = (
        fisa_module.resolve_backend(use_gpu)
        if hasattr(fisa_module, "resolve_backend")
        else ("gpu" if use_gpu else "cpu")
    )

    train_data, test_data = make_dataset(
        n_samples=args.samples,
        n_features=args.features,
        n_classes=args.classes,
        seed=args.seed,
    )

    fkg = fisa_module.fkg.FKG()
    fkg.set_use_gpu(use_gpu)

    train_start = time.perf_counter()
    fkg.train(train_data, args.classes)
    train_ms = (time.perf_counter() - train_start) * 1000.0

    infer_start = time.perf_counter()
    predicted = []
    actual = []
    confidences = []
    for row in test_data:
        cls, conf = fkg.predict(row[:-1])
        predicted.append(int(cls))
        actual.append(int(row[-1]))
        confidences.append(float(conf))
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    accuracy = (correct / len(actual) * 100.0) if actual else 0.0
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

    print(f"[INFO] module path        : {getattr(fisa_module, '__file__', '<unknown>')}")
    print(f"[INFO] import time        : {import_ms:.3f} ms")
    print(f"[INFO] gpu compiled       : {gpu_compiled}")
    print(f"[INFO] gpu available      : {gpu_available}")
    print(f"[INFO] backend used       : {effective_backend}")
    print(f"[INFO] fkg.is_using_gpu   : {fkg.is_using_gpu()}")
    print(f"[INFO] train time         : {train_ms:.3f} ms")
    print(f"[INFO] inference time     : {infer_ms:.3f} ms")
    print(f"[INFO] test size          : {len(actual)}")
    print(f"[INFO] accuracy           : {accuracy:.2f}% ({correct}/{len(actual)})")
    print(f"[INFO] avg confidence     : {avg_conf:.6f}")
    print(f"[INFO] first 10 predict   : {predicted[:10]}")
    print(f"[INFO] first 10 actual    : {actual[:10]}")
    print("[OK] test_fkg_small completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
