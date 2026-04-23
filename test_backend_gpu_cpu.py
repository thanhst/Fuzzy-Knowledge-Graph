#!/usr/bin/env python
"""
Quick runtime check for backend selection (CPU/GPU) in fisa_module.

Usage:
    python test_backend_gpu_cpu.py --backend auto
    python test_backend_gpu_cpu.py --backend cpu
    python test_backend_gpu_cpu.py --backend gpu
"""

import argparse
import os
import sys
import time
from typing import List


def resolve_paths(module_dir: str) -> str:
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

    dll_dirs = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dll_dirs.append(os.path.join(cuda_path, "bin"))
    dll_dirs.append(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")

    py_dir = os.path.dirname(sys.executable)
    if py_dir:
        dll_dirs.append(py_dir)

    system_root = os.environ.get("SystemRoot")
    if system_root:
        dll_dirs.append(os.path.join(system_root, "System32"))

    handles: List[object] = []
    for path in dll_dirs:
        if os.path.isdir(path):
            try:
                handles.append(os.add_dll_directory(path))
            except OSError:
                pass
    return handles


def choose_gpu(backend: str, gpu_compiled: bool, gpu_available: bool) -> bool:
    if backend == "cpu":
        return False

    if backend == "gpu":
        if not gpu_compiled:
            print("[INFO] Yeu cau GPU nhung module duoc build khong co ho tro GPU.")
            print("[INFO] Dang su dung CPU.")
            return False
        if not gpu_available:
            print("[INFO] Yeu cau GPU nhung may khong co GPU kha dung.")
            print("[INFO] Dang su dung CPU.")
            return False
        return True

    # auto
    return gpu_compiled and gpu_available


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Backend request mode",
    )
    parser.add_argument(
        "--module-dir",
        default="auto",
        choices=["auto", "source", "gpu"],
        help="Where to prioritize loading fisa_module from",
    )
    args = parser.parse_args()

    source_dir = resolve_paths(args.module_dir)
    print(f"[INFO] Source dir: {source_dir}")
    _dll_handles = configure_windows_dll_dirs()

    import_start = time.perf_counter()
    try:
        import fisa_module
    except Exception as exc:
        print("[ERROR] Khong import duoc fisa_module.")
        print(f"[ERROR] {exc}")
        return 1
    import_ms = (time.perf_counter() - import_start) * 1000.0

    print(f"[INFO] Module path   : {getattr(fisa_module, '__file__', '<unknown>')}")
    print(f"[INFO] import time    : {import_ms:.3f} ms")

    if hasattr(fisa_module, "GPU_COMPILED"):
        gpu_compiled = bool(fisa_module.GPU_COMPILED)
    elif hasattr(fisa_module, "is_gpu_compiled"):
        gpu_compiled = bool(fisa_module.is_gpu_compiled())
    else:
        gpu_compiled = False

    if hasattr(fisa_module, "is_gpu_available"):
        gpu_available = bool(fisa_module.is_gpu_available())
    else:
        gpu_available = False

    use_gpu = choose_gpu(args.backend, gpu_compiled, gpu_available)
    if hasattr(fisa_module, "resolve_backend"):
        effective_backend = fisa_module.resolve_backend(use_gpu)
    else:
        effective_backend = "gpu" if use_gpu else "cpu"

    print(f"[INFO] backend request : {args.backend}")
    print(f"[INFO] GPU compiled    : {gpu_compiled}")
    print(f"[INFO] GPU available   : {gpu_available}")
    print(f"[INFO] backend used    : {effective_backend}")

    if not hasattr(fisa_module, "fkg") or not hasattr(fisa_module.fkg, "FKG"):
        print("[ERROR] Module fisa_module khong co submodule fkg/FKG.")
        print("[ERROR] Hay build lai module moi.")
        return 1

    base = [
        [1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0],
        [2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 1.0, 3.0, 3.0, 2.0, 2.0],
    ]
    sample = [1.0, 2.0, 1.0, 2.0, 3.0, 1.0]

    fkg = fisa_module.fkg.FKG()
    if not hasattr(fkg, "set_use_gpu"):
        print("[ERROR] fisa_module hien tai chua co API set_use_gpu.")
        print("[ERROR] Hay build lai theo pipeline moi.")
        return 1

    train_start = time.perf_counter()
    fkg.set_use_gpu(use_gpu)
    fkg.train(base, 2)
    train_ms = (time.perf_counter() - train_start) * 1000.0

    predict_start = time.perf_counter()
    predicted_class, confidence = fkg.predict(sample)
    predict_ms = (time.perf_counter() - predict_start) * 1000.0

    print(f"[INFO] fkg.get_use_gpu : {fkg.get_use_gpu()}")
    print(f"[INFO] fkg.is_using_gpu: {fkg.is_using_gpu()}")
    print(f"[INFO] train time      : {train_ms:.3f} ms")
    print(f"[INFO] predict time    : {predict_ms:.3f} ms")
    print(f"[INFO] predict class   : {predicted_class}")
    print(f"[INFO] confidence      : {confidence:.6f}")
    print("[OK] Backend test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
