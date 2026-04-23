"""
FKG/FIS smoke test aligned with current fisa_module API.

This test prints:
- requested/effective backend (CPU or GPU)
- train/inference runtime
- basic result quality
"""

import argparse
import os
import random
import sys
import time


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _configure_paths(module_dir: str) -> None:
    root = _repo_root()
    source_dir = os.path.join(root, "Source")
    gpu_dir = os.path.join(root, "GPU", "Source")

    if module_dir == "source":
        candidates = [source_dir]
    elif module_dir == "gpu":
        candidates = [gpu_dir, source_dir]
    else:
        candidates = [gpu_dir, source_dir]

    for path in reversed(candidates):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _configure_windows_dll() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates = []
    cuda = os.environ.get("CUDA_PATH")
    if cuda:
        candidates.append(os.path.join(cuda, "bin"))
    candidates.append(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    candidates.append(os.path.dirname(sys.executable))
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.append(os.path.join(system_root, "System32"))

    for path in candidates:
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass


def _select_backend(requested: str, gpu_compiled: bool, gpu_available: bool) -> bool:
    if requested == "cpu":
        return False
    if requested == "gpu":
        return gpu_compiled and gpu_available
    return gpu_compiled and gpu_available


def _make_data(n_samples: int, n_features: int, n_classes: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        features = [float(rng.randint(1, 6)) for _ in range(n_features)]
        score = int(sum(features) + features[0] + 2 * features[-1])
        label = float((score % n_classes) + 1)
        rows.append(features + [label])
    return rows


def run_smoke(backend: str, module_dir: str) -> int:
    _configure_paths(module_dir)
    _configure_windows_dll()

    import fisa_module

    gpu_compiled = bool(getattr(fisa_module, "GPU_COMPILED", False))
    gpu_available = bool(
        fisa_module.is_gpu_available() if hasattr(fisa_module, "is_gpu_available") else False
    )
    use_gpu = _select_backend(backend, gpu_compiled, gpu_available)
    effective_backend = (
        fisa_module.resolve_backend(use_gpu)
        if hasattr(fisa_module, "resolve_backend")
        else ("gpu" if use_gpu else "cpu")
    )

    data = _make_data(n_samples=140, n_features=6, n_classes=3, seed=123)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    print("=" * 60)
    print("Running FKG/FIS smoke test")
    print("=" * 60)
    print(f"[INFO] module path      : {getattr(fisa_module, '__file__', '<unknown>')}")
    print(f"[INFO] backend request  : {backend}")
    print(f"[INFO] gpu compiled     : {gpu_compiled}")
    print(f"[INFO] gpu available    : {gpu_available}")
    print(f"[INFO] backend used     : {effective_backend}")

    fkg = fisa_module.fkg.FKG()
    fkg.set_use_gpu(use_gpu)

    t0 = time.perf_counter()
    fkg.train(train_data, 3)
    train_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    predicted = []
    actual = []
    for row in test_data:
        pred, _conf = fkg.predict(row[:-1])
        predicted.append(int(pred))
        actual.append(int(row[-1]))
    infer_ms = (time.perf_counter() - t1) * 1000.0

    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    acc = (correct / len(actual) * 100.0) if actual else 0.0

    # FIS smoke: train + one prediction to validate API path.
    fis = fisa_module.fis.FIS()
    fis.set_use_gpu(use_gpu)
    fis.train(train_data)
    fis_pred = fis.predict(test_data[0][:-1]) if test_data else -1

    print(f"[INFO] fkg.is_using_gpu : {fkg.is_using_gpu()}")
    print(f"[INFO] fkg train time   : {train_ms:.3f} ms")
    print(f"[INFO] fkg infer time   : {infer_ms:.3f} ms")
    print(f"[INFO] fkg accuracy     : {acc:.2f}% ({correct}/{len(actual)})")
    print(f"[INFO] fis sample pred  : {fis_pred}")
    print("=" * 60)

    # Smoke threshold: only require code path to execute and produce sane labels.
    ok = len(predicted) == len(actual) and len(predicted) > 0
    print(f"Results: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--module-dir", default="auto", choices=["auto", "source", "gpu"])
    args = parser.parse_args()
    raise SystemExit(run_smoke(args.backend, args.module_dir))
