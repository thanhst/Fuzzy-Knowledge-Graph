#!/usr/bin/env python3
"""
Quick consistency checks for FKG matrices and predictions.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np


def _setup_imports() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_dir = os.path.join(root, "Source")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        candidates = []
        cuda = os.environ.get("CUDA_PATH")
        if cuda:
            candidates.append(os.path.join(cuda, "bin"))
        candidates.append(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
        candidates.append(os.path.dirname(sys.executable))
        candidates.append(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"))
        for p in candidates:
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except OSError:
                    pass


def _make_data(n: int = 160, features: int = 6, classes: int = 3):
    rng = random.Random(101)
    data = []
    for _ in range(n):
        feat = [float(rng.randint(1, 7)) for _ in range(features)]
        label = float((int(sum(feat) + feat[0] * 3 + feat[-1]) % classes) + 1)
        data.append(feat + [label])
    return data


def main() -> int:
    _setup_imports()
    import fisa_module  # noqa: PLC0415

    train_data = _make_data()

    fkg_cpu = fisa_module.fkg.FKG()
    fkg_cpu.set_use_gpu(False)
    fkg_cpu.train(train_data, 3)

    cpu_A = np.asarray(fkg_cpu.get_A(), dtype=np.float64)
    cpu_M = np.asarray(fkg_cpu.get_M(), dtype=np.float64)
    cpu_B = np.asarray(fkg_cpu.get_B(), dtype=np.float64)
    cpu_C = np.asarray(fkg_cpu.get_C(), dtype=np.float64)

    assert cpu_A.ndim == 2 and cpu_A.shape[0] == len(train_data)
    assert cpu_M.ndim == 2 and cpu_M.shape[0] == len(train_data)
    assert cpu_B.ndim == 2 and cpu_B.shape[0] == len(train_data)
    assert cpu_C.ndim == 2 and cpu_C.shape[0] == len(train_data)

    gpu_ok = bool(getattr(fisa_module, "GPU_COMPILED", False)) and bool(fisa_module.is_gpu_available())
    if gpu_ok:
        fkg_gpu = fisa_module.fkg.FKG()
        fkg_gpu.set_use_gpu(True)
        fkg_gpu.train(train_data, 3)
        gpu_A = np.asarray(fkg_gpu.get_A(), dtype=np.float64)
        gpu_M = np.asarray(fkg_gpu.get_M(), dtype=np.float64)
        gpu_B = np.asarray(fkg_gpu.get_B(), dtype=np.float64)
        gpu_C = np.asarray(fkg_gpu.get_C(), dtype=np.float64)

        assert np.max(np.abs(cpu_A - gpu_A)) < 1e-9
        assert np.max(np.abs(cpu_M - gpu_M)) < 1e-9
        assert np.max(np.abs(cpu_B - gpu_B)) < 1e-9
        assert np.max(np.abs(cpu_C - gpu_C)) < 1e-9

        sample = train_data[0][:-1]
        cls_cpu, _ = fkg_cpu.predict(sample)
        cls_gpu, _ = fkg_gpu.predict(sample)
        assert int(cls_cpu) == int(cls_gpu)

    print("PASS: matrix consistency checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
