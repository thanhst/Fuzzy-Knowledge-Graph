# FIS-FKG CUDA Progress Report

Updated: **2026-04-02**

## 1) What was implemented

This update completes real CUDA execution for both **FKG** and **FIS** in the `Source/` codebase.

- FKG CUDA kernels were already implemented and optimized (including cached batch inference).
- FIS now has real CUDA kernels for:
  - 1D FCM membership update
  - center update
  - objective computation
  - GPU-based rule generation pipeline

## 2) New FIS CUDA components

Added files:

- `Source/Include/FIS_CUDA_Kernels.h`
- `Source/Src/FIS_CUDA_Kernels.cu`

Integrated into:

- `Source/Src/FIS.cpp`
  - `FIS::fcmGPU(...)` now calls CUDA kernel wrapper (`fcm1DGPU`).
  - `FIS::ruleGenerateGPU(...)` now calls CUDA pipeline wrapper (`ruleGenerateFIS_GPU`).
  - Safe CPU fallback remains active when CUDA fails or is unavailable.

Build updates:

- `Source/CMakeLists.txt` includes `Src/FIS_CUDA_Kernels.cu` and `Include/FIS_CUDA_Kernels.h`.
- `Source/CMakeLists_CUDA.txt` also includes `Src/FIS_CUDA_Kernels.cu`.

## 3) Full-flow benchmark script (FIS + FKG)

Added:

- `Source/tests/test_fis_fkg_full_flow_gpu_cpu.py`

This script runs both **CPU and GPU** flows for:

- FIS train/infer
- FKG train/infer

Datasets:

- ICTA
- Diabetic Retinopathy Feature FT Selection

Convenience batch files:

- `Bat run/Test_FIS_FKG_Full_Flow.bat`
- `Test_FIS_FKG_Full_Flow.bat`

## 4) Latest benchmark results (cold + warm)

Command used:

```powershell
python Source/tests/test_fis_fkg_full_flow_gpu_cpu.py --dataset both --module-dir source --bins 6 --warm-repeats 1 --test-ratio 0.3 --seed 42 --out-json result/full_flow_benchmark_cold_warm.json
```

### ICTA (train=537, test=231)

- FIS CPU:
  - train cold/warm: `74.38 / 73.76 ms`
  - infer cold/warm: `1.28 / 1.26 ms`
  - accuracy: `64.94%`
- FIS GPU:
  - train cold/warm: `1412.32 / 496.83 ms`
  - infer cold/warm: `1.22 / 1.01 ms`
  - accuracy: `64.94%`
- FKG CPU:
  - train cold/warm: `6.85 / 7.40 ms`
  - infer cold/warm: `25.98 / 25.71 ms`
  - accuracy: `65.37%`
- FKG GPU:
  - train cold/warm: `13.54 / 8.13 ms`
  - infer cold/warm: `8.09 / 5.99 ms`
  - accuracy: `65.37%`
- CPU/GPU prediction match:
  - FIS: `100%`
  - FKG: `100%`

### Feature Selection dataset (train=21274, test=9118)

- FIS CPU:
  - train cold/warm: `1205.90 / 1150.15 ms`
  - infer cold/warm: `77.86 / 84.49 ms`
  - accuracy: `91.42%`
- FIS GPU:
  - train cold/warm: `17966.95 / 17648.54 ms`
  - infer cold/warm: `71.93 / 152.90 ms`
  - accuracy: `91.42%`
- FKG CPU:
  - train cold/warm: `3645.51 / 5090.66 ms`
  - infer cold/warm: `75914.88 / 76147.53 ms`
  - accuracy: `59.71%`
- FKG GPU:
  - train cold/warm: `55889.18 / 56937.34 ms`
  - infer cold/warm: `11812.67 / 11177.02 ms`
  - accuracy: `59.71%`
- CPU/GPU prediction match:
  - FIS: `100%`
  - FKG: `100%`

Benchmark report file:

- `result/full_flow_benchmark_cold_warm.json`

## 5) Notes

- FIS CUDA is now functional and numerically consistent with CPU.
- Benchmark now reports both cold and warm timings for train/infer.
- On current datasets, FIS GPU train is slower than CPU due iterative kernel-launch overhead and cold-start cost.
- FKG GPU remains clearly beneficial for inference throughput in both datasets.
- Further FIS GPU optimization should focus on:
  - reducing per-iteration kernel launch overhead,
  - fusing FCM kernels,
  - keeping convergence checks on device when possible.
