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

## 4) Latest benchmark results

Command used:

```powershell
python Source/tests/test_fis_fkg_full_flow_gpu_cpu.py --dataset both --module-dir source --bins 6 --test-ratio 0.3 --seed 42 --out-json result/full_flow_benchmark_compact.json
```

### ICTA (train=537, test=231)

- FIS CPU:
  - train: `74.76 ms`
  - infer: `1.22 ms`
  - accuracy: `64.94%`
- FIS GPU:
  - train: `1426.93 ms` (cold GPU context overhead)
  - infer: `1.27 ms`
  - accuracy: `64.94%`
- FKG CPU:
  - train: `6.66 ms`
  - infer: `25.95 ms`
  - accuracy: `65.37%`
- FKG GPU:
  - train: `13.14 ms`
  - infer: `7.79 ms`
  - accuracy: `65.37%`
- CPU/GPU prediction match:
  - FIS: `100%`
  - FKG: `100%`

### Feature Selection dataset (train=21274, test=9118)

- FIS CPU:
  - train: `1145.00 ms`
  - infer: `73.99 ms`
  - accuracy: `91.42%`
- FIS GPU:
  - train: `18042.80 ms`
  - infer: `122.13 ms`
  - accuracy: `91.42%`
- FKG CPU:
  - train: `3403.68 ms`
  - infer: `75522.03 ms`
  - accuracy: `59.71%`
- FKG GPU:
  - train: `55565.85 ms`
  - infer: `11726.41 ms`
  - accuracy: `59.71%`
- CPU/GPU prediction match:
  - FIS: `100%`
  - FKG: `100%`

## 5) Notes

- FIS CUDA is now functional and numerically consistent with CPU.
- On current datasets, FIS GPU train is slower than CPU due iterative kernel-launch overhead and cold-start cost.
- FKG GPU remains clearly beneficial for inference throughput in both datasets.
- Further FIS GPU optimization should focus on:
  - reducing per-iteration kernel launch overhead,
  - fusing FCM kernels,
  - keeping convergence checks on device when possible.

