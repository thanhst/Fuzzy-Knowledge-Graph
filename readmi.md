# FKG CUDA Progress Report

Updated: **2026-04-02**

## 1) Scope

This update focuses on real CUDA execution for FKG and practical GPU optimization for train/infer flow in FIS-FKG pipeline.

## 2) Completed updates

- Implemented real CUDA kernels in `Source/Src/FKG_CUDA_Kernels.cu`.
- Enabled CUDA build path in `Source/CMakeLists.txt`.
- Added matrix and GPU APIs in C++ and Python bindings.
- Added bat scripts under `Bat run/` for build and test automation.
- Added comparison and consistency tests:
  - `Source/tests/test_fkg_python_vs_cpp_cuda.py`
  - `Source/tests/test_fkg_matrix_consistency.py`
  - `Source/tests/test_icta_gpu.py`

## 3) New optimization in this round

### 3.1 GPU inference cache (major)

Added persistent device cache for inference:

- `createFisaDeviceCache(...)`
- `destroyFisaDeviceCache(...)`
- `fisaGPUWithCache(...)`
- `fisaBatchGPUWithCache(...)`

Cache keeps `base`, `C`, and `comb3` on GPU memory across predictions.

### 3.2 Batch CUDA kernel for FISA

Added `KernelFisaDBatch` so multiple samples are inferred in one GPU launch.
This removes repeated per-sample launch/memory overhead.

### 3.3 FKG runtime path updated

`Source/Src/FKG.cpp` now:

- builds/invalidates cache on train lifecycle,
- uses cached GPU path for `predict(...)`,
- uses cached batch path for `predictBatch(...)`,
- exposes `predictBatchWithConfidence(...)`.

### 3.4 Python API and ICTA test updated

- `Source/Python/bindings.cpp` exposes `predict_batch_with_confidence`.
- `Source/tests/test_icta_gpu.py` now uses batch inference when available.

## 4) Measured results

### ICTA script (`Source/tests/test_icta_gpu.py`, train=537, test=231)

- GPU train time: `~1252.1 ms` (cold process, includes CUDA init)
- GPU infer total: `~11.7 ms`
- CPU train time: `~9.0 ms`
- CPU infer total: `~29.5 ms`
- Accuracy: same (`65.80%` in this run)

### Same process repeated timing (GPU)

- train[0]: `~878.5 ms` (cold init)
- train[1]: `~10.3 ms`
- train[2]: `~9.9 ms`
- batch infer: `~8-11 ms` for 231 samples
- old per-sample loop infer: `~529-561 ms` for 231 samples

Conclusion: inference bottleneck is fixed; cold train still dominated by first CUDA context initialization.

## 5) Updated files (this optimization round)

- `Source/Src/FKG_CUDA_Kernels.cu`
- `Source/Include/FKG_CUDA_Kernels.h`
- `Source/Src/FKG.cpp`
- `Source/Include/FKG.h`
- `Source/Python/bindings.cpp`
- `Source/tests/test_icta_gpu.py`
- `Source/tests/test_fkg_python_vs_cpp_cuda.py`

## 6) Remaining high-impact tasks

- Add optional warmup stage before benchmark timing to separate cold init cost.
- Reuse input/output device buffers for batch inference to reduce remaining allocations.
- Add CUDA streams for overlap between H2D/D2H copy and compute.
- Consider compact integer encoding for discrete data to reduce bandwidth.
