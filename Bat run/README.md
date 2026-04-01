# Bat Run Scripts

This folder contains the main Windows `.bat` scripts for building and testing FIS/FKG CPU-CUDA flows.

## Build scripts

- `Build_FKG_CUDA.bat`
  - Build `fisa_module` with CUDA backend.
  - Supports `--fallback-cpu` when CUDA/MSVC is unavailable.
- `Build_FISA_CUDA.bat`
  - Alias wrapper to CUDA build flow.

## Test scripts

- `Test_Backend_GPU_CPU.bat [backend] [module_dir]`
  - Check runtime backend resolution (`auto|cpu|gpu`).
- `Test_FKG_Python_vs_CPP_CUDA.bat [backend] [module_dir]`
  - Compare pure Python vs C++ CPU vs CUDA for FKG.
- `Test_FIS_FKG_Full_Flow.bat [dataset] [module_dir] [bins]`
  - Run full FIS + FKG CPU/GPU benchmark.
  - `dataset`: `icta | feature_selection | both`
  - default: `both`
- `Run_Full_GPU_Validation.bat`
  - Full validation pipeline (build + smoke + consistency + benchmark).

## Quick start

```bat
Bat run\Build_FKG_CUDA.bat --fallback-cpu
Bat run\Test_Backend_GPU_CPU.bat gpu source
Bat run\Test_FKG_Python_vs_CPP_CUDA.bat auto source
Bat run\Test_FIS_FKG_Full_Flow.bat both source 6
Bat run\Run_Full_GPU_Validation.bat
```

