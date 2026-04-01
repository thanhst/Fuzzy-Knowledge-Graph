# Bat run

Thư mục này chứa các script `.bat` chính để build/test FKG-FIS (CPU/CUDA) theo pipeline mới.

## Scripts

- `Build_FKG_CUDA.bat`
  - Build module `fisa_module` với CUDA (NVIDIA + MSVC).
  - Có hỗ trợ `--fallback-cpu` để tự chuyển qua build CPU nếu máy thiếu CUDA/MSVC.
- `Build_FISA_CUDA.bat`
  - Alias gọi `Build_FKG_CUDA.bat`.
- `Test_Backend_GPU_CPU.bat [backend] [module_dir]`
  - Kiểm tra runtime backend (`auto|cpu|gpu`).
- `Test_FKG_Python_vs_CPP_CUDA.bat [backend] [module_dir]`
  - So sánh Python thuần vs C++ CPU vs CUDA về thời gian, accuracy, sai khác ma trận A/B/C/M.
- `Run_Full_GPU_Validation.bat`
  - Chạy full pipeline: build, smoke test, consistency test, benchmark.

## Ví dụ chạy nhanh

```bat
Bat run\Build_FKG_CUDA.bat --fallback-cpu
Bat run\Test_Backend_GPU_CPU.bat gpu source
Bat run\Test_FKG_Python_vs_CPP_CUDA.bat auto source
Bat run\Run_Full_GPU_Validation.bat
```
