# FISA Module (FKG + FIS) - CPU/CUDA

`fisa_module` is a Python extension (C++/CUDA) that provides:

- `fkg`: Fuzzy Knowledge Graph (A/M/B/C, FISA infer, CPU/GPU benchmark)
- `fis`: Fuzzy Inference System (CPU/GPU FCM + rule generation)

## Main points

- Real CUDA kernels implemented in `Source/Src/model/FKG_CUDA_Kernels.cu`.
- Real CUDA kernels implemented in `Source/Src/model/FIS_CUDA_Kernels.cu`.
- End-to-end GPU matrix pipeline: `calculateABCM_GPU(...)`.
- Cached GPU inference path to remove repeated setup overhead:
  - `createFisaDeviceCache(...)`
  - `fisaGPUWithCache(...)`
  - `fisaBatchGPUWithCache(...)`
- Batch inference API now available in Python:
  - `FKG.predict_batch_with_confidence(inputs)`

## Build

### Quick build by bat (recommended)

From project root:

```bat
Bat run\Build_FKG_CUDA.bat --fallback-cpu
```

CPU-only build:

```bat
Build_FISA_CPU.bat
```

### Build with CMake

CPU:

```powershell
cd Source
cmake -S . -B build_cpu -DUSE_CUDA=OFF -DUSE_GPU=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu --config Release -j 8
```

CUDA:

```powershell
cd Source
cmake -S . -B build_cuda -DUSE_CUDA=ON -DUSE_GPU=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda --config Release -j 8
```

## Python quick usage

```python
import fisa_module

fkg = fisa_module.fkg.FKG()
fkg.set_use_gpu(True)
fkg.train(train_data, n_classes)

# Single sample
pred_class, conf = fkg.predict(sample_features)

# Batch (recommended for GPU)
results = fkg.predict_batch_with_confidence(test_inputs)
# results: list[(class_id, confidence)]

A = fkg.get_A()
M = fkg.get_M()
B = fkg.get_B()
C = fkg.get_C()
```

## Tests

Backend check:

```bat
Bat run\Test_Backend_GPU_CPU.bat [backend] [module_dir]
```

Python vs C++ CPU vs CUDA:

```bat
Bat run\Test_FKG_Python_vs_CPP_CUDA.bat [backend] [module_dir]
```

ICTA benchmark:

```powershell
python Source/tests/test_icta_gpu.py --backend gpu --module-dir source
```

Full flow benchmark (FIS + FKG, CPU vs GPU, ICTA + Feature Selection):

```powershell
python Source/tests/test_fis_fkg_full_flow_gpu_cpu.py --dataset both --module-dir source --warm-repeats 1
```

Matrix consistency:

```powershell
python Source/tests/test_fkg_matrix_consistency.py
```

## Notes on GPU timing

- First GPU train in a new process includes CUDA context initialization and can be much slower.
- Use `--warm-repeats` to report both cold and warm timings in one benchmark report.
- For realistic throughput, compare warm timings and batch inference API behavior.
- On current ICTA setup, GPU batch infer is significantly faster than CPU infer.
