# Bao cao tien do cap nhat FKG CUDA + FIS-FKG

Cap nhat: **2026-04-02**

## 1) Tong quan

Da chuyen FKG sang CUDA kernel thuc te va toi uu lai luong infer GPU theo huong batch + cache de giam overhead.

## 2) Noi dung da cap nhat

- Viet CUDA kernel that trong `Source/Src/FKG_CUDA_Kernels.cu` cho cac buoc A/M/B/C/FISA.
- Bat build CUDA trong `Source/CMakeLists.txt`.
- Them API C++/Python cho duong infer nhanh:
  - `createFisaDeviceCache(...)`
  - `destroyFisaDeviceCache(...)`
  - `fisaGPUWithCache(...)`
  - `fisaBatchGPUWithCache(...)`
  - `FKG.predict_batch_with_confidence(...)`
- Cap nhat `Source/Src/FKG.cpp` de:
  - quan ly vong doi GPU cache,
  - predict GPU theo cache,
  - predict batch GPU de tan dung thong luong.
- Sap xep script chay/build vao thu muc `Bat run/`.
- Bo sung test so sanh va test consistency:
  - `Source/tests/test_fkg_python_vs_cpp_cuda.py`
  - `Source/tests/test_fkg_matrix_consistency.py`
  - `Source/tests/test_icta_gpu.py`

## 3) Ket qua do toc do (ICTA)

Cau hinh test:

- Train: 537 mau
- Test: 231 mau
- Dataset: `Source_code/data/ICTA/ICTA.csv`

Retest moi nhat:

- GPU (cold process):
  - Train: `953.422 ms`
  - Infer tong: `10.559 ms` (`0.046 ms/mau`)
- CPU:
  - Train: `39.347 ms`
  - Infer tong: `28.558 ms` (`0.124 ms/mau`)

Warm-run trong cung process GPU:

- `train[0] = 847.008 ms` (co chi phi khoi tao CUDA context)
- `train[1] = 9.100 ms`
- `train[2] = 9.995 ms`
- `infer batch = 8.088 - 10.041 ms` / 231 mau

Ket luan:

- Nut that infer da duoc xu ly (GPU infer nhanh hon CPU ro ret).
- Train GPU lan dau van bi anh huong boi khoi tao CUDA context.

## 4) Huong dan chay nhanh

Build CUDA:

```bat
Bat run\Build_FKG_CUDA.bat --fallback-cpu
```

Test backend CPU/GPU:

```bat
Bat run\Test_Backend_GPU_CPU.bat gpu source
```

Test ICTA:

```powershell
python Source/tests/test_icta_gpu.py --backend gpu --module-dir source --csv Source_code/data/ICTA/ICTA.csv
```

Test so sanh Python vs C++ CPU vs CUDA:

```bat
Bat run\Test_FKG_Python_vs_CPP_CUDA.bat auto source
```

## 5) Ke hoach toi uu tiep

- Them warmup mode trong benchmark de tach cold-start.
- Tai su dung buffer input/output tren GPU cho infer lap.
- Nghien cuu stream + overlap copy/compute cho batch lon.
- Neu can throughput cao hon nua: xem xet toi uu layout du lieu roi rac (int packing).
