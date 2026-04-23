# Wheel Release Guide (CPU/GPU)

## 1) Su that can biet truoc

- Khong the co **mot wheel duy nhat** chay tren moi OS/Python/gpu.
- Wheel Python phu thuoc:
  - He dieu hanh + kien truc (`win_amd64`, `manylinux_x86_64`, ...)
  - Version Python (`cp310`, `cp311`, `cp313`, ...)
- GPU wheel CUDA chi chay tren NVIDIA + runtime CUDA tuong thich.

## 2) Cach phat hanh de "may nao cung cai duoc"

Ban phat hanh **nhieu wheel** theo matrix:

- CPU wheels:
  - Windows (cp310/cp311/cp312/cp313)
  - Linux (cp310/cp311/cp312/cp313)
- GPU wheels (Windows CUDA):
  - Tach theo CUDA major (vi du `cuda12`)
  - Van tach theo version Python

Client `pip` se tu chon wheel phu hop voi may.

## 3) Build wheel tai local

### CPU
```bat
GPU\bat\Build_Wheel.bat cpu
```

### GPU (CUDA)
```bat
GPU\bat\Build_Wheel.bat gpu
```

File wheel se nam trong:
- `dist/wheels/cpu/*.whl`
- `dist/wheels/gpu/*.whl`

## 4) Khuyen nghi khi push GitHub Release

- Upload day du wheel artifacts cho tung target.
- Dat ten release ro backend:
  - `fisa_module-...-cpu-...whl`
  - `fisa_module-...-gpu-cuda12-...whl`
- Dinh kem release note ghi ro:
  - Python versions ho tro
  - CUDA runtime can co (neu la GPU wheel)
  - Fallback CPU neu khong co GPU
