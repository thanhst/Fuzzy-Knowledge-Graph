# FISA Module - C++ Implementation

High-performance C++ implementation of Fuzzy Knowledge Graph (FKG) and Fuzzy Inference System (FIS) with parallel processing support.

## Features

### FKG (Fuzzy Knowledge Graph)
- `calculateA()` - 4-combination attribute matching matrix
- `calculateM()` - Attribute matching with class labels
- `calculateB()` - B matrix computation
- `calculateC()` - Class-based scoring
- `FISA()` - Fuzzy Inference System for FKG
- `minMaxNormalize()` - Min-max normalization
- `gaussianNormalize()` - Gaussian normalization
- `sampling()` - Sampling for FKGS

### FIS (Fuzzy Inference System)
- `fcmFunction()` - Fuzzy C-Means clustering
- `GaussMF()` - Gaussian Membership Function
- `ruleGenerate()` - Rule generation using FCM
- `ruleWeight()` - Rule weight calculation
- `fuzzifyInput()` - Fuzzify input data
- `matchRule()` - Match fuzzified input with rules
- `testFIS()` - FIS inference

## Parallel Processing

The implementation uses OpenMP for parallel processing:
- Parallel matrix operations
- Parallel distance calculations
- Parallel metric computations

## Requirements

- C++17 compatible compiler
- Python 3.7+
- pybind11 >= 2.10.0
- NumPy >= 1.19.0
- OpenMP (optional, for parallel processing)

## Installation

### Using pip
```bash
pip install .
```

### Using CMake directly
```bash
cd Source
mkdir build && cd build
cmake ..
make
```

### Development installation
```bash
pip install -e .[dev]
```

## Wheel Distribution

Ban khong the co 1 wheel duy nhat cho moi may/OS/Python/GPU.
Can phat hanh nhieu wheel theo target.

- Build CPU wheel:
```bat
GPU\bat\Build_Wheel.bat cpu
```

- Build GPU wheel (CUDA):
```bat
GPU\bat\Build_Wheel.bat gpu
```

Output:
- `dist/wheels/cpu/*.whl`
- `dist/wheels/gpu/*.whl`

Chi tiet release xem: `docs/WHEEL_RELEASE.md`.

## Usage

### Python API

```python
import numpy as np
import fisa_module as fisa

# FKG Example
base = np.array([[1, 2, 3, 4, 1], 
                 [1, 2, 3, 5, 2], 
                 [2, 3, 4, 5, 2]], dtype=float)

# Calculate matrices
A = fisa.fkg.calculateA(base)
M = fisa.fkg.calculateM(base)
B = fisa.fkg.calculateB(base, A, M)
C = fisa.fkg.calculateC(base, B, n_classes=2)

# Normalize
C_norm = fisa.fkg.min_max_normalize(C)

# Predict
input_sample = np.array([1, 2, 3, 4], dtype=float)
predicted_class, confidence = fisa.fkg.FISA(base, C_norm, input_sample, n_classes=2)

# FIS Example
sigma_M = [1.0, 1.0, 1.0]
centers = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
rule_list = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
input_data = np.array([0.5, 0.5, 0.5])

predicted = fisa.fis.test_fis(input_data, rule_list, sigma_M, centers)
```

## Performance

The C++ implementation provides significant speedup over pure Python:
- **calculateA**: ~10-50x faster with parallel processing
- **calculateM**: ~5-20x faster
- **FCM**: ~5-15x faster
- **FISA**: ~20-100x faster

## License

MIT License
