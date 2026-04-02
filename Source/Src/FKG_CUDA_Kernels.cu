/**
 * @file FKG_CUDA_Kernels.cu
 * @brief CUDA kernels + host wrappers for FKG.
 */

#if FUZZY_USE_CUDA

#include "FKG_CUDA_Kernels.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace Fuzzy {
namespace CUDA {
namespace {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), count_(0) {}
    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    cudaError_t allocate(size_t count) {
        count_ = count;
        if (count_ == 0) {
            ptr_ = nullptr;
            return cudaSuccess;
        }
        return cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T));
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T* ptr_;
    size_t count_;
};

inline void setError(std::string* error_message, const char* stage, cudaError_t err) {
    if (error_message == nullptr) {
        return;
    }
    std::ostringstream oss;
    oss << stage << ": " << cudaGetErrorString(err);
    *error_message = oss.str();
}

inline int combination(int k, int n) {
    if (k < 0 || n < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n - k) {
        k = n - k;
    }
    int result = 1;
    for (int i = 1; i <= k; ++i) {
        result = (result * (n - k + i)) / i;
    }
    return result;
}

inline std::vector<int3> buildComb3(int num_features) {
    std::vector<int3> comb3;
    comb3.reserve(static_cast<size_t>(combination(3, num_features)));
    for (int a = 0; a < num_features - 2; ++a) {
        for (int b = a + 1; b < num_features - 1; ++b) {
            for (int c = b + 1; c < num_features; ++c) {
                int3 v;
                v.x = a;
                v.y = b;
                v.z = c;
                comb3.push_back(v);
            }
        }
    }
    return comb3;
}

inline std::vector<int4> buildComb4(int num_features) {
    std::vector<int4> comb4;
    comb4.reserve(static_cast<size_t>(combination(4, num_features)));
    for (int a = 0; a < num_features - 3; ++a) {
        for (int b = a + 1; b < num_features - 2; ++b) {
            for (int c = b + 1; c < num_features - 1; ++c) {
                for (int d = c + 1; d < num_features; ++d) {
                    int4 v;
                    v.x = a;
                    v.y = b;
                    v.z = c;
                    v.w = d;
                    comb4.push_back(v);
                }
            }
        }
    }
    return comb4;
}

inline std::vector<double> flattenMatrix(const Matrix& matrix) {
    if (matrix.empty() || matrix[0].empty()) {
        return {};
    }
    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();
    std::vector<double> out(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        std::memcpy(out.data() + r * cols, matrix[r].data(), cols * sizeof(double));
    }
    return out;
}

inline void reshapeMatrix(const std::vector<double>& flat, int rows, int cols, Matrix& matrix) {
    matrix.assign(static_cast<size_t>(rows), std::vector<double>(static_cast<size_t>(cols), 0.0));
    if (rows == 0 || cols == 0) {
        return;
    }
    for (int r = 0; r < rows; ++r) {
        std::memcpy(matrix[r].data(), flat.data() + static_cast<size_t>(r) * static_cast<size_t>(cols),
                    static_cast<size_t>(cols) * sizeof(double));
    }
}

inline bool validateRectangular(const Matrix& matrix, int* rows, int* cols) {
    if (matrix.empty() || matrix[0].empty()) {
        return false;
    }
    const int r = static_cast<int>(matrix.size());
    const int c = static_cast<int>(matrix[0].size());
    for (int i = 1; i < r; ++i) {
        if (static_cast<int>(matrix[i].size()) != c) {
            return false;
        }
    }
    *rows = r;
    *cols = c;
    return true;
}

constexpr int kBlockSize = 256;

struct HostLookupKey {
    int combLabel;
    int v0;
    int v1;
    int v2;

    bool operator==(const HostLookupKey& other) const {
        return combLabel == other.combLabel &&
               v0 == other.v0 &&
               v1 == other.v1 &&
               v2 == other.v2;
    }
};

struct HostLookupKeyHash {
    size_t operator()(const HostLookupKey& key) const {
        size_t h = std::hash<int>{}(key.combLabel);
        h ^= std::hash<int>{}(key.v0) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.v1) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.v2) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct HostLookupEntry {
    int4 key;
    double value;
};

inline int roundToIntSafe(double value) {
    return static_cast<int>(std::llround(value));
}

inline bool isNearlyInteger(double value, double tol = 1e-8) {
    return std::abs(value - static_cast<double>(roundToIntSafe(value))) <= tol;
}

inline bool compareInt4(const int4& a, const int4& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    return a.w < b.w;
}

inline bool buildFisaLookupTable(const Matrix& base, const Matrix& C,
                                 const std::vector<int3>& comb3,
                                 int nClasses, int fullCols,
                                 std::vector<int4>* outKeys,
                                 std::vector<double>* outValues) {
    if (outKeys == nullptr || outValues == nullptr) {
        return false;
    }
    outKeys->clear();
    outValues->clear();

    if (base.empty() || base[0].size() < 2 || C.empty()) {
        return false;
    }

    const int rows = static_cast<int>(base.size());
    const int cols = static_cast<int>(base[0].size());
    const int numComb3 = static_cast<int>(comb3.size());
    if (rows <= 1 || numComb3 <= 0 || nClasses <= 0) {
        return false;
    }

    std::unordered_map<HostLookupKey, double, HostLookupKeyHash> lookup;
    lookup.reserve(static_cast<size_t>(std::max(1024, rows * std::max(1, nClasses))));

    for (int r = 0; r < rows - 1; ++r) {
        const double rawLabel = base[static_cast<size_t>(r)][static_cast<size_t>(cols - 1)];
        if (!isNearlyInteger(rawLabel)) {
            return false;
        }
        const int label = roundToIntSafe(rawLabel);
        if (label < 1 || label > nClasses) {
            continue;
        }

        for (int combIdx = 0; combIdx < numComb3; ++combIdx) {
            const int3 comb = comb3[static_cast<size_t>(combIdx)];
            const double rawV0 = base[static_cast<size_t>(r)][static_cast<size_t>(comb.x)];
            const double rawV1 = base[static_cast<size_t>(r)][static_cast<size_t>(comb.y)];
            const double rawV2 = base[static_cast<size_t>(r)][static_cast<size_t>(comb.z)];
            if (!isNearlyInteger(rawV0) || !isNearlyInteger(rawV1) || !isNearlyInteger(rawV2)) {
                return false;
            }

            const int combLabel = (label - 1) * numComb3 + combIdx;
            if (combLabel < 0 || combLabel >= fullCols) {
                continue;
            }

            const HostLookupKey key{
                combLabel,
                roundToIntSafe(rawV0),
                roundToIntSafe(rawV1),
                roundToIntSafe(rawV2)
            };
            // Keep last-row overwrite behavior to match legacy implementation.
            lookup[key] = C[static_cast<size_t>(r)][static_cast<size_t>(combLabel)];
        }
    }

    if (lookup.empty()) {
        return false;
    }

    std::vector<HostLookupEntry> entries;
    entries.reserve(lookup.size());
    for (const auto& item : lookup) {
        HostLookupEntry entry{};
        entry.key.x = item.first.combLabel;
        entry.key.y = item.first.v0;
        entry.key.z = item.first.v1;
        entry.key.w = item.first.v2;
        entry.value = item.second;
        entries.push_back(entry);
    }

    std::sort(entries.begin(), entries.end(), [](const HostLookupEntry& lhs, const HostLookupEntry& rhs) {
        return compareInt4(lhs.key, rhs.key);
    });

    outKeys->reserve(entries.size());
    outValues->reserve(entries.size());
    for (const auto& entry : entries) {
        outKeys->push_back(entry.key);
        outValues->push_back(entry.value);
    }
    return true;
}

// (r1, comb4)
__global__ void KernelCalculateA(const double* base, const int4* comb4, int numComb4,
                                 int rows, int cols, double invRows, double* outA) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * numComb4;
    if (idx >= total) {
        return;
    }

    const int r1 = idx / numComb4;
    const int combIdx = idx - r1 * numComb4;
    const int4 comb = comb4[combIdx];

    const double v0 = base[r1 * cols + comb.x];
    const double v1 = base[r1 * cols + comb.y];
    const double v2 = base[r1 * cols + comb.z];
    const double v3 = base[r1 * cols + comb.w];

    double count = 0.0;
    for (int r2 = 0; r2 < rows; ++r2) {
        const int offset = r2 * cols;
        if (base[offset + comb.x] == v0 &&
            base[offset + comb.y] == v1 &&
            base[offset + comb.z] == v2 &&
            base[offset + comb.w] == v3) {
            count += 1.0;
        }
    }

    outA[idx] = count * invRows;
}

// (r1, feature)
__global__ void KernelCalculateM(const double* base, int rows, int cols,
                                 double invRows, double* outM) {
    const int numFeatures = cols - 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * numFeatures;
    if (idx >= total) {
        return;
    }

    const int r1 = idx / numFeatures;
    const int feature = idx - r1 * numFeatures;

    const double v = base[r1 * cols + feature];
    const double cls = base[r1 * cols + (cols - 1)];

    double count = 0.0;
    for (int r2 = 0; r2 < rows; ++r2) {
        const int offset = r2 * cols;
        if (base[offset + feature] == v && base[offset + (cols - 1)] == cls) {
            count += 1.0;
        }
    }
    outM[idx] = count * invRows;
}

// (row)
__global__ void KernelRowSum(const double* inA, int rows, int colsA, double* outRowSum) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const int offset = r * colsA;
    double sum = 0.0;
    for (int i = 0; i < colsA; ++i) {
        sum += inA[offset + i];
    }
    outRowSum[r] = sum;
}

// (r, comb3)
__global__ void KernelCalculateB(const double* inM, const double* rowSumA, const int3* comb3,
                                 int numComb3, int rows, int numFeatures, double* outB) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * numComb3;
    if (idx >= total) {
        return;
    }

    const int r = idx / numComb3;
    const int combIdx = idx - r * numComb3;
    const int3 comb = comb3[combIdx];
    const int rowOffset = r * numFeatures;

    const double m0 = inM[rowOffset + comb.x];
    const double m1 = inM[rowOffset + comb.y];
    const double m2 = inM[rowOffset + comb.z];
    const double minM = fmin(m0, fmin(m1, m2));

    outB[idx] = rowSumA[r] * minM;
}

// (r1, (label,comb3))
__global__ void KernelCalculateC(const double* base, const double* inB, const int3* comb3,
                                 int numComb3, int rows, int cols, int nClasses, int fullCols,
                                 double* outC) {
    const int activeCols = nClasses * numComb3;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * activeCols;
    if (idx >= total) {
        return;
    }

    const int r1 = idx / activeCols;
    const int localCol = idx - r1 * activeCols;

    const int label = (localCol / numComb3) + 1;
    const int combIdx = localCol % numComb3;
    const int3 comb = comb3[combIdx];

    const double v0 = base[r1 * cols + comb.x];
    const double v1 = base[r1 * cols + comb.y];
    const double v2 = base[r1 * cols + comb.z];

    double sum = 0.0;
    for (int r2 = 0; r2 < rows; ++r2) {
        const int offset = r2 * cols;
        if (base[offset + comb.x] == v0 &&
            base[offset + comb.y] == v1 &&
            base[offset + comb.z] == v2 &&
            static_cast<int>(base[offset + (cols - 1)]) == label) {
            sum += inB[r2 * numComb3 + combIdx];
        }
    }

    outC[r1 * fullCols + localCol] = sum;
}

__device__ int compareInt4Device(const int4& key, int x, int y, int z, int w) {
    if (key.x != x) return (key.x < x) ? -1 : 1;
    if (key.y != y) return (key.y < y) ? -1 : 1;
    if (key.z != z) return (key.z < z) ? -1 : 1;
    if (key.w != w) return (key.w < w) ? -1 : 1;
    return 0;
}

__device__ double lookupFisaValue(const int4* keys, const double* values, int count,
                                  int combLabel, int v0, int v1, int v2) {
    int lo = 0;
    int hi = count - 1;
    while (lo <= hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const int cmp = compareInt4Device(keys[mid], combLabel, v0, v1, v2);
        if (cmp == 0) {
            return values[mid];
        }
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return 0.0;
}

// (label)
__global__ void KernelFisaD(const double* base, const double* C, const double* input,
                            const int3* comb3, int numComb3, int rows, int cols,
                            int nClasses, int fullCols, double* outD) {
    const int classIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (classIdx >= nClasses) {
        return;
    }

    const int label = classIdx + 1;
    bool initialized = false;
    double maxVal = 0.0;
    double minVal = 0.0;

    for (int combIdx = 0; combIdx < numComb3; ++combIdx) {
        const int3 comb = comb3[combIdx];
        double value = 0.0;

        // Keep row-1 behavior to match legacy Python implementation exactly.
        for (int r = 0; r < rows - 1; ++r) {
            const int baseOffset = r * cols;
            if (base[baseOffset + comb.x] == input[comb.x] &&
                base[baseOffset + comb.y] == input[comb.y] &&
                base[baseOffset + comb.z] == input[comb.z] &&
                static_cast<int>(base[baseOffset + (cols - 1)]) == label) {
                const int cCol = (label - 1) * numComb3 + combIdx;
                if (cCol < fullCols) {
                    value = C[r * fullCols + cCol];
                }
            }
        }

        if (!initialized) {
            maxVal = value;
            minVal = value;
            initialized = true;
        } else {
            maxVal = fmax(maxVal, value);
            minVal = fmin(minVal, value);
        }
    }

    outD[classIdx] = initialized ? (maxVal + minVal) : 0.0;
}

// (label) optimized path with prebuilt lookup table.
__global__ void KernelFisaDLookup(const int4* lookupKeys, const double* lookupValues, int lookupSize,
                                  const double* input, const int3* comb3, int numComb3,
                                  int nClasses, double* outD) {
    const int classIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (classIdx >= nClasses) {
        return;
    }

    bool initialized = false;
    double maxVal = 0.0;
    double minVal = 0.0;

    for (int combIdx = 0; combIdx < numComb3; ++combIdx) {
        const int3 comb = comb3[combIdx];
        const int v0 = __double2int_rn(input[comb.x]);
        const int v1 = __double2int_rn(input[comb.y]);
        const int v2 = __double2int_rn(input[comb.z]);
        const int combLabel = classIdx * numComb3 + combIdx;

        const double value = lookupFisaValue(lookupKeys, lookupValues, lookupSize, combLabel, v0, v1, v2);
        if (!initialized) {
            maxVal = value;
            minVal = value;
            initialized = true;
        } else {
            maxVal = fmax(maxVal, value);
            minVal = fmin(minVal, value);
        }
    }

    outD[classIdx] = initialized ? (maxVal + minVal) : 0.0;
}

// (sample, label)
__global__ void KernelFisaDBatch(const double* base, const double* C, const double* inputs,
                                 const int3* comb3, int numComb3, int rows, int cols,
                                 int nClasses, int fullCols, int numInputs, double* outD) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = numInputs * nClasses;
    if (idx >= total) {
        return;
    }

    const int sampleIdx = idx / nClasses;
    const int classIdx = idx - sampleIdx * nClasses;
    const int label = classIdx + 1;
    const double* input = inputs + static_cast<size_t>(sampleIdx) * static_cast<size_t>(cols - 1);

    bool initialized = false;
    double maxVal = 0.0;
    double minVal = 0.0;

    for (int combIdx = 0; combIdx < numComb3; ++combIdx) {
        const int3 comb = comb3[combIdx];
        double value = 0.0;

        // Keep row-1 behavior to match legacy Python implementation exactly.
        for (int r = 0; r < rows - 1; ++r) {
            const int baseOffset = r * cols;
            if (base[baseOffset + comb.x] == input[comb.x] &&
                base[baseOffset + comb.y] == input[comb.y] &&
                base[baseOffset + comb.z] == input[comb.z] &&
                static_cast<int>(base[baseOffset + (cols - 1)]) == label) {
                const int cCol = (label - 1) * numComb3 + combIdx;
                if (cCol < fullCols) {
                    value = C[r * fullCols + cCol];
                }
            }
        }

        if (!initialized) {
            maxVal = value;
            minVal = value;
            initialized = true;
        } else {
            maxVal = fmax(maxVal, value);
            minVal = fmin(minVal, value);
        }
    }

    outD[idx] = initialized ? (maxVal + minVal) : 0.0;
}

// (sample, label) optimized path with prebuilt lookup table.
__global__ void KernelFisaDBatchLookup(const int4* lookupKeys, const double* lookupValues, int lookupSize,
                                       const double* inputs, const int3* comb3, int numComb3,
                                       int nClasses, int featureCount, int numInputs, double* outD) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = numInputs * nClasses;
    if (idx >= total) {
        return;
    }

    const int sampleIdx = idx / nClasses;
    const int classIdx = idx - sampleIdx * nClasses;
    const double* input = inputs + static_cast<size_t>(sampleIdx) * static_cast<size_t>(featureCount);

    bool initialized = false;
    double maxVal = 0.0;
    double minVal = 0.0;

    for (int combIdx = 0; combIdx < numComb3; ++combIdx) {
        const int3 comb = comb3[combIdx];
        const int v0 = __double2int_rn(input[comb.x]);
        const int v1 = __double2int_rn(input[comb.y]);
        const int v2 = __double2int_rn(input[comb.z]);
        const int combLabel = classIdx * numComb3 + combIdx;

        const double value = lookupFisaValue(lookupKeys, lookupValues, lookupSize, combLabel, v0, v1, v2);
        if (!initialized) {
            maxVal = value;
            minVal = value;
            initialized = true;
        } else {
            maxVal = fmax(maxVal, value);
            minVal = fmin(minVal, value);
        }
    }

    outD[idx] = initialized ? (maxVal + minVal) : 0.0;
}

inline cudaError_t checkCudaKernel(const char* stage, std::string* error_message) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        setError(error_message, stage, err);
        return err;
    }
    return cudaSuccess;
}

cudaError_t runABCM(const Matrix& base, int n_classes,
                    Matrix* outA, Matrix* outM, Matrix* outB, Matrix* outC,
                    std::string* error_message) {
    int rows = 0;
    int cols = 0;
    if (!validateRectangular(base, &rows, &cols)) {
        if (error_message != nullptr) {
            *error_message = "Input base matrix is empty or not rectangular.";
        }
        return cudaErrorInvalidValue;
    }
    if (cols < 2) {
        if (error_message != nullptr) {
            *error_message = "Input base matrix must contain at least one feature and one label column.";
        }
        return cudaErrorInvalidValue;
    }

    const int numFeatures = cols - 1;
    const int numComb4 = combination(4, numFeatures);
    const int numComb3 = combination(3, numFeatures);
    const int fullColsC = 6 * numComb3;
    const int activeClassCount = std::max(0, std::min(n_classes, 6));

    if (numComb4 <= 0 || numComb3 <= 0 || activeClassCount <= 0) {
        if (error_message != nullptr) {
            *error_message = "Need at least 4 feature columns and n_classes in [1, 6] for FKG CUDA.";
        }
        return cudaErrorInvalidValue;
    }

    const std::vector<double> hBase = flattenMatrix(base);
    const std::vector<int4> hComb4 = buildComb4(numFeatures);
    const std::vector<int3> hComb3 = buildComb3(numFeatures);

    DeviceBuffer<double> dBase;
    DeviceBuffer<int4> dComb4;
    DeviceBuffer<int3> dComb3;
    DeviceBuffer<double> dA;
    DeviceBuffer<double> dM;
    DeviceBuffer<double> dRowSumA;
    DeviceBuffer<double> dB;
    DeviceBuffer<double> dC;

    cudaError_t err = cudaSuccess;

    err = dBase.allocate(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(base)", err); return err; }
    err = dComb4.allocate(hComb4.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(comb4)", err); return err; }
    err = dComb3.allocate(hComb3.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(comb3)", err); return err; }
    err = dA.allocate(static_cast<size_t>(rows) * static_cast<size_t>(numComb4));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(A)", err); return err; }
    err = dM.allocate(static_cast<size_t>(rows) * static_cast<size_t>(numFeatures));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(M)", err); return err; }
    err = dRowSumA.allocate(static_cast<size_t>(rows));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(sumA)", err); return err; }
    err = dB.allocate(static_cast<size_t>(rows) * static_cast<size_t>(numComb3));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(B)", err); return err; }
    err = dC.allocate(static_cast<size_t>(rows) * static_cast<size_t>(fullColsC));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(C)", err); return err; }

    err = cudaMemcpy(dBase.get(), hBase.data(),
                     static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(double),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(base)", err); return err; }

    err = cudaMemcpy(dComb4.get(), hComb4.data(), hComb4.size() * sizeof(int4), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(comb4)", err); return err; }
    err = cudaMemcpy(dComb3.get(), hComb3.data(), hComb3.size() * sizeof(int3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(comb3)", err); return err; }

    const double invRows = 1.0 / static_cast<double>(rows);

    const int totalA = rows * numComb4;
    KernelCalculateA<<<(totalA + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        dBase.get(), dComb4.get(), numComb4, rows, cols, invRows, dA.get());
    err = checkCudaKernel("KernelCalculateA", error_message);
    if (err != cudaSuccess) { return err; }

    const int totalM = rows * numFeatures;
    KernelCalculateM<<<(totalM + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        dBase.get(), rows, cols, invRows, dM.get());
    err = checkCudaKernel("KernelCalculateM", error_message);
    if (err != cudaSuccess) { return err; }

    KernelRowSum<<<(rows + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        dA.get(), rows, numComb4, dRowSumA.get());
    err = checkCudaKernel("KernelRowSum", error_message);
    if (err != cudaSuccess) { return err; }

    const int totalB = rows * numComb3;
    KernelCalculateB<<<(totalB + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        dM.get(), dRowSumA.get(), dComb3.get(), numComb3, rows, numFeatures, dB.get());
    err = checkCudaKernel("KernelCalculateB", error_message);
    if (err != cudaSuccess) { return err; }

    err = cudaMemset(dC.get(), 0, static_cast<size_t>(rows) * static_cast<size_t>(fullColsC) * sizeof(double));
    if (err != cudaSuccess) { setError(error_message, "cudaMemset(C)", err); return err; }

    const int totalC = rows * activeClassCount * numComb3;
    KernelCalculateC<<<(totalC + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        dBase.get(), dB.get(), dComb3.get(), numComb3, rows, cols, activeClassCount, fullColsC, dC.get());
    err = checkCudaKernel("KernelCalculateC", error_message);
    if (err != cudaSuccess) { return err; }

    std::vector<double> hA(static_cast<size_t>(rows) * static_cast<size_t>(numComb4));
    std::vector<double> hM(static_cast<size_t>(rows) * static_cast<size_t>(numFeatures));
    std::vector<double> hB(static_cast<size_t>(rows) * static_cast<size_t>(numComb3));
    std::vector<double> hC(static_cast<size_t>(rows) * static_cast<size_t>(fullColsC));

    err = cudaMemcpy(hA.data(), dA.get(), hA.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(A)", err); return err; }
    err = cudaMemcpy(hM.data(), dM.get(), hM.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(M)", err); return err; }
    err = cudaMemcpy(hB.data(), dB.get(), hB.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(B)", err); return err; }
    err = cudaMemcpy(hC.data(), dC.get(), hC.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(C)", err); return err; }

    if (outA != nullptr) {
        reshapeMatrix(hA, rows, numComb4, *outA);
    }
    if (outM != nullptr) {
        reshapeMatrix(hM, rows, numFeatures, *outM);
    }
    if (outB != nullptr) {
        reshapeMatrix(hB, rows, numComb3, *outB);
    }
    if (outC != nullptr) {
        reshapeMatrix(hC, rows, fullColsC, *outC);
    }

    return cudaSuccess;
}

} // namespace

cudaError_t calculateA_GPU(const Matrix& base, Matrix& A, std::string* error_message) {
    return runABCM(base, 1, &A, nullptr, nullptr, nullptr, error_message);
}

cudaError_t calculateM_GPU(const Matrix& base, Matrix& M, std::string* error_message) {
    return runABCM(base, 1, nullptr, &M, nullptr, nullptr, error_message);
}

cudaError_t calculateB_GPU(const Matrix& A, const Matrix& M, Matrix& B, int columns,
                           std::string* error_message) {
    if (A.empty() || M.empty()) {
        if (error_message != nullptr) {
            *error_message = "Input matrices A/M must not be empty.";
        }
        return cudaErrorInvalidValue;
    }
    if (static_cast<int>(A.size()) != static_cast<int>(M.size())) {
        if (error_message != nullptr) {
            *error_message = "A and M row counts must match.";
        }
        return cudaErrorInvalidValue;
    }
    if (columns < 2) {
        if (error_message != nullptr) {
            *error_message = "columns must be >= 2.";
        }
        return cudaErrorInvalidValue;
    }

    const int rows = static_cast<int>(A.size());
    const int aCols = static_cast<int>(A[0].size());
    const int numFeatures = columns - 1;
    const int numComb3 = combination(3, numFeatures);
    if (numComb3 <= 0) {
        if (error_message != nullptr) {
            *error_message = "Need at least 3 feature columns for B matrix.";
        }
        return cudaErrorInvalidValue;
    }

    std::vector<double> hA = flattenMatrix(A);
    std::vector<double> hM = flattenMatrix(M);
    std::vector<int3> hComb3 = buildComb3(numFeatures);
    std::vector<double> hB(static_cast<size_t>(rows) * static_cast<size_t>(numComb3), 0.0);

    DeviceBuffer<double> dA;
    DeviceBuffer<double> dM;
    DeviceBuffer<double> dRowSumA;
    DeviceBuffer<int3> dComb3;
    DeviceBuffer<double> dB;

    cudaError_t err = dA.allocate(hA.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(A)", err); return err; }
    err = dM.allocate(hM.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(M)", err); return err; }
    err = dRowSumA.allocate(static_cast<size_t>(rows));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(sumA)", err); return err; }
    err = dComb3.allocate(hComb3.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(comb3)", err); return err; }
    err = dB.allocate(hB.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(B)", err); return err; }

    err = cudaMemcpy(dA.get(), hA.data(), hA.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(A)", err); return err; }
    err = cudaMemcpy(dM.get(), hM.data(), hM.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(M)", err); return err; }
    err = cudaMemcpy(dComb3.get(), hComb3.data(), hComb3.size() * sizeof(int3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(comb3)", err); return err; }

    KernelRowSum<<<(rows + kBlockSize - 1) / kBlockSize, kBlockSize>>>(dA.get(), rows, aCols, dRowSumA.get());
    err = checkCudaKernel("KernelRowSum", error_message);
    if (err != cudaSuccess) { return err; }

    const int totalB = rows * numComb3;
    KernelCalculateB<<<(totalB + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        dM.get(), dRowSumA.get(), dComb3.get(), numComb3, rows, numFeatures, dB.get());
    err = checkCudaKernel("KernelCalculateB", error_message);
    if (err != cudaSuccess) { return err; }

    err = cudaMemcpy(hB.data(), dB.get(), hB.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(B)", err); return err; }

    reshapeMatrix(hB, rows, numComb3, B);
    return cudaSuccess;
}

cudaError_t calculateC_GPU(const Matrix& base, const Matrix& B, Matrix& C, int columns, int n_classes,
                           std::string* error_message) {
    if (base.empty() || B.empty()) {
        if (error_message != nullptr) {
            *error_message = "Input matrices base/B must not be empty.";
        }
        return cudaErrorInvalidValue;
    }
    if (static_cast<int>(base.size()) != static_cast<int>(B.size())) {
        if (error_message != nullptr) {
            *error_message = "base and B row counts must match.";
        }
        return cudaErrorInvalidValue;
    }
    if (columns < 2) {
        if (error_message != nullptr) {
            *error_message = "columns must be >= 2.";
        }
        return cudaErrorInvalidValue;
    }

    const int rows = static_cast<int>(base.size());
    const int cols = columns;
    const int numFeatures = cols - 1;
    const int numComb3 = combination(3, numFeatures);
    const int fullColsC = 6 * numComb3;
    const int activeClassCount = std::max(0, std::min(n_classes, 6));

    if (numComb3 <= 0 || activeClassCount <= 0) {
        if (error_message != nullptr) {
            *error_message = "Need at least 3 feature columns and n_classes in [1, 6] for C matrix.";
        }
        return cudaErrorInvalidValue;
    }

    std::vector<double> hBase = flattenMatrix(base);
    std::vector<double> hB = flattenMatrix(B);
    std::vector<int3> hComb3 = buildComb3(numFeatures);
    std::vector<double> hC(static_cast<size_t>(rows) * static_cast<size_t>(fullColsC), 0.0);

    DeviceBuffer<double> dBase;
    DeviceBuffer<double> dB;
    DeviceBuffer<int3> dComb3;
    DeviceBuffer<double> dC;

    cudaError_t err = dBase.allocate(hBase.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(base)", err); return err; }
    err = dB.allocate(hB.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(B)", err); return err; }
    err = dComb3.allocate(hComb3.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(comb3)", err); return err; }
    err = dC.allocate(hC.size());
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(C)", err); return err; }

    err = cudaMemcpy(dBase.get(), hBase.data(), hBase.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(base)", err); return err; }
    err = cudaMemcpy(dB.get(), hB.data(), hB.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(B)", err); return err; }
    err = cudaMemcpy(dComb3.get(), hComb3.data(), hComb3.size() * sizeof(int3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(comb3)", err); return err; }
    err = cudaMemset(dC.get(), 0, hC.size() * sizeof(double));
    if (err != cudaSuccess) { setError(error_message, "cudaMemset(C)", err); return err; }

    const int totalC = rows * activeClassCount * numComb3;
    KernelCalculateC<<<(totalC + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        dBase.get(), dB.get(), dComb3.get(), numComb3, rows, cols, activeClassCount, fullColsC, dC.get());
    err = checkCudaKernel("KernelCalculateC", error_message);
    if (err != cudaSuccess) { return err; }

    err = cudaMemcpy(hC.data(), dC.get(), hC.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(C)", err); return err; }

    reshapeMatrix(hC, rows, fullColsC, C);
    return cudaSuccess;
}

cudaError_t calculateABCM_GPU(const Matrix& base, int n_classes,
                              Matrix& A, Matrix& M, Matrix& B, Matrix& C,
                              std::string* error_message) {
    return runABCM(base, n_classes, &A, &M, &B, &C, error_message);
}

cudaError_t fisaGPU(const Matrix& base, const Matrix& C, const std::vector<double>& input,
                    int n_classes, int& result_class, double& result_confidence,
                    std::vector<double>* d_values, std::string* error_message) {
    FisaDeviceCache cache;
    cudaError_t err = createFisaDeviceCache(base, C, n_classes, cache, error_message);
    if (err != cudaSuccess) {
        return err;
    }
    err = fisaGPUWithCache(cache, input, result_class, result_confidence, d_values, error_message);
    destroyFisaDeviceCache(cache);
    return err;
}

cudaError_t createFisaDeviceCache(const Matrix& base, const Matrix& C, int n_classes,
                                  FisaDeviceCache& cache, std::string* error_message) {
    destroyFisaDeviceCache(cache);

    int rows = 0;
    int cols = 0;
    if (!validateRectangular(base, &rows, &cols)) {
        if (error_message != nullptr) {
            *error_message = "Input base matrix is empty or not rectangular.";
        }
        return cudaErrorInvalidValue;
    }

    int cRows = 0;
    int cCols = 0;
    if (!validateRectangular(C, &cRows, &cCols) || cRows != rows) {
        if (error_message != nullptr) {
            *error_message = "Input C matrix is empty, not rectangular, or row count mismatch.";
        }
        return cudaErrorInvalidValue;
    }

    const int numFeatures = cols - 1;
    const int numComb3 = combination(3, numFeatures);
    const int activeClassCount = std::max(0, std::min(n_classes, 6));
    if (numComb3 <= 0 || activeClassCount <= 0) {
        if (error_message != nullptr) {
            *error_message = "Need at least 3 feature columns and n_classes in [1, 6] for FISA.";
        }
        return cudaErrorInvalidValue;
    }

    if (cCols < activeClassCount * numComb3) {
        if (error_message != nullptr) {
            *error_message = "C matrix does not have enough columns for requested classes.";
        }
        return cudaErrorInvalidValue;
    }

    const std::vector<int3> hComb3 = buildComb3(numFeatures);

    double* dBase = nullptr;
    double* dC = nullptr;
    int3* dComb3 = nullptr;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dComb3), hComb3.size() * sizeof(int3));
    if (err != cudaSuccess) {
        setError(error_message, "cudaMalloc(cache.comb3)", err);
        return err;
    }

    err = cudaMemcpy(dComb3, hComb3.data(), hComb3.size() * sizeof(int3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        setError(error_message, "cudaMemcpy(cache.comb3)", err);
        cudaFree(dComb3);
        return err;
    }

    int4* dLookupKeys = nullptr;
    double* dLookupValues = nullptr;
    int lookupSize = 0;
    int useLookup = 0;

    std::vector<int4> hLookupKeys;
    std::vector<double> hLookupValues;
    if (buildFisaLookupTable(base, C, hComb3, activeClassCount, cCols, &hLookupKeys, &hLookupValues) &&
        !hLookupKeys.empty() &&
        hLookupKeys.size() == hLookupValues.size()) {
        cudaError_t lookupErr =
            cudaMalloc(reinterpret_cast<void**>(&dLookupKeys), hLookupKeys.size() * sizeof(int4));
        if (lookupErr == cudaSuccess) {
            lookupErr = cudaMalloc(reinterpret_cast<void**>(&dLookupValues),
                                   hLookupValues.size() * sizeof(double));
        }
        if (lookupErr == cudaSuccess) {
            lookupErr = cudaMemcpy(dLookupKeys, hLookupKeys.data(),
                                   hLookupKeys.size() * sizeof(int4),
                                   cudaMemcpyHostToDevice);
        }
        if (lookupErr == cudaSuccess) {
            lookupErr = cudaMemcpy(dLookupValues, hLookupValues.data(),
                                   hLookupValues.size() * sizeof(double),
                                   cudaMemcpyHostToDevice);
        }
        if (lookupErr == cudaSuccess) {
            lookupSize = static_cast<int>(hLookupKeys.size());
            useLookup = (lookupSize > 0) ? 1 : 0;
        } else {
            if (dLookupValues != nullptr) {
                cudaFree(dLookupValues);
                dLookupValues = nullptr;
            }
            if (dLookupKeys != nullptr) {
                cudaFree(dLookupKeys);
                dLookupKeys = nullptr;
            }
            lookupSize = 0;
            useLookup = 0;
        }
    }

    if (useLookup == 0) {
        const std::vector<double> hBase = flattenMatrix(base);
        const std::vector<double> hC = flattenMatrix(C);

        err = cudaMalloc(reinterpret_cast<void**>(&dBase), hBase.size() * sizeof(double));
        if (err != cudaSuccess) {
            setError(error_message, "cudaMalloc(cache.base)", err);
            if (dLookupValues != nullptr) cudaFree(dLookupValues);
            if (dLookupKeys != nullptr) cudaFree(dLookupKeys);
            cudaFree(dComb3);
            return err;
        }

        err = cudaMalloc(reinterpret_cast<void**>(&dC), hC.size() * sizeof(double));
        if (err != cudaSuccess) {
            setError(error_message, "cudaMalloc(cache.C)", err);
            cudaFree(dBase);
            if (dLookupValues != nullptr) cudaFree(dLookupValues);
            if (dLookupKeys != nullptr) cudaFree(dLookupKeys);
            cudaFree(dComb3);
            return err;
        }

        err = cudaMemcpy(dBase, hBase.data(), hBase.size() * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            setError(error_message, "cudaMemcpy(cache.base)", err);
            cudaFree(dC);
            cudaFree(dBase);
            if (dLookupValues != nullptr) cudaFree(dLookupValues);
            if (dLookupKeys != nullptr) cudaFree(dLookupKeys);
            cudaFree(dComb3);
            return err;
        }

        err = cudaMemcpy(dC, hC.data(), hC.size() * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            setError(error_message, "cudaMemcpy(cache.C)", err);
            cudaFree(dC);
            cudaFree(dBase);
            if (dLookupValues != nullptr) cudaFree(dLookupValues);
            if (dLookupKeys != nullptr) cudaFree(dLookupKeys);
            cudaFree(dComb3);
            return err;
        }
    }

    cache.dBase = dBase;
    cache.dC = dC;
    cache.dComb3 = dComb3;
    cache.dLookupKeys = dLookupKeys;
    cache.dLookupValues = dLookupValues;
    cache.rows = rows;
    cache.cols = cols;
    cache.fullCols = cCols;
    cache.numComb3 = numComb3;
    cache.nClasses = activeClassCount;
    cache.lookupSize = lookupSize;
    cache.useLookup = useLookup;
    return cudaSuccess;
}

void destroyFisaDeviceCache(FisaDeviceCache& cache) {
    if (cache.dLookupValues != nullptr) {
        cudaFree(cache.dLookupValues);
        cache.dLookupValues = nullptr;
    }
    if (cache.dLookupKeys != nullptr) {
        cudaFree(cache.dLookupKeys);
        cache.dLookupKeys = nullptr;
    }
    if (cache.dComb3 != nullptr) {
        cudaFree(cache.dComb3);
        cache.dComb3 = nullptr;
    }
    if (cache.dC != nullptr) {
        cudaFree(cache.dC);
        cache.dC = nullptr;
    }
    if (cache.dBase != nullptr) {
        cudaFree(cache.dBase);
        cache.dBase = nullptr;
    }
    cache.rows = 0;
    cache.cols = 0;
    cache.fullCols = 0;
    cache.numComb3 = 0;
    cache.nClasses = 0;
    cache.lookupSize = 0;
    cache.useLookup = 0;
}

cudaError_t fisaGPUWithCache(const FisaDeviceCache& cache, const std::vector<double>& input,
                             int& result_class, double& result_confidence,
                             std::vector<double>* d_values, std::string* error_message) {
    if (cache.dComb3 == nullptr ||
        cache.rows <= 0 || cache.cols <= 1 || cache.numComb3 <= 0 || cache.nClasses <= 0) {
        if (error_message != nullptr) {
            *error_message = "FISA device cache is not initialized.";
        }
        return cudaErrorInvalidResourceHandle;
    }
    if (cache.useLookup == 0 && (cache.dBase == nullptr || cache.dC == nullptr)) {
        if (error_message != nullptr) {
            *error_message = "FISA cache missing base/C tensors.";
        }
        return cudaErrorInvalidResourceHandle;
    }

    const int featureCount = cache.cols - 1;
    if (static_cast<int>(input.size()) != featureCount) {
        if (error_message != nullptr) {
            *error_message = "Input sample size must match cache feature count.";
        }
        return cudaErrorInvalidValue;
    }
    bool canUseLookup = (cache.useLookup != 0 && cache.dLookupKeys != nullptr &&
                         cache.dLookupValues != nullptr && cache.lookupSize > 0);
    if (canUseLookup) {
        for (double v : input) {
            if (!isNearlyInteger(v)) {
                canUseLookup = false;
                break;
            }
        }
    }

    DeviceBuffer<double> dInput;
    DeviceBuffer<double> dD;
    std::vector<double> hD(static_cast<size_t>(cache.nClasses), 0.0);

    cudaError_t err = dInput.allocate(static_cast<size_t>(featureCount));
    if (err != cudaSuccess) {
        setError(error_message, "cudaMalloc(input)", err);
        return err;
    }
    err = dD.allocate(static_cast<size_t>(cache.nClasses));
    if (err != cudaSuccess) {
        setError(error_message, "cudaMalloc(D)", err);
        return err;
    }

    err = cudaMemcpy(dInput.get(), input.data(), static_cast<size_t>(featureCount) * sizeof(double),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        setError(error_message, "cudaMemcpy(input)", err);
        return err;
    }

    if (canUseLookup) {
        KernelFisaDLookup<<<(cache.nClasses + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            cache.dLookupKeys, cache.dLookupValues, cache.lookupSize, dInput.get(),
            cache.dComb3, cache.numComb3, cache.nClasses, dD.get());
        err = checkCudaKernel("KernelFisaDLookup", error_message);
    } else {
        KernelFisaD<<<(cache.nClasses + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            cache.dBase, cache.dC, dInput.get(), cache.dComb3, cache.numComb3, cache.rows,
            cache.cols, cache.nClasses, cache.fullCols, dD.get());
        err = checkCudaKernel("KernelFisaD", error_message);
    }
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMemcpy(hD.data(), dD.get(), hD.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        setError(error_message, "cudaMemcpy(D)", err);
        return err;
    }

    int best = 1;
    double maxD = hD[0];
    double sumD = hD[0];
    for (int classIdx = 1; classIdx < cache.nClasses; ++classIdx) {
        sumD += hD[static_cast<size_t>(classIdx)];
        if (hD[static_cast<size_t>(classIdx)] > maxD) {
            maxD = hD[static_cast<size_t>(classIdx)];
            best = classIdx + 1;
        }
    }

    result_class = best;
    result_confidence = (sumD > 0.0) ? (maxD / sumD) : 0.0;
    if (d_values != nullptr) {
        d_values->assign(hD.begin(), hD.end());
    }
    return cudaSuccess;
}

cudaError_t fisaBatchGPUWithCache(const FisaDeviceCache& cache, const Matrix& inputs,
                                  std::vector<int>& result_classes,
                                  std::vector<double>& result_confidences,
                                  std::string* error_message) {
    result_classes.clear();
    result_confidences.clear();

    if (inputs.empty()) {
        return cudaSuccess;
    }

    if (cache.dComb3 == nullptr ||
        cache.rows <= 0 || cache.cols <= 1 || cache.numComb3 <= 0 || cache.nClasses <= 0) {
        if (error_message != nullptr) {
            *error_message = "FISA device cache is not initialized.";
        }
        return cudaErrorInvalidResourceHandle;
    }
    if (cache.useLookup == 0 && (cache.dBase == nullptr || cache.dC == nullptr)) {
        if (error_message != nullptr) {
            *error_message = "FISA cache missing base/C tensors.";
        }
        return cudaErrorInvalidResourceHandle;
    }

    const int numInputs = static_cast<int>(inputs.size());
    const int featureCount = cache.cols - 1;
    bool canUseLookup = (cache.useLookup != 0 && cache.dLookupKeys != nullptr &&
                         cache.dLookupValues != nullptr && cache.lookupSize > 0);
    std::vector<double> hInputs;
    hInputs.resize(static_cast<size_t>(numInputs) * static_cast<size_t>(featureCount), 0.0);

    for (int i = 0; i < numInputs; ++i) {
        if (static_cast<int>(inputs[static_cast<size_t>(i)].size()) != featureCount) {
            if (error_message != nullptr) {
                *error_message = "All input rows must match cache feature count.";
            }
            return cudaErrorInvalidValue;
        }
        if (canUseLookup) {
            for (double v : inputs[static_cast<size_t>(i)]) {
                if (!isNearlyInteger(v)) {
                    canUseLookup = false;
                    break;
                }
            }
        }
        std::memcpy(hInputs.data() + static_cast<size_t>(i) * static_cast<size_t>(featureCount),
                    inputs[static_cast<size_t>(i)].data(),
                    static_cast<size_t>(featureCount) * sizeof(double));
    }

    DeviceBuffer<double> dInputs;
    DeviceBuffer<double> dBatchD;
    std::vector<double> hBatchD(static_cast<size_t>(numInputs) * static_cast<size_t>(cache.nClasses), 0.0);

    cudaError_t err = dInputs.allocate(hInputs.size());
    if (err != cudaSuccess) {
        setError(error_message, "cudaMalloc(inputs)", err);
        return err;
    }
    err = dBatchD.allocate(hBatchD.size());
    if (err != cudaSuccess) {
        setError(error_message, "cudaMalloc(batchD)", err);
        return err;
    }

    err = cudaMemcpy(dInputs.get(), hInputs.data(), hInputs.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        setError(error_message, "cudaMemcpy(inputs)", err);
        return err;
    }

    const int totalThreads = numInputs * cache.nClasses;
    if (canUseLookup) {
        KernelFisaDBatchLookup<<<(totalThreads + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            cache.dLookupKeys, cache.dLookupValues, cache.lookupSize, dInputs.get(),
            cache.dComb3, cache.numComb3, cache.nClasses, featureCount, numInputs, dBatchD.get());
        err = checkCudaKernel("KernelFisaDBatchLookup", error_message);
    } else {
        KernelFisaDBatch<<<(totalThreads + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            cache.dBase, cache.dC, dInputs.get(), cache.dComb3, cache.numComb3, cache.rows,
            cache.cols, cache.nClasses, cache.fullCols, numInputs, dBatchD.get());
        err = checkCudaKernel("KernelFisaDBatch", error_message);
    }
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMemcpy(hBatchD.data(), dBatchD.get(), hBatchD.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        setError(error_message, "cudaMemcpy(batchD)", err);
        return err;
    }

    result_classes.resize(static_cast<size_t>(numInputs), 1);
    result_confidences.resize(static_cast<size_t>(numInputs), 0.0);
    for (int i = 0; i < numInputs; ++i) {
        const size_t baseOffset = static_cast<size_t>(i) * static_cast<size_t>(cache.nClasses);
        int best = 1;
        double maxD = hBatchD[baseOffset];
        double sumD = hBatchD[baseOffset];
        for (int classIdx = 1; classIdx < cache.nClasses; ++classIdx) {
            const double value = hBatchD[baseOffset + static_cast<size_t>(classIdx)];
            sumD += value;
            if (value > maxD) {
                maxD = value;
                best = classIdx + 1;
            }
        }
        result_classes[static_cast<size_t>(i)] = best;
        result_confidences[static_cast<size_t>(i)] = (sumD > 0.0) ? (maxD / sumD) : 0.0;
    }
    return cudaSuccess;
}

} // namespace CUDA
} // namespace Fuzzy

#endif // FUZZY_USE_CUDA
