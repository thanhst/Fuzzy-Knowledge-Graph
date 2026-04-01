/**
 * @file FKG.cpp
 * @brief Fuzzy Knowledge Graph (FKG) - Correct Implementation matching Python source
 * @version 2.1
 * 
 * Implements the exact FKG algorithm from Python source:
 * - calculateA: Count matching 4-tuple combinations / row
 * - calculateM: Count matching attributes between rows
 * - calculateB: sum(A[r]) * min(M[r][a], M[r][b], M[r][c])
 * - calculateC: Class-based aggregation with 6 * C(3, n) columns
 */

#include "FKG.h"
#if FUZZY_USE_CUDA
#include <cuda_runtime.h>
#include "FKG_CUDA_Kernels.h"
#endif
#include <array>
#include <cstddef>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <chrono>
#include <atomic>
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Fuzzy {

// ============================================================================
// Performance Configuration
// ============================================================================

static int g_defaultNumThreads = 0;

void setDefaultThreads(int n) { g_defaultNumThreads = n; }
int getOptimalThreadCount() {
    if (g_defaultNumThreads > 0) return g_defaultNumThreads;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ============================================================================
// Combination helper
// ============================================================================

int combination(int k, int n) {
    if (k < 0 || n < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n - k) {
        k = n - k;
    }

    long long result = 1;
    for (int i = 1; i <= k; i++) {
        result = result * (n - k + i) / i;
    }
    return static_cast<int>(result);
}

namespace {

struct Comb3 {
    int a;
    int b;
    int c;
};

struct Comb4 {
    int a;
    int b;
    int c;
    int d;
};

inline std::vector<Comb3> buildComb3(int numFeatures) {
    std::vector<Comb3> combinations;
    combinations.reserve(static_cast<size_t>(combination(3, numFeatures)));
    for (int a = 0; a < numFeatures - 2; ++a) {
        for (int b = a + 1; b < numFeatures - 1; ++b) {
            for (int c = b + 1; c < numFeatures; ++c) {
                combinations.push_back(Comb3{a, b, c});
            }
        }
    }
    return combinations;
}

inline std::vector<Comb4> buildComb4(int numFeatures) {
    std::vector<Comb4> combinations;
    combinations.reserve(static_cast<size_t>(combination(4, numFeatures)));
    for (int a = 0; a < numFeatures - 3; ++a) {
        for (int b = a + 1; b < numFeatures - 2; ++b) {
            for (int c = b + 1; c < numFeatures - 1; ++c) {
                for (int d = c + 1; d < numFeatures; ++d) {
                    combinations.push_back(Comb4{a, b, c, d});
                }
            }
        }
    }
    return combinations;
}

inline uint64_t bitsOf(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct Key4 {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;

    bool operator==(const Key4& other) const {
        return a == other.a && b == other.b && c == other.c && d == other.d;
    }
};

struct Key2 {
    uint64_t a;
    uint64_t b;

    bool operator==(const Key2& other) const {
        return a == other.a && b == other.b;
    }
};

struct Key3Class {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    int label;

    bool operator==(const Key3Class& other) const {
        return a == other.a && b == other.b && c == other.c && label == other.label;
    }
};

struct Key4Hash {
    size_t operator()(const Key4& key) const {
        size_t h = std::hash<uint64_t>{}(key.a);
        h ^= std::hash<uint64_t>{}(key.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.d) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Key2Hash {
    size_t operator()(const Key2& key) const {
        size_t h = std::hash<uint64_t>{}(key.a);
        h ^= std::hash<uint64_t>{}(key.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Key3ClassHash {
    size_t operator()(const Key3Class& key) const {
        size_t h = std::hash<uint64_t>{}(key.a);
        h ^= std::hash<uint64_t>{}(key.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.label) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

inline int detectClassCount(const Matrix& base) {
    std::set<int> labels;
    for (const auto& row : base) {
        labels.insert(static_cast<int>(row.back()));
    }
    return static_cast<int>(labels.size());
}

#if FUZZY_USE_CUDA
struct CudaInferenceCacheHandle {
    CUDA::FisaDeviceCache cache;
};
#endif

} // namespace

// ============================================================================
// FKG Class Implementation
// ============================================================================

FKG::FKG() : n_classes_(2), trained_(false), useGPU_(false)
#if FUZZY_USE_CUDA
    , cudaInferenceCache_(nullptr)
#endif
{
    config_ = PerformanceConfig();
    metrics_ = PerformanceMetrics{0.0, 0, 1};
}

FKG::FKG(const PerformanceConfig& config) 
    : n_classes_(2), trained_(false), config_(config), useGPU_(false)
#if FUZZY_USE_CUDA
    , cudaInferenceCache_(nullptr)
#endif
{
    metrics_ = PerformanceMetrics{0.0, 0, 1};
}

FKG::~FKG() {
#if FUZZY_USE_CUDA
    invalidateGPUCache();
#endif
}

void FKG::train(const Matrix& base) {
    auto start = std::chrono::high_resolution_clock::now();

    n_classes_ = detectClassCount(base);
    train(base, n_classes_);
    
    auto end = std::chrono::high_resolution_clock::now();
    metrics_.computeTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void FKG::train(const Matrix& base, int n_classes) {
#if FUZZY_USE_CUDA
    invalidateGPUCache();
#endif
    base_ = base;
    n_classes_ = n_classes;
    ComputedMatrices matrices = computeMatrices(base_, n_classes_, isUsingGPU());
    A_ = std::move(matrices.A);
    M_ = std::move(matrices.M);
    B_ = std::move(matrices.B);
    C_ = std::move(matrices.C);
    
    trained_ = true;
    metrics_.numThreadsUsed = getOptimalThreadCount();
}

std::pair<int, double> FKG::predict(const std::vector<double>& input) const {
    if (!trained_) {
        return {1, 0.0};
    }
    if (isUsingGPU()) {
        return predictGPUCached(input);
    }
    return fisa(base_, C_, input, n_classes_);
}

std::vector<int> FKG::predictBatch(const Matrix& inputs) const {
    if (!trained_ || inputs.empty()) {
        return {};
    }

    if (isUsingGPU()) {
        const auto gpuResults = predictBatchWithConfidenceGPUCached(inputs);
        std::vector<int> classes;
        classes.reserve(gpuResults.size());
        for (const auto& item : gpuResults) {
            classes.push_back(item.first);
        }
        return classes;
    }

    return predictBatchParallel(inputs, getOptimalThreadCount());
}

std::vector<std::pair<int, double>> FKG::predictBatchWithConfidence(const Matrix& inputs) const {
    if (!trained_ || inputs.empty()) {
        return {};
    }

    if (isUsingGPU()) {
        return predictBatchWithConfidenceGPUCached(inputs);
    }

    const int n = static_cast<int>(inputs.size());
    std::vector<std::pair<int, double>> results(static_cast<size_t>(n));
    int numThreads = getOptimalThreadCount();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 1024)
    for (int i = 0; i < n; ++i) {
        results[static_cast<size_t>(i)] = fisa(base_, C_, inputs[static_cast<size_t>(i)], n_classes_);
    }
    return results;
}

std::vector<int> FKG::predictBatchParallel(const Matrix& inputs, int numThreads) const {
    if (!trained_ || inputs.empty()) {
        return {};
    }

    if (isUsingGPU()) {
        const auto gpuResults = predictBatchWithConfidenceGPUCached(inputs);
        std::vector<int> classes;
        classes.reserve(gpuResults.size());
        for (const auto& item : gpuResults) {
            classes.push_back(item.first);
        }
        return classes;
    }
    
    int n = static_cast<int>(inputs.size());
    std::vector<int> predictions(n);
    
    if (numThreads <= 0) numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 1024)
    for (int i = 0; i < n; i++) {
        auto result = fisa(base_, C_, inputs[i], n_classes_);
        predictions[i] = result.first;
    }
    
    return predictions;
}

// ============================================================================
// calculateA: Count matching 4-tuple combinations / row (EXACT PYTHON)
// ============================================================================

Matrix FKG::calculateA_Parallel(const Matrix& base) {
    if (base.empty() || base[0].empty()) {
        return {};
    }

    const int row = static_cast<int>(base.size());
    const int colum = static_cast<int>(base[0].size());
    const int numFeatures = colum - 1;
    const std::vector<Comb4> comb4 = buildComb4(numFeatures);
    const int numComb = static_cast<int>(comb4.size());

    Matrix A(row, std::vector<double>(numComb, 0.0));
    if (row == 0 || numComb == 0) {
        return A;
    }

    const double invRows = 1.0 / static_cast<double>(row);
    const int numThreads = getOptimalThreadCount();

    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 1)
    for (int combIdx = 0; combIdx < numComb; ++combIdx) {
        const Comb4 comb = comb4[combIdx];

        std::unordered_map<Key4, int, Key4Hash> counts;
        counts.reserve(static_cast<size_t>(row) * 2);
        std::vector<Key4> rowKeys(static_cast<size_t>(row));

        for (int r = 0; r < row; ++r) {
            const Key4 key{
                bitsOf(base[r][comb.a]),
                bitsOf(base[r][comb.b]),
                bitsOf(base[r][comb.c]),
                bitsOf(base[r][comb.d])
            };
            rowKeys[static_cast<size_t>(r)] = key;
            ++counts[key];
        }

        for (int r = 0; r < row; ++r) {
            A[r][combIdx] = static_cast<double>(counts[rowKeys[static_cast<size_t>(r)]]) * invRows;
        }
    }

    return A;
}

Matrix FKG::calculateA(const Matrix& base) {
    return calculateA_Parallel(base);
}

// ============================================================================
// calculateM: Count matching attributes between rows (EXACT PYTHON)
// ============================================================================

Matrix FKG::calculateM(const Matrix& base) {
    if (base.empty() || base[0].empty()) {
        return {};
    }

    const int row = static_cast<int>(base.size());
    const int colum = static_cast<int>(base[0].size());
    const int numFeatures = colum - 1;

    Matrix M(row, std::vector<double>(numFeatures, 0.0));
    if (row == 0 || numFeatures == 0) {
        return M;
    }

    const double invRows = 1.0 / static_cast<double>(row);
    const int numThreads = getOptimalThreadCount();

    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 1)
    for (int feature = 0; feature < numFeatures; ++feature) {
        std::unordered_map<Key2, int, Key2Hash> counts;
        counts.reserve(static_cast<size_t>(row) * 2);
        std::vector<Key2> rowKeys(static_cast<size_t>(row));

        for (int r = 0; r < row; ++r) {
            const Key2 key{bitsOf(base[r][feature]), bitsOf(base[r][colum - 1])};
            rowKeys[static_cast<size_t>(r)] = key;
            ++counts[key];
        }

        for (int r = 0; r < row; ++r) {
            M[r][feature] = static_cast<double>(counts[rowKeys[static_cast<size_t>(r)]]) * invRows;
        }
    }

    return M;
}

// ============================================================================
// calculateB: sum(A[r]) * min(M[r][a], M[r][b], M[r][c]) (EXACT PYTHON)
// ============================================================================

Matrix FKG::calculateB_Parallel(const Matrix& base, const Matrix& A, const Matrix& M) {
    if (base.empty() || base[0].empty()) {
        return {};
    }

    const int row = static_cast<int>(base.size());
    const int colum = static_cast<int>(base[0].size());
    const int numFeatures = colum - 1;
    const std::vector<Comb3> comb3 = buildComb3(numFeatures);
    const int numComb3 = static_cast<int>(comb3.size());

    Matrix B(row, std::vector<double>(numComb3, 0.0));
    if (row == 0 || numComb3 == 0) {
        return B;
    }

    const int numThreads = getOptimalThreadCount();

    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 256)
    for (int r = 0; r < row; ++r) {
        const double sumA = std::accumulate(A[r].begin(), A[r].end(), 0.0);
        for (int combIdx = 0; combIdx < numComb3; ++combIdx) {
            const Comb3 comb = comb3[combIdx];
            const double minM = std::min({M[r][comb.a], M[r][comb.b], M[r][comb.c]});
            B[r][combIdx] = sumA * minM;
        }
    }

    return B;
}

Matrix FKG::calculateB(const Matrix& base, const Matrix& A, const Matrix& M) {
    return calculateB_Parallel(base, A, M);
}

// ============================================================================
// calculateC: 6 * C(3, n-1) columns, class-based aggregation (EXACT PYTHON)
// ============================================================================

Matrix FKG::calculateC_Parallel(const Matrix& base, const Matrix& B, int n_classes) {
    if (base.empty() || base[0].empty()) {
        return {};
    }

    const int row = static_cast<int>(base.size());
    const int colum = static_cast<int>(base[0].size());
    const int numFeatures = colum - 1;
    const std::vector<Comb3> comb3 = buildComb3(numFeatures);
    const int numComb3 = static_cast<int>(comb3.size());
    const int cols = 6 * numComb3;  // Keep exact legacy output shape.

    Matrix C(row, std::vector<double>(cols, 0.0));
    if (row == 0 || numComb3 == 0) {
        return C;
    }

    const int classCount = std::max(0, std::min(n_classes, 6));
    const int numThreads = getOptimalThreadCount();

    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 1)
    for (int combIdx = 0; combIdx < numComb3; ++combIdx) {
        const Comb3 comb = comb3[combIdx];

        std::unordered_map<Key3Class, double, Key3ClassHash> aggregated;
        aggregated.reserve(static_cast<size_t>(row) * 2);

        for (int r = 0; r < row; ++r) {
            const Key3Class key{
                bitsOf(base[r][comb.a]),
                bitsOf(base[r][comb.b]),
                bitsOf(base[r][comb.c]),
                static_cast<int>(base[r][colum - 1])
            };
            aggregated[key] += B[r][combIdx];
        }

        for (int r = 0; r < row; ++r) {
            const uint64_t v0 = bitsOf(base[r][comb.a]);
            const uint64_t v1 = bitsOf(base[r][comb.b]);
            const uint64_t v2 = bitsOf(base[r][comb.c]);
            for (int label = 1; label <= classCount; ++label) {
                const int cIndex = (label - 1) * numComb3 + combIdx;
                if (cIndex >= cols) {
                    continue;
                }
                const Key3Class key{v0, v1, v2, label};
                auto it = aggregated.find(key);
                C[r][cIndex] = (it != aggregated.end()) ? it->second : 0.0;
            }
        }
    }

    return C;
}

Matrix FKG::calculateC(const Matrix& base, const Matrix& B, int n_classes) {
    return calculateC_Parallel(base, B, n_classes);
}

FKG::ComputedMatrices FKG::computeMatrices(const Matrix& base, int n_classes, bool prefer_gpu) {
    ComputedMatrices result;
    if (base.empty() || base[0].empty()) {
        return result;
    }

    const int classCount = (n_classes > 0) ? n_classes : detectClassCount(base);

#if FUZZY_USE_CUDA
    if (prefer_gpu && isGPUAvailable()) {
        std::string cudaError;
        const cudaError_t status = CUDA::calculateABCM_GPU(
            base, classCount, result.A, result.M, result.B, result.C, &cudaError);
        if (status == cudaSuccess) {
            result.C = minMaxNormalize(result.C);
            return result;
        }
        std::cerr << "CUDA pipeline failed, falling back to CPU: "
                  << cudaError << std::endl;
    }
#else
    (void)prefer_gpu;
#endif

    result.A = calculateA_Parallel(base);
    result.M = calculateM(base);
    result.B = calculateB_Parallel(base, result.A, result.M);
    result.C = calculateC_Parallel(base, result.B, classCount);
    result.C = minMaxNormalize(result.C);
    return result;
}

// ============================================================================
// FISA Inference (EXACT PYTHON)
// ============================================================================

std::pair<int, double> FKG::fisa(const Matrix& base, const Matrix& C,
                                  const std::vector<double>& input, int n_classes) {
    if (base.empty() || base[0].empty()) {
        return {1, 0.0};
    }

    const int colum = static_cast<int>(base[0].size());
    const int row = static_cast<int>(base.size());
    const int cols = combination(3, colum - 1);
    if (cols <= 0 || n_classes <= 0) {
        return {1, 0.0};
    }

    const std::vector<Comb3> comb3 = buildComb3(colum - 1);
    std::vector<std::vector<double>> cByClass(static_cast<size_t>(n_classes + 1),
                                              std::vector<double>(static_cast<size_t>(cols), 0.0));

    for (int combIdx = 0; combIdx < cols; ++combIdx) {
        const Comb3 comb = comb3[combIdx];
        for (int r = 0; r < row - 1; ++r) {
            if (base[r][comb.a] == input[comb.a] &&
                base[r][comb.b] == input[comb.b] &&
                base[r][comb.c] == input[comb.c]) {
                const int label = static_cast<int>(base[r][colum - 1]);
                if (label >= 1 && label <= n_classes) {
                    const int cIndex = combIdx + (label - 1) * cols;
                    if (cIndex < static_cast<int>(C[r].size())) {
                        cByClass[static_cast<size_t>(label)][static_cast<size_t>(combIdx)] =
                            C[r][cIndex];
                    }
                }
            }
        }
    }

    int bestClass = 1;
    double maxD = std::numeric_limits<double>::lowest();
    double sumD = 0.0;

    for (int label = 1; label <= n_classes; ++label) {
        const auto& vec = cByClass[static_cast<size_t>(label)];
        const auto minmax = std::minmax_element(vec.begin(), vec.end());
        const double d = *minmax.second + *minmax.first;
        sumD += d;
        if (d > maxD) {
            maxD = d;
            bestClass = label;
        }
    }

    const double confidence = (sumD > 0.0) ? (maxD / sumD) : 0.0;
    return {bestClass, confidence};
}

FKG::FISAResult FKG::FISAWithConfidence(const Matrix& base, const Matrix& C,
                                         const std::vector<double>& input, int n_classes) {
    FISAResult result;
    result.bestClass = 1;
    result.confidence = 0.0;
    result.D.assign(static_cast<size_t>(std::max(0, n_classes)), 0.0);

    if (base.empty() || base[0].empty()) {
        return result;
    }

    const int colum = static_cast<int>(base[0].size());
    const int row = static_cast<int>(base.size());
    const int cols = combination(3, colum - 1);
    if (cols <= 0 || n_classes <= 0) {
        return result;
    }

    const std::vector<Comb3> comb3 = buildComb3(colum - 1);
    std::vector<std::vector<double>> cByClass(static_cast<size_t>(n_classes + 1),
                                              std::vector<double>(static_cast<size_t>(cols), 0.0));

    for (int combIdx = 0; combIdx < cols; ++combIdx) {
        const Comb3 comb = comb3[combIdx];
        for (int r = 0; r < row - 1; ++r) {
            if (base[r][comb.a] == input[comb.a] &&
                base[r][comb.b] == input[comb.b] &&
                base[r][comb.c] == input[comb.c]) {
                const int label = static_cast<int>(base[r][colum - 1]);
                if (label >= 1 && label <= n_classes) {
                    const int cIndex = combIdx + (label - 1) * cols;
                    if (cIndex < static_cast<int>(C[r].size())) {
                        cByClass[static_cast<size_t>(label)][static_cast<size_t>(combIdx)] =
                            C[r][cIndex];
                    }
                }
            }
        }
    }

    double maxD = std::numeric_limits<double>::lowest();
    double sumD = 0.0;
    for (int label = 1; label <= n_classes; ++label) {
        const auto& vec = cByClass[static_cast<size_t>(label)];
        const auto minmax = std::minmax_element(vec.begin(), vec.end());
        const double d = *minmax.second + *minmax.first;
        result.D[static_cast<size_t>(label - 1)] = d;
        sumD += d;
        if (d > maxD) {
            maxD = d;
            result.bestClass = label;
        }
    }

    result.confidence = (sumD > 0.0) ? (maxD / sumD) : 0.0;
    return result;
}

// ============================================================================
// Normalization
// ============================================================================

Matrix FKG::minMaxNormalize(const Matrix& C) {
    if (C.empty() || C[0].empty()) return C;
    
    int rows = static_cast<int>(C.size());
    int cols = static_cast<int>(C[0].size());
    
    Matrix normalized(rows, std::vector<double>(cols));
    
    std::vector<double> mins(cols, std::numeric_limits<double>::max());
    std::vector<double> maxs(cols, std::numeric_limits<double>::lowest());
    
    #pragma omp parallel
    {
        std::vector<double> local_mins(cols, std::numeric_limits<double>::max());
        std::vector<double> local_maxs(cols, std::numeric_limits<double>::lowest());
        
        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (C[i][j] < local_mins[j]) local_mins[j] = C[i][j];
                if (C[i][j] > local_maxs[j]) local_maxs[j] = C[i][j];
            }
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < cols; j++) {
                if (local_mins[j] < mins[j]) mins[j] = local_mins[j];
                if (local_maxs[j] > maxs[j]) maxs[j] = local_maxs[j];
            }
        }
    }
    
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double range = maxs[j] - mins[j];
            normalized[i][j] = (range > 0) ? ((C[i][j] - mins[j]) / range) : 0.0;
        }
    }
    
    return normalized;
}

Matrix FKG::gaussianNormalize(const Matrix& C) {
    if (C.empty() || C[0].empty()) return C;
    
    int rows = static_cast<int>(C.size());
    int cols = static_cast<int>(C[0].size());
    
    Matrix normalized(rows, std::vector<double>(cols));
    
    // Compute mean
    std::vector<double> sum(cols, 0.0);
    #pragma omp parallel
    {
        std::vector<double> local_sum(cols, 0.0);
        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                local_sum[j] += C[i][j];
            }
        }
        #pragma omp critical
        {
            for (int j = 0; j < cols; j++) {
                sum[j] += local_sum[j];
            }
        }
    }
    
    std::vector<double> mean(cols);
    for (int j = 0; j < cols; j++) {
        mean[j] = sum[j] / rows;
    }
    
    // Compute variance
    std::vector<double> sq_diff_sum(cols, 0.0);
    #pragma omp parallel
    {
        std::vector<double> local_sq_diff(cols, 0.0);
        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = C[i][j] - mean[j];
                local_sq_diff[j] += diff * diff;
            }
        }
        #pragma omp critical
        {
            for (int j = 0; j < cols; j++) {
                sq_diff_sum[j] += local_sq_diff[j];
            }
        }
    }
    
    std::vector<double> std_dev(cols);
    for (int j = 0; j < cols; j++) {
        std_dev[j] = std::sqrt(sq_diff_sum[j] / rows);
        if (std_dev[j] < 1e-10) std_dev[j] = 1.0;
    }
    
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            normalized[i][j] = (C[i][j] - mean[j]) / std_dev[j];
        }
    }
    
    return normalized;
}

// ============================================================================
// Metrics
// ============================================================================

double FKG::accuracy(const std::vector<int>& predicted, const std::vector<int>& actual) {
    if (predicted.empty()) return 0.0;

    int correct = 0;
    int n = static_cast<int>(predicted.size());
    int numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads) reduction(+:correct)
    for (int i = 0; i < n; i++) {
        if (predicted[i] == actual[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) * 100.0 / n;
}

std::vector<double> FKG::precisionPerClass(const std::vector<int>& predicted, 
                                             const std::vector<int>& actual) {
    std::unordered_set<int> uniqueLabels;
    for (int label : predicted) uniqueLabels.insert(label);
    for (int label : actual) uniqueLabels.insert(label);
    
    std::vector<int> labels(uniqueLabels.begin(), uniqueLabels.end());
    std::sort(labels.begin(), labels.end());
    
    std::vector<double> precisions;
    for (int label : labels) {
        int TP = 0;
        int FP = 0;

        #pragma omp parallel for reduction(+:TP,FP)
        for (int i = 0; i < static_cast<int>(predicted.size()); i++) {
            if (predicted[i] == label && actual[i] == label) TP++;
            if (predicted[i] == label && actual[i] != label) FP++;
        }
        
        if (TP + FP > 0) {
            precisions.push_back(static_cast<double>(TP) / (TP + FP) * 100.0);
        } else {
            precisions.push_back(0.0);
        }
    }
    
    return precisions;
}

std::vector<double> FKG::recallPerClass(const std::vector<int>& predicted, 
                                          const std::vector<int>& actual) {
    std::unordered_set<int> uniqueLabels;
    for (int label : predicted) uniqueLabels.insert(label);
    for (int label : actual) uniqueLabels.insert(label);
    
    std::vector<int> labels(uniqueLabels.begin(), uniqueLabels.end());
    std::sort(labels.begin(), labels.end());
    
    std::vector<double> recalls;
    for (int label : labels) {
        int TP = 0;
        int FN = 0;

        #pragma omp parallel for reduction(+:TP,FN)
        for (int i = 0; i < static_cast<int>(predicted.size()); i++) {
            if (predicted[i] == label && actual[i] == label) TP++;
            if (predicted[i] != label && actual[i] == label) FN++;
        }
        
        if (TP + FN > 0) {
            recalls.push_back(static_cast<double>(TP) / (TP + FN) * 100.0);
        } else {
            recalls.push_back(0.0);
        }
    }
    
    return recalls;
}

std::vector<double> FKG::f1PerClass(const std::vector<int>& predicted, 
                                      const std::vector<int>& actual) {
    std::vector<double> precision = precisionPerClass(predicted, actual);
    std::vector<double> recall = recallPerClass(predicted, actual);
    
    std::vector<double> f1;
    for (size_t i = 0; i < precision.size(); i++) {
        if (precision[i] + recall[i] > 0) {
            f1.push_back(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]));
        } else {
            f1.push_back(0.0);
        }
    }
    
    return f1;
}

// ============================================================================
// FKGS Class Implementation
// ============================================================================

FKGS::FKGS() : FKG(), ran_(50.0), e_(0.1) {}

FKGS::FKGS(double ran, double e) : FKG(), ran_(ran), e_(e) {}

FKGS::FKGS(const PerformanceConfig& config, double ran, double e) 
    : FKG(config), ran_(ran), e_(e) {}

FKGS::~FKGS() {}

// diff: count matching attributes / (m-1) (EXACT PYTHON)
double FKGS::diff(const std::vector<int>& Rule1, const std::vector<int>& Rule2) {
    // Check class match first
    if (Rule1.back() != Rule2.back()) {
        return -1.0;
    }
    
    int m = static_cast<int>(Rule1.size());
    int count = 0;
    
    for (int i = 0; i < m - 1; i++) {
        if (Rule1[i] == Rule2[i]) {
            count++;
        }
    }
    
    return static_cast<double>(count) / (m - 1);
}

// sampling: k-nearest neighbor based sampling (EXACT PYTHON)
Matrix FKGS::samplingParallel(const Matrix& base, double ran, double e, int numThreads) {
    int num = static_cast<int>(base.size());
    int targetSize = static_cast<int>(num * ran / 100.0);
    int k = 2;  // k-nearest neighbors parameter
    
    if (numThreads <= 0) numThreads = getOptimalThreadCount();
    
    Matrix R;
    std::vector<int> listIndex;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, num - 1);

    auto toIntRule = [](const std::vector<double>& row) {
        std::vector<int> out(row.size(), 0);
        for (size_t i = 0; i < row.size(); ++i) {
            out[i] = static_cast<int>(std::llround(row[i]));
        }
        return out;
    };
    
    while (static_cast<int>(R.size()) < targetSize) {
        int index = dis(gen);
        
        bool found = false;
        for (int idx : listIndex) {
            if (idx == index) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            // Build T with k-nearest neighbors
            std::vector<int> T;
            T.push_back(index);
            
            // Add neighbors within k distance
            for (int i = std::max(0, index - k); i <= std::min(num - 1, index + k); i++) {
                if (i == index) continue;
                
                bool alreadyInList = false;
                for (int idx : listIndex) {
                    if (i == idx) {
                        alreadyInList = true;
                        break;
                    }
                }
                if (alreadyInList) continue;
                
                // Convert to int vectors for diff calculation
                std::vector<int> r1 = toIntRule(base[index]);
                std::vector<int> r2 = toIntRule(base[i]);
                
                if (diff(r1, r2) < 1.0 - e) {
                    T.push_back(i);
                }
            }
            
            // Filter T based on mutual similarity
            std::vector<int> filteredT;
            for (int i : T) {
                bool temp = true;
                for (int j = 0; j < static_cast<int>(R.size()); j++) {
                    std::vector<int> r1 = toIntRule(base[i]);
                    std::vector<int> r2 = toIntRule(R[j]);
                    
                    if (diff(r1, r2) >= 1.0 - e) {
                        temp = false;
                        break;
                    }
                }
                if (temp) {
                    filteredT.push_back(i);
                }
            }
            
            // Add to R
            for (int i : filteredT) {
                R.push_back(base[i]);
                listIndex.push_back(i);
            }
        }
    }
    
    return R;
}

Matrix FKGS::sampling(const Matrix& base, double ran, double e) {
    return samplingParallel(base, ran, e, getOptimalThreadCount());
}

#if FUZZY_USE_CUDA
void FKG::invalidateGPUCache() const {
    if (cudaInferenceCache_ == nullptr) {
        return;
    }
    CudaInferenceCacheHandle* handle =
        reinterpret_cast<CudaInferenceCacheHandle*>(cudaInferenceCache_);
    CUDA::destroyFisaDeviceCache(handle->cache);
    delete handle;
    cudaInferenceCache_ = nullptr;
}

bool FKG::ensureGPUCache() const {
    if (cudaInferenceCache_ != nullptr) {
        return true;
    }
    if (!trained_ || base_.empty() || C_.empty()) {
        return false;
    }

    CudaInferenceCacheHandle* handle = new CudaInferenceCacheHandle();
    std::string cudaError;
    const cudaError_t status = CUDA::createFisaDeviceCache(base_, C_, n_classes_,
                                                           handle->cache, &cudaError);
    if (status != cudaSuccess) {
        std::cerr << "CUDA cache init failed, fallback to CPU: " << cudaError << std::endl;
        delete handle;
        return false;
    }
    cudaInferenceCache_ = handle;
    return true;
}

std::pair<int, double> FKG::predictGPUCached(const std::vector<double>& input) const {
    if (!ensureGPUCache()) {
        return fisa(base_, C_, input, n_classes_);
    }

    CudaInferenceCacheHandle* handle =
        reinterpret_cast<CudaInferenceCacheHandle*>(cudaInferenceCache_);
    int resultClass = 1;
    double resultConfidence = 0.0;
    std::string cudaError;
    const cudaError_t status = CUDA::fisaGPUWithCache(handle->cache, input, resultClass,
                                                      resultConfidence, nullptr, &cudaError);
    if (status != cudaSuccess) {
        std::cerr << "CUDA cached predict failed, fallback to CPU: " << cudaError << std::endl;
        invalidateGPUCache();
        return fisa(base_, C_, input, n_classes_);
    }
    return {resultClass, resultConfidence};
}

std::vector<std::pair<int, double>> FKG::predictBatchWithConfidenceGPUCached(
    const Matrix& inputs) const {
    if (!ensureGPUCache()) {
        std::vector<std::pair<int, double>> cpuResults;
        cpuResults.reserve(inputs.size());
        for (const auto& input : inputs) {
            cpuResults.push_back(fisa(base_, C_, input, n_classes_));
        }
        return cpuResults;
    }

    CudaInferenceCacheHandle* handle =
        reinterpret_cast<CudaInferenceCacheHandle*>(cudaInferenceCache_);
    std::vector<int> resultClasses;
    std::vector<double> resultConfidences;
    std::string cudaError;
    const cudaError_t status = CUDA::fisaBatchGPUWithCache(handle->cache, inputs, resultClasses,
                                                           resultConfidences, &cudaError);
    if (status != cudaSuccess) {
        std::cerr << "CUDA batch cached predict failed, fallback to CPU: "
                  << cudaError << std::endl;
        invalidateGPUCache();
        std::vector<std::pair<int, double>> cpuResults;
        cpuResults.reserve(inputs.size());
        for (const auto& input : inputs) {
            cpuResults.push_back(fisa(base_, C_, input, n_classes_));
        }
        return cpuResults;
    }

    std::vector<std::pair<int, double>> out;
    out.reserve(resultClasses.size());
    for (size_t i = 0; i < resultClasses.size(); ++i) {
        out.emplace_back(resultClasses[i], resultConfidences[i]);
    }
    return out;
}
#endif

// ============================================================================
// GPU Interface (runtime fallback to CPU)
// ============================================================================

bool FKG::isGPUCompiled() {
#if FUZZY_USE_CUDA || FUZZY_USE_GPU
    return true;
#else
    return false;
#endif
}

bool FKG::isGPUAvailable() {
#if FUZZY_USE_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess) && (deviceCount > 0);
#elif FUZZY_USE_GPU
    try {
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        return !devices.empty();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

Matrix FKG::calculateA_GPU(const Matrix& base) {
#if FUZZY_USE_CUDA
    if (!isGPUAvailable()) {
        return calculateA_Parallel(base);
    }
    std::vector<std::vector<double>> A;
    std::string cudaError;
    cudaError_t err = CUDA::calculateA_GPU(base, A, &cudaError);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in calculateA_GPU, falling back to CPU: "
                  << cudaError << std::endl;
        return calculateA_Parallel(base);
    }
    return A;
#else
    return calculateA_Parallel(base);
#endif
}

Matrix FKG::calculateM_GPU(const Matrix& base) {
#if FUZZY_USE_CUDA
    if (!isGPUAvailable()) {
        return calculateM(base);
    }
    std::vector<std::vector<double>> M;
    std::string cudaError;
    cudaError_t err = CUDA::calculateM_GPU(base, M, &cudaError);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in calculateM_GPU, falling back to CPU: "
                  << cudaError << std::endl;
        return calculateM(base);
    }
    return M;
#else
    return calculateM(base);
#endif
}

Matrix FKG::calculateB_GPU(const Matrix& base, const Matrix& A, const Matrix& M) {
#if FUZZY_USE_CUDA
    if (!isGPUAvailable()) {
        return calculateB_Parallel(base, A, M);
    }
    int colum = static_cast<int>(base[0].size());
    std::vector<std::vector<double>> B;
    std::string cudaError;
    cudaError_t err = CUDA::calculateB_GPU(A, M, B, colum, &cudaError);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in calculateB_GPU, falling back to CPU: "
                  << cudaError << std::endl;
        return calculateB_Parallel(base, A, M);
    }
    return B;
#else
    return calculateB_Parallel(base, A, M);
#endif
}

Matrix FKG::calculateC_GPU(const Matrix& base, const Matrix& B, int n_classes) {
#if FUZZY_USE_CUDA
    if (!isGPUAvailable()) {
        return calculateC_Parallel(base, B, n_classes);
    }
    int colum = static_cast<int>(base[0].size());
    std::vector<std::vector<double>> C;
    std::string cudaError;
    cudaError_t err = CUDA::calculateC_GPU(base, B, C, colum, n_classes, &cudaError);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in calculateC_GPU, falling back to CPU: "
                  << cudaError << std::endl;
        return calculateC_Parallel(base, B, n_classes);
    }
    return C;
#else
    return calculateC_Parallel(base, B, n_classes);
#endif
}

std::pair<int, double> FKG::fisaGPU(const Matrix& base, const Matrix& C,
                                   const std::vector<double>& input, int n_classes) {
#if FUZZY_USE_CUDA
    if (!isGPUAvailable()) {
        return fisa(base, C, input, n_classes);
    }
    int result_class = 1;
    double result_confidence = 0.0;
    std::string cudaError;
    cudaError_t err = CUDA::fisaGPU(base, C, input, n_classes, result_class, result_confidence,
                                    nullptr, &cudaError);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in fisaGPU, falling back to CPU: "
                  << cudaError << std::endl;
        return fisa(base, C, input, n_classes);
    }
    return {result_class, result_confidence};
#else
    return fisa(base, C, input, n_classes);
#endif
}

FKG::FISAResult FKG::FISAWithConfidenceGPU(const Matrix& base, const Matrix& C,
                                           const std::vector<double>& input, int n_classes) {
#if FUZZY_USE_CUDA
    if (!isGPUAvailable()) {
        return FISAWithConfidence(base, C, input, n_classes);
    }
    FISAResult result;
    result.bestClass = 1;
    result.confidence = 0.0;

    std::string cudaError;
    cudaError_t err = CUDA::fisaGPU(base, C, input, n_classes, result.bestClass,
                                    result.confidence, &result.D, &cudaError);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in FISAWithConfidenceGPU, falling back to CPU: "
                  << cudaError << std::endl;
        return FISAWithConfidence(base, C, input, n_classes);
    }
    return result;
#else
    return FISAWithConfidence(base, C, input, n_classes);
#endif
}

bool FKG::verifyGPUvsCPU(const Matrix& testData, double tolerance) {
    if (testData.empty()) {
        return false;
    }

    const bool originalUseGPU = getUseGPU();
    setUseGPU(false);
    train(testData);
    std::vector<int> cpuPred = predictBatch(testData);

    setUseGPU(true);
    train(testData);
    std::vector<int> gpuPred = predictBatch(testData);
    setUseGPU(originalUseGPU);

    if (cpuPred.size() != gpuPred.size()) {
        return false;
    }
    for (size_t i = 0; i < cpuPred.size(); i++) {
        if (std::abs(cpuPred[i] - gpuPred[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

FKG::BenchmarkResult FKG::benchmark(const Matrix& testData) {
    BenchmarkResult result = {0.0, 0.0, 1.0, false, 0.0};
    if (testData.empty()) {
        return result;
    }

    const bool originalUseGPU = getUseGPU();

    auto cpuStart = std::chrono::high_resolution_clock::now();
    setUseGPU(false);
    train(testData);
    std::vector<int> cpuPred = predictBatch(testData);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    result.cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    auto gpuStart = std::chrono::high_resolution_clock::now();
    setUseGPU(true);
    train(testData);
    std::vector<int> gpuPred = predictBatch(testData);
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    result.gpuTimeMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();

    if (result.gpuTimeMs > 0.0) {
        result.speedup = result.cpuTimeMs / result.gpuTimeMs;
    }

    result.resultsMatch = (cpuPred.size() == gpuPred.size());
    if (result.resultsMatch) {
        for (size_t i = 0; i < cpuPred.size(); i++) {
            const double diff = std::abs(cpuPred[i] - gpuPred[i]);
            if (diff > result.maxDiff) {
                result.maxDiff = diff;
            }
            if (cpuPred[i] != gpuPred[i]) {
                result.resultsMatch = false;
            }
        }
    }

    setUseGPU(originalUseGPU);
    return result;
}

} // namespace Fuzzy
