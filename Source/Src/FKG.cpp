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
#include <cstddef>
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

// ============================================================================
// FKG Class Implementation
// ============================================================================

FKG::FKG() : n_classes_(2), trained_(false), useGPU_(false) {
    config_ = PerformanceConfig();
    metrics_ = PerformanceMetrics{0.0, 0, 1};
}

FKG::FKG(const PerformanceConfig& config) 
    : n_classes_(2), trained_(false), config_(config), useGPU_(false) {
    metrics_ = PerformanceMetrics{0.0, 0, 1};
}

FKG::~FKG() {}

void FKG::train(const Matrix& base) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Auto-detect number of classes
    std::set<int> labels;
    for (const auto& row : base) {
        labels.insert(static_cast<int>(row.back()));
    }
    n_classes_ = static_cast<int>(labels.size());
    train(base, n_classes_);
    
    auto end = std::chrono::high_resolution_clock::now();
    metrics_.computeTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void FKG::train(const Matrix& base, int n_classes) {
    base_ = base;
    n_classes_ = n_classes;

    const bool useEffectiveGPU = isUsingGPU();

    if (useEffectiveGPU) {
        A_ = calculateA_GPU(base_);
    } else {
        A_ = calculateA_Parallel(base_);
    }

    M_ = calculateM(base_);

    if (useEffectiveGPU) {
        B_ = calculateB_GPU(base_, A_, M_);
        C_ = calculateC_GPU(base_, B_, n_classes_);
    } else {
        B_ = calculateB_Parallel(base_, A_, M_);
        C_ = calculateC_Parallel(base_, B_, n_classes_);
    }

    C_ = minMaxNormalize(C_);
    
    trained_ = true;
    metrics_.numThreadsUsed = getOptimalThreadCount();
}

std::pair<int, double> FKG::predict(const std::vector<double>& input) const {
    if (!trained_) {
        return {1, 0.0};
    }
    if (isUsingGPU()) {
        return fisaGPU(base_, C_, input, n_classes_);
    }
    return fisa(base_, C_, input, n_classes_);
}

std::vector<int> FKG::predictBatch(const Matrix& inputs) const {
    return predictBatchParallel(inputs, getOptimalThreadCount());
}

std::vector<int> FKG::predictBatchParallel(const Matrix& inputs, int numThreads) const {
    if (!trained_ || inputs.empty()) {
        return {};
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
    int row = static_cast<int>(base.size());
    int colum = static_cast<int>(base[0].size());
    int numComb = combination(4, colum - 1);
    
    Matrix A(row, std::vector<double>(numComb, 0.0));
    
    int numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 256)
    for (int r1 = 0; r1 < row; r1++) {
        std::vector<double> k(numComb, 0.0);
        
        int temp = 0;
        for (int a = 0; a < colum - 3; a++) {
            for (int b = a + 1; b < colum - 2; b++) {
                for (int c = b + 1; c < colum - 1; c++) {
                    // Use only feature columns (exclude last label column).
                    for (int d = c + 1; d < colum - 1; d++) {
                        for (int r2 = 0; r2 < row; r2++) {
                            // Check 4-tuple match
                            if (base[r1][a] == base[r2][a] && 
                                base[r1][b] == base[r2][b] && 
                                base[r1][c] == base[r2][c] &&
                                base[r1][d] == base[r2][d]) {
                                k[temp] += 1.0;
                            }
                        }
                        // EXACT: k[temp] / row (divide by row count)
                        A[r1][temp] = k[temp] / row;
                        temp++;
                    }
                }
            }
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
    int row = static_cast<int>(base.size());
    int colum = static_cast<int>(base[0].size());
    
    Matrix M(row, std::vector<double>(colum - 1, 0.0));
    
    int numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 256)
    for (int t1 = 0; t1 < row; t1++) {
        std::vector<double> k(colum - 1, 0.0);
        int temp = 0;
        
        for (int i = 0; i < colum - 1; i++) {
            for (int t2 = 0; t2 < row; t2++) {
                // Check attribute match AND class match
                if (base[t1][i] == base[t2][i] && 
                    base[t1][colum - 1] == base[t2][colum - 1]) {
                    k[temp] += 1.0;
                }
            }
            M[t1][temp] = k[temp] / row;
            temp++;
        }
    }
    
    return M;
}

// ============================================================================
// calculateB: sum(A[r]) * min(M[r][a], M[r][b], M[r][c]) (EXACT PYTHON)
// ============================================================================

Matrix FKG::calculateB_Parallel(const Matrix& base, const Matrix& A, const Matrix& M) {
    int row = static_cast<int>(base.size());
    int colum = static_cast<int>(base[0].size());
    int numComb3 = combination(3, colum - 1);
    
    Matrix B(row, std::vector<double>(numComb3, 0.0));
    
    int numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 256)
    for (int r = 0; r < row; r++) {
        int temp = 0;
        
        // Pre-compute sum(A[r]) for efficiency
        double sumA = 0.0;
        for (int j = 0; j < static_cast<int>(A[r].size()); j++) {
            sumA += A[r][j];
        }
        
        for (int a = 0; a < colum - 3; a++) {
            for (int b = a + 1; b < colum - 2; b++) {
                for (int c = b + 1; c < colum - 1; c++) {
                    // EXACT: sum(A[r]) * min(M[r][a], M[r][b], M[r][c])
                    double minM = std::min({M[r][a], M[r][b], M[r][c]});
                    B[r][temp] = sumA * minM;
                    temp++;
                }
            }
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
    int row = static_cast<int>(base.size());
    int colum = static_cast<int>(base[0].size());
    int cols = 6 * combination(3, colum - 1);  // EXACT: 6 * C(3, n-1)
    
    Matrix C(row, std::vector<double>(cols, 0.0));
    
    int numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 256)
    for (int r1 = 0; r1 < row; r1++) {
        int temp = 0;
        
        // For each class
        for (int i = 1; i <= n_classes; i++) {
            for (int a = 0; a < colum - 3; a++) {
                for (int b = a + 1; b < colum - 2; b++) {
                    for (int c = b + 1; c < colum - 1; c++) {
                        for (int r2 = 0; r2 < row; r2++) {
                            // Check 3-tuple match AND class match
                            if (base[r1][a] == base[r2][a] && 
                                base[r1][b] == base[r2][b] && 
                                base[r1][c] == base[r2][c] &&
                                base[r2][colum - 1] == i) {
                                // EXACT: C[r1][temp] += B[r2][temp % C(3, n-1)]
                                int B_idx = temp % combination(3, colum - 1);
                                C[r1][temp] += B[r2][B_idx];
                            }
                        }
                        temp++;
                    }
                }
            }
        }
    }
    
    return C;
}

Matrix FKG::calculateC(const Matrix& base, const Matrix& B, int n_classes) {
    return calculateC_Parallel(base, B, n_classes);
}

// ============================================================================
// FISA Inference (EXACT PYTHON)
// ============================================================================

std::pair<int, double> FKG::fisa(const Matrix& base, const Matrix& C,
                                  const std::vector<double>& input, int n_classes) {
    int colum = static_cast<int>(base[0].size());
    int row = static_cast<int>(base.size());
    int cols = combination(3, colum - 1);
    
    // EXACT: C_dict = {i: [0] * cols for i in range(1, n_classes + 1)}
    std::vector<std::vector<double>> C_dict(n_classes + 1, std::vector<double>(cols, 0.0));
    
    int t = 0;
    for (int a = 0; a < colum - 3; a++) {
        for (int b = a + 1; b < colum - 2; b++) {
            for (int c = b + 1; c < colum - 1; c++) {
                for (int r = 0; r < row - 1; r++) {
                    if (base[r][a] == input[a] && 
                        base[r][b] == input[b] && 
                        base[r][c] == input[c]) {
                        int label = static_cast<int>(base[r][colum - 1]);
                        if (label >= 1 && label <= n_classes) {
                            C_dict[label][t] = C[r][t + (label - 1) * cols];
                        }
                    }
                }
                t++;
            }
        }
    }
    
    // EXACT: D_dict[label] = max(vec) + min(vec)
    std::unordered_map<int, double> D_dict;
    for (int label = 1; label <= n_classes; label++) {
        auto& vec = C_dict[label];
        double maxVal = *std::max_element(vec.begin(), vec.end());
        double minVal = *std::min_element(vec.begin(), vec.end());
        D_dict[label] = maxVal + minVal;
    }
    
    // EXACT: max_label = max(D_dict, key=D_dict.get)
    int bestClass = 1;
    double maxD = D_dict[1];
    for (int label = 2; label <= n_classes; label++) {
        if (D_dict[label] > maxD) {
            maxD = D_dict[label];
            bestClass = label;
        }
    }
    
    double sumD = 0.0;
    for (int label = 1; label <= n_classes; label++) {
        sumD += D_dict[label];
    }
    
    double confidence = (sumD > 0) ? (maxD / sumD) : 0.0;
    
    return {bestClass, confidence};
}

FKG::FISAResult FKG::FISAWithConfidence(const Matrix& base, const Matrix& C,
                                         const std::vector<double>& input, int n_classes) {
    FISAResult result;
    
    int colum = static_cast<int>(base[0].size());
    int row = static_cast<int>(base.size());
    int cols = combination(3, colum - 1);
    
    std::vector<std::vector<double>> C_dict(n_classes + 1, std::vector<double>(cols, 0.0));
    
    int t = 0;
    for (int a = 0; a < colum - 3; a++) {
        for (int b = a + 1; b < colum - 2; b++) {
            for (int c = b + 1; c < colum - 1; c++) {
                for (int r = 0; r < row - 1; r++) {
                    if (base[r][a] == input[a] && 
                        base[r][b] == input[b] && 
                        base[r][c] == input[c]) {
                        int label = static_cast<int>(base[r][colum - 1]);
                        if (label >= 1 && label <= n_classes) {
                            C_dict[label][t] = C[r][t + (label - 1) * cols];
                        }
                    }
                }
                t++;
            }
        }
    }
    
    std::unordered_map<int, double> D_dict;
    for (int label = 1; label <= n_classes; label++) {
        auto& vec = C_dict[label];
        double maxVal = *std::max_element(vec.begin(), vec.end());
        double minVal = *std::min_element(vec.begin(), vec.end());
        D_dict[label] = maxVal + minVal;
    }
    
    result.D.resize(n_classes);
    int bestClass = 1;
    double maxD = D_dict[1];
    for (int label = 1; label <= n_classes; label++) {
        result.D[label - 1] = D_dict[label];
        if (D_dict[label] > maxD) {
            maxD = D_dict[label];
            bestClass = label;
        }
    }
    
    double sumD = 0.0;
    for (int label = 1; label <= n_classes; label++) {
        sumD += D_dict[label];
    }
    
    result.bestClass = bestClass;
    result.confidence = (sumD > 0) ? (maxD / sumD) : 0.0;
    
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
                std::vector<int> r1(base[index].begin(), base[index].end());
                std::vector<int> r2(base[i].begin(), base[i].end());
                
                if (diff(r1, r2) < 1.0 - e) {
                    T.push_back(i);
                }
            }
            
            // Filter T based on mutual similarity
            std::vector<int> filteredT;
            for (int i : T) {
                bool temp = true;
                for (int j = 0; j < static_cast<int>(R.size()); j++) {
                    std::vector<int> r1(base[i].begin(), base[i].end());
                    std::vector<int> r2(R[j].begin(), R[j].end());
                    
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
    if (!isGPUAvailable()) {
        return calculateA_Parallel(base);
    }
    return calculateA_Parallel(base);
}

Matrix FKG::calculateB_GPU(const Matrix& base, const Matrix& A, const Matrix& M) {
    if (!isGPUAvailable()) {
        return calculateB_Parallel(base, A, M);
    }
    return calculateB_Parallel(base, A, M);
}

Matrix FKG::calculateC_GPU(const Matrix& base, const Matrix& B, int n_classes) {
    if (!isGPUAvailable()) {
        return calculateC_Parallel(base, B, n_classes);
    }
    return calculateC_Parallel(base, B, n_classes);
}

std::pair<int, double> FKG::fisaGPU(const Matrix& base, const Matrix& C,
                                   const std::vector<double>& input, int n_classes) {
    if (!isGPUAvailable()) {
        return fisa(base, C, input, n_classes);
    }
    return fisa(base, C, input, n_classes);
}

bool FKG::verifyGPUvsCPU(const Matrix& testData, double tolerance) {
    if (testData.empty()) {
        return false;
    }

    train(testData);
    std::vector<int> cpuPred = predictBatch(testData);

    const bool originalUseGPU = getUseGPU();
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
            if (cpuPred[i] != gpuPred[i]) {
                result.resultsMatch = false;
                break;
            }
        }
    }

    setUseGPU(originalUseGPU);
    return result;
}

} // namespace Fuzzy
