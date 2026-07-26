/**
 * @file FIS.cpp
 * @brief Fuzzy Inference System (FIS) - High-Performance Implementation
 * @version 2.0
 * 
 * Key Optimizations:
 * 1. OpenMP parallel FCM with thread pooling
 * 2. SIMD-friendly loops
 * 3. Reduced memory allocations
 * 4. Early exit conditions
 * 5. Atomic operations for thread safety
 */

#include "FIS.h"
#if FUZZY_USE_CUDA
#include "FIS_CUDA_Kernels.h"
#endif
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <chrono>
#include <limits>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Fuzzy {

// Type alias
using IntMatrix = std::vector<std::vector<int>>;

// Get optimal thread count
static int getOptimalThreadCount(int requested = 0) {
    if (requested > 0) return requested;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ============================================================================
// FIS Class Implementation
// ============================================================================

FIS::FIS() : m_(2.0), eps_(1e-5), max_iter_(200), trained_(false), useGPU_(false) {
    config_ = FISPerformanceConfig();
}

FIS::FIS(const FISPerformanceConfig& config) 
    : m_(config.m), eps_(config.eps), max_iter_(config.maxIter), trained_(false),
      config_(config), useGPU_(false) {
    if (config.numThreads > 0) {
        config_.numThreads = config.numThreads;
    }
}

FIS::FIS(const std::vector<int>& n_clusters, double m, double eps, int max_iter)
    : n_clusters_(n_clusters), m_(m), eps_(eps), max_iter_(max_iter), trained_(false), useGPU_(false) {
    config_ = FISPerformanceConfig();
    config_.m = m;
    config_.eps = eps;
    config_.maxIter = max_iter;
}

FIS::~FIS() {}

void FIS::train(const Matrix& data) {
    trainParallel(data, config_.numThreads);
}

void FIS::trainParallel(const Matrix& data, int numThreads) {
    if (data.empty() || data[0].size() < 2) {
        trained_ = false;
        rules_.clear();
        centers_.clear();
        sigma_.clear();
        min_vals_.clear();
        max_vals_.clear();
        return;
    }

    const int w = static_cast<int>(data[0].size());
    const int rowCount = static_cast<int>(data.size());

    // Calculate min/max values for all columns (features + label column for output FCM init).
    min_vals_.assign(w, std::numeric_limits<double>::max());
    max_vals_.assign(w, std::numeric_limits<double>::lowest());

    if (numThreads <= 0) numThreads = getOptimalThreadCount();

    #pragma omp parallel num_threads(numThreads)
    {
        std::vector<double> local_mins(static_cast<size_t>(w));
        std::vector<double> local_maxs(static_cast<size_t>(w));

        for (int i = 0; i < w; ++i) {
            local_mins[static_cast<size_t>(i)] = data[0][i];
            local_maxs[static_cast<size_t>(i)] = data[0][i];
        }

        #pragma omp for schedule(dynamic, 256)
        for (int r = 0; r < rowCount; ++r) {
            for (int i = 0; i < w; ++i) {
                if (data[r][i] < local_mins[static_cast<size_t>(i)]) {
                    local_mins[static_cast<size_t>(i)] = data[r][i];
                }
                if (data[r][i] > local_maxs[static_cast<size_t>(i)]) {
                    local_maxs[static_cast<size_t>(i)] = data[r][i];
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < w; ++i) {
                if (local_mins[static_cast<size_t>(i)] < min_vals_[static_cast<size_t>(i)]) {
                    min_vals_[static_cast<size_t>(i)] = local_mins[static_cast<size_t>(i)];
                }
                if (local_maxs[static_cast<size_t>(i)] > max_vals_[static_cast<size_t>(i)]) {
                    max_vals_[static_cast<size_t>(i)] = local_maxs[static_cast<size_t>(i)];
                }
            }
        }
    }

    // Ensure cluster vector has one value per column (including output/label column).
    if (n_clusters_.empty()) {
        n_clusters_.assign(static_cast<size_t>(w), 3);
    } else if (static_cast<int>(n_clusters_.size()) < w) {
        const int fallbackCluster = std::max(2, n_clusters_.back());
        n_clusters_.resize(static_cast<size_t>(w), fallbackCluster);
    }

    // Output cluster count should reflect class cardinality (at least 2).
    std::unordered_set<int> labelSet;
    labelSet.reserve(static_cast<size_t>(rowCount));
    for (const auto& r : data) {
        labelSet.insert(static_cast<int>(r.back()));
    }
    const int detectedOutputClusters = std::max(2, static_cast<int>(labelSet.size()));
    n_clusters_[static_cast<size_t>(w - 1)] =
        std::max(detectedOutputClusters, n_clusters_[static_cast<size_t>(w - 1)]);

    RuleGenerationResult result;
    if (isUsingGPU()) {
        result = ruleGenerateGPU(data, n_clusters_, min_vals_, max_vals_, m_, eps_, max_iter_);
    } else {
        result = ruleGenerate(data, n_clusters_, min_vals_, max_vals_, m_, eps_, max_iter_);
    }

    rules_ = result.rules;
    centers_ = result.centers;

    // Sigma only applies to input features.
    sigma_.resize(static_cast<size_t>(w - 1));
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < w - 1; ++i) {
        sigma_[static_cast<size_t>(i)] = computeSigma(centers_[static_cast<size_t>(i)]);
    }

    trained_ = true;
}

int FIS::predict(const std::vector<double>& input) const {
    if (!trained_) {
        return 1;
    }
    
    auto fuzzy_input = fuzzifyInput(input, sigma_, centers_);
    return matchRule(fuzzy_input, rules_);
}

std::vector<int> FIS::predictBatch(const Matrix& inputs) const {
    return predictBatchParallel(inputs, config_.numThreads);
}

std::vector<int> FIS::predictBatchParallel(const Matrix& inputs, int numThreads) const {
    if (!trained_ || inputs.empty()) {
        return {};
    }
    
    int n = static_cast<int>(inputs.size());
    std::vector<int> predictions(n);
    
    if (numThreads <= 0) numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 1024)
    for (int i = 0; i < n; i++) {
        predictions[i] = predict(inputs[i]);
    }
    
    return predictions;
}

// ============================================================================
// Static Methods - SIMD-Optimized Membership Functions
// ============================================================================

double FIS::gaussMF(double x, double sigma, double center) {
    if (sigma <= 0) sigma = 1e-10;
    double diff = x - center;
    return std::exp(-(diff * diff) / (2.0 * sigma * sigma));
}

double FIS::gaussMF(double x, int label, int MFnumber, double sigma,
                    const std::vector<double>& centers) {
    if (label >= 1 && label <= MFnumber) {
        return gaussMF(x, sigma, centers[label - 1]);
    }
    return 0.0;
}

double FIS::triangleMF(double x, double a, double b, double c) {
    if (x <= a || x >= c) return 0.0;
    if (x < b) return (x - a) / (b - a);
    return (c - x) / (c - b);
}

double FIS::trapezoidMF(double x, double a, double b, double c, double d) {
    if (x <= a || x >= d) return 0.0;
    if (x >= b && x <= c) return 1.0;
    if (x < b) return (x - a) / (b - a);
    return (d - x) / (d - c);
}

double FIS::sigmoidMF(double x, double center, double slope) {
    return 1.0 / (1.0 + std::exp(-slope * (x - center)));
}

double FIS::expMF(double x, double center, double sigma) {
    if (sigma <= 0) sigma = 1e-10;
    return std::exp(-std::abs(x - center) / sigma);
}

double FIS::computeSigma(const std::vector<double>& center_vector) {
    double d = 0.0;
    int len = static_cast<int>(center_vector.size());
    
    if (len == 2) {
        d = std::abs(center_vector[0] - center_vector[1]);
    } else {
        // Parallel distance computation
        #pragma omp parallel
        {
            double local_d = 0.0;
            #pragma omp for
            for (int i = 0; i < len - 1; i++) {
                for (int j = i + 1; j < len; j++) {
                    double d_temp = std::abs(center_vector[i] - center_vector[j]);
                    if (d_temp > local_d) local_d = d_temp;
                }
            }
            #pragma omp critical
            {
                if (local_d > d) d = local_d;
            }
        }
    }
    
    double sigma = std::abs(d) / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    while (sigma < 1.0) {
        sigma *= 10.0;
    }
    return sigma;
}

// ============================================================================
// Parallel FCM (Fuzzy C-Means)
// ============================================================================

std::pair<Matrix, Matrix> FIS::fcm(const std::vector<double>& X, int C,
                                    const std::vector<double>& V_init,
                                    double m, double eps, int max_iter) {
    return fcmParallel(X, C, V_init, m, eps, max_iter, 0);
}

std::pair<Matrix, Matrix> FIS::fcmParallel(const std::vector<double>& X, int C,
                                             const std::vector<double>& V_init,
                                             double m, double eps, int max_iter,
                                             int numThreads) {
    int N = static_cast<int>(X.size());
    
    if (numThreads <= 0) numThreads = getOptimalThreadCount();
    
    Matrix V(1, std::vector<double>(C, 0.0));
    for (int j = 0; j < C; j++) {
        V[0][j] = V_init[j];
    }
    
    Matrix U(C, std::vector<double>(N, 0.0));
    
    double J_prev = std::numeric_limits<double>::infinity();
    
    for (int count = 0; count < max_iter; count++) {
        // Compute distances in parallel
        Matrix dist(N, std::vector<double>(C, 0.0));
        
        #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 256)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < C; j++) {
                double diff = X[i] - V[0][j];
                dist[i][j] = std::sqrt(diff * diff);
                if (dist[i][j] < eps) {
                    dist[i][j] = eps;
                }
            }
        }
        
        // Update membership in parallel
        Matrix U_new(C, std::vector<double>(N, 0.0));
        
        #pragma omp parallel for num_threads(numThreads) schedule(dynamic, 256)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < C; j++) {
                double sum_term = 0.0;
                for (int k = 0; k < C; k++) {
                    double ratio = std::pow(dist[i][j] / dist[i][k], 2.0 / (m - 1.0));
                    sum_term += ratio;
                }
                U_new[j][i] = (sum_term > 0) ? (1.0 / sum_term) : 0.0;
            }
        }
        
        // Normalize
        #pragma omp parallel for num_threads(numThreads)
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < C; j++) {
                sum += U_new[j][i];
            }
            if (sum > 0) {
                for (int j = 0; j < C; j++) {
                    U_new[j][i] /= sum;
                }
            }
        }
        
        // Update centers in parallel
        Matrix V_new(1, std::vector<double>(C, 0.0));
        
        #pragma omp parallel for num_threads(numThreads)
        for (int j = 0; j < C; j++) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (int i = 0; i < N; i++) {
                double u_m = std::pow(U_new[j][i], m);
                numerator += u_m * X[i];
                denominator += u_m;
            }
            V_new[0][j] = (denominator > 0) ? (numerator / denominator) : V[0][j];
        }
        
        // Compute objective function in parallel
        double J = 0.0;
        #pragma omp parallel for num_threads(numThreads) reduction(+:J)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < C; j++) {
                J += std::pow(U_new[j][i], m) * (dist[i][j] * dist[i][j]);
            }
        }
        
        if (std::abs(J - J_prev) < eps) {
            V = V_new;
            U = U_new;
            break;
        }
        
        J_prev = J;
        V = V_new;
        U = U_new;
    }
    
    return {V, U};
}

// ============================================================================
// Rule Generation with Parallel FCM
// ============================================================================

FIS::RuleGenerationResult FIS::ruleGenerate(const Matrix& train_data,
                                            const std::vector<int>& cluster,
                                            const std::vector<double>& min_vals,
                                            const std::vector<double>& max_vals,
                                            double m, double eps, int max_iter) {
    const int h = static_cast<int>(train_data.size());
    const int w = static_cast<int>(train_data[0].size());

    RuleGenerationResult result;
    result.rules = Matrix(h, std::vector<double>(w, 0.0));
    result.centers.resize(w - 1);

    Matrix U_output;
    const int numThreads = getOptimalThreadCount();

    auto buildInitCenters = [](double minVal, double maxVal, int c) {
        std::vector<double> init;
        c = std::max(1, c);
        if (c == 1) {
            init.push_back((minVal + maxVal) / 2.0);
            return init;
        }
        if (c == 2) {
            init = {minVal, maxVal};
            return init;
        }
        if (c == 3) {
            init = {minVal, (minVal + maxVal) / 2.0, maxVal};
            return init;
        }

        const double seg = (maxVal - minVal) / static_cast<double>(c - 1);
        init.reserve(static_cast<size_t>(c));
        for (int j = 0; j < c; ++j) {
            init.push_back(minVal + static_cast<double>(j) * seg);
        }
        return init;
    };

    // Process input features in parallel.
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int i = 0; i < w - 1; ++i) {
        std::vector<double> feature(h);
        for (int j = 0; j < h; ++j) {
            feature[j] = train_data[j][i];
        }

        const int clusterVal = (i < static_cast<int>(cluster.size())) ? cluster[i] : 3;
        const int c = std::max(1, clusterVal);
        const std::vector<double> V_init = buildInitCenters(min_vals[i], max_vals[i], c);

        Matrix centers;
        Matrix U;
        std::tie(centers, U) = fcmParallel(feature, c, V_init, m, eps, max_iter, numThreads);

        result.centers[i] = centers[0];

        for (int j = 0; j < h; ++j) {
            int maxIdx = 0;
            double maxVal = U[0][j];
            for (int k = 1; k < c; ++k) {
                if (U[k][j] > maxVal) {
                    maxVal = U[k][j];
                    maxIdx = k;
                }
            }
            result.rules[j][i] = maxIdx + 1;
        }
    }

    // Process output label column independently (matches Python logic).
    {
        const int outputIdx = w - 1;
        const int clusterOut =
            (outputIdx < static_cast<int>(cluster.size())) ? cluster[outputIdx] : 2;
        const int cOut = std::max(2, clusterOut);
        std::vector<double> outputValues(h);
        for (int j = 0; j < h; ++j) {
            outputValues[j] = train_data[j][outputIdx];
        }

        const std::vector<double> V_init =
            buildInitCenters(min_vals[outputIdx], max_vals[outputIdx], cOut);

        Matrix outputCenters;
        std::tie(outputCenters, U_output) =
            fcmParallel(outputValues, cOut, V_init, m, eps, max_iter, numThreads);
    }

    #pragma omp parallel for num_threads(numThreads)
    for (int j = 0; j < h; ++j) {
        const int c = static_cast<int>(U_output.size());
        if (c <= 0) {
            result.rules[j][w - 1] = 1.0;
            continue;
        }
        int maxIdx = 0;
        double maxVal = U_output[0][j];
        for (int k = 1; k < c; ++k) {
            if (U_output[k][j] > maxVal) {
                maxVal = U_output[k][j];
                maxIdx = k;
            }
        }
        result.rules[j][w - 1] = maxIdx + 1;
    }

    result.U = U_output;
    return result;
}

std::pair<Matrix, std::vector<double>> FIS::ruleWeight(const Matrix& rules,
                                                          const Matrix& data,
                                                          const std::vector<int>& cluster,
                                                          const Matrix& center_vector) {
    int data_num = static_cast<int>(data.size());
    int attribute_num = static_cast<int>(data[0].size());
    
    std::vector<double> sigma(attribute_num, 0.0);
    Matrix t(data_num, std::vector<double>(attribute_num, 0.0));
    
    int numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int feature_index = 0; feature_index < attribute_num; feature_index++) {
        std::vector<double> feature_data(data_num);
        for (int i = 0; i < data_num; i++) {
            feature_data[i] = data[i][feature_index];
        }
        
        std::vector<int> rule_index(data_num);
        for (int i = 0; i < data_num; i++) {
            rule_index[i] = static_cast<int>(rules[i][feature_index]);
        }
        
        int mf_number = cluster[feature_index];
        sigma[feature_index] = computeSigma(center_vector[feature_index]);
        
        for (int i = 0; i < data_num; i++) {
            t[i][feature_index] = gaussMF(feature_data[i], rule_index[i], 
                                          mf_number, sigma[feature_index],
                                          center_vector[feature_index]);
        }
    }
    
    return {t, sigma};
}

// ============================================================================
// Inference with Parallel Processing
// ============================================================================

std::vector<int> FIS::fuzzifyInput(const std::vector<double>& input_data,
                                    const std::vector<double>& sigma_M,
                                    const Matrix& centers) {
    int num_features = static_cast<int>(input_data.size());
    std::vector<int> fuzzy_values(num_features);
    
    int numThreads = getOptimalThreadCount();
    
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < num_features; i++) {
        const auto& center_vector = centers[i];
        double sigma = sigma_M[i];
        int num_centers = static_cast<int>(center_vector.size());
        
        double maxMembership = -1.0;
        int bestLabel = 1;
        
        // SIMD hint for compilers that support OpenMP SIMD without extra flags.
#if !defined(_MSC_VER)
        #pragma omp simd
#endif
        for (int label = 1; label <= num_centers; label++) {
            double membership = gaussMF(input_data[i], label, num_centers, sigma, center_vector);
            if (membership > maxMembership) {
                maxMembership = membership;
                bestLabel = label;
            }
        }
        
        fuzzy_values[i] = bestLabel;
    }
    
    return fuzzy_values;
}

int FIS::matchRule(const std::vector<int>& fuzzy_input, const Matrix& ruleList) {
    int num_rules = static_cast<int>(ruleList.size());
    if (num_rules == 0) return -1;
    
    int num_attrs = static_cast<int>(ruleList[0].size()) - 1;
    std::unordered_map<int, int> labelCounts;
    std::vector<int> labelOrder;
    
    for (int i = 0; i < num_rules; i++) {
        bool match = true;
        for (int j = 0; j < num_attrs; j++) {
            if (fuzzy_input[j] != static_cast<int>(ruleList[i][j])) {
                match = false;
                break;
            }
        }
        if (match) {
            const int label = static_cast<int>(ruleList[i][num_attrs]);
            if (labelCounts.find(label) == labelCounts.end()) {
                labelOrder.push_back(label);
            }
            labelCounts[label] += 1;
        }
    }

    if (!labelOrder.empty()) {
        int bestLabel = labelOrder[0];
        int bestCount = labelCounts[bestLabel];
        for (int label : labelOrder) {
            const int count = labelCounts[label];
            if (count > bestCount) {
                bestLabel = label;
                bestCount = count;
            }
        }
        return bestLabel;
    }
    
    return -1;
}

// ============================================================================
// Parallel Metrics
// ============================================================================

double FIS::accuracyScore(const std::vector<int>& predicted, const std::vector<int>& actual) {
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
    return static_cast<double>(correct) / n * 100.0;
}

double FIS::precisionScore(const std::vector<int>& predicted, const std::vector<int>& actual) {
    std::unordered_set<int> uniqueLabels;
    for (int label : predicted) uniqueLabels.insert(label);
    for (int label : actual) uniqueLabels.insert(label);
    
    double sumPrecision = 0.0;
    int numLabels = 0;
    int numThreads = getOptimalThreadCount();
    
    for (int label : uniqueLabels) {
        int TP = 0;
        int FP = 0;

        #pragma omp parallel for num_threads(numThreads) reduction(+:TP,FP)
        for (int i = 0; i < static_cast<int>(predicted.size()); i++) {
            if (predicted[i] == label && actual[i] == label) TP++;
            if (predicted[i] == label && actual[i] != label) FP++;
        }
        
        if (TP + FP > 0) {
            sumPrecision += static_cast<double>(TP) / (TP + FP);
            numLabels++;
        }
    }
    
    return (numLabels > 0) ? (sumPrecision / numLabels * 100.0) : 0.0;
}

double FIS::recallScore(const std::vector<int>& predicted, const std::vector<int>& actual) {
    std::unordered_set<int> uniqueLabels;
    for (int label : predicted) uniqueLabels.insert(label);
    for (int label : actual) uniqueLabels.insert(label);
    
    double sumRecall = 0.0;
    int numLabels = 0;
    int numThreads = getOptimalThreadCount();
    
    for (int label : uniqueLabels) {
        int TP = 0;
        int FN = 0;

        #pragma omp parallel for num_threads(numThreads) reduction(+:TP,FN)
        for (int i = 0; i < static_cast<int>(predicted.size()); i++) {
            if (predicted[i] == label && actual[i] == label) TP++;
            if (predicted[i] != label && actual[i] == label) FN++;
        }
        
        if (TP + FN > 0) {
            sumRecall += static_cast<double>(TP) / (TP + FN);
            numLabels++;
        }
    }
    
    return (numLabels > 0) ? (sumRecall / numLabels * 100.0) : 0.0;
}

double FIS::f1Score(const std::vector<int>& predicted, const std::vector<int>& actual) {
    double precision = precisionScore(predicted, actual);
    double recall = recallScore(predicted, actual);
    
    if (precision + recall > 0) {
        return 2.0 * precision * recall / (precision + recall);
    }
    return 0.0;
}

IntMatrix FIS::confusionMatrix(const std::vector<int>& predicted, const std::vector<int>& actual) {
    std::unordered_set<int> uniqueLabels;
    for (int label : predicted) uniqueLabels.insert(label);
    for (int label : actual) uniqueLabels.insert(label);
    
    std::vector<int> labels;
    for (int label : uniqueLabels) {
        labels.push_back(label);
    }
    std::sort(labels.begin(), labels.end());
    
    int n = static_cast<int>(labels.size());
    IntMatrix cm(n, std::vector<int>(n, 0));
    
    for (size_t i = 0; i < predicted.size(); i++) {
        int predIdx = -1, actualIdx = -1;
        for (int j = 0; j < n; j++) {
            if (labels[j] == predicted[i]) predIdx = j;
            if (labels[j] == actual[i]) actualIdx = j;
        }
        if (predIdx >= 0 && actualIdx >= 0) {
            cm[actualIdx][predIdx]++;
        }
    }
    
    return cm;
}

// ============================================================================
// GPU Interface (runtime fallback to CPU)
// ============================================================================

bool FIS::isGPUCompiled() {
#if FUZZY_USE_CUDA || FUZZY_USE_GPU
    return true;
#else
    return false;
#endif
}

bool FIS::isGPUAvailable() {
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

std::pair<Matrix, Matrix> FIS::fcmGPU(const std::vector<double>& X, int C,
                                       const std::vector<double>& V_init,
                                       double m, double eps, int max_iter) {
    if (!isGPUAvailable()) {
        return fcmParallel(X, C, V_init, m, eps, max_iter);
    }
#if FUZZY_USE_CUDA
    Matrix centers;
    Matrix U;
    std::string cudaError;
    const cudaError_t status = CUDA::fcm1DGPU(X, C, V_init, m, eps, max_iter,
                                              centers, U, &cudaError);
    if (status == cudaSuccess) {
        return {centers, U};
    }
    std::cerr << "CUDA error in FIS::fcmGPU, falling back to CPU: "
              << cudaError << std::endl;
#endif
    return fcmParallel(X, C, V_init, m, eps, max_iter);
}

FIS::RuleGenerationResult FIS::ruleGenerateGPU(const Matrix& data,
                                              const std::vector<int>& cluster,
                                              const std::vector<double>& min_vals,
                                              const std::vector<double>& max_vals,
                                              double m, double eps, int max_iter) {
    if (!isGPUAvailable()) {
        return ruleGenerate(data, cluster, min_vals, max_vals, m, eps, max_iter);
    }
#if FUZZY_USE_CUDA
    RuleGenerationResult result;
    std::string cudaError;
    const cudaError_t status = CUDA::ruleGenerateFIS_GPU(data, cluster, min_vals, max_vals,
                                                          m, eps, max_iter,
                                                          result.rules, result.centers, result.U,
                                                          &cudaError);
    if (status == cudaSuccess) {
        return result;
    }
    std::cerr << "CUDA error in FIS::ruleGenerateGPU, falling back to CPU: "
              << cudaError << std::endl;
#endif
    return ruleGenerate(data, cluster, min_vals, max_vals, m, eps, max_iter);
}

bool FIS::verifyGPUvsCPU(const Matrix& testData, double tolerance) {
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

FIS::BenchmarkResult FIS::benchmark(const Matrix& testData) {
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
