/**
 * @file FIS.h
 * @brief Fuzzy Inference System (FIS) - High-Performance C++ Implementation
 * @author Optimized for maximum parallel processing
 * @version 2.0
 * 
 * Namespace: Fuzzy
 * Class: FIS
 * 
 * Optimization Features:
 * - OpenMP parallel FCM clustering
 * - SIMD vectorization
 * - Cache-optimized memory access
 * - Thread-safe operations
 * - GPU/CUDA support (conditional)
 */

#ifndef FUZZY_FIS_H
#define FUZZY_FIS_H

#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include <memory>

// ============================================================================
// GPU Configuration Options
// ============================================================================

// Enable GPU support (set to 1 to enable, 0 to disable)
#ifndef FUZZY_USE_GPU
#define FUZZY_USE_GPU 0
#endif

// Enable CUDA support (set to 1 to enable, 0 to disable)
#ifndef FUZZY_USE_CUDA
#define FUZZY_USE_CUDA 0
#endif

// Use SYCL for GPU (requires oneAPI/DPC++)
#if FUZZY_USE_GPU && !FUZZY_USE_CUDA
#include <sycl/sycl.hpp>
#endif

// Use CUDA for GPU (requires NVIDIA CUDA Toolkit)
#if FUZZY_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace Fuzzy {

// Type definitions
using Matrix = std::vector<std::vector<double>>;
using IntMatrix = std::vector<std::vector<int>>;

// Performance configuration for FIS
struct FISPerformanceConfig {
    int numThreads;
    double m;              // Fuzziness parameter
    double eps;            // Convergence threshold
    int maxIter;           // Maximum iterations
    bool useSIMD;
    
    FISPerformanceConfig() : numThreads(0), m(2.0), eps(1e-5), 
                            maxIter(200), useSIMD(true) {}
};

/**
 * @brief FIS - Fuzzy Inference System main class (High Performance)
 */
class FIS {
public:
    FIS();
    explicit FIS(const FISPerformanceConfig& config);
    FIS(const std::vector<int>& n_clusters, double m = 2.0, 
        double eps = 1e-5, int max_iter = 200);
    ~FIS();

    void train(const Matrix& data);
    
    // Parallel training with thread pool
    void trainParallel(const Matrix& data, int numThreads = 0);
    
    int predict(const std::vector<double>& input) const;
    std::vector<int> predictBatch(const Matrix& inputs) const;
    
    // Parallel batch prediction
    std::vector<int> predictBatchParallel(const Matrix& inputs, int numThreads = 0) const;

    // Static methods for membership functions (SIMD-optimized)
    static double gaussMF(double x, double sigma, double center);
    static double gaussMF(double x, int label, int MFnumber, double sigma, 
                          const std::vector<double>& centers);
    static double triangleMF(double x, double a, double b, double c);
    static double trapezoidMF(double x, double a, double b, double c, double d);
    static double sigmoidMF(double x, double center, double slope);
    static double expMF(double x, double center, double sigma);
    static double computeSigma(const std::vector<double>& center_vector);

    // FCM with parallel processing
    static std::pair<Matrix, Matrix> fcm(const std::vector<double>& X, int C,
                                         const std::vector<double>& V_init,
                                         double m = 2.0, double eps = 1e-5, 
                                         int max_iter = 100);
    
    // Parallel FCM
    static std::pair<Matrix, Matrix> fcmParallel(const std::vector<double>& X, int C,
                                             const std::vector<double>& V_init,
                                             double m, double eps, int max_iter,
                                             int numThreads = 0);

    // Rule processing
    struct RuleGenerationResult {
        Matrix rules;
        Matrix centers;
        Matrix U;
    };

    static RuleGenerationResult ruleGenerate(const Matrix& data,
                                          const std::vector<int>& cluster,
                                          const std::vector<double>& min_vals,
                                          const std::vector<double>& max_vals,
                                          double m = 2.0, double eps = 1e-5, 
                                          int max_iter = 200);

    static std::pair<Matrix, std::vector<double>> ruleWeight(const Matrix& rules,
                                                               const Matrix& data,
                                                               const std::vector<int>& cluster,
                                                               const Matrix& center_vector);

    static std::vector<int> fuzzifyInput(const std::vector<double>& input_data,
                                      const std::vector<double>& sigma_M,
                                      const Matrix& centers);

    static int matchRule(const std::vector<int>& fuzzy_input, const Matrix& ruleList);

    // Metrics with parallel processing
    static double accuracyScore(const std::vector<int>& predicted, 
                                const std::vector<int>& actual);
    static double precisionScore(const std::vector<int>& predicted, 
                                 const std::vector<int>& actual);
    static double recallScore(const std::vector<int>& predicted, 
                               const std::vector<int>& actual);
    static double f1Score(const std::vector<int>& predicted, 
                          const std::vector<int>& actual);
    static IntMatrix confusionMatrix(const std::vector<int>& predicted, 
                                     const std::vector<int>& actual);

    // Getters
    const Matrix& getRules() const { return rules_; }
    const Matrix& getCenters() const { return centers_; }
    const std::vector<double>& getSigma() const { return sigma_; }
    bool isTrained() const { return trained_; }
    const FISPerformanceConfig& getConfig() const { return config_; }
    
    // ========================================================================
    // GPU Support Methods (runtime selection with safe fallback)
    // ========================================================================

    // Enable/disable GPU preference at runtime
    void setUseGPU(bool use) { useGPU_ = use; }
    bool getUseGPU() const { return useGPU_; }
    bool isUsingGPU() const { return useGPU_ && isGPUAvailable(); }

    // Build/runtime GPU capabilities
    static bool isGPUCompiled();
    static bool isGPUAvailable();
    
    // GPU methods (fallback to CPU when GPU is unavailable)
    static std::pair<Matrix, Matrix> fcmGPU(const std::vector<double>& X, int C,
                                            const std::vector<double>& V_init,
                                            double m = 2.0, double eps = 1e-5, 
                                            int max_iter = 100);
    
    // GPU-accelerated rule generation
    static RuleGenerationResult ruleGenerateGPU(const Matrix& data,
                                                const std::vector<int>& cluster,
                                                const std::vector<double>& min_vals,
                                                const std::vector<double>& max_vals,
                                                double m = 2.0, double eps = 1e-5, 
                                                int max_iter = 200);
    
    // Verification & Benchmarking
    bool verifyGPUvsCPU(const Matrix& testData, double tolerance = 1e-6);
    struct BenchmarkResult {
        double gpuTimeMs;
        double cpuTimeMs;
        double speedup;
        bool resultsMatch;
        double maxDiff;
    };
    BenchmarkResult benchmark(const Matrix& testData);
    
private:
    FISPerformanceConfig config_;
    std::vector<int> n_clusters_;
    double m_;
    double eps_;
    int max_iter_;
    
    Matrix rules_;
    Matrix centers_;
    std::vector<double> sigma_;
    std::vector<double> min_vals_;
    std::vector<double> max_vals_;
    
    bool trained_;
    
    // Runtime backend preference
    bool useGPU_;

    // GPU-related members (conditional on build flags)
#if FUZZY_USE_GPU
    mutable std::unique_ptr<sycl::queue> syclQueue_;
#elif FUZZY_USE_CUDA
    mutable void* cudaStream_;
#endif
    
    // Pre-allocated buffers for parallel processing
    mutable std::vector<double> inputBuffer_;
    mutable std::vector<int> outputBuffer_;
};

} // namespace Fuzzy

#endif
