/**
 * @file FKG.h
 * @brief Fuzzy Knowledge Graph (FKG) - High-Performance C++ Implementation
 * @author Optimized for maximum parallel processing with SIMD
 * @version 2.0
 * 
 * Namespace: Fuzzy
 * Classes: FKG, FKGS
 * 
 * Optimization Features:
 * - OpenMP parallel processing
 * - SIMD vectorization hints
 * - Memory-aligned data structures
 * - Cache-optimized algorithms
 * - Thread-local storage
 * - GPU/CUDA support (conditional)
 */

#ifndef FUZZY_FKG_H
#define FUZZY_FKG_H

#include <vector>
#include <utility>
#include <cstdint>
#include <memory>
#include <string>

// ============================================================================
// GPU Configuration Options
// ============================================================================

// Enable GPU support (set to 1 to enable, 0 to disable)
#ifndef FUZZY_USE_GPU
#define FUZZY_USE_GPU 0
#endif

// GPU Backend selection:
// - Use SYCL for Intel GPUs / oneAPI (default when FUZZY_USE_GPU=1)
// - Use CUDA for NVIDIA GPUs (set FUZZY_USE_CUDA=1)
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

// Type definitions with alignment hints
using Matrix = std::vector<std::vector<double>>;
using MatrixFloat = std::vector<float>;

// Alignment for cache efficiency
#ifdef __GNUC__
#define ALIGNED(x) __attribute__((aligned(x)))
#else
#define ALIGNED(x)
#endif

// Performance configuration
struct PerformanceConfig {
    int numThreads;           // Number of OpenMP threads (0 = auto)
    bool useSIMD;            // Enable SIMD optimizations
    bool useCacheOptimized;   // Cache-optimized algorithms
    bool useThreadLocal;     // Use thread-local storage
    
    PerformanceConfig() : numThreads(0), useSIMD(true), 
                         useCacheOptimized(true), useThreadLocal(true) {}
};

/**
 * @brief FKG - Fuzzy Knowledge Graph main class (High Performance)
 */
class FKG {
public:
    FKG();
    explicit FKG(const PerformanceConfig& config);
    ~FKG();

    void train(const Matrix& base);
    void train(const Matrix& base, int n_classes);

    std::pair<int, double> predict(const std::vector<double>& input) const;
    std::vector<int> predictBatch(const Matrix& inputs) const;
    
    // Parallel batch prediction with thread pool
    std::vector<int> predictBatchParallel(const Matrix& inputs, int numThreads = 0) const;

    // Static utility methods with SIMD support
    static Matrix calculateA(const Matrix& base);
    static Matrix calculateM(const Matrix& base);
    static Matrix calculateB(const Matrix& base, const Matrix& A, const Matrix& M);
    static Matrix calculateC(const Matrix& base, const Matrix& B, int n_classes);
    
    // Optimized versions
    static Matrix calculateA_Parallel(const Matrix& base);
    static Matrix calculateB_Parallel(const Matrix& base, const Matrix& A, const Matrix& M);
    static Matrix calculateC_Parallel(const Matrix& base, const Matrix& B, int n_classes);
    
    // FISA with full D values
    static std::pair<int, double> fisa(const Matrix& base, const Matrix& C,
                                       const std::vector<double>& input, int n_classes);
    
    struct FISAResult {
        int bestClass;
        double confidence;
        std::vector<double> D;
    };
    
    static FISAResult FISAWithConfidence(const Matrix& base, const Matrix& C,
                                          const std::vector<double>& input, int n_classes);
    
    // SIMD-accelerated normalization
    static Matrix minMaxNormalize(const Matrix& C);
    static Matrix gaussianNormalize(const Matrix& C);
    
    // Metrics with parallel processing
    static double accuracy(const std::vector<int>& predicted, const std::vector<int>& actual);
    static std::vector<double> precisionPerClass(const std::vector<int>& predicted, 
                                                   const std::vector<int>& actual);
    static std::vector<double> recallPerClass(const std::vector<int>& predicted, 
                                              const std::vector<int>& actual);
    static std::vector<double> f1PerClass(const std::vector<int>& predicted, 
                                           const std::vector<int>& actual);
    
    // Performance metrics
    struct PerformanceMetrics {
        double computeTimeMs;
        int64_t memoryUsageBytes;
        int numThreadsUsed;
    };
    
    PerformanceMetrics getMetrics() const { return metrics_; }
    void resetMetrics() { metrics_ = PerformanceMetrics(); }

    // Getters
    int getNumClasses() const { return n_classes_; }
    const Matrix& getA() const { return A_; }
    const Matrix& getM() const { return M_; }
    const Matrix& getB() const { return B_; }
    const Matrix& getC() const { return C_; }
    const Matrix& getBase() const { return base_; }
    bool isTrained() const { return trained_; }
    const PerformanceConfig& getConfig() const { return config_; }
    
    // ========================================================================
    // GPU Support Methods (runtime selection with safe fallback)
    // ========================================================================

    // Enable/disable GPU preference at runtime
    void setUseGPU(bool use) { useGPU_ = use; }
    bool getUseGPU() const { return useGPU_; }

    // Check current effective backend ("gpu" only when requested and available)
    bool isUsingGPU() const { return useGPU_ && isGPUAvailable(); }

    // Build/runtime GPU capabilities
    static bool isGPUCompiled();
    static bool isGPUAvailable();
    
    // GPU methods (fallback to CPU when GPU is unavailable)
    static Matrix calculateA_GPU(const Matrix& base);
    static Matrix calculateB_GPU(const Matrix& base, const Matrix& A, const Matrix& M);
    static Matrix calculateC_GPU(const Matrix& base, const Matrix& B, int n_classes);
    static std::pair<int, double> fisaGPU(const Matrix& base, const Matrix& C,
                                          const std::vector<double>& input, int n_classes);
    
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
    Matrix base_;
    Matrix A_;
    Matrix M_;
    Matrix B_;
    Matrix C_;
    int n_classes_;
    bool trained_;
    PerformanceConfig config_;
    PerformanceMetrics metrics_;
    
    // Runtime backend preference
    bool useGPU_;

    // GPU-related members (conditional on build flags)
#if FUZZY_USE_GPU
    mutable std::unique_ptr<sycl::queue> syclQueue_;
#elif FUZZY_USE_CUDA
    mutable void* cudaStream_;  // CUDA stream handle
#endif
    
    // Pre-allocated buffers for parallel processing
    mutable std::vector<double> inputBuffer_;
    mutable std::vector<int> outputBuffer_;
};

/**
 * @brief FKGS - Fuzzy Knowledge Graph with Sampling (High Performance)
 */
class FKGS : public FKG {
public:
    FKGS();
    explicit FKGS(double ran, double e);
    explicit FKGS(const PerformanceConfig& config, double ran = 50.0, double e = 0.1);
    ~FKGS();

    static Matrix sampling(const Matrix& base, double ran, double e);
    
    // Parallel sampling
    static Matrix samplingParallel(const Matrix& base, double ran, double e, int numThreads = 0);
    
    static double diff(const std::vector<int>& Rule1, const std::vector<int>& Rule2);

private:
    double ran_;
    double e_;
};

} // namespace Fuzzy

#endif
