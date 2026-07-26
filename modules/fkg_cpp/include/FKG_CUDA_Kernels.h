/**
 * @file FKG_CUDA_Kernels.h
 * @brief Host-side CUDA wrappers for FKG GPU acceleration.
 */

#ifndef FKG_CUDA_KERNELS_H
#define FKG_CUDA_KERNELS_H

#if FUZZY_USE_CUDA

#include <cstddef>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace Fuzzy {

using Matrix = std::vector<std::vector<double>>;

namespace CUDA {

struct FisaGPUOutput {
    int bestClass = 1;
    double confidence = 0.0;
    std::vector<double> dValues;
};

struct FisaDeviceCache {
    double* dBase = nullptr;
    double* dC = nullptr;
    int3* dComb3 = nullptr;
    int4* dLookupKeys = nullptr;
    double* dLookupValues = nullptr;
    double* dInput = nullptr;
    double* dD = nullptr;
    double* dBatchInputs = nullptr;
    double* dBatchD = nullptr;
    double* hPinnedD = nullptr;
    double* hPinnedBatchD = nullptr;
    void* stream = nullptr; // cudaStream_t
    int rows = 0;
    int cols = 0;
    int fullCols = 0;
    int numComb3 = 0;
    int nClasses = 0;
    int lookupSize = 0;
    int useLookup = 0;
    std::size_t inputCapacity = 0;
    std::size_t dCapacity = 0;
    std::size_t batchInputCapacity = 0;
    std::size_t batchDCapacity = 0;
    std::size_t pinnedDCapacity = 0;
    std::size_t pinnedBatchDCapacity = 0;
};

// Compute each matrix independently.
cudaError_t calculateA_GPU(const Matrix& base, Matrix& A, std::string* error_message = nullptr);
cudaError_t calculateM_GPU(const Matrix& base, Matrix& M, std::string* error_message = nullptr);
cudaError_t calculateB_GPU(const Matrix& A, const Matrix& M, Matrix& B, int columns,
                           std::string* error_message = nullptr);
cudaError_t calculateC_GPU(const Matrix& base, const Matrix& B, Matrix& C, int columns, int n_classes,
                           std::string* error_message = nullptr);

// End-to-end pipeline (recommended for performance).
cudaError_t calculateABCM_GPU(const Matrix& base, int n_classes,
                              Matrix& A, Matrix& M, Matrix& B, Matrix& C,
                              std::string* error_message = nullptr);

// Single-sample FISA on GPU. If d_values is provided, D(label) will be returned.
cudaError_t fisaGPU(const Matrix& base, const Matrix& C, const std::vector<double>& input,
                    int n_classes, int& result_class, double& result_confidence,
                    std::vector<double>* d_values = nullptr,
                    std::string* error_message = nullptr);

// Persistent cache APIs for high-throughput inference.
cudaError_t createFisaDeviceCache(const Matrix& base, const Matrix& C, int n_classes,
                                  FisaDeviceCache& cache,
                                  std::string* error_message = nullptr);

void destroyFisaDeviceCache(FisaDeviceCache& cache);

cudaError_t fisaGPUWithCache(FisaDeviceCache& cache, const std::vector<double>& input,
                             int& result_class, double& result_confidence,
                             std::vector<double>* d_values = nullptr,
                             std::string* error_message = nullptr);

cudaError_t fisaBatchGPUWithCache(FisaDeviceCache& cache, const Matrix& inputs,
                                  std::vector<int>& result_classes,
                                  std::vector<double>& result_confidences,
                                  std::string* error_message = nullptr);

} // namespace CUDA
} // namespace Fuzzy

#endif // FUZZY_USE_CUDA

#endif // FKG_CUDA_KERNELS_H
