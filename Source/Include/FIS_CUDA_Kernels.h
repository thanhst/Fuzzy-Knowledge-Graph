/**
 * @file FIS_CUDA_Kernels.h
 * @brief Host-side CUDA wrappers for FIS GPU acceleration.
 */

#ifndef FIS_CUDA_KERNELS_H
#define FIS_CUDA_KERNELS_H

#if FUZZY_USE_CUDA

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace Fuzzy {

using Matrix = std::vector<std::vector<double>>;

namespace CUDA {

// 1D FCM on GPU (for one feature column).
cudaError_t fcm1DGPU(const std::vector<double>& X, int C,
                     const std::vector<double>& V_init,
                     double m, double eps, int max_iter,
                     Matrix& V_out, Matrix& U_out,
                     std::string* error_message = nullptr);

// Full FIS rule generation path on GPU (feature-wise FCM + output FCM).
cudaError_t ruleGenerateFIS_GPU(const Matrix& data,
                                const std::vector<int>& cluster,
                                const std::vector<double>& min_vals,
                                const std::vector<double>& max_vals,
                                double m, double eps, int max_iter,
                                Matrix& rules_out,
                                Matrix& centers_out,
                                Matrix& U_out,
                                std::string* error_message = nullptr);

} // namespace CUDA
} // namespace Fuzzy

#endif // FUZZY_USE_CUDA

#endif // FIS_CUDA_KERNELS_H

