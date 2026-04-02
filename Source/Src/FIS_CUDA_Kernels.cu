/**
 * @file FIS_CUDA_Kernels.cu
 * @brief CUDA kernels + host wrappers for FIS (FCM and rule generation).
 */

#if FUZZY_USE_CUDA

#include "FIS_CUDA_Kernels.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
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

inline std::vector<double> buildInitCenters(double minVal, double maxVal, int c) {
    std::vector<double> out;
    c = std::max(1, c);
    if (c == 1) {
        out.push_back((minVal + maxVal) / 2.0);
        return out;
    }
    if (c == 2) {
        out = {minVal, maxVal};
        return out;
    }
    if (c == 3) {
        out = {minVal, (minVal + maxVal) / 2.0, maxVal};
        return out;
    }

    out.reserve(static_cast<size_t>(c));
    const double seg = (maxVal - minVal) / static_cast<double>(c - 1);
    for (int i = 0; i < c; ++i) {
        out.push_back(minVal + static_cast<double>(i) * seg);
    }
    return out;
}

inline cudaError_t checkKernel(const char* stage, std::string* error_message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        setError(error_message, stage, err);
        return err;
    }
    return cudaSuccess;
}

constexpr int kBlockSize = 256;

// idx = i * C + j
__global__ void KernelUpdateMembership(const double* X, const double* centers,
                                       int N, int C, double exponent, double eps,
                                       int useSquareExponent, double* U) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C;
    if (idx >= total) {
        return;
    }

    const int i = idx / C;
    const int j = idx - i * C;
    double dij = fabs(X[i] - centers[j]);
    if (dij < eps) {
        dij = eps;
    }

    double denom = 0.0;
    for (int k = 0; k < C; ++k) {
        double dik = fabs(X[i] - centers[k]);
        if (dik < eps) {
            dik = eps;
        }
        const double ratio = dij / dik;
        if (useSquareExponent != 0) {
            denom += ratio * ratio;
        } else {
            denom += pow(ratio, exponent);
        }
    }

    U[idx] = (denom > 0.0) ? (1.0 / denom) : 0.0;
}

// idx = i
__global__ void KernelNormalizeMembership(double* U, int N, int C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const int offset = i * C;
    double sum = 0.0;
    for (int j = 0; j < C; ++j) {
        sum += U[offset + j];
    }

    if (sum > 0.0) {
        for (int j = 0; j < C; ++j) {
            U[offset + j] /= sum;
        }
    }
}

// idx = cluster j
__global__ void KernelUpdateCenters(const double* X, const double* U,
                                    int N, int C, double m,
                                    int useSquareM,
                                    const double* centersPrev,
                                    double* centersNext) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= C) {
        return;
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < N; ++i) {
        const double u = U[i * C + j];
        const double um = (useSquareM != 0) ? (u * u) : pow(u, m);
        numerator += um * X[i];
        denominator += um;
    }

    centersNext[j] = (denominator > 0.0) ? (numerator / denominator) : centersPrev[j];
}

// Block-reduced max center shift. Host reduces partial maxima.
__global__ void KernelMaxCenterShiftPartial(const double* centersPrev,
                                            const double* centersNext,
                                            int C, double* partialMax) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < C) {
        local = fabs(centersNext[i] - centersPrev[i]);
    }

    __shared__ double sdata[kBlockSize];
    sdata[threadIdx.x] = local;
    __syncthreads();

    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partialMax[blockIdx.x] = sdata[0];
    }
}

} // namespace

cudaError_t fcm1DGPU(const std::vector<double>& X, int C,
                     const std::vector<double>& V_init,
                     double m, double eps, int max_iter,
                     Matrix& V_out, Matrix& U_out,
                     std::string* error_message) {
    V_out.clear();
    U_out.clear();

    const int N = static_cast<int>(X.size());
    if (N <= 0 || C <= 0 || static_cast<int>(V_init.size()) != C) {
        if (error_message != nullptr) {
            *error_message = "Invalid FCM input shape.";
        }
        return cudaErrorInvalidValue;
    }
    if (m <= 1.0) {
        if (error_message != nullptr) {
            *error_message = "m must be > 1.0 for FCM.";
        }
        return cudaErrorInvalidValue;
    }
    if (max_iter <= 0) {
        max_iter = 1;
    }
    if (eps <= 0.0) {
        eps = 1e-10;
    }

    DeviceBuffer<double> dX;
    DeviceBuffer<double> dCentersA;
    DeviceBuffer<double> dCentersB;
    DeviceBuffer<double> dU;
    DeviceBuffer<double> dCenterShiftPartial;

    cudaError_t err = dX.allocate(static_cast<size_t>(N));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(X)", err); return err; }
    err = dCentersA.allocate(static_cast<size_t>(C));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(VA)", err); return err; }
    err = dCentersB.allocate(static_cast<size_t>(C));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(VB)", err); return err; }
    err = dU.allocate(static_cast<size_t>(N) * static_cast<size_t>(C));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(U)", err); return err; }
    const int centerBlocks = (C + kBlockSize - 1) / kBlockSize;
    err = dCenterShiftPartial.allocate(static_cast<size_t>(centerBlocks));
    if (err != cudaSuccess) { setError(error_message, "cudaMalloc(centerShiftPartial)", err); return err; }

    err = cudaMemcpy(dX.get(), X.data(), static_cast<size_t>(N) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(X)", err); return err; }
    err = cudaMemcpy(dCentersA.get(), V_init.data(), static_cast<size_t>(C) * sizeof(double),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { setError(error_message, "cudaMemcpy(V_init)", err); return err; }

    double* currentCenters = dCentersA.get();
    double* nextCenters = dCentersB.get();
    const double exponent = 2.0 / (m - 1.0);
    const int useSquareExponent = (std::abs(exponent - 2.0) <= 1e-12) ? 1 : 0;
    const int useSquareM = (std::abs(m - 2.0) <= 1e-12) ? 1 : 0;
    bool converged = false;

    std::vector<double> hCenterShiftPartial(static_cast<size_t>(centerBlocks), 0.0);
    std::vector<double> hCenters(static_cast<size_t>(C), 0.0);
    std::vector<double> hU(static_cast<size_t>(N) * static_cast<size_t>(C), 0.0);

    const int totalNC = N * C;
    for (int iter = 0; iter < max_iter; ++iter) {
        KernelUpdateMembership<<<(totalNC + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            dX.get(), currentCenters, N, C, exponent, eps, useSquareExponent, dU.get());
        err = checkKernel("KernelUpdateMembership", error_message);
        if (err != cudaSuccess) {
            return err;
        }

        KernelNormalizeMembership<<<(N + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            dU.get(), N, C);
        err = checkKernel("KernelNormalizeMembership", error_message);
        if (err != cudaSuccess) {
            return err;
        }

        KernelUpdateCenters<<<(C + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            dX.get(), dU.get(), N, C, m, useSquareM, currentCenters, nextCenters);
        err = checkKernel("KernelUpdateCenters", error_message);
        if (err != cudaSuccess) {
            return err;
        }

        KernelMaxCenterShiftPartial<<<centerBlocks, kBlockSize>>>(
            currentCenters, nextCenters, C, dCenterShiftPartial.get());
        err = checkKernel("KernelMaxCenterShiftPartial", error_message);
        if (err != cudaSuccess) {
            return err;
        }

        err = cudaMemcpy(hCenterShiftPartial.data(), dCenterShiftPartial.get(),
                         static_cast<size_t>(centerBlocks) * sizeof(double),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            setError(error_message, "cudaMemcpy(centerShiftPartial)", err);
            return err;
        }
        double maxShift = 0.0;
        for (double value : hCenterShiftPartial) {
            if (value > maxShift) {
                maxShift = value;
            }
        }

        if (maxShift < eps) {
            err = cudaMemcpy(hCenters.data(), nextCenters, static_cast<size_t>(C) * sizeof(double),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                setError(error_message, "cudaMemcpy(V_final)", err);
                return err;
            }
            err = cudaMemcpy(hU.data(), dU.get(),
                             static_cast<size_t>(N) * static_cast<size_t>(C) * sizeof(double),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                setError(error_message, "cudaMemcpy(U_final)", err);
                return err;
            }
            converged = true;
            break;
        }

        std::swap(currentCenters, nextCenters);
    }

    if (!converged) {
        err = cudaMemcpy(hCenters.data(), currentCenters, static_cast<size_t>(C) * sizeof(double),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            setError(error_message, "cudaMemcpy(V_last)", err);
            return err;
        }
        err = cudaMemcpy(hU.data(), dU.get(),
                         static_cast<size_t>(N) * static_cast<size_t>(C) * sizeof(double),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            setError(error_message, "cudaMemcpy(U_last)", err);
            return err;
        }
    }

    V_out.assign(1, std::vector<double>(static_cast<size_t>(C), 0.0));
    for (int j = 0; j < C; ++j) {
        V_out[0][static_cast<size_t>(j)] = hCenters[static_cast<size_t>(j)];
    }

    U_out.assign(static_cast<size_t>(C), std::vector<double>(static_cast<size_t>(N), 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
            U_out[static_cast<size_t>(j)][static_cast<size_t>(i)] =
                hU[static_cast<size_t>(i) * static_cast<size_t>(C) + static_cast<size_t>(j)];
        }
    }
    return cudaSuccess;
}

cudaError_t ruleGenerateFIS_GPU(const Matrix& data,
                                const std::vector<int>& cluster,
                                const std::vector<double>& min_vals,
                                const std::vector<double>& max_vals,
                                double m, double eps, int max_iter,
                                Matrix& rules_out,
                                Matrix& centers_out,
                                Matrix& U_out,
                                std::string* error_message) {
    rules_out.clear();
    centers_out.clear();
    U_out.clear();

    int rows = 0;
    int cols = 0;
    if (!validateRectangular(data, &rows, &cols) || cols < 2) {
        if (error_message != nullptr) {
            *error_message = "Input data must be non-empty rectangular matrix with label column.";
        }
        return cudaErrorInvalidValue;
    }
    if (static_cast<int>(min_vals.size()) < cols || static_cast<int>(max_vals.size()) < cols) {
        if (error_message != nullptr) {
            *error_message = "min_vals/max_vals size mismatch.";
        }
        return cudaErrorInvalidValue;
    }

    rules_out.assign(static_cast<size_t>(rows), std::vector<double>(static_cast<size_t>(cols), 0.0));
    centers_out.resize(static_cast<size_t>(cols - 1));

    for (int feature = 0; feature < cols - 1; ++feature) {
        std::vector<double> X(static_cast<size_t>(rows), 0.0);
        for (int r = 0; r < rows; ++r) {
            X[static_cast<size_t>(r)] = data[static_cast<size_t>(r)][static_cast<size_t>(feature)];
        }

        const int clusterVal =
            (feature < static_cast<int>(cluster.size())) ? cluster[static_cast<size_t>(feature)] : 3;
        const int c = std::max(1, clusterVal);
        const std::vector<double> initCenters =
            buildInitCenters(min_vals[static_cast<size_t>(feature)],
                             max_vals[static_cast<size_t>(feature)], c);

        Matrix featureCenters;
        Matrix U;
        const cudaError_t err = fcm1DGPU(X, c, initCenters, m, eps, max_iter,
                                         featureCenters, U, error_message);
        if (err != cudaSuccess) {
            return err;
        }

        centers_out[static_cast<size_t>(feature)] = featureCenters[0];

        for (int r = 0; r < rows; ++r) {
            int maxIdx = 0;
            double maxVal = U[0][static_cast<size_t>(r)];
            for (int k = 1; k < c; ++k) {
                const double candidate = U[static_cast<size_t>(k)][static_cast<size_t>(r)];
                if (candidate > maxVal) {
                    maxVal = candidate;
                    maxIdx = k;
                }
            }
            rules_out[static_cast<size_t>(r)][static_cast<size_t>(feature)] =
                static_cast<double>(maxIdx + 1);
        }
    }

    const int outputIdx = cols - 1;
    const int outputCluster =
        (outputIdx < static_cast<int>(cluster.size())) ? cluster[static_cast<size_t>(outputIdx)] : 2;
    const int cOut = std::max(2, outputCluster);

    std::vector<double> outputValues(static_cast<size_t>(rows), 0.0);
    for (int r = 0; r < rows; ++r) {
        outputValues[static_cast<size_t>(r)] = data[static_cast<size_t>(r)][static_cast<size_t>(outputIdx)];
    }

    const std::vector<double> outputInit =
        buildInitCenters(min_vals[static_cast<size_t>(outputIdx)],
                         max_vals[static_cast<size_t>(outputIdx)], cOut);

    Matrix outputCenters;
    Matrix outputU;
    cudaError_t err = fcm1DGPU(outputValues, cOut, outputInit, m, eps, max_iter,
                               outputCenters, outputU, error_message);
    if (err != cudaSuccess) {
        return err;
    }

    for (int r = 0; r < rows; ++r) {
        int maxIdx = 0;
        double maxVal = outputU[0][static_cast<size_t>(r)];
        for (int k = 1; k < cOut; ++k) {
            const double candidate = outputU[static_cast<size_t>(k)][static_cast<size_t>(r)];
            if (candidate > maxVal) {
                maxVal = candidate;
                maxIdx = k;
            }
        }
        rules_out[static_cast<size_t>(r)][static_cast<size_t>(outputIdx)] =
            static_cast<double>(maxIdx + 1);
    }

    U_out = std::move(outputU);
    return cudaSuccess;
}

} // namespace CUDA
} // namespace Fuzzy

#endif // FUZZY_USE_CUDA
