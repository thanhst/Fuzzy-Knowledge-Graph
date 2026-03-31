/**
 * @file bindings.cpp
 * @brief Python bindings for FKG and FIS modules using pybind11
 * @version 2.0 - Optimized for high performance
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "FKG.h"
#include "FIS.h"
#include "Utils.h"

namespace py = pybind11;
using namespace Fuzzy;

// ============================================================================
// Helper Functions
// ============================================================================

Matrix numpyToMatrix(const py::array_t<double>& arr) {
    py::buffer_info buf = arr.request();
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    
    Matrix result(rows, std::vector<double>(cols));
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = ptr[i * cols + j];
        }
    }
    return result;
}

py::array_t<double> matrixToNumpy(const Matrix& mat) {
    if (mat.empty()) return py::array_t<double>(0);
    
    py::array_t<double> result({mat.size(), mat[0].size()});
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < mat.size(); i++) {
        for (size_t j = 0; j < mat[0].size(); j++) {
            ptr[i * mat[0].size() + j] = mat[i][j];
        }
    }
    return result;
}

IntMatrix numpyToIntMatrix(const py::array_t<int>& arr) {
    py::buffer_info buf = arr.request();
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    
    IntMatrix result(rows, std::vector<int>(cols));
    int* ptr = static_cast<int*>(buf.ptr);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = ptr[i * cols + j];
        }
    }
    return result;
}

py::array_t<int> intMatrixToNumpy(const IntMatrix& mat) {
    if (mat.empty()) return py::array_t<int>(0);
    
    py::array_t<int> result({mat.size(), mat[0].size()});
    py::buffer_info buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    for (size_t i = 0; i < mat.size(); i++) {
        for (size_t j = 0; j < mat[0].size(); j++) {
            ptr[i * mat[0].size() + j] = mat[i][j];
        }
    }
    return result;
}

// ============================================================================
// FKG Class Bindings (Optimized)
// ============================================================================

void bind_fkg_class(py::module& m) {
    py::class_<PerformanceConfig>(m, "PerformanceConfig")
        .def(py::init<>())
        .def_readwrite("numThreads", &PerformanceConfig::numThreads)
        .def_readwrite("useSIMD", &PerformanceConfig::useSIMD)
        .def_readwrite("useCacheOptimized", &PerformanceConfig::useCacheOptimized)
        .def_readwrite("useThreadLocal", &PerformanceConfig::useThreadLocal);
    
    py::class_<FKG::PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readwrite("computeTimeMs", &FKG::PerformanceMetrics::computeTimeMs)
        .def_readwrite("memoryUsageBytes", &FKG::PerformanceMetrics::memoryUsageBytes)
        .def_readwrite("numThreadsUsed", &FKG::PerformanceMetrics::numThreadsUsed);
    
    py::class_<FKG>(m, "FKG")
        .def(py::init<>(), "Create FKG with default config")
        .def(py::init<const PerformanceConfig&>(), "Create FKG with custom performance config")
        .def("set_use_gpu", &FKG::setUseGPU, "Request GPU backend", py::arg("use_gpu"))
        .def("get_use_gpu", &FKG::getUseGPU, "Get requested GPU backend")
        .def("is_using_gpu", &FKG::isUsingGPU, "Get effective backend (GPU only when available)")
        .def("train", static_cast<void(FKG::*)(const Matrix&)>(&FKG::train), "Train FKG")
        .def("train", static_cast<void(FKG::*)(const Matrix&, int)>(&FKG::train), 
             "Train FKG with specified classes", py::arg("base"), py::arg("n_classes"))
        .def("predict", &FKG::predict, "Predict single input")
        .def("predict_batch", &FKG::predictBatch, "Predict batch (auto parallel)")
        .def("predict_batch_parallel", &FKG::predictBatchParallel, 
             "Predict batch with specified threads", py::arg("inputs"), py::arg("numThreads") = 0)
        .def("get_base", &FKG::getBase, "Get base matrix")
        .def("get_A", &FKG::getA, "Get A matrix")
        .def("get_M", &FKG::getM, "Get M matrix")
        .def("get_B", &FKG::getB, "Get B matrix")
        .def("get_C", &FKG::getC, "Get C matrix")
        .def("is_trained", &FKG::isTrained, "Check if trained")
        .def("get_metrics", &FKG::getMetrics, "Get performance metrics")
        .def("reset_metrics", &FKG::resetMetrics, "Reset metrics")
        .def("get_config", &FKG::getConfig, "Get performance config")
        // Static methods
        .def_static("calculateA", &FKG::calculateA, "Calculate A matrix")
        .def_static("calculateA_parallel", &FKG::calculateA_Parallel, "Calculate A matrix (parallel)")
        .def_static("calculateM", &FKG::calculateM, "Calculate M matrix")
        .def_static("calculateB", &FKG::calculateB, "Calculate B matrix")
        .def_static("calculateB_parallel", &FKG::calculateB_Parallel, "Calculate B matrix (parallel)")
        .def_static("calculateC", &FKG::calculateC, "Calculate C matrix")
        .def_static("calculateC_parallel", &FKG::calculateC_Parallel, "Calculate C matrix (parallel)")
        .def_static("fisa", &FKG::fisa, "FISA inference")
        .def_static("fisa_with_confidence", &FKG::FISAWithConfidence, "FISA with confidence")
        .def_static("min_max_normalize", &FKG::minMaxNormalize, "Min-max normalize")
        .def_static("gaussian_normalize", &FKG::gaussianNormalize, "Gaussian normalize");
    
    py::class_<FKGS, FKG>(m, "FKGS")
        .def(py::init<>(), "Create FKGS with defaults")
        .def(py::init<double, double>(), "Create FKGS", py::arg("ran") = 50.0, py::arg("e") = 0.1)
        .def(py::init<const PerformanceConfig&, double, double>(), 
             "Create FKGS with config", py::arg("config"), py::arg("ran") = 50.0, py::arg("e") = 0.1)
        .def("sample", &FKGS::sampling, "Sample data");
}

// ============================================================================
// FIS Class Bindings (Optimized)
// ============================================================================

void bind_fis_class(py::module& m) {
    py::class_<FISPerformanceConfig>(m, "FISPerformanceConfig")
        .def(py::init<>())
        .def_readwrite("numThreads", &FISPerformanceConfig::numThreads)
        .def_readwrite("m", &FISPerformanceConfig::m)
        .def_readwrite("eps", &FISPerformanceConfig::eps)
        .def_readwrite("maxIter", &FISPerformanceConfig::maxIter)
        .def_readwrite("useSIMD", &FISPerformanceConfig::useSIMD);
    
    py::class_<FIS>(m, "FIS")
        .def(py::init<>(), "Create FIS with defaults")
        .def(py::init<const FISPerformanceConfig&>(), "Create FIS with config")
        .def(py::init<const std::vector<int>&, double, double, int>(), 
             "Create FIS with clusters", 
             py::arg("n_clusters"), py::arg("m") = 2.0, py::arg("eps") = 1e-5, py::arg("max_iter") = 200)
        .def("set_use_gpu", &FIS::setUseGPU, "Request GPU backend", py::arg("use_gpu"))
        .def("get_use_gpu", &FIS::getUseGPU, "Get requested GPU backend")
        .def("is_using_gpu", &FIS::isUsingGPU, "Get effective backend (GPU only when available)")
        .def("train", static_cast<void(FIS::*)(const Matrix&)>(&FIS::train), "Train FIS")
        .def("train_parallel", &FIS::trainParallel, 
             "Train FIS (parallel)", py::arg("data"), py::arg("numThreads") = 0)
        .def("predict", &FIS::predict, "Predict single input")
        .def("predict_batch", &FIS::predictBatch, "Predict batch")
        .def("predict_batch_parallel", &FIS::predictBatchParallel, 
             "Predict batch (parallel)", py::arg("inputs"), py::arg("numThreads") = 0)
        .def("get_rules", &FIS::getRules, "Get rules")
        .def("get_centers", &FIS::getCenters, "Get centers")
        .def("get_sigma", &FIS::getSigma, "Get sigma values")
        .def("is_trained", &FIS::isTrained, "Check if trained")
        .def("get_config", &FIS::getConfig, "Get config");
}

// ============================================================================
// Utility Functions Bindings
// ============================================================================

void bind_utils(py::module& m) {
    // Normalization
    m.def("min_max_normalize", [](const py::array_t<double>& input) {
        Matrix mat = numpyToMatrix(input);
        return matrixToNumpy(FKG::minMaxNormalize(mat));
    }, "Min-max normalization");
    
    m.def("gaussian_normalize", [](const py::array_t<double>& input) {
        Matrix mat = numpyToMatrix(input);
        return matrixToNumpy(FKG::gaussianNormalize(mat));
    }, "Gaussian normalization");
    
    // Sampling
    m.def("sampling", [](const py::array_t<double>& input, double ran, double e) {
        Matrix mat = numpyToMatrix(input);
        return matrixToNumpy(FKGS::sampling(mat, ran, e));
    }, "Sampling", py::arg("base"), py::arg("ran"), py::arg("e"));
    
    m.def("sampling_parallel", [](const py::array_t<double>& input, double ran, double e, int numThreads) {
        Matrix mat = numpyToMatrix(input);
        return matrixToNumpy(FKGS::samplingParallel(mat, ran, e, numThreads));
    }, "Sampling (parallel)", py::arg("base"), py::arg("ran"), py::arg("e"), py::arg("numThreads") = 0);
    
    // Metrics
    m.def("accuracy", [](const std::vector<int>& predicted,
                        const std::vector<int>& actual) {
        return FKG::accuracy(predicted, actual);
    }, "Calculate accuracy");
    
    m.def("precision_per_class", [](const std::vector<int>& predicted,
                                    const std::vector<int>& actual) {
        return FKG::precisionPerClass(predicted, actual);
    }, "Precision per class");
    
    m.def("recall_per_class", [](const std::vector<int>& predicted,
                                  const std::vector<int>& actual) {
        return FKG::recallPerClass(predicted, actual);
    }, "Recall per class");
    
    m.def("f1_per_class", [](const std::vector<int>& predicted,
                              const std::vector<int>& actual) {
        return FKG::f1PerClass(predicted, actual);
    }, "F1 per class");
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(fisa_module, m) {
    m.doc() = "FKG and FIS modules for fuzzy inference systems (v2.0 - High Performance)";
    
    // FKG class
    py::module fkg = m.def_submodule("fkg", "Fuzzy Knowledge Graph module (optimized)");
    bind_fkg_class(fkg);
    
    // FIS class
    py::module fis = m.def_submodule("fis", "Fuzzy Inference System module (optimized)");
    bind_fis_class(fis);
    
    // Utils
    py::module utils = m.def_submodule("utils", "Utility functions");
    bind_utils(utils);

    m.def("is_gpu_compiled", &FKG::isGPUCompiled, "Check whether module was built with GPU support");
    m.def("is_gpu_available", &FKG::isGPUAvailable, "Check if GPU is currently available");
    m.def("resolve_backend", [](bool use_gpu) {
        if (use_gpu && FKG::isGPUCompiled() && FKG::isGPUAvailable()) {
            return std::string("gpu");
        }
        return std::string("cpu");
    }, py::arg("use_gpu") = false, "Resolve effective backend");
    
    // ============================================================================
    // GPU Support Bindings (Conditional Compilation)
    // ============================================================================
    
    #if FUZZY_USE_GPU || FUZZY_USE_CUDA
    py::class_<FKG::BenchmarkResult>(m, "FKGBenchmarkResult")
        .def(py::init<>())
        .def_readwrite("gpuTimeMs", &FKG::BenchmarkResult::gpuTimeMs)
        .def_readwrite("cpuTimeMs", &FKG::BenchmarkResult::cpuTimeMs)
        .def_readwrite("speedup", &FKG::BenchmarkResult::speedup)
        .def_readwrite("resultsMatch", &FKG::BenchmarkResult::resultsMatch)
        .def_readwrite("maxDiff", &FKG::BenchmarkResult::maxDiff);
    
    // FIS GPU is not implemented yet, only FKG
    // py::class_<FIS::BenchmarkResult>(m, "FISBenchmarkResult")
    //     .def(py::init<>())
    //     .def_readwrite("gpuTimeMs", &FIS::BenchmarkResult::gpuTimeMs)
    //     .def_readwrite("cpuTimeMs", &FIS::BenchmarkResult::cpuTimeMs)
    //     .def_readwrite("speedup", &FIS::BenchmarkResult::speedup)
    //     .def_readwrite("resultsMatch", &FIS::BenchmarkResult::resultsMatch)
    //     .def_readwrite("maxDiff", &FIS::BenchmarkResult::maxDiff);
    
    m.attr("GPU_ENABLED") = true;
    #else
    m.attr("GPU_ENABLED") = false;
    #endif
    m.attr("GPU_COMPILED") = FKG::isGPUCompiled();
    m.attr("GPU_AVAILABLE") = FKG::isGPUAvailable();
}
