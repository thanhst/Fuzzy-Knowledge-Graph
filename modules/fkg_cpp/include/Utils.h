/**
 * @file Utils.h
 * @brief Common utilities and data structures for FKG and FIS modules
 */

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

namespace Utils {

// Type definitions
using Matrix2D = std::vector<std::vector<double>>;
using StringMatrix = std::vector<std::vector<std::string>>;

/**
 * @brief Convert vector to string
 */
template<typename T>
std::string vectorToString(const std::vector<T>& v, const std::string& delimiter = ", ") {
    if (v.empty()) return "[]";
    
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        oss << v[i];
        if (i < v.size() - 1) oss << delimiter;
    }
    oss << "]";
    return oss.str();
}

/**
 * @brief Print matrix to console
 */
void printMatrix(const Matrix2D& matrix, const std::string& name = "Matrix");

/**
 * @brief Get matrix dimensions
 */
std::pair<size_t, size_t> getDimensions(const Matrix2D& matrix);

/**
 * @brief Transpose matrix
 */
Matrix2D transpose(const Matrix2D& matrix);

/**
 * @brief Flatten matrix to 1D vector
 */
std::vector<double> flatten(const Matrix2D& matrix);

/**
 * @brief Reshape vector to 2D matrix
 */
Matrix2D reshape(const std::vector<double>& flat, size_t rows, size_t cols);

/**
 * @brief L1 normalization
 */
std::vector<double> L1Normalize(const std::vector<double>& v);

/**
 * @brief L2 normalization
 */
std::vector<double> L2Normalize(const std::vector<double>& v);

/**
 * @brief Calculate loss function value
 */
double lossFunction(const std::vector<double>& predict_percent);

// ============================================================================
// Data Conversion Utilities
// ============================================================================

/**
 * @brief Convert std::vector to Python list (via pybind11 compatible)
 */
template<typename T>
std::vector<std::vector<T>> pythonListToMatrix(const std::vector<std::vector<T>>& input) {
    return input;
}

/**
 * @brief Create identity matrix
 */
Matrix2D identity(size_t size);

/**
 * @brief Create zero matrix
 */
Matrix2D zeros(size_t rows, size_t cols);

/**
 * @brief Create ones matrix
 */
Matrix2D ones(size_t rows, size_t cols);

} // namespace Utils

#endif // UTILS_H
