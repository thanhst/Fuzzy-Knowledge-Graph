/**
 * @file Utils.cpp
 * @brief Implementation of common utilities
 */

#include "Utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace Utils {

void printMatrix(const Matrix2D& matrix, const std::string& name) {
    std::cout << name << " (" << matrix.size() << "x" 
              << (matrix.empty() ? 0 : matrix[0].size()) << "):" << std::endl;
    
    for (size_t i = 0; i < matrix.size(); i++) {
        std::cout << "  [";
        for (size_t j = 0; j < matrix[i].size(); j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                     << matrix[i][j];
            if (j < matrix[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

std::pair<size_t, size_t> getDimensions(const Matrix2D& matrix) {
    if (matrix.empty()) return {0, 0};
    return {matrix.size(), matrix[0].size()};
}

Matrix2D transpose(const Matrix2D& matrix) {
    if (matrix.empty()) return {};
    
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    Matrix2D result(cols, std::vector<double>(rows));
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    
    return result;
}

std::vector<double> flatten(const Matrix2D& matrix) {
    std::vector<double> result;
    
    for (const auto& row : matrix) {
        result.insert(result.end(), row.begin(), row.end());
    }
    
    return result;
}

Matrix2D reshape(const std::vector<double>& flat, size_t rows, size_t cols) {
    if (flat.size() != rows * cols) {
        throw std::invalid_argument("Vector size does not match requested dimensions");
    }
    
    Matrix2D result(rows, std::vector<double>(cols));
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = flat[i * cols + j];
        }
    }
    
    return result;
}

std::vector<double> L1Normalize(const std::vector<double>& v) {
    double sum = 0.0;
    for (double val : v) {
        sum += std::abs(val);
    }
    
    if (sum == 0.0) return v;
    
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = v[i] / sum;
    }
    
    return result;
}

std::vector<double> L2Normalize(const std::vector<double>& v) {
    double sum = 0.0;
    for (double val : v) {
        sum += val * val;
    }
    sum = std::sqrt(sum);
    
    if (sum == 0.0) return v;
    
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = v[i] / sum;
    }
    
    return result;
}

double lossFunction(const std::vector<double>& predict_percent) {
    if (predict_percent.empty()) return 0.0;
    
    double loss = 0.0;
    for (double p : predict_percent) {
        loss += (1.0 - p) * (1.0 - p);
    }
    
    return loss / predict_percent.size();
}

Matrix2D identity(size_t size) {
    Matrix2D result(size, std::vector<double>(size, 0.0));
    
    for (size_t i = 0; i < size; i++) {
        result[i][i] = 1.0;
    }
    
    return result;
}

Matrix2D zeros(size_t rows, size_t cols) {
    return Matrix2D(rows, std::vector<double>(cols, 0.0));
}

Matrix2D ones(size_t rows, size_t cols) {
    return Matrix2D(rows, std::vector<double>(cols, 1.0));
}

} // namespace Utils
