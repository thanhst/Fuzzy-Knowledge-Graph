#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <set>
#include <map>
#include <numeric>  // Fix missing header for accumulate
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;
using namespace std;

typedef vector<vector<double>> Matrix;

int combination(int k, int n) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    int res = 1;
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    return res;
}

vector<vector<double>> calculateA(const vector<vector<int>>& base) {
    size_t row = base.size();
    size_t colum = base[0].size();
    int comb = combination(4, colum - 1);
    vector<vector<double>> A(row, vector<double>(comb, 0.0));

    for (size_t r1 = 0; r1 < row; r1++) {
        vector<int> k(comb, 0);
        size_t temp = 0;

        for (size_t a = 0; a < colum - 4; a++) {
            for (size_t b = a + 1; b < colum - 3; b++) {
                for (size_t c = b + 1; c < colum - 2; c++) {
                    for (size_t d = c + 1; d < colum - 1; d++) {
                        for (size_t r2 = 0; r2 < row; r2++) {
                            if (base[r1][a] == base[r2][a] &&
                                base[r1][b] == base[r2][b] &&
                                base[r1][c] == base[r2][c] &&
                                base[r1][d] == base[r2][d]) {
                                k[temp]++;
                            }
                        }
                        A[r1][temp] = static_cast<double>(k[temp]) / row;
                        temp++;
                    }
                }
            }
        }
    }
    return A;
}

vector<vector<double>> calculateM(const vector<vector<int>>& base) {
    size_t row = base.size();
    size_t colum = base[0].size();
    vector<vector<double>> M(row, vector<double>(colum - 1, 0.0));

    for (size_t t1 = 0; t1 < row; t1++) {
        vector<int> k(colum - 1, 0);
        size_t temp = 0;

        for (size_t i = 0; i < colum - 1; i++) {
            for (size_t t2 = 0; t2 < row; t2++) {
                if (base[t1][i] == base[t2][i] && base[t1][colum - 1] == base[t2][colum - 1]) {
                    k[temp]++;
                }
            }
            M[t1][temp] = static_cast<double>(k[temp]) / row;
            temp++;
        }
    }
    return M;
}

vector<vector<double>> calculateB(const vector<vector<int>>& base, const vector<vector<double>>& A, const vector<vector<double>>& M) {
    size_t row = base.size();
    size_t colum = base[0].size();
    int comb = combination(3, colum - 1);
    vector<vector<double>> B(row, vector<double>(comb, 0.0));

    for (size_t r = 0; r < row; r++) {
        size_t temp = 0;

        for (size_t a = 0; a < colum - 3; a++) {
            for (size_t b = a + 1; b < colum - 2; b++) {
                for (size_t c = b + 1; c < colum - 1; c++) {
                    double min_val = min({M[r][a], M[r][b], M[r][c]});
                    double sum_A_r = accumulate(A[r].begin(), A[r].end(), 0.0);
                    B[r][temp] = sum_A_r * min_val;
                    temp++;
                }
            }
        }
    }
    return B;
}

vector<vector<double>> calculateC(const vector<vector<int>>& base, const vector<vector<double>>& B) {
    size_t row = base.size();
    size_t colum = base[0].size();
    size_t cols = 2 * combination(3, colum - 1);
    vector<vector<double>> C(row, vector<double>(cols, 0.0));

    for (size_t r1 = 0; r1 < row; r1++) {
        size_t temp = 0;

        for (size_t i = 0; i < 2; i++) {
            for (size_t a = 0; a < colum - 3; a++) {
                for (size_t b = a + 1; b < colum - 2; b++) {
                    for (size_t c = b + 1; c < colum - 1; c++) {
                        for (size_t r2 = 0; r2 < row; r2++) {
                            if (base[r1][a] == base[r2][a] &&
                                base[r1][b] == base[r2][b] &&
                                base[r1][c] == base[r2][c] &&
                                base[r2][colum - 1] == i) {
                                C[r1][temp] += B[r2][temp % combination(3, colum - 1)];
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

// pair<int, double> computeFISA(const vector<vector<int>>& base, const vector<vector<double>>& C, const vector<int>& list, int n_labels) {
//     size_t colum = base[0].size();
//     size_t row = base.size();

//     int cols = combination(3, colum - 1);

//     // Create a vector of vectors to hold the values for each label
//     vector<vector<double>> C_labels(n_labels, vector<double>(cols, 0));

//     size_t t = 0;

//     // Iterate over all possible combinations of three columns
//     for (size_t a = 0; a < colum - 3; a++) {
//         for (size_t b = a + 1; b < colum - 2; b++) {
//             for (size_t c = b + 1; c < colum - 1; c++) {
//                 // Iterate through all rows
//                 for (size_t r = 0; r < row - 1; r++) {
//                     // Check each label and update the corresponding vector in C_labels
//                     for (int label = 0; label < n_labels; label++) {
//                         if (base[r][a] == list[a] && base[r][b] == list[b] && base[r][c] == list[c] && base[r][colum - 1] == label) {
//                             C_labels[label][t] = C[r][t + label * cols];
//                         }
//                     }
//                 }
//                 t++;
//             }
//         }
//     }

//     // Compute D0, D1, ..., Dn-1
//     vector<double> D(n_labels, 0);
//     for (int i = 0; i < n_labels; i++) {
//         D[i] = *max_element(C_labels[i].begin(), C_labels[i].end()) + *min_element(C_labels[i].begin(), C_labels[i].end());
//     }

//     // Find the label with the highest D value
//     double max_D = *max_element(D.begin(), D.end());
//     int label_index = distance(D.begin(), find(D.begin(), D.end(), max_D));

//     // Return the label with the highest D value and its ratio
//     double sum_D = accumulate(D.begin(), D.end(), 0.0);
//     return {label_index, D[label_index] / sum_D};
// }
pair<int, double> computeFISA(const vector<vector<int>>& base, const vector<vector<double>>& C, const vector<int>& list) {
    size_t colum = base[0].size();
    size_t row = base.size();

    int cols = combination(3, colum - 1);

    vector<double> C0(cols, 0);
    vector<double> C1(cols, 0);

    size_t t = 0;

    for (size_t a = 0; a < colum - 3; a++) {
        for (size_t b = a + 1; b < colum - 2; b++) {
            for (size_t c = b + 1; c < colum - 1; c++) {
                for (size_t r = 0; r < row - 1; r++) {
                    if (base[r][a] == list[a] && base[r][b] == list[b] && base[r][c] == list[c] && base[r][colum - 1] == 0) {
                        C0[t] = C[r][t + 0 * cols];
                    }
                    if (base[r][a] == list[a] && base[r][b] == list[b] && base[r][c] == list[c] && base[r][colum - 1] == 1) {
                        C1[t] = C[r][t + 1 * cols];
                    }
                }
                t++;
            }
        }
    }

    double D0 = *max_element(C0.begin(), C0.end()) + *min_element(C0.begin(), C0.end());
    double D1 = *max_element(C1.begin(), C1.end()) + *min_element(C1.begin(), C1.end());

    if (D0 > 9 * D1) {
        return {0, D0 / (D0 + D1)};
    } else {
        return {1, D1 / (D0 + D1)};
    }
}


PYBIND11_MODULE(fisa_module, m) {
    m.def("computeFISA", &computeFISA);
    m.def("calculateA", &calculateA);
    m.def("calculateM", &calculateM);
    m.def("calculateB", &calculateB);
    m.def("calculateC", &calculateC);
}