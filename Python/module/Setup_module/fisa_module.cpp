#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <iostream> // Include iostream for cout

namespace py = pybind11;
using namespace std;

typedef vector<vector<double>> Matrix;

int combination(int k, int n) {
    if (k == 0 || k == n) {
        return 1;
    }
    if (k == 1) {
        return n;
    }
    return combination(k - 1, n - 1) + combination(k, n - 1);
}

vector<vector<double>> caculateA(const vector<vector<int>>& base) {
    size_t row = base.size();
    size_t colum = base[0].size();
    int comb = combination(4, colum - 1);
    vector<vector<double>> A(row, vector<double>(comb, 0.0));

    for (size_t r1 = 0; r1 < row; r1++) {
        vector<int> k(comb, 0);
        int temp = 0;
        
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

vector<vector<double>> caculateM(const vector<vector<int>>& base) {
    size_t row = base.size();
    size_t colum = base[0].size();
    vector<vector<double>> M(row, vector<double>(colum - 1, 0.0));

    for (size_t t1 = 0; t1 < row; t1++) {
        vector<int> k(colum - 1, 0);
        int temp = 0;
        
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

vector<vector<double>> caculateB(const vector<vector<int>>& base, const vector<vector<double>>& A, const vector<vector<double>>& M) {
    size_t row = base.size();
    size_t colum = base[0].size();
    int comb = combination(3, colum - 1);
    vector<vector<double>> B(row, vector<double>(comb, 0.0));

    for (size_t r = 0; r < row; r++) {
        int temp = 0;

        for (size_t a = 0; a < colum - 3; a++) {
            for (size_t b = a + 1; b < colum - 2; b++) {
                for (size_t c = b + 1; c < colum - 1; c++) {
                    double min_val = min({M[r][a], M[r][b], M[r][c]});
                    double sum_A_r = 0;
                    for (size_t i = 0; i < A[r].size(); i++) {
                        sum_A_r += A[r][i];  // Sum all elements in A[r]
                    }
                    B[r][temp] = sum_A_r * min_val;
                    temp++;
                }
            }
        }
    }

    return B;
}

vector<vector<double>> caculateC(const vector<vector<int>>& base, const vector<vector<double>>& B) {
    size_t row = base.size();
    size_t colum = base[0].size();
    size_t cols = 2 * combination(3, colum - 1);  
    vector<vector<double>> C(row, vector<double>(cols, 0.0));

    for (size_t r1 = 0; r1 < row; r1++) {
        int temp = 0;
        
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
    m.def("computeFISA", &computeFISA, "Compute FISA matrix");
    m.def("caculateA", &caculateA, "Compute A");
    m.def("caculateM", &caculateM, "Compute M");
    m.def("caculateB", &caculateB, "Compute B");
    m.def("caculateC", &caculateC, "Compute C");
}
