#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <set>

namespace py = pybind11;
using namespace std;

typedef vector<vector<double>> Matrix;

int combination(int k, int n)
{
    if (k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    int res = 1;
    for (int i = 0; i < k; ++i)
    {
        res *= (n - i);
        res /= (i + 1);
    }
    return res;
}

std::vector<std::vector<double>> calculateA(const std::vector<std::vector<int>> &base)
{
    int row = base.size();
    int col = base[0].size();
    int comb = (col - 1) * (col - 2) * (col - 3) * (col - 4) / 24;
    std::vector<std::vector<double>> A(row, std::vector<double>(comb, 0));

    for (int r1 = 0; r1 < row; ++r1)
    {
        int temp = 0;
        for (int a = 0; a < col - 4; ++a)
        {
            for (int b = a + 1; b < col - 3; ++b)
            {
                for (int c = b + 1; c < col - 2; ++c)
                {
                    for (int d = c + 1; d < col - 1; ++d)
                    {
                        int count = 0;
                        for (int r2 = 0; r2 < row; ++r2)
                        {
                            if (base[r1][a] == base[r2][a] && base[r1][b] == base[r2][b] &&
                                base[r1][c] == base[r2][c] && base[r1][d] == base[r2][d])
                            {
                                count++;
                            }
                        }
                        A[r1][temp++] = static_cast<double>(count) / row;
                    }
                }
            }
        }
    }
    cout << "done A" << endl;
    return A;
}

vector<vector<double>> calculateM(const vector<vector<int>> &base)
{
    int row = base.size();
    if (row == 0)
        return {};

    int colum = base[0].size();
    vector<vector<double>> M(row, vector<double>(colum - 1, 0));

    for (int t1 = 0; t1 < row; ++t1)
    {
        vector<int> k(colum - 1, 0);
        int temp = 0;

        for (int i = 0; i < colum - 1; ++i)
        {
            for (int t2 = 0; t2 < row; ++t2)
            {
                if (base[t1][i] == base[t2][i] && base[t1][colum - 1] == base[t2][colum - 1])
                {
                    k[i]++;
                }
            }
            M[t1][temp] = static_cast<double>(k[temp]) / row;
            temp++;
        }
    }

    return M;
}

vector<vector<double>> calculateB(const vector<vector<int>> &base, const vector<vector<double>> &A, const vector<vector<double>> &M)
{
    int row = base.size();
    if (row == 0)
        return {};

    int colum = base[0].size();
    int comb_size = combination(3, colum - 1);

    vector<vector<double>> B(row, vector<double>(comb_size, 0));

    for (int r = 0; r < row; ++r)
    {
        int temp = 0;
        for (int a = 0; a < colum - 3; ++a)
        {
            for (int b = a + 1; b < colum - 2; ++b)
            {
                for (int c = b + 1; c < colum - 1; ++c)
                {
                    B[r][temp] = accumulate(A[r].begin(), A[r].end(), 0.0) *min({M[r][a], M[r][b], M[r][c]});
                    temp++;
                }
            }
        }
    }

    cout << "done B" << endl;
    return B;
}

vector<vector<double>> calculateC(const vector<vector<int>> &base, const vector<vector<double>> &B)
{
    int row = base.size();
    if (row == 0)
        return {};

    int colum = base[0].size();
    int comb_size = combination(3, colum - 1);
    int cols = 6 * comb_size;

    vector<vector<double>> C(row, vector<double>(cols, 0.0));

    for (int r1 = 0; r1 < row; ++r1)
    {
        int temp = 0;
        for (int i = 1; i <= 6; ++i)
        {
            for (int a = 0; a < colum - 3; ++a)
            {
                for (int b = a + 1; b < colum - 2; ++b)
                {
                    for (int c = b + 1; c < colum - 1; ++c)
                    {
                        for (int r2 = 0; r2 < row; ++r2)
                        {
                            if (base[r1][a] == base[r2][a] &&
                                base[r1][b] == base[r2][b] &&
                                base[r1][c] == base[r2][c] &&
                                base[r2][colum - 1] == i)
                            {
                                C[r1][temp] += B[r2][temp % comb_size];
                            }
                        }
                        temp++;
                    }
                }
            }
        }
    }

    cout << "done C" << endl;
    return C;
}

pair<int, double> FISA(const vector<vector<int>> &base, const vector<vector<double>> &C, const vector<int> &list)
{
    int row = base.size();
    if (row == 0)
        return {0, 0.0};

    int colum = base[0].size();
    int cols = combination(3, colum - 1);

    vector<vector<double>> C_values(6, vector<double>(cols, 0.0));

    int t = 0;
    for (int a = 0; a < colum - 3; ++a)
    {
        for (int b = a + 1; b < colum - 2; ++b)
        {
            for (int c = b + 1; c < colum - 1; ++c)
            {
                for (int r = 0; r < row ; ++r)
                {
                    if (base[r][a] == list[a] && base[r][b] == list[b] && base[r][c] == list[c])
                    {
                        int label = base[r][colum - 1]-1;
                        C_values[label][t] = C[r][t + (label * cols)];
                    }
                }
                t++;
            }
        }
    }

    vector<double> D(6, 0.0);
    for (int i = 0; i < 6; ++i)
    {
        if (!C_values[i].empty())
        {
            D[i] = *max_element(C_values[i].begin(), C_values[i].end()) + *min_element(C_values[i].begin(), C_values[i].end());
        }
    }

    double D_sum = std::accumulate(D.begin(), D.end(), 0.0);
    int bestIndex = std::max_element(D.begin(), D.end()) - D.begin();
    double confidence = (D_sum > 0) ? D[bestIndex] / D_sum : 0.0;
    return {bestIndex + 1, confidence};
}

PYBIND11_MODULE(fisa_module, m)
{
    m.def("FISA", &FISA);
    m.def("calculateA", &calculateA);
    m.def("calculateM", &calculateM);
    m.def("calculateB", &calculateB);
    m.def("calculateC", &calculateC);
}