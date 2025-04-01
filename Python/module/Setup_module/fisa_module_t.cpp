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
    int comb = combination(4, col - 1); // Sửa lại công thức tổ hợp
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
        // Khởi tạo vector k đúng cách
        vector<int> k(colum - 1, 0);

        for (int i = 0; i < colum - 1; ++i)
        {
            for (int t2 = 0; t2 < row; ++t2)
            {
                if (base[t1][i] == base[t2][i] && base[t1][colum - 1] == base[t2][colum - 1])
                {
                    k[i]++; // Sửa: sử dụng i thay vì temp
                }
            }
            M[t1][i] = static_cast<double>(k[i]) / row; // Sửa: sử dụng i thay vì temp
        }
    }

    cout << "done M" << endl;
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
                    // Thay đổi: Chỉ sử dụng thông tin liên quan đến (a,b,c)
                    // thay vì tính tổng toàn bộ A[r]
                    double a_sum = 0.0;
                    int count = 0;
                    
                    // Tính trung bình của các phần tử A liên quan đến a, b, c
                    for (int d = 0; d < colum - 1; ++d) {
                        if (d != a && d != b && d != c) {
                            int idx = 0;
                            // Tính chỉ số tương ứng trong ma trận A
                            // (Đây là một ước lượng, cần điều chỉnh tùy thuộc vào cách thức lưu trữ A)
                            if (d > c) idx = temp;
                            else if (d > b) idx = temp - 1;
                            else if (d > a) idx = temp - 2;
                            else idx = temp - 3;
                            
                            if (idx >= 0 && idx < A[r].size()) {
                                a_sum += A[r][idx];
                                count++;
                            }
                        }
                    }
                    
                    double avg_a = (count > 0) ? a_sum / count : 0.0;
                    B[r][temp] = avg_a * min({M[r][a], M[r][b], M[r][c]});
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
        for (int label = 1; label <= 6; ++label) {
            int offset = (label - 1) * comb_size;
            int idx = 0;
            
            for (int a = 0; a < colum - 3; ++a)
            {
                for (int b = a + 1; b < colum - 2; ++b)
                {
                    for (int c = b + 1; c < colum - 1; ++c)
                    {
                        C[r1][offset + idx] = 0.0; // Reset trước khi tích lũy
                        int count = 0;
                        
                        for (int r2 = 0; r2 < row; ++r2)
                        {
                            if (base[r1][a] == base[r2][a] &&
                                base[r1][b] == base[r2][b] &&
                                base[r1][c] == base[r2][c] &&
                                base[r2][colum - 1] == label)
                            {
                                C[r1][offset + idx] += B[r2][idx];
                                count++;
                            }
                        }
                        
                        // Lấy trung bình nếu có nhiều mẫu thỏa mãn
                        if (count > 0) {
                            C[r1][offset + idx] /= count;
                        }
                        
                        idx++;
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

    // Debug: In ra kích thước các ma trận
    cout << "Matrix sizes - row: " << row << ", cols: " << cols << endl;

    vector<vector<double>> C_values(6, vector<double>(cols, 0.0));
    vector<int> counts(6, 0); // Đếm số mẫu cho mỗi nhãn

    int t = 0;
    for (int a = 0; a < colum - 3; ++a)
    {
        for (int b = a + 1; b < colum - 2; ++b)
        {
            for (int c = b + 1; c < colum - 1; ++c)
            {
                for (int label = 0; label < 6; ++label) {
                    C_values[label][t] = 0.0; // Reset trước khi tích lũy
                    counts[label] = 0;
                }
                
                for (int r = 0; r < row; ++r)
                {
                    if (base[r][a] == list[a] && base[r][b] == list[b] && base[r][c] == list[c])
                    {
                        int label = base[r][colum - 1] - 1;
                        if (label >= 0 && label < 6) {
                            C_values[label][t] += C[r][t + (label * cols)];
                            counts[label]++;
                        }
                    }
                }
                
                // Lấy trung bình nếu có nhiều mẫu
                for (int label = 0; label < 6; ++label) {
                    if (counts[label] > 0) {
                        C_values[label][t] /= counts[label];
                    }
                }
                
                t++;
            }
        }
    }

    // Debug: In ra một số giá trị của C_values
    cout << "Some C_values:" << endl;
    for (int i = 0; i < 6; ++i) {
        cout << "Label " << (i+1) << ": ";
        for (int j = 0; j < min(5, (int)C_values[i].size()); ++j) {
            cout << C_values[i][j] << " ";
        }
        cout << endl;
    }

    vector<double> D(6, 0.0);
    for (int i = 0; i < 6; ++i)
    {
        if (!C_values[i].empty())
        {
            double sum = accumulate(C_values[i].begin(), C_values[i].end(), 0.0);
            int non_zero = count_if(C_values[i].begin(), C_values[i].end(),[](double x) { return x > 0.0; });
            
            if (non_zero > 0) {
                D[i] = sum / non_zero;
            } else {
                D[i] = 0.0;
            }
        }
    }

    // cout << "D values: ";
    // for (int i = 0; i < 6; ++i) {
    //     cout << D[i] << " ";
    // }
    // cout << endl;

    int bestIndex = max_element(D.begin(), D.end()) - D.begin();
    
    vector<int> candidates;
    double max_d = D[bestIndex];
    for (int i = 0; i < 6; ++i) {
        if (fabs(D[i] - max_d) < 1e-9) {
            candidates.push_back(i);
        }
    }
    
    if (candidates.size() > 1) {
        vector<int> label_counts(6, 0);
        for (int r = 0; r < row; ++r) {
            int label = base[r][colum - 1] - 1;
            if (label >= 0 && label < 6) {
                label_counts[label]++;
            }
        }
        
        int max_count = -1;
        for (int candidate : candidates) {
            if (label_counts[candidate] > max_count) {
                max_count = label_counts[candidate];
                bestIndex = candidate;
            }
        }
    }
    
    double D_sum = accumulate(D.begin(), D.end(), 0.0);
    double confidence = (D_sum > 0.001) ? D[bestIndex] / D_sum : 0.0;
    
    if (confidence < 0.2 && candidates.size() == 1) {
        confidence = 0.2;
    }
    
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