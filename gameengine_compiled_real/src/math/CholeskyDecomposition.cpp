// FILEPATH: src/math/CholeskyDecomposition.cpp
#include "CholeskyDecomposition.h"
#include <cmath>

bool cholesky_decomposition(Tensor* A, Tensor* L) {
    if (A->c0.size() != A->c1.size() || A->c0.size() != L->c0.size()) return false;
    int n = A->c0.size();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            if (j == i) {
                for (int k = 0; k < j; ++k) {
                    sum += L->c0[i][k] * L->c0[k][j];
                }
                L->c0[i][i] = std::sqrt(std::max(static_cast<double>(A->c0[i][i]) - sum, 1e-6)); // Avoid negative square root
            } else {
                for (int k = 0; k < j; ++k) {
                    sum += L->c0[i][k] * L->c0[k][j];
                }
                if (L->c0[j][j] > 1e-6) {
                    L->c0[i][j] = (static_cast<double>(A->c0[i][j]) - sum) / L->c0[j][j];
                } else {
                    return false; // Division by zero in Cholesky decomposition
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            L->c0[i][j] = 0.0f; // Zero out the upper triangle
        }
    }

    return true;
}

