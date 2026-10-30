// FILEPATH: src/math/cholesky_solver.cpp
#include "CholeskyDecomposition.h"
#include <stdexcept>
#include <cmath>

bool cholesky_solve(const Tensor* L, const float* b, float* x) {
    int n = L->c0.size();

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
            sum += L->c0[i][k] * x[k];
        }
        x[i] = (b[i] - sum) / L->c0[i][i];
    }

    // Back substitution to get the solution vector x
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int k = i + 1; k < n; ++k) {
            sum += L->c0[k][i] * x[k];
        }
        x[i] = (x[i] - sum) / L->c0[i][i];
    }

    return true;
}
