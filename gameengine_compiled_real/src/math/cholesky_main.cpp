// FILEPATH: src/math/cholesky_main.cpp
#include "CholeskyDecomposition.h"
#include <iostream>
#include <cmath>

int main() {
    Tensor A, L;
    int n = 3;
    A.c0.resize(n, std::vector<float>(n));
    L.c0.resize(n, std::vector<float>(n));

    // Example matrix A (must be positive definite)
    A.set(0, 0, 4.0f);
    A.set(0, 1, 12.0f);
    A.set(0, 2, -16.0f);
    A.set(1, 0, 12.0f);
    A.set(1, 1, 37.0f);
    A.set(1, 2, -43.0f);
    A.set(2, 0, -16.0f);
    A.set(2, 1, -43.0f);
    A.set(2, 2, 98.0f);

    if (cholesky_decompose(&A, &L)) {
        std::cout << "Successful Cholesky decomposition:" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << L.c0[i][j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Cholesky decomposition failed." << std::endl;
    }

    return 0;
}
