```cpp
// FILEPATH: tests/math_invariants_test.cpp

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

// Pure C++ Cholesky decomposition function
bool cholesky_decompose(const Eigen::MatrixXf& A, Eigen::MatrixXf& L) {
    int n = A.rows();
    L.setZero(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = 0;
            if (j == i) {
                for (int k = 0; k < j; ++k)
                    s += L(j, k) * L(j, k);
                L(i, j) = std::sqrt(A(i, i) - s);
            } else {
                for (int k = 0; k < j; ++k)
                    s += L(i, k) * L(j, k);
                L(i, j) = (A(i, j) - s) / L(j, j);
            }
        }
    }

    return true;
}

// Function to check if a matrix is lower triangular
bool is_lower_triangular(const Eigen::MatrixXf& M) {
    int n = M.rows();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (M(i, j) != 0)
                return false;
        }
    }
    return true;
}

// Function to check if the diagonal of a matrix is positive
bool has_positive_diagonal(const Eigen::MatrixXf& M) {
    int n = M.rows();
    for (int i = 0; i < n; ++i) {
        if (M(i, i) <= 0)
            return false;
    }
    return true;
}

// Main test function
int main() {
    // Generate a random matrix B and construct SPD A = B * B^T
    int n = 5;
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf A = B * B.transpose();

    // Pure C++ Cholesky decomposition
    Eigen::MatrixXf L_cpp(n, n);
    if (!cholesky_decompose(A, L_cpp)) {
        std::cerr << "Pure C++ Cholesky decomposition failed." << std::endl;
        return 1;
    }

    // Check properties of the pure C++ L matrix
    if (!is_lower_triangular(L_cpp) || !has_positive_diagonal(L_cpp)) {
        std::cerr << "Pure C++ L matrix is not lower triangular or does not have positive diagonal." << std::endl;
        return 1;
    }

    // Calculate residual for pure C++ Cholesky
    float resid_cpp = (A - L_cpp * L_cpp.transpose()).cwiseAbs().maxCoeff();
    if (resid_cpp >= 1e-3) {
        std::cerr << "Pure C++ Cholesky decomposition has large residual: " << resid_cpp << std::endl;
        return 1;
    }

    // Eigen::LLT decomposition
#if __has_include(<Eigen/Cholesky>)
    Eigen::MatrixXf L_eigen(n, n);
    float resid_eigen = 0.0f;
    if (!llt_eigen(A, L_eigen, &resid_eigen)) {
        std::cerr << "Eigen::LLT decomposition failed." << std::endl;
        return 1;
    }

    // Check properties of the Eigen L matrix
    if (!is_lower_triangular(L_eigen) || !has_positive_diagonal(L_eigen)) {
        std::cerr << "Eigen L matrix is not lower triangular or does not have positive diagonal." << std::endl;
        return 1;
    }

    // Compare residuals between pure C++ and Eigen
    if (std::abs(resid_cpp - resid_eigen) >= 1e-3) {
        std::cerr << "Discrepancy in residuals between pure C++ and Eigen: "
                  << resid_cpp << ", " << resid_eigen << std::endl;
        return 1;
    }
#endif

    std::cout << "All tests passed." << std::endl;
    return 0;
}
```