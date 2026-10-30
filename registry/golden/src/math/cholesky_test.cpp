// FILEPATH: src/math/cholesky_test.cpp

#include <Eigen/Dense>
#include "cholesky_decompose.h"
#include "cholesky_eigen.h"

void test_cholesky() {
    Eigen::Matrix3f A;
    A << 4, 12, -16,
         12, 37, -43,
        -16, -43, 98;

    Eigen::MatrixXf L_out(3, 3);
    float pure_resid = 0.0f;
    bool success_pure = cholesky_decompose(A, L_out, &pure_resid);

    #if __has_include(<Eigen/Cholesky>)
    float eigen_resid = 0.0f;
    bool success_eigen = llt_eigen(A, L_out, &eigen_resid);
    #endif

    if (success_pure) {
        std::cout << "Pure C++ Cholesky Success: Residual = " << pure_resid << std::endl;
        #if __has_include(<Eigen/Cholesky>)
        std::cout << "Eigen Cholesky Success: Residual = " << eigen_resid << std::endl;
        #endif
    } else {
        std::cout << "Pure C++ Cholesky Failed" << std::endl;
        #if __has_include(<Eigen/Cholesky>)
        if (!success_eigen) {
            std::cout << "Eigen Cholesky Also Failed" << std::endl;
        }
        #endif
    }
}

int main() {
    test_cholesky();
    return 0;
}
