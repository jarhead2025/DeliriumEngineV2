// FILEPATH: src/physics/LocalFEM.cpp
#include "cholesky_solver.h"
#include <iostream>
#include <vector>

void test_cholesky() {
  std::vector<std::vector<float>> A = {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};
  std::vector<std::vector<float>> L;
  bool success = cholesky_decompose(A, L);
  if (success) {
    std::cout << "Cholesky decomposition successful." << std::endl;
  } else {
    std::cout << "Cholesky decomposition failed." << std::endl;
  }

  Eigen::MatrixXf A_eigen(3, 3);
  A_eigen << 2, -1, 0,
             -1, 2, -1,
              0, -1, 2;
  Eigen::MatrixXf L_out;
  float resid = 0.0f;
  success = llt_eigen(A_eigen, L_out, &resid);
  if (success) {
    std::cout << "Eigen LLT successful with residual: " << resid << std::endl;
  } else {
    std::cout << "Eigen LLT failed." << std::endl;
  }
}

int main() {
  test_cholesky();
  return 0;
}
