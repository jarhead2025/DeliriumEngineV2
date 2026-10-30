// FILEPATH: src/math/solvers.cpp

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <cmath>

namespace math {
namespace solvers {

bool cholesky_decompose(const std::vector<std::vector<float>>& A,
                        std::vector<std::vector<float>>& L) {
  const int n = (int)A.size();
  L.assign(n, std::vector<float>(n, 0.f));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = 0.0;
      for (int k = 0; k < j; ++k) sum += (double)L[i][k] * (double)L[j][k];
      if (i == j) {
        double v = (double)A[i][i] - sum;
        if (v <= 1e-12f) return false;
        L[i][j] = (float)std::sqrt(v);
      } else {
        if (std::fabs(L[j][j]) < 1e-12f) return false;
        L[i][j] = (float)(((double)A[i][j] - sum) / (double)L[j][j]);
      }
    }
  }
  return true;
}

bool llt_eigen(const Eigen::MatrixXf& A, Eigen::MatrixXf& L_out, float* resid) {
  if (!A.isApprox(A.transpose())) return false; // Ensure A is symmetric
  Eigen::LLT<Eigen::MatrixXf> llt(A);
  if (llt.info() != Eigen::Success) return false;
  L_out = llt.matrixL();
  if (resid) *resid = (A - L_out * L_out.transpose()).cwiseAbs().maxCoeff();
  return true;
}

void cross_check_cholesky_decompose(const std::vector<std::vector<float>>& A) {
  std::vector<std::vector<float>> L_cpp, L_eigen;
  bool success_cpp = cholesky_decompose(A, L_cpp);
  Eigen::MatrixXf A_eigen(A.size(), A.size());
  for (size_t i = 0; i < A.size(); ++i) {
    for (size_t j = 0; j < A.size(); ++j) {
      A_eigen(i, j) = A[i][j];
    }
  }
  
  Eigen::MatrixXf L_matrix;
  float resid_cpp, resid_eigen;
  bool success_eigen = llt_eigen(A_eigen, L_matrix, &resid_eigen);

  if (success_cpp && success_eigen) {
    std::vector<std::vector<float>> L_eigen_vec(L_matrix.rows(), std::vector<float>(L_matrix.cols()));
    for (size_t i = 0; i < L_matrix.rows(); ++i) {
      for (size_t j = 0; j < L_matrix.cols(); ++j) {
        L_eigen_vec[i][j] = L_matrix(i, j);
      }
    }
    
    resid_cpp = 0.0f;
    for (size_t i = 0; i < A.size(); ++i) {
      for (size_t j = 0; j < A.size(); ++j) {
        resid_cpp = std::max(resid_cpp, std::fabs(A[i][j] - (L_cpp[i][j] * L_cpp[j][i])));
      }
    }
    
    const float epsilon = 1e-6f;
    if (std::fabs(resid_cpp - resid_eigen) < epsilon) {
      // Both methods produce similar residuals, verified.
      return;
    } else {
      // Residuals differ significantly, investigate further.
    }
  } else {
    // One or both methods failed decomposition.
  }
}

} // namespace solvers
} // namespace math
