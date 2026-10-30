// FILEPATH: src/math/cholesky_eigen.cpp

#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace delirium {

#if __has_include(<Eigen/Cholesky>)

bool llt_eigen(const Eigen::MatrixXf& A, Eigen::MatrixXf& L_out, float* resid) {
    // Using Eigen's LLT for Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXf> llt(A);
    if (llt.info() != Eigen::Success) return false;
    L_out = llt.matrixL();
    if (resid) *resid = (A - L_out * L_out.transpose()).cwiseAbs().maxCoeff();
    return true;
}

#endif

} // namespace delirium
