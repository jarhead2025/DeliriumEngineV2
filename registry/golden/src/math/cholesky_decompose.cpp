// FILEPATH: src/math/cholesky_decompose.cpp

#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace delirium {

bool cholesky_decompose(const Eigen::MatrixXf& A, Eigen::MatrixXf& L_out, float* resid) {
    // Pure C++ Cholesky decomposition using Eigen's LLT
    Eigen::LLT<Eigen::MatrixXf> llt(A);
    if (llt.info() != Eigen::Success) return false;
    L_out = llt.matrixL();
    if (resid) *resid = (A - L_out * L_out.transpose()).cwiseAbs().maxCoeff();
    return true;
}

} // namespace delirium
