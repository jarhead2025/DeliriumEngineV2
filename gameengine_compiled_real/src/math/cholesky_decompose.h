// FILEPATH: src/math/cholesky_decompose.h

#ifndef SRC_MATH_CHOLESKY_DECOMPOSE_H_
#define SRC_MATH_CHOLESKY_DECOMPOSE_H_

#include <Eigen/Dense>
#include <Eigen/Cholesky>

bool cholesky_decompose(const Eigen::MatrixXf& A, Eigen::MatrixXf& L_out, float* resid);
bool cholesky_eigen(const Eigen::MatrixXf& A, Eigen::MatrixXf& L_out, float* resid);

#endif  // SRC_MATH_CHOLESKY_DECOMPOSE_H_
