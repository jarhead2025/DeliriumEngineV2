---
agent: pipeline
topic: eigen_llt
updated: 2026-09-15T14:20:23+00:00
---

Eigen::LLT work order for SPD matrices.


=== AGENT WORK ORDER: Eigen::LLT (headers are installed) ===
Math agents MUST execute the following when touching solvers or math tests:

1. In src/math (or equivalent), provide:
   - pure C++ cholesky_decompose / LLT
   - #if __has_include(<Eigen/Cholesky>) branch using Eigen::LLT<MatrixXf>
2. Verify with properties only (A = B*B^T residual), never a memorized L table.
3. Cross-check pure residual vs Eigen residual on the same A.
4. Use tools: search_code_index / trace_symbol for existing solvers;
   search_reference_code for LocalFEM dense_solve patterns;
   write_agent_cheatsheet("eigen_llt", ...) after a working patch.
5. FILEPATH discipline: complete files only, no placeholders, no gtest.
6. Include style:
   #include <Eigen/Dense>
   #include <Eigen/Cholesky>
   Compile assumes -I to eigen3 (pipeline adds this when detected).

Snippet to adapt (not copy blindly — fit project types):
```cpp
#include <Eigen/Dense>
#include <Eigen/Cholesky>
bool llt_eigen(const Eigen::MatrixXf& A, Eigen::MatrixXf& L_out, float* resid) {
  Eigen::LLT<Eigen::MatrixXf> llt(A);
  if (llt.info() != Eigen::Success) return false;
  L_out = llt.matrixL();
  if (resid) *resid = (A - L_out * L_out.transpose()).cwiseAbs().maxCoeff();
  return true;
}
```


// Eigen::LLT path (headers available on this machine)
#include <Eigen/Dense>
#include <Eigen/Cholesky>
// A must be symmetric positive definite (e.g. A = B * B.transpose())
Eigen::MatrixXf A = /* SPD */;
Eigen::LLT<Eigen::MatrixXf> llt(A);
if (llt.info() != Eigen::Success) {
  // not SPD or numerical failure — fall back to pure C++ or report false
} else {
  Eigen::MatrixXf L = llt.matrixL();           // lower triangular
  Eigen::MatrixXf recon = L * L.transpose();   // should approx A
  const float resid = (A - recon).cwiseAbs().maxCoeff();
  // property: resid < 1e-4f (float) or 1e-10 (double)
}
// Prefer MatrixXf / MatrixXd; for fixed 3x3 tests Matrix3f is fine.
