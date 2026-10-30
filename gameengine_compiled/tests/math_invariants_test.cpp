// Property-based math harness - pure C++ LLT + optional Eigen::LLT cross-check.
#include <cmath>
#include <iostream>
#include <vector>
#if defined(DELIRIUM_HAS_EIGEN) || __has_include(<Eigen/Cholesky>)
#  include <Eigen/Dense>
#  include <Eigen/Cholesky>
#  define DELIRIUM_USE_EIGEN 1
#else
#  define DELIRIUM_USE_EIGEN 0
#endif

static int g_failed = 0;
static int g_passed = 0;

#define CHECK(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << (msg) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    ++g_failed; \
  } else { ++g_passed; } \
} while (0)

static bool chol_ref(const std::vector<std::vector<float>>& A,
                     std::vector<std::vector<float>>& L) {
  const int n = (int)A.size();
  L.assign(n, std::vector<float>(n, 0.f));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = 0.0;
      for (int k = 0; k < j; ++k) sum += (double)L[i][k] * (double)L[j][k];
      if (i == j) {
        double v = (double)A[i][i] - sum;
        if (v <= 1e-12) return false;
        L[i][j] = (float)std::sqrt(v);
      } else {
        if (std::fabs(L[j][j]) < 1e-12) return false;
        L[i][j] = (float)(((double)A[i][j] - sum) / (double)L[j][j]);
      }
    }
  }
  return true;
}

static double max_llt_residual(const std::vector<std::vector<float>>& A,
                               const std::vector<std::vector<float>>& L) {
  const int n = (int)A.size();
  double maxe = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double s = 0.0;
      for (int k = 0; k <= j; ++k) s += (double)L[i][k] * (double)L[j][k];
      maxe = std::max(maxe, std::fabs(s - (double)A[i][j]));
    }
  }
  return maxe;
}

static void fill_spd_bbt(std::vector<std::vector<float>>& A) {
  const float B[3][3] = {{2,0,0},{6,1,0},{8,-5,3}};
  A.assign(3, std::vector<float>(3, 0.f));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += (double)B[i][k] * (double)B[j][k];
      A[i][j] = (float)s;
    }
}

static void test_property_spd_from_bbt() {
  std::vector<std::vector<float>> A;
  fill_spd_bbt(A);
  std::vector<std::vector<float>> L;
  CHECK((chol_ref(A, L)), "pure chol_ref on SPD A=BB^T");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (j > i) CHECK((std::fabs(L[i][j]) < 1e-6), "L upper ~0");
    }
    CHECK((L[i][i] > 0.f), "L diag positive");
  }
  double resid = max_llt_residual(A, L);
  CHECK((resid < 1e-3), "pure max |A-LLT| < 1e-3");
  std::cout << "pure LLT residual=" << resid << "\n";
}

#if DELIRIUM_USE_EIGEN
static void test_eigen_llt() {
  std::vector<std::vector<float>> A;
  fill_spd_bbt(A);
  Eigen::Matrix3f Ae;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) Ae(i, j) = A[i][j];
  Eigen::LLT<Eigen::Matrix3f> llt(Ae);
  CHECK((llt.info() == Eigen::Success), "Eigen::LLT info Success");
  Eigen::Matrix3f Lmat = llt.matrixL();
  Eigen::Matrix3f recon = Lmat * Lmat.transpose();
  double maxe = (double)(Ae - recon).cwiseAbs().maxCoeff();
  CHECK((maxe < 1e-4), "Eigen |A-LLT| residual");
  std::cout << "Eigen::LLT residual=" << maxe << "\n";
  std::vector<std::vector<float>> Lpure;
  CHECK((chol_ref(A, Lpure)), "pure path for cross-check");
  double pure_r = max_llt_residual(A, Lpure);
  CHECK((pure_r < 1e-3 && maxe < 1e-4), "pure and Eigen both accept SPD A");
  std::cout << "cross-check pure=" << pure_r << " eigen=" << maxe << "\n";
}
#endif

static void test_dtype() {
  CHECK((sizeof(float) == 4), "float is 4 bytes");
  CHECK((sizeof(double) == 8), "double is 8 bytes");
  CHECK((std::isfinite(1.0f) && std::isfinite(1.0)), "finite samples");
}

int main() {
  std::cout << "DeliRium math invariants (property-based"
#if DELIRIUM_USE_EIGEN
            << " + Eigen::LLT"
#endif
            << ")\n";
  test_dtype();
  test_property_spd_from_bbt();
#if DELIRIUM_USE_EIGEN
  test_eigen_llt();
#else
  std::cout << "Eigen::LLT skipped (headers not found; pure path only)\n";
#endif
  std::cout << "Passed=" << g_passed << " Failed=" << g_failed << "\n";
  return g_failed == 0 ? 0 : 1;
}

