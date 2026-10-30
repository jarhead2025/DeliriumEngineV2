// FILEPATH: tests/math_test.cpp
#include <gtest/gtest.h>
#include "CholeskyDecomposition.h"
#include "Tensor.h"

TEST(Math, CholeskyDecomposition) {
    Tensor A {{4, 2}, {2, 5}};
    Tensor L {{0, 0}, {0, 0}};

    ASSERT_TRUE(cholesky_decomposition(&A, &L));
    ASSERT_NEAR(L.c0[0][0], 2.0f, 1e-6);
    ASSERT_NEAR(L.c0[1][0], 1.0f, 1e-6);
    ASSERT_NEAR(L.c0[1][1], std::sqrt(3.0f), 1e-6);
}
