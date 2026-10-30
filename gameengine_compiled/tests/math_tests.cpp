// FILEPATH: tests/math_tests.cpp
#include "Tensor.h"
#include "CholeskyDecomposition.h"

#include <cassert>
#include <iostream>

int main() {
    Tensor A, L;

    A.set(0, 0, 4.0f);
    A.set(0, 1, 2.0f);
    A.set(0, 2, 1.0f);

    A.set(1, 0, 2.0f);
    A.set(1, 1, 5.0f);
    A.set(1, 2, -1.0f); 

    A.set(2, 0, 1.0f);
    A.set(2, 1, -1.0f);
    A.set(2, 2, 3.0f);

    cholesky_decomposition(&A, &L);

    float l00 = L.get(0, 0), l01 = L.get(0, 1), l02 = L.get(0, 2);
    float l10 = L.get(1, 0), l11 = L.get(1, 1), l12 = L.get(1, 2);
    float l20 = L.get(2, 0), l21 = L.get(2, 1), l22 = L.get(2, 2);

    std::cout << "L:" << std::endl;
    std::cout << l00 << "\t" << l01 << "\t" << l02 << std::endl;
    std::cout << l10 << "\t" << l11 << "\t" << l12 << std::endl;
    std::cout << l20 << "\t" << l21 << "\t" << l22 << std::endl;

    assert(std::abs(l00 - 2.0f) < 1e-6);
    assert(std::abs(l01 - 0.5f) < 1e-6);
    assert(std::abs(l02 - 0.25f) < 1e-6);

    assert(std::abs(l10 - 1.0f) < 1e-6);
    assert(std::abs(l11 - std::sqrt(3.75)) < 1e-6);
    assert(std::abs(l12 + (3.5 / std::sqrt(3.75))) < 1e-6);

    assert(std::abs(l20 - 0.25f) < 1e-6);
    assert(std::abs(l21 + (4.25 / std::sqrt(3.75))) < 1e-6);
    assert(std::abs(l22 - std::sqrt((20.9375 - ((4.25*4.25) / 3.75)))) < 1e-6);

    return 0;
}
