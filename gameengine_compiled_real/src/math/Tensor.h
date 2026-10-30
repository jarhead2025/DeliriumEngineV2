// FILEPATH: src/math/Tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cmath>

struct Tensor {
    std::vector<std::vector<float>> c0;
    std::vector<std::vector<float>> c1;

    // Default constructor for a 3x3 tensor
    Tensor() : c0(3, std::vector<float>(3, 0.0f)), c1(3, std::vector<float>(3, 0.0f)) {}

    float get(int i, int j) const {
        return c0[i][j];
    }

    void set(int i, int j, float value) {
        c0[i][j] = value;
    }
};

#endif // TENSOR_H

