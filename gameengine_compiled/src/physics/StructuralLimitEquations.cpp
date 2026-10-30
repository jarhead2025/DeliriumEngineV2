#include "../memory_management/StructuralLimitEquations.h"

float calculate_bending_stiffness(float youngsModulus, float momentOfInertia, float length) {
    return (youngsModulus * momentOfInertia) / length;
}

float calculate_lateral_strain(float poissonsRatio, float strainXDirection) {
    return poissonsRatio * strainXDirection;
}
