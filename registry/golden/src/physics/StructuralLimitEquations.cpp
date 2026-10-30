// FILEPATH: src/physics/StructuralLimitEquations.cpp
#include "StructuralLimitEquations.h"
#include <unordered_map>
#include <iostream>

std::unordered_map<SLEConstraintType, std::string> constraintNameMap = {
    {SLEConstraintType::Linear, "Linear"},
    {SLEConstraintType::Quadratic, "Quadratic"},
    {SLEConstraintType::Cubic, "Cubic"}
};

SLESolver* StructuralLimitEquations::solver;

void StructuralLimitEquations::initialize(SLESolver* s) {
    solver = s;
}

SLESolver* StructuralLimitEquations::getSolver() {
    return solver;
}
