// FILEPATH: src/physics/StructuralLimitEquations.h
#ifndef STRUCTURAL_LIMIT_EQUATIONS_H
#define STRUCTURAL_LIMIT_EQUATIONS_H

enum class SLEConstraintType {
    Linear,
    Quadratic,
    Cubic
};

class StructuralLimitEquations {
public:
    static void initialize(SLESolver* s);
    static SLESolver* getSolver();

private:
    static SLESolver* solver;
};

#endif // STRUCTURAL_LIMIT_EQUATIONS_H
