#ifndef STRUCTURALLIMITEQUATIONS_H
#define STRUCTURALLIMITEQUATIONS_H

#include <cstddef>
#include <cstdint>

class StructuralLimitEquations {
public:
    StructuralLimitEquations() = default;
    ~StructuralLimitEquations() = default;
    void allocate();
    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);
};

#endif // STRUCTURALLIMITEQUATIONS_H
