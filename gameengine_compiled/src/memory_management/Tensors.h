// FILEPATH: src/memory_management/Tensors.h
#ifndef TENSORS_H
#define TENSORS_H

#include <cstddef>
#include <cstdint>
#include <vector>

class Tensors {
public:
    Tensors() = default;
    ~Tensors() = default;

    void* allocate(std::size_t bytes) {
        return aligned_alloc(16, bytes);
    }

    void deallocate(void* ptr) {
        free(ptr);
    }
};

#endif // TENSORS_H
