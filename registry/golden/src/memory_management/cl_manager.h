#ifndef CL_MANAGER_H
#define CL_MANAGER_H

#include <cstddef>
#include <cstdint>

class cl_manager {
public:
    cl_manager() = default;
    ~cl_manager() = default;
    void allocate();
    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);
};

#endif // CL_MANAGER_H
