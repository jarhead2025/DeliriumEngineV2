#ifndef ANCHORPOOL_SNAPPED_H
#define ANCHORPOOL_SNAPPED_H

#include <cstddef>
#include <cstdint>

class AnchorPool_snapped {
public:
    AnchorPool_snapped() = default;
    ~AnchorPool_snapped() = default;
    void allocate();
    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);
};

#endif // ANCHORPOOL_SNAPPED_H
