// TriInfill: triangular average-fill between sample dots (structured grid + Delaunay).
#pragma once
#include <cstdint>
#include <vector>
#include "../core/Types.h"

namespace DeliRium {

struct SampleDot {
    float sx, sy;   // screen pixels
    float z;        // depth (smaller = nearer)
    float r, g, b;  // 0..255
};

// Structured: dots laid out as rows of `su` (row-major). Builds 2 tris per cell.
void TriInfillStructured(const std::vector<SampleDot>& dots, int su, int sv,
                         int W, int H, std::vector<uint32_t>& rgba,
                         std::vector<float>& zbuf);

// Unstructured: Bowyer–Watson Delaunay in 2D, then barycentric fill.
void TriInfillDelaunay(const std::vector<SampleDot>& dots, int W, int H,
                       std::vector<uint32_t>& rgba, std::vector<float>& zbuf);

} // namespace DeliRium
