// FILEPATH: src/material/MaterialProfile.h
#ifndef MATERIAL_PROFILE_H
#define MATERIAL_PROFILE_H

#include <CL/cl.hpp>

class MaterialProfile {
public:
    void setTexture(const cl::Image2D& texture);
    const cl::Image2D& getTexture() const;

private:
    cl::Image2D texture;
};

#endif // MATERIAL_PROFILE_H
