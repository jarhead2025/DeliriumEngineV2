// FILEPATH: src/material/MaterialProfile.cpp
#include "MaterialProfile.h"
#include <stdexcept>

void MaterialProfile::setTexture(const cl::Image2D& texture) {
    if (texture.getInfo<CL_IMAGE_FORMAT>().image_channel_order != CL_RGBA)
        throw std::invalid_argument("Invalid channel order for material texture. Expected CL_RGBA.");

    this->texture = texture;
}

const cl::Image2D& MaterialProfile::getTexture() const {
    return texture;
}
