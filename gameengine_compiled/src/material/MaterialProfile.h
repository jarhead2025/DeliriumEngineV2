// FILEPATH: src/material/MaterialProfile.h
#ifndef MATERIAL_PROFILE_H
#define MATERIAL_PROFILE_H

#include <string>

struct MaterialProfile {
public:
    std::string name;
    float youngsModulus;
    float poissonsRatio;
    float densityValue;
    float modalDampingState;

    void set_youngs_modulus(float modulus);
    void set_poissons_ratio(float ratio);
    void set_density(float density);
    void set_modal_damping_state(float dampingState);

    MaterialProfile(const std::string& name, float youngsModulus, float poissonsRatio, float density, float modalDampingState)
        : name(name), youngsModulus(youngsModulus), poissonsRatio(poissonsRatio), densityValue(density), modalDampingState(modalDampingState) {}

    MaterialProfile() : name(""), youngsModulus(0.0f), poissonsRatio(0.0f), densityValue(0.0f), modalDampingState(0.0f) {}
};

#endif // MATERIAL_PROFILE_H
