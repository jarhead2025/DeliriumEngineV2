// FILEPATH: src/material/MaterialProfile.cpp
#include "MaterialProfile.h"

void MaterialProfile::set_youngs_modulus(float modulus) {
    youngsModulus = modulus;
}

void MaterialProfile::set_poissons_ratio(float ratio) {
    poissonsRatio = ratio;
}

void MaterialProfile::set_density(float density) {
    densityValue = density;
}

void MaterialProfile::set_modal_damping_state(float dampingState) {
    modalDampingState = dampingState;
}
