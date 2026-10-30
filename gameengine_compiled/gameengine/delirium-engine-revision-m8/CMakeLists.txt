cmake_minimum_required(VERSION 3.16)
project(delirium_m8 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(delirium_m8_core STATIC
    src/body_topology.cpp
    src/material_registry.cpp
    src/modal_system.cpp
    src/basis_functions.cpp
    src/elastic_solver.cpp
    src/damage_accumulator.cpp
    src/welded_seam.cpp
    src/spatial_grid.cpp
    src/surfel_cache.cpp
    src/contact_detector.cpp
    src/thermal_simulator.cpp
    src/keyframe_animator.cpp
    src/animation_timeline.cpp
    src/serializer.cpp
    src/constraint_solver.cpp
    src/collision_response.cpp
    src/external_forces.cpp
    src/fracture_system.cpp
    src/multi_body_manager.cpp
    src/body_runtime.cpp
    src/frustum_culler.cpp
    src/shared_material_pool.cpp
    src/interaction_layer.cpp
)

target_include_directories(delirium_m8_core PUBLIC include)

add_executable(delirium_m8_tests tests/m8_tests.cpp)
target_link_libraries(delirium_m8_tests PRIVATE delirium_m8_core)
target_compile_options(delirium_m8_tests PRIVATE -Wall -Wextra -Werror -Wmisleading-indentation)