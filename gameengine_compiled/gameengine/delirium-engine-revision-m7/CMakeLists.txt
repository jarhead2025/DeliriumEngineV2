cmake_minimum_required(VERSION 3.16)
project(delirium_m7 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(delirium_m7_core STATIC
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
    src/body_runtime.cpp
)

target_include_directories(delirium_m7_core PUBLIC include)

add_executable(delirium_m7_tests tests/m7_tests.cpp)
target_link_libraries(delirium_m7_tests PRIVATE delirium_m7_core)
target_compile_options(delirium_m7_tests PRIVATE -Wall -Wextra -Werror -Wmisleading-indentation)