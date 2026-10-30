cmake_minimum_required(VERSION 3.20)
project(delirium_engine_revision_m6 VERSION 0.6.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(delirium_m6_core
    src/body_topology.cpp
    src/material_registry.cpp
    src/modal_system.cpp
    src/elastic_solver.cpp
    src/damage_accumulator.cpp
    src/basis_functions.cpp
    src/welded_seam.cpp
    src/spatial_grid.cpp
    src/surfel_cache.cpp
    src/contact_detector.cpp
    src/thermal_simulator.cpp
    src/keyframe_animator.cpp
    src/animation_timeline.cpp
    src/serializer.cpp
    src/body_runtime.cpp)
target_include_directories(delirium_m6_core PUBLIC include)
target_compile_features(delirium_m6_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(delirium_m6_core PRIVATE /W4)
else()
    target_compile_options(delirium_m6_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(delirium_m6_tests tests/m6_tests.cpp)
target_link_libraries(delirium_m6_tests PRIVATE delirium_m6_core)

enable_testing()
add_test(NAME m6_core_contracts COMMAND delirium_m6_tests)