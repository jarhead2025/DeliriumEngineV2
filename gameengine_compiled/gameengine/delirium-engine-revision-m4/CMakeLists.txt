cmake_minimum_required(VERSION 3.20)
project(delirium_engine_revision_m4 VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DELIRIUM_M4_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(delirium_m4_core
    src/body_topology.cpp
    src/material_registry.cpp
    src/modal_system.cpp
    src/elastic_solver.cpp
    src/damage_accumulator.cpp
    src/basis_functions.cpp
    src/welded_seam.cpp
    src/spatial_grid.cpp
    src/surfel_cache.cpp
    src/body_runtime.cpp)
target_include_directories(delirium_m4_core PUBLIC include)
target_compile_features(delirium_m4_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(delirium_m4_core PRIVATE /W4)
else()
    target_compile_options(delirium_m4_core PRIVATE -Wall -Wextra -Wpedantic)
    if(DELIRIUM_M4_WARNINGS_AS_ERRORS)
        target_compile_options(delirium_m4_core PRIVATE -Werror)
    endif()
endif()

add_executable(delirium_m4_tests tests/m4_tests.cpp)
target_link_libraries(delirium_m4_tests PRIVATE delirium_m4_core)

enable_testing()
add_test(NAME m4_core_contracts COMMAND delirium_m4_tests)