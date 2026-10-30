cmake_minimum_required(VERSION 3.20)
project(delirium_engine_revision_m2 VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DELIRIUM_M2_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(DELIRIUM_M2_ENABLE_OPENCL "Enable OpenCL adapter (requires OpenCL SDK)" OFF)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(delirium_m2_core
    src/types.cpp
    src/body_topology.cpp
    src/articulation.cpp
    src/pose_controller.cpp
    src/surface_evaluator.cpp
    src/surfel_cache.cpp
    src/opencl_adapter.cpp
    src/body_runtime.cpp)

target_include_directories(delirium_m2_core PUBLIC include)
target_compile_features(delirium_m2_core PUBLIC cxx_std_17)

# Requires GCC 11+ or Clang 13+ for full C++17 support
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
        message(WARNING "GCC < 11 may have incomplete C++17 support — consider GCC 11+")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
        message(WARNING "Clang < 13 may have incomplete C++17 support — consider Clang 13+")
    endif()
endif()

if(MSVC)
    target_compile_options(delirium_m2_core PRIVATE /W4)
else()
    target_compile_options(delirium_m2_core PRIVATE -Wall -Wextra -Wpedantic)
    if(DELIRIUM_M2_WARNINGS_AS_ERRORS)
        target_compile_options(delirium_m2_core PRIVATE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# Optional OpenCL
# ---------------------------------------------------------------------------
if(DELIRIUM_M2_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        target_link_libraries(delirium_m2_core PRIVATE OpenCL::OpenCL)
        target_compile_definitions(delirium_m2_core PRIVATE DELIRIUM_M2_HAS_OPENCL)
        message(STATUS "OpenCL found — adapter will link against OpenCL SDK")
    else()
        message(WARNING "OpenCL requested but not found — using stub adapter")
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_executable(delirium_m2_tests tests/m2_tests.cpp)
target_link_libraries(delirium_m2_tests PRIVATE delirium_m2_core)

enable_testing()
add_test(NAME m2_core_contracts COMMAND delirium_m2_tests)