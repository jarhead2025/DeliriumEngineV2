cmake_minimum_required(VERSION 3.20)
project(delirium_engine_v2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Dependencies
# ============================================================================
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

# ============================================================================
# Layer 0: Math core (header-only)
# ============================================================================
add_library(delirium_math INTERFACE)
target_include_directories(delirium_math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ============================================================================
# Layer 1: Geometry (patch, body, seam)
# ============================================================================
add_library(delirium_geometry STATIC
    src/delirium/geometry/patch/patch.cpp
)
target_include_directories(delirium_geometry PUBLIC src)
target_link_libraries(delirium_geometry PUBLIC delirium_math)

# ============================================================================
# Layer 2: Rendering (surfel, lighting, OpenGL)
# ============================================================================
add_library(delirium_render STATIC
    src/delirium/render/surfel/surfel_cache.cpp
    src/delirium/render/renderer.cpp
)
target_include_directories(delirium_render PUBLIC src)
target_link_libraries(delirium_render PUBLIC delirium_geometry OpenGL::GL SDL2::SDL2)

# ============================================================================
# Layer 3: Physics (modal, collision, material)
# ============================================================================
add_library(delirium_physics STATIC
    src/delirium/physics/modal/modal_solver.cpp
    src/delirium/physics/collision/collision_detector.cpp
)
target_include_directories(delirium_physics PUBLIC src)
target_link_libraries(delirium_physics PUBLIC delirium_geometry)

# ============================================================================
# Integration: Core engine
# ============================================================================
add_library(delirium_core STATIC
    src/delirium/core/engine.cpp
    src/delirium/core/scene.cpp
)
target_include_directories(delirium_core PUBLIC src)
target_link_libraries(delirium_core PUBLIC
    delirium_geometry
    delirium_render
    delirium_physics
)

# ============================================================================
# Executable: Demo application
# ============================================================================
add_executable(delirium_demo main.cpp)
target_link_libraries(delirium_demo PRIVATE delirium_core)
target_compile_options(delirium_demo PRIVATE -Wall -Wextra)

# ============================================================================
# Tests
# ============================================================================
enable_testing()

add_executable(test_math tests/math/test_basis.cpp)
target_link_libraries(test_math PRIVATE delirium_math)
add_test(NAME math_basis COMMAND test_math)

add_executable(test_geometry tests/geometry/test_patch.cpp)
target_link_libraries(test_geometry PRIVATE delirium_geometry)
add_test(NAME geometry_patch COMMAND test_geometry)

# ============================================================================
# Install
# ============================================================================
install(TARGETS delirium_demo DESTINATION bin)