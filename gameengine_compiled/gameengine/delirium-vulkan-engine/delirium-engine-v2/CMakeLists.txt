cmake_minimum_required(VERSION 3.16)
project(delirium_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find dependencies ──────────────────────────────────────────────
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

# ── Core library (M8-aligned, 64-byte records) ────────────────────
add_library(delirium_core STATIC
    core/types.h
    core/body_topology.h  core/body_topology.cpp
    core/body_state.h
    core/material_registry.h  core/material_registry.cpp
    core/basis_functions.h  core/basis_functions.cpp
    core/modal_system.h  core/modal_system.cpp
    core/surfel_cache.h  core/surfel_cache.cpp
    core/body_runtime.h  core/body_runtime.cpp
    core/scene.h  core/scene.cpp
    core/engine.h  core/engine.cpp
)
target_include_directories(delirium_core PUBLIC ${CMAKE_SOURCE_DIR})

# ── OpenGL rendering library (ray marching + compute path) ────────
add_library(delirium_render STATIC
    core/opengl/context.hpp  core/opengl/context.cpp
    core/opengl/shader.hpp  core/opengl/shader.cpp
    core/opengl/buffer.hpp  core/opengl/buffer.cpp
    core/window/window.hpp  core/window/window.cpp
    core/window/input.hpp  core/window/input.cpp
    core/render/raymarch_renderer.hpp  core/render/raymarch_renderer.cpp
    core/render/compute_renderer.hpp  core/render/compute_renderer.cpp
)
target_link_libraries(delirium_render PUBLIC delirium_core SDL2::SDL2 OpenGL::GL GLEW::GLEW)
target_include_directories(delirium_render PUBLIC ${CMAKE_SOURCE_DIR})

# ── Surfel rendering library (point sprite path) ───────────────────
add_library(delirium_surfel STATIC
    render/opengl_context.h  render/opengl_context.cpp
    render/shader_program.h  render/shader_program.cpp
    render/surfel_renderer.h  render/surfel_renderer.cpp
    render/camera.h  render/camera.cpp
)
target_link_libraries(delirium_surfel PUBLIC delirium_core SDL2::SDL2 OpenGL::GL GLEW::GLEW)
target_include_directories(delirium_surfel PUBLIC ${CMAKE_SOURCE_DIR})

# ── AI module ───────────────────────────────────────────────────────
add_library(delirium_ai STATIC
    ai/domain_classifier.h  ai/domain_classifier.cpp
    ai/material_science.h  ai/material_science.cpp
    ai/organic_generator.h  ai/organic_generator.cpp
    ai/inorganic_generator.h  ai/inorganic_generator.cpp
    ai/generator.h  ai/generator.cpp
)
target_link_libraries(delirium_ai PUBLIC delirium_core)
target_include_directories(delirium_ai PUBLIC ${CMAKE_SOURCE_DIR})

# ── Visual demo: ray marching (full-screen quad, GPU shader) ───────
add_executable(demo_raymarch demo_scene.cpp demo_scene.hpp demo_raymarch_main.cpp)
target_link_libraries(demo_raymarch PRIVATE delirium_render delirium_ai)
set_target_properties(demo_raymarch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
target_compile_definitions(demo_raymarch PRIVATE SHADER_DIR="${SHADER_DIR}")

# ── Visual demo: surfel point sprites (surfel cache → VBO → GL_POINTS) ─
add_executable(demo_surfel demo_visual.cpp)
target_link_libraries(demo_surfel PRIVATE delirium_surfel delirium_ai)
target_compile_definitions(demo_surfel PRIVATE SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders")
set_target_properties(demo_surfel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ── Interactive workshop (AI + raymarch + surfel) ─────────────────
add_executable(demo_workshop demo_workshop.cpp)
target_link_libraries(demo_workshop PRIVATE delirium_render delirium_surfel delirium_ai)
target_compile_definitions(demo_workshop PRIVATE SHADER_DIR="${SHADER_DIR}")
set_target_properties(demo_workshop PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ── CPU demos (no OpenGL) ──────────────────────────────────────────
add_executable(demo_basis demo_basis.cpp)
target_link_libraries(demo_basis PRIVATE delirium_core)

add_executable(demo_ai demo_ai.cpp)
target_link_libraries(demo_ai PRIVATE delirium_ai)

add_executable(demo_seam demo_seam.cpp)
target_link_libraries(demo_seam PRIVATE delirium_core)

add_executable(demo_physics demo_physics.cpp)
target_link_libraries(demo_physics PRIVATE delirium_core)

add_executable(demo_render demo_render.cpp)
target_link_libraries(demo_render PRIVATE delirium_core)

# ── Deformation test (live physics + raymarch render) ────────────
add_executable(demo_deform demo_deform.cpp)
target_link_libraries(demo_deform PRIVATE delirium_render delirium_core)
target_compile_definitions(demo_deform PRIVATE SHADER_DIR="${SHADER_DIR}")
set_target_properties(demo_deform PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ── OpenCL demo (v2 body → bake → proven surfel_march.cl pipeline) ──
find_package(OpenCL REQUIRED)
add_executable(demo_opencl demo_opencl.cpp)
target_link_libraries(demo_opencl PRIVATE delirium_core delirium_ai OpenCL::OpenCL SDL2::SDL2)
set_target_properties(demo_opencl PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# Copy proven kernel to build dir so it's found at runtime
add_custom_command(TARGET demo_opencl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/kernels
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/../delirium-engine/kernels/surfel_march.cl
        ${CMAKE_BINARY_DIR}/kernels/surfel_march.cl
)