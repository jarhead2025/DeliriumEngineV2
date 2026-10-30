cmake_minimum_required(VERSION 3.16)
project(delirium_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

# Physics engine (M8)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../delirium-engine-revision-m8
                 ${CMAKE_BINARY_DIR}/m8 EXCLUDE_FROM_ALL)

# Core engine library
add_library(delirium_core STATIC
    core/opengl/context.cpp
    core/opengl/shader.cpp
    core/opengl/buffer.cpp
    core/window/window.cpp
    core/window/input.cpp
    core/camera/camera.cpp
    core/obj_loader/obj_loader.cpp
    core/obj_loader/obj_model_builder.cpp
    core/basis_fitter/basis_fitter.cpp
)
target_include_directories(delirium_core PUBLIC core)
target_link_libraries(delirium_core PUBLIC OpenGL::GL SDL2::SDL2 delirium_m8_core)
target_compile_options(delirium_core PRIVATE -Wall -Wextra)

# Scene layer
add_library(delirium_scene STATIC
    scene/surfel_renderer.cpp
    scene/raymarch_renderer.cpp
    scene/demo_scene.cpp
)
target_include_directories(delirium_scene PUBLIC scene)
target_link_libraries(delirium_scene PUBLIC delirium_core delirium_m8_core)
target_compile_definitions(delirium_scene PRIVATE
    SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")

# Main executable
add_executable(delirium_engine main.cpp)
target_link_libraries(delirium_engine PRIVATE delirium_scene delirium_m8_core)
target_compile_options(delirium_engine PRIVATE -Wall -Wextra)