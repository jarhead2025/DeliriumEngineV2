cmake_minimum_required(VERSION 3.20)
project(delirium_engine_revision_m1 VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DELIRIUM_M1_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(delirium_m1_core
    src/body_topology.cpp
    src/body_runtime.cpp
    src/articulation.cpp)
target_include_directories(delirium_m1_core PUBLIC include)
target_compile_features(delirium_m1_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(delirium_m1_core PRIVATE /W4)
else()
    target_compile_options(delirium_m1_core PRIVATE -Wall -Wextra -Wpedantic)
    if(DELIRIUM_M1_WARNINGS_AS_ERRORS)
        target_compile_options(delirium_m1_core PRIVATE -Werror)
    endif()
endif()

add_executable(delirium_m1_tests tests/m1_tests.cpp)
target_link_libraries(delirium_m1_tests PRIVATE delirium_m1_core)

enable_testing()
add_test(NAME m1_core_contracts COMMAND delirium_m1_tests)
