cmake_minimum_required(VERSION 3.20)
project(delirium_engine_revision_m0 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DELIRIUM_M0_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(OpenCL REQUIRED)

# ---------------------------------------------------------------------------
# M0 — OpenCL device probe (must stay green; default `all` target)
# ---------------------------------------------------------------------------
add_executable(delirium_m0_probe src/opencl_probe.cpp)
target_link_libraries(delirium_m0_probe PRIVATE OpenCL::OpenCL)
target_compile_features(delirium_m0_probe PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(delirium_m0_probe PRIVATE /W4)
else()
  target_compile_options(delirium_m0_probe PRIVATE -Wall -Wextra -Wpedantic)
  if(DELIRIUM_M0_WARNINGS_AS_ERRORS)
    target_compile_options(delirium_m0_probe PRIVATE -Werror)
  endif()
endif()

enable_testing()
add_test(NAME m0_opencl_probe COMMAND delirium_m0_probe --quiet)

# ---------------------------------------------------------------------------
# M1 — torus deform (EXCLUDE_FROM_ALL: does not break M0 `cmake --build`)
# Build with: cmake --build build_delirium --target delirium_m1_torus
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/01_torus_deform/main.cpp")
  file(GLOB_RECURSE DELIRIUM_M1_ENGINE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/engine/src/*.cpp"
  )
  add_executable(delirium_m1_torus
    examples/01_torus_deform/main.cpp
    ${DELIRIUM_M1_ENGINE_SOURCES}
  )
  set_target_properties(delirium_m1_torus PROPERTIES EXCLUDE_FROM_ALL TRUE)
  target_include_directories(delirium_m1_torus PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/engine/include
    ${CMAKE_SOURCE_DIR}/engine/src
  )
  target_compile_features(delirium_m1_torus PRIVATE cxx_std_17)
  target_link_libraries(delirium_m1_torus PRIVATE OpenCL::OpenCL)
endif()

# --- delirium_g3_frame (Godot-like + OpenCL) ---
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/g3_frame/main.cpp")
  add_executable(delirium_g3_frame
    engine/src/core/Node.cpp
    engine/src/core/SceneTree.cpp
    engine/src/servers/RenderServer.cpp
    engine/src/servers/PhysicsServer.cpp
    engine/src/scene/SplatView.cpp
    examples/g3_frame/main.cpp
  )
  target_include_directories(delirium_g3_frame PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
  )
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND)
    target_link_libraries(delirium_g3_frame PRIVATE OpenCL::OpenCL)
  else()
    target_link_libraries(delirium_g3_frame PRIVATE OpenCL)
  endif()
endif()

# --- Optional SDL2 for E2 window (headless build still works without it) ---
find_package(SDL2 QUIET)
if(TARGET delirium_m1_torus)
  if(SDL2_FOUND)
    target_compile_definitions(delirium_m1_torus PRIVATE DELIRIUM_HAS_SDL=1)
    if(TARGET SDL2::SDL2)
      target_link_libraries(delirium_m1_torus PRIVATE SDL2::SDL2)
    else()
      target_include_directories(delirium_m1_torus PRIVATE ${SDL2_INCLUDE_DIRS})
      target_link_libraries(delirium_m1_torus PRIVATE ${SDL2_LIBRARIES})
    endif()
    message(STATUS "DeliRium: SDL2 found -- DELIRIUM_HAS_SDL=1")
  else()
    message(STATUS "DeliRium: SDL2 not found -- headless Engine only")
  endif()
endif()

# ---------------------------------------------------------------------------
# validate_* CPU test suite (17 milestone gates; 14 present under tests/).
# Each is a standalone main() that loads assets/torus.psf and self-checks a
# subsystem. Built against a shared static lib of the engine sources so they
# stay in sync with the M1 binary. Wired into CTest; working dir = project
# root so the default "assets/torus.psf" path resolves.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE DELIRIUM_TEST_ENGINE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/engine/src/*.cpp"
)
add_library(delirium_engine_lib STATIC ${DELIRIUM_TEST_ENGINE_SOURCES})
target_include_directories(delirium_engine_lib PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/engine/include
  ${CMAKE_SOURCE_DIR}/engine/src
)
target_compile_features(delirium_engine_lib PUBLIC cxx_std_17)
target_link_libraries(delirium_engine_lib PUBLIC OpenCL::OpenCL)
if(SDL2_FOUND)
  target_compile_definitions(delirium_engine_lib PUBLIC DELIRIUM_HAS_SDL=1)
  if(TARGET SDL2::SDL2)
    target_link_libraries(delirium_engine_lib PUBLIC SDL2::SDL2)
  else()
    target_include_directories(delirium_engine_lib PUBLIC ${SDL2_INCLUDE_DIRS})
    target_link_libraries(delirium_engine_lib PUBLIC ${SDL2_LIBRARIES})
  endif()
endif()

# ---------------------------------------------------------------------------
# delirium_visual_stress - interactive GPU stress test (poke/dent/refine/
# coarsen/fracture/morph, live render + screenshots). Links the shared lib.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/03_visual_stress/main.cpp")
  add_executable(delirium_visual_stress
    examples/03_visual_stress/main.cpp
  )
  target_link_libraries(delirium_visual_stress PRIVATE delirium_engine_lib)
endif()

# ---------------------------------------------------------------------------
# delirium_visual_gallery - HEADLESS render gallery (no window/display).
# One PPM per subsystem + render path (20 scenes) into visual_gallery/.
#   cmake --build build_delirium --target delirium_visual_gallery
#   ./build_delirium/delirium_visual_gallery            (from project root)
#   options: --out DIR  --size WxH  --only A,B,C  --list
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/visual_gallery/main.cpp")
  add_executable(delirium_visual_gallery
    examples/visual_gallery/main.cpp
  )
  target_link_libraries(delirium_visual_gallery PRIVATE delirium_engine_lib)
endif()

# ---------------------------------------------------------------------------
# delirium_playground - LIVE interactive playground (SDL window).
# Drive every subsystem in real time: poke/dent/refine/coarsen/fracture/
# morph/restitch/weld + live render-path switching (1/2/3) + auto-orbit.
#   cmake --build build_delirium --target delirium_playground
#   ./build_delirium/delirium_playground                (from project root)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/playground/main.cpp")
  add_executable(delirium_playground
    examples/playground/main.cpp
  )
  target_link_libraries(delirium_playground PRIVATE delirium_engine_lib)
endif()

# ---------------------------------------------------------------------------
# delirium_realtime_deform - HEADLESS real-time benchmark WHILE deforming.
# Sustained deform workload + per-frame GPU render, times deform/render/total.
#   cmake --build build_delirium --target delirium_realtime_deform
#   ./build_delirium/delirium_realtime_deform            (from project root)
#   options: --workload modal|amr|dent|mixed  --path march|raymarch|splat
#            --frames N  --warmup N  --size WxH  --dt T  --shot FRAME FILE
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/realtime_deform/main.cpp")
  add_executable(delirium_realtime_deform
    examples/realtime_deform/main.cpp
  )
  target_link_libraries(delirium_realtime_deform PRIVATE delirium_engine_lib)
endif()

# ---------------------------------------------------------------------------
# delirium_recon_window - LIVE window: generic element-system reconstruction.
# Four objects reconstructed IN MEMORY (orb / bump / wave / mound) from the
# element-system baker, rendered live with the full playground control set.
#   cmake --build build_delirium --target delirium_recon_window
#   ./build_delirium/delirium_recon_window               (from project root)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/recon_window/main.cpp")
  add_executable(delirium_recon_window
    examples/recon_window/main.cpp
  )
  target_link_libraries(delirium_recon_window PRIVATE delirium_engine_lib)
  # 10s recorder writes PNG frames (zlib deflate) + encodes MP4 via ffmpeg
  find_package(ZLIB)
  if(ZLIB_FOUND)
    target_link_libraries(delirium_recon_window PRIVATE ZLIB::ZLIB)
  else()
    target_link_libraries(delirium_recon_window PRIVATE z)
  endif()
endif()

# delirium_torus_live - LIVE toroidal deformation demo (SDL window).
# Loads assets/torus.psf + assets/torus.modes and drives a continuous
# traveling wave (impulse injected at a rotating tube point every 10
# frames) so the march path is exercised under sustained deformation.
# Full playground control set + W toggles the auto-wave.
#   cmake --build build_delirium --target delirium_torus_live
#   ./build_delirium/delirium_torus_live                (from project root)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/torus_live/main.cpp")
  add_executable(delirium_torus_live
    examples/torus_live/main.cpp
  )
  target_link_libraries(delirium_torus_live PRIVATE delirium_engine_lib)
endif()

# delirium_recon_diag - HEADLESS black-screen diagnostic (no SDL/window).
# Projects each generated body to screen, raycasts it, and counts
# non-background pixels per render path (march/raymarch/splat), with the
# torus as a known-good control.
#   cmake --build build_delirium --target delirium_recon_diag
#   ./build_delirium/delirium_recon_diag               (from project root)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/recon_diag/main.cpp")
  add_executable(delirium_recon_diag
    examples/recon_diag/main.cpp
  )
  target_link_libraries(delirium_recon_diag PRIVATE delirium_engine_lib)
endif()

file(GLOB DELIRIUM_VALIDATE_TESTS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/validate_*.cpp")
foreach(test_src ${DELIRIUM_VALIDATE_TESTS})
  get_filename_component(test_name ${test_src} NAME_WE)   # e.g. validate_eval
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE delirium_engine_lib)
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

# ---------------------------------------------------------------------------
# perf_engine - performance report (NOT a pass/fail gate; always exits 0).
# Times load/eval/queries/modal/AMR/refit/seams/GPU-flush. Build + run:
#   cmake --build build_delirium --target perf_engine
#   ./build_delirium/perf_engine            # (run from project root)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/perf_engine.cpp")
  add_executable(perf_engine tests/perf_engine.cpp)
  target_link_libraries(perf_engine PRIVATE delirium_engine_lib)
  # wire into ctest too, but it self-reports and exits 0, so it never fails
  add_test(NAME perf_engine COMMAND perf_engine)
  set_tests_properties(perf_engine PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
