cmake_minimum_required(VERSION 3.20)
project(delirium_engine_revision_m0 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DELIRIUM_M0_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(OpenCL REQUIRED)

# nlohmann/json is NOT required. The engine's JSON handling (the local-LLM
# client) uses the in-tree strict reader in ai/ElementTreeParser.h, so the
# build has no external JSON dependency and works identically on a machine
# that does not have the system package.
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  message(STATUS "DeliRium: nlohmann_json found (not required)")
else()
  message(STATUS "DeliRium: nlohmann_json not found (not required -- in-tree JSON reader)")
endif()

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
# delirium_toroid_seam_probe - MEASURE the toroid's seam structure.
#
# WHY: the user proposed that two patch sets should NOT have to match edges
# tightly ("breathing space"), with a deliberately made "empty path" mapped
# to compensating anchors for the second set. Whether that is available
# depends on HOW the edges currently resolve -- by exact anchor index, by
# position (cross-chart), or not at all. That is a measurement, not an
# opinion, so it gets a probe.
#   cmake --build build_delirium --target delirium_toroid_seam_probe
#   ./build_delirium/delirium_toroid_seam_probe assets/torus.psf
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/toroid_seam_probe/main.cpp")
  add_executable(delirium_toroid_seam_probe
    examples/toroid_seam_probe/main.cpp
  )
  target_link_libraries(delirium_toroid_seam_probe PRIVATE delirium_engine_lib)
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

# ui_font_probe - renders the embedded 5x7 font atlas to a PPM for visual
# inspection. SDL2_ttf is unavailable on this machine, so the workspace UI
# depends on the embedded glyph table being correct -- a picture proves it.
#   cmake --build build_delirium --target ui_font_probe
#   ./build_delirium/ui_font_probe /tmp/font.ppm
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/ui_font_probe/main.cpp")
  add_executable(ui_font_probe examples/ui_font_probe/main.cpp)
  target_link_libraries(ui_font_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# llm_probe - proves the local-LLM client talks to a real local server.
# Standalone (no window): pings the server, lists the models, then does one
# streaming chat round-trip and prints the answer as it arrives.
#   cmake --build build_delirium --target llm_probe
#   ./build_delirium/llm_probe [--host H --port P --model M --prompt "..."]
#   ./build_delirium/llm_probe --list
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/llm_probe/main.cpp")
  add_executable(llm_probe examples/llm_probe/main.cpp)
  target_link_libraries(llm_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# delirium_workspace - the docked editor: scene tree / 3D viewport /
# inspector / AI console, wired to a local Ollama-compatible LLM.
#   cmake --build build_delirium --target delirium_workspace
#   ./build_delirium/delirium_workspace                  (from project root)
#   options: --host H --port P --model M --object NAME --no-probe
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/workspace/main.cpp")
  add_executable(delirium_workspace
    examples/workspace/main.cpp
  )
  target_link_libraries(delirium_workspace PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# ai_pipeline_probe - proves the WHOLE AI path, headless:
#   prompt -> LLMClient (socket/SSE) -> model reply -> JSON extraction ->
#   element-tree parse -> ElementSystem::BakeBody -> a real PSFBody.
# Unlike llm_probe (transport only) this exercises the geometry half, so a
# failure names the BROKEN LINK rather than "the AI button did nothing".
#   cmake --build build_delirium --target ai_pipeline_probe
#   ./build_delirium/ai_pipeline_probe [--prompt "..."] [--model M]
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/ai_pipeline_probe/main.cpp")
  add_executable(ai_pipeline_probe
    examples/ai_pipeline_probe/main.cpp
  )
  target_link_libraries(ai_pipeline_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# geo_tool_probe - DOES THE LOCAL MODEL ACTUALLY DEPLOY THE GEOMETRIC
# ANALYSIS TOOLS?
#   The registry's maths is unit-tested (tests/validate_geo_tools.cpp).
#   This probe asks the LIVE model a task that REQUIRES the tool, runs the
#   tool loop, and checks that (a) the model called the tool and (b) the
#   final answer carries the right numbers. Run it with and without the
#   tools -- the difference is the measurement.
#   cmake --build build_delirium --target geo_tool_probe
#   ./build_delirium/geo_tool_probe --model qwen2.5:7b-instruct-q8_0
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/geo_tool_probe/main.cpp")
  add_executable(geo_tool_probe
    examples/geo_tool_probe/main.cpp
  )
  target_link_libraries(geo_tool_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# doc_learner_probe - drives the DOCUMENT-driven reconstruction loop
# against a LIVE local model:
#   technical document -> LLMClient -> element tree -> validate -> BAKE
#   -> a real PSFBody, with the failure logged and the escalation ladder
#   (retry with the failure context, optionally against a LARGER model)
#   exercised.
#   cmake --build build_delirium --target doc_learner_probe
#   ./build_delirium/doc_learner_probe [--model M] [--escalate-model M2]
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/doc_learner_probe/main.cpp")
  add_executable(doc_learner_probe
    examples/doc_learner_probe/main.cpp
  )
  target_link_libraries(doc_learner_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# complex_scene - renders COMPLEX scenes: a multi-part closed assembly
# (the gearbox housing the document learner bakes) and a FIVE-BODY scene
# that exercises the multi-body compositing contract (body 0 clears, all
# bodies composite, only the last resolves). Each scene is MEASURED, so a
# scene that silently renders one object instead of five is caught rather
# than admired.
#
# It also carries the RENDER gates that CTest cannot otherwise see:
#   --scene sphere       the closed-sphere A/B (a closed body must present
#                        the same silhouette from both sides)
#   --scene paths        all five render paths, each with its own banner
#   --scene determinism  the same scene rendered twice must be BYTE-IDENTICAL
#                        (this is the gate for GPU races; the march path had
#                        one -- an unguarded store in build_grid -- that made
#                        1.09% of pixels differ between two runs)
#
#   cmake --build build_delirium --target complex_scene
#   ./build_delirium/complex_scene [--out DIR] [--scene all|five|gearbox|sphere|paths|determinism]
#
# The render scenes are wired into CTest below: they are the only evidence
# that the renderer actually produces pixels.
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/complex_scene/main.cpp")
  add_executable(complex_scene
    examples/complex_scene/main.cpp
  )
  target_link_libraries(complex_scene PRIVATE delirium_engine_lib)

  # The render gates. Each gets its own output directory under the build
  # tree so they cannot interfere with each other or with a developer's
  # own runs.
  set(DELIRIUM_COMPLEX_SCENE_OUT "${CMAKE_BINARY_DIR}/render_gates")
  foreach(scene sphere paths determinism)
    add_test(NAME render_${scene}
             COMMAND complex_scene --out "${DELIRIUM_COMPLEX_SCENE_OUT}/${scene}" --scene ${scene})
    set_tests_properties(render_${scene} PROPERTIES
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 600)
  endforeach()

  # The DERIVED-mesh render gate: the same closed sphere, baked two ways,
  # from the same tree, with the same cameras.
  #
  # WHY IT IS A SEPARATE TARGET: the derived path (G22) derives its cells
  # from the point set instead of placing them on a lattice. A derivation
  # that has never been RENDERED is a derivation whose visual quality is
  # unknown -- the record-level gates measure residual, connectivity,
  # tiling and coverage, none of which is a pixel.
  add_test(NAME render_derived
           COMMAND complex_scene --out "${DELIRIUM_COMPLEX_SCENE_OUT}/derived" --scene derived)
  set_tests_properties(render_derived PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    TIMEOUT 600)
endif()

# -----------------------------------------------------------------------
# sheet_probe - WHY THE DERIVED SEAM SITS AT A DIFFERENT HEIGHT THAN THE
# GRID'S.
#
# The render comparison showed the derived sphere's seam discontinuity 8
# rows higher than the grid's. This compares the two bodies' anchor z
# distributions -- the per-sheet z ranges and how many anchors straddle
# z=0 -- so the cause is named rather than guessed.
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/complex_scene/sheet_probe.cpp")
  add_executable(sheet_probe examples/complex_scene/sheet_probe.cpp)
  target_link_libraries(sheet_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# torus_probe - attributes a geometry loss to a specific layer (field ->
# anchors -> patches). Diagnostic: when a bounded substance bakes to a
# flat slab, this names the stage that dropped the relief instead of
# leaving it to guesswork.
#   cmake --build build_delirium --target torus_probe
#   ./build_delirium/torus_probe
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/ai_pipeline_probe/torus_probe.cpp")
  add_executable(torus_probe
    examples/ai_pipeline_probe/torus_probe.cpp
  )
  target_link_libraries(torus_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# torus_bake_probe - attributes the torus's SHREDDED SURFACE to a specific
# layer: degenerate (out-of-support) anchors, patches that reference them,
# the baker-vs-renderer corner CONVENTION, and analytic-vs-baked fidelity.
# Diagnostic, not a gate -- it prints the numbers that identify the defect.
#   cmake --build build_delirium --target torus_bake_probe
#   ./build_delirium/torus_bake_probe
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/torus_bake_probe/main.cpp")
  add_executable(torus_bake_probe
    examples/torus_bake_probe/main.cpp
  )
  target_link_libraries(torus_bake_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# camera_probe - is the yaw-dependent silhouette change in the CAMERA or in
# the RENDERER?  Projects the body's real anchor cloud AND the analytic
# dome through the same Camera::BuildMVP the renderer uses, per yaw.
# Diagnostic: prints the numbers that separate the two.
#   cmake --build build_delirium --target camera_probe
#   ./build_delirium/camera_probe
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/camera_probe/main.cpp")
  add_executable(camera_probe
    examples/camera_probe/main.cpp
  )
  target_link_libraries(camera_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# sphere_probe - shows why an ellipsoid with a plane child came out as a
# flat face, and proves the depth-weighted contribution fixes it.
#   cmake --build build_delirium --target sphere_probe
#   ./build_delirium/sphere_probe
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/ai_pipeline_probe/sphere_probe.cpp")
  add_executable(sphere_probe
    examples/ai_pipeline_probe/sphere_probe.cpp
  )
  target_link_libraries(sphere_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# closed_body_probe - MEASURES which origin/scale actually closes a body.
# The doc-learner system prompt teaches the model a rule for closing a
# sphere with two sheets; this probe checks that the rule it teaches is
# the rule the baker actually implements, instead of trusting the prose.
#   cmake --build build_delirium --target closed_body_probe
#   ./build_delirium/closed_body_probe
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/ai_pipeline_probe/closed_body_probe.cpp")
  add_executable(closed_body_probe
    examples/ai_pipeline_probe/closed_body_probe.cpp
  )
  target_link_libraries(closed_body_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# make_sphere_asset - regenerate assets/sphere.psf as a VALID binary.
#
# The asset was a legacy TEXT file (a parametric-surface description with
# `#` comments), so the binary loader rejected it and the gallery's scene N
# reported a misleading "MISSING" message. This tool bakes a CLOSED sphere
# with the engine's own baker -- so the bbox, the s_scale, the position
# quantisation and the corner-vanishing coefficients are all correct by
# construction -- and writes the v3 container. It READS THE FILE BACK
# before reporting success, so "wrote it" and "it loads" are the same
# claim.
#   cmake --build build_delirium --target make_sphere_asset
#   ./build_delirium/make_sphere_asset --out assets/sphere.psf
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/make_sphere_asset/main.cpp")
  add_executable(make_sphere_asset
    examples/make_sphere_asset/main.cpp
  )
  target_link_libraries(make_sphere_asset PRIVATE delirium_engine_lib)
endif()

file(GLOB DELIRIUM_VALIDATE_TESTS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/validate_*.cpp")
foreach(test_src ${DELIRIUM_VALIDATE_TESTS})
  get_filename_component(test_name ${test_src} NAME_WE)   # e.g. validate_eval
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE delirium_engine_lib)
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

# -----------------------------------------------------------------------
# validate_workspace_render - the RENDER gate for the editor workspace.
#
# It drives the REAL app and reads the REAL composited window pixels, so it
# can see things the C++ suite structurally cannot: that the docked panels
# paint, that a torus has a real hole, that the app closes itself on a
# complete frame.
#
# It needs a DISPLAY and a live local model, and it takes minutes (the AI
# path waits on the model). So it is a LABELLED gate: it is registered, but
# CTest skips it unless DELIRIUM_WORKSPACE_GATE=1 is set. That keeps a
# headless CI run green while the gate stays one command away:
#
#   DELIRIUM_WORKSPACE_GATE=1 ctest -R validate_workspace_render -V
#
# The build directory is passed explicitly so the gate finds the binaries
# wherever CMake put them.
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/validate_workspace_render.py")
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME validate_workspace_render
             COMMAND ${Python3_EXECUTABLE}
                     "${CMAKE_SOURCE_DIR}/tests/validate_workspace_render.py")
    set_tests_properties(validate_workspace_render PROPERTIES
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 900
      ENVIRONMENT "DELIRIUM_BUILD_DIR=${CMAKE_BINARY_DIR}"
      LABELS "display;model")
    # Skipped by default: it needs a display and a local model.
    set_property(TEST validate_workspace_render
                 PROPERTY SKIP_RETURN_CODE 77)
  endif()
endif()

# ---------------------------------------------------------------------------
# validate_layered_splat - the gate for the LAYERED DISTANCE-PLANE SPLAT.
#
# The user's method: surface samples are bucketed into distance shells, the
# impacts in a shell are AVERAGED, and the shells are stacked back-to-front.
# The property under test is about PIXELS -- does the averaging smooth the
# dither without collapsing coverage or blowing out to white -- so the C++
# suite structurally cannot see it (it renders nothing).
#
# It needs a DISPLAY (the gallery opens an OpenCL context), so it is
# LABELLED and skipped by default like the workspace gate:
#
#   DELIRIUM_LAYERED_GATE=1 ctest -R validate_layered_splat -V
#
# The build directory is passed explicitly so the gate finds the binaries
# wherever CMake put them.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/validate_layered_splat.py")
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME validate_layered_splat
             COMMAND ${Python3_EXECUTABLE}
                     "${CMAKE_SOURCE_DIR}/tests/validate_layered_splat.py")
    set_tests_properties(validate_layered_splat PROPERTIES
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 1200
      ENVIRONMENT "DELIRIUM_BUILD_DIR=${CMAKE_BINARY_DIR}"
      LABELS "display")
    # Skipped by default: it needs a display.
    set_property(TEST validate_layered_splat
                 PROPERTY SKIP_RETURN_CODE 77)
  endif()
endif()

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

# -----------------------------------------------------------------------
# make_cube_asset - bake the MODEL-GENERATED cube (the closed-body recipe
# the model applied by itself: upper "box" sheet + mirrored lower sheet)
# and write it as a valid binary .psf asset. The cube is the sharpest lens
# on the render pipelines: flat faces with constant normals turn any
# normal/shading artifact into crisp edges or missing faces.
#   cmake --build build_delirium --target make_cube_asset
#   ./build_delirium/make_cube_asset --out assets/cube.psf
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/make_cube_asset/main.cpp")
  add_executable(make_cube_asset
    examples/make_cube_asset/main.cpp
  )
  target_link_libraries(make_cube_asset PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# cube_diag - WHERE did the model-generated cube's side walls go?
# The anchors span a perfect cube, but the render shows two flat sheets.
# This measures the anchor distribution and the patch normals to name the
# stage that dropped the walls.
#   cmake --build build_delirium --target cube_diag
#   ./build_delirium/cube_diag assets/cube.psf
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/make_cube_asset/cube_diag.cpp")
  add_executable(cube_diag examples/make_cube_asset/cube_diag.cpp)
  target_link_libraries(cube_diag PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# domain_probe - does the baker's sampling ever visit the cube's wall band?
# The cube diag showed zero wall anchors; this names the cause: the
# substance's Domain() vs its own wall band.
#   cmake --build build_delirium --target domain_probe
#   ./build_delirium/domain_probe
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/make_cube_asset/domain_probe.cpp")
  add_executable(domain_probe examples/make_cube_asset/domain_probe.cpp)
  target_link_libraries(domain_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# bake_probe - instrument the cube bake: the substance yields 188 wall
# samples on the grid, but the baked body had SIDE-WALL 0. This probe
# bakes the cube tree and reports the distribution, naming the dropping
# stage.
#   cmake --build build_delirium --target bake_probe
#   ./build_delirium/bake_probe
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/make_cube_asset/bake_probe.cpp")
  add_executable(bake_probe examples/make_cube_asset/bake_probe.cpp)
  target_link_libraries(bake_probe PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# cube_diag2 - the cube's two sheets: does the mirrored one seal the box
# or poke out behind? Bakes the tree with and without the mirrored sheet
# and compares extents.
#   cmake --build build_delirium --target cube_diag2
#   ./build_delirium/cube_diag2
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/make_cube_asset/cube_diag2.cpp")
  add_executable(cube_diag2 examples/make_cube_asset/cube_diag2.cpp)
  target_link_libraries(cube_diag2 PRIVATE delirium_engine_lib)
endif()

# -----------------------------------------------------------------------
# seam_diag - measure the torus's equator serration at the record level.
# The cube work proved shading/normals/winding correct; the remaining
# artifact is the serrated teeth at the seam. This names the seam's
# geometry: rim radii distribution, straddling patches, duplicate anchors.
#   cmake --build build_delirium --target seam_diag
#   ./build_delirium/seam_diag
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/make_cube_asset/seam_diag.cpp")
  add_executable(seam_diag examples/make_cube_asset/seam_diag.cpp)
  target_link_libraries(seam_diag PRIVATE delirium_engine_lib)
endif()
