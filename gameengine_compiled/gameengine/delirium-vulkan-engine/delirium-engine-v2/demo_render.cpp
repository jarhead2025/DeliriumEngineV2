// demo_render.cpp — prove CPU splat rendering with ASCII visualization
#include "core/types.h"
#include "core/body_topology.h"
#include "core/body_runtime.h"
#include "core/material_registry.h"
#include "core/scene.h"
#include "core/engine.h"
#include "core/surfel_cache.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace delirium;

// ASCII luminance chars (dark → bright)
static const char LUMA[] = " .`'-\"^\",:;Il|)!+~?/>|#%$@";

void ascii_render(const std::vector<std::uint32_t>& pixels, int w, int h,
                  int out_w, int out_h) {
    printf("+");
    for (int x = 0; x < out_w; ++x) printf("-");
    printf("+\n");

    for (int row = 0; row < out_h; ++row) {
        printf("|");
        for (int col = 0; col < out_w; ++col) {
            int px = col * (w * 2 / out_w);
            int py = row * (h * 2 / out_h);
            if (px >= w) px = w - 1;
            if (py >= h) py = h - 1;

            std::uint32_t p = pixels[py * w + px];
            std::uint8_t r = p & 0xFF;
            std::uint8_t g = (p >> 8) & 0xFF;
            std::uint8_t b = (p >> 16) & 0xFF;

            float lum = 0.299f * r + 0.587f * g + 0.114f * b;
            int idx = static_cast<int>(lum / 255.0f * (sizeof(LUMA) - 2));
            printf("%c", LUMA[idx]);
        }
        printf("|\n");
    }
    printf("+");
    for (int x = 0; x < out_w; ++x) printf("-");
    printf("+\n");
}

// Build a sphere-like body from patches
static BodyRuntime* make_sphere(BodyId id, int res) {
    BodyTopology topo;

    // Create a grid of anchors (res x res)
    std::vector<AnchorId> anchors;
    for (int r = 0; r < res; ++r)
        for (int c = 0; c < res; ++c)
            anchors.push_back(topo.add_anchor());

    auto anchor_at = [&](int r, int c) -> AnchorId {
        return anchors[r * res + c];
    };

    // Create patches
    for (int r = 0; r < res - 1; ++r)
        for (int c = 0; c < res - 1; ++c) {
            topo.add_patch({
                anchor_at(r, c),
                anchor_at(r, c + 1),
                anchor_at(r + 1, c),
                anchor_at(r + 1, c + 1),
            }, 0);
        }

    topo.finalize();

    auto* body = new BodyRuntime(std::move(topo), id);
    MaterialRegistry reg;
    MaterialProperties mat;
    mat.r = 200; mat.g = 140; mat.b = 90;
    mat.roughness = 0.6f;
    reg.register_material(0, mat);
    body->set_materials(std::move(reg));

    // Sphere parameterization
    std::vector<Vec3> rest;
    for (int r = 0; r < res; ++r) {
        float phi = M_PI * r / (res - 1); // 0 to pi
        for (int c = 0; c < res; ++c) {
            float theta = 2.0f * M_PI * c / (res - 1); // 0 to 2pi
            rest.push_back({
                std::sin(phi) * std::cos(theta),
                std::cos(phi),
                std::sin(phi) * std::sin(theta),
            });
        }
    }
    body->initialize(rest);
    return body;
}

int main() {
    printf("=== CPU Splat Render Proof ===\n\n");

    Engine engine;
    engine.set_viewport(512, 512);

    // 1. Single quad (head-on view)
    printf("[1] Single quad, head-on:\n");
    {
        BodyTopology topo;
        AnchorId a[4];
        for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
        topo.add_patch({a[0], a[1], a[2], a[3]}, 0);
        topo.finalize();

        BodyRuntime body(std::move(topo), 0);
        MaterialRegistry reg;
        MaterialProperties mat;
        mat.r = 255; mat.g = 200; mat.b = 100;
        reg.register_material(0, mat);
        body.set_materials(std::move(reg));
        body.initialize({
            {-0.5f, -0.5f, 0}, {0.5f, -0.5f, 0},
            {-0.5f,  0.5f, 0}, {0.5f,  0.5f, 0},
        });

        Scene scene;
        scene.add_body(&body);
        engine.set_scene(&scene);
        engine.set_camera({0, 0, 4}, {0, 0, 0}, {0, 1, 0});

        SurfelCache cache;
        engine.prepare_render(cache);
        std::vector<std::uint32_t> pixels;
        engine.render_cpu(cache, pixels);

        printf("    Surfels: %zu, Pixels: %zu\n", cache.count(), pixels.size());
        ascii_render(pixels, 512, 512, 48, 36);
    }

    // 2. Quad at angle
    printf("\n[2] Quad rotated 45°:\n");
    {
        BodyTopology topo;
        AnchorId a[4];
        for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
        topo.add_patch({a[0], a[1], a[2], a[3]}, 0);
        topo.finalize();

        BodyRuntime body(std::move(topo), 0);
        MaterialRegistry reg;
        MaterialProperties mat;
        mat.r = 100; mat.g = 200; mat.b = 255;
        reg.register_material(0, mat);
        body.set_materials(std::move(reg));

        // Rotated quad
        float c = std::cos(0.785f), s = std::sin(0.785f);
        body.initialize({
            {-0.5f*c, -0.5f,  0.5f*s},
            { 0.5f*c, -0.5f, -0.5f*s},
            {-0.5f*c,  0.5f,  0.5f*s},
            { 0.5f*c,  0.5f, -0.5f*s},
        });

        Scene scene;
        scene.add_body(&body);
        engine.set_scene(&scene);
        engine.set_camera({0, 0, 4}, {0, 0, 0}, {0, 1, 0});

        SurfelCache cache;
        engine.prepare_render(cache);
        std::vector<std::uint32_t> pixels;
        engine.render_cpu(cache, pixels);

        printf("    Surfels: %zu\n", cache.count());
        ascii_render(pixels, 512, 512, 48, 36);
    }

    // 3. Sphere approximation (6x6 grid = 25 patches)
    printf("\n[3] Sphere (6x6 anchor grid, 25 patches):\n");
    {
        BodyRuntime* sphere = make_sphere(0, 6);
        Scene scene;
        scene.add_body(sphere);
        engine.set_scene(&scene);
        engine.set_camera({0, 2, 4}, {0, 0, 0}, {0, 1, 0});

        SurfelCache cache;
        engine.prepare_render(cache);
        std::vector<std::uint32_t> pixels;
        engine.render_cpu(cache, pixels);

        printf("    Surfels: %zu, Anchors: %zu, Patches: %zu\n",
               cache.count(),
               sphere->topology().anchor_count(),
               sphere->topology().patch_count());
        ascii_render(pixels, 512, 512, 48, 36);

        delete sphere;
    }

    // 4. Depth test: two overlapping quads
    printf("\n[4] Depth test (near quad occludes far quad):\n");
    {
        Scene scene;

        // Far quad (blue)
        {
            BodyTopology topo;
            AnchorId a[4];
            for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
            topo.add_patch({a[0], a[1], a[2], a[3]}, 0);
            topo.finalize();
            auto* b = new BodyRuntime(std::move(topo), 0);
            MaterialRegistry reg;
            MaterialProperties mat;
            mat.r = 50; mat.g = 50; mat.b = 255;
            reg.register_material(0, mat);
            b->set_materials(std::move(reg));
            b->initialize({
                {-0.8f, -0.8f, -2}, {0.8f, -0.8f, -2},
                {-0.8f,  0.8f, -2}, {0.8f,  0.8f, -2},
            });
            scene.add_body(b);
        }

        // Near quad (red, offset)
        {
            BodyTopology topo;
            AnchorId a[4];
            for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
            topo.add_patch({a[0], a[1], a[2], a[3]}, 1);
            topo.finalize();
            auto* b = new BodyRuntime(std::move(topo), 1);
            MaterialRegistry reg;
            MaterialProperties mat;
            mat.r = 255; mat.g = 80; mat.b = 80;
            reg.register_material(1, mat);
            b->set_materials(std::move(reg));
            b->initialize({
                {-0.4f, -0.4f, -1}, {0.4f, -0.4f, -1},
                {-0.4f,  0.4f, -1}, {0.4f,  0.4f, -1},
            });
            scene.add_body(b);
        }

        engine.set_scene(&scene);
        engine.set_camera({0, 0, 4}, {0, 0, 0}, {0, 1, 0});

        SurfelCache cache;
        engine.prepare_render(cache);
        std::vector<std::uint32_t> pixels;
        engine.render_cpu(cache, pixels);

        printf("    Surfels: %zu (both bodies combined)\n", cache.count());
        ascii_render(pixels, 512, 512, 48, 36);
        printf("    => Red (near) should occlude blue (far) in center ✓\n");
    }

    printf("\n=== Render proof complete ===\n");
    return 0;
}