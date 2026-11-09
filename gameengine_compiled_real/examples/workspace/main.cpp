// =============================================================================
// delirium_workspace - the DeliRium editor: a docked, Godot-like UI around
// the live 3D viewport, wired to a local LLM.
//
//   +--------------------------------------------------------------+
//   | status bar                                                   |
//   +----------------+--------------------------------+-----------+
//   | Scene Tree     |        3D VIEWPORT              | Inspector |
//   +----------------+--------------------------------+-----------+
//   | AI Console: prompt -> local model -> element tree -> body    |
//   +--------------------------------------------------------------+
//
// WHAT THIS IS: the engine's own UI (core/Workspace.h) driven by a thin
// app. The app owns the window, the scene and the startup objects; the
// workspace owns the panels, the camera and the AI loop.
//
// THE AI PATH: the console sends the prompt to a local Ollama-compatible
// server, the reply is parsed as an element tree (ai/ElementTreeParser.h),
// baked by the element-system baker, and dropped into the scene as a new
// object. The MODEL writes the geometry; the engine owns the math.
//
//   Build:  cmake --build build_delirium --target delirium_workspace
//   Run:    ./build_delirium/delirium_workspace        (from project root)
//
// Options:
//   --host H --port P --model M    point the AI console at a local server
//   --object NAME                 load assets/NAME.psf at startup
//   --semantic NAME                build a procedural body from NAME
//                                 (orb / torus / bump / field) instead of
//                                 the default orb -- proves the generator's
//                                 vocabulary without waiting on the model
//   --no-probe                    skip the local-server reachability probe
//   --auto-quit SECONDS           close the app by itself after N seconds
//                                 (for scripted/automated checks: the app
//                                 ends on a COMPLETE frame, with no
//                                 external kill and no half-written state)
//   --auto-frames N               close the app by itself after N RENDERED
//                                 FRAMES -- the DETERMINISTIC quit. The
//                                 render is not frame-stable, so a pixel
//                                 measurement taken at a clock-defined frame
//                                 is not reproducible. Use this, not
//                                 --auto-quit, when a gate asserts on pixels.
//   --shot FILE.png               capture the window to FILE.png before
//                                 quitting (proves the UI actually painted)
//   --auto-prompt TEXT            type TEXT into the AI console and press
//                                 Generate automatically once the window
//                                 is up. This drives the SAME path a
//                                 click takes (SubmitPrompt -> AskModel ->
//                                 parse -> bake -> AddFromTree), so a run
//                                 with this flag proves the in-app AI
//                                 loop, not just the transport.
//
// Controls:
//   LMB (viewport) .... apply the active tool
//   LMB drag .......... orbit the camera
//   MMB drag .......... pan
//   wheel ............. dolly
//   1..6 .............. select the tool (poke/dent/refine/coarsen/fracture/move)
//   Move tool + drag .. slide the selected object
//   Esc ............... quit
// =============================================================================
#include "delirium/core/Engine.h"
#include "delirium/core/GameObject.h"
#include "delirium/core/Scene.h"
#include "delirium/core/Workspace.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

using namespace DeliRium;

int main(int argc, char** argv) {
    // --- CLI ---------------------------------------------------------------
    std::string host = "127.0.0.1";
    int    port  = 11434;
    std::string model = "qwen2.5:7b-instruct-q8_0";
    std::string object;
    bool probe = true;
    double auto_quit = 0.0;
    int auto_frames = 0;
    std::string shot;
    std::string auto_prompt;
    std::string semantic = "orb";
    int render_path = -1;   // -1 = the workspace's default
    // Camera placement, so a capture can be taken from a KNOWN angle.
    // -1 = "not specified" (the workspace's own framing decides).
    float cam_yaw = -999.f, cam_pitch = -999.f, cam_dist = -1.f;
    // --- document mode (G15) ----------------------------------------------
    std::string document;        // the technical document to interpret
    bool        doc_mode = false;
    std::string doc_domain = "machine";
    std::string escalate_model;  // the larger model rung
    std::string doc_corpus;      // where the corpus is saved/loaded
    bool        auto_document = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string(def);
        };
        if (a == "--host")       host = next("127.0.0.1");
        else if (a == "--port")  port = std::atoi(next("11434").c_str());
        else if (a == "--model") model = next("qwen2.5:7b-instruct-q8_0");
        else if (a == "--object") object = next("torus");
        else if (a == "--no-probe") probe = false;
        else if (a == "--auto-quit") auto_quit = std::atof(next("0").c_str());
        else if (a == "--auto-frames") auto_frames = std::atoi(next("0").c_str());
        else if (a == "--shot") shot = next("");
        else if (a == "--auto-prompt") auto_prompt = next("");
        else if (a == "--semantic") semantic = next("orb");
        // --document PATH: switch the console into DOCUMENT mode and load
        // the technical document at PATH. The document pipeline runs the
        // corpus, the closed-body rule and the escalation ladder -- a
        // different contract from the semantic prompt.
        else if (a == "--document") { document = next(""); doc_mode = true; }
        // --doc-domain D: the retrieval prior's domain ("machine",
        // "biological", ...). A same-domain worked example outranks a
        // cross-domain one.
        else if (a == "--doc-domain") doc_domain = next("machine");
        // --escalate-model M: the LARGER model consulted when the local one
        // fails a document. Empty = no escalation rung.
        else if (a == "--escalate-model") escalate_model = next("");
        // --doc-corpus PATH: where the retrieval corpus is saved/loaded.
        // It PERSISTS across documents -- that is what makes the second
        // document better than the first.
        else if (a == "--doc-corpus") doc_corpus = next("");
        // --auto-document: interpret the loaded document automatically once
        // the model is online, then quit. This is the scripted path.
        else if (a == "--auto-document") auto_document = true;
        // --render-path N: 0 march, 1 raymarch, 2 splat classic,
        // 3 splat layered, 4 splat area (the pixel-footprint pyramid).
        else if (a == "--render-path") render_path = std::atoi(next("-1").c_str());
        // --cam-yaw / --cam-pitch / --cam-dist: pin the orbit camera so a
        // capture is taken from a KNOWN angle. A view-dependent defect (a
        // normal that points the wrong way on one side of the object) is
        // invisible unless the angle is an INPUT, not an uncontrolled
        // variable that drifts between runs.
        else if (a == "--cam-yaw")   cam_yaw   = std::atof(next("0").c_str());
        else if (a == "--cam-pitch") cam_pitch = std::atof(next("0").c_str());
        else if (a == "--cam-dist")  cam_dist  = std::atof(next("0").c_str());
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        }
    }
    if (auto_quit > 0.0) {
        // The app closes ITSELF, on a complete frame. This is how the
        // automated check runs it: no external kill, no half-written state.
        Engine::Instance().SetAutoQuitSeconds(auto_quit);
    }
    if (auto_frames > 0) {
        // The DETERMINISTIC quit: close after N rendered FRAMES rather than
        // N seconds. See Engine::SetAutoQuitFrames.
        Engine::Instance().SetAutoQuitFrames(auto_frames);
    }
    if (!shot.empty()) Engine::Instance().SetScreenshotOnExit(shot);

    // --- engine + scene ----------------------------------------------------
    Engine& engine = Engine::Instance();
    if (!engine.Initialize(1440, 900, "DeliRium Workspace")) {
        std::fprintf(stderr, "engine init failed\n");
        return 1;
    }
    Scene* scene = engine.CreateScene("workspace");

    // Two starter objects so the scene tree is not empty on launch:
    //   - a procedural body baked from an element tree (the same path the
    //     AI console uses -- so the AI result lands in a scene that already
    //     demonstrates the mechanism)
    //   - optionally a .psf loaded from disk
    {
        auto tree = ProceduralElementGenerator::Build(semantic);
        if (tree) {
            std::string why;
            // Same resolution the AI path uses: the starter orb must look
            // like the objects the console creates, not like a different
            // (coarser) kind of object.
            auto body = ElementSystem::BakeBody(*tree, 48, 1.0f, why);
            if (body) {
                GameObject* o = scene->Create("orb", std::move(body));
                if (o) o->GetTransform().position = { -0.7f, 0.f, 0.f };
            } else {
                std::fprintf(stderr, "starter bake failed: %s\n", why.c_str());
            }
        }
    }
    if (!object.empty()) {
        const std::string path = "assets/" + object + ".psf";
        auto body = PSFBody::FromFile(path.c_str());
        if (body) {
            GameObject* o = scene->Create(object, std::move(body));
            if (o) o->GetTransform().position = { 0.7f, 0.f, 0.f };
            std::printf("loaded %s\n", path.c_str());
        } else {
            std::fprintf(stderr, "could not load %s\n", path.c_str());
        }
    }

    // --- the workspace -----------------------------------------------------
    Workspace ws(engine);
    LLM::Config cfg = ws.Llm().GetConfig();
    cfg.host = host;
    cfg.port = port;
    cfg.model = model;
    ws.SetLlmConfig(cfg);
    ws.Attach();
    // Pin the camera AFTER Attach(): the workspace frames the scene during
    // Attach (computing the distance from the objects' world bounds), so an
    // override applied before that would be overwritten.
    if (cam_yaw > -900.f)   ws.SetOrbitYaw(cam_yaw);
    if (cam_pitch > -900.f) ws.SetOrbitPitch(cam_pitch);
    if (cam_dist > 0.f)     ws.SetOrbitDist(cam_dist);
    if (render_path >= 0) {
        // The render path is a RUNTIME value the renderer reads every
        // frame, so this is instant -- the same mechanism the viewport's
        // path buttons use. The renderer announces the change itself.
        ws.SetRenderPath(render_path);
    }

    // --- document mode (G15) ----------------------------------------------
    //
    // The corpus is LOADED before the document is interpreted, so the FIRST
    // document already benefits from every document interpreted before it.
    // That ordering is the whole point of the corpus -- a corpus that is
    // saved after the interpretation is a log, not memory.
    if (doc_mode || !document.empty()) {
        ws.SetDocumentMode(true);
        ws.SetDocumentDomain(doc_domain);
        if (!escalate_model.empty()) ws.SetEscalateModel(escalate_model);
        if (!doc_corpus.empty()) {
            ws.SetDocCorpusPath(doc_corpus);
            ws.LoadDocCorpus();
        }
        if (!document.empty()) {
            std::printf("document: %s (domain=%s)\n", document.c_str(),
                        doc_domain.c_str());
            ws.LoadDocument(document);
        }
    }
    if (!probe) {
        // The probe is what turns the status bar from "not probed" into
        // "online"/"offline"; skipping it is only useful for a headless
        // smoke test.
        ws.Llm().SetConfig(cfg);
    }
    std::printf("workspace up: model %s at %s:%d\n", model.c_str(), host.c_str(), port);

    if (!auto_prompt.empty()) {
        // Drive the REAL console path, one frame after the window exists:
        //   SubmitPrompt -> AskModel -> (stream) -> HandleModelReply
        //                -> ParseElementTree -> BakeBody -> AddFromTree
        // That is exactly what pressing Generate does, so a run with this
        // flag is evidence the in-app AI loop works, not just the wire.
        const std::string p = auto_prompt;
        ws.PostNextFrame([&ws, p] {
            std::printf("[auto-prompt] submitting: %s\n", p.c_str());
            if (!ws.SubmitPrompt(p))
                std::printf("[auto-prompt] submit failed\n");
            else
                std::printf("[auto-prompt] request sent\n");
        });
    }

    if (auto_document && !document.empty()) {
        // Drive the REAL document path.
        //
        // It must WAIT for the model to come online: the document path
        // refuses to run against an offline model (there is no point
        // sending a request that cannot be delivered), and the probe is
        // ASYNCHRONOUS -- it runs on its own thread and lands in a later
        // frame's PollProbe().
        //
        // So the driver is a STATE MACHINE, not a loop: it is armed here and
        // it fires on a LATER frame. A loop inside this callback would spin
        // on a condition that only a LATER frame can satisfy -- the probe
        // result is set by PollProbe(), which runs in OnFrame(). It would
        // deadlock on the very thing it waits for.
        ws.SetAutoDocumentRunner([&ws] {
            // Returns true once the interpretation has been attempted.
            return ws.InterpretLoadedDocument();
        });
    }

    engine.Run();
    engine.Shutdown();
    std::printf("workspace exited cleanly\n");
    return 0;
}
