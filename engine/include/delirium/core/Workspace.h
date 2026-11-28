// =============================================================================
// Workspace.h - The docked editor UI for DeliRium.
//
// LAYOUT (Godot-like, fixed):
//
//   +--------------------------------------------------------------+
//   | Status bar: engine state, fps, selection, LLM state          |
//   +----------------+--------------------------------+-----------+
//   |                |                                |           |
//   |  Scene Tree    |        3D VIEWPORT             | Inspector |
//   |  (left dock)   |   (the GPU image, centred)     | (right)   |
//   |                |                                |           |
//   +----------------+--------------------------------+-----------+
//   | AI CONSOLE: prompt input, model reply, generate button      |
//   +--------------------------------------------------------------+
//
// WHY IMMEDIATE MODE (see ui/UI.h): the layout is FIXED, so a retained
// widget tree would be machinery without a purpose. The workspace computes
// a rect per panel, hands it to the toolkit, and reads back the hit.
//
// THE TWO PASSES: the engine calls the FRAME hook before the scene
// updates (that is where the UI UPDATE pass runs -- hit-testing, widget
// callbacks, viewport resize) and the OVERLAY hook during Present() (the
// PAINT pass). Splitting them means a button that creates an object is
// reflected in the SAME frame's paint.
//
// CAMERA: the workspace owns the camera (Engine::CameraControl::External).
// Left-drag orbits, middle-drag pans, wheel dollies. The engine never
// touches the camera in that mode, so the UI and the view never fight.
//
// THE AI PATH: the console sends a prompt to a local Ollama-compatible
// server (llm/LLMClient.h), parses the reply as an element tree
// (ai/ElementTreeParser.h), bakes it (ai/ElementSystem.h) and drops the
// resulting body into the scene. The model writes the GEOMETRY; the engine
// owns the MATH.
//
// MOVING OBJECTS: a GameObject's placement lives in its Transform, OUTSIDE
// the frozen 64-byte records, and the renderer applies it as a GPU model
// matrix. The inspector exposes position/rotation/scale as drag fields, so
// an object can be dragged around without ever re-quantizing an anchor.
// =============================================================================
#pragma once
#include <functional>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "delirium/core/Engine.h"
#include "delirium/core/GameObject.h"
#include "delirium/core/Scene.h"
#include "delirium/core/Transform.h"
#include "delirium/ui/UI.h"
#include "delirium/ui/Font.h"
#include "delirium/ui/InspectorPanel.h"
#include "delirium/script/ScriptRuntime.h"
#include "delirium/script/ClingInterpreter.h"
#include "delirium/ai/DocumentIngestor.h"
#include "delirium/ai/SpecToGraph.h"
#include "delirium/ai/ReconstructionValidator.h"
#include "delirium/llm/LLMClient.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/DocLearner.h"
#include "delirium/ai/ReconstructionModel.h"
#include "delirium/physics/PhysicsWorld.h"

namespace DeliRium {

// The sculpt tools the viewport applies on click.
enum class Tool {
    Poke = 0,   // impulse at the hit point
    Dent,        // radial dent inward
    Refine,      // split patches near the hit
    Coarsen,     // merge patches near the hit
    Fracture,    // cut the body along the camera-parallel plane
    Move,        // drag the selected object along the camera-right axis
    // NAVIGATION -- not a sculpt tool. A dedicated mode in which the mouse
    // drives the CAMERA instead of the geometry, so the viewport can be
    // inspected without ever risking a poke on the object.
    //
    // WHY A MODE AT ALL: the user reported that dragging to look around
    // kept colliding with the tools -- a drag that started over a panel
    // picked up a slider, and a drag that started in the viewport applied
    // a poke at the end. Separating "look" from "act" needs a mode, and a
    // mode the user can enter deliberately is more predictable than one
    // inferred from which modifier is held.
    Navigate,
};

// A named viewpoint. The editor keeps several and can jump between them;
// each remembers its own orbit state.
struct Viewpoint {
    std::string name;
    float yaw = 0.6f, pitch = 0.35f, dist = 4.0f;
    Vec3  target{0.f, 0.f, 0.f};
};

class Workspace {
public:
    explicit Workspace(Engine& eng);
    ~Workspace();

    // Install the engine hooks. Call AFTER the scene and its objects exist.
    void Attach();

    // --- layout -----------------------------------------------------------
    // The panel rectangles, in WINDOW pixels. Public so a test (and the
    // status bar) can reason about the layout without a window.
    struct Layout {
        int win_w = 0, win_h = 0;
        int status_h = 0;
        int console_h = 0;
        int left_w = 0, right_w = 0;
        int toolbar_h = 0;   // U5: reserved strip at the top of the viewport
        UI::Rect status, toolbar, left, viewport, right, console;
    };
    const Layout& GetLayout() const { return layout_; }
    // Recompute the layout for a window size. Pure function of the size:
    // the panels shrink proportionally on a small window.
    void ComputeLayout(int w, int h);

    // --- THE UI-DECIDES-FIRST DECISION -------------------------------------
    // Would a click at (x, y) reach the VIEWPORT (and therefore apply a
    // tool, start an orbit, or drag an object)?
    //
    // This is the decision the user's bug lived in, extracted so it can be
    // TESTED directly. The rule:
    //   - the gizmo wins (it is not a UI widget -- it IS a viewport action);
    //   - a point outside the viewport rect is not a viewport action;
    //   - a point inside a UI WIDGET's rect is consumed by the UI, even if
    //     the widget pokes into the viewport rect (the panels touch the
    //     viewport's edges).
    //
    // The widget rects come from the PREVIOUS frame's pass (immediate mode
    // is one frame stale by design -- the same discipline ImGui's
    // IsItemHovered uses), so a widget that only just appeared cannot veto
    // this frame's click. That is why the map is double-buffered.
    bool ClickReachesViewport(int x, int y) const;

    // Pull the orbit camera back until the whole scene fits in the view.
    // Computed from the world bounds of every object (each object's
    // Transform included, since placement lives outside the frozen
    // records). Called on attach; call it again after adding objects.
    void FrameScene();

    // --- state ------------------------------------------------------------
    size_t SelectedIndex() const { return selected_; }
    GameObject* Selected() const;
    void Select(size_t index);

    // --- the AI path ------------------------------------------------------
    // Send `prompt` to the local model. Returns false if a request is
    // already in flight.
    bool AskModel(const std::string& prompt);
    // Drive the console the way the Generate button does: put the text in
    // the field, clear the field, and send. This is the SAME path a click
    // takes -- an automated check that calls this is testing the real
    // button, not a side door into the parser.
    bool SubmitPrompt(const std::string& prompt);
    // Run `fn` once, on the NEXT frame. Used by a scripted run to drive the
    // real UI after the window exists (e.g. type into the console and press
    // Generate). Deferred rather than immediate because the first frame is
    // what establishes the layout the widgets hit-test against.
    void PostNextFrame(std::function<void()> fn);
    // True once a posted action has run.
    bool RanNextFrameAction() const { return ran_next_frame_; }
    // The console log (newest last). The AI panel shows the tail.
    //
    // It returns a COPY, not a reference: the document worker thread (G15)
    // appends to the log while the main thread reads it, and a reference
    // would outlive the lock that protects it. A copy is cheap (the log is
    // capped at 400 lines) and it cannot dangle.
    std::vector<std::string> Log() const {
        std::lock_guard<std::mutex> guard(log_mutex_);
        return log_;
    }
    void Logf(const char* fmt, ...);

    // Build a body from an element tree and add it to the scene. Returns
    // the new object (nullptr on failure, with the reason logged).
    GameObject* AddFromTree(const std::unique_ptr<Element>& tree, const std::string& name);

    // The LLM client (exposed so the app can configure the host/port).
    LLM::LLMClient& Llm() { return llm_; }
    void SetLlmConfig(const LLM::Config& cfg) { llm_.SetConfig(cfg); }
    // True once the reachability probe has run and the server answered.
    bool ModelOnline() const { return model_online_; }
    const std::string& ModelStatus() const { return model_status_; }

    // FPS + frame counter, for the status bar.
    int  Fps() const { return fps_; }
    float LastFrameMs() const { return frame_ms_; }

    // The active sculpt tool.
    Tool ActiveTool() const { return tool_; }
    void SetTool(Tool t) { tool_ = t; }

    // --- viewpoints --------------------------------------------------------
    // Save the current orbit under `name` (or update it if it exists) and
    // make it active.
    void SaveViewpoint(const std::string& name);
    // Jump to a saved viewpoint. Returns false if the index is out of range.
    bool GotoViewpoint(int index);
    int  ViewpointCount() const { return int(viewpoints_.size()); }
    int  ActiveViewpoint() const { return active_view_; }
    const std::vector<Viewpoint>& Viewpoints() const { return viewpoints_; }

    // --- the navigation gizmo ----------------------------------------------
    // Where the gizmo is drawn, in window pixels. Public so a test can
    // reason about it without a window.
    const UI::Rect& GizmoRect() const { return gizmo_rect_; }

    // The UI context. Public so a test can register a widget rect and
    // then assert that the workspace's input decision honours it -- the
    // UI-DECIDES-FIRST contract spans both sides, and each side needs to
    // be drivable from a test.
    UI::Context& GetUI() { return ui_; }
    const UI::Context& GetUI() const { return ui_; }

    // ======================================================================
    // DOCUMENT MODE (G15)
    //
    // The document loop, made usable from the editor: load a technical
    // document, interpret it through the document pipeline (corpus +
    // closed-body rule + error log + escalation), and drop the resulting
    // body into the scene.
    // ======================================================================

    // Switch the console between the SEMANTIC contract ("make a torus") and
    // the DOCUMENT contract (a path to a technical document).
    void SetDocumentMode(bool on) { doc_mode_ = on; }
    bool DocumentMode() const { return doc_mode_; }

    // Read a document from disk into the document slot. Returns false with
    // the reason logged (a missing file, or an empty one).
    bool LoadDocument(const std::string& path);
    // The document's text (empty if none is loaded).
    const std::string& DocumentText() const { return doc_text_; }
    const std::string& DocumentPath() const { return doc_path_; }
    // The domain used for the retrieval prior.
    void SetDocumentDomain(const std::string& d) { doc_domain_ = d; }
    const std::string& DocumentDomain() const { return doc_domain_; }

    // Run the document pipeline. This is the SAME path the console's
    // "interpret" button takes.
    //
    // It is ASYNCHRONOUS. The model call takes 20-280 s, and the learner
    // drives the model itself (InterpretDocument owns the request and the
    // escalation ladder), so it cannot be expressed in the workspace's
    // asynchronous LLM state machine without restructuring the learner.
    //
    // It runs on a WORKER THREAD: the document path is driven from
    // OnFrame(), and a blocking call there would freeze the window for
    // minutes -- which reads as a crashed app, not as a slow model. The
    // worker does the model call, the parse and the validation; the MAIN
    // thread does the baking and the scene insertion, because the scene is
    // not thread-safe.
    //
    // Returns true if the request was STARTED (not if it succeeded).
    bool InterpretLoadedDocument();

    // Poll the document worker. Called from the frame hook; it applies the
    // result on the main thread when the worker finishes.
    void PollDocument();

    // The corpus: the accumulated successes and failures.
    const AI::DocErrorLog& DocCorpus() const { return doc_log_; }
    // Save/load the corpus so it survives a restart.
    bool SaveDocCorpus();
    bool LoadDocCorpus();
    // The corpus's size, for the status line.
    size_t DocCorpusSize() const { return doc_log_.Size(); }
    size_t DocCorpusFailures() const { return doc_log_.FailureCount(); }
    size_t DocCorpusSuccesses() const { return doc_log_.SuccessCount(); }
    // The escalation model (empty = none). Set it to a larger model's name
    // to enable the ladder's second rung.
    void SetEscalateModel(const std::string& m) { doc_escalate_model_ = m; }
    const std::string& EscalateModel() const { return doc_escalate_model_; }
    // Where the corpus is saved/loaded. It must be settable, because the
    // corpus is a FILE and the path a user wants is not the default.
    void SetDocCorpusPath(const std::string& p) { doc_corpus_path_ = p; }
    const std::string& DocCorpusPath() const { return doc_corpus_path_; }
    // True while a document request is in flight.
    bool DocBusy() const { return doc_busy_; }

    // Arm an AUTOMATIC document interpretation.
    //
    // The callback fires on a LATER frame, once the model has been probed
    // as online -- the probe is asynchronous, so the first frame is too
    // early. The callback is invoked at most once, and the runner is
    // disarmed afterwards whether it succeeded or failed.
    //
    // WHY IT IS A CALLBACK AND NOT A LOOP: the probe result is set by
    // PollProbe(), which runs in OnFrame(). A loop that waited for it from
    // inside a frame callback would deadlock on the very thing it waits
    // for -- the condition it needs can only be satisfied by the next
    // frame, and the next frame cannot start until the current one returns.
    void SetAutoDocumentRunner(std::function<bool()> fn) {
        auto_doc_runner_ = std::move(fn);
        auto_doc_armed_ = true;
    }
    // True if an automatic document interpretation has been attempted (or
    // was never armed). A test uses this to know the run is over.
    bool AutoDocumentDone() const { return !auto_doc_armed_; }

private:
    Engine&  engine_;
    UI::Context ui_;
    BitmapFont font_;
    LLM::LLMClient llm_;
    // M4 WS4 physics-first editing panel: displays editable physical
    // properties grouped by module with units + assumptions, and a
    // conservation ledger check before committing edits.
    InspectorPanel inspector_panel_;
    // M5 WS5 script runtime + Cling interpreter: execute .drs scripts and
    // inline expressions against the PhysicsWorld/Scene state.
    ScriptRuntime script_runtime_;
    ClingInterpreter cling_interp_;
    // M6 WS2 document reconstruction: parse PDF/CAD/mesh specs → DocumentSpec
    // → StructuralGraph constraints → validate reconstructed body.
    DocumentIngestor doc_ingestor_;
    SpecToGraph doc_spec_to_graph_;
    ReconstructionValidator doc_validator_;

    Layout layout_;
    size_t selected_ = static_cast<size_t>(-1);
    int    fps_ = 0;
    float  frame_ms_ = 0.0f;
    // The editable display name of the selected object. GameObject::name is
    // const, so the rename lives here.
    std::string edit_name_;
    // The name typed into the "view name" field when saving a viewpoint.
    std::string view_name_;

    // --- camera (the workspace owns it) -----------------------------------
    float orbit_yaw_ = 0.6f;
    float orbit_pitch_ = 0.35f;
    float orbit_dist_ = 4.0f;
    Vec3  orbit_target_{0.f, 0.f, 0.f};
    bool  orbiting_ = false;
    int   orbit_last_x_ = 0, orbit_last_y_ = 0;
    bool  panning_ = false;
    int   pan_last_x_ = 0, pan_last_y_ = 0;
    bool  dragging_object_ = false;   // the Move tool is dragging
    int   drag_last_x_ = 0;
    // U6: drag-to-orbit in sculpt modes. A sculpt click applies the tool; a
    // drag (movement past the threshold) orbits instead. The flag records
    // that a sculpt drag is active; the origin records where it started so
    // the orbit delta is measured from the button-down point.
    bool  tool_dragging_ = false;
    int   tool_drag_origin_x_ = 0, tool_drag_origin_y_ = 0;
    bool  tool_drag_applied_ = false;   // the tool impulse already fired

    // --- viewpoints --------------------------------------------------------
    // Named camera states the user can jump between. Each remembers its own
    // orbit, so switching views does not lose the framing.
    std::vector<Viewpoint> viewpoints_;
    int   active_view_ = 0;
    // The gizmo: a small sphere-like widget drawn in the viewport corner
    // that the user drags to rotate the camera. It is SEPARATE from the
    // viewport's own pixels, so dragging it can never hit the geometry.
    UI::Rect gizmo_rect_;
    bool  gizmo_drag_ = false;
    int   gizmo_last_x_ = 0, gizmo_last_y_ = 0;

    // --- the multi-view capture sweep --------------------------------------
    // The orbit sweep is a scripted capture loop: pin viewpoint k, wait
    // settle_frames for the render to catch up (the renderer's surfel bake
    // and grid are rebuilt for the new angle), queue the capture, advance.
    // settle_frames > 1 exists because a capture taken on the SAME frame as
    // the camera change reads a frame rendered from the OLD camera.
    bool        sweep_armed_ = false;
    // The deferred arm: the sweep starts when the next AI object lands.
    bool        sweep_deferred_ = false;
    // One-time startup log of the viewport rect (U2): external probes need
    // the origin, not just the size, to place clicks precisely.
    bool        layout_logged_ = false;
    bool        sweep_vary_pitch_ = true;
    int         sweep_count_ = 0;
    int         sweep_next_ = 0;          // next viewpoint index to capture
    int         sweep_settle_ = 10;       // frames to wait after pinning
    int         sweep_settle_left_ = 0;
    bool        vary_pitch_ = true;
    std::string sweep_prefix_;
    int         sweep_captured_ = 0;
    // One step of the sweep per frame (see ArmOrbitSweep).
    void SweepStep();

    // --- keyboard camera movement -----------------------------------------
    // WASD / arrows move the camera in its own frame; Q/E rise and fall.
    // The keys are read as HELD state, so the camera moves continuously
    // while they are down.
    bool  cam_key_fwd_ = false, cam_key_back_ = false;
    bool  cam_key_left_ = false, cam_key_right_ = false;
    bool  cam_key_up_ = false, cam_key_down_ = false;
    bool  cam_key_roll_ = false;

    // --- AI panel ---------------------------------------------------------
    std::string prompt_;
    std::string model_host_ = "127.0.0.1";
    int         model_port_ = 11434;
    std::string model_name_ = "qwen2.5:7b-instruct-q8_0";
    bool        model_online_ = false;
    bool        model_probed_ = false;
    std::string model_status_ = "not probed";
    std::vector<std::string> log_;
    // Guards the log: the document worker thread (G15) appends to it while
    // the main thread reads it for the console panel.
    mutable std::mutex log_mutex_;
    int         console_scroll_ = 0;
    int         tree_scroll_ = 0;
    bool        busy_ = false;

    // --- DOCUMENT MODE (G15) ----------------------------------------------
    //
    // The console has two contracts, and mixing them is why the document
    // loop was not usable:
    //
    //   SEMANTIC  "make a torus"  -> one prompt -> an element tree. The
    //                                prompt IS the specification.
    //   DOCUMENT  a path to a file -> the file IS the specification, and
    //                                the pipeline runs the DOCUMENT path:
    //                                the retrieval corpus, the closed-body
    //                                rule, the error log, and the escalation
    //                                ladder.
    //
    // The two need DIFFERENT prompts, DIFFERENT validation (the document
    // path rejects a single sheet) and DIFFERENT bookkeeping (the corpus).
    // Running a document through the semantic path would skip every one of
    // those -- and would silently accept a one-face body, which is exactly
    // the failure the document path exists to prevent.
    bool        doc_mode_ = false;
    // The document's path, and the text read from it.
    std::string doc_path_;
    std::string doc_text_;
    // The document's domain ("machine", "biological", ...). It selects the
    // retrieval prior: a same-domain worked example outranks a
    // cross-domain one.
    std::string doc_domain_ = "machine";
    // The corpus. It PERSISTS across documents -- that is what makes the
    // second document better than the first.
    AI::DocErrorLog doc_log_;
    // Where the corpus is saved/loaded, so it survives a restart.
    std::string doc_corpus_path_ = "/tmp/delirium_doc_corpus.txt";
    // The escalation rung: the larger model consulted when the local one
    // fails. Empty = no escalation.
    std::string doc_escalate_model_;
    // True while a DOCUMENT request is in flight. The document path drives
    // the model itself (through InterpretDocument), so the workspace's own
    // LLM state machine is not used for it.
    bool        doc_busy_ = false;
    // The automatic document runner (G15): armed by the app, fired on a
    // LATER frame once the model has been probed as online.
    std::function<bool()> auto_doc_runner_;
    bool        auto_doc_armed_ = false;
    // The document WORKER (G15).
    //
    // The model call takes minutes, and it is driven from OnFrame() -- so
    // it must not block the frame. The worker thread does the model call,
    // the parse and the validation; the main thread applies the result
    // (the bake and the scene insertion), because the scene is not
    // thread-safe.
    std::thread             doc_thread_;
    // Whether there is a RESULT waiting to be applied on the main thread.
    //
    // It is ATOMIC with release/acquire ordering because the worker writes
    // the result fields (ok, reason, tree) and THEN sets this flag, while
    // the main thread reads the flag and THEN reads those fields. The
    // ordering is what makes the fields visible.
    //
    // WHY A SEPARATE FLAG: the obvious design -- "done" means "apply the
    // result" -- cannot distinguish "the worker finished" from "no worker
    // ever ran", because both start as TRUE. That ambiguity made
    // PollDocument() report a spurious "document: FAILED" (with an empty
    // reason) and save an empty corpus on EVERY launch -- including
    // launches that never asked for a document.
    std::atomic<bool>       doc_has_result_{false};
    bool                    doc_thread_ok_ = false;
    std::string             doc_thread_reason_;
    // `Element` lives in the DeliRium namespace (not DeliRium::AI) -- the
    // AI namespace holds the PARSER and the LEARNER, and the element tree
    // itself is the engine's data model.
    std::unique_ptr<Element> doc_thread_tree_;
    // The corpus is TOUCHED by the worker (it records the attempt), and
    // READ by the main thread. The worker owns it while it runs; the main
    // thread only reads it after the worker has finished.

    // --- viewport picking -------------------------------------------------
    bool hover_viewport_ = false;

    // --- ML reconstruction backend ----------------------------------------
    // The reconstruction model holds the RMW1 weights once loaded. It is
    // kept in the workspace (not the engine) because inference is a UI
    // action triggered by the ML control strip, not a per-frame operation.
    ReconstructionModel recon_model_;
    std::string recon_weights_path_;   // last loaded weights path (for status)

    // Where the inspector content ended in the right dock (so the physics
    // panel can draw below it within the same dock).
    int inspector_bottom_ = 0;
    // Where the physics panel ended (so grammar/assembly can draw below it).
    int physics_bottom_ = 0;
    // Where the grammar/assembly panel ended (so profiler can draw below it).
    int grammar_bottom_ = 0;
    // Where the physics-controls panel ended (so profiler can draw below it).
    int physics_controls_bottom_ = 0;

    // --- grammar / assembly state -----------------------------------------
    // Bonding distance threshold for the grammar compiler (world units).
    float grammar_bond_dist_ = 0.05f;
    // SE(3) alignment tolerance for bond validation.
    float grammar_align_eps_ = 0.01f;
    // Mass conservation tolerance for bond validation.
    float grammar_mass_tol_ = 1e-3f;
    // Assembly config: max bonds to commit.
    int assembly_max_bonds_ = 64;
    // Assembly config: minimum bond strength to commit.
    float assembly_min_strength_ = 0.3f;
    // Whether closed loops are allowed in assembly.
    bool assembly_allow_loops_ = false;
    // Last assembly result (committed bonds count).
    int assembly_last_committed_ = 0;
    int assembly_last_rejected_ = 0;
    // Whether the last assembly run produced a graph.
    bool assembly_has_graph_ = false;

    // --- physics modules state (physics-controls panel) -------------------
    // These mirror PhysicsWorld toggles so the panel can read/write them.
    // The panel pushes the state into PhysicsWorld each frame it changes.
    bool phys_gravity_on_ = true;       // gravity enabled
    Vec3 phys_gravity_ = { 0.f, 0.f, -9.81f };
    bool phys_body_body_coll_ = true;   // body-body collision enabled
    bool phys_thermal_on_ = false;      // thermal coupling enabled
    bool phys_charge_on_ = false;       // charge coupling enabled
    bool phys_electrostatic_on_ = false;
    bool phys_magnetostatic_on_ = false;
    bool phys_aero_on_ = false;
    bool phys_hydro_on_ = false;
    bool phys_gpu_physics_ = false;     // GPU collision dispatch
    bool phys_gpu_nbody_ = false;       // GPU N-body pairwise forces
    bool phys_gpu_fem_ = false;         // GPU FEM continuum solve
    bool phys_gpu_reflow_ = false;      // GPU anchor-reflow
    bool phys_gpu_modal_ = false;       // GPU modal writeback
    int  phys_weld_budget_ = 4;         // weld passes per frame per object
    float phys_aero_density_ = 1.225f;  // kg/m³ (air at sea level)
    float phys_aero_cd_ = 0.47f;        // drag coefficient (sphere)
    float phys_hydro_density_ = 1000.f; // kg/m³ (water)
    // True if the physics-controls panel has pushed its state this frame.
    bool phys_controls_dirty_ = true;

    // --- recon_window feature integration (Phase 5) -----------------------
    // Toast notification: a short-lived message shown top-center of the
    // viewport (fracture/rec/weld/etc. feedback).
    std::string toast_text_;
    float       toast_age_ = 0.f;      // seconds since set; fades after 2.5s
    bool        toast_active_ = false;
    void        SetToast(const std::string& msg) {
        toast_text_ = msg; toast_age_ = 0.f; toast_active_ = true;
    }
    // Auto-orbit: continuous camera rotation around the orbit target.
    bool        auto_orbit_ = false;
    float       auto_orbit_speed_ = 0.3f;  // radians/sec
    // Recording: capture the final framebuffer each frame for up to 10s,
    // then encode to MP4 via ffmpeg (background thread). Mirrors recon_window's
    // Recorder but driven through the workspace's renderer hook.
    struct Recorder {
        std::string dir;
        int w = 0, h = 0;
        std::chrono::steady_clock::time_point t0;
        int n = 0;
        bool active = false;
        bool ffmpeg_ok = false;
        void start(int W, int H, const std::string& base);
        void frame(const std::vector<unsigned char>& rgba);
        void finalize();
    } recorder_;
    void StartRecording();
    void StopRecording();
    void PollRecording();

    // --- blueprint / biology interpretation mode --------------------------
    // The console's document path can interpret two specialized domains:
    //   BLUEPRINT : technical drawings, specs, dimensions -> element tree
    //   BIOLOGY   : anatomical structures, organic forms, tissue layers
    // The mode selects the system prompt, the validation rules, and the
    // reconstruction hints the LLM receives. Default is the generic MACHINE
    // domain (existing behaviour).
    enum class InterpretDomain { Machine, Blueprint, Biology };
    InterpretDomain interp_domain_ = InterpretDomain::Machine;
    // True when the inspector should display physical units alongside values.
    bool show_units_ = true;

    // --- captured SDL input -----------------------------------------------
    // The engine's input hook fires once per SDL event; the UI's UPDATE
    // pass runs later (during Present). These members carry the input from
    // the event hook to the UI pass, and the ONE-SHOT flags are cleared
    // once the UI has consumed them.
    int  mouse_x_ = 0, mouse_y_ = 0;
    bool mouse_down_ = false;
    bool mouse_clicked_ = false;
    bool mouse_released_ = false;
    bool mouse_right_down_ = false;
    bool mouse_right_clicked_ = false;
    bool mouse_middle_down_ = false;
    int  wheel_ = 0;
    bool shift_ = false, ctrl_ = false, alt_ = false;
    std::string text_input_;
    int  key_backspace_ = 0, key_delete_ = 0;
    int  key_left_ = 0, key_right_ = 0, key_up_ = 0, key_down_ = 0;
    int  key_home_ = 0, key_end_ = 0;
    int  key_enter_ = 0, key_escape_ = 0, key_tab_ = 0;
    // Mouse motion accumulated since the last UI pass (orbit / pan deltas).
    int  motion_dx_ = 0, motion_dy_ = 0;
    bool font_ready_ = false;

    // The active sculpt tool.
    Tool  tool_ = Tool::Poke;
    float tool_strength_ = 0.25f;

    // The active RENDER PATH (the 4 compositing modes the renderer
    // offers). The renderer reads it every frame, so switching it in the
    // UI is immediate -- no restart, no env var.
    //
    // The default is the LAYERED splat: the user's distance-plane method
    // (impacts bucketed into distance shells, averaged within each
    // shell, the shells stacked back-to-front). Measured on the gallery's
    // splat scene it beats the classic splat on every metric:
    //   classic : 16108 lit, 145 holes, 2.95% dither
    //   layered : 20808 lit, 138 holes, 2.19% dither
    int  render_path_ = 6;   // 0 march, 1 raymarch, 2 splat classic, 3 splat layered,
                            // 4 splat area (the pixel-footprint pyramid), 5 equation-layered,
                            // 6 tri-infill (the optimized GPU path). DEFAULT = tri-infill.
    static const char* RenderPathName(int i);
    // Push render_path_ into the renderer (the renderer reads its path
    // every frame, so the switch is instant).
    void ApplyRenderPath();

public:
    // Select the render path by index (0 march, 1 raymarch, 2 splat
    // classic, 3 splat layered, 4 splat area, 5 equation-layered). Used by
    // the viewport's path buttons and by the app's --render-path flag.
    void SetRenderPath(int i) {
        if (i < 0 || i > 5) return;
        render_path_ = i;
        ApplyRenderPath();
    }
    int  GetRenderPath() const { return render_path_; }

    // --- blueprint / biology interpretation domain ------------------------
    // Select the domain the document/console interpreter uses. Machine is
    // the default (generic element-tree generation). Blueprint adds
    // dimension/spec validation; Biology adds anatomical/organic hints.
    void SetInterpDomain(InterpretDomain d) { interp_domain_ = d; }
    InterpretDomain InterpDomain() const { return interp_domain_; }
    // Toggle whether the inspector shows physical units (kg, Pa, m/s²).
    void SetShowUnits(bool on) { show_units_ = on; }
    bool ShowUnits() const { return show_units_; }

    // --- physics diagnostics ---------------------------------------------
    // Access the last-step conservation diagnostics and solver state.
    // Returns BY VALUE (engine_.Conservation/SolverState return by value too);
    // returning a reference to the temporary would dangle (B4 bug guard).
    PhysicsConservation Conservation() const {
        return engine_.Conservation();
    }
    PhysicsSolverState  SolverState()  const {
        return engine_.SolverState();
    }

    // --- ML reconstruction control ---------------------------------------
    // Load RMW1 weights for the neural reconstruction backend. Returns
    // false if the file cannot be read or the format is invalid.
    bool LoadReconWeights(const std::string& path);
    // True once weights have been loaded and the backend is ready.
    bool ReconWeightsLoaded() const { return recon_model_.loaded(); }
    // Infer a body from a semantic request using the loaded weights.
    // Returns the new object (nullptr on failure). Uses the CURRENT domain
    // for any domain-specific hints passed to the model.
    GameObject* ReconInfer(const std::string& semantic);

    // --- camera placement (for reproducible captures) ---------------------
    //
    // WHY THIS EXISTS: a pixel gate that measures "the object covers X% of
    // the viewport" is only comparable between runs if the CAMERA is the
    // same. The orbit camera is driven by mouse drags and by the framing
    // pass, so two runs can land on different angles -- and a
    // view-dependent defect (a normal that points the wrong way on one side)
    // then shows up as a flaky gate instead of a reproducible number.
    //
    // These setters let a harness pin the orbit, so "rotate the camera" is
    // an INPUT to a measurement rather than an uncontrolled variable.
    void SetOrbitYaw(float y)   { orbit_yaw_ = y; }
    void SetOrbitPitch(float p) { orbit_pitch_ = p; }
    void SetOrbitDist(float d)  { orbit_dist_ = d; }
    void SetOrbitTarget(const Vec3& t) { orbit_target_ = t; }

    // --- the multi-view capture sweep ---------------------------------------
    //
    // WHY THIS EXISTS: a single pinned-angle capture proves ONE view. The
    // user's point is stronger than that: a defect that hides on one side
    // of an object (an antiparallel normal, a seam that only breaks from
    // below) is only VISIBLE if the app shows the object from SEVERAL
    // perspectives, so the evidence is pixels the user can look at rather
    // than gate numbers they have to trust.
    //
    // Arm a sweep of `count` viewpoints spread over azimuth (and, when
    // `vary_pitch`, alternating elevation), all at the scene's framed
    // distance. On each subsequent frame the sweep advances: it pins the
    // camera to viewpoint k, queues a window capture (drained AFTER
    // Present, so capture == render), and when the last viewpoint is done
    // it requests the app to quit. The result is `count` PPMs named
    // `<prefix>_<k>.ppm`, each already staring at the target object.
    void ArmOrbitSweep(int count, const std::string& prefix,
                       bool vary_pitch = true, int settle_frames = 10);
    // True once every queued viewpoint has been captured (or none was
    // armed). A harness uses this to know the run is over.
    bool OrbitSweepDone() const { return sweep_armed_ && sweep_next_ >= sweep_count_; }
    // The number of captures actually written (for the app to report).
    int OrbitSweepCaptured() const { return sweep_captured_; }
    // Arm the sweep so it STARTS when the next AI-generated object lands.
    //
    // WHY: a model reply takes tens of seconds, and the sweep quits the app
    // when it finishes -- so arming both together would capture the STARTUP
    // scene and exit long before the model answers. Deferred arming waits
    // for AddFromTree to succeed, then frames the scene and starts.
    void ArmOrbitSweepOnNextObject(int count, const std::string& prefix,
                                   bool vary_pitch = true, int settle_frames = 10) {
        sweep_deferred_ = true;
        sweep_count_ = count;
        sweep_prefix_ = prefix;
        sweep_vary_pitch_ = vary_pitch;
        sweep_settle_ = settle_frames;
    }
    bool OrbitSweepDeferred() const { return sweep_deferred_; }
private:

    // --- model reachability probe (background thread) ---------------------
    std::thread probe_thread_;
    bool probe_pending_ = false;
    bool probe_result_ = false;

    // --- engine hooks -----------------------------------------------------
    // A one-shot action to run on the next frame (see PostNextFrame).
    std::function<void()> next_frame_action_;
    bool ran_next_frame_ = false;
    void OnFrame();                 // layout + viewport resize + camera
    void OnOverlay(void* r, int w, int h);   // UI update+draw pass
    bool OnInput(const void* ev);

    // Update+draw helpers.
    void DrawStatusBar();
    void DrawSceneTree();
    void DrawInspector();
    void DrawConsole();
    void DrawViewportOverlay();
    // The navigation gizmo: a sphere-like widget in the viewport corner.
    void DrawGizmo();
    void LayoutGizmo();
    // True when the pointer is over the gizmo.
    bool OverGizmo(int x, int y) const;
    // Physics diagnostics panel (right dock, below inspector when a body
    // is selected). Shows conservation counters and solver state.
    void DrawPhysicsPanel();
    // Physics controls panel (right dock, below physics diagnostics).
    // Exposes toggles/sliders for gravity, thermal/charge coupling, field
    // modules (electrostatic/magnetostatic/aero/hydro), GPU physics dispatch,
    // and weld budget. Pushes state into PhysicsWorld each frame it changes.
    void DrawPhysicsControlsPanel();
    // Script console panel (bottom dock, tabbed with AI Console).
    // Loads/runs .drs scripts, evaluates inline expressions via Cling,
    // shows command log + breakpoint controls + hot reload.
    void DrawScriptConsole();
    // Document reconstruction panel (right dock, below profiler).
    // Parses PDF/CAD/mesh specs → DocumentSpec → graph constraints →
    // validates reconstructed body against the spec.
    void DrawDocumentPanel();
    // Grammar / Assembly panel (right dock, below physics). Shows structural
    // grammar validation, assembly planning controls, and graph diagnostics.
    void DrawGrammarAssemblyPanel();
    // Profiler panel (right dock, below grammar). Shows TriInfill GPU path
    // timing diagnostics (sample/build/raster/upload per frame).
    void DrawProfilerPanel();
    // ML reconstruction control strip (toolbar, next to render-path buttons).
    void DrawMLStrip();
    // Procedural body quick-add strip (toolbar, right of ML strip). Buttons
    // for orb/bump/wave/mound via ElementSystem::BakeBody.
    void DrawProceduralStrip();
    // Toast notification rendering (viewport overlay, top-center).
    void DrawToast();

    // AI plumbing.
    void PollModel();
    void StartProbe();
    void PollProbe();
    void HandleModelReply(const std::string& reply);
    // A SYNCHRONOUS model call: send, wait, return the reply.
    //
    // WHY IT EXISTS: the document path (G15) drives the model itself --
    // InterpretDocument owns the request and the escalation ladder -- and
    // that cannot be expressed in the workspace's asynchronous LLM state
    // machine without restructuring the learner. So the document path needs
    // a blocking call, and this is it.
    //
    // It uses its OWN LLMClient rather than the workspace's `llm_`, because
    // that client is a single in-flight state machine: overwriting its
    // config mid-request would corrupt a request already running.
    std::string AskModelSync(const LLM::Config& cfg,
                             const std::string& system,
                             const std::string& user);
    void UpdateCamera();
    void HandleCameraInput();     // orbit / pan / dolly / object drag
    void HandleKeyboardCamera(float dt);  // WASD-style camera movement
    void ApplyViewpoint(const Viewpoint& v);
    Vec3  AimPoint() const;          // the point under the cursor in the viewport
    void  ApplyTool(Tool tool);
    void  AddProcedural(const char* semantic);
};

} // namespace DeliRium
