// FILEPATH: engine/src/core/Workspace.cpp
// The docked editor: layout, panels, camera, and the AI -> geometry loop.
#include "delirium/core/Workspace.h"

#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/render/Camera.h"
#include "delirium/render/SplatRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

namespace DeliRium {

using namespace UI;

namespace {
// Panel widths as a fraction of the window. The docks shrink on a small
// window but never below their minimum, so the viewport always has room.
constexpr float kLeftFrac   = 0.20f;
constexpr float kRightFrac  = 0.24f;
constexpr int   kMinLeft    = 150;
constexpr int   kMinRight   = 170;
constexpr int   kMinConsole = 110;
constexpr int   kStatusH    = 22;
constexpr int   kConsoleFrac = 26;   // % of window height
constexpr int   kRowH       = 14;    // one widget row
} // namespace

Workspace::Workspace(Engine& eng) : engine_(eng), llm_() {
    LLM::Config cfg;
    cfg.host = model_host_;
    cfg.port = model_port_;
    cfg.model = model_name_;
    cfg.system = AI::ElementTreeSystemPrompt();
    llm_.SetConfig(cfg);
    Logf("workspace ready");
}

Workspace::~Workspace() {
    // The probe thread may still be blocked in a connect() -- join it so the
    // process never outlives a socket in flight.
    if (probe_thread_.joinable()) probe_thread_.join();
    // The DOCUMENT worker (G15) may still be waiting on a model reply.
    // It MUST be joined: a std::thread destroyed while joinable calls
    // std::terminate, so the app would abort on exit instead of shutting
    // down cleanly. The join waits for the model (up to its timeout), which
    // is the price of a clean exit.
    if (doc_thread_.joinable()) doc_thread_.join();
    llm_.Cancel();
}

void Workspace::Attach() {
    engine_.SetCameraControl(Engine::CameraControl::External);
    engine_.SetFrameHook([this] { OnFrame(); });
    engine_.SetOverlayHook([this](void* r, int w, int h) { OnOverlay(r, w, h); });
    engine_.SetInputHook([this](const void* e) { return OnInput(e); });
    StartProbe();

    // Apply the initial render path (the layered splat by default).
    // The renderer seeds its path from the env vars at Initialize, so
    // without this the workspace would start on the march path no matter
    // what the user last chose.
    ApplyRenderPath();

    // AUTO-SELECT the first object.
    //
    // WHY: an editor that opens with an empty Inspector reads as "nothing
    // loaded" even when the scene is full. Godot selects the first node on
    // open.
    if (selected_ == static_cast<size_t>(-1)) {
        if (Scene* s = engine_.CurrentScene(); s && !s->Objects().empty())
            Select(0);
    }

    // FRAME THE SCENE.
    //
    // WHY: the camera must be pulled back far enough to see the whole
    // scene, not just whatever is near the origin. A fixed distance (4.0)
    // is wrong for every body size: a large procedurally-baked body fills
    // the viewport and reads as one unrecognisable blob, while a small one
    // is a speck. The framing is computed from the ACTUAL world bounds of
    // every object -- including each object's Transform, because placement
    // lives outside the frozen records.
    FrameScene();
}

void Workspace::FrameScene() {
    Scene* s = engine_.CurrentScene();
    if (!s) return;

    // Union of every object's world-space bounds.
    Vec3 mn{ 1e30f, 1e30f, 1e30f }, mx{ -1e30f, -1e30f, -1e30f };
    bool any = false;
    for (const auto& o : s->Objects()) {
        if (!o || !o->HasBody()) continue;
        const Vec3 bmn = o->Body().BBoxMin();
        const Vec3 bsp = o->Body().BBoxSpan();
        const Mat4 m = o->GetTransform().ToMatrix();
        for (int c = 0; c < 8; ++c) {
            const Vec3 corner{ bmn.x + ((c&1)?bsp.x:0.f),
                               bmn.y + ((c&2)?bsp.y:0.f),
                               bmn.z + ((c&4)?bsp.z:0.f) };
            const Vec3 w = Mat4_TransformPoint(m, corner);
            mn.x = std::min(mn.x, w.x); mn.y = std::min(mn.y, w.y); mn.z = std::min(mn.z, w.z);
            mx.x = std::max(mx.x, w.x); mx.y = std::max(mx.y, w.y); mx.z = std::max(mx.z, w.z);
        }
        any = true;
    }
    if (!any) return;

    const Vec3 centre{ (mn.x+mx.x)*0.5f, (mn.y+mx.y)*0.5f, (mn.z+mx.z)*0.5f };
    const Vec3 span{ mx.x-mn.x, mx.y-mn.y, mx.z-mn.z };
    const float radius = std::sqrt(span.x*span.x + span.y*span.y + span.z*span.z) * 0.5f;

    orbit_target_ = centre;
    // Pull back far enough to fit the bounding sphere in the view, with
    // margin. The camera FOV is 60 degrees, so the half-angle is 30 deg:
    // d = radius / sin(30 deg) = 2*radius. The 1.35x margin keeps the
    // objects off the very edge of the viewport (and off the panels).
    const float want = std::max(0.5f, radius * 2.0f * 1.35f);
    orbit_dist_ = want;

    // Keep the pitch/yaw but make sure the user is looking at the centre.
    Logf("framed scene: centre (%.2f %.2f %.2f) radius %.2f dist %.2f",
         centre.x, centre.y, centre.z, radius, orbit_dist_);
}

// =============================================================================
// The multi-view capture sweep
// =============================================================================
//
// WHY A SWEEP: a single capture proves one view. A defect that hides on one
// side of an object (an antiparallel normal, a seam that only breaks from
// below) is only VISIBLE from several perspectives -- and the evidence a
// human can check is pixels, not gate numbers. The sweep pins the camera,
// lets the render settle (the renderer rebuilds its surfel bake and grid
// for the new angle), captures the COMPOSITED window (panels included, so
// the context is visible), and advances.
void Workspace::ArmOrbitSweep(int count, const std::string& prefix,
                              bool vary_pitch, int settle_frames) {
    sweep_count_ = std::max(1, count);
    sweep_prefix_ = prefix;
    vary_pitch_ = vary_pitch;
    sweep_vary_pitch_ = vary_pitch;
    sweep_deferred_ = false;   // an immediate arm cancels any pending defer
    sweep_settle_ = std::max(1, settle_frames);
    sweep_next_ = 0;
    sweep_captured_ = 0;
    sweep_armed_ = true;
    sweep_settle_left_ = 0;   // pin the FIRST viewpoint on the next frame
    Logf("orbit sweep armed: %d view(s), prefix '%s', settle %d frame(s)",
         sweep_count_, prefix.c_str(), sweep_settle_);
}

// One sweep step per frame: settle -> capture -> advance. Runs BEFORE the
// camera update in OnFrame so the pinned viewpoint is the one rendered.
void Workspace::SweepStep() {
    if (!sweep_armed_) return;

    // Done: every viewpoint captured. Ask the app to close itself.
    if (sweep_next_ >= sweep_count_) {
        sweep_armed_ = false;
        Logf("orbit sweep complete: %d capture(s) written",
             sweep_captured_);
        engine_.RequestQuit();
        return;
    }

    // PIN the viewpoint on the first frame of each step.
    if (sweep_settle_left_ == 0) {
        const float yaw   = float(sweep_next_) * (2.0f * 3.14159265f)
                            / float(sweep_count_);
        // Alternate elevation across the sweep so both hemispheres are
        // seen: +20 deg, -15 deg, +10 deg, -5 deg ... A defect that only
        // shows from below needs a below camera to exist.
        const float pitch = vary_pitch_
            ? 0.35f * ((sweep_next_ % 2 == 0) ? 1.0f : -0.5f)
            : 0.35f;
        SetOrbitYaw(yaw);
        SetOrbitPitch(pitch);
        // Distance stays at the FRAMED value -- FrameScene() already pulled
        // the camera back so the whole object fits; overriding it here
        // would crop the object and defeat the purpose.
        sweep_settle_left_ = sweep_settle_;
        std::printf("workspace: sweep viewpoint %d/%d  yaw=%.2f pitch=%.2f\n",
                    sweep_next_ + 1, sweep_count_, yaw, pitch);
        return;   // render this frame from the new camera first
    }

    // Still settling: the render needs a few frames at the new angle (the
    // surfel bake and the grid are rebuilt per camera), and a capture taken
    // on the same frame as the pin would read the OLD camera's frame.
    if (--sweep_settle_left_ > 0) return;

    // Capture now. The request is drained AFTER Present(), so the pixels
    // written are the pixels this frame rendered.
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s_%d.ppm", sweep_prefix_.c_str(), sweep_next_);
    engine_.RequestWindowCapture(buf);
    std::printf("workspace: sweep capture requested: %s\n", buf);
    ++sweep_next_;
    sweep_settle_left_ = 0;   // pin the next viewpoint on the next frame
}

// =============================================================================
// Layout
// =============================================================================
void Workspace::ComputeLayout(int w, int h) {
    Layout L;
    L.win_w = w;
    L.win_h = h;
    L.status_h = kStatusH;
    // The console takes a share of the height, but the viewport needs at
    // least a third of it.
    int console = h * kConsoleFrac / 100;
    if (console < kMinConsole) console = kMinConsole;
    if (console > h / 3) console = h / 3;
    L.console_h = console;

    int left = int(float(w) * kLeftFrac);
    if (left < kMinLeft) left = kMinLeft;
    int right = int(float(w) * kRightFrac);
    if (right < kMinRight) right = kMinRight;
    // Never let the docks eat the whole width.
    if (left + right > w - 120) {
        const int excess = left + right - (w - 120);
        left = std::max(80, left - excess / 2);
        right = std::max(90, right - excess / 2);
    }
    L.left_w = left;
    L.right_w = right;

    const int body_top = L.status_h;
    const int body_h = h - L.status_h - L.console_h;
    L.status   = Rect{ 0, 0, w, L.status_h };
    L.left     = Rect{ 0, body_top, left, body_h };
    L.viewport = Rect{ left, body_top, w - left - right, body_h };
    L.right    = Rect{ w - right, body_top, right, body_h };
    L.console  = Rect{ 0, body_top + body_h, w, L.console_h };
    layout_ = L;
    // The gizmo's rect is DERIVED from the viewport, so it must be
    // recomputed whenever the layout is -- otherwise the gizmo's position
    // and the viewport's disagree (the gizmo would still be where the
    // PREVIOUS layout put it). Doing it here also means the geometry is
    // available before the first draw, so the input decision and the
    // draw can never disagree about where the gizmo is.
    LayoutGizmo();
}

// =============================================================================
// Scene / selection
// =============================================================================
GameObject* Workspace::Selected() const {
    Scene* s = engine_.CurrentScene();
    if (!s) return nullptr;
    const auto& objs = s->Objects();
    if (selected_ >= objs.size()) return nullptr;
    return objs[selected_].get();
}

void Workspace::Select(size_t index) {
    Scene* s = engine_.CurrentScene();
    if (!s) return;
    const auto& objs = s->Objects();
    if (index < objs.size() && objs[index]) {
        selected_ = index;
        edit_name_ = objs[index]->name;
        // Frame the selection: aim the orbit target at the object's centre.
        // The object's centre is its LOCAL bbox centre pushed through the
        // transform -- the placement lives outside the frozen records.
        if (GameObject* o = objs[index].get(); o && o->HasBody()) {
            const Vec3 mn = o->Body().BBoxMin();
            const Vec3 sp = o->Body().BBoxSpan();
            const Vec3 local{ mn.x + sp.x * 0.5f, mn.y + sp.y * 0.5f, mn.z + sp.z * 0.5f };
            orbit_target_ = Mat4_TransformPoint(o->GetTransform().ToMatrix(), local);
        }
    }
}

// =============================================================================
// Logging
// =============================================================================
void Workspace::Logf(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    // The log is written from the MAIN thread AND from the document worker
    // thread (G15) -- the worker's model call can fail, and that failure
    // must be reported. A std::vector append from two threads is a data
    // race, so the append is guarded.
    std::lock_guard<std::mutex> guard(log_mutex_);
    log_.emplace_back(buf);
    if (log_.size() > 400) log_.erase(log_.begin(), log_.begin() + 100);
}

// =============================================================================
// Camera (the workspace owns it)
// =============================================================================
void Workspace::UpdateCamera() {
    Scene* s = engine_.CurrentScene();
    if (!s) return;
    Camera& cam = s->GetCamera();
    const float cp = std::cos(orbit_pitch_);
    const Vec3 dir{ std::sin(orbit_yaw_) * cp,
                    std::sin(orbit_pitch_),
                    std::cos(orbit_yaw_) * cp };
    cam.position = { orbit_target_.x + dir.x * orbit_dist_,
                     orbit_target_.y + dir.y * orbit_dist_,
                     orbit_target_.z + dir.z * orbit_dist_ };
    cam.target = orbit_target_;
    cam.up = { 0.f, 1.f, 0.f };
}

Vec3 Workspace::AimPoint() const {
    // The point on the selected object under the CURSOR. Uses the viewer's
    // own ray math so the pick matches what the user sees.
    GameObject* o = Selected();
    if (!o || !o->HasBody()) return orbit_target_;
    Scene* s = engine_.CurrentScene();
    if (!s) return orbit_target_;
    const Camera& cam = s->GetCamera();
    // The cursor is in WINDOW coordinates; convert to viewport coordinates.
    const int vx = ui_.mouse_x - layout_.viewport.x;
    const int vy = ui_.mouse_y - layout_.viewport.y;
    if (vx < 0 || vy < 0 || vx >= layout_.viewport.w || vy >= layout_.viewport.h)
        return orbit_target_;
    Vec3 origin, dir;
    // PixelRay expects the FRAME size; the renderer works at the viewport
    // size, so use the viewport dimensions.
    cam.PixelRay(float(vx), float(vy), layout_.viewport.w, layout_.viewport.h, origin, dir);
    // The body's surface is in LOCAL space; the ray is in WORLD space. Pull
    // the ray into local space before the intersection test, or the pick
    // misses every moved object.
    const Transform& tr = o->GetTransform();
    const Vec3 lo = tr.InversePoint(origin);
    const Vec3 ld = tr.InverseVector(dir);
    SurfaceHit hit;
    if (o->Body().Raycast(lo, ld, hit)) {
        // Bring the hit back to world space for the caller. The matrix is
        // the exact forward map (local -> world), so the round trip is exact.
        return Mat4_TransformPoint(tr.ToMatrix(), hit.p);
    }
    return orbit_target_;
}

// =============================================================================
// The AI path
// =============================================================================
bool Workspace::AskModel(const std::string& prompt) {
    if (prompt.empty()) return false;
    if (llm_.IsBusy()) {
        Logf("model busy -- request ignored");
        return false;
    }
    LLM::Config cfg = llm_.GetConfig();
    cfg.system = AI::ElementTreeSystemPrompt();
    llm_.SetConfig(cfg);
    Logf("> %s", prompt.c_str());
    if (!llm_.Send(prompt)) {
        Logf("could not send request");
        return false;
    }
    busy_ = true;
    return true;
}

bool Workspace::SubmitPrompt(const std::string& prompt) {
    // Mirror the Generate button exactly: the text goes through the field,
    // the field is cleared, and the request goes out.
    prompt_ = prompt;
    const std::string p = prompt_;
    prompt_.clear();
    return AskModel(p);
}

void Workspace::PostNextFrame(std::function<void()> fn) {
    next_frame_action_ = std::move(fn);
    ran_next_frame_ = false;
}

void Workspace::PollModel() {
    if (!llm_.IsBusy() && !busy_) return;
    switch (llm_.CurrentState()) {
        case LLM::State::Failed: {
            Logf("model error: %s", llm_.Error().c_str());
            busy_ = false;
            model_status_ = "error";
            break;
        }
        case LLM::State::Done: {
            const std::string reply = llm_.TakeResponse();
            busy_ = false;
            model_status_ = "ready";
            HandleModelReply(reply);
            break;
        }
        default:
            model_status_ = "thinking";
            break;
    }
}

void Workspace::HandleModelReply(const std::string& reply) {
    if (reply.empty()) {
        Logf("model returned nothing");
        return;
    }
    // A model often wraps its JSON in prose or a markdown fence: extract the
    // first balanced object before parsing.
    std::string json = reply;
    AI::JsonValue probe;
    std::string why;
    if (!AI::ParseJson(json, probe, why)) {
        std::string extracted;
        if (AI::ExtractJsonObject(reply, extracted)) {
            json = extracted;
            Logf("extracted JSON from the reply");
        }
    }
    std::unique_ptr<Element> tree;
    if (!AI::ParseElementTree(json, tree, why)) {
        Logf("invalid element tree: %s", why.c_str());
        Logf("reply: %.200s", reply.c_str());
        return;
    }
    Logf("parsed: %s", AI::DescribeTree(*tree).c_str());
    GameObject* obj = AddFromTree(tree, "generated");
    if (obj) {
        Logf("created '%s' (%zu patches, %zu anchors)", obj->name.c_str(),
             obj->Body().Patches().size(), obj->Body().AnchorBlob().size());
    }
}

GameObject* Workspace::AddFromTree(const std::unique_ptr<Element>& tree,
                                   const std::string& name) {
    if (!tree) return nullptr;
    Scene* s = engine_.CurrentScene();
    if (!s) { Logf("no scene"); return nullptr; }
    std::string why;
    // Bake at a resolution that can RESOLVE the substance.
    //
    // The baker reconstructs each patch as a tangent plane (bilinear corner
    // positions + a normal-direction displacement), so a feature whose size
    // approaches the sample spacing gets CHORDED -- a torus's ring flattens
    // into a disc. The sample spacing is extent/(N-1); at N=12 it is 0.091,
    // comparable to a default tube radius of 0.12, and the body kept only
    // 19% of the field's relief. At N=48 it keeps 67%.
    //
    // The bake costs ~8 ms, which is nothing next to the model's reply
    // time, so there is no reason to ship the coarse version.
    const int kBakeGrid = 48;
    auto body = ElementSystem::BakeBody(*tree, kBakeGrid, 1.0f, why);
    if (!body) {
        Logf("bake failed: %s", why.c_str());
        return nullptr;
    }
    // Place the new object so it does not sit exactly on top of the others.
    const size_t n = s->Objects().size();
    const float offset = float(n) * 1.4f;
    GameObject* obj = s->Create(name, std::move(body));
    if (!obj) { Logf("scene rejected the object"); return nullptr; }
    obj->GetTransform().position = { offset - 1.4f, 0.f, 0.f };
    Select(s->Objects().size() - 1);
    // The DEFERRED orbit sweep (ArmOrbitSweepOnNextObject) starts HERE:
    // the AI object has landed, so frame the scene on it and begin the
    // multi-view capture. FrameScene() recomputes the distance from the
    // new object's world bounds first, so every viewpoint frames it.
    if (sweep_deferred_) {
        FrameScene();
        ArmOrbitSweep(sweep_count_, sweep_prefix_, sweep_vary_pitch_,
                      sweep_settle_);
        Logf("orbit sweep started on the generated object");
    }
    return obj;
}

// =============================================================================
// DOCUMENT MODE (G15)
// =============================================================================
std::string Workspace::AskModelSync(const LLM::Config& cfg,
                                    const std::string& system,
                                    const std::string& user) {
    // A FRESH client per call: the workspace's `llm_` is a single in-flight
    // state machine, and overwriting its config while a request is running
    // would corrupt that request. The document path is a blocking call, so
    // it can own its client for the duration.
    LLM::Config c = cfg;
    c.system = system;
    // A document prompt is LARGE (the system prompt alone is ~2,900
    // tokens -- 11.6 KB -- plus the document and the worked examples), so
    // the token budget and the timeout are both raised above the defaults.
    // The default 1024 tokens is not enough for a schema-shaped JSON
    // answer.
    //
    // The timeout is 900 s rather than 300 s because a MEASURED successful
    // run took 276.4 s on a loaded machine. A 300 s ceiling sat only 8%
    // above the worst observed case, so any additional load turned a
    // working request into a "transport" failure -- indistinguishable from
    // "the model is unreachable".
    if (c.max_tokens < 1400) c.max_tokens = 1400;
    if (c.timeout_seconds < 900) c.timeout_seconds = 900;

    LLM::LLMClient client(c);
    if (!client.Send(user)) {
        Logf("document: could not start the request");
        return {};
    }
    // Wait for the reply. The timeout is the CLIENT's own timeout (it fails
    // the request), so the loop is bounded by it; the extra bound here is a
    // belt-and-braces ceiling so a client that never transitions cannot
    // hang the UI forever.
    const int max_ticks = c.timeout_seconds * 20 + 60;   // 50 ms per tick
    int ticks = 0;
    while (client.IsBusy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (++ticks > max_ticks) {
            client.Cancel();
            Logf("document: the request timed out after %ds", c.timeout_seconds);
            return {};
        }
    }
    if (client.CurrentState() == LLM::State::Failed) {
        Logf("document: model error: %s", client.Error().c_str());
        return {};
    }
    return client.TakeResponse();
}

bool Workspace::LoadDocument(const std::string& path) {
    if (path.empty()) {
        Logf("document: no path given");
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        // A missing document is the most likely mistake, and the message
        // must NAME THE PATH -- "document not found" without the path is
        // useless when the user typed it.
        Logf("document: cannot read '%s'", path.c_str());
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (text.empty()) {
        Logf("document: '%s' is empty", path.c_str());
        return false;
    }
    doc_path_ = path;
    doc_text_ = std::move(text);
    Logf("document: loaded '%s' (%zu bytes, domain=%s)", path.c_str(),
         doc_text_.size(), doc_domain_.c_str());
    return true;
}

bool Workspace::SaveDocCorpus() {
    if (!doc_log_.Save(doc_corpus_path_)) {
        Logf("corpus: could not save '%s'", doc_corpus_path_.c_str());
        return false;
    }
    Logf("corpus: saved %zu entries to '%s'", doc_log_.Size(),
         doc_corpus_path_.c_str());
    return true;
}

bool Workspace::LoadDocCorpus() {
    if (!doc_log_.Load(doc_corpus_path_)) {
        // A missing corpus is NOT an error -- it is the cold-start case.
        Logf("corpus: no corpus at '%s' yet (starting cold)", doc_corpus_path_.c_str());
        return false;
    }
    Logf("corpus: loaded %zu entries (%zu ok, %zu failed) from '%s'",
         doc_log_.Size(), doc_log_.SuccessCount(), doc_log_.FailureCount(),
         doc_corpus_path_.c_str());
    return true;
}

bool Workspace::InterpretLoadedDocument() {
    if (doc_text_.empty()) {
        Logf("document: nothing loaded -- set a document path first");
        return false;
    }
    if (doc_busy_) {
        Logf("document: already working");
        return false;
    }
    if (!model_online_) {
        Logf("document: the local model is offline");
        return false;
    }
    // A previous worker must be joined before a new one starts -- joining
    // a joinable std::thread destroys it.
    if (doc_thread_.joinable()) doc_thread_.join();

    doc_busy_ = true;
    doc_has_result_.store(false, std::memory_order_release);
    doc_thread_ok_ = false;
    doc_thread_reason_.clear();
    doc_thread_tree_.reset();

    // The document id is the file's stem: it identifies the document in the
    // corpus and in the error log, so it must be stable and human-readable.
    std::string id = doc_path_;
    const size_t slash = id.find_last_of('/');
    if (slash != std::string::npos) id = id.substr(slash + 1);

    AI::Document doc;
    doc.id = id;
    doc.text = doc_text_;
    doc.domain = doc_domain_;

    Logf("document: interpreting '%s' (%zu bytes)", id.c_str(), doc.text.size());

    // The retrieval corpus is consulted for the worked examples -- and the
    // log line makes it VISIBLE, because a learning system whose learning is
    // invisible is indistinguishable from one that has not learned.
    const auto retrieved = doc_log_.RetrieveSimilar(doc.text, doc.domain, 3);
    if (retrieved.empty()) {
        Logf("document: corpus has no similar success (cold start)");
    } else {
        Logf("document: corpus supplied %zu worked example(s), best '%s' (%.0f%%)",
             retrieved.size(), retrieved[0].document_id.c_str(),
             retrieved[0].similarity * 100.0);
    }

    // The worker gets its OWN COPY of the corpus. The corpus is written by
    // the worker (it records the attempt) and read by the main thread (the
    // console shows its size) -- sharing one object across the two would be
    // a data race on a std::vector. The copy is merged back when the worker
    // finishes, so the corpus still accumulates.
    AI::DocErrorLog worker_log = doc_log_;
    const LLM::Config base = llm_.GetConfig();
    const std::string escalate_model = doc_escalate_model_;

    doc_thread_ = std::thread([this, doc, base, escalate_model,
                               worker_log]() mutable {
        // The model call. The document path drives the model ITSELF, so the
        // workspace's asynchronous LLM client is not used -- the learner
        // owns the request and the ladder.
        const auto ask = [this, base](const std::string& system,
                                      const std::string& user) {
            return AskModelSync(base, system, user);
        };

        AI::DocResult r = AI::InterpretDocument(doc, ask, worker_log,
                                                /*escalate=*/false);

        // The ESCALATION rung: if the local model failed and a larger model
        // is configured, the SAME document is retried against it with the
        // failure reason, the error log and the worked examples attached.
        if (!r.ok && !escalate_model.empty()) {
            LLM::Config big = base;
            big.model = escalate_model;
            const auto ask_big = [this, big](const std::string& system,
                                             const std::string& user) {
                return AskModelSync(big, system, user);
            };
            const std::function<std::string(const std::string&, const std::string&)>
                larger_fn = ask_big;
            r = AI::InterpretDocument(doc, ask_big, worker_log,
                                      /*escalate=*/true, &larger_fn);
        }

        doc_thread_ok_ = r.ok;
        doc_thread_reason_ = r.ok ? std::string() : r.reason;
        doc_thread_tree_ = std::move(r.tree);
        // The updated corpus goes back to the main thread.
        doc_log_ = std::move(worker_log);
        // The flag goes LAST, with release ordering: the main thread
        // acquires it and then reads the fields above, so they are visible.
        doc_has_result_.store(true, std::memory_order_release);
    });

    return true;
}

void Workspace::PollDocument() {
    // The worker sets `doc_has_result_` when it finishes. This function
    // runs on the MAIN thread, and it applies the result here -- the scene
    // is not thread-safe, so the bake and the insertion must not happen on
    // the worker.
    if (!doc_has_result_.load(std::memory_order_acquire)) return;
    if (doc_thread_.joinable()) doc_thread_.join();

    doc_busy_ = false;
    doc_has_result_.store(false, std::memory_order_release);  // reset for the next run

    if (!doc_thread_ok_) {
        Logf("document: FAILED -- %s", doc_thread_reason_.c_str());
        Logf("document: corpus is now %zu entries (%zu ok, %zu failed)",
             doc_log_.Size(), doc_log_.SuccessCount(), doc_log_.FailureCount());
        SaveDocCorpus();
        return;
    }

    Logf("document: parsed -- %s", AI::DescribeTree(*doc_thread_tree_).c_str());
    GameObject* obj = AddFromTree(doc_thread_tree_, "doc:" + doc_path_);
    if (obj) {
        Logf("document: created '%s' (%zu patches, %zu anchors)", obj->name.c_str(),
             obj->Body().Patches().size(), obj->Body().AnchorBlob().size());
    }
    Logf("document: corpus is now %zu entries (%zu ok, %zu failed)",
         doc_log_.Size(), doc_log_.SuccessCount(), doc_log_.FailureCount());
    // Persist immediately: the corpus is the system's memory, and a memory
    // that is lost on exit is not a memory.
    SaveDocCorpus();
}

// =============================================================================
// Model reachability probe
// =============================================================================
void Workspace::StartProbe() {
    if (probe_pending_) return;
    probe_pending_ = true;
    const std::string host = model_host_;
    const int port = model_port_;
    if (probe_thread_.joinable()) probe_thread_.join();
    probe_thread_ = std::thread([this, host, port] {
        const bool up = LLM::LLMClient::Ping(host, port, 3);
        probe_result_ = up;
        probe_pending_ = false;
    });
}

void Workspace::PollProbe() {
    if (probe_pending_ || model_probed_) return;
    model_probed_ = true;
    model_online_ = probe_result_;
    model_status_ = model_online_ ? "online" : "offline";
    Logf("local model %s:%d %s", model_host_.c_str(), model_port_,
         model_online_ ? "online" : "offline");
}

// =============================================================================
// Frame hooks
// =============================================================================
void Workspace::OnFrame() {
    const int w = engine_.WindowWidth();
    const int h = engine_.WindowHeight();
    if (w <= 0 || h <= 0) return;
    ComputeLayout(w, h);

    // The viewport renders at ITS OWN resolution, so the GPU work matches
    // what the user sees. Resizing reallocates the frame buffers.
    if (SplatRenderer* r = engine_.GetRenderer()) {
        if (layout_.viewport.w > 0 && layout_.viewport.h > 0)
            r->Resize(layout_.viewport.w, layout_.viewport.h);
        r->SetPresentRect(layout_.viewport.x, layout_.viewport.y,
                          layout_.viewport.w, layout_.viewport.h);
    }

    // The orbit sweep (if armed) pins the camera BEFORE UpdateCamera(), so
    // the frame that follows renders from the pinned viewpoint.
    SweepStep();
    HandleCameraInput();
    HandleKeyboardCamera(1.0f / 60.0f);
    UpdateCamera();
    PollModel();
    PollProbe();

    // Apply the document worker's result, if it finished. This is where the
    // bake and the scene insertion happen -- on the MAIN thread, because the
    // scene is not thread-safe.
    PollDocument();

    // The AUTOMATIC DOCUMENT runner (G15).
    //
    // It fires here -- AFTER PollProbe() -- because the probe result only
    // lands in PollProbe(), and the document path refuses to run against an
    // offline model. Firing it before the probe would mean the very first
    // run always failed with "the local model is offline".
    //
    // It is disarmed BEFORE the callback runs, so a callback that somehow
    // re-enters the frame cannot fire it twice.
    //
    // The callback returns whether the request was STARTED (the work is
    // asynchronous now), so the log line says "started" -- not "succeeded".
    // The outcome arrives later, through PollDocument().
    if (auto_doc_armed_ && model_online_ && auto_doc_runner_) {
        auto_doc_armed_ = false;
        auto fn = std::move(auto_doc_runner_);
        auto_doc_runner_ = nullptr;
        const bool started = fn();
        Logf("document: automatic interpretation %s",
             started ? "started" : "could not start");
    }

    // GIVE THE CPU BACK WHILE THE MODEL WORKS.
    //
    // The editor renders every frame at full resolution; a local 7B model
    // needs the same cores. At 60 FPS the editor is the bottleneck and a
    // reply that takes 30 s of model time stretches to minutes -- which
    // reads as "the Generate button is broken" rather than "the model is
    // slow". Dropping to 10 FPS while a request is in flight costs nothing
    // visually (the user is reading the console, not the viewport) and
    // hands the cores back to the model.
    //
    // The DOCUMENT path needs the same throttle: its model call runs on a
    // worker thread, so the editor's frame rate competes with it exactly
    // as the semantic path's does.
    if (busy_ || doc_busy_)
        engine_.SetMaxFps(10.0);
    else
        engine_.SetMaxFps(60.0);

    // Run a posted one-shot action AFTER the layout exists, so a scripted
    // interaction hits the same widgets (and the same rects) a click would.
    if (next_frame_action_) {
        auto fn = std::move(next_frame_action_);
        next_frame_action_ = nullptr;
        ran_next_frame_ = true;
        fn();
    }

    // The engine owns the frame timing; the status bar shows it.
    fps_ = engine_.Fps();
    frame_ms_ = engine_.LastFrameMs();
}

// =============================================================================
// Camera input: orbit, pan, dolly, and object dragging.
//
// THE INTERACTION MODEL (Godot-like): the mouse only drives the camera when
// it is OVER THE VIEWPORT. Over a panel it belongs to the UI. That is why
// the checks are against the viewport rect, not the whole window.
// =============================================================================
void Workspace::HandleCameraInput() {
    // THE UI-DECIDES-FIRST RULE applies to the CONTINUOUS camera inputs
    // too, not just the click: the wheel over a slider or a number field
    // that pokes into the viewport rect must scroll THAT widget, not
    // dolly the camera. And a middle-button drag that started on the
    // viewport but wandered over a panel must keep panning (a drag that
    // started is owned by the viewport), while a middle-button drag that
    // STARTED on a panel must never pan.
    const bool over_viewport = ClickReachesViewport(mouse_x_, mouse_y_);
    hover_viewport_ = over_viewport;

    // --- dolly: the wheel over the viewport zooms ------------------------
    if (over_viewport && wheel_ != 0) {
        orbit_dist_ *= (1.0f - 0.12f * float(wheel_));
        if (orbit_dist_ < 0.25f) orbit_dist_ = 0.25f;
        if (orbit_dist_ > 60.f) orbit_dist_ = 60.f;
    }

    // --- the GIZMO: a sphere-like widget that rotates the camera ---------
    //
    // WHY THE GIZMO IS SEPARATE FROM THE VIEWPORT: the user asked for the
    // whole camera drag to happen "through an external sphere-like tool".
    // Keeping it in its own small rect means dragging it CANNOT land on
    // the geometry, so looking around never risks a poke.
    if (gizmo_drag_) {
        const int dx = mouse_x_ - gizmo_last_x_;
        const int dy = mouse_y_ - gizmo_last_y_;
        gizmo_last_x_ = mouse_x_;
        gizmo_last_y_ = mouse_y_;
        orbit_yaw_ += float(dx) * 0.010f;
        orbit_pitch_ -= float(dy) * 0.010f;
        const float lim = 1.45f;
        if (orbit_pitch_ > lim) orbit_pitch_ = lim;
        if (orbit_pitch_ < -lim) orbit_pitch_ = -lim;
        return;
    }

    // --- pan: middle-drag moves the orbit target -------------------------
    if (panning_) {
        const int dx = mouse_x_ - pan_last_x_;
        const int dy = mouse_y_ - pan_last_y_;
        pan_last_x_ = mouse_x_;
        pan_last_y_ = mouse_y_;
        if (dx || dy) {
            // Pan in the camera's screen plane, scaled by the distance so
            // the drag tracks the cursor at any zoom level.
            Scene* s = engine_.CurrentScene();
            const float scale = orbit_dist_ * 0.0016f;
            float rx = 1.f, rz = 0.f;
            if (s) {
                const Camera& cam = s->GetCamera();
                Vec3 fwd = cam.target - cam.position;
                const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
                if (fl > 1e-6f) {
                    fwd = fwd * (1.f/fl);
                    rx = fwd.z; rz = -fwd.x;   // camera right (y-up)
                    const float rl = std::sqrt(rx*rx + rz*rz);
                    if (rl > 1e-6f) { rx /= rl; rz /= rl; }
                }
            }
            orbit_target_.x += (-rx * dx) * scale;
            orbit_target_.z += (-rz * dx) * scale;
            orbit_target_.y += (dy) * scale;
        }
        return;
    }

    // --- orbit: left-drag rotates the camera ------------------------------
    // ONLY when the drag started over the viewport. The flag is set on the
    // button-down and cleared on release, so a drag that wanders off the
    // viewport still orbits (as it should) but a drag that started on a
    // panel never does.
    if (orbiting_) {
        orbit_yaw_ += float(motion_dx_) * 0.006f;
        orbit_pitch_ -= float(motion_dy_) * 0.006f;
        const float lim = 1.45f;
        if (orbit_pitch_ > lim) orbit_pitch_ = lim;
        if (orbit_pitch_ < -lim) orbit_pitch_ = -lim;
    }

    // --- object drag: the Move tool slides the selected object ------------
    if (dragging_object_) {
        GameObject* o = Selected();
        if (o) {
            const int dx = mouse_x_ - drag_last_x_;
            drag_last_x_ = mouse_x_;
            if (dx) {
                // Slide along the camera's right axis, scaled by the distance
                // so the object tracks the cursor at any zoom.
                Scene* s = engine_.CurrentScene();
                float rx = 1.f, rz = 0.f;
                if (s) {
                    const Camera& cam = s->GetCamera();
                    Vec3 fwd = cam.target - cam.position;
                    const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
                    if (fl > 1e-6f) {
                        fwd = fwd * (1.f/fl);
                        rx = fwd.z; rz = -fwd.x;
                        const float rl = std::sqrt(rx*rx + rz*rz);
                        if (rl > 1e-6f) { rx /= rl; rz /= rl; }
                    }
                }
                const float amt = float(dx) * orbit_dist_ * 0.0016f;
                o->GetTransform().position.x += rx * amt;
                o->GetTransform().position.z += rz * amt;
            }
        }
    }
}

void Workspace::HandleKeyboardCamera(float dt) {
    // The keys move the CAMERA in its own frame: W/S along the view
    // direction, A/D along the view's right, Q/E along world up.
    //
    // WHY THIS CANNOT CONFLICT WITH THE TOOLS: the sculpt tools act on the
    // GEOMETRY under the cursor; these keys act on the CAMERA. They are
    // different subjects, so holding W while poking does not drag the
    // object and poking while holding W does not move the camera.
    const bool any = cam_key_fwd_ || cam_key_back_ || cam_key_left_ ||
                     cam_key_right_ || cam_key_up_ || cam_key_down_ ||
                     cam_key_roll_;
    if (!any) return;

    const float speed = orbit_dist_ * 0.9f * dt;
    Scene* s = engine_.CurrentScene();
    float fx = 0.f, fy = 0.f, fz = 0.f;
    float rx = 1.f, rz = 0.f;
    if (s) {
        const Camera& cam = s->GetCamera();
        Vec3 fwd = cam.target - cam.position;
        const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
        if (fl > 1e-6f) {
            fwd = fwd * (1.f/fl);
            fx = fwd.x; fy = fwd.y; fz = fwd.z;
            rx = fwd.z; rz = -fwd.x;
            const float rl = std::sqrt(rx*rx + rz*rz);
            if (rl > 1e-6f) { rx /= rl; rz /= rl; }
        }
    }
    if (cam_key_fwd_) {
        orbit_target_.x += fx * speed; orbit_target_.y += fy * speed; orbit_target_.z += fz * speed;
    }
    if (cam_key_back_) {
        orbit_target_.x -= fx * speed; orbit_target_.y -= fy * speed; orbit_target_.z -= fz * speed;
    }
    if (cam_key_left_)  { orbit_target_.x -= rx * speed; orbit_target_.z -= rz * speed; }
    if (cam_key_right_) { orbit_target_.x += rx * speed; orbit_target_.z += rz * speed; }
    if (cam_key_up_)    { orbit_target_.y += speed; }
    if (cam_key_down_)  { orbit_target_.y -= speed; }
    if (cam_key_roll_)  { orbit_yaw_ += 1.4f * dt; }
}

void Workspace::ApplyViewpoint(const Viewpoint& v) {
    orbit_yaw_ = v.yaw;
    orbit_pitch_ = v.pitch;
    orbit_dist_ = v.dist;
    orbit_target_ = v.target;
}

void Workspace::SaveViewpoint(const std::string& name) {
    Viewpoint v;
    v.name = name;
    v.yaw = orbit_yaw_; v.pitch = orbit_pitch_;
    v.dist = orbit_dist_; v.target = orbit_target_;
    for (size_t i = 0; i < viewpoints_.size(); ++i) {
        if (viewpoints_[i].name == name) {
            viewpoints_[i] = v;
            active_view_ = int(i);
            Logf("viewpoint '%s' updated", name.c_str());
            return;
        }
    }
    viewpoints_.push_back(v);
    active_view_ = int(viewpoints_.size()) - 1;
    Logf("viewpoint '%s' saved", name.c_str());
}

bool Workspace::GotoViewpoint(int index) {
    if (index < 0 || index >= int(viewpoints_.size())) return false;
    ApplyViewpoint(viewpoints_[size_t(index)]);
    active_view_ = index;
    Logf("viewpoint: %s", viewpoints_[size_t(index)].name.c_str());
    return true;
}

void Workspace::OnOverlay(void* r, int w, int h) {
    if (!r) return;
    // The font atlas needs the SDL renderer; build it once, lazily.
    if (!font_ready_) {
        font_ready_ = font_.Init(static_cast<SDL_Renderer*>(r));
        if (!font_ready_) std::fprintf(stderr, "Workspace: font atlas failed\n");
    }

    // Push the captured input into the UI context. The engine's input hook
    // runs per SDL EVENT; the UI pass runs once per FRAME, so the flags are
    // latched here and cleared after the pass.
    ui_.mouse_x = mouse_x_;
    ui_.mouse_y = mouse_y_;
    ui_.mouse_down = mouse_down_;
    ui_.mouse_clicked = mouse_clicked_;
    ui_.mouse_released = mouse_released_;
    ui_.mouse_right_down = mouse_right_down_;
    ui_.mouse_right_clicked = mouse_right_clicked_;
    ui_.wheel = wheel_;
    ui_.shift = shift_;
    ui_.ctrl = ctrl_;
    ui_.alt = alt_;
    ui_.text_input = text_input_;
    ui_.key_backspace = key_backspace_;
    ui_.key_delete = key_delete_;
    ui_.key_left = key_left_;
    ui_.key_right = key_right_;
    ui_.key_up = key_up_;
    ui_.key_down = key_down_;
    ui_.key_home = key_home_;
    ui_.key_end = key_end_;
    ui_.key_enter = key_enter_;
    ui_.key_escape = key_escape_;
    ui_.key_tab = key_tab_;

    ui_.BeginFrame(static_cast<SDL_Renderer*>(r), &font_, w, h);

    DrawStatusBar();
    DrawSceneTree();
    DrawInspector();
    DrawConsole();
    DrawViewportOverlay();
    DrawGizmo();

    ui_.EndFrame();

    // Clear the ONE-SHOT input flags: they described a single frame.
    mouse_clicked_ = false;
    mouse_released_ = false;
    mouse_right_clicked_ = false;
    wheel_ = 0;
    text_input_.clear();
    key_backspace_ = key_delete_ = 0;
    key_left_ = key_right_ = key_up_ = key_down_ = 0;
    key_home_ = key_end_ = key_enter_ = key_escape_ = key_tab_ = 0;
    motion_dx_ = motion_dy_ = 0;
}

// =============================================================================
// Panels
// =============================================================================
void Workspace::DrawStatusBar() {
    const Theme& th = ThemeDefault();
    const Rect r = layout_.status;
    ui_.FillRect(r, th.bg_header);
    ui_.DrawRect(r, th.border);

    Scene* s = engine_.CurrentScene();
    const size_t n = s ? s->Objects().size() : 0;
    GameObject* sel = Selected();

    char buf[256];
    std::snprintf(buf, sizeof(buf), "DeliRium  %d FPS  %.1f ms  |  %zu obj  |  sel: %s",
                  fps_, frame_ms_, n, sel ? sel->name.c_str() : "none");
    ui_.Text(Rect{ r.x + 6, r.y + (r.h - BitmapFont::kGlyphH)/2, r.w - 200, BitmapFont::kGlyphH },
             buf, th.fg);

    const char* tool_names[] = { "poke", "dent", "refine", "coarsen", "fracture", "move", "navigate" };
    std::snprintf(buf, sizeof(buf), "tool: %s", tool_names[int(tool_)]);
    ui_.Text(Rect{ r.Right() - 300, r.y + (r.h - BitmapFont::kGlyphH)/2, 120, BitmapFont::kGlyphH },
             buf, th.fg_accent);

    // The active render path, so the user always knows which compositor
    // produced the image.
    std::snprintf(buf, sizeof(buf), "path: %s", RenderPathName(render_path_));
    ui_.Text(Rect{ r.Right() - 470, r.y + (r.h - BitmapFont::kGlyphH)/2, 160, BitmapFont::kGlyphH },
             buf, th.fg_dim);

    std::snprintf(buf, sizeof(buf), "model: %s", model_status_.c_str());
    const unsigned char* mc = model_online_ ? th.fg_good
                           : (model_status_ == "error" ? th.fg_bad : th.fg_dim);
    ui_.Text(Rect{ r.Right() - 170, r.y + (r.h - BitmapFont::kGlyphH)/2, 164, BitmapFont::kGlyphH },
             buf, mc);
}

void Workspace::DrawSceneTree() {
    const Theme& th = ThemeDefault();
    Rect content = ui_.Panel(layout_.left, "Scene");
    Scene* s = engine_.CurrentScene();
    if (!s) {
        ui_.Text(content, "no scene", th.fg_dim);
        return;
    }
    const auto& objs = s->Objects();
    const int row_h = kRowH;
    const int content_h = int(objs.size()) * row_h;
    tree_scroll_ = ui_.ScrollArea(content, "tree_scroll", &tree_scroll_, content_h);

    int y = content.y - tree_scroll_;
    for (size_t i = 0; i < objs.size(); ++i) {
        const Rect row{ content.x, y, content.w, row_h };
        if (row.Bottom() < content.y || row.y > content.Bottom()) { y += row_h; continue; }
        const bool sel = (i == selected_);
        if (sel) ui_.FillRect(row, th.bg_active);
        const char* marker = sel ? ">" : " ";
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s %s", marker, objs[i]->name.c_str());
        ui_.TextClipped(row, buf, sel ? th.fg : th.fg_dim);
        if (ui_.MouseClicked(row)) Select(i);
        y += row_h;
    }

    // The "add" row: a procedural starter body, no model needed.
    const Rect add{ content.x, content.Bottom() - row_h, content.w, row_h };
    ui_.HSeparator(Rect{ add.x, add.y - 1, add.w, 1 });
    if (ui_.Button(add, "+ procedural body")) AddProcedural("orb");
}

void Workspace::DrawInspector() {
    const Theme& th = ThemeDefault();
    Rect content = ui_.Panel(layout_.right, "Inspector");
    GameObject* o = Selected();
    if (!o) {
        ui_.Text(content, "nothing selected", th.fg_dim);
        return;
    }
    const int row_h = kRowH;
    int y = content.y + 2;

    // --- name -------------------------------------------------------------
    // GameObject::name is const (an object keeps its identity), so the
    // editable display name lives in the workspace, keyed by the selected
    // index. Renaming does not touch the object.
    ui_.Text(Rect{ content.x, y, content.w, row_h }, "name", th.fg_dim);
    y += row_h;
    ui_.TextField(Rect{ content.x, y, content.w, row_h }, "name", &edit_name_);
    y += row_h + 4;

    // --- transform --------------------------------------------------------
    // The placement lives OUTSIDE the frozen 64-byte records, so editing it
    // never re-quantizes an anchor -- dragging an object is free.
    ui_.Text(Rect{ content.x, y, content.w, row_h }, "transform", th.fg_accent);
    y += row_h;

    Transform& tr = o->GetTransform();
    const int lab_w = 16;
    const int val_w = content.w - lab_w - 4;
    const char axes[] = { 'x', 'y', 'z' };

    ui_.Text(Rect{ content.x, y, lab_w, row_h }, "pos", th.fg_dim);
    float* pos[3] = { &tr.position.x, &tr.position.y, &tr.position.z };
    for (int a = 0; a < 3; ++a) {
        const Rect fr{ content.x + lab_w + a * (val_w/3 + 2), y, val_w/3 - 2, row_h };
        ui_.NumberField(fr, std::string("pos") + axes[a], pos[a], 0.01f, -100.f, 100.f, 3);
    }
    y += row_h;

    ui_.Text(Rect{ content.x, y, lab_w, row_h }, "scl", th.fg_dim);
    float* scl[3] = { &tr.scale.x, &tr.scale.y, &tr.scale.z };
    for (int a = 0; a < 3; ++a) {
        const Rect fr{ content.x + lab_w + a * (val_w/3 + 2), y, val_w/3 - 2, row_h };
        ui_.NumberField(fr, std::string("scl") + axes[a], scl[a], 0.01f, 0.01f, 100.f, 3);
    }
    y += row_h + 6;

    // --- body stats -------------------------------------------------------
    if (o->HasBody()) {
        const PSFBody& b = o->Body();
        char buf[256];
        ui_.Text(Rect{ content.x, y, content.w, row_h }, "geometry", th.fg_accent);
        y += row_h;
        std::snprintf(buf, sizeof(buf), "patches: %zu", b.Patches().size());
        ui_.Text(Rect{ content.x, y, content.w, row_h }, buf, th.fg_dim);
        y += row_h;
        std::snprintf(buf, sizeof(buf), "anchors: %zu", b.AnchorBlob().size());
        ui_.Text(Rect{ content.x, y, content.w, row_h }, buf, th.fg_dim);
        y += row_h;
        const Vec3 mn = b.BBoxMin(), sp = b.BBoxSpan();
        std::snprintf(buf, sizeof(buf), "bbox: %.2f %.2f %.2f", sp.x, sp.y, sp.z);
        ui_.Text(Rect{ content.x, y, content.w, row_h }, buf, th.fg_dim);
        y += row_h;
        if (o->Fragment()) {
            ui_.Text(Rect{ content.x, y, content.w, row_h }, "fractured", th.fg_warn);
            y += row_h;
        }
        y += 4;
    }

    // --- viewpoints --------------------------------------------------------
    //
    // The user asked to be able to "view from multiple cameras". Each
    // viewpoint stores its own orbit, so switching between them does not
    // lose the framing -- and each can be reached from the keyboard.
    {
        ui_.Text(Rect{ content.x, y, content.w, row_h }, "viewpoints", th.fg_accent);
        y += row_h;
        for (size_t i = 0; i < viewpoints_.size(); ++i) {
            const Rect vr{ content.x, y, content.w - 46, row_h };
            bool active = (int(i) == active_view_);
            if (ui_.ToggleButton(vr, viewpoints_[i].name, &active))
                GotoViewpoint(int(i));
            // A "go" button per viewpoint, so one click restores it.
            const Rect gb{ content.x + content.w - 42, y, 38, row_h };
            if (ui_.Button(gb, "go")) GotoViewpoint(int(i));
            y += row_h;
        }
        // Save the current framing under a name.
        const Rect sr{ content.x, y, content.w - 52, row_h };
        ui_.TextField(sr, "view_name", &view_name_, "view name");
        const Rect sb{ content.x + content.w - 48, y, 44, row_h };
        if (ui_.Button(sb, "save")) {
            if (!view_name_.empty()) {
                SaveViewpoint(view_name_);
                view_name_.clear();
            }
        }
        y += row_h + 4;
    }

    // --- material ---------------------------------------------------------
    ui_.Text(Rect{ content.x, y, content.w, row_h }, "material", th.fg_accent);
    y += row_h;
    Material& m = o->Mat();
    ui_.Text(Rect{ content.x, y, 46, row_h }, "density", th.fg_dim);
    ui_.NumberField(Rect{ content.x + 48, y, content.w - 48, row_h }, "density",
                    &m.density, 10.f, 0.1f, 20000.f, 1);
    y += row_h;
    ui_.Text(Rect{ content.x, y, 46, row_h }, "young", th.fg_dim);
    ui_.NumberField(Rect{ content.x + 48, y, content.w - 48, row_h }, "young",
                    &m.youngs_modulus, 10000.f, 1.f, 1e9f, 0);
    y += row_h;
    // The weld category (how the body joins a neighbour) as a drop-down.
    static const std::vector<std::string> weld_names = {
        "rigid", "elastic", "soft", "fluid", "brittle" };
    int weld = int(m.weld);
    if (ui_.ComboBox(Rect{ content.x + 48, y, content.w - 48, row_h }, "weld",
                     weld_names, &weld))
        m.weld = WeldCategory(weld);
    y += row_h + 6;
}

void Workspace::DrawConsole() {
    const Theme& th = ThemeDefault();
    Rect content = ui_.Panel(layout_.console, "AI Console");

    // The prompt field on the left, the model controls on the right.
    const int btn_w = 74;
    const Rect prompt_r{ content.x + 4, content.y + 2, content.w - btn_w - 12, kRowH };
    ui_.TextField(prompt_r, "prompt", &prompt_, "describe a shape...");

    const Rect send_r{ prompt_r.Right() + 6, prompt_r.y, btn_w, kRowH };
    const bool can_send = !busy_ && !prompt_.empty();
    if (ui_.Button(send_r, busy_ ? "thinking" : "generate")) {
        if (can_send) {
            const std::string p = prompt_;
            prompt_.clear();
            AskModel(p);
        } else if (busy_) {
            llm_.Cancel();
            busy_ = false;
            Logf("cancelled");
        }
    }

    // The log fills the rest of the console.
    Rect log_r{ content.x + 4, prompt_r.Bottom() + 3, content.w - 8,
                content.Bottom() - prompt_r.Bottom() - 5 };

    // ---------------------------------------------------------------------
    // THE DOCUMENT ROW (G15).
    //
    // The console has TWO contracts, and they are mutually exclusive:
    //
    //   SEMANTIC  "make a torus"        -- the prompt IS the spec.
    //   DOCUMENT  a path to a .txt file -- the FILE is the spec, and the
    //                                       document pipeline runs (corpus,
    //                                       closed-body rule, error log,
    //                                       escalation ladder).
    //
    // The toggle is explicit rather than inferred from the text, because
    // the two paths need different prompts AND different validation: the
    // document path REJECTS a single sheet (a closed body needs several),
    // while the semantic path allows one. Inferring the mode from the text
    // would make that rejection unpredictable -- the same prompt could be
    // accepted or refused depending on how it was phrased.
    // ---------------------------------------------------------------------
    {
        const int toggle_w = 78;
        const Rect toggle_r{ content.x + 4, prompt_r.Bottom() + 3, toggle_w, kRowH };
        const char* mode_label = doc_mode_ ? "document" : "semantic";
        if (ui_.Button(toggle_r, mode_label)) {
            doc_mode_ = !doc_mode_;
            Logf("console: %s mode", doc_mode_ ? "DOCUMENT" : "semantic");
        }

        if (doc_mode_) {
            // The document path field: the path to the technical document.
            const Rect path_r{ toggle_r.Right() + 6, toggle_r.y,
                               content.w - toggle_r.w - 130, kRowH };
            ui_.TextField(path_r, "doc_path", &doc_path_, "document .txt path...");
            const Rect load_r{ path_r.Right() + 6, path_r.y, 52, kRowH };
            if (ui_.Button(load_r, "load")) {
                if (!doc_path_.empty()) LoadDocument(doc_path_);
            }
            const Rect run_r{ load_r.Right() + 6, load_r.y, 66, kRowH };
            const bool can_run = !doc_busy_ && !doc_text_.empty() && model_online_;
            if (ui_.Button(run_r, doc_busy_ ? "working" : "interpret")) {
                if (can_run) InterpretLoadedDocument();
            }
        } else {
            // The corpus readout: the learning system's memory, made
            // visible. A learning loop whose learning is invisible cannot
            // be distinguished from one that has not learned.
            const Rect corpus_r{ toggle_r.Right() + 6, toggle_r.y,
                                 content.w - toggle_r.w - 12, kRowH };
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "corpus: %zu ok / %zu failed  (domain: %s)",
                          doc_log_.SuccessCount(), doc_log_.FailureCount(),
                          doc_domain_.c_str());
            ui_.Text(corpus_r, buf, th.fg_dim);
        }
    }
    // The log starts BELOW the document row.
    log_r.y = prompt_r.Bottom() + 3 + kRowH + 4;
    log_r.h = content.Bottom() - log_r.y - 5;

    // SHOW THE STREAM WHILE THE MODEL WORKS.
    //
    // A local model can take a minute to answer. A console that shows
    // nothing until the reply is complete looks identical to a hung app,
    // and the user clicks Generate again. Echoing the partial text as it
    // streams proves the round trip is alive and is the cheapest possible
    // progress bar.
    if (busy_) {
        const std::string partial = llm_.Partial();
        if (!partial.empty()) {
            // Keep the tail: a long JSON reply does not fit on one line.
            std::string tail = partial;
            const size_t max_chars = size_t(log_r.w / 6);
            if (tail.size() > max_chars) tail = tail.substr(tail.size() - max_chars);
            ui_.Text(Rect{ log_r.x, log_r.y, log_r.w, kRowH },
                     (std::string("...") + tail).c_str(), th.fg_dim);
        }
    }

    // Pin the view to the newest line when the user has not scrolled up.
    if (ui_.mouse_y > log_r.y && ui_.mouse_y < log_r.Bottom() && ui_.wheel != 0)
        console_scroll_ += 0;  // LogView adjusts it
    // A COPY of the log: the document worker thread appends to it while the
    // main thread paints it, and LogView takes a reference.
    const std::vector<std::string> log_copy = Log();
    ui_.LogView(log_r, "console_log", log_copy, &console_scroll_);
    // Auto-follow the tail: keep the scroll pinned to the bottom.
    const int lh = BitmapFont::kGlyphH + 2;
    const int visible = std::max(1, (log_r.h - 4) / lh);
    const int max_scroll = std::max(0, int(log_copy.size()) - visible);
    if (console_scroll_ >= max_scroll - 1) console_scroll_ = max_scroll;
}

void Workspace::LayoutGizmo() {
    // The gizmo lives in the viewport's TOP-RIGHT corner, inset from the
    // edge so it never overlaps the tool strip (top-left).
    const int size = 54;
    const int margin = 8;
    gizmo_rect_.x = layout_.viewport.Right() - size - margin;
    gizmo_rect_.y = layout_.viewport.y + margin;
    gizmo_rect_.w = size;
    gizmo_rect_.h = size;
}

bool Workspace::OverGizmo(int x, int y) const {
    return gizmo_rect_.w > 0 && gizmo_rect_.Contains(x, y);
}

void Workspace::DrawGizmo() {
    const Theme& th = ThemeDefault();
    LayoutGizmo();
    if (gizmo_rect_.w <= 0) return;
    const Rect g = gizmo_rect_;
    const int cx = g.x + g.w / 2;
    const int cy = g.y + g.h / 2;
    const int r = g.w / 2 - 3;

    // A SPHERE-LIKE NAVIGATION WIDGET: a circle with a latitude/longitude
    // grid and a marker showing where the camera is looking.
    //
    // WHY IT IS DRAWN AS A SPHERE: the user asked for "an external sphere
    // like tool" to drive the rotation. A sphere reads as an orientation
    // ball (the Blender/Maya idiom) -- the marker on its surface shows the
    // view direction, so you can see where you are before you drag.
    ui_.FillRect(Rect{ g.x, g.y, g.w, g.h }, th.bg_panel);
    ui_.DrawRect(g, th.border);

    // The sphere's disc.
    for (int dy = -r; dy <= r; ++dy) {
        const int half = int(float(std::sqrt(float(r * r - dy * dy))) + 0.5f);
        ui_.FillRect(Rect{ cx - half, cy + dy, half * 2, 1 }, th.fg_dim);
    }
    // Latitude lines (ellipses seen edge-on): horizontal chords.
    for (int k = -1; k <= 1; ++k) {
        const int yoff = k * r / 2;
        const int half = int(float(std::sqrt(float(r * r - yoff * yoff))) + 0.5f);
        ui_.FillRect(Rect{ cx - half, cy + yoff, half * 2, 1 }, th.border);
    }
    // The view-direction marker: a dot on the sphere's disc at the angle
    // the camera is looking from.
    const float sy = std::sin(orbit_yaw_);
    const float sp = std::sin(orbit_pitch_);
    const int mx = cx + int(sy * float(r) * 0.82f);
    const int my = cy - int(sp * float(r) * 0.82f);
    ui_.FillRect(Rect{ mx - 3, my - 3, 6, 6 }, th.fg_accent);

    ui_.Text(Rect{ g.x, g.y + g.h - 9, g.w, 8 }, "nav", th.fg_dim, 1, true);
}

const char* Workspace::RenderPathName(int i) {
    switch (i) {
        case 0:  return "march";
        case 1:  return "raymarch";
        case 2:  return "splat";
        case 3:  return "layered";
        default: return "area";
    }
}

// Push the selected render path into the renderer.
//
// The renderer's active path is a RUNTIME value (`path_`), read by
// RenderBody every frame -- that is what makes this switch instant.
// (The old env vars -- DELIRIUM_MARCH / DELIRIUM_RAYMARCH -- are read
// once at Initialize, so setenv() at runtime does nothing. That is why
// the gallery's old setenv-based scene selection silently rendered the
// wrong path; it now calls SetPath directly.)
void Workspace::ApplyRenderPath() {
    SplatRenderer* r = engine_.GetRenderer();
    if (!r) return;
    // EXPLICIT cases, no fall-through: a `default` that maps an unknown
    // index to the newest path would silently switch a body to the wrong
    // compositor. An out-of-range index keeps the current path.
    switch (render_path_) {
        case 0:  r->SetPath(SplatRenderer::Path::March);        break;
        case 1:  r->SetPath(SplatRenderer::Path::Raymarch);     break;
        case 2:  r->SetPath(SplatRenderer::Path::SplatClassic); break;
        case 3:  r->SetPath(SplatRenderer::Path::SplatLayered); break;
        case 4:  r->SetPath(SplatRenderer::Path::SplatArea);    break;
        default: break;
    }
}

void Workspace::DrawViewportOverlay() {
    const Theme& th = ThemeDefault();
    const Rect v = layout_.viewport;
    if (v.w <= 0 || v.h <= 0) return;

    // A 1px frame around the viewport so the panel edges read clearly.
    ui_.DrawRect(v, th.border);

    // The tool strip, top-left of the viewport.
    const char* tool_names[] = { "poke", "dent", "refine", "coarsen", "fract", "move", "nav" };
    const int ntools = 7;
    const int bw = 42;
    for (int i = 0; i < ntools; ++i) {
        const Rect br{ v.x + 6 + i * (bw + 2), v.y + 6, bw, kRowH };
        bool state = (int(tool_) == i);
        if (ui_.ToggleButton(br, tool_names[i], &state)) tool_ = Tool(i);
    }

    // The strength slider, under the strip.
    const Rect sr{ v.x + 6, v.y + 6 + kRowH + 3, 120, 10 };
    ui_.Text(Rect{ v.x + 130, v.y + 6 + kRowH + 1, 70, kRowH }, "strength",
             th.fg_dim);
    ui_.Slider(sr, "strength", &tool_strength_, 0.02f, 1.0f);

    // THE RENDER PATH SELECTOR, under the strength slider.
    //
    // The renderer offers four compositing modes and reads the active one
    // EVERY FRAME (path_ is a runtime value, not an env var), so this
    // switch is immediate -- no restart. The layered splat is the
    // DEFAULT: it is the user's distance-plane method, and it beats the
    // classic splat on coverage, holes and dither.
    //
    // The buttons sit under the strength slider, left-aligned, so they
    // never overlap the nav ball (top-right) or the tool strip.
    {
        static const char* path_names[] = { "march", "raym", "splat", "layer", "area" };
        const int npaths = 5;
        const int pbw = 46;
        const int py = v.y + 6 + kRowH + 3 + 15;
        for (int i = 0; i < npaths; ++i) {
            const Rect br{ v.x + 6 + i * (pbw + 2), py, pbw, kRowH };
            bool state = (render_path_ == i);
            if (ui_.ToggleButton(br, path_names[i], &state)) {
                render_path_ = i;
                ApplyRenderPath();
            }
        }
        ui_.Text(Rect{ v.x + 6 + npaths * (pbw + 2) + 4, py, 200, kRowH },
                 "render path", th.fg_dim);
    }

    // The hint line, bottom-left of the viewport.
    //
    // The hints changed with the interaction model: the camera is now moved
    // by the GIZMO, the keyboard, or Navigate mode -- NOT by dragging in
    // the viewport, which is reserved for the sculpt tools.
    const char* hint =
        "LMB tool  |  drag the nav ball (top-right) to orbit  |  WASD move cam  |  "
        "MMB pan  |  wheel zoom";
    ui_.Text(Rect{ v.x + 6, v.Bottom() - 14, v.w - 12, kRowH }, hint, th.fg_dim);
}

// =============================================================================
// Tools
// =============================================================================
void Workspace::ApplyTool(Tool tool) {
    GameObject* o = Selected();
    if (!o || !o->HasBody()) return;
    const Vec3 p = AimPoint();
    switch (tool) {
        case Tool::Poke: {
            // An impulse along the camera forward, at the hit point.
            Scene* s = engine_.CurrentScene();
            if (!s) return;
            const Camera& cam = s->GetCamera();
            Vec3 fwd = cam.target - cam.position;
            const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
            if (fl < 1e-6f) return;
            fwd = fwd * (1.f/fl);
            o->Poke(p, fwd * (tool_strength_ * 2.f));
            Logf("poke at %.2f %.2f %.2f", p.x, p.y, p.z);
            break;
        }
        case Tool::Dent:
            o->DentLocally(p, 0.22f, tool_strength_ * 0.15f);
            Logf("dent at %.2f %.2f %.2f", p.x, p.y, p.z);
            break;
        case Tool::Refine:
            o->Body().RefineWhere(p, 0.25f, 0.05f, 24);
            Logf("refined near %.2f %.2f %.2f", p.x, p.y, p.z);
            break;
        case Tool::Coarsen:
            o->Body().CoarsenWhere(p, 0.25f, 0.02f, 24);
            Logf("coarsened near %.2f %.2f %.2f", p.x, p.y, p.z);
            break;
        case Tool::Fracture: {
            Scene* s = engine_.CurrentScene();
            if (!s) return;
            const Camera& cam = s->GetCamera();
            Vec3 fwd = cam.target - cam.position;
            const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
            if (fl < 1e-6f) return;
            fwd = fwd * (1.f/fl);
            o->Fracture(p, tool_strength_, fwd);
            Logf("fracture at %.2f %.2f %.2f", p.x, p.y, p.z);
            break;
        }
        case Tool::Move:
            // The drag handler moves the object; a click alone does nothing.
            break;
    }
}

void Workspace::AddProcedural(const char* semantic) {
    Scene* s = engine_.CurrentScene();
    if (!s) return;
    auto tree = ProceduralElementGenerator::Build(semantic);
    if (!tree) { Logf("procedural build failed"); return; }
    GameObject* o = AddFromTree(tree, semantic);
    if (o) Logf("added procedural '%s'", semantic);
}

// =============================================================================
// Input
// =============================================================================
bool Workspace::ClickReachesViewport(int x, int y) const {
    // The gizmo is a viewport ACTION (it is not a UI widget), so it wins.
    if (OverGizmo(x, y)) return true;
    // Outside the viewport: not a viewport action.
    if (!layout_.viewport.Contains(x, y)) return false;
    // Inside the viewport, but inside a UI WIDGET's rect: the UI consumed
    // the click. The widget rects are the PREVIOUS frame's (immediate mode
    // is one frame stale by design) -- so a widget that only just appeared
    // cannot veto this frame's click. That is why the map is double-buffered.
    if (ui_.HotTest(x, y)) return false;
    return true;
}

bool Workspace::OnInput(const void* ev) {
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    const SDL_Event& e = *static_cast<const SDL_Event*>(ev);

    switch (e.type) {
        case SDL_MOUSEMOTION:
            mouse_x_ = e.motion.x;
            mouse_y_ = e.motion.y;
            motion_dx_ += e.motion.xrel;
            motion_dy_ += e.motion.yrel;
            break;
        case SDL_MOUSEBUTTONDOWN:
            mouse_x_ = e.button.x;
            mouse_y_ = e.button.y;
            if (e.button.button == SDL_BUTTON_LEFT) {
                mouse_down_ = true;
                mouse_clicked_ = true;
                // The GIZMO wins over everything: it is a small sphere in
                // the viewport corner, and dragging it rotates the camera
                // without ever touching the geometry.
                if (OverGizmo(mouse_x_, mouse_y_)) {
                    gizmo_drag_ = true;
                    gizmo_last_x_ = mouse_x_;
                    gizmo_last_y_ = mouse_y_;
                    break;
                }
                // A click inside the viewport applies the tool; a click on a
                // panel is consumed by the UI pass.
                //
                // THE UI-DECIDES-FIRST RULE: the panels TOUCH the viewport's
                // edges (the tool strip, the render-path row, the console's
                // generate button), so a click aimed at a button can land 1px
                // inside the viewport rect -- and the old code applied the
                // tool AND fired the button (or started an orbit that ate the
                // button's click). The UI pass runs later in the frame, too
                // late to veto. HotTest() reads the PREVIOUS frame's widget
                // rects (immediate mode: the layout is one frame stale by
                // design), and any widget rect wins over the viewport. The
                // gizmo is still tested first -- it is not a UI widget.
                if (ClickReachesViewport(mouse_x_, mouse_y_)) {
                    if (tool_ == Tool::Navigate) {
                        // Navigation mode: the drag looks around, and NO
                        // tool is applied -- not even on release.
                        orbiting_ = true;
                        orbit_last_x_ = mouse_x_;
                        orbit_last_y_ = mouse_y_;
                    } else if (tool_ == Tool::Move) {
                        // The Move tool drags the object instead of orbiting.
                        dragging_object_ = true;
                        drag_last_x_ = mouse_x_;
                    } else {
                        // A sculpt tool: the click applies it. The camera
                        // does NOT orbit on a sculpt click -- that was the
                        // overlap the user reported ("the mouse dragging
                        // mechanic to move things around overlaps with
                        // clicking and moving the other options"). The
                        // camera is moved by the gizmo, the keyboard, or
                        // Navigate mode.
                        ApplyTool(tool_);
                    }
                }
            } else if (e.button.button == SDL_BUTTON_RIGHT) {
                mouse_right_down_ = true;
                mouse_right_clicked_ = true;
            } else if (e.button.button == SDL_BUTTON_MIDDLE) {
                mouse_middle_down_ = true;
                // A middle-button drag that STARTS on a panel must never
                // pan -- the panel owns that click. (Once a pan has
                // started over the viewport it keeps panning even if the
                // cursor wanders over a panel: a drag belongs to whoever
                // started it.)
                if (ClickReachesViewport(mouse_x_, mouse_y_)) {
                    panning_ = true;
                    pan_last_x_ = e.button.x;
                    pan_last_y_ = e.button.y;
                }
            }
            break;
        case SDL_MOUSEBUTTONUP:
            mouse_x_ = e.button.x;
            mouse_y_ = e.button.y;
            if (e.button.button == SDL_BUTTON_LEFT) {
                mouse_down_ = false;
                mouse_released_ = true;
                dragging_object_ = false;
                orbiting_ = false;
                gizmo_drag_ = false;
            } else if (e.button.button == SDL_BUTTON_RIGHT) {
                mouse_right_down_ = false;
            } else if (e.button.button == SDL_BUTTON_MIDDLE) {
                mouse_middle_down_ = false;
                panning_ = false;
            }
            break;
        case SDL_MOUSEWHEEL:
            wheel_ += e.wheel.y;
            break;
        case SDL_TEXTINPUT:
            text_input_ += e.text.text;
            break;
        case SDL_KEYDOWN: {
            const SDL_Keymod mod = SDL_GetModState();
            shift_ = (mod & KMOD_SHIFT) != 0;
            ctrl_ = (mod & KMOD_CTRL) != 0;
            alt_ = (mod & KMOD_ALT) != 0;
            switch (e.key.keysym.sym) {
                case SDLK_BACKSPACE: key_backspace_++; break;
                case SDLK_DELETE:    key_delete_++; break;
                case SDLK_LEFT:      key_left_++; break;
                case SDLK_RIGHT:     key_right_++; break;
                case SDLK_UP:        key_up_++; break;
                case SDLK_DOWN:      key_down_++; break;
                case SDLK_HOME:      key_home_++; break;
                case SDLK_END:       key_end_++; break;
                case SDLK_RETURN: case SDLK_KP_ENTER: key_enter_++; break;
                case SDLK_ESCAPE:    key_escape_++; break;
                case SDLK_TAB:       key_tab_++; break;
                // --- camera keys (held state, applied every frame) ---
                // WHY CAMERA KEYS: the user asked for the camera to be
                // moved "either by keycommands or by hand". The keys move
                // the CAMERA (not the geometry), so they can never conflict
                // with a sculpt tool or a widget.
                case SDLK_w: cam_key_fwd_ = true; break;
                case SDLK_s: cam_key_back_ = true; break;
                case SDLK_a: cam_key_left_ = true; break;
                case SDLK_d: cam_key_right_ = true; break;
                case SDLK_e: cam_key_up_ = true; break;
                case SDLK_q: cam_key_down_ = true; break;
                case SDLK_z: cam_key_roll_ = true; break;
                default: break;
            }
            // Tool hotkeys: 1..6 select the active tool, 7 is Navigate.
            //
            // WHY 7 AND NOT A MODIFIER: the user's complaint was that
            // dragging to look around collided with the tools. Holding a
            // modifier still leaves the two gestures sharing one button, so
            // the collision can still happen (and a modifier is easy to
            // forget mid-gesture). A separate mode makes "look" and "act"
            // mutually exclusive BY CONSTRUCTION.
            if (e.key.keysym.sym >= SDLK_1 && e.key.keysym.sym <= SDLK_6) {
                tool_ = Tool(int(e.key.keysym.sym - SDLK_1));
                Logf("tool: %d", int(tool_) + 1);
            } else if (e.key.keysym.sym == SDLK_7) {
                tool_ = Tool::Navigate;
                Logf("tool: navigate (camera)");
            } else if (e.key.keysym.sym >= SDLK_F1 && e.key.keysym.sym <= SDLK_F8) {
                // F1..F8 jump to the saved viewpoints. This is the keyboard
                // half of "view from multiple cameras" -- the inspector
                // buttons are the mouse half.
                const int idx = int(e.key.keysym.sym - SDLK_F1);
                if (!GotoViewpoint(idx))
                    Logf("no viewpoint %d", idx + 1);
            }
            break;
        }
        case SDL_KEYUP: {
            shift_ = ctrl_ = alt_ = false;
            switch (e.key.keysym.sym) {
                case SDLK_w: cam_key_fwd_ = false; break;
                case SDLK_s: cam_key_back_ = false; break;
                case SDLK_a: cam_key_left_ = false; break;
                case SDLK_d: cam_key_right_ = false; break;
                case SDLK_e: cam_key_up_ = false; break;
                case SDLK_q: cam_key_down_ = false; break;
                case SDLK_z: cam_key_roll_ = false; break;
                default: break;
            }
            break;
        }
        default:
            break;
    }
    // The editor consumes EVERYTHING: the engine must not quit on Escape
    // (the user may be clearing a text field) and must not move the camera.
    return true;
#else
    (void)ev;
    return false;
#endif
}

} // namespace DeliRium
