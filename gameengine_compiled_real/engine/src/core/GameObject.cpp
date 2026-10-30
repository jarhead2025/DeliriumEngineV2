#include <cmath>
#include <cstring>
#include "delirium/core/GameObject.h"
#include <cstdio>

namespace DeliRium {

GameObject::GameObject(std::string n, IReconstructor& maker,
                       const ReconstructionRequest& req)
    : name(std::move(n)) {
    material = req.material;
    body = maker.Reconstruct(req);
    if (!body) std::fprintf(stderr, "[%s] reconstruction failed (%s)\n",
                            name.c_str(), req.semantic.c_str());
}

bool GameObject::LoadModes(const std::string& path) {
    return body && dynamics.LoadModes(path, *body);
}

void GameObject::PokeSurface(uint32_t s, float u, float v, float j) {
    dynamics.ApplyImpulseAt(s, u, v, j);
}

void GameObject::Poke(const Vec3& world_point, const Vec3& impulse) {
    if (!body || dynamics.ModeCount() == 0) return;
    SurfaceHit hit;
    if (!body->ClosestPointEx(world_point, hit)) return;
    const Vec3 n = body->Patches()[hit.patch]
                       .EvaluateNormalField(hit.u, hit.v);
    const float j = impulse.x*n.x + impulse.y*n.y + impulse.z*n.z;
    dynamics.ApplyImpulseAt(hit.patch, hit.u, hit.v, j);
}
void GameObject::DentLocally(const Vec3& p, float radius, float depth) {
    if (!body) return;
    body->RefineWhere(p, radius, radius * 0.8f);
    body->ApplyRadialDent(p, radius, depth);
}
void GameObject::Fracture(const Vec3& p, float energy, const Vec3& cam_forward) {
    if (!body || energy <= 0) return;
    SurfaceHit hit;
    if (!body->ClosestPointEx(p, hit)) return;
    auto& pool = body->AnchorsConst();
    Vec3 cen{}; uint32_t N=0;
    for (uint32_t a=0;a<pool.Count();++a){ Vec3 q=pool.Position(a);
        cen.x+=q.x; cen.y+=q.y; cen.z+=q.z; ++N; }
    if (N) { cen.x/=N; cen.y/=N; cen.z/=N; }
    // CAMERA-PARALLEL CUT PLANE (2026-10-29): the plane normal is the camera
    // forward (image-plane-parallel), not the old radial (hit.p - cen), which
    // was only camera-parallel when the object's centroid sat on the center
    // ray. The plane still passes through the midpoint (cen + (hit-cen)/2).
    // Callers pass normalize(cam.target - cam.position); the default {0,0,1}
    // keeps old behavior only if a caller omits it.
    Vec3 n = cam_forward;
    float nl = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
    if (nl < 1e-9f) {
        n = { hit.p.x-cen.x, hit.p.y-cen.y, hit.p.z-cen.z };
        nl = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
        if (nl < 1e-9f) {
            n = body->Patches()[hit.patch].EvaluateNormal(hit.u, hit.v); nl=1;
        }
    }
    n = { n.x/nl, n.y/nl, n.z/nl };
    const Vec3 mid{ cen.x + (hit.p.x-cen.x)*0.5f,
                    cen.y + (hit.p.y-cen.y)*0.5f,
                    cen.z + (hit.p.z-cen.z)*0.5f };
    const float d = mid.x*n.x + mid.y*n.y + mid.z*n.z;

    auto patch_side = [&](uint32_t pid){
        const SurfaceRecord& s = body->SurfaceBlob()[pid];
        Vec3 ctr{};
        for (int k=0;k<4;++k){ Vec3 a=pool.Position(s.anchor_idx[k]);
            ctr.x+=a.x*0.25f; ctr.y+=a.y*0.25f; ctr.z+=a.z*0.25f; }
        return (ctr.x*n.x + ctr.y*n.y + ctr.z*n.z) - d;
    };
    std::vector<uint32_t> minority;
    for (uint32_t pid=0; pid<body->SurfaceBlob().size(); ++pid)
        if (patch_side(pid) > 0) minority.push_back(pid);
    if (minority.empty() || minority.size()==body->SurfaceBlob().size())
        return;

    fragments = body->DetachPatches(minority);
    body->Seams().Build(*body);
}

void GameObject::Morph(uint32_t pattern_id, float k) {
    if (!body) return;
    if (pattern_id >= 16) return;
    auto& patches = const_cast<std::vector<Patch>&>(body->Patches());
    for (Patch& pt : patches) pt.AddColorK(pattern_id, k);
    body->MarkAnchorsDirty();
}

void GameObject::RecalculateStructure() {
    if (!body) return;
    body->Seams().Build(*body);
    body->Seams().FitAll(*body);
    body->MarkAnchorsDirty();
}
void GameObject::Update(float) {}
void GameObject::FixedUpdate(float fdt) {
    if (!body || dynamics.ModeCount() == 0) return;
    // NUEVO (verificado en aislamiento antes de aplicar aca): si el
    // material tiene una curva tension-deformacion real cargada, el
    // amortiguamiento modal se deriva de ella (fragil suena, ductil
    // amortigua) en vez de quedar fijo en el valor que psf_modes.py
    // horneo en el .modes. Si no hay curva (caso de HOY, todo objeto
    // existente), SetDampingFromMaterial es un no-op verificado -- cero
    // cambio de comportamiento para nada de lo que ya funciona.
    dynamics.SetDampingFromMaterial(material);
    dynamics.Step(fdt);
    dynamics.WriteBack(*body);
}

void NPC::SetPose(const std::vector<BoneRecord>& pose) {
    if (!body || pose.empty()) return;
    skeleton = pose;
    body->ExpandBBox(0.5f);
    auto& pool = body->Anchors();
    const uint32_t na = (uint32_t)body->AnchorBlob().size();
    for (uint32_t a=0; a<na; ++a) {
        const Vec3 ap = pool.Position(a);
        float wsum=0; Vec3 disp{};
        for (const BoneRecord& b : pose) {
            const Vec3 bind{b.bind_pos[0],b.bind_pos[1],b.bind_pos[2]};
            const float dx=ap.x-bind.x, dy=ap.y-bind.y, dz=ap.z-bind.z;
            const float dist2 = dx*dx+dy*dy+dz*dz + 1e-6f;
            const float w = 1.0f/dist2;
            float tx,ty,tz;
            std::memcpy(&tx, &b.reserved[0], 4);
            std::memcpy(&ty, &b.reserved[4], 4);
            std::memcpy(&tz, &b.reserved[8], 4);
            disp.x += w*tx; disp.y += w*ty; disp.z += w*tz; wsum += w;
        }
        if (wsum > 0) pool.SetPosition(a, { ap.x + disp.x/wsum,
            ap.y + disp.y/wsum, ap.z + disp.z/wsum });
    }
    body->MarkAnchorsDirty();
    body->Seams().Build(*body);
    std::vector<uint32_t> all(body->Patches().size());
    for (uint32_t i=0;i<all.size();++i) all[i]=i;
    body->Seams().RestitchAround(*body, all);
}
void NPC::Update(float dt) { GameObject::Update(dt); }

} // namespace DeliRium
