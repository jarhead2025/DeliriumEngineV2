#include "delirium/physics/PhysicsWorld.h"
#include "delirium/psf/Seam.h"
#include <cmath>

namespace DeliRium {

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return { a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x };
}
static float dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

void PhysicsWorld::Step(float dt) {
    for (GameObject* o : objects) {
        RigidState& r = o->Rigid();
        Transform&  t = o->GetTransform();
        r.velocity.x += gravity.x*dt; r.velocity.y += gravity.y*dt;
        r.velocity.z += gravity.z*dt;
        t.position.x += r.velocity.x*dt; t.position.y += r.velocity.y*dt;
        t.position.z += r.velocity.z*dt;
        Quat& q = t.rotation;
        const Quat w{r.omega.x, r.omega.y, r.omega.z, 0};
        Quat dq{ 0.5f*( w.w*q.x + w.x*q.w + w.y*q.z - w.z*q.y),
                 0.5f*( w.w*q.y - w.x*q.z + w.y*q.w + w.z*q.x),
                 0.5f*( w.w*q.z + w.x*q.y - w.y*q.x + w.z*q.w),
                 0.5f*( w.w*q.w - w.x*q.x - w.y*q.y - w.z*q.z) };
        q.x += dq.x*dt; q.y += dq.y*dt; q.z += dq.z*dt; q.w += dq.w*dt;
        const float ql = std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
        q.x/=ql; q.y/=ql; q.z/=ql; q.w/=ql;

        std::vector<Contact> cs;
        for (const Plane& pl : planes)
            CollisionAnalytic::BodyVsPlane(o->Body(), t, pl.n, pl.d, cs);

        for (int sweep = 0; sweep < 4; ++sweep)
            for (const Contact& c : cs) {
                const Vec3 ra{ c.point.x-t.position.x,
                               c.point.y-t.position.y,
                               c.point.z-t.position.z };
                Vec3 u{ r.velocity.x + (r.omega.y*ra.z - r.omega.z*ra.y),
                        r.velocity.y + (r.omega.z*ra.x - r.omega.x*ra.z),
                        r.velocity.z + (r.omega.x*ra.y - r.omega.y*ra.x) };
                const float vn = dot(u, c.normal);
                if (vn >= 0) continue;
                const Vec3 rxn = cross(ra, c.normal);
                const float k = r.inv_mass + r.inv_inertia*dot(rxn, rxn);
                const float rest_v = 0.5f;
                const float e = (-vn > rest_v) ? r.restitution : 0.0f;
                const float j = -(1.0f + e) * vn / k;
                const Vec3 imp{ c.normal.x*j, c.normal.y*j, c.normal.z*j };
                r.velocity.x += imp.x*r.inv_mass;
                r.velocity.y += imp.y*r.inv_mass;
                r.velocity.z += imp.z*r.inv_mass;
                const Vec3 rxi = cross(ra, imp);
                r.omega.x += rxi.x*r.inv_inertia;
                r.omega.y += rxi.y*r.inv_inertia;
                r.omega.z += rxi.z*r.inv_inertia;
                Vec3 ut{ u.x - vn*c.normal.x, u.y - vn*c.normal.y,
                         u.z - vn*c.normal.z };
                const float utl = std::sqrt(dot(ut, ut));
                if (utl > 1e-6f) {
                    const float jt = std::fmin(r.friction*j, utl/k);
                    r.velocity.x -= ut.x/utl*jt*r.inv_mass;
                    r.velocity.y -= ut.y/utl*jt*r.inv_mass;
                    r.velocity.z -= ut.z/utl*jt*r.inv_mass;
                }
                if (sweep == 0 && o->Dynamics().ModeCount() > 0)
                    o->Dynamics().ApplyImpulseAt(c.patch, c.u, c.v,
                                                 -0.5f * j * r.inv_mass);
            }

        float deepest = 0;
        for (const Contact& c : cs) deepest = std::fmax(deepest, c.depth);
        if (deepest > 1e-4f) {
            const float corr = 0.8f * (deepest - 1e-4f);
            const Vec3& n = cs[0].normal;
            t.position.x += n.x*corr; t.position.y += n.y*corr;
            t.position.z += n.z*corr;
        }
    }

    for (size_t i = 0; i < objects.size(); ++i) {
        if (relax_debt[i] <= 0.0f || !objects[i]->HasBody()) continue;
        GameObject* o = objects[i];
        const Vec3 osp = o->Body().BBoxSpan();
        const float odiag = std::sqrt(osp.x*osp.x+osp.y*osp.y+osp.z*osp.z);
        WeldProfile prof = o->Mat().EffectiveWeldProfile(odiag, dt);
        const int dose = std::max(1,
            (int)std::round(weld_budget * (0.3f + prof.relax)));
        WeldProfile fp = prof; fp.iterations = dose;
        o->Body().Seams().Build(o->Body());
        o->Body().Seams().HarmonicWeld(o->Body(), fp);
        relax_debt[i] -= (0.4f + prof.relax);
        if (relax_debt[i] < 0.0f) relax_debt[i] = 0.0f;
    }
}

void PhysicsWorld::RequestWeld(GameObject* obj, float amount) {
    for (size_t i = 0; i < objects.size(); ++i)
        if (objects[i] == obj) {
            relax_debt[i] += std::fmax(0.0f, amount);
            return;
        }
}

void PhysicsWorld::PromoteToFEM(GameObject* obj, const Vec3& center,
                                float radius, int iters) {
    if (!obj) return;
    LocalFEM fem;
    fem.BindRegion(obj->Body(), center, radius);
    fem.AssembleSystem(obj->Mat());   // BUGFIX: use the object's REAL material
                                       // (was: a fresh Material with a
                                       // hardcoded youngs_modulus=5e5f,
                                       // ignoring obj->Mat() entirely)
    for (int i = 0; i < iters; ++i) fem.Step(1.0f/240.0f);
    fem.WriteBack();
}

} // namespace DeliRium
