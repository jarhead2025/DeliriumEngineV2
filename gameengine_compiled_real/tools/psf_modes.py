#!/usr/bin/env python3
"""
psf_modes.py - Offline modal analysis on the PSF patch basis.
==============================================================
Builds a reduced deformation model directly on the format's native
degrees of freedom (the 8 corner-vanishing coefficients per patch):

  stiffness K : membrane energy  integral |grad h|^2  per patch
                + SEAM COUPLING: penalty (a_L - a_R)^2 + (b_L - b_R)^2
                on the 2 coefficients owning each shared edge
                -> eigenmodes cannot open cracks on coupled edges
  mass M      : consistent mass  integral B_i B_j  (block diagonal)

Generalized eigenproblem K phi = w^2 M phi via per-block Cholesky of M
(numpy only). Lowest n modes -> .modes file the engine integrates.

Honest approximations (documented, milestone 2 scope):
  - per-patch metric approximated by patch area (no full first
    fundamental form); modes are physically plausible, not exact FEM
  - wrap seams (duplicated anchors) and T-junctions are NOT coupled yet
    (SeamGraph milestone handles those); the validation test measures
    gaps on coupled edges

.modes layout: "PSM\0" | u32 version=1 | u32 n_modes | u32 n_surf |
               f32 zeta | f32 freq[n_modes] (rad/s) |
               f32 shapes[n_modes][n_surf*8]
"""
import struct, sys
import numpy as np

PSF_IN   = sys.argv[1] if len(sys.argv) > 1 else "assets/torus.psf"
MODES_OUT= sys.argv[2] if len(sys.argv) > 2 else "assets/torus.modes"
N_MODES  = int(sys.argv[3]) if len(sys.argv) > 3 else 12
ZETA     = 0.03          # modal damping ratio
SEAM_W   = 50.0          # seam penalty weight (vs membrane stiffness ~1)
STIFF    = 4000.0        # global stiffness scale -> audible-ish freqs

blob = open(PSF_IN, "rb").read()
_, ver, na, ns, nb, flags = struct.unpack_from("<4sIIIII", blob, 0)
assert ver == 3
A = blob[64:64+64*na]; S = blob[64+64*na:64+64*na+64*ns]
bbmin  = np.array(struct.unpack_from("<3f", blob, 24))
bbspan = np.array(struct.unpack_from("<3f", blob, 36))

apos = np.array([struct.unpack_from("<3H", A, i*64+32) for i in range(na)],
                np.float64)/65535.0*bbspan + bbmin
aidx = np.array([struct.unpack_from("<4H", S, s*64) for s in range(ns)],
                np.int64)

# ---------------------------------------------------------------------------
# element matrices on the 8-function corner-vanishing basis (Gauss 4x4,
# exact for these polynomial degrees)
# ---------------------------------------------------------------------------
def basis(u, v):
    bu, bv = u*(1-u), v*(1-v); au, av = 2*u-1, 2*v-1
    return np.array([bu*(1-v), bu*v, bv*(1-u), bv*u,
                     bu*au*(1-v), bu*au*v, bv*av*(1-u), bv*av*u])

def basis_du(u, v):
    bu, bv = u*(1-u), v*(1-v); au, av = 2*u-1, 2*v-1
    dbu = 1-2*u; dcu = dbu*au + 2*bu
    return np.array([dbu*(1-v), dbu*v, -bv, bv,
                     dcu*(1-v), dcu*v, -bv*av, bv*av])

def basis_dv(u, v):
    bu, bv = u*(1-u), v*(1-v); au, av = 2*u-1, 2*v-1
    dbv = 1-2*v; dcv = dbv*av + 2*bv
    return np.array([-bu, bu, dbv*(1-u), dbv*u,
                     -bu*au, bu*au, dcv*(1-u), dcv*u])

gp, gw = np.polynomial.legendre.leggauss(4)
gp = (gp+1)/2; gw = gw/2
Ke = np.zeros((8,8)); Me = np.zeros((8,8))
for i,(u,wu) in enumerate(zip(gp,gw)):
    for j,(v,wv) in enumerate(zip(gp,gw)):
        B, Du, Dv = basis(u,v), basis_du(u,v), basis_dv(u,v)
        Ke += wu*wv*(np.outer(Du,Du)+np.outer(Dv,Dv))
        Me += wu*wv*np.outer(B,B)

# ---------------------------------------------------------------------------
# JOINT assembly (v2): DOFs = anchor normal-displacements (na) + patch
# coefficients (ns*8). Local scalar field per patch:
#   dh(u,v) = sum_k a_k * w_k(u,v)  +  sum_i c_i * B_i(u,v)
# with w_k the bilinear corner hats. SHARED anchors couple neighboring
# patches -> true GLOBAL object modes (bending/ovalling), not per-patch
# flutter. Seam penalty still ties edge coefficient pairs.
# ---------------------------------------------------------------------------
def corner_hats(u, v):
    return np.array([(1-u)*(1-v), (1-u)*v, u*v, u*(1-v)])  # A0 A1 A2 A3
def corner_hats_du(u, v):
    return np.array([-(1-v), -v, v, (1-v)])
def corner_hats_dv(u, v):
    return np.array([-(1-u), (1-u), u, -u])

# 12x12 local element matrices over [a0..a3, c0..c7]
Ke12 = np.zeros((12,12)); Me12 = np.zeros((12,12))
for u,wu in zip(gp,gw):
    for v,wv in zip(gp,gw):
        Bv = np.concatenate([corner_hats(u,v),    basis(u,v)])
        Du = np.concatenate([corner_hats_du(u,v), basis_du(u,v)])
        Dv = np.concatenate([corner_hats_dv(u,v), basis_dv(u,v)])
        Ke12 += wu*wv*(np.outer(Du,Du)+np.outer(Dv,Dv))
        Me12 += wu*wv*np.outer(Bv,Bv)

ndof = na + ns*8
K = np.zeros((ndof,ndof)); M = np.zeros((ndof,ndof))
areas = np.zeros(ns)
for s in range(ns):
    A4 = apos[aidx[s]]
    area = 0.5*(np.linalg.norm(np.cross(A4[1]-A4[0],A4[3]-A4[0]))
              + np.linalg.norm(np.cross(A4[1]-A4[2],A4[3]-A4[2])))
    areas[s] = max(area, 1e-12)
    dof = np.concatenate([aidx[s], na + s*8 + np.arange(8)])
    K[np.ix_(dof,dof)] += STIFF*Ke12
    M[np.ix_(dof,dof)] += areas[s]*Me12

EDGES = {"v0": (0,4,(0,3)), "v1": (1,5,(1,2)),
         "u0": (2,6,(0,1)), "u1": (3,7,(3,2))}
by_pair = {}
for s in range(ns):
    for e,(c0,c1,(p,q)) in EDGES.items():
        key = tuple(sorted((int(aidx[s,p]), int(aidx[s,q]))))
        by_pair.setdefault(key, []).append((s,e,int(aidx[s,p])))
coupled = 0
for key, lst in by_pair.items():
    if len(lst) != 2: continue
    (sa,ea,starta),(sb,eb,startb) = lst
    ca0,ca1,_ = EDGES[ea]; cb0,cb1,_ = EDGES[eb]
    sgn = 1.0 if starta == startb else -1.0
    ia0, ia1 = na+sa*8+ca0, na+sa*8+ca1
    ib0, ib1 = na+sb*8+cb0, na+sb*8+cb1
    w = SEAM_W*STIFF
    K[ia0,ia0]+=w; K[ib0,ib0]+=w; K[ia0,ib0]-=w; K[ib0,ia0]-=w
    K[ia1,ia1]+=w; K[ib1,ib1]+=w; K[ia1,ib1]-=sgn*w; K[ib1,ia1]-=sgn*w
    coupled += 1
print(f"{ns} patches + {na} anchors = {ndof} DOFs, {coupled} coupled edges")

# generalized eigenproblem (M no longer block-diagonal: dense Cholesky)
M += 1e-9*np.eye(ndof)                  # numerical floor
L = np.linalg.cholesky(M)
Linv = np.linalg.inv(L)
Kt = Linv @ K @ Linv.T
Kt = 0.5*(Kt+Kt.T)
w2, Vv = np.linalg.eigh(Kt)
keep = np.where(w2 > 1e-6*w2.max())[0][:N_MODES]
freqs = np.sqrt(w2[keep])
shapes = (Linv.T @ Vv[:,keep]).T
shapes /= np.max(np.abs(shapes), axis=1, keepdims=True)

# v2 layout: "PSM\0" | u32 ver=2 | u32 n_modes | u32 n_surf | u32 n_anchor
#            | f32 zeta | freq[n_modes]
#            | per mode: f32 anchor_amp[n_anchor] then f32 coef[n_surf*8]
out  = struct.pack("<4sIIIIf", b"PSM\0", 2, len(keep), ns, na, ZETA)
out += np.asarray(freqs, np.float32).tobytes()
out += np.asarray(shapes, np.float32).tobytes()
open(MODES_OUT, "wb").write(out)
nz = float((np.abs(shapes[:, na:].reshape(len(keep), ns, 8)).max(2) > 1e-6).mean())
print(f"wrote {MODES_OUT} (v2): {len(keep)} modes, "
      f"f = {np.round(freqs/(2*np.pi),2)} Hz, "
      f"patch participation {100*nz:.0f}%")
