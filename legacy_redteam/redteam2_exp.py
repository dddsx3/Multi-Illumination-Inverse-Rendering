"""
Second red-team round: numerical verification of the calibration-confidence
continuum proposal (expert round 6).

N1  Formula (1)-(2): DeltaF(Lambda) == Schur complement of the hybrid
    information matrix == inverse of the x-block of HFIM^{-1}  (known-answer test)
N2  Proposition (3): Lambda_1 <= Lambda_2  =>  DeltaF(Lambda_1) <= DeltaF(Lambda_2)
    (Loewner), + scalar-lambda sweep monotonicity of tr F and tr F^+
N3  Spectral form (4): direct vs thin-SVD formula; rank-deficient-B case:
    plain inverse at Lambda=0 fails, pinv limit is continuous
N4  Soft gauge (5): DeltaF(Lambda)a = A^T B (B^T B + Lambda)^{-1} Lambda c_bar
    (closed form derived in this round), and the anisotropic
    Lambda_phi = diag(0, lambda_d, lambda_d) case: gauge survives direction
    calibration, lifted linearly in lambda_L
N5  The "who chooses Lambda" problem: calibrated/uncalibrated information
    ratio, and sensitivity of predictions to 4x Lambda misspecification
"""
import numpy as np

rng = np.random.default_rng(20260907)

def sh_basis(n):
    x, y, z = n[:,0], n[:,1], n[:,2]
    A0,A1,A2,A3,A4 = 0.282095,0.488603,1.092548,0.315392,0.546274
    return np.stack([A0*np.ones_like(x),A1*y,A1*z,A1*x,A2*x*y,A2*y*z,
                     A3*(3*z**2-1),A2*x*z,A4*(x**2-y**2)],axis=1)

def rand_dirs(n, zmin=0.15):
    d = rng.normal(size=(n,3)); d/=np.linalg.norm(d,axis=1,keepdims=True)
    d[:,2]=np.abs(d[:,2])+zmin
    return d/np.linalg.norm(d,axis=1,keepdims=True)

def scene(P, N, seed_rng):
    n = rand_dirs(P); Y = sh_basis(n)
    a = np.exp(seed_rng.uniform(np.log(0.3), np.log(1.0), size=P))
    lights, Bs, ss = [], [], []
    for _ in range(N):
        c = sh_basis(rand_dirs(1)[0][None,:])[0]
        s_full = Y @ c; H = (s_full > 0).astype(float)
        s = np.maximum(s_full, 0.0)
        ss.append(s); Bs.append((a*H)[:,None]*Y); lights.append(c)
    return Y, a, ss, Bs, lights

def DeltaF(s, B, Lam, pinv_tol=1e-12):
    """A^T [I - B(B^T B + Lam)^{-1} B^T] A  with A = D(s) (diag).
    Implemented as D(s) M D(s) directly (P x P)."""
    G = B.T @ B + Lam
    w = np.linalg.lstsq(G, B.T, rcond=None)[0]     # (9,P)
    M = np.eye(len(s)) - B @ w                     # I - B G^{-1} B^T
    return s[:,None] * M * s[None,:]

# ---------------------------------------------------------------- N1
print("="*70); print("N1  known-answer: DeltaF(Lam) vs hybrid-Schur vs [HFIM^-1]_xx^-1"); print("="*70)
P = 800
Y, a, ss, Bs, lights = scene(P, 1, rng)
s, B = ss[0], Bs[0]
idx0 = np.where(s > 0)[0]          # restrict to active pixels: unlit albedo unconstrained
s, B = s[idx0], B[idx0]
Lam = np.diag(rng.uniform(0.1, 2.0, size=9))
dF = DeltaF(s, B, Lam)
P = s.shape[0]
HFIM = np.zeros((P+9, P+9))
HFIM[:P,:P] = np.diag(s**2)
HFIM[:P,P:] = s[:,None]*B
HFIM[P:,:P] = (s[:,None]*B).T
HFIM[P:,P:] = B.T@B + Lam
schur = HFIM[:P,:P] - HFIM[:P,P:] @ np.linalg.inv(HFIM[P:,P:]) @ HFIM[P:,:P]
inv_xx = np.linalg.inv(np.linalg.inv(HFIM)[:P,:P])
e1 = np.linalg.norm(dF - schur)/np.linalg.norm(schur)
e2 = np.linalg.norm(dF - inv_xx)/np.linalg.norm(inv_xx)
print(f"  rel err vs Schur complement      = {e1:.2e}")
print(f"  rel err vs [HFIM^-1]_xx^{-1}     = {e2:.2e}")
print("  -> DeltaF(Lam) is exactly the hybrid (deterministic-x / Gaussian-nuisance)")
print("     information matrix; its inverse = HCRLB covariance (tight in linear-Gaussian)")

# ---------------------------------------------------------------- N2
print("\n"+"="*70); print("N2  calibration monotonicity (Loewner)"); print("="*70)
Y2, a2, ss2, Bs2, _ = scene(1200, 4, rng)
def F_eff(Lams):
    F = np.zeros((1200,1200))
    for s,B,L in zip(ss2, Bs2, Lams):
        F += DeltaF(s, B, L)
    return F
viol = 0; trials = 30
for t in range(trials):
    L1 = np.diag(rng.uniform(0, 2, size=9))
    dL = np.diag(rng.uniform(0, 2, size=9))
    E = DeltaF_sum = None
    F1 = F_eff([L1]*4); F2 = F_eff([L1+dL]*4)
    Ev = np.linalg.eigvalsh(F2 - F1)
    if Ev.min() < -1e-9 * max(abs(Ev).max(), 1e-30):
        viol += 1
print(f"  Loewner violations in {trials} random paired scenes: {viol}")
# scalar sweep
lams = np.logspace(-3, 4, 15)
trs, trinvs = [], []
act = np.zeros(1200, bool)
for s in ss2: act |= s > 0
idx = np.where(act)[0]
Linf = F_eff([np.zeros((9,9))]*4)*0 + sum(np.outer(s,s) for s in ss2)  # Lambda=inf limit
for lam in lams:
    F = F_eff([lam*np.eye(9)]*4)
    Fo = F[np.ix_(idx,idx)]
    ev = np.linalg.eigvalsh(Fo)
    ev = ev[ev > 1e-11*ev.max()]
    trs.append(ev.sum()); trinvs.append((1/ev).sum())
print(f"  tr F_eff(lambda) monotone nondecreasing: {all(trs[i+1]>=trs[i]-1e-9*trs[-1] for i in range(len(trs)-1))}")
print(f"  tr F_eff(lambda)^+ monotone nonincreasing: {all(trinvs[i+1]<=trinvs[i]+1e-9*trinvs[0] for i in range(len(trinvs)-1))}")
Fi = Linf[np.ix_(idx,idx)]; evi = np.linalg.eigvalsh(Fi); evi = evi[evi>1e-11*evi.max()]
F0 = F_eff([np.full((9,9),1e-10)]*4)[np.ix_(idx,idx)]
ev0 = np.linalg.eigvalsh(F0); ev0 = ev0[ev0>1e-11*ev0.max()]
print(f"  info ratio calibrated/uncalibrated: tr F(inf)/tr F(0) = {evi.sum()/ev0.sum():.2f}"
      f"   (how much of a calibrated image's info the unknown lighting eats)")

# ---------------------------------------------------------------- N3
print("\n"+"="*70); print("N3  spectral form (4) + rank-deficient erratum"); print("="*70)
Bf = rng.normal(size=(600,9))
for lam in [1e-3, 1.0, 100.0]:
    Mdir = np.eye(600) - Bf @ np.linalg.solve(Bf.T@Bf + lam*np.eye(9), Bf.T)
    U, sv, _ = np.linalg.svd(Bf, full_matrices=False)
    Msvd = (np.eye(600) - U@U.T) + U @ np.diag(lam/(sv**2+lam)) @ U.T
    print(f"  lam={lam:8.0e}: |M_direct - M_spectral|/|M| = "
          f"{np.linalg.norm(Mdir-Msvd)/np.linalg.norm(Mdir):.2e}")
Br = Bf[:, :4]      # rank 4 < 9: B^T B singular
try:
    np.linalg.solve(Br.T@Br, Br.T)
    print("  plain inverse at Lam=0 (rank-def B): unexpectedly succeeded")
except np.linalg.LinAlgError:
    print("  ERRATUM CONFIRMED: eq(1)(2) at Lam=0 needs (B^T B)^dag, plain inverse fails")
M0 = np.eye(600) - Br @ np.linalg.pinv(Br.T@Br) @ Br.T
Mlt = np.eye(600) - Br @ np.linalg.solve(Br.T@Br + 1e-10*np.eye(4), Br.T)
print(f"  continuity: |M(0^+ via pinv) - M(1e-10)|/|M(0)| = "
      f"{np.linalg.norm(M0-Mlt)/np.linalg.norm(M0):.2e}  (continuum is well-defined with dag)")

# ---------------------------------------------------------------- N4
print("\n"+"="*70); print("N4  soft gauge: closed form + anisotropic calibration prior"); print("="*70)
c_bar = lights[0]
Y3, a3, ss3, Bs3, lights3 = scene(1200, 3, rng)
# cleaner: per-image closed form check first
print("  per-image closed form  DeltaF(Lam)a == A^T B (B^T B + Lam)^{-1} Lam c_bar:")
worst = 0.0
for s,B,L,c in zip(ss3, Bs3, [np.diag(rng.uniform(0.1,2.0,size=9)) for _ in range(3)], lights3):
    dF = DeltaF(s,B,L)
    lhs = dF @ a3
    rhs = (s[:,None]*B) @ np.linalg.solve(B.T@B + L, L @ c)
    worst = max(worst, np.linalg.norm(lhs-rhs)/max(np.linalg.norm(rhs),1e-12))
print(f"    worst rel err = {worst:.2e}")
print("  anisotropic phi-space prior diag(lambda_L, lam_d, lam_d):")
def J_phi(c, d, eps=1e-6):
    """d dc/dphi: columns [c (log-intensity), dc/dtheta, dc/dphi_angle]."""
    d1 = d.copy()
    def rot_dir(dd, axis, ang):
        axis = np.asarray(axis,float); axis/=np.linalg.norm(axis)
        return dd*np.cos(ang)+np.cross(axis,dd)*np.sin(ang)+axis*np.dot(axis,dd)*(1-np.cos(ang))
    dth = rot_dir(d1,[0,1,0],eps); dph = rot_dir(d1,[0,0,1],eps)
    return np.stack([c, (sh_basis(dth[None,:])[0]-sh_basis(d1[None,:])[0])/eps,
                        (sh_basis(dph[None,:])[0]-sh_basis(d1[None,:])[0])/eps], axis=1)
# rebuild scene with stored directions
n3 = rand_dirs(1200); Y3 = sh_basis(n3)
a3 = np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=1200))
dirs3 = [rand_dirs(1)[0] for _ in range(3)]
ss3, Bs3, cs3 = [], [], []
for d in dirs3:
    c = sh_basis(d[None,:])[0]
    s_full = Y3 @ c; H = (s_full>0).astype(float)
    ss3.append(np.maximum(s_full,0)); Bs3.append((a3*H)[:,None]*Y3); cs3.append(c)
for lam_L in [0.0, 1e-4, 1e-2, 1.0]:
    Fsum = np.zeros((1200,1200)); gn = 0.0
    for s,B,c,d in zip(ss3, Bs3, cs3, dirs3):
        Jp = J_phi(c, d)
        Bphi = B @ Jp                      # P x 3
        Lam = np.diag([lam_L, 1.0, 1.0])   # lambda_d = 1 fixed
        dF = DeltaF(s, Bphi, Lam)
        Fsum += dF
        gn += np.linalg.norm(dF @ a3)
    Fscale = np.linalg.norm(Fsum, 2)
    print(f"    lambda_L={lam_L:8.0e}:  ||F_eff a|| / ||F_eff||_2 = {gn/Fscale:.3e}")
# smallest nonzero eigenvalue vs lambda_L on observable set
print("  gauge-mode lifting (smallest nonzero eigval of F_eff on observable set):")
act = np.zeros(1200, bool)
for s in ss3: act |= s>0
idx = np.where(act)[0]
for lam_L in [0.0, 1e-4, 1e-2, 1.0]:
    Fsum = np.zeros((1200,1200))
    for s,B,c,d in zip(ss3, Bs3, cs3, dirs3):
        Jp = J_phi(c, d); Bphi = B @ Jp
        Fsum += DeltaF(s, Bphi, np.diag([lam_L, 1.0, 1.0]))
    ev = np.linalg.eigvalsh(Fsum[np.ix_(idx,idx)])
    nz = ev[ev > 1e-11*ev.max()]
    print(f"    lambda_L={lam_L:8.0e}:  lam_min/lam_max = {nz[0]/nz[-1]:.3e}")

# ---------------------------------------------------------------- N5
print("\n"+"="*70); print("N5  who chooses Lambda? misspecification sensitivity"); print("="*70)
print("  predicted weak-mode variance  tr F_eff^+  under true vs 4x-misspecified lambda:")
for lam_true in [1e-2, 1.0, 100.0]:
    def Finv_tr(lam):
        F = np.zeros((1200,1200))
        for s,B in zip(ss3, Bs3):
            F += DeltaF(s, B, lam*np.eye(9))
        Fo = F[np.ix_(idx,idx)]
        ev = np.linalg.eigvalsh(Fo); ev = ev[ev>1e-11*ev.max()]
        return (1/ev).sum(), ev
    t_true, ev_true = Finv_tr(lam_true)
    t_miss, ev_miss = Finv_tr(lam_true*4)
    print(f"    lambda={lam_true:8.0e}: tr F^+ = {t_true:.3e};  4x misspecification changes it by "
          f"{(t_miss-t_true)/t_true*100:+.1f}%   weakest-mode var by {(1/ev_miss[0]-1/ev_true[0])/(1/ev_true[0])*100:+.1f}%")
