"""
Third (final) red-team round: attack the expert round-7 "pre-freeze theory
closure" package.  Every new mathematical claim is verified numerically.

V1  Proposition 1 (affine-Gaussian finite-sample identity):
    y = A x + B dc + eps, dc ~ N(0, Sum_c), eps ~ N(0, s2 I)
    joint MAP estimator  x^ = DF^{-1} A^T M y  with  DF = A^T M A,
    M = I - B(B^T B + Lam)^{-1} B^T,  Lam = s2 * Sum_c^{-1}
    => marginal Cov(x^) = s2 * DF^{-1}  EXACTLY (MC over joint ensemble);
       fixed-dc conditional ensemble shows BIAS with unchanged variance
       (validates the Gate-B "joint sampling" requirement quantitatively)
V1b unit test: joint block-diagonal nuisance DF == sum of per-image DF_k
V2  Gauge Rayleigh closed form:  a^T DF(lam) a = sum_i alpha_i^2 s_i^2 lam/(s_i^2+lam),
    B^T B = V diag(s^2) V^T, alpha = V^T c_bar;  limits: linear lifting (lam->0),
    saturation to ||A a||^2 (lam->inf)
V3  Retention spectrum R = F_inf^{-1/2} DF F_inf^{-1/2}:  0 <= rho_j <= 1;
    closed-form gauge retention curve rho_gauge(lam) == measured Rayleigh quotient;
    observability crossover lam_star measured vs closed-form-only prediction
V4  Parameterization invariance:  c-space prior Sum_c = J_phi Sum_phi J_phi^T
    (rank-3) vs phi-space prior give IDENTICAL DF (machine precision, via direct
    marginal inverse);  PITFALL demo: lam*I in raw SH coords != lam*I in phi coords
V5  Lam = s2 * Sum_c^{-1} scaling: (s2, Sum_c) -> (t*s2, t^2 Sum_c) leaves the
    retention spectrum invariant;  changing s2 alone at fixed Sum_c shifts the
    continuum position (quantified)
V6  Mode tracking: eigenvector rotation of R(lam) between lam=1e-2 and lam=1e2
    for N=1 (expect ~0: U-basis is lam-invariant) vs N=3 (rotation quantified)
"""
import time
import numpy as np

t0 = time.time()
rng = np.random.default_rng(20260907)


def sh_basis(n):
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    A0, A1, A2, A3, A4 = 0.282095, 0.488603, 1.092548, 0.315392, 0.546274
    return np.stack([A0 * np.ones_like(x), A1 * y, A1 * z, A1 * x,
                     A2 * x * y, A2 * y * z, A3 * (3 * z**2 - 1), A2 * x * z,
                     A4 * (x**2 - y**2)], axis=1)


def rand_dirs(n, zmin=0.15):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2]) + zmin
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def make_scene(P, N):
    n = rand_dirs(P)
    Y = sh_basis(n)
    a = np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=P))
    dirs = [rand_dirs(1)[0] for _ in range(N)]
    ss, Bs, cs = [], [], []
    for d in dirs:
        c = sh_basis(d[None, :])[0]
        s_full = Y @ c
        H = (s_full > 0).astype(float)
        ss.append(np.maximum(s_full, 0.0))
        Bs.append((a * H)[:, None] * Y)   # B_k = D(a) H Y   (P x 9)
        cs.append(c)
    return Y, a, ss, Bs, cs, dirs


def DeltaF(s, B, Lam):
    """A^T [I - B(B^T B + Lam)^{-1} B^T] A with A = D(s); lstsq path only."""
    G = B.T @ B + Lam
    w = np.linalg.lstsq(G, B.T, rcond=None)[0]
    M = np.eye(len(s)) - B @ w
    return s[:, None] * M * s[None, :]


# =====================================================================
print("=" * 72)
print("V1  Proposition 1: finite-sample covariance identity + Gate-B bias")
print("=" * 72)
P, N, sigma, sig_c = 1200, 3, 0.02, 0.05
Y, a, ss, Bs, cs, dirs = make_scene(P, N)
act = np.zeros(P, bool)
for s in ss:
    act |= s > 0
idx = np.where(act)[0]
O = len(idx)
s_l = [s[idx] for s in ss]
B_l = [B[idx] for B in Bs]
A_l = [np.diag(s) for s in s_l]
a_O = a[idx]
A_st = np.vstack(A_l)                       # (3O, O)
B_blk = np.zeros((N * O, N * 9))            # block-diagonal: image k carries its own dc_k
for k in range(N):
    B_blk[k * O:(k + 1) * O, k * 9:(k + 1) * 9] = B_l[k]
Lam = (sigma / sig_c) ** 2 * np.eye(9)      # per-image Lambda = s2 * Sum_c^{-1}
Lam_blk = np.kron(np.eye(N), Lam)
G = B_blk.T @ B_blk + Lam_blk
Ginv_Bt = np.linalg.lstsq(G, B_blk.T, rcond=None)[0]        # (27, 3O)
M = np.eye(N * O) - B_blk @ Ginv_Bt
DF = A_st.T @ M @ A_st                                     # (O, O)
DFinv = np.linalg.inv(DF)
print(f"  |O|={O},  cond(DF) = {np.linalg.cond(DF):.3e}  (gauge lifted by full-rank prior)")

# V1b: joint block == sum of per-image
DF_sum = np.zeros((O, O))
for s, B in zip(s_l, B_l):
    DF_sum += DeltaF(s, B, Lam)
print(f"  [V1b] joint vs sum of per-image DF: rel err = "
      f"{np.linalg.norm(DF - DF_sum) / np.linalg.norm(DF):.2e}")

TRIALS = 2000
dc_all = rng.normal(0, sig_c, size=(TRIALS, N, 9))
E = np.empty((TRIALS, O))
for t in range(TRIALS):
    dcst = dc_all[t].reshape(-1)                          # stacked (27,)
    y = A_st @ a_O + B_blk @ dcst + rng.normal(0, sigma, size=N * O)
    My = y - B_blk @ (Ginv_Bt @ y)
    E[t] = DFinv @ (A_st.T @ My) - a_O
Cov_mc = np.cov(E.T)
crb_diag = sigma ** 2 * np.diag(DFinv)
ratio = np.diag(Cov_mc) / crb_diag
print(f"  [V1 marginal ensemble, {TRIALS} trials] Cov(x^)_diag / (s2*DF^-1)_diag:")
print(f"    median = {np.median(ratio):.4f}   IQR = "
      f"[{np.percentile(ratio, 25):.4f}, {np.percentile(ratio, 75):.4f}]")
print(f"    rel Frobenius err of full Cov: "
      f"{np.linalg.norm(Cov_mc - sigma**2 * DFinv) / np.linalg.norm(sigma**2 * DFinv):.4f}")

# fixed-dc conditional ensemble: same variance, nonzero bias
dc_fix = dc_all[0].reshape(-1)
bias = DFinv @ A_st.T @ M @ (B_blk @ dc_fix)
Ef = np.empty((TRIALS, O))
for t in range(TRIALS):
    y = A_st @ a_O + B_blk @ dc_fix + rng.normal(0, sigma, size=N * O)
    My = y - B_blk @ (Ginv_Bt @ y)
    Ef[t] = DFinv @ (A_st.T @ My) - a_O
print(f"  [V1 fixed-dc ensemble] mean error norm / bias norm = "
      f"{np.linalg.norm(Ef.mean(0)) / np.linalg.norm(bias):.4f}  (1 => bias explained)")
print(f"    conditional bias ||bias||/||a_O|| = {np.linalg.norm(bias) / np.linalg.norm(a[idx]):.3e}"
      f"   marginal mean ||E.mean||/||bias|| = {np.linalg.norm(E.mean(0)) / np.linalg.norm(bias):.3e}")
cond_pred = sigma ** 2 * np.diag(DFinv @ (A_st.T @ (M @ M) @ A_st) @ DFinv)
ratio_f = np.diag(np.cov(Ef.T)) / cond_pred
print(f"    conditional variance vs sigma^2 * DF^-1 A^T M^2 A DF^-1: median ratio = {np.median(ratio_f):.4f}")
print(f"    (conditional var < marginal var: M is a contraction with Lam>0, not idempotent)")

# =====================================================================
print("\n" + "=" * 72)
print("V2  gauge Rayleigh closed form + limits")
print("=" * 72)
s, B, c = s_l[0], B_l[0], cs[0]
i1 = np.where(s > 0)[0]
s, B, a1 = s[i1], B[i1], a_O[i1]           # restrict to this lamp's own active set
cbar = -c                                    # gauge nuisance direction
GtB = B.T @ B
ev9, V9 = np.linalg.eigh(GtB)
alpha = V9.T @ cbar
lhs, rhs = [], []
lams = np.logspace(-6, 6, 25)
for lam in lams:
    dF = DeltaF(s, B, lam * np.eye(9))
    lhs.append(a1 @ dF @ a1)
    rhs.append(np.sum(alpha**2 * ev9 * lam / (ev9 + lam)))
lhs, rhs = np.array(lhs), np.array(rhs)
rel = np.abs(lhs - rhs) / np.abs(rhs)
print(f"  max rel err over 25-decade lam grid: {rel.max():.2e}")
sat = np.linalg.norm(s * a1) ** 2        # ||A a||^2 = ||B c_bar||^2
print(f"  saturation: r(lam=1e6)/||Aa||^2 = {lhs[-1] / sat:.6f}  (theory sum alpha_i^2 s_i^2 = {sat:.4f})")
pos = ev9 > 1e-10 * ev9.max()
print(f"  linear lifting: r(1e-6)/1e-6 = {lhs[0] / 1e-6:.4f},  "
      f"sum alpha_i^2 (s_i>0) = {np.sum((alpha**2)[pos]):.4f}")

# =====================================================================
print("\n" + "=" * 72)
print("V3  retention spectrum: bounds, closed-form gauge curve, crossover")
print("=" * 72)
Finf = np.diag(s ** 2)                       # calibrated single-image Fisher on O
Finv_h = np.diag(1.0 / s)                    # F_inf^{-1/2} = diag(1/s)  (F_inf = diag(s^2))
for lam in [1e-4, 1e-2, 1.0, 1e2]:
    dF = DeltaF(s, B, lam * np.eye(9))
    R = Finv_h @ dF @ Finv_h
    rho = np.linalg.eigvalsh(R)
    ok = (rho.min() > -1e-12) and (rho.max() < 1 + 1e-12)
    print(f"  lam={lam:8.0e}: rho in [{rho.min():.6f}, {rho.max():.6f}]  bounds ok: {ok}")
rho_g = lhs / sat                            # closed-form gauge retention curve
meas = []
for lam in lams:
    dF = DeltaF(s, B, lam * np.eye(9))
    meas.append((a1 @ dF @ a1) / (a1 @ Finf @ a1))
meas = np.array(meas)
print(f"  closed-form rho_gauge(lam) vs measured Rayleigh: max rel err = "
      f"{np.abs(rho_g - meas).max() / meas.max():.2e}")
# retention-metric check: gauge deficit vs weakest-mode deficit never cross at 1/lambda order
d_gauge = np.sum(alpha**2 * ev9**2) / np.sum(alpha**2 * ev9)   # alpha-weighted mean s^2
print(f"  retention-metric deficit coefficients: gauge {d_gauge:.4f} vs weakest mode "
      f"{ev9.max():.4f}  -> rho_gauge > rho_min for large lam (no crossover in R-metric)")
# crossover in ABSOLUTE info units (round-2 setting): gauge curve vs intrinsic floor
floor_abs = np.linalg.eigvalsh(DeltaF(s, B, np.zeros((9, 9))))[0] * 0  # placeholder
M0 = DeltaF(s, B, np.full((9, 9), 1e-12))
ev0 = np.linalg.eigvalsh(M0)
floor_abs = ev0[ev0 > 1e-10 * ev0.max()][0]                    # intrinsic weakest-mode info
cross = rho_g * sat >= floor_abs                               # gauge info vs floor (absolute)
lam_star = lams[np.argmax(cross)] if cross.any() else np.nan
lam_star_pred = floor_abs / np.sum((alpha**2)[ev9 > 1e-10 * ev9.max()])   # linear-regime predictor
print(f"  absolute-unit crossover: floor = {floor_abs:.3e};  measured lam_star = {lam_star:.3e},"
      f"  linear predictor floor/sum(alpha^2) = {lam_star_pred:.3e},  ratio = {lam_star_pred/lam_star:.2f}")

# =====================================================================
print("\n" + "=" * 72)
print("V4  parameterization invariance + raw-coordinate pitfall")
print("=" * 72)
P4 = 600
Y4, a4, ss4, Bs4, cs4, dirs4 = make_scene(P4, 1)
s4, B4, c4 = ss4[0], Bs4[0], cs4[0]
i4 = np.where(s4 > 0)[0]
s4, B4 = s4[i4], B4[i4]
A4 = np.diag(s4)


def J_phi(c, d, eps=1e-6):
    def rot(dd, axis, ang):
        axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
        return dd * np.cos(ang) + np.cross(axis, dd) * np.sin(ang) \
            + axis * np.dot(axis, dd) * (1 - np.cos(ang))
    dth = rot(d, [0, 1, 0], eps)
    dph = rot(d, [0, 0, 1], eps)
    return np.stack([c, (sh_basis(dth[None, :])[0] - sh_basis(d[None, :])[0]) / eps,
                     (sh_basis(dph[None, :])[0] - sh_basis(d[None, :])[0]) / eps], axis=1)


Jp = J_phi(c4, dirs4[0])
Sig_phi = np.diag(np.array([0.05, 0.10, 0.10]) ** 2)   # (log-I, theta, phi) prior
Sig_c = Jp @ Sig_phi @ Jp.T                            # rank-3 in 9-dim SH space
s2 = 0.02 ** 2
# (a) phi-space profiled form
Lphi = s2 * np.linalg.inv(Sig_phi)
dF_phi = DeltaF(s4, B4 @ Jp, Lphi)
# (b) c-space via direct marginal inverse (no pinv subtleties)
Marg = s2 * np.eye(len(s4)) + B4 @ Sig_c @ B4.T        # s2 I + B Sum_c B^T
dF_c = s2 * (A4 @ np.linalg.solve(Marg, A4))          # A^T Marg^{-1} A, A diagonal
e_inv = np.linalg.norm(dF_phi - dF_c) / np.linalg.norm(dF_c)
print(f"  phi-space profiled DF vs c-space marginal DF: rel err = {e_inv:.2e}")
# pitfall: lam*I in raw SH coordinates vs lam*I in phi coordinates
lam_nom = 0.16
dF_raw = DeltaF(s4, B4, lam_nom * np.eye(9))
dF_phiI = DeltaF(s4, B4 @ Jp, lam_nom * np.eye(3))
dF_whit = DeltaF(s4, B4, lam_nom * np.linalg.pinv(Sig_c))   # "same lambda" via whitening
d1 = np.linalg.norm(dF_raw - dF_whit) / np.linalg.norm(dF_whit)
d2 = np.linalg.norm(dF_raw - dF_phiI) / np.linalg.norm(dF_phiI)
print(f"  PITFALL, same nominal lam={lam_nom}:")
print(f"    lam*I in raw SH coords vs whitened-correct form : rel diff = {d1:.3f}")
print(f"    lam*I in raw SH coords vs lam*I in phi coords   : rel diff = {d2:.3f}"
      f"   (coordinate-dependent => Revision #8 justified)")

# =====================================================================
print("\n" + "=" * 72)
print("V5  Lambda = s2 * Sum_c^{-1} scaling consistency")
print("=" * 72)
i5 = np.where(s_l[0] > 0)[0]
s, B = s_l[0][i5], B_l[0][i5]
Fh = np.diag(1.0 / s)                        # F_inf^{-1/2}


def retention(lam):
    return np.linalg.eigvalsh(Fh @ DeltaF(s, B, lam * np.eye(9)) @ Fh)


t = 10.0
lam1 = sigma ** 2 / sig_c ** 2
lam2 = (t ** 2 * sigma ** 2) / (t ** 2 * sig_c ** 2)
lam3 = (t ** 2 * sigma ** 2) / sig_c ** 2
r1, r2, r3 = retention(lam1), retention(lam2), retention(lam3)
print(f"  (s2, Sum_c) -> ({t:.0f}*s2, {t**2:.0f}*Sum_c): lam {lam1:.4f} -> {lam2:.4f}, "
      f"max |rho diff| = {np.abs(r1 - r2).max():.2e}  (invariance ok)")
print(f"  s2 alone x{t**2:.0f} at fixed Sum_c: lam -> {lam3:.4f} ({lam3/lam1:.0f}x shift), "
      f"max |rho diff| = {np.abs(r1 - r3).max():.3f}  (continuum position moves)")

# =====================================================================
print("\n" + "=" * 72)
print("V6  mode tracking: eigenbasis rotation of R(lam)")
print("=" * 72)


def principal_angle(V1, V2, k=9):
    U, sv, _ = np.linalg.svd(V1[:, :k].T @ V2[:, :k])
    return np.degrees(np.arccos(np.clip(sv[-1], -1, 1)))   # largest principal angle


for tag, (s_list, B_list) in [("N=1", (s_l[:1], B_l[:1])), ("N=3", (s_l, B_l))]:
    idxN = np.where(np.logical_or.reduce([sv > 0 for sv in s_list]))[0]
    sl = [sv[idxN] for sv in s_list]
    bl = [bv[idxN] for bv in B_list]
    Finf = sum(np.diag(sv ** 2) for sv in sl)
    w9, Vinf = np.linalg.eigh(Finf)
    Fh = Vinf @ np.diag(1 / np.sqrt(w9)) @ Vinf.T
    Vs = {}
    for lam in [1e-2, 1e2]:
        Ftot = sum(DeltaF(sv, bv, lam * np.eye(9)) for sv, bv in zip(sl, bl))
        Rl = Fh @ Ftot @ Fh
        _, Vl = np.linalg.eigh(Rl)
        Vs[lam] = Vl[:, :9]                     # bottom-9 (least-retained) modes
    ang = principal_angle(Vs[1e-2], Vs[1e2])
    print(f"  {tag}: largest principal angle of bottom-9 (least-retained) mode subspace, lam 1e-2 vs 1e2: "
          f"{ang:.2f} deg")

print(f"\n[total runtime {time.time() - t0:.0f}s]")
