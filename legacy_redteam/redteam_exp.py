"""
Red-team numerical verification of the v2 expert-review design document.  (v2, fixed)

E1  Corrected rank claim:  rank(DF_k | active set) = m_k - 9   (old claim: <= 9)
E2  Gauge identity:        DF_k a = 0 exactly
E3  Identifiability:       N=1 -> 9-dim kernel; N>=2 distinct lights -> 1-dim (span a);
    near-identical lights -> still 1-dim but tiny eigenvalue (conditioning cliff)
E4  CRB calibration in matched world: Monte-Carlo ALS covariance vs (Q^T F Q)^{-1}
E5  Mismatch attack: near-field point lights rendered / far-field SH-9 inferred.
    Does the Fisher-based prediction of light-subset value survive mismatch?
E6  Magnitude of the near-field structural mismatch (noiseless).
"""
import time
import numpy as np
from scipy.stats import spearmanr

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

def fisher_dFk(Y, a, c):
    """Marginalized single-light Fisher DF_k = D(s)(I - P_{Jc})D(s),  P x P.
    Model mu = a * [Y c]_+:  Ja = D(s),  Jc = D(a) H Y."""
    s_full = Y @ c
    H = (s_full > 0).astype(float)
    s = np.maximum(s_full, 0.0)
    Jc = (a * H)[:, None] * Y
    Q, _ = np.linalg.qr(Jc)
    Pj = Q @ Q.T
    DF = s[:, None] * (np.eye(len(a)) - Pj) * s[None, :]
    return DF, (H > 0), s

# =================================================================
print("=" * 70); print("E1/E2/E3  (P=3000)"); print("=" * 70)
P = 3000
n = rand_dirs(P)
Y = sh_basis(n)
a = np.exp(rng.uniform(np.log(0.2), np.log(1.0), size=P))
N_L = 8
lights = np.array([sh_basis(rand_dirs(1)[0][None, :])[0] for _ in range(N_L)])

DFk, masks = [], []
for k in range(N_L):
    dF, H, s = fisher_dFk(Y, a, lights[k])
    DFk.append(dF); masks.append(H)

print("\n[E1] rank(DF_k restricted to its active set) vs claim max(m_k-9, 0):")
ok = 0
for k in range(N_L):
    idx = np.where(masks[k])[0]
    m_k = len(idx)
    Fo = DFk[k][np.ix_(idx, idx)]
    ev = np.linalg.eigvalsh(Fo)
    r = int((ev > 1e-9 * ev.max()).sum())
    claim = max(m_k - 9, 0)
    ok += (r == claim)
    print(f"  light {k}: m_k={m_k:5d}  rank={r:5d}  claim={claim:5d}  "
          f"{'OK' if r == claim else 'MISMATCH'}   old-claim(rank<=9): "
          f"{'violated' if r > 9 else 'ok'}")
print(f"  -> corrected rank claim: {ok}/{N_L}")

print("\n[E2] gauge identity  ||DF_k a|| / (||DF_k||_2 ||a||):")
worst = max(np.linalg.norm(dF @ a) / (np.linalg.norm(dF, 2) * np.linalg.norm(a))
            for dF in DFk)
print(f"  worst relative residual = {worst:.3e}")

def nullinfo(F, idx, tol_rel=1e-9):
    Fo = F[np.ix_(idx, idx)]
    ev, V = np.linalg.eigh(Fo)
    nd = int((ev <= tol_rel * ev.max()).sum())
    return nd, ev, V[:, 0]

print("\n[E3] kernel dimension on observable set O:")
for Ntest in [1, 2, 3]:
    act = np.zeros(P, bool)
    for k in range(Ntest):
        act |= masks[k]
    idx = np.where(act)[0]
    F = sum(DFk[k] for k in range(Ntest))
    nd, ev, v0 = nullinfo(F, idx)
    ao = a[act] / np.linalg.norm(a[act])
    print(f"  N={Ntest}: |O|={act.sum():5d}  nullity={nd:3d}  "
          f"cos(eigvec_min, a)={abs(v0 @ ao):.6f}")

print("\n[E3b] conditioning cliff:")
def rot(d, axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return (d * np.cos(ang) + np.cross(axis, d) * np.sin(ang)
            + axis * np.dot(axis, d) * (1 - np.cos(ang)))
d1 = rand_dirs(1)[0]
for name, d2 in [("2 deg apart", rot(d1, [1, 0, 0], np.deg2rad(2.0))),
                 ("90 deg apart", rot(d1, [0, 1, 0], np.deg2rad(90.0)))]:
    Fs, act = [], np.zeros(P, bool)
    for d in [d1, d2]:
        dF, H, s = fisher_dFk(Y, a, sh_basis(d[None, :])[0])
        Fs.append(dF); act |= H
    idx = np.where(act)[0]
    Fo = (Fs[0] + Fs[1])[np.ix_(idx, idx)]
    ev = np.linalg.eigvalsh(Fo)
    nz = ev[ev > 1e-12 * ev.max()]
    print(f"  {name:14s}: nullity={int((ev <= 1e-12*ev.max()).sum())},  "
          f"lambda_min/lambda_max = {nz[0]/nz[-1]:.3e}")

# =================================================================
print("\n" + "=" * 70); print("E4  CRB calibration (matched world)"); print("=" * 70)
P4, N4, sigma = 1000, 3, 0.02
n4 = rand_dirs(P4)
Y4 = sh_basis(n4)
a4 = np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=P4))
while True:
    cl = np.array([sh_basis(rand_dirs(1)[0][None, :])[0] for _ in range(N4)])
    S4 = np.array([np.maximum(Y4 @ c, 0) for c in cl])
    M4 = S4 > 0
    if np.all(M4.any(axis=0)) and M4.sum(1).min() > 300:
        break
F4 = np.zeros((P4, P4))
for k in range(N4):
    dF, _, _ = fisher_dFk(Y4, a4, cl[k])
    F4 += dF
U = np.linalg.svd(a4[:, None], full_matrices=True)[0]     # U[:,1:]: basis of a-perp
Qa = U[:, 1:]
crb_diag = sigma**2 * np.diag(np.linalg.inv(Qa.T @ F4 @ Qa))

def als_fit(obs, iters=150):
    a_est = np.ones(P4)
    s_est = np.maximum(obs, 1e-3)
    for _ in range(iters):
        c_new = np.zeros((N4, 9))
        for k in range(N4):
            mk = s_est[k] > 0
            c_new[k] = np.linalg.lstsq(Y4[mk] * a_est[mk, None], obs[k][mk], rcond=None)[0]
        s_est = np.maximum(c_new @ Y4.T, 0)
        num = (s_est * obs).sum(0)
        a_est = np.maximum(num / ((s_est**2).sum(0) + 1e-12), 1e-6)
    return a_est

TRIALS = 200
E = np.zeros((TRIALS, P4))
for t in range(TRIALS):
    obs = a4 * S4 + rng.normal(0, sigma, size=(N4, P4))
    ah = als_fit(obs)
    E[t] = ((ah @ a4) / (ah @ ah)) * ah - a4
mc_diag = np.diag(np.cov(Qa.T @ E.T))
ratio = mc_diag / crb_diag
print(f"  trials={TRIALS}  sigma={sigma}  P={P4}  N={N4}")
print(f"  median MC/CRB variance ratio = {np.median(ratio):.3f}")
print(f"  fraction in [0.8,1.25]: {np.mean((ratio>0.8)&(ratio<1.25)):.3f}   "
      f"IQR [{np.percentile(ratio,25):.3f}, {np.percentile(ratio,75):.3f}]")

# =================================================================
print("\n" + "=" * 70); print("E5/E6  mismatch experiment (sphere, near-field lights)"); print("=" * 70)
P5 = 1800
dirs = rand_dirs(P5, zmin=0.20)
R, Dist = 0.8, 2.2
X5 = R * dirs
n5 = dirs
Y5 = sh_basis(n5)
a5 = np.clip(0.55 + 0.30 * np.sin(2*np.pi*dirs[:,0]) * np.cos(2*np.pi*dirs[:,1]), 0.15, 0.95)

K = 16
az = np.linspace(0, 2*np.pi, K, endpoint=False) + 0.1
el = np.linspace(np.deg2rad(20), np.deg2rad(70), K)
d5 = np.stack([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], 1)

def render_near(noise=0.005):
    Lp = Dist * d5
    diff = Lp[None, :, :] - X5[:, None, :]
    dist2 = (diff**2).sum(-1)
    ldir = diff / np.sqrt(dist2)[..., None]
    ndl = np.maximum((n5[:, None, :] * ldir).sum(-1), 0)
    return a5[:, None] * ndl / dist2 + rng.normal(0, noise, (P5, K))

def render_far(noise=0.005):
    ndl = np.maximum(Y5 @ sh_basis(d5).T, 0)   # exactly in model class
    return a5[:, None] * ndl / Dist**2 + rng.normal(0, noise, (P5, K))

def als_uncal(obs, iters=120):
    Kc = obs.shape[1]
    a_est = np.ones(P5)
    s_est = np.maximum(obs, 1e-4)
    for _ in range(iters):
        c_est = np.zeros((Kc, 9))
        for k in range(Kc):
            mk = s_est[:, k] > 0
            if mk.sum() >= 9:
                c_est[k] = np.linalg.lstsq(Y5[mk] * a_est[mk, None], obs[mk, k], rcond=None)[0]
        s_est = np.maximum(c_est @ Y5.T, 0).T
        a_est = np.maximum((s_est * obs).sum(1) / ((s_est**2).sum(1) + 1e-9), 1e-6)
    return a_est, c_est

def albedo_err(ah, a_true):
    t = (ah @ a_true) / (ah @ ah)
    return np.linalg.norm(t * ah - a_true) / np.linalg.norm(a_true)

def fisher_pred(ck, ak, sub):
    F = np.zeros((P5, P5)); act = np.zeros(P5, bool)
    for j, k in enumerate(sub):
        dF, H, s = fisher_dFk(Y5, ak, ck[j])
        F += dF; act |= H
    idx = np.where(act)[0]
    ev = np.linalg.eigvalsh(F[np.ix_(idx, idx)])
    ev = ev[ev > 1e-10 * ev.max()]
    return (1.0 / ev).sum()                    # A-optimal proxy (lower = predicted better)

def world_experiment(render_fn, tag):
    obs = render_fn()
    a_full, _ = als_uncal(obs)
    print(f"  [{tag}] full-16 albedo rel.err = {albedo_err(a_full, a5):.4f}")
    Es = [albedo_err(als_uncal(obs[:, [k]])[0], a5) for k in range(K)]
    print(f"  [{tag}] single-light err: min={min(Es):.3f} max={max(Es):.3f}")
    rng_sub = np.random.default_rng(7)
    subs = [list(rng_sub.choice(K, 3, replace=False)) for _ in range(24)]
    sE, sG = [], []
    for sub in subs:
        ak, ck = als_uncal(obs[:, sub])
        sE.append(albedo_err(ak, a5))
        sG.append(fisher_pred(ck, ak, sub))
    rho = spearmanr(sG, sE).statistic
    print(f"  [{tag}] 24 subsets x3 lights: Spearman rho(Fisher-pred, actual err) = {rho:+.3f}")
    return rho

rho_m = world_experiment(render_far, "matched  far-field")
rho_x = world_experiment(render_near, "mismatch near-field")

obs_nn = render_near(noise=0.0)
a_nn, _ = als_uncal(obs_nn)
obs_fn = render_far(noise=0.0)
a_fn, _ = als_uncal(obs_fn)
print(f"\n[E6] noiseless: near-field fit by far-field model err = {albedo_err(a_nn, a5):.4f}"
      f"   (matched reference = {albedo_err(a_fn, a5):.4f})")

print(f"\nSummary: matched rho={rho_m:+.3f}, mismatched rho={rho_x:+.3f}, "
      f"E6 mismatch err vs matched = {albedo_err(a_nn,a5):.4f} vs {albedo_err(a_fn,a5):.4f}")
print(f"[total runtime {time.time()-t0:.0f}s]")
