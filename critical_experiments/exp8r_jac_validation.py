# 临时验证: 解析 Jacobian vs 中心差分 FD(数值一致性) + 耗时对比
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp8r_diligent_discrimination_v3 as m

def fd_jac(fun, x, h=1e-6):
    """自实现中心差分(本机 scipy 1.17+ 已移除 approx_derivative)。"""
    m = fun(x).size
    J = np.zeros((m, x.size))
    for j in range(x.size):
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        J[:, j] = (fun(xp) - fun(xm)) / (2 * h)
    return J

RNG = np.random.default_rng(20260906)
d = Path("D:/data/DiLiGenT/pmsData/ballPNG")
dirs, n_gt_full, I_norm, _mask = m.load_object(d)   # I_norm: (96,P) 已归一, n_gt_full: (P,3)

# 子集: N=3 固定种子抽光; P=300 固定种子抽像素
sel = RNG.choice(96, 3, replace=False)
idx = RNG.choice(n_gt_full.shape[0], 300, replace=False)
n_k = n_gt_full[idx]
I_k = I_norm[sel][:, idx]
dirs_sub = dirs[sel]
rho_full, a_, b_, _ = m.calibrate(n_k, dirs_sub, I_k)
P, N = 300, 3

# 与 joint_trf 相同的 x0 构造
x0 = np.concatenate([rho_full * 0.8, np.column_stack([
    np.array([np.mean(I_k[k][I_k[k] > 0]) / max(np.mean(rho_full), 1e-6) for k in range(N)]),
    dirs_sub[:, :2]]).ravel()])

def residual_of(x):
    rho = x[:P]
    parms = x[P:].reshape(N, 3)
    alphas = parms[:, 0]
    xy = parms[:, 1:]
    z = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
    dirs = np.column_stack([xy, z])
    nl = np.clip(n_k @ dirs.T, 0, None)
    I_mod = rho[:, None] * (alphas[None, :] * np.ones((P, N))) * nl
    return (I_k.T - I_mod).T.ravel()

worst = 0.0
for tag, x in [("x0", x0),
               ("扰动1", x0 + RNG.normal(0, 0.03, x0.size)),
               ("扰动2", x0 + RNG.normal(0, 0.1, x0.size))]:
    J_fd = fd_jac(residual_of, x)
    J = m.jac_r_joint(x, P, n_k)
    denom = np.maximum(np.abs(J_fd).max(0), 1.0)
    rel = np.abs(J - J_fd).max(0) / denom
    print(f"[{tag}] 最差列相对误差 = {rel.max():.3e} (列均值 {rel.mean():.3e})")
    worst = max(worst, rel.max())

res = m.least_squares(residual_of, x0, method='trf', max_nfev=60, verbose=0)
rho = res.x[:P]; parms = res.x[P:].reshape(N, 3); alphas = parms[:, 0]; xy = parms[:, 1:]
z = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
nl = np.clip(n_k @ np.column_stack([xy, z]).T, 0, None)
m0 = (nl > 0).astype(float)
J = np.zeros((N * P, P + 3 * N))
for k in range(N):
    rows = slice(k * P, (k + 1) * P)
    J[rows, np.arange(P)] = -alphas[k] * nl[:, k]
    J[rows, P + 3 * k] = -rho * nl[:, k]
    J[rows, P + 3 * k + 1] = -rho * alphas[k] * (n_k[:, 0] - (xy[k, 0] / z[k]) * n_k[:, 2]) * m0[:, k]
    J[rows, P + 3 * k + 2] = -rho * alphas[k] * (n_k[:, 1] - (xy[k, 1] / z[k]) * n_k[:, 2]) * m0[:, k]
J = m.jac_r_joint(res.x, P, n_k)
J_fd = fd_jac(residual_of, res.x)
rel = np.abs(J - J_fd).max(0) / np.maximum(np.abs(J_fd).max(0), 1.0)
print(f"[解 x*] 最差列相对误差 = {rel.max():.3e}")
worst = max(worst, rel.max())
print("阈值 1e-5:", "PASS" if worst < 1e-5 else "FAIL")

# 耗时对比: 同子集同样 60 nfev
os.environ["EXP8R_JAC"] = "fd"
t0 = time.time(); res_fd = m.joint_trf(I_k, rho_full * 0.8, dirs_sub, n_k, a_, b_)
t_fd = time.time() - t0
os.environ["EXP8R_JAC"] = "analytic"
t0 = time.time(); res_an = m.joint_trf(I_k, rho_full * 0.8, dirs_sub, n_k, a_, b_)
t_an = time.time() - t0
print(f"耗时: FD={t_fd:.1f}s  analytic={t_an:.1f}s  加速={t_fd/max(t_an,1e-9):.1f}x")
print(f"cost: FD={res_fd.cost:.6f}  analytic={res_an.cost:.6f}  最优性差异={abs(res_fd.cost-res_an.cost)/(res_fd.cost+1e-12):.2e}")
print(f"nfev: FD={res_fd.nfev}  analytic={res_an.nfev}")