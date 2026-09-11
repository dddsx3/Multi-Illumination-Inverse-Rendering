"""B2 · joint_map：联合 MAP/剖面估计器（从 exp8R v3 合格估计器侧移植）。

来源：critical_experiments/exp8r_diligent_discrimination_v3.py 的 joint_trf/jac_r_joint_sparse/
als_init/calibrate/pixel_guard（合格部分：trf + 解析稀疏 Jacobian + ALS 初值 + 多起点 +
像素守卫；FD 一致性 4e-9、同最优解 cost 差 4e-11、跨机 LAE 对账 0.0002°）。
**禁止移植其 diagnose_trace()**（两处代数错误，诊断层走 diagnostics.py 的变体 A 参考实现）。

移植约束（迁移原则"公式单源化"）：数值行为与 legacy 逐位一致（同输入同种子同输出），
接口改为显式数组输入/输出；本模块不知道 DiLiGenT 路径（数据由 datasets 层喂入）。

多起点初值约束（GT 显式声明）：本模块所有函数只接受数据驱动输入；GT 参数只允许出现在
调用方的评分环节。init 断言由调用方执行（estimators.gauss_newton 提供辅助）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def calibrate(n_gt, dirs, I_norm, convention="legacy"):
    """全 96 光朗伯拟合 → ρ, 噪声模型 (a,b), 朗伯残差。

    来源：exp8r v3 calibrate（逐位一致移植）。

    convention（数学冻结 v1.0 M0-1）：np.polyfit(x, y, 1) 返回 [slope, intercept]，
    legacy（默认）保留历史错位顺序（a←slope、b←intercept；exp8r 移植逐位一致
    契约），corrected 返回 a=intercept、b=slope（与模型 Var ≈ a + b·I 一致）。
    退化守卫判定同一对象（slope < 0 → 回退 (1, 0)）。
    """
    nl = np.clip(n_gt @ dirs.T, 0, None)
    rho = (I_norm.T * nl).sum(1) / np.maximum((nl * nl).sum(1), 1e-12)
    I_hat = (rho[:, None] * nl).T
    resid = I_norm - I_hat
    sel = I_hat.ravel() > 0.01
    if sel.sum() < 100:
        a_, b_ = 1.0, 0.0
    else:
        slope, intercept = np.polyfit(I_hat.ravel()[sel],
                                      (resid.ravel() ** 2)[sel], 1)
        if slope < 0:
            a_, b_ = 1.0, 0.0
        elif convention == "legacy":
            a_, b_ = float(slope), float(intercept)     # 历史错位顺序（a←slope）
        else:
            a_, b_ = float(intercept), float(slope)     # corrected：(a, b) 按模型定义
    return rho, float(a_), float(b_), float(np.linalg.norm(resid) / np.linalg.norm(I_norm))


def pixel_guard(n_masked):
    """子集内被 mask（阴影/n·l≤0）的光 <2 → 该像素整子集剔除。

    来源：exp8r v3 pixel_guard；参数改为只吃 n_masked (P, N)（调用方先算亮光数）。
    """
    return n_masked.sum(1) >= 2


def als_init(I_sub, rho0, dirs_sub, n_gt):
    """SH-2 ALS 式粗解提方向初值（几何已知：只迭代 ρ 和方向；禁真值）。

    来源：exp8r v3 als_init（逐位一致移植）。
    """
    rho = rho0.copy()
    new_dirs = None
    for _ in range(50):
        new_dirs = []
        A = rho[:, None] * n_gt
        for k in range(dirs_sub.shape[0]):
            b = I_sub[k]
            l = np.linalg.lstsq(A, b, rcond=None)[0]
            nrm = np.linalg.norm(l)
            if nrm > 1e-9:
                new_dirs.append(l / nrm)
            else:
                new_dirs.append(dirs_sub[k])
        nl = np.clip(n_gt @ np.array(new_dirs).T, 0, None)
        rho = (I_sub.T * nl).sum(1) / np.maximum((nl * nl).sum(1), 1e-12)
    return rho, np.array(new_dirs)


def jac_r_joint_sparse(x, P, n_gt):
    """残差 r[(k,p)] 的解析稀疏 Jacobian（csr）。

    来源：exp8r v3 jac_r_joint_sparse（逐位一致移植；np.diag→显式三元组同构）。
    x = [ρ(P) | (α_k, x_k, y_k)×N]；行序 = 光主序 (k*P+p)。
    trf 对稀疏 J 走 LSMR；紧公差 tr_options 保同最优解（见 joint_map）。
    """
    from scipy import sparse
    N = (x.size - P) // 3
    rho = x[:P]
    parms = x[P:].reshape(N, 3)
    alphas = parms[:, 0]
    xy = parms[:, 1:]
    z = np.sqrt(np.maximum(1 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 1e-12))
    nl = np.clip(n_gt @ np.column_stack([xy, z]).T, 0, None)
    m0 = (nl > 0).astype(float)
    rows, cols, vals = [], [], []
    for k in range(N):
        r = np.arange(k * P, (k + 1) * P)
        rows += [r, r, r, r]
        cols += [np.arange(P),
                 np.full(P, P + 3 * k),
                 np.full(P, P + 3 * k + 1),
                 np.full(P, P + 3 * k + 2)]
        vals += [-alphas[k] * nl[:, k],
                 -rho * nl[:, k],
                 -rho * alphas[k] * (n_gt[:, 0] - (xy[k, 0] / z[k]) * n_gt[:, 2]) * m0[:, k],
                 -rho * alphas[k] * (n_gt[:, 1] - (xy[k, 1] / z[k]) * n_gt[:, 2]) * m0[:, k]]
    return sparse.csr_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N * P, P + 3 * N))


def joint_residual(x, I_sub, n_gt):
    """残差 r[(k,p)] = I_obs[k,p] − ρ_p·α_k·clip(n·l̂,0,None)（行序 = 光主序）。

    与 legacy joint_trf 内部 residual 逐位一致；导出供 FD 一致性闸与外部评分使用。
    x = [ρ(P) | (α_k, x_k, y_k)×N]，z = √(1−x²−y²)。
    """
    N = I_sub.shape[0]
    P = len(n_gt)
    rho = x[:P]
    parms = x[P:].reshape(N, 3)
    alphas = parms[:, 0]
    xy = parms[:, 1:]
    z = np.sqrt(np.maximum(1 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 1e-12))
    dirs = np.column_stack([xy, z])
    nl = np.clip(n_gt @ dirs.T, 0, None)
    I_mod = rho[:, None] * (alphas[None, :] * np.ones((P, N))) * nl
    return (I_sub.T - I_mod).T.ravel()


def joint_map(I_sub, rho0, dirs0, n_gt, n_iters=60, tight_lsmr=True):
    """trf 联合 MAP 估计 {ρ} ∪ {α_i, l̂_i}×N（exp8R v3 joint_trf 移植版）。

    参数化：l = [x, y] → z = √(1−x²−y²)；残差见 joint_residual。
    返回 dict：rho(P,) / alphas(N,) / dirs(N,3) / cost / optimality / x（原始参数向量）。

    LSMR 紧公差（legacy 实测 P=2000/N=3：与 FD 稠密路径 cost 差 4e-11 = 同一最优解）；
    tight_lsmr=False 回退默认 trf 选项（仅供诊断对比，实验路径必须用默认）。
    """
    N = I_sub.shape[0]
    P = len(n_gt)

    def unpack(x):
        rho = x[:P]
        parms = x[P:].reshape(N, 3)
        alphas = parms[:, 0]
        xy = parms[:, 1:]
        z = np.sqrt(np.maximum(1 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 1e-12))
        dirs = np.column_stack([xy, z])
        return rho, alphas, dirs

    x0 = np.concatenate([rho0, np.column_stack([
        np.array([np.mean(I_sub[k][I_sub[k] > 0]) / max(np.mean(rho0), 1e-6)
                  for k in range(N)]),
        dirs0[:, :2]]).ravel()])
    kw = dict(method="trf", max_nfev=n_iters, verbose=0)
    if tight_lsmr:
        kw["tr_options"] = {"maxiter": 500, "atol": 1e-12, "btol": 1e-12}
    res = least_squares(lambda x: joint_residual(x, I_sub, n_gt), x0,
                        jac=lambda x: jac_r_joint_sparse(x, P, n_gt), **kw)
    rho_e, alpha_e, dirs_e = unpack(res.x)
    return dict(rho=rho_e, alphas=alpha_e, dirs=dirs_e, cost=float(res.cost),
                optimality=float(res.optimality), x=res.x)
