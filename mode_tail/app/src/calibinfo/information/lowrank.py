"""M2 · low-rank routes for the per-light whitened block structure（L5）。

结构（per-light 块对角 nuisance，whitened 坐标；与 allocation 状态同构）：
    ΔF(t) = diag(F∞) − Σ_{k∈active} u_k K_k(t) u_kᵀ,  K_k = (M0k + t_k Λ0k)⁻¹
    R     = F∞^{-1/2} ΔF F∞^{-1/2} = I − V Vᵀ,
    V     = F∞^{-1/2} [u_k K_k^{1/2}]_{k∈active}   （P × 3|active| 瘦阵）

推论（L5 恒等式，tests/test_math_foundations.py 机器锁定）：
    spec(R) = {1}^{P−rank(V)} ⊕ (1 − spec(VᵀV))；
    tr ΔF⁻¹ = tr D⁻¹ + tr(G⁻¹ VᵀD⁻²V),  G = I − VᵀD⁻¹V（push-through 恒等式），
全部从 3|active|×3|active| 的小矩阵出发——稠密 P×P 路线的 15–46 倍加速，
且解锁 P ≫ 1200 的全分辨率规模（P=54600 时稠密路线需 22 GB，本路线 455 ms）。

数值纪律：
  - K_k^{1/2} 用 eigh **对称**平方根（禁 Cholesky/sqrtm——对称性进 V 的谱结构）；
  - D = diag(F∞) 要求 F∞ 严格为正（实场景满足；近零像素必须在调用方截断并记录）；
  - G = I − VᵀD⁻¹V 的 PD 性 ⇔ ΔF ≻ 0；破坏即抛错（禁静默正则化）。

来源：吸收自评估报告的 probe（results/certification/provenance/e4_convex.py，
低秩 f/grad 路线，其 f_dense 交叉校验随附）；provenance/reconciliation.json
记录了 probe 复现对账。绑定测试：tests/test_lowrank_identity.py。
"""

from __future__ import annotations

import numpy as np

from calibinfo.allocation.blocks import sym_inv


def _sym_sqrt(M: np.ndarray) -> np.ndarray:
    """对称 PSD 平方根（eigh 构造；对称性是 V 谱结构成立的前提）。"""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def _build_V(finf_diag, u, M0, lam, active, t, whiten):
    """因式分解因子（P × 3|active|）。whiten 语义必须选对：
      - whiten=True  → V_R = F∞^{-1/2}[u K^{1/2}]：R = I − V_R V_Rᵀ（谱用）；
      - whiten=False → V_Δ = [u K^{1/2}]：  ΔF = D − V_Δ V_Δᵀ（tr ΔF⁻¹ 用）。
    两者张冠李戴会产生 ~1.5% 的谱/迹偏差（v1.1 修复的实际 bug）。"""
    idx = np.flatnonzero(active)
    scale = (1.0 / np.sqrt(np.asarray(finf_diag, float))) if whiten         else np.ones_like(finf_diag)
    cols = []
    for k in idx:
        K = sym_inv(M0[k] + t[k] * lam[k])
        cols.append(scale[:, None] * (u[k] @ _sym_sqrt(K)))
    return np.concatenate(cols, axis=1), idx


def retention_spectrum_lowrank(finf_diag, u, M0, lam, active, t,
                               unit_tol=1e-9):
    """L5 恒等式全谱：spec(R) = {1}^{P−rank(V)} ⊕ (1 − spec(VᵀV))。

    返回 dict(rho (P,) 升序, n_unit（=P−rank(V)）, n_active_blocks=3|active|,
    eig_VtV (3L,))。与稠密路线的对拍见 tests/test_lowrank_identity.py
    （≤1e-10）。"""
    finf_diag = np.asarray(finf_diag, float)
    P = finf_diag.shape[0]
    V, idx = _build_V(finf_diag, u, M0, lam, active, t, whiten=True)
    mu = np.linalg.eigvalsh(V.T @ V)                     # (3L,) ∈ [0, 1]
    mu = np.clip(mu, 0.0, 1.0)
    n_unit = P - int(np.sum(mu > unit_tol))
    rho = np.sort(np.concatenate([np.full(n_unit, 1.0), 1.0 - mu[mu > unit_tol]]))
    return dict(rho=rho, n_unit=n_unit,
                n_active_blocks=int(V.shape[1]), eig_VtV=mu)


def woodbury_trace_inv(finf_diag, u, M0, lam, active, t):
    """tr ΔF(t)⁻¹（低秩 Woodbury 路线；要求 ΔF ≻ 0，即 G = I − VᵀD⁻¹V ≻ 0）。"""
    finf_diag = np.asarray(finf_diag, float)
    finf_diag = np.asarray(finf_diag, float)
    Dinv = 1.0 / finf_diag
    V, _idx = _build_V(finf_diag, u, M0, lam, active, t, whiten=False)
    Z = Dinv[:, None] * V
    G = np.eye(V.shape[1]) - V.T @ Z
    w = np.linalg.eigvalsh(0.5 * (G + G.T))
    if w.min() <= max(1.0, float(w.max())) * 1e-12:
        raise np.linalg.LinAlgError(
            f"G = I - V^T D^-1 V 数值奇异（min eig {w.min():.3e}）——"
            "ΔF 非 PD：近零 F∞ 像素须在调用方截断并记录")
    Ginv = np.linalg.inv(G)
    Z2 = (Dinv ** 2)[:, None] * V
    return float(Dinv.sum() + np.trace(Ginv @ (V.T @ Z2)))


def woodbury_trace_inv_grad(finf_diag, u, M0, lam, active, t):
    """tr ΔF⁻¹ 及其梯度（低秩路线；梯度对装配后 ΔF，FD 对拍见测试）：
    ∇_{t_k} = −tr(K_k Λ0k K_k · (u_kᵀ ΔF⁻² u_k))。"""
    finf_diag = np.asarray(finf_diag, float)
    Dinv = 1.0 / finf_diag
    idx = np.flatnonzero(active)
    V, _idx = _build_V(finf_diag, u, M0, lam, active, t, whiten=False)
    Z = Dinv[:, None] * V
    G = np.eye(V.shape[1]) - V.T @ Z
    Ginv = np.linalg.inv(G)
    Z2 = (Dinv ** 2)[:, None] * V
    f = float(Dinv.sum() + np.trace(Ginv @ (V.T @ Z2)))
    U = np.concatenate([u[k] for k in idx], axis=1)      # (P, 3L)
    Aall = Dinv[:, None] * U
    Yall = Aall + Z @ (Ginv @ (Z.T @ U))                 # ΔF⁻¹ U 的列块
    grad = np.zeros(u.shape[0])                        # 每灯一个分量（v1.1：原误用 P）
    for j, k in enumerate(idx):
        K = sym_inv(M0[k] + t[k] * lam[k])
        Y = Yall[:, 3 * j:3 * j + 3]
        grad[k] = -float(np.trace(K @ lam[k] @ K @ (Y.T @ Y)))
    return f, grad
