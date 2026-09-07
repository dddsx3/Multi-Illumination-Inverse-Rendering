"""B1 · schur：ΔF(Λ) 的唯一 Schur 实现 + 独立边缘化路线（交叉验证专用）。

模型：y = A x + B δc + ε，ε~N(0, σ²I)（先白化，见 whitening.whiten_system），
δc~N(0, Σ_c)，Λ = σ²Σ_c⁻¹。ΔF(Λ) = Aᵀ[I − B(BᵀB+Λ)⁻¹Bᵀ]A。

红线 RL-solve（宪法 §0 + 红队 N3-fix）：Λ=0/秩亏路径禁用 np.linalg.solve——
本模块统一走 SVD/lstsq（pinv 极限），并在 delta_f 内显式断言。
红线 RL-thin-svd：谱性质仅在 thin-SVD 形式下成立（零奇异方向由 I−UUᵀ 项承载）。
绑定测试：test_information_modules.py + test_rank_deficient_lambda0.py（红队 V 系列）。
"""

from __future__ import annotations

import numpy as np


def _lam_matrix(Lam, q):
    """把标量 λ / (q,) 向量 / (q,q) 矩阵统一成 (q,q)。"""
    Lam = np.asarray(Lam, dtype=float)
    if Lam.ndim == 0:
        return Lam * np.eye(q)
    if Lam.ndim == 1:
        assert Lam.shape[0] == q
        return np.diag(Lam)
    assert Lam.shape == (q, q)
    return Lam


def delta_f(A, B, Lam, rank_policy="pinv"):
    """唯一 Schur 实现：ΔF(Λ) = Aᵀ M A，M = I − B(BᵀB+Λ)⁻¹Bᵀ。

    参数
    ----
    A : (m, n) 主参数设计矩阵（x 空间）
    B : (m, q) nuisance 设计矩阵（δc 空间）
    Lam : 标量 λ（各向同性）/ (q,) 对角 / (q,q) 矩阵，Λ = σ²Σ_c⁻¹
    rank_policy : "pinv"（默认，Λ=0/秩亏自动 SVD/lstsq 极限）或
                  "raise_if_singular"（数值奇异则抛错）

    返回
    ----
    (DeltaF (n,n), M (m,m), diagnostics dict)
    diagnostics: rank（G 的数值秩）、method（实现路径）、g_min/g_max（G 特征值范围）

    红线：本函数任何路径不调用 np.linalg.solve（RL-solve；已知答案测试盯防）。
    M 为 (m,m) 稠密矩阵——CI 尺度（m≤数千）可用；更大规模走低秩/MatrixFree（卡 CI03 预算）。
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    m, q = B.shape
    assert A.shape[0] == m
    G = B.T @ B + _lam_matrix(Lam, q)
    # Λ=0 自动 SVD/pinv（lstsq 即 SVD pinv 最小范数解）；统一路径，绝无 solve
    ev = np.linalg.eigvalsh(G)
    g_min, g_max = float(ev.min()), float(ev.max())
    singular = g_min <= max(abs(g_max), 1.0) * 1e-12
    if singular and rank_policy == "raise_if_singular":
        raise np.linalg.LinAlgError(
            f"G=BᵀB+Λ 数值奇异 (g_min={g_min:.3e})；rank_policy=raise_if_singular")
    w = np.linalg.lstsq(G, B.T, rcond=None)[0]        # (q, m)：G⁻¹Bᵀ 的 pinv 极限
    M = np.eye(m) - B @ w
    DeltaF = A.T @ M @ A
    diag = dict(rank=int(np.linalg.matrix_rank(G)),
                method="lstsq_svd_pinv",
                singular=singular, g_min=g_min, g_max=g_max)
    return DeltaF, M, diag


def delta_f_marginal(A, B, Sigma_c, sigma):
    """独立 Route A：ΔF = σ²·Aᵀ(σ²I + BΣ_cBᵀ)⁻¹A（边缘协方差直接逆，乘 σ² 回
    Schur 单位——Woodbury: V⁻¹ = σ⁻²M ⇒ σ²AᵀV⁻¹A = AᵀMA = ΔF，红队 V4 同口径）。

    **只用于交叉验证/测试**（迁移矩阵 B1 行 2），禁止实验主路径调用——
    主路径一律 delta_f（Schur），两路线逐元素相对误差是 CI01 的第一主指标。
    σ² > 0 时 V 非奇异，可用 solve；σ=0 拒绝（走 delta_f 的 Λ=0 pinv 路径）。
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    Sigma_c = np.asarray(Sigma_c, float)
    m, q = B.shape
    assert A.shape[0] == m
    sigma2 = float(sigma) ** 2
    assert sigma2 > 0, "Route A 要求 σ²>0（σ=0 走 delta_f 的 Λ=0 路径）"
    V = sigma2 * np.eye(m) + B @ Sigma_c @ B.T
    X = np.linalg.solve(V, A)                          # V 非奇异（σ²>0）
    return sigma2 * (A.T @ X), dict(method="marginal_direct_inverse",
                                    cond_V=float(np.linalg.cond(V)))
