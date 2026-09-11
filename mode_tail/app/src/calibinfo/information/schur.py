"""B1 · schur：ΔF(Λ) 的唯一 Schur 实现 + 独立边缘化路线（交叉验证专用）。

模型：y = A x + B δc + ε，ε~N(0, σ²I)（先白化，见 whitening.whiten_system）。
Λ 的语义（数学冻结 v1.0 §5，必须区分两种对象）：
  - delta_f 的 Λ 是 **PSD profiling precision**（二次惩罚 cᵀΛc 的精度阵）：
    奇异时零特征方向 = flat / 不受惩罚，完全合法；
  - **proper Gaussian covariance** Σ_c 奇异时，零特征方向 = 几乎必然为零
    （硬支撑）——二者语义相反。此时**禁止**把 Λ = σ²Σ_c⁺ 塞进 delta_f 的
    无约束 profiling 并宣称等价（会丢掉支撑约束；已知答案：B=[1,1]、
    Σ_c=diag(1,0)、σ=1、A=[1] 时捷径给 0，正确值 0.5）。
    正确路线：协方差因子 / 边缘化（本模块 delta_f_marginal 对任意
    Σ_c≽0 成立，或 nuisance_factor 取 Σ_c=LLᵀ 后用 C=BL）。
  - Σ_c ≻ 0（满秩）时 Λ = σ²Σ_c⁻¹ 与边缘化等价（Woodbury），两路线互为交叉验证。

 solve（N3-fix）：Λ=0/秩亏路径禁用 np.linalg.solve——
本模块统一走 SVD/lstsq（pinv 极限），并在 delta_f 内显式断言。
thin-SVD 谱性质：谱性质仅在 thin-SVD 形式下成立（零奇异方向由 I−UUᵀ 项承载）。
绑定测试：test_information_modules.py + test_rank_deficient_lambda0.py（V 系列）
+ test_singular_covariance.py（MF-0.5 奇异协方差门）。
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

    Λ 语义 = **PSD profiling precision**（二次惩罚 cᵀΛc；奇异时零特征方向
    = flat / 不受惩罚）。proper Gaussian covariance 的边缘化路线见
    delta_f_marginal / delta_f_marginal_factor——不要把奇异 Σ_c 的伪逆
    塞进本函数并宣称等价（语义相反，见模块 docstring 与 MF-0.5 已知答案）。

    参数
    ----
    A : (m, n) 主参数设计矩阵（x 空间）
    B : (m, q) nuisance 设计矩阵（δc 空间）
    Lam : 标量 λ（各向同性）/ (q,) 对角 / (q,q) 矩阵，profiling precision
    rank_policy : "pinv"（默认，Λ=0/秩亏自动 SVD/lstsq 极限）或
                  "raise_if_singular"（数值奇异则抛错）

    返回
    ----
    (DeltaF (n,n), M (m,m), diagnostics dict)
    diagnostics: rank（G 的数值秩）、method（实现路径）、g_min/g_max（G 特征值范围）

    本函数任何路径不调用 np.linalg.solve（已知答案测试盯防）。
    M 为 (m,m) 稠密矩阵——CI 尺度（m≤数千）可用；更大规模走低秩/MatrixFree（CI03 预算）。
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
    Schur 单位——Woodbury: V⁻¹ = σ⁻²M ⇒ σ²AᵀV⁻¹A = AᵀMA = ΔF，同口径）。

    对任意 Σ_c ≽ 0 精确成立（含奇异协方差：σ²I 保证 V 非奇异，支撑语义由
    Σ_c 自身承载——这是奇异 Gaussian covariance 的合法处理路线；
    对偶的 profiling-precision 伪逆捷径 Σ_c⁺ 语义不同，禁止混用）。

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


def delta_f_marginal_factor(A, B, Sigma_c, sigma, rtol=1e-12):
    """奇异协方差主定理路线（数学冻结 v1.0 §13）：thin 因子 Σ_c = LLᵀ
    （L: q×r_c，r_c = rank(Σ_c)），C = BL（或白化后 Σ_y^{-1/2}BL），

        F_x = Ãᵀ (I + CCᵀ)⁻¹ Ã，
        Woodbury: (I + CCᵀ)⁻¹ = I − C(I + CᵀC)⁻¹Cᵀ
        ⇒ ΔF = σ²·Aᵀ[I − C(I + CᵀC)⁻¹Cᵀ]A（同 Schur 单位）。

    对任意 Σ_c ≽ 0 精确成立、自动保留支撑，且 q 大而 r_c 小时可显著省算力
    （(r_c, r_c) 逆代替 (m, m) 逆）。本函数是 proper structured Gaussian
    uncertainty 的正式实现；与 delta_f_marginal 在数值上应一致（MF-0.5 盯防）。

    只用于交叉验证/测试（同 Route A 纪律）；返回 (DeltaF, meta)。
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    Sigma_c = 0.5 * (np.asarray(Sigma_c, float)
                     + np.asarray(Sigma_c, float).T)
    m, q = B.shape
    assert A.shape[0] == m and Sigma_c.shape == (q, q)
    sigma2 = float(sigma) ** 2
    assert sigma2 > 0, "要求 σ²>0（σ=0 无边缘协方差对象）"
    w, V = np.linalg.eigh(Sigma_c)
    keep = w > max(w.max(), 1.0) * rtol
    if not keep.any():
        raise ValueError("Σ_c = 0：c 几乎必然为零，无 nuisance 可边缘化")
    L = V[:, keep] * np.sqrt(w[keep])[None, :]         # (q, r_c) thin 因子
    C = B @ L / sigma                                  # (m, r_c)：白化后 nuisance（σ 归一）
    G = np.eye(C.shape[1]) + C.T @ C                   # (r_c, r_c) 正定
    X = np.linalg.solve(G, C.T @ A)                    # (r_c, n)
    # ΔF = σ²AᵀV⁻¹A = Aᵀ(I+CCᵀ)⁻¹A（σ² 已并入 C 的归一），Woodbury 展开：
    DeltaF = A.T @ A - A.T @ C @ X
    meta = dict(method="covariance_factor_marginal",
                rank_c=int(keep.sum()),
                cond_G=float(np.linalg.cond(G)))
    return DeltaF, meta


def nuisance_factor(Sigma_c, rtol=1e-12):
    """奇异/满秩 proper Gaussian covariance 的支持保持参数化（数学冻结 v1.0 §6）：

        Σ_c = L Lᵀ，L: (q, r_c)，r_c = rank(Σ_c)；c = Lz，z ~ N(0, I_{r_c})。

    白化后 nuisance 矩阵 C = B L（L 零列空间 = Σ_c 的近零谱方向被自动剔除，
    支撑语义保留）。这是奇异 Σ_c 的**唯一正式表述**；把 Σ_c 的伪逆当作
    profiling precision 塞进 delta_f 是语义错误（零协方差 ≠ 零精度）。
    返回 (L, r_c, meta)。
    """
    Sigma_c = 0.5 * (np.asarray(Sigma_c, float)
                     + np.asarray(Sigma_c, float).T)
    q = Sigma_c.shape[0]
    w, V = np.linalg.eigh(Sigma_c)
    keep = w > max(w.max(), 1.0) * rtol
    L = V[:, keep] * np.sqrt(np.maximum(w[keep], 0.0))[None, :]
    meta = dict(rank_c=int(keep.sum()), q=int(q),
                g_min=float(w.min()), g_max=float(w.max()))
    return L, int(keep.sum()), meta
