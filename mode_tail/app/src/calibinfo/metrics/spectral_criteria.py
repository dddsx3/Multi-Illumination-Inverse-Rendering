"""Prediction-side deterministic computation（第五章；唯一合法的 scalar 基线来源）。

全部函数确定性（无 rng）；谱界检查先于 log 保护（T5.5：禁止静默 clip）；
并联和走 Anderson–Duffin 一般式 S:Λ = S(S + Λ)⁺Λ（T10.2，禁裸 inv/solve）。
"""
from __future__ import annotations

import numpy as np


def lambda_min(F: np.ndarray) -> float:
    """P1：λmin 必须走 eigvalsh（禁 F.min()——全矩阵语义下取到非对角 0）。"""
    return float(np.linalg.eigvalsh(np.asarray(F, float)).min())


def trace_ratio(F: np.ndarray) -> float:
    """tr(F) / d（P2 的 trace ratio 口径；F 须正半定对角化语义）。"""
    F = np.asarray(F, float)
    return float(np.trace(F) / F.shape[0])


def logdet_deficit(F: np.ndarray, log_eps: float = 1e-12) -> float:
    """Δlogdet = Σ_i log max(r_i, log_eps)（相对 log det I = 0 的亏量；P2 口径）。"""
    r = np.linalg.eigvalsh(np.asarray(F, float))
    return float(np.log(np.maximum(r, log_eps)).sum())


def parallel_sum(S: np.ndarray, Lam: np.ndarray) -> np.ndarray:
    """Anderson–Duffin 并联和 S:Λ = S(S + Λ)⁺Λ（一般式，允许 S/Λ 半正定秩亏）。

    实现走对称化 + eigh/pinv 稳定路径；禁裸 inv/solve（solve 约束）。
    """
    S = np.asarray(S, float)
    Lam = np.asarray(Lam, float)
    S = (S + S.T) / 2
    Lam = (Lam + Lam.T) / 2
    # S(S + Λ)⁺Λ：对 G = S + Λ 做对称 eigh，构造 G⁺ 的对称伪逆
    w, V = np.linalg.eigh(S + Lam)
    tol = max(w.max(), 1.0) * 1e-12
    winv = np.where(w > tol, 1.0 / np.where(w > tol, w, 1.0), 0.0)
    Gplus = (V * winv) @ V.T
    return S @ Gplus @ Lam


def gauge_parallel_form(B: np.ndarray, cbar: np.ndarray, Lam: np.ndarray) -> float:
    """P4 右端：c̄ᵀ[(BᵀB):Λ]c̄（gauge 恒等式的并联和形式）。"""
    G = np.asarray(B, float).T @ np.asarray(B, float)
    return float(np.asarray(cbar, float) @ parallel_sum(G, Lam) @ np.asarray(cbar, float))


def identifiable_subspace(F_inf: np.ndarray, rank_relative_tol: float = 1e-10):
    """T5.4：q = #{r_i > tol·max_i r_i}；返回 (q, basis V_q, w_q)。"""
    w, V = np.linalg.eigh(np.asarray(F_inf, float))
    tol = rank_relative_tol * max(w.max(), 1e-300)
    keep = w > tol
    return int(keep.sum()), V[:, keep], w[keep]


def retention_spectrum_full(DeltaF: np.ndarray, F_inf: np.ndarray,
                            rank_relative_tol: float = 1e-10,
                            clip_tol: float = 1e-10,
                            log_eps: float = 1e-12):
    """全谱 retention（identifiable subspace 内）+ 谱界检查（T5.5：越界报错，禁静默 clip）。

    返回 dict(q, r (q,), r_log (q,), basis, Finf_proj)
    """
    DeltaF = np.asarray(DeltaF, float)
    F_inf = np.asarray(F_inf, float)
    q, Vq, wq = identifiable_subspace(F_inf, rank_relative_tol)
    # F∞^{-1/2} 在子空间内（正定平方根），全空间投影坐标
    Fh_q = Vq @ np.diag(1.0 / np.sqrt(wq)) @ Vq.T
    R = Fh_q @ DeltaF @ Fh_q
    R = (R + R.T) / 2
    r = np.linalg.eigvalsh(R)
    # T5.5 谱界：先检查后保护
    if r.min() < -clip_tol or r.max() > 1.0 + clip_tol:
        raise ValueError(
            f"retention 谱越界 [{r.min():.3e}, {r.max():.3e}]，违反 0≼R≼I（禁静默 clip）")
    r_log = np.maximum(r, log_eps)
    return dict(q=q, r=r, r_log=r_log, basis=Vq, Finf_proj=Fh_q)


def identifiable_overlap(R: np.ndarray, F_inf: np.ndarray, ajs: np.ndarray,
                         rank_relative_tol: float = 1e-10) -> np.ndarray:
    """P5/T8.1：E-min 方向与 tracked mode 的 F∞-度量 overlap（秩亏安全）。

    q_j = |a_jᵀ F∞ v_min| / sqrt((a_jᵀF∞a_j)(v_minᵀF∞v_min))，
    全部在 identifiable subspace 内计算（先投影，禁全空间直接套公式）。
    """
    F_inf = np.asarray(F_inf, float)
    R = np.asarray(R, float)
    ajs = np.atleast_2d(np.asarray(ajs, float))
    q, Vq, wq = identifiable_subspace(F_inf, rank_relative_tol)
    P = Vq @ Vq.T                                     # 子空间投影
    ev, V = np.linalg.eigh((Vq.T @ R @ Vq))
    v_min = Vq @ V[:, 0]                              # 子空间内最小特征向量
    Fv = F_inf @ (P @ v_min)                          # 子空间内一致的 F∞ 作用
    out = []
    for a in ajs:
        a_proj = P @ a
        Fa = F_inf @ a_proj
        num = abs(a_proj @ Fv)
        den = np.sqrt(max(a_proj @ Fa, 0.0) * (v_min @ Fv))
        out.append(num / den if den > 0 else 0.0)
    return np.asarray(out)


# ------------------------------------------------- Predictor definitions
def predictors_from_spectrum(r: np.ndarray, q: int, log_eps: float = 1e-12):
    """P_trace / P_logdet / P_emin（越大越坏）。"""
    r = np.asarray(r, float)
    return dict(
        P_trace=1.0 - float(r.mean()),
        P_logdet=float(-np.log(np.maximum(r, log_eps)).mean()),
        P_emin=1.0 - float(r.min()),
    )


def p_mode_from_pred_deg(pred_deg: np.ndarray) -> float:
    """P_mode = 1 - min_j rho_j = 1 - 1/max_j pred_deg_j（frozen artifact 直读）。"""
    return 1.0 - 1.0 / float(np.max(pred_deg))
